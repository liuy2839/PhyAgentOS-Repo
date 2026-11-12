"""π0.5 策略节点（forge_policy 统一协议适配）—— 方案 a：WS 客户端 + 本地服务器。

架构（依赖隔离）：π0.5 的 openpi 依赖（jax/torch 与 RoboDojo 的 websockets 版本
冲突等）隔离在独立 venv（`XPolicyLab/policy/Pi_05/openpi/`）。本节点进程不加载
模型，而是：
  1. 启动时**拉起本地策略服务器**子进程（Pi_05/openpi venv 里的
     `client_server.ws.__main__`，XPolicyLab 的 PolicyServer 承载 Pi_05.Model）；
  2. 轮询直到服务器就绪（加载完权重能响应——几分钟，est_startup_s 已声明）；
  3. 经 **WsModelClient**（XPolicyLab 现成客户端，`client.call(func, obs)` 通用
     分发 update_obs/get_action/reset）驱动推理——这就是 RoboDojo 官方 eval 链路
     用的同一客户端（评测环境里被 stub 的正是它，这里是真用）；
  4. runtime 停止时先停服务器子进程（进程组整组清理，不留占显存的孤儿）。

模型侧事实（zy2d 实测核对，2026-11-11）：
  * Pi_05.Model 与 FastWAM/G05 同族（ModelTemplate）：reset / update_obs /
    get_action → 动作**带名 dict**（unpack_robot_state 后），同 fastwam 的展平；
  * deploy.yml（arx_x5）：`action_type: joint`、`train_config_name:
    pi05_base_aloha_full_sim_arx-x5_seed_0`、`repo_id: arx_x5_sim`、
    `checkpoint_num: 59999`、`protocol: ws`、`port: 6000`；
  * 权重（pro5000 zy2d）：`/data2/kangning/models/robotwin_pi05/
    RoboTwin-lerobot_v30-aloha_agilex-joint-0`（model_path 指到它；
    目录名里的 aloha_agilex 是历史命名，训练配置是 arx-x5）；
  * XPolicyLab 检出：`/data2/kangning/fastwam_eval/RoboDojo/XPolicyLab`。

一次 `get_action()` 的返回是**一段 chunk**（openpi 的 action chunk 经
unpack_robot_state 后的列表——与 fastwam 的 get_action 返回同语义），所以
step() 的节奏与 fastwam 节点一致：chunk 队列逐帧吐、末动作 (vec, True)。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 让同仓库的 forge_policy SDK 可导入（部署环境装包后可去掉）
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "policy" / "src"))

from forge_policy import Policy, PolicyContract, PolicyRuntime  # noqa: E402


def log(msg: str) -> None:
    print(f"[pi05_policy] {msg}", flush=True)


# ---- 部署路径（zy2d 实测；换机用环境变量覆盖）--------------------------------

XPOLICYLAB_ROOT = os.environ.get(
    "XPOLICYLAB_ROOT", "/data2/kangning/fastwam_eval/RoboDojo/XPolicyLab"
)
PI05_POLICY_DIR = f"{XPOLICYLAB_ROOT}/policy/Pi_05"
PI05_OPENPI_PYTHON = os.environ.get(
    "PI05_OPENPI_PYTHON", "/data1/liuyi/pi05_openpi_venv/bin/python"
)
PI05_MODEL_DIR = os.environ.get(
    "PI05_MODEL_DIR",
    "/data2/kangning/models/robotwin_pi05/RoboTwin-lerobot_v30-aloha_agilex-joint-0",
)
PI05_PORT = int(os.environ.get("PI05_PORT", "6008"))   # 6000 是官方默认，避开常用冲突
PI05_CHECKPOINT_NUM = os.environ.get("PI05_CHECKPOINT_NUM", "29999")  # zy2d 实测只有 29999
PI05_REPO_ID = os.environ.get("PI05_REPO_ID", "RoboTwin_lerobot_v30")  # norm stats 子目录名（zy2d 实测）
PI05_MEM_FRACTION = os.environ.get("PI05_MEM_FRACTION", "0.15")  # 共享卡限 jax 预留（≈11GB/72G）
PI05_GPU = os.environ.get("PI05_GPU", "2")            # 绑卡（跑前 nvidia-smi 确认最空卡）

SERVER_READY_TIMEOUT_S = float(os.environ.get("PI05_SERVER_READY_TIMEOUT_S", "900"))


class Pi05ServerHandle:
    """本地 π0.5 策略服务器子进程的拉起、就绪轮询与整组清理。"""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if not os.path.isfile(PI05_OPENPI_PYTHON):
            raise RuntimeError(
                f"π0.5 openpi venv 的 python 不存在：{PI05_OPENPI_PYTHON}——"
                "先在 zy2d 上跑 Pi_05/setup_eval_policy_server.sh 建 venv"
            )
        env = dict(os.environ)
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", PI05_MEM_FRACTION)
        env.setdefault("CUDA_VISIBLE_DEVICES", PI05_GPU)
        env.setdefault("XDG_CACHE_HOME", "/tmp/pi05_cache")
        # 服务器启动走 XPolicyLab 的官方入口 setup_policy_server.py：
        #   --config-path deploy.yml --overrides key=value...
        # （overrides 机制把 model_path/port 等键直接注进 deploy_cfg；
        #   setup 脚本内部 import XPolicyLab.policy.<policy_name>.model
        #   并以 PolicyServer 承载 Model。PYTHONPATH 需要 RoboDojo 仓库根
        #   （XPolicyLab 的父目录）+ openpi 的 src。）
        repo_root = str(Path(XPOLICYLAB_ROOT).parent)
        openpi_src = f"{PI05_POLICY_DIR}/openpi/src"
        env["PYTHONPATH"] = ":".join(
            p for p in (env.get("PYTHONPATH", ""), repo_root, openpi_src) if p
        )
        cmd = [
            PI05_OPENPI_PYTHON, "-u", f"{XPOLICYLAB_ROOT}/setup_policy_server.py",
            "--config_path", f"{PI05_POLICY_DIR}/deploy.yml",
            "--overrides",
            "policy_name=Pi_05",
            f"model_path={PI05_MODEL_DIR}",
            f"checkpoint_num={PI05_CHECKPOINT_NUM}",
            "train_config_name=pi05_base_aloha_full_sim_arx-x5_seed_0",
            f"repo_id={PI05_REPO_ID}",
            "action_type=joint",
            "env_cfg_type=arx_x5",
            f"port={PI05_PORT}",
            "host=127.0.0.1",
        ]
        log(f"拉起 π0.5 策略服务器：{' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd, cwd=XPOLICYLAB_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,        # 独立进程组：退出时整组清理
        )

    def wait_ready(self, client: Any) -> None:
        """轮询服务器：能完成一次 reset 调用即就绪（权重加载完成）。"""
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"π0.5 服务器进程退出（code={self.proc.returncode}）——"
                    "看上面的服务器日志（通常是权重路径/openpi 依赖问题）"
                )
            try:
                client.call("reset")
                log(f"π0.5 服务器就绪（端口 {PI05_PORT}）")
                return
            except Exception as exc:  # noqa: BLE001 连不上/未就绪都继续等
                if int(time.monotonic()) % 30 == 0:
                    log(f"等待 π0.5 服务器就绪…（{exc}）")
                time.sleep(2.0)
        raise RuntimeError(
            f"π0.5 服务器 {SERVER_READY_TIMEOUT_S:.0f}s 内未就绪（端口 {PI05_PORT}）"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        self.proc = None
        log("π0.5 服务器已停止")


class Pi05Policy(Policy):
    """π0.5 的 forge_policy 四方法：经 WsModelClient 驱动远端 Pi_05.Model。

    观测还原与 chunk 节奏与 fastwam 节点同款：线上 float64 → XPolicyLab 标准
    观测（夹爪 list / 其余 float32，图像 RGB 不转换）；get_action 返回一段
    chunk → 队列逐帧吐，末动作 (vec, True)。
    """

    def __init__(self, client: Any, server: Pi05ServerHandle) -> None:
        self._client = client
        self._server = server
        self._instruction = ""
        self._chunk: list[np.ndarray] = []
        self._chunk_pos = 0

    # ---- forge_policy.Policy 四方法 ----

    def reset_episode(self) -> None:
        # 官方每集双 reset 的节奏照抄（与 fastwam 一致）；WS 客户端的 reset
        # 还会把服务端 step 计数归零。
        self._client.call("reset")
        self._client.call("reset")
        self._chunk = []
        self._chunk_pos = 0

    def set_instruction(self, text: str) -> None:
        self._instruction = str(text)
        log(f"指令：{self._instruction}")

    def step(self, obs: dict[str, Any]) -> Any:
        # 先无条件 update_obs（官方节奏：chunk 内每步都喂）；
        # chunk 用尽才 get_action（一次推理产一段）。
        model_obs = self._to_model_obs(obs)
        self._client.call("update_obs", model_obs)
        if self._chunk_pos >= len(self._chunk):
            actions = self._client.call("get_action")
            if not actions:
                log("π0.5 返回空 chunk，本帧不动作（评测端 watchdog 会重发观测）")
                return None
            self._chunk = [self._flatten(a) for a in actions]
            self._chunk_pos = 0
        vec = self._chunk[self._chunk_pos]
        last = self._chunk_pos + 1 == len(self._chunk)
        self._chunk_pos += 1
        return vec, last

    def shutdown(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        self._server.stop()

    # ---- 内部 ----

    def _to_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """PolicyRuntime 扁平 obs → XPolicyLab 标准观测（与 fastwam 节点同源）。"""
        by_name = dict(zip(obs.get("state_names", []), obs.get("state", []), strict=False))
        st: dict[str, Any] = {}
        for key in (
            "left_arm_joint_state", "right_arm_joint_state",
            "left_ee_pose", "right_ee_pose",
        ):
            dim = 6 if "arm_joint" in key else 7
            st[key] = np.asarray(
                [by_name.get(f"{key}.{i}", 0.0) for i in range(dim)], dtype=np.float32
            )
        for key in ("left_ee_joint_state", "right_ee_joint_state"):
            st[key] = [float(by_name.get(f"{key}.0", 0.0))]
        return {
            "instruction": self._instruction,
            "vision": {cam: {"color": img} for cam, img in obs.get("images", {}).items()},
            "state": st,
        }

    def _flatten(self, action: Any) -> np.ndarray:
        """Pi_05 输出（unpack_robot_state 后的带名 dict）→ 14 维向量。"""
        if isinstance(action, dict):
            parts: list[np.ndarray] = []
            for key in (
                "left_arm_joint_state", "left_ee_joint_state",
                "right_arm_joint_state", "right_ee_joint_state",
            ):
                if key not in action:
                    raise KeyError(f"π0.5 动作缺少 {key}；实得键={sorted(action)}")
                parts.append(np.asarray(action[key], dtype=np.float32).reshape(-1))
            return np.concatenate(parts).astype(np.float32)
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 14:
            raise ValueError(f"π0.5 动作应为 14 维，收到 {arr.size}")
        return arr


def _action_names_for() -> tuple[str, ...]:
    names: list[str] = []
    for p in ("left_", "right_"):
        names.extend(f"{p}arm_joint_state.{i}" for i in range(6))
        names.extend(f"{p}ee_joint_state.{i}" for i in range(1))
    return tuple(names)


def build_client_and_server() -> tuple[Any, Pi05ServerHandle]:
    """拉起本地 π0.5 服务器并建 WsModelClient。加载几分钟——runtime 的 ready
    要等这里返回才上报，评测端按 est_startup_s 预算等待。"""
    xpolicy = XPOLICYLAB_ROOT
    for p in (str(xpolicy), str(Path(xpolicy).parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    server = Pi05ServerHandle()
    server.start()
    from XPolicyLab.client_server.ws.model_client import WsModelClient

    client = WsModelClient(
        url=f"ws://127.0.0.1:{PI05_PORT}",
        evaluation_id="pi05-forge",
        trial_id="pi05-forge",
        action_case_id="pi05_case",
        max_connect_seconds=SERVER_READY_TIMEOUT_S,
    )
    server.wait_ready(client)
    return client, server


def main() -> int:
    action_names = _action_names_for()
    contract = PolicyContract(
        action_names=action_names,
        cameras=("image/head", "image/left_wrist", "image/right_wrist"),
        emit_chunk_end=True,
    )
    client, server = build_client_and_server()
    policy = Pi05Policy(client, server)
    runtime = PolicyRuntime(policy, contract)
    # 契约自报（协议 v2 握手）：随 ready 上报，评测端校验后启动回合。
    runtime.set_handshake(
        policy_id="pi05",
        action_type="joint",
        weight_id=f"pi05-robodojo-arx-x5-joint-ckpt{PI05_CHECKPOINT_NUM}",
        obs_state_dim=28,
        est_startup_s=SERVER_READY_TIMEOUT_S,
    )
    log(f"π0.5 节点就绪：动作 {contract.action_dim} 维（WS:{PI05_PORT}），等待评测端命令")
    return runtime.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
