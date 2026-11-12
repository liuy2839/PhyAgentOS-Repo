"""FastWAM 策略节点（模块 B）：XPolicyLab 加载器 + forge_policy 四方法适配。

**解耦边界**：与 ReKep 节点（nodes/policies/rekep）同构——算法实现
`forge_policy.Policy` 四方法，Dora 收发全部由 `PolicyRuntime` 承担。本文件只
负责把 XPolicyLab 的策略适配器（`XPolicyLab/policy/<POLICY_NAME>/model.Model`，
FastWAM_80k_forge 与 G05 同一入口）包成四方法，并**复刻官方 rollout 节奏**
（XPolicyLab/policy/demo_policy/deploy.py）：

    model.reset()
    while not is_episode_end():
        model.update_obs(obs); actions = model.get_action()      # 一次一个 chunk
        for i, a in enumerate(actions):
            take_action(a)
            if is_episode_end() or i+1 == len(actions): break
            model.update_obs(next_obs)                            # chunk 内也每步喂

对应到四方法：
    reset_episode  → model.reset()（官方每集 reset 两次的节奏照抄：双 reset）
    set_instruction→ 记录指令（action chunk 用尽后重推理时生效）
    step           → update_obs 每帧都喂；chunk 未用尽时直接吐下一个动作
    chunk 边界     → 吐出 chunk 最后一个动作后置 mark_chunk_end（哨兵随
                     **本 chunk 末动作**发出，语义见 forge_policy.runtime）

环境变量（与原 policy_node.py / launcher 逐字一致）：
    XPOLICYLAB_ROOT  XPolicyLab 检出目录（必填）
    POLICY_NAME      FastWAM_80k_forge（默认 G05）
    ENV_CFG_TYPE / ACTION_TYPE / TASK_NAME / CKPT_NAME
    G05_PROCESSOR_PATH（仅 G05 必填；FastWAM 权重路径在其 deploy.yml 里）
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

# 让同仓库的 forge_policy SDK 可导入（部署环境装包后可去掉）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "policy", "src")))

from forge_policy import Policy, PolicyContract, PolicyRuntime  # noqa: E402


def log(msg: str) -> None:
    print(f"[fastwam_policy] {msg}", flush=True)


def _build_obs_spec_from_dims(arm_dims, ee_dims) -> tuple[tuple[str, int], ...]:
    """由 get_robot_action_dim_info 的结果推出观测规格（28 维），按键名字典序。

    与评测节点侧 build_obs_spec() 的算法逐字一致 —— 单测注入 fake dims
    即可覆盖，不需要 XPolicyLab。
    """
    prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        spec.append((f"{p}arm_joint_state", int(ad)))
        spec.append((f"{p}ee_joint_state", int(ed)))
        spec.append((f"{p}ee_pose", 7))
    return tuple(sorted(spec, key=lambda kv: kv[0]))


def _build_state_spec_from_dims(arm_dims, ee_dims, action_type: str) -> tuple[tuple[str, int], ...]:
    if action_type not in ("joint", "ee"):
        raise ValueError(f'action_type 必须是 "joint" 或 "ee"，收到 {action_type!r}')
    prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        spec.append((f"{p}arm_joint_state", int(ad)) if action_type == "joint" else (f"{p}ee_pose", 7))
        spec.append((f"{p}ee_joint_state", int(ed)))
    return tuple(spec)


def _robot_dims(env_cfg_type: str):
    """XPolicyLab 的机器人维度信息（模块级隔离：单测不 import 本函数）。"""
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    return list(info["arm_dim"]), list(info["ee_dim"])


def build_obs_spec(env_cfg_type: str) -> tuple[tuple[str, int], ...]:
    return _build_obs_spec_from_dims(*_robot_dims(env_cfg_type))


def build_state_spec(env_cfg_type: str, action_type: str) -> tuple[tuple[str, int], ...]:
    return _build_state_spec_from_dims(*_robot_dims(env_cfg_type), action_type)


def _flatten_action(action: dict[str, Any], spec) -> np.ndarray:
    parts = []
    for key, dim in spec:
        v = action.get(key)
        if v is None:
            raise KeyError(f"策略动作缺少 {key}；实得键={sorted(action)}")
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"{key} 维度应为 {dim}，收到 {arr.size}")
        parts.append(arr)
    return np.concatenate(parts).astype(np.float32)


def build_model() -> Any:
    """按 XPolicyLab 的约定加载策略适配器。

    注意 CWD：官方 setup_eval_* 脚本要求 CWD 是 policy 目录（策略里有相对路径）。
    """
    xpl_root = os.environ.get("XPOLICYLAB_ROOT")
    if not xpl_root or not os.path.isdir(xpl_root):
        raise RuntimeError("必须设置 XPOLICYLAB_ROOT 指向 XPolicyLab 检出目录")
    parent = os.path.dirname(os.path.abspath(xpl_root))
    for p in (parent, xpl_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    policy = os.environ.get("POLICY_NAME", "G05")
    import importlib

    from utils.load_file import load_yaml  # type: ignore

    deploy = load_yaml(os.path.join(xpl_root, "policy", policy, "deploy.yml")) or {}
    # **不能用 setdefault**：deploy.yml 里这些键是存在的、值为 null（demo_policy
    # 的模板就长这样），setdefault 不覆盖 None，结果 Model 拿到 env_cfg_type=None
    # 去找 `env_cfg/None.yml`。必须显式判空。
    for _k, _v in (
        ("env_cfg_type", os.environ.get("ENV_CFG_TYPE", "arx_x5")),
        ("action_type", os.environ.get("ACTION_TYPE", "joint")),
        ("bench_name", "RoboDojo"),
        ("task_name", os.environ.get("TASK_NAME")),
        ("seed", 0),
    ):
        if deploy.get(_k) is None and _v is not None:
            deploy[_k] = _v
    ckpt = os.environ.get("CKPT_NAME")
    if ckpt:
        deploy["ckpt_name"] = ckpt

    # G05 resolves weights from G05_CKPT_PATH inside its model adapter.  The
    # processor path is likewise deployment-specific; when supplied, replace
    # any stale path from an upstream deploy.yml with the portable override.
    if policy == "G05":
        processor_path = os.environ.get("G05_PROCESSOR_PATH")
        if not processor_path:
            raise RuntimeError(
                "G05 requires G05_PROCESSOR_PATH; do not rely on a machine-specific "
                "processor path from the upstream deploy.yml"
            )
        overrides = [
            str(item)
            for item in (deploy.get("hydra_overrides") or [])
            if not str(item).startswith(
                ("model.model_arch.hf_processor_path=", "model.processor.hf_processor_path=")
            )
        ]
        overrides.append(f"model.model_arch.hf_processor_path={processor_path}")
        deploy["hydra_overrides"] = overrides

    os.chdir(os.path.join(xpl_root, "policy", policy))
    mod = importlib.import_module(f"XPolicyLab.policy.{policy}.model")
    log(f"加载策略 {policy}（action_type={deploy['action_type']}, env_cfg={deploy['env_cfg_type']}）")
    return mod.Model(deploy)


class FastWAMPolicy(Policy):
    """把 XPolicyLab 的 model.Model 包成 forge_policy.Policy 四方法。

    **观测 → 动作的类型还原**：XPolicyLab 的标准观测里，夹爪（ee_joint_state）
    是 Python list、其余是 float32 ndarray。传输层统一是 float64，还原时必须
    按官方类型拼回去 —— `_encode_observation` 把关节转回 float32，ee_pose 却按
    float64 用，类型错了会静默走进位姿还原的数学里。图像已是 RGB，不做任何
    通道转换（任何 COLOR_BGR2RGB 都是 bug）。
    """

    def __init__(
        self, model: Any, arm_dims, ee_dims, action_type: str, obs_spec=None
    ) -> None:
        """`obs_spec` 可注入（单测用）；默认由 XPolicyLab 维度信息推导。"""
        self._model = model
        self._obs_spec = obs_spec or _build_obs_spec_from_dims(arm_dims, ee_dims)
        self._state_spec = _build_state_spec_from_dims(arm_dims, ee_dims, action_type)
        self._instruction = ""
        self._chunk: list[np.ndarray] = []
        self._chunk_pos = 0

    # ---- forge_policy.Policy 四方法 ----

    def reset_episode(self) -> None:
        # 官方每集调两次 reset（env.reset() 里一次 + eval_one_episode 开头
        # 一次），对有内部状态的策略这个次数是有意义的，所以照抄双 reset。
        self._model.reset()
        self._model.reset()

    def set_instruction(self, text: str) -> None:
        self._instruction = str(text)
        log(f"指令：{self._instruction}")

    def step(self, obs: dict[str, Any]) -> "np.ndarray | tuple[np.ndarray, bool] | None":
        # **先无条件 update_obs**（官方节奏坑 2：有策略吃历史帧，只在需要
        # 动作时喂会静默劣化），chunk 用尽才调 get_action 推新 chunk。
        model_obs = self._to_model_obs(obs)
        self._model.update_obs(model_obs)
        if self._chunk_pos >= len(self._chunk):
            actions = self._model.get_action()
            if not actions:
                log("策略返回空 chunk，本帧不动作（评测端 watchdog 会重发观测）")
                return None
            self._chunk = [_flatten_action(a, self._state_spec) for a in actions]
            self._chunk_pos = 0
        vec = self._chunk[self._chunk_pos]
        last = self._chunk_pos + 1 == len(self._chunk)
        self._chunk_pos += 1
        # 元组声明 is_last：runtime 自动给末动作附 effort 哨兵，评测端补查
        # 那次 is_episode_end() 的时机与官方（chunk 末动作之后、下一帧观测
        # 之前）逐字对齐——不再依赖手工 mark_chunk_end 的置位时机。
        return vec, last

    # ---- 内部 ----

    def _to_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """PolicyRuntime 的扁平 obs → XPolicyLab 标准观测格式。"""
        by_name = dict(zip(obs.get("state_names", []), obs.get("state", []), strict=False))
        st: dict[str, Any] = {}
        for key, dim in self._obs_spec:
            seg = np.asarray([by_name.get(f"{key}.{i}", 0.0) for i in range(dim)], dtype=np.float64)
            if key.endswith("ee_joint_state"):
                st[key] = [float(x) for x in seg]           # 夹爪：Python list
            else:
                st[key] = np.asarray(seg, dtype=np.float32)  # 其余：float32
        return {
            "instruction": self._instruction,
            "vision": {cam: {"color": img} for cam, img in obs.get("images", {}).items()},
            "state": st,
        }


def _action_names(state_spec) -> tuple[str, ...]:
    return tuple(f"{key}.{i}" for key, dim in state_spec for i in range(dim))


def main() -> int:
    env_cfg_type = os.environ.get("ENV_CFG_TYPE", "arx_x5")
    action_type = os.environ.get("ACTION_TYPE", "joint")
    arm_dims, ee_dims = _robot_dims(env_cfg_type)

    state_spec = _build_state_spec_from_dims(arm_dims, ee_dims, action_type)
    contract = PolicyContract(
        action_names=_action_names(state_spec),
        cameras=("image/head", "image/left_wrist", "image/right_wrist"),
        emit_chunk_end=True,
    )
    policy = FastWAMPolicy(build_model(), arm_dims, ee_dims, action_type)
    runtime = PolicyRuntime(policy, contract)
    # 契约自报（协议 v2 握手）：随 ready 上报，评测端校验后启动回合。
    runtime.set_handshake(
        policy_id=os.environ.get("POLICY_NAME", "G05"),
        action_type=action_type,
        weight_id=os.environ.get("WEIGHT_ID"),
        weight_sha256=os.environ.get("WEIGHT_SHA256"),
        obs_state_dim=sum(d for _, d in policy._obs_spec),
    )
    obs_spec = policy._obs_spec
    log(f"动作 {contract.action_dim} 维 / 观测 {sum(d for _, d in obs_spec)} 维")
    return runtime.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
