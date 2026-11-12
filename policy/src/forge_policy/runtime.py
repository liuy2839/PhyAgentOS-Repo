"""PolicyRuntime：统一策略节点的 Dora 事件循环。

职责（算法与传输分离，见 SKILL_DEVELOPMENT §4.2）：
- 解析/聚合输入帧：proprio_state + contract 声明的相机全部到齐才算一帧；
- 分发 PolicyCommand：start/set_instruction/reset_scene/stop/cancel；
- 观测驱动推理：一帧完整观测调一次 policy.step()，至多发一个 action；
- JointCommand 构造（name/position/mode='position'，chunk_end 填 effort 哨兵）；
- policy_status 状态上报（JSON 文本）。
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable
import numpy as np

from .base import Policy, PolicyError
from .contract import (
    CHUNK_END_SENTINEL,
    MIN_POLICY_COMMAND_QUEUE,
    POLICY_STATUS_CANCELLED,
    POLICY_STATUS_FAILED,
    POLICY_STATUS_IDLE,
    POLICY_STATUS_READY,
    POLICY_STATUS_RUNNING,
    POLICY_STATUS_STOPPED,
    RETRYABLE_ERROR_CODES,
    TERMINAL_COMMANDS,
    PolicyContract,
)
from .msgs import Image, JointCommand, JointState, PolicyCommand


def log(msg: str) -> None:
    print(f"[policy_runtime] {msg}", flush=True)


def _payload_text(value: Any) -> str:
    """把 dora 传来的 JSON 文本负载还原成 str（bytes / Arrow / list[int] / str 都认）。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    if hasattr(value, "to_pylist"):
        rows = value.to_pylist()
        if len(rows) == 1:
            row = rows[0]
            if isinstance(row, str):
                return row
            if isinstance(row, dict) and len(row) == 1:
                return _payload_text(next(iter(row.values())))
            return _payload_text(row)
        return _payload_text(rows)
    if hasattr(value, "as_py"):
        return _payload_text(value.as_py())
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], int):
            return bytes(value).decode("utf-8")
        if len(value) == 1:
            return _payload_text(value[0])
    raise TypeError(f"无法解析 JSON 文本负载: {type(value)}")


class PolicyRuntime:
    def __init__(
        self,
        policy: Policy,
        contract: PolicyContract,
        *,
        node: Any = None,
        logger: Callable[[str], None] = log,
    ) -> None:
        self.policy = policy
        self.contract = contract
        if node is None:
            from dora import Node

            node = Node()
        self.node = node
        self._log = logger
        self._phase = POLICY_STATUS_IDLE
        self._instruction = ""
        self._state: np.ndarray | None = None
        self._state_names: list[str] = []
        self._images: dict[str, np.ndarray] = {}
        self._depths: dict[str, np.ndarray] = {}
        self._segs: dict[str, np.ndarray] = {}
        self._scene_objects: list[dict[str, Any]] | None = None
        self._camera_info: dict[str, Any] | None = None
        self._fresh_inputs: set[str] = set()
        self._missing_logged: set[tuple[str, ...]] = set()
        self._last_complete_obs: dict[str, Any] | None = None
        self._chunk_pending_end = False
        self._failed = False
        self._sent = 0
        # ---- 协议 v2（与评测节点 robodojo_eval/episode_service 对应）----
        # 当前回合身份：评测端每帧观测随 obs_meta {episode_id, seq} 同拍发出，
        # 动作经 action_meta 回带同标签——异回合动作被评测端机械丢弃。
        self._episode_id: str | None = None
        self._obs_seq: int | None = None
        # ready 载荷的契约自报（评测端握手门禁的数据源）：由策略/入口在构造
        # 后经 set_handshake() 填写；run() 开头随 ready 一次性上报。
        self._handshake_payload: dict[str, Any] = {}

    def set_handshake(self, **payload: Any) -> None:
        """填写 ready 时的契约自报（policy_id/action_type/action_names/cameras/
        weight_id/weight_sha256/est_startup_s/emit_chunk_end/chunk_per_action/
        obs_state_dim——字段表见本目录 HANDSHAKE_FIELDS.md）。"""
        self._handshake_payload.update(payload)

    # -------------------------------------------------------------- status
    def _emit_status(self, state: str, **extra: Any) -> None:
        payload = {"status": state, **extra}
        try:
            self.node.send_output(
                "policy_status", json.dumps(payload, ensure_ascii=False).encode("utf-8")
            )
        except Exception as exc:  # 输出未在 dataflow 声明时忽略
            self._log(f"policy_status 输出不可用: {exc}")

    def _emit_action_meta(self) -> None:
        """随动作同拍发出身份标签（回带当前回合的 episode_id + seq）。"""
        if self._episode_id is None:
            return
        try:
            self.node.send_output(
                "action_meta",
                json.dumps({"episode_id": self._episode_id, "seq": self._obs_seq},
                           ensure_ascii=False).encode("utf-8"),
            )
        except Exception as exc:  # 输出未在 dataflow 声明时忽略
            self._log(f"action_meta 输出不可用: {exc}")

    # ------------------------------------------------------------ commands
    def _handle_command(self, cmd: PolicyCommand) -> bool:
        """返回 False 表示进程应退出（stop/cancel）。"""
        command = cmd.command
        if command in TERMINAL_COMMANDS:
            self._log(f"收到 {command}，退出")
            self.policy.shutdown()
            self._emit_status(
                POLICY_STATUS_CANCELLED if command == "cancel" else POLICY_STATUS_STOPPED
            )
            return False
        if command == "start":
            self._failed = False
            self._phase = POLICY_STATUS_RUNNING
            self._emit_status(self._phase, instruction=self._instruction)
            self._retry_last_obs()
        elif command == "set_instruction":
            payload = json.loads(cmd.inputs_json or "{}")
            text = str(payload.get("instruction", ""))
            self._instruction = text
            self.policy.set_instruction(text)
            self._log(f"指令：{text}")
            self._retry_last_obs()
        elif command == "reset_scene":
            # 跨通道无序：reset 可能清掉已到达的观测帧；评测端 watchdog 会
            # 幂等重发观测，这里直接丢弃半帧即可。
            # 评测端只在启动时发一次 start，之后每集发 reset_scene —— 因此
            # reset 不改变运行相位（running 保持 running），只清 episode 状态。
            self.policy.reset_episode()
            self._state = None
            self._images, self._depths, self._segs = {}, {}, {}
            self._scene_objects = None
            self._camera_info = None
            self._fresh_inputs.clear()
            self._last_complete_obs = None
            self._chunk_pending_end = False
            self._failed = False
            self._episode_id = None          # 新回合：旧身份作废
            self._obs_seq = None
            if self._phase == POLICY_STATUS_IDLE:
                self._phase = POLICY_STATUS_READY
            # reset 确认（协议 v2）：评测端收到才发首帧——观测先于 reset 被
            # 消费/清空的乱序死锁从结构上消除（超时兜底在评测端）。
            self._emit_status(self._phase)
            self._emit_status("reset_ack")
        elif command in ("pause", "resume"):
            # 协议预留：评测端当前不发，保持运行态切换语义。
            if command == "pause":
                self._phase = POLICY_STATUS_READY
            elif self._instruction or self._phase != POLICY_STATUS_IDLE:
                self._phase = POLICY_STATUS_RUNNING
            self._emit_status(self._phase)
        else:
            self._log(f"忽略未知命令 {command!r}")
        return True

    # -------------------------------------------------------- observation
    def _try_step(self) -> None:
        """观测收齐且处于 running 才推理；一帧至多一个 action。

        临时性失败（如指令尚未到达）不置 _failed：缓存完整观测，
        等下一条命令（set_instruction/start）到达时重试同一帧。
        """
        if self._phase != POLICY_STATUS_RUNNING or self._failed:
            return
        if self._state is None:
            return
        required = self.contract.required_inputs()
        if not required.issubset(self._fresh_inputs):
            missing = tuple(sorted(required - self._fresh_inputs))
            if missing not in self._missing_logged:
                self._missing_logged.add(missing)
                self._log(f"帧不完整，等待输入: {list(missing)}")
            return
        obs = {
            "state": self._state,
            "state_names": list(self._state_names),
            "images": dict(self._images),
            "depths": dict(self._depths),
            "segs": dict(self._segs),
            "scene_objects": self._scene_objects,
            "camera_info": self._camera_info,
        }
        self._last_complete_obs = obs
        self._step_on(obs)

    def _retry_last_obs(self) -> None:
        """命令（指令/启动）到达后重试缓存的完整观测帧。"""
        if self._phase != POLICY_STATUS_RUNNING or self._failed:
            return
        if self._last_complete_obs is not None:
            self._step_on(self._last_complete_obs)

    def _step_on(self, obs: dict[str, Any]) -> None:
        self._fresh_inputs.clear()
        try:
            action = self.policy.step(obs)
        except PolicyError as exc:
            if exc.code in RETRYABLE_ERROR_CODES:
                # 跨通道乱序导致的暂时缺料：不判死，等命令重试
                self._log(f"暂不能推理（{exc.code}），缓存观测等待命令")
                return
            self._failed = True
            self._log(f"策略失败: {exc.code}: {exc}")
            self._emit_status(POLICY_STATUS_FAILED, error_code=exc.code, message=str(exc))
            return
        if action is None:
            return
        self._last_complete_obs = None

        # 解析 step() 的返回形态：
        #   ndarray            → (动作, 非末动作)（旧用法，零改动兼容）
        #   (ndarray, bool)    → (动作, is_last)——is_last=True 的动作由
        #                        runtime 自动附 effort 哨兵（推荐用法）
        #   其它               → 结构化失败（错误码 invalid_step_return）
        is_last = False
        if isinstance(action, tuple):
            if (
                len(action) == 2
                and isinstance(action[1], (bool, np.bool_))
                and isinstance(action[0], np.ndarray)
            ):
                vec_raw, is_last = action[0], bool(action[1])
            else:
                self._failed = True
                self._log(f"step 返回的元组形态非法: {type(action[0]).__name__}, {type(action[1]).__name__}"
                          if len(action) == 2 else f"step 返回的元组长度非法: {len(action)}")
                self._emit_status(
                    POLICY_STATUS_FAILED,
                    error_code="invalid_step_return",
                    message="step 应返回 np.ndarray、None 或 (np.ndarray, bool) 元组",
                )
                return
        else:
            vec_raw = action

        vec = np.asarray(vec_raw, dtype=np.float64).reshape(-1)
        if vec.size != self.contract.action_dim:
            self._failed = True
            self._emit_status(
                POLICY_STATUS_FAILED,
                error_code="action_dim_mismatch",
                message=f"期望 {self.contract.action_dim} 维，得到 {vec.size}",
            )
            return
        if not np.all(np.isfinite(vec)):
            self._failed = True
            self._emit_status(
                POLICY_STATUS_FAILED, error_code="non_finite_action", message="动作含非有限值"
            )
            return
        self._send_action(vec, is_last=is_last)
    def _send_action(self, vec: np.ndarray, is_last: bool = False) -> None:
        """发送动作；按三优先级决定是否附 effort 哨兵：

        1. chunk_per_action=True        → 每帧制策略，每个动作都是 N=1 chunk
                                          的末动作，恒附哨兵；
        2. 元组返回的 is_last=True      → chunk 制策略声明本动作即 chunk 末
                                          （推荐用法，时机由接口保证）；
        3. _chunk_pending_end 置位      → deprecated 的手工置位路径，仅为
                                          兼容保留（见 mark_chunk_end）。
        """
        names = list(self.contract.action_names)
        cmd = JointCommand(
            name=names,
            position=[float(x) for x in vec],
            mode="position",
        )
        if self.contract.chunk_per_action or is_last or (
            self.contract.emit_chunk_end and self._chunk_pending_end
        ):
            cmd.effort = [CHUNK_END_SENTINEL] * len(names)
            self._chunk_pending_end = False
        self.node.send_output("action", cmd.to_arrow())
        # 身份标签随动作同拍发出（协议 v2）：评测端按 episode_id 校验，异回合
        # 动作机械丢弃。动作消息本身是固定 Arrow 格式不能内嵌，走独立 JSON 通道。
        self._emit_action_meta()
        self._sent += 1
        if self._sent == 1 or self._sent % 50 == 0:
            self._log(f"已发送 {self._sent} 个动作")

    def mark_chunk_end(self) -> None:
        """[DEPRECATED] 手工声明"上一个动作是 chunk 末动作"。

        旧机制：置位后下一次发送附 effort 哨兵。只有在 step() **返回末动作
        之前**调用才能得到正确语义（哨兵随末动作发出）；在返回后调用会把
        哨兵错位到下一 chunk 的第一个动作——这正是本方法被替代的原因：
        正确性完全依赖调用方手工把握时机。

        新用法：chunk 制策略在 step() 里直接 `return (action, is_last)`，
        runtime 自动在 is_last=True 的动作上打哨兵；每帧制策略在 contract
        设 chunk_per_action=True。本方法仅为兼容保留，不再建议调用。
        """
        self._chunk_pending_end = True

    # ---------------------------------------------------------------- run
    def run(self) -> int:
        # ready + 契约自报（协议 v2）：评测端握手门禁的数据源。policy_id 等
        # 由 set_handshake() 填写；contract 的字段作为默认兜底。
        ready_payload: dict[str, Any] = {
            "action_names": list(self.contract.action_names),
            "cameras": list(self.contract.cameras),
            "emit_chunk_end": self.contract.emit_chunk_end,
            "chunk_per_action": self.contract.chunk_per_action,
            **self._handshake_payload,
        }
        self._emit_status("ready", **ready_payload)
        self._emit_status(POLICY_STATUS_IDLE)
        self._log(
            f"就绪：动作 {self.contract.action_dim} 维 "
            f"{list(self.contract.action_names)}；相机 {list(self.contract.cameras)}；"
            f"depth {list(self.contract.depth_cameras)}；seg {list(self.contract.seg_cameras)}"
        )
        started = time.monotonic()
        try:
            for event in self.node:
                if event["type"] == "STOP":
                    break
                if event["type"] != "INPUT":
                    continue
                eid = event["id"]
                if eid == "policy_command":
                    cmd = PolicyCommand.from_arrow(event["value"])
                    if not self._handle_command(cmd):
                        break
                    continue
                if eid == "obs_meta":
                    # 观测身份标签（协议 v2）：记录当前回合的 episode_id + seq，
                    # 下一个动作经 action_meta 回带——异回合动作被评测端丢弃。
                    try:
                        meta = json.loads(_payload_text(event["value"]))
                        self._episode_id = meta.get("episode_id")
                        self._obs_seq = meta.get("seq")
                    except Exception as exc:  # noqa: BLE001
                        self._log(f"obs_meta 解析失败: {exc}")
                    continue
                if eid == "proprio_state":
                    state = JointState.from_arrow(event["value"])
                    self._state = np.asarray(state.position, dtype=np.float64)
                    self._state_names = list(state.name)
                    self._try_step()
                elif eid in self.contract.cameras:
                    img = Image.from_arrow(event["value"]).to_numpy()
                    self._images[eid] = np.ascontiguousarray(img, dtype=np.uint8)
                    self._fresh_inputs.add(eid)
                    self._try_step()
                elif eid in self.contract.depth_cameras:
                    img = Image.from_arrow(event["value"])
                    self._depths[eid] = img.to_numpy()
                    self._fresh_inputs.add(eid)
                    self._try_step()
                elif eid in self.contract.seg_cameras:
                    img = Image.from_arrow(event["value"]).to_numpy()
                    self._segs[eid] = img
                    self._fresh_inputs.add(eid)
                    self._try_step()
                elif eid == "scene_objects" and self.contract.scene_objects:
                    try:
                        self._scene_objects = json.loads(_payload_text(event["value"]))
                    except Exception as exc:  # noqa: BLE001
                        self._log(f"scene_objects 解析失败: {exc}")
                        continue
                    self._fresh_inputs.add(eid)
                    self._try_step()
                elif eid == "camera_info" and self.contract.camera_info:
                    try:
                        self._camera_info = json.loads(_payload_text(event["value"]))
                    except Exception as exc:  # noqa: BLE001
                        self._log(f"camera_info 解析失败: {exc}")
                        continue
                    self._fresh_inputs.add(eid)
                    self._try_step()
                # tick 等其它输入：心跳，不推理
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.policy.shutdown()
            except Exception:  # noqa: BLE001 退出路径不再抛
                pass
        self._log(
            f"退出，共发出 {self._sent} 个动作，运行 {time.monotonic() - started:.1f}s"
        )
        return 0


def validate_dataflow_hint() -> None:
    """启动时提醒 dataflow 必须给 policy_command 足够队列深度（见 contract 注释）。"""
    print(
        f"[policy_runtime] 提示：policy_command 输入 queue_size 应 ≥ {MIN_POLICY_COMMAND_QUEUE}",
        file=sys.stderr,
    )
