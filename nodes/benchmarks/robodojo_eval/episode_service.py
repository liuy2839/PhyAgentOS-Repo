"""评测发起节点的单回合服务逻辑 —— 纯逻辑状态机，dora / Isaac 都不 import。

一次 `eval_command` = 一个完整回合（接口①）：

    {"command": "start_episode", "task": "arrange_largest_number",
     "layout_id": 0, "round": 2, "instruction": "...（可省）",
     "output_dir": "/abs/dir（可省）"}

    {"command": "stop", "reason": "..."}

回合结束按接口② 经 `eval_result` 发出结果 JSON，并落盘
`<output_dir>/episode_result.json`；三路相机各落一个 mp4（videos 字段）。

设计照搬原 forge 流程 serve.py 的可测形态：本类不碰 dora/Isaac，事件由外面
喂（`process_event`），`run()` 只是 dora 事件循环的薄壳。单测用假 node +
假 env 就能覆盖整个状态机。

与原流程的协议差别（解耦点）：
  * `eval_command` / `eval_status` / `eval_result` 取代 bench_tool 的
    bench_command / benchmark_status / benchmark_result —— 不再有 forge-tool
    工具面、gateway、session/租约那套；
  * 批次（task_ids × init_state_ids × num_runs）不复存在：编排（模块 D）
    逐回合发命令，指令逐回合给定 —— 这正是 v2 文档接口① 的形状。

policy_command 序列（回合边界）：
    start → set_instruction(占位) → reset_scene → [reset 后] set_instruction(最终)
`start` 不能漏：策略端（裸节点与 forge_policy SDK 皆然）只在 running 相位才
推理，收不到 start 就永远不动作、双向等死。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# eval_command 的载荷类型
CMD_START = "start_episode"
CMD_STOP = "stop"
CMD_RUN_BATCH = "run"           # 批次命令（batch.py 展开，逐回合自动驱动）
CMD_DESCRIBE = "describe"       # 能力快照（Query 语义，工具接口的 describe）
CMD_STATUS = "status"           # 进度查询（run Session 的 status）

# 与原 forge 流程逐字相同的哨兵值（policy 把 chunk 末动作的 effort 整条填成
# 它，评测侧据此补那次 is_episode_end()）—— 两边必须一致，协议不改。
CHUNK_END_SENTINEL = 1.0

TERMINATION_DONE = "done"          # episode 正常结束（成败由 success 字段区分）
TERMINATION_TIMED_OUT = "timed_out"
TERMINATION_STOPPED = "stopped"
TERMINATION_FAILED = "failed"

STATUS_SUCCEEDED = "succeeded"     # 回合跑完了（任务成败看 success）
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"


def _is_timeout(event: Any) -> bool:
    """dora 把 `next(timeout=)` 的超时表达成 ERROR 事件，不是返回 None。"""
    if not isinstance(event, dict) or event.get("type") != "ERROR":
        return False
    return "timed out" in str(event.get("error", "")).lower()


def _decode_text(value: Any) -> str | None:
    """dora 文本通道的载荷可能是 pyarrow string array / bytes / str。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if hasattr(value, "as_py"):
        value = value.as_py()
        return value if isinstance(value, str) else None
    try:
        import pyarrow as pa

        if isinstance(value, pa.Array):
            inner = value[0].as_py()
            return inner if isinstance(inner, str) else None
    except Exception:  # noqa: BLE001
        pass
    return None


class TeeLog:
    """stdout + 回合日志文件双写（接口② 的 log 字段）。"""

    def __init__(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self._fh.write(str(msg) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class _Episode:
    """一个正在跑的回合的全部状态。"""

    task: str
    layout_id: int
    round: int | None
    given_instruction: str | None
    output_dir: str
    max_steps: int
    effective_instruction: str = ""
    steps: int = 0
    final_score: float | None = None
    latencies: list[float] = field(default_factory=list)
    obs_sent_at: float | None = None
    awaiting_action: bool = False
    recorder: Any = None            # PerCameraRecorder
    tee: Any = None                 # TeeLog
    log_path: str = ""
    finished: bool = False
    # ---- 协议层（episode_id + seq + 看门狗放弃上限）----
    episode_id: str = ""
    seq: int = 0                    # 当前已发布的观测序号（下一帧 = seq）
    republishes: int = 0            # 当前序号上的观测重发次数
    dropped_actions: int = 0        # 被入口/身份校验拒绝的动作计数
    step_started_at: float | None = None   # 本步开始等待的时刻（放弃上限计时）
    last_status: str = ""           # 终局状态（批次摘要用）
    last_success: bool = False


class EpisodeDriver:
    """单回合状态机。**不碰 tool 协议 / dora / Isaac**，事件由外面喂，可以单测。

    `env_factory` 是延迟环境工厂：首次 start_episode 才建 RoboDojoEpisodeEnv
    （Isaac 已由 main.py 起好）。env 跨回合复用 —— 换 task 时 env 内部自己重建，
    同 task 复用（与官方批次间 close 语义一致，见 robodojo_env 注释）。

    协议状态（与策略节点共同遵守，见 handshake.py）：
      * `handshake`：策略 ready 自报的契约；回合启动前必须已通过校验；
      * `reset_acked`：策略对最近一次 reset_scene 的确认（收到后才发首帧——
        从结构上消除"观测先于 reset 被消费、被清空"的跨通道乱序死锁；
        确认超时由 `_RESET_ACK_TIMEOUT_S` 兜底，退化为旧行为）；
      * `policy_failed`：策略上报 failed（error_code）——回合立即进入明确
        失败终态，不再无限等待一个死掉的策略。
    """

    # reset 确认等待上限（秒）。正常几十毫秒；策略还在推理上一回合尾帧时可能
    # 稍慢。超时则按旧协议继续（发首帧 + 看门狗重发兜底），不因此挂死。
    _RESET_ACK_TIMEOUT_S = 5.0

    def __init__(
        self,
        node: Any,
        cfg: Any,
        env_factory: Callable[[Callable[[str], None]], Any],
        log: Callable[[str], None] = print,
    ) -> None:
        self.node = node
        self.cfg = cfg
        self._make_env = env_factory
        self._log = log
        self.env: Any = None
        self.episode: _Episode | None = None
        self.batch: Any = None              # BatchRunState（批次进行中非 None）
        # ---- 协议状态 ----
        self.handshake: Any = None          # PolicyHandshake（校验通过后）
        self.handshake_errors: list[str] = []
        self.reset_acked: bool = False
        self.reset_pending_since: float | None = None
        self.policy_failed: dict[str, Any] | None = None   # {"error_code": ..., "message": ...}
        self.policy_stopped: bool = False
        self._episode_counter = 0
        self._pending_first_obs: dict[str, Any] | None = None   # reset_ack 前暂存的首帧
        self._last_action_meta: dict[str, Any] | None = None    # 最近一条 action_meta
        # 动作入口（《修改方案》：停止时关闭动作入口）——Driver 级状态，跨回合
        # 持有：stop 置 False 后，迟到的动作从状态上不可能再被消费（不是碰巧
        # 被丢）；下一条 start/run 显式重开。start 时若入口是关的说明上一场
        # 停过，重开是自然语义。
        self.admission_open = True
        self.admission_dropped = 0      # 入口关闭期间拒绝的动作计数

    # ---- 协议：policy_status 处理 ----

    def on_policy_status(self, value: Any) -> None:
        """策略节点的状态/应答通道（协议层主角）。"""
        text = _decode_text(value)
        if text is None:
            return
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            self._log(f"忽略非法 policy_status：{text[:80]!r}")
            return
        status = payload.get("status")
        if status == "ready":
            self._on_ready(payload)
        elif status in ("failed", "error"):
            self.policy_failed = {
                "error_code": payload.get("error_code", "policy_failed"),
                "message": payload.get("message", ""),
            }
            # 回合进行中：立即收尾（明确终态），不再等一个死掉的策略
            if self.episode is not None and not self.episode.finished:
                self._fail_current(
                    "policy_failed",
                    f"策略失败：{self.policy_failed['error_code']} {self.policy_failed['message']}",
                )
        elif status == "stopped":
            self.policy_stopped = True
        elif status == "reset_ack":
            self.reset_acked = True
            self.reset_pending_since = None
            # ack 到达即发布暂存的首帧（不等下一个看门狗超时——那是兜底路径）
            self._maybe_publish_first()

    def _on_ready(self, payload: dict[str, Any]) -> None:
        """握手：契约自报校验。通过才允许回合启动；不通过记录错误并拒绝。"""
        from handshake import PolicyHandshake, check_handshake

        try:
            hs = PolicyHandshake.from_status_payload(payload)
        except ValueError as exc:
            self.handshake = None
            self.handshake_errors = [str(exc)]
            self._log(f"握手载荷非法：{exc}")
            return
        result = check_handshake(hs, self.cfg.camera_map)
        if result.ok:
            self.handshake = hs
            self.handshake_errors = []
            self._log(
                f"握手通过：policy_id={hs.policy_id} action_type={hs.action_type} "
                f"动作 {len(hs.action_names)} 维 相机 {len(hs.cameras)} 路"
                + (f" weight_id={hs.weight_id}" if hs.weight_id else "")
            )
            for w in result.warnings:
                self._log(f"握手警告：{w}")
        else:
            self.handshake = None
            self.handshake_errors = result.errors
            for e in result.errors:
                self._log(f"握手不通过：{e}")

    # ---- 出站 ----

    def _emit(self, channel: str, payload: dict[str, Any]) -> None:
        import pyarrow as pa

        self.node.send_output(channel, pa.array([json.dumps(payload, ensure_ascii=False)]))

    def _emit_status(self, payload: dict[str, Any]) -> None:
        self._emit("eval_status", payload)
        self._log(json.dumps(payload, ensure_ascii=False))

    def _policy_command(self, command: str, inputs: dict[str, Any] | None = None) -> None:
        from messages import PolicyCommand

        msg = PolicyCommand.from_inputs(policy_id=self.cfg.policy_id, command=command, inputs=inputs or {})
        self.node.send_output("policy_command", msg.to_arrow())

    def _log_line(self, msg: str) -> None:
        ep = self.episode
        if ep is not None and ep.tee is not None:
            ep.tee(msg)
        else:
            self._log(msg)

    # ---- 观测发布 ----

    def _publish_obs(self, obs: dict[str, Any]) -> None:
        import pyarrow as pa

        from messages import Image, JointState

        ep = self.episode
        env = self.env
        state = env.to_state(obs)
        self.node.send_output(
            "proprio_state",
            JointState.from_np(
                state, env.state_joint_names(), "position"
            ).to_arrow(),
        )
        frames = env.to_images(obs)
        for output_id, frame in frames.items():
            self.node.send_output(output_id, Image.from_numpy(frame, "rgb8").to_arrow())
        # 观测身份标签（协议层）：与观测同拍发出，策略端据此在动作上回带同标签。
        # forge_msgs 的 JointState/Image 是固定字段格式不能内嵌，所以走独立 JSON
        # 通道（obs_meta）；同拍性由本函数内先后 send + dora 单发送端保序保证。
        self._emit("obs_meta", {
            "episode_id": ep.episode_id,
            "seq": ep.seq,
        })
        # 可选观测通道：env 提供 to_depths/to_segs/to_scene_objects/to_camera_info
        # 且 cfg.publish_* 开启才输出（编码约定同原 runner：32FC1 米 / 16UC1 标签 /
        # JSON 文本）。默认全关，FastWAM 链路行为不变；ReKep 类策略开 scene_objects。
        _depths = getattr(env, "to_depths", None)
        if callable(_depths):
            for output_id, depth in _depths(obs).items():
                depth32 = np.ascontiguousarray(depth, dtype=np.float32)
                self.node.send_output(
                    output_id, Image.from_numpy(depth32, "32FC1").to_arrow()
                )
        _segs = getattr(env, "to_segs", None)
        if callable(_segs):
            for output_id, seg in _segs(obs).items():
                seg16 = np.ascontiguousarray(seg, dtype=np.uint16)
                self.node.send_output(
                    output_id, Image.from_numpy(seg16, "16UC1").to_arrow()
                )
        _objs = getattr(env, "to_scene_objects", None)
        if callable(_objs):
            scene_objects = _objs(obs)
            if scene_objects:
                self.node.send_output(
                    "scene_objects",
                    pa.array([json.dumps(scene_objects, ensure_ascii=False)]),
                )
        _cam = getattr(env, "to_camera_info", None)
        if callable(_cam):
            camera_info = _cam(obs)
            if camera_info:
                self.node.send_output(
                    "camera_info",
                    pa.array([json.dumps(camera_info, ensure_ascii=False)]),
                )
        ep.recorder.add(frames)
        ep.obs_sent_at = time.time()
        ep.awaiting_action = True

    def _republish_if_stalled(self, timeout_s: float) -> None:
        """等 action 太久就重发同一帧观测；超过放弃上限则回合 failed 终态。

        为什么需要重发：`policy_command`（reset_scene/set_instruction）与
        `proprio_state`/`image/*` 是**不同 dora 通道，跨通道到达顺序没有保证**。
        观测若先于 reset_scene 被 policy 取出，会被 reset 处理清掉 → policy 等
        下一帧、本节点等 action，双向死锁且不报错。重发幂等（obs_meta 带同一
        seq，策略端按 seq 识别"这是同一帧"）。

        放弃上限（《修改方案》：等待超时进入明确终态）：同一序号上重发次数或
        累计等待时间超过阈值，判定策略已死（区别于"推理慢"），回合 failed
        收尾并留下证据（等了多久 / 重发几次 / 卡在哪个 seq）。两个阈值都来自
        config（give_up_after_s / max_republishes；0 = 不限，保持旧的无穷耐心）。
        """
        ep = self.episode
        if ep is None or ep.finished or not ep.awaiting_action:
            return
        if ep.obs_sent_at is None:
            return
        waited = time.time() - ep.obs_sent_at
        if waited < timeout_s:
            return
        # ---- 放弃上限检查（先于重发）----
        give_up_after_s = getattr(self.cfg, "give_up_after_s", 0) or 0
        max_republishes = getattr(self.cfg, "max_republishes", 0) or 0
        total_wait = time.time() - (ep.step_started_at or ep.obs_sent_at)
        if (give_up_after_s and total_wait >= give_up_after_s) or (
            max_republishes and ep.republishes >= max_republishes
        ):
            self._fail_current(
                "policy_wait_timeout",
                f"等待 action 超时放弃：seq={ep.seq} 已重发观测 {ep.republishes} 次、"
                f"累计等待 {total_wait:.0f}s（give_up_after_s={give_up_after_s} "
                f"max_republishes={max_republishes}）",
            )
            return
        ep.republishes += 1
        self._log_line(
            f"等 action 超过 {timeout_s:.0f}s，重发观测（seq={ep.seq}，"
            f"第 {ep.republishes} 次重发）"
        )
        obs = self.env.last_obs
        if obs:
            self._publish_obs(obs)

    # ---- 回合生命周期 ----

    def _start_episode(self, cmd: dict[str, Any]) -> None:
        if self.episode is not None and not self.episode.finished:
            self._emit_status({
                "command": "busy",
                "task": self.episode.task,
                "steps": self.episode.steps,
            })
            return

        # ---- 协议门禁：握手必须已通过（《修改方案》：校验契约与 ready 后启动回合）----
        if self.handshake is None:
            self._emit_status({
                "command": "rejected",
                "error": "策略握手未通过（未收到 ready 或契约校验失败）",
                "handshake_errors": self.handshake_errors,
            })
            return

        task = str(cmd.get("task") or "").strip()
        if not task:
            self._emit_status({"command": "rejected", "error": "eval_command 缺少 task"})
            return
        # 新回合 = 动作入口重开（上一场若 stop 过，这里自然恢复）
        self.admission_open = True
        # 参数三分（《修改方案》）：layout_id（场景选择）/ seed（eval_cfg 随机化，
        # 命令未给则用 cfg.seed）/ max_steps（步数预算，reset 后按任务 step_lim
        # 修正）。v2 接口① 的 "seed" 字段按布局 ID 同义处理（RoboDojo 特性），
        # 与 cfg.seed（过程随机化）是两个不同的旋钮。
        layout_id = cmd.get("layout_id")
        seed = cmd.get("seed")
        if layout_id is None and seed is not None:
            layout_id = seed
        if layout_id is not None and seed is not None and int(layout_id) != int(seed):
            self._emit_status({
                "command": "rejected",
                "error": f"layout_id={layout_id} 与 seed={seed} 不一致（RoboDojo 里二者同义）",
            })
            return
        layout_id = int(layout_id) if layout_id is not None else 0
        if cmd.get("max_steps") is not None:
            self.cfg.max_steps_default = int(cmd["max_steps"])
        round_no = cmd.get("round")
        given = cmd.get("instruction")
        given = str(given).strip() if isinstance(given, str) and given.strip() else None

        # 唯一回合目录（《修改方案》）：默认 tag 里带递增回合号，同命令重跑不覆盖。
        output_dir = cmd.get("output_dir")
        self._episode_counter += 1
        ep_counter = self._episode_counter
        if not output_dir:
            tag = f"{task}_r{round_no if round_no is not None else 0}_layout{layout_id}"
            output_dir = os.path.join(self.cfg.result_dir, f"{tag}", f"run{ep_counter:04d}")
        output_dir = os.path.abspath(str(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        # 环境惰性建立：首次命令才碰 Isaac（起 app 是 main.py 的事）。
        # 通用身份 → 握手覆盖（《修改方案》）：官方 env_cfg 组装需要 policy_name
        # （deploy_cfg 字段必填、影响官方结果文件目录归属），以握手自报为准，
        # 配置里的 "auto" 只是占位。env 持有 cfg 的活引用，改 cfg 即生效。
        if self.handshake is not None and self.handshake.policy_id:
            self.cfg.policy_id = self.handshake.policy_id
        if self.env is None:
            self.env = self._make_env(self._log)

        # 回合身份（协议层）：动作必须回带它，不匹配的必然丢弃。
        episode_id = f"{task}-L{layout_id}-{ep_counter:04d}"

        tee = TeeLog(os.path.join(output_dir, "episode.log"))
        tee(
            f"[robodojo_eval] 回合开始 task={task} layout_id={layout_id} "
            f"round={round_no} episode_id={episode_id} output_dir={output_dir}"
        )
        if given:
            tee(f"[robodojo_eval] 指定指令：{given}")

        recorder_cls = self._recorder_class()
        ep = _Episode(
            task=task,
            layout_id=layout_id,
            round=int(round_no) if round_no is not None else None,
            given_instruction=given,
            output_dir=output_dir,
            # step_lim 由 env 在 reset 后按任务回填（robodojo_env.reset_episode）；
            # 这里先给兜底值。
            max_steps=int(self.cfg.max_steps_default),
            recorder=recorder_cls(os.path.join(output_dir, "videos"), fps=30.0, log=tee),
            tee=tee,
            log_path=os.path.join(output_dir, "episode.log"),
            episode_id=episode_id,
        )
        self.episode = ep
        self.policy_failed = None
        self.policy_stopped = False

        try:
            self.env.set_log(tee)
            # 命令序列（顺序照原 runner，start 不能漏 —— 策略端只在 running
            # 相位推理）：
            #   1. start：进 running 相位
            #   2. set_instruction：占位（给定值或任务名）
            #   3. reset_scene：清策略上一回合的动作队列与观测缓存
            #   4. [reset 确认后] set_instruction：最终指令 + 发首帧
            # 协议层新增：reset_scene 之后**等策略的 reset_ack**（policy_status
            # 通道）再发首帧 —— 观测先于 reset 被消费/清空的乱序死锁从结构上
            # 消除。确认等待在下面的循环里做（事件循环继续处理 policy_status）。
            self.reset_acked = False
            self.reset_pending_since = time.time()
            self._policy_command("start")
            self._policy_command("set_instruction", {"instruction": given or task})
            self._policy_command("reset_scene")

            obs = self.env.reset_episode(task, layout_id)
            if self.env.episode_step_lim:
                ep.max_steps = int(self.env.episode_step_lim)

            # 最终防线（协议 v2）：env 已建，用活推导的分量名表做终检——
            # 静态快检挡不住的（env_cfg_type 变更、机器人换代）在这里兜住。
            # 不通过 → 回合 failed（此时尚未发首帧，成本只有 reset）。
            if self.handshake is not None and hasattr(self.env, "action_component_names"):
                from handshake import check_handshake_live

                live = check_handshake_live(
                    self.handshake,
                    self.cfg.camera_map,
                    live_action_names=self.env.action_component_names(),
                    live_state_names=self.env.state_joint_names(),
                )
                if not live.ok:
                    for e in live.errors:
                        self._log_line(f"活校验不通过：{e}")
                    self._finalize(
                        TERMINATION_FAILED,
                        status=STATUS_FAILED,
                        error_code="handshake_live_mismatch",
                        error_message="；".join(live.errors),
                    )
                    return

            env_instr = self.env.env_instruction
            if given:
                ep.effective_instruction = given
                if env_instr and env_instr != given:
                    tee(
                        f"[robodojo_eval] 给定指令覆盖环境指令（环境原指令：{env_instr}）"
                    )
                self._policy_command("set_instruction", {"instruction": given})
            elif env_instr:
                ep.effective_instruction = env_instr
                self._policy_command("set_instruction", {"instruction": env_instr})
            else:
                ep.effective_instruction = task

            # reset 确认协议：发完 reset_scene 后**暂缓首帧**，进入
            # `waiting_reset_ack` 相位。事件循环的看门狗分支（process_event →
            # on_timeout）负责检查：ack 到了 → 发首帧；超过 _RESET_ACK_TIMEOUT_S
            # 没到 → 按旧协议直接发首帧（看门狗重发兜底）。不能在这里 sleep 等待
            # —— 阻塞的是 dora 事件循环，policy_status 就永远到不了。
            self._emit_status({
                "command": "started",
                "task": task,
                "layout_id": layout_id,
                "round": ep.round,
                "episode_id": episode_id,
                "max_steps": ep.max_steps,
                "instruction": ep.effective_instruction,
                "output_dir": output_dir,
            })
            self._log_line(
                f"回合就绪（episode_id={episode_id}），等待 reset 确认后发首帧，"
                f"指令：{ep.effective_instruction}，max_steps={ep.max_steps}"
            )
            self._pending_first_obs = obs      # 暂存首帧，reset_ack 后由 _maybe_publish_first 发布
        except Exception as exc:  # noqa: BLE001 启动失败收敛成 failed 回合，不冲出循环
            self._log_line(f"回合启动失败：{type(exc).__name__}: {exc}")
            self._finalize(
                TERMINATION_FAILED,
                status=STATUS_FAILED,
                error_code="episode_start_error",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def _maybe_publish_first(self) -> None:
        """reset 确认协议的发布点：ack 到了 / 超时了，二选一发布暂存的首帧。"""
        ep = self.episode
        obs = getattr(self, "_pending_first_obs", None)
        if ep is None or ep.finished or obs is None or ep.awaiting_action:
            return
        if self.reset_acked:
            self._log_line("已收到 reset 确认，发布首帧")
        else:
            waited = time.time() - (self.reset_pending_since or time.time())
            if waited < self._RESET_ACK_TIMEOUT_S:
                return        # 还没到兜底时限，继续等 ack
            self._log_line(
                f"reset 确认等待超时（{waited:.0f}s），按旧协议发布首帧（看门狗重发兜底）"
            )
        self._pending_first_obs = None
        self.reset_pending_since = None
        ep.step_started_at = time.time()
        self._publish_obs(obs)

    def _recorder_class(self) -> type:
        from episode_recorder import PerCameraRecorder

        return PerCameraRecorder

    def _finalize(
        self,
        termination: str,
        *,
        status: str,
        success: bool = False,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        ep = self.episode
        if ep is None or ep.finished:
            return
        ep.finished = True

        videos: dict[str, str] = {}
        if ep.recorder is not None:
            videos = ep.recorder.save()
        if ep.tee is not None:
            ep.tee(
                f"[robodojo_eval] 回合结束 termination={termination} success={success} "
                f"score={ep.final_score} steps={ep.steps}"
            )
            ep.tee.close()

        result_path = os.path.join(ep.output_dir, "episode_result.json")
        mean_ms = (
            sum(ep.latencies) / len(ep.latencies) if ep.latencies else None
        )
        # 实际生效参数（《修改方案》：结果保存实际生效参数）——命令给的是"要的"，
        # 这里记录"跑的"：布局（env 可能按清单解析）、步数上限（按任务 step_lim
        # 修正）、随机化 seed、握手到的策略身份。
        effective: dict[str, Any] = {
            "layout_id": ep.layout_id,
            "max_steps": ep.max_steps,
            "eval_cfg_seed": getattr(self.cfg, "seed", None),
            "instruction": ep.effective_instruction,
        }
        if self.handshake is not None:
            effective["policy_id"] = self.handshake.policy_id
            effective["weight_id"] = self.handshake.weight_id
        # 官方原生结果文件（《修改方案》：结果保存官方 metrics）——RoboDojo 官方
        # eval 链路的 _result.json 是判分的原始证据，路径收进结果供复核。
        native = None
        if self.env is not None and hasattr(self.env, "native_result_path"):
            try:
                native = self.env.native_result_path()
            except Exception:  # noqa: BLE001
                native = None
        payload: dict[str, Any] = {
            # ---- v2 文档接口② 的最小集合 ----
            "task": ep.task,
            "round": ep.round,
            "layout_id": ep.layout_id,
            "success": bool(success),
            "score": ep.final_score,
            "steps": ep.steps,
            "instruction": ep.effective_instruction,
            "videos": videos,
            "result_json": result_path,
            "log": ep.log_path,
            # ---- 补充字段（编排/反思可直接用，也可忽略）----
            "status": status,
            "termination": termination,
            "episode_id": ep.episode_id,
            "env_instruction": (
                self.env.env_instruction
                if self.env is not None and hasattr(self.env, "env_instruction")
                else None
            ),
            "mean_policy_latency_ms": mean_ms,
            "reproducibility": (
                self.env.reproducibility()
                if self.env is not None and hasattr(self.env, "reproducibility")
                else {}
            ),
            "effective": effective,
            "native_result_json": native,
            "protocol": {
                "handshake_ok": self.handshake is not None,
                "policy_id": self.handshake.policy_id if self.handshake else None,
                "reset_acked": bool(self.reset_acked),
                "dropped_actions": ep.dropped_actions,
                "republishes_total": ep.republishes,
            },
            "error_code": error_code,
            "error_message": error_message,
        }
        # 原子写入（《修改方案》）：先写临时文件、写完 os.replace 到位——最终
        # 路径上要么是完整旧文件、要么是完整新文件，永远不会出现写一半的残缺
        # JSON（比没有文件更糟：它看起来像结果）。写失败作为明确错误上抛到
        # 结果消息，不再静默吞掉后照常"交付"。
        write_error: str | None = None
        try:
            tmp_path = result_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, result_path)
        except Exception as exc:  # noqa: BLE001
            write_error = f"{type(exc).__name__}: {exc}"
            self._log(f"结果落盘失败：{write_error}")
        if write_error is not None:
            payload["status"] = STATUS_FAILED
            payload["error_code"] = error_code or "result_write_error"
            payload["error_message"] = error_message or write_error

        self._emit("eval_result", payload)
        self._log(
            f"回合结束 status={payload['status']} success={success} score={ep.final_score} "
            f"steps={ep.steps} → {result_path}"
        )
        ep.last_status = payload["status"]
        ep.last_success = bool(success)
        self.episode = None
        # 批次进行中：记录摘要并驱动下一回合（或收批次）。注意在 episode=None
        # 之后调用——下一回合的 _start_episode 需要"当前无回合"状态。
        self._advance_batch(ep)

    # ---- 入站 ----

    def on_eval_command(self, text: str) -> int | None:
        """处理一条 eval_command。返回非 None 表示进程该退出。"""
        try:
            cmd = json.loads(text)
        except (TypeError, ValueError) as exc:
            self._log(f"忽略非法 eval_command：{exc}")
            return None
        if not isinstance(cmd, dict):
            self._log("eval_command 必须是 JSON 对象，忽略")
            return None
        kind = cmd.get("command")
        if kind == CMD_START:
            self._start_episode(cmd)
            return None
        if kind == CMD_STOP:
            if self.episode is None or self.episode.finished:
                return None
            # 关闭动作入口（《修改方案》：停止时关闭动作入口）——Driver 级
            # 状态，跨回合持有：此后迟到的动作从状态上不可能再被消费。
            self.admission_open = False
            self.episode.admission_open = False
            self._log_line(f"收到 stop（{cmd.get('reason')}），动作入口已关闭，收尾本回合")
            # 批次进行中：回合照常收尾（stopped），但批次不再开下一回合
            if self.batch is not None:
                self.batch.status = "stopped"
            self._finalize(TERMINATION_STOPPED, status=STATUS_STOPPED)
            return None
        if kind == CMD_RUN_BATCH:
            return self._start_batch(cmd)
        if kind == CMD_DESCRIBE:
            self._emit("eval_status", {"command": "describe", **self._describe_payload()})
            return None
        if kind == CMD_STATUS:
            self._emit("eval_status", {"command": "status", **self._status_payload()})
            return None
        self._log(f"未知 eval_command：{kind!r}，忽略")
        return None

    # ---- 工具接口（describe / status）----

    def _describe_payload(self) -> dict[str, Any]:
        """能力快照（describe Query 语义）：环境契约 + 任务清单 + 运行状态。

        数据源是 handshake.env_contract()（相机/状态/动作空间集合/判分口径）
        ——策略端据此可在启动前核对自身需求，编排层据此构造 run。
        """
        from handshake import env_contract

        # env 已建时用活推导（真源）；未建时 env_contract 自动退到静态快照。
        live_a = self.env.action_component_names() if (
            self.env is not None and hasattr(self.env, "action_component_names")
        ) else None
        live_s = self.env.state_joint_names() if (
            self.env is not None and hasattr(self.env, "state_joint_names")
        ) else None
        payload: dict[str, Any] = {
            "benchmark": self.cfg.benchmark,
            "env_cfg_type": self.cfg.env_cfg_type,
            "action_type_default": self.cfg.action_type,
            "camera_map": dict(self.cfg.camera_map),
            "env_contract": env_contract(
                self.cfg.camera_map,
                live_action_names=live_a,
                live_state_names=live_s,
                env_cfg_type=self.cfg.env_cfg_type,
            ),
            "init_states_per_task": int(getattr(self.cfg, "init_states_per_task", 50)),
            "defaults": {
                "num_runs": 1,
                "seed": int(getattr(self.cfg, "seed", 0)),
                "max_steps": int(getattr(self.cfg, "max_steps_default", 1050)),
                "give_up_after_s": float(getattr(self.cfg, "give_up_after_s", 900)),
            },
            "handshake_ok": self.handshake is not None,
            "policy": (
                {
                    "policy_id": self.handshake.policy_id,
                    "action_type": self.handshake.action_type,
                    "weight_id": self.handshake.weight_id,
                }
                if self.handshake is not None
                else None
            ),
        }
        tasks = self._task_inventory()
        if tasks:
            payload["tasks"] = tasks
        return payload

    def _status_payload(self) -> dict[str, Any]:
        """进度查询（run Session 的 status）：当前回合/批次进度。"""
        payload: dict[str, Any] = {
            "admission_open": self.admission_open,
            "handshake_ok": self.handshake is not None,
        }
        ep = self.episode
        if ep is not None:
            payload["episode"] = {
                "episode_id": ep.episode_id,
                "task": ep.task,
                "layout_id": ep.layout_id,
                "round": ep.round,
                "steps": ep.steps,
                "max_steps": ep.max_steps,
                "seq": ep.seq,
                "awaiting_action": ep.awaiting_action,
                "reset_acked": self.reset_acked,
            }
        if self.batch is not None:
            payload["batch"] = {
                "batch_id": self.batch.batch_id,
                "suite": self.batch.plan.suite,
                "status": self.batch.status,
                "completed_episodes": len(self.batch.episode_summaries),
                "total_episodes": self.batch.plan.total,
                "successes": sum(1 for s in self.batch.episode_summaries if s.get("success")),
            }
        if self.policy_failed is not None:
            payload["policy_failed"] = self.policy_failed
        return payload

    # ---- 批次层（评测调度：批次展开 → 逐回合驱动 → 汇总）----

    def _start_batch(self, cmd: dict[str, Any]) -> int | None:
        """run 命令：校验参数、展开回合队列、启动第一个回合。"""
        import uuid

        from batch import (
            BatchRunState,
            BatchValidationError,
            build_batch_plan,
            episode_command_from_plan,
        )

        if self.episode is not None and not self.episode.finished:
            self._emit_status({"command": "busy", "reason": "batch/episode running"})
            return None
        try:
            plan = build_batch_plan(
                cmd,
                all_tasks=self._task_inventory(cmd),
                num_layouts_per_task=int(getattr(self.cfg, "init_states_per_task", 50)),
                default_output_root=self.cfg.result_dir,
            )
        except BatchValidationError as exc:
            self._emit_status({"command": "rejected", "error": str(exc)})
            return None

        batch_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.batch = BatchRunState(plan=plan, batch_id=batch_id)
        self._log(
            f"批次 {batch_id}：suite={plan.suite} tasks={len(plan.tasks)} "
            f"布局 {len(plan.layout_ids)} × {plan.num_runs} 次 = {plan.total} 回合"
        )
        self._emit("eval_status", {
            "command": "batch_started",
            "batch_id": batch_id,
            "total_episodes": plan.total,
            "suite": plan.suite,
            "tasks": plan.tasks,
        })
        task, layout_id, run_index = self.batch.current()
        first = episode_command_from_plan(plan, task, layout_id, run_index, batch_id)
        self.on_eval_command(json.dumps(first, ensure_ascii=False))
        # 第一回合没起来（握手门禁/参数拒绝/env 建失败）→ 批次立即收尾，
        # 不能让它卡在 running 永远等一个不会来的回合终局。
        if self.batch is not None and self.episode is None:
            self.batch.status = "failed"
            from batch import write_batch_result

            path = write_batch_result(
                self.batch, os.path.join(
                    self.batch.plan.output_root, f"batch_result_{batch_id}.json")
            )
            self._emit("eval_result", {
                "kind": "batch",
                **self.batch.summarize(),
                "batch_result_json": path,
                "error": "第一回合启动被拒（见 eval_status）",
            })
            self._log(f"批次 {batch_id} 启动失败收尾 → {path}")
            self.batch = None
        return None

    def _task_inventory(self, cmd: dict[str, Any] | None = None) -> list[str]:
        """suite 内可用任务清单。优先级：命令显式给的 tasks > env 的 registry > 空。

        空清单 + suite 名本身会被 build_batch_plan 当作单任务处理——这使得
        "没接 Isaac（测试/离线）"或"任务清单暂不可得"时批次仍可按单任务跑。
        """
        if cmd and isinstance(cmd.get("tasks"), list) and cmd["tasks"]:
            return [str(t) for t in cmd["tasks"]]
        if self.env is not None and hasattr(self.env, "available_tasks"):
            try:
                tasks = self.env.available_tasks()
                if tasks:
                    return list(tasks)
            except Exception:  # noqa: BLE001
                pass
        return []

    def _advance_batch(self, ep: "_Episode") -> None:
        """回合终局后由 _finalize 调用：记录摘要、开下一回合或收批次。"""
        if self.batch is None:
            return
        from batch import episode_command_from_plan, write_batch_result

        self.batch.record_episode({
            "task": ep.task,
            "layout_id": ep.layout_id,
            "run": ep.round,
            "episode_id": ep.episode_id,
            "status": ep.last_status,
            "success": ep.last_success,
            "score": ep.final_score,
            "result_json": os.path.join(ep.output_dir, "episode_result.json"),
        })
        nxt = self.batch.current()
        if nxt is None or self.batch.status == "stopped":
            self.batch.status = self.batch.status if self.batch.status == "stopped" else "succeeded"
            path = write_batch_result(
                self.batch, os.path.join(self.batch.plan.output_root,
                                         f"batch_result_{self.batch.batch_id}.json")
            )
            self._emit("eval_result", {
                "kind": "batch",
                **self.batch.summarize(),
                "batch_result_json": path,
            })
            self._log(
                f"批次 {self.batch.batch_id} 结束：{self.batch.status}，"
                f"{self.batch.completed if hasattr(self.batch, 'completed') else len(self.batch.episode_summaries)}"
                f"/{self.batch.plan.total} → {path}"
            )
            self.batch = None
            return
        task, layout_id, run_index = nxt
        nxt_cmd = episode_command_from_plan(
            self.batch.plan, task, layout_id, run_index, self.batch.batch_id
        )
        self.on_eval_command(json.dumps(nxt_cmd, ensure_ascii=False))

    def on_action(self, value: Any, meta: dict[str, Any] | None = None) -> None:
        # 动作入口（《修改方案》：停止时关闭动作入口）：stop 后入口关死，
        # 迟到动作在此被拒并计数——从状态上不可能被消费，不是碰巧被丢。
        if not self.admission_open:
            self.admission_dropped += 1
            self._log(f"动作入口已关闭（stop 后），丢弃迟到动作（累计 {self.admission_dropped}）")
            return
        ep = self.episode
        # 非 1:1 的多余 action 直接丢弃：重发观测后策略会多推一次，不丢的话
        # env 会被多走一步；回合终局后 chunk 里排队中的旧动作同理。
        if ep is None or ep.finished or not ep.awaiting_action:
            if ep is not None and not ep.finished:
                ep.dropped_actions += 1
            return
        from messages import JointCommand

        try:
            command = JointCommand.from_arrow(value)
        except Exception as exc:  # noqa: BLE001
            self._fail_current("invalid_action", f"JointCommand 解析失败：{exc}")
            return

        action_raw = command.position
        if not action_raw:
            self._fail_current("invalid_action", "action 为空")
            return

        # 分量名校验：只读 position 会让"policy 把分量顺序发错"完全静默——
        # 机器人照动、每步都有值、分数悄悄归零。逐字比对，不一致宁可炸。
        expected_names = self.env.action_component_names()
        got = list(command.name or [])
        if got != expected_names:
            self._fail_current(
                "action_name_mismatch",
                f"动作分量名不匹配：期望 {len(expected_names)} 个分量"
                f"（{expected_names[:3]}...），收到 {len(got)} 个",
            )
            return

        # 回合身份校验（协议层）：动作回带的 episode_id 必须与当前回合一致。
        # action_meta 通道（JSON）与 action 同拍由策略端发出；旧回合迟到的
        # 动作在这里被机械拒绝——回合隔离漏洞（《修改方案》第 2 条）的封堵点。
        if meta is not None and meta.get("episode_id") not in (None, ep.episode_id):
            ep.dropped_actions += 1
            self._log_line(
                f"丢弃异回合动作（meta.episode_id={meta.get('episode_id')!r} ≠ "
                f"当前 {ep.episode_id!r}，累计丢弃 {ep.dropped_actions}）"
            )
            return

        ep.latencies.append((time.time() - ep.obs_sent_at) * 1000.0)
        ep.awaiting_action = False
        ep.republishes = 0            # 本序号已满足，放弃计数归零
        ep.seq += 1                   # 下一个观测的序号

        # chunk 边界标记：RoboDojo 的 is_episode_end() 带副作用，官方每个
        # chunk 在末动作之后、下一帧观测之前多查一次；policy 在 chunk 最后
        # 一个动作的 effort 填哨兵，这里交给 env.step() 补那次调用。
        effort = list(command.effort or [])
        chunk_end = bool(effort) and effort[0] == CHUNK_END_SENTINEL

        try:
            outcome = self.env.step(action_raw, chunk_end=chunk_end)
        except Exception as exc:  # noqa: BLE001 单步失败不该打死节点
            self._fail_current("step_error", f"{type(exc).__name__}: {exc}")
            return
        ep.steps += 1

        if outcome.done:
            ep.final_score = outcome.score
            self._finalize(
                TERMINATION_DONE,
                status=STATUS_SUCCEEDED,
                success=outcome.success,
            )
            return
        if ep.steps >= ep.max_steps:
            ep.final_score = outcome.score
            self._finalize(
                TERMINATION_TIMED_OUT,
                status=STATUS_SUCCEEDED,
                success=False,
            )
            return
        ep.step_started_at = time.time()
        self._publish_obs(outcome.obs)

    def _fail_current(self, code: str, message: str) -> None:
        self._log_line(f"回合失败：{code} {message}")
        self._finalize(
            TERMINATION_FAILED,
            status=STATUS_FAILED,
            error_code=code,
            error_message=message,
        )

    def on_timeout(self) -> None:
        # reset 确认协议的发布点：ack 已到或超时兜底到了，发布暂存的首帧
        self._maybe_publish_first()
        self._republish_if_stalled(self.cfg.watchdog_s)

    # ---- 事件循环 ----

    def process_event(self, event: Any) -> int | None:
        if event is None or _is_timeout(event):
            self.on_timeout()
            return None

        etype = event.get("type") if isinstance(event, dict) else None
        if etype == "STOP":
            if self.episode is not None and not self.episode.finished:
                self._finalize(TERMINATION_STOPPED, status=STATUS_STOPPED)
            return 0
        if etype == "ERROR":
            self._log(f"dora error: {event.get('error')}")
            if self.episode is not None and not self.episode.finished:
                self._finalize(
                    TERMINATION_FAILED,
                    status=STATUS_FAILED,
                    error_code="dora_error",
                    error_message=str(event.get("error")),
                )
            return 1
        if etype != "INPUT":
            return None

        eid = event.get("id")
        if eid == "eval_command":
            text = _decode_text(event.get("value"))
            if text is None:
                self._log(f"eval_command 载荷类型异常：{type(event.get('value'))}，忽略")
                return None
            return self.on_eval_command(text)
        if eid == "policy_status":
            self.on_policy_status(event.get("value"))
            # ready 可能在首帧暂存期间到达——不推进回合，但握手状态变了
            return None
        if eid == "action":
            self.on_action(event.get("value"))
            return None
        if eid == "action_meta":
            # 策略端随 action 同拍发出的身份标签（JSON）。dora 跨通道顺序无保证，
            # 稳妥做法：缓存最近一条 meta，供下一条 action 校验（action 先到就
            # 等 meta 一拍——由 pending_action_meta 的消费逻辑保证不卡死）。
            text = _decode_text(event.get("value"))
            try:
                self._last_action_meta = json.loads(text) if text else None
            except (TypeError, ValueError):
                self._last_action_meta = None
            return None
        # 其它输入只当心跳，绝不推进 episode
        return None

    def close(self) -> None:
        if self.episode is not None and not self.episode.finished:
            if self.episode.recorder is not None:
                self.episode.recorder.close()
            if self.episode.tee is not None:
                self.episode.tee.close()
            self.episode.finished = True
        if self.env is not None:
            self.env.close()

    def run(self) -> int:
        self._log(
            f"评测发起节点就绪：env_cfg={self.cfg.env_cfg_type} action_type={self.cfg.action_type} "
            f"policy_id={self.cfg.policy_id}，等待 eval_command"
        )
        rc = 0
        while True:
            # 必须带 timeout：空闲时无限阻塞，看门狗与 dora 心跳都无从驱动。
            event = self.node.next(timeout=self.cfg.watchdog_s)
            outcome = self.process_event(event)
            if outcome is not None:
                rc = outcome
                break
        return rc
