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


class EpisodeDriver:
    """单回合状态机。**不碰 tool 协议 / dora / Isaac**，事件由外面喂，可以单测。

    `env_factory` 是延迟环境工厂：首次 start_episode 才建 RoboDojoEpisodeEnv
    （Isaac 已由 main.py 起好）。env 跨回合复用 —— 换 task 时 env 内部自己重建，
    同 task 复用（与官方批次间 close 语义一致，见 robodojo_env 注释）。
    """

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
        """等 action 太久就重发同一帧观测。

        为什么需要：`policy_command`（reset_scene/set_instruction）与
        `proprio_state`/`image/*` 是**不同 dora 通道，跨通道到达顺序没有保证**。
        观测若先于 reset_scene 被 policy 取出，会被 reset 处理清掉 → policy 等
        下一帧、本节点等 action，双向死锁且不报错。重发幂等，代价只是策略多推
        一次；不重发的代价是整回合挂死。收齐观测后 policy 的 retryable
        NO_INSTRUCTION 机制也是同一道保险的另一端。
        """
        ep = self.episode
        if ep is None or ep.finished or not ep.awaiting_action:
            return
        if ep.obs_sent_at is None or (time.time() - ep.obs_sent_at) < timeout_s:
            return
        self._log_line(f"等 action 超过 {timeout_s:.0f}s，重发观测（step {ep.steps}）")
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

        task = str(cmd.get("task") or "").strip()
        if not task:
            self._emit_status({"command": "rejected", "error": "eval_command 缺少 task"})
            return
        # v2 接口① 同时给了 seed 和 layout_id —— RoboDojo 的 seed 就是布局 ID。
        # layout_id 优先；只给 seed 时按同义处理；两者都给且不同 → 拒绝（宁可炸掉）。
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
        round_no = cmd.get("round")
        given = cmd.get("instruction")
        given = str(given).strip() if isinstance(given, str) and given.strip() else None

        output_dir = cmd.get("output_dir")
        if not output_dir:
            tag = f"{task}_r{round_no if round_no is not None else 0}_layout{layout_id}"
            output_dir = os.path.join(self.cfg.result_dir, tag)
        output_dir = os.path.abspath(str(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        # 环境惰性建立：首次命令才碰 Isaac（起 app 是 main.py 的事）。
        if self.env is None:
            self.env = self._make_env(self._log)

        tee = TeeLog(os.path.join(output_dir, "episode.log"))
        tee(
            f"[robodojo_eval] 回合开始 task={task} layout_id={layout_id} "
            f"round={round_no} output_dir={output_dir}"
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
        )
        self.episode = ep

        try:
            self.env.set_log(tee)
            # 命令序列（顺序照原 runner，start 不能漏 —— 策略端只在 running
            # 相位推理）：
            #   1. start：进 running 相位
            #   2. set_instruction：占位（给定值或任务名）
            #   3. reset_scene：清策略上一回合的动作队列与观测缓存
            #   4. [reset 后] set_instruction：最终指令（给定值优先，否则环境
            #      按场景生成的原指令）—— RoboDojo 的指令 reset 前不存在
            self._policy_command("start")
            self._policy_command("set_instruction", {"instruction": given or task})
            self._policy_command("reset_scene")

            obs = self.env.reset_episode(task, layout_id)
            if self.env.episode_step_lim:
                ep.max_steps = int(self.env.episode_step_lim)

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

            self._emit_status({
                "command": "started",
                "task": task,
                "layout_id": layout_id,
                "round": ep.round,
                "max_steps": ep.max_steps,
                "instruction": ep.effective_instruction,
                "output_dir": output_dir,
            })
            self._log_line(f"首帧观测已取，指令：{ep.effective_instruction}，max_steps={ep.max_steps}")
            self._publish_obs(obs)
        except Exception as exc:  # noqa: BLE001 启动失败收敛成 failed 回合，不冲出循环
            self._log_line(f"回合启动失败：{type(exc).__name__}: {exc}")
            self._finalize(
                TERMINATION_FAILED,
                status=STATUS_FAILED,
                error_code="episode_start_error",
                error_message=f"{type(exc).__name__}: {exc}",
            )

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
            "env_instruction": self.env.env_instruction if self.env is not None else None,
            "mean_policy_latency_ms": mean_ms,
            "reproducibility": self.env.reproducibility() if self.env is not None else {},
            "error_code": error_code,
            "error_message": error_message,
        }
        try:
            with open(result_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            self._log(f"结果落盘失败：{type(exc).__name__}: {exc}")

        self._emit("eval_result", payload)
        self._log(
            f"回合结束 status={status} success={success} score={ep.final_score} "
            f"steps={ep.steps} → {result_path}"
        )
        self.episode = None

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
            self._log_line(f"收到 stop（{cmd.get('reason')}），立即收尾本回合")
            self._finalize(TERMINATION_STOPPED, status=STATUS_STOPPED)
            return None
        self._log(f"未知 eval_command：{kind!r}，忽略")
        return None

    def on_action(self, value: Any) -> None:
        ep = self.episode
        # 非 1:1 的多余 action 直接丢弃：重发观测后策略会多推一次，不丢的话
        # env 会被多走一步；回合终局后 chunk 里排队中的旧动作同理。
        if ep is None or ep.finished or not ep.awaiting_action:
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

        ep.latencies.append((time.time() - ep.obs_sent_at) * 1000.0)
        ep.awaiting_action = False

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
        if eid == "action":
            self.on_action(event.get("value"))
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
