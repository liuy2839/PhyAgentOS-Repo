"""统一策略节点传输协议（Forge Policy Wire Contract）常量与校验。

约定（与评测节点的锁步语义一致）：
- 输入 topic：
    proprio_state : forge_msgs.JointState（命名关节状态）
    image/<cam>   : forge_msgs.Image, encoding=rgb8
    depth/<cam>   : forge_msgs.Image, encoding=32FC1, 单位 meters（可选）
    seg/<cam>     : forge_msgs.Image, encoding=mono16, 物体实例标签（可选）
    policy_command: forge_msgs.PolicyCommand（dataflow 中 queue_size 必须 ≥32）
    tick          : 仅作心跳/看门狗，绝不在 tick 上推理
- 输出 topic：
    action        : forge_msgs.JointCommand，name/position 与评测端期望逐字一致
    policy_status : JSON 文本（状态机 + 错误码）
- 生命周期状态机：idle → ready → running → (idle | terminal)
- 推理时机：观测驱动。收齐一帧完整观测（proprio + 声明的全部相机）才调用
  策略 step()，至多产出一个 action；评测端按 1:1 锁步消费。
- chunk 边界（RoboDojo 兼容项，非通用语义）：RoboDojo 的 is_episode_end()
  带副作用，官方 rollout 每个 chunk 在「末动作之后、下一帧观测之前」多查
  一次（每 chunk 2N+1 次），调用次数影响成绩。chunk 末动作的
  JointCommand.effort 整条填 CHUNK_END_SENTINEL，评测端据此补查那一次。
  声明方式二选一：
    * chunk 制策略（如 FastWAM）：step() 返回 (action, is_last) 元组，
      is_last=True 的动作由 runtime 自动附哨兵——推荐，时机由接口保证；
    * 每帧制策略（如 ReKep，等价于官方 N=1 chunk，每个动作都是决策单元
      的末动作）：contract 设 chunk_per_action=True，runtime 给每个动作
      附哨兵。
"""

from __future__ import annotations

from dataclasses import dataclass

# benchmarks/_core/adapter.py 的同名约定；RoboDojo 的 is_episode_end 有副作用，
# 官方 rollout 每 chunk 会多查一次结束，此标记让评测端补上那一次。
CHUNK_END_SENTINEL = 1.0

# dataflow 里 policy_command 输入的最低队列深度：episode 边界会连发
# set_instruction + reset_scene 两条命令，默认深度会挤掉前一条且不报错。
MIN_POLICY_COMMAND_QUEUE = 32

POLICY_STATUS_IDLE = "idle"
POLICY_STATUS_READY = "ready"
POLICY_STATUS_RUNNING = "running"
POLICY_STATUS_FAILED = "failed"
POLICY_STATUS_CANCELLED = "cancelled"
POLICY_STATUS_STOPPED = "stopped"

# PolicyCommand.command 的合法值（评测端 _core 目前使用 start/set_instruction/
# reset_scene/stop；cancel 为协议统一补充，评测端不发送但策略端必须能处理）。
TERMINAL_COMMANDS = frozenset({"stop", "cancel"})
KNOWN_COMMANDS = frozenset(
    {"start", "set_instruction", "reset_scene", "pause", "resume"} | TERMINAL_COMMANDS
)

# 可重试的策略错误码：由跨 topic 乱序引起的"暂时缺料"（如指令晚于首帧观测到达）。
# runtime 收到这些错误时不判死 episode，缓存观测帧，等下一条命令重试。
# 评测端 _core runner 的 set_instruction/reset_scene 与观测是不同 dora 通道，
# 到达顺序无保证——这正是需要该机制的场景。
RETRYABLE_ERROR_CODES = frozenset({"NO_INSTRUCTION"})


@dataclass(frozen=True)
class PolicyContract:
    """一个策略节点的协议声明。

    cameras / depth_cameras / seg_cameras 是节点**要求收齐才算一帧**的输入
    topic 集合；action_names 是输出 JointCommand.name 的期望顺序，评测端
    （_core runner 的 action_component_names 校验）会逐字比对。
    """

    action_names: tuple[str, ...]
    cameras: tuple[str, ...] = ()
    depth_cameras: tuple[str, ...] = ()
    seg_cameras: tuple[str, ...] = ()
    scene_objects: bool = False
    camera_info: bool = False
    emit_chunk_end: bool = False
    # 每帧制策略（每帧重新决策，等价于官方 N=1 chunk）：每个发出的动作都
    # 视为决策单元的末动作、附 effort 哨兵。必须与 emit_chunk_end=True
    # 同用（否则是语义矛盾，构造即拒绝）。
    chunk_per_action: bool = False

    def __post_init__(self) -> None:
        if not self.action_names:
            raise ValueError("action_names 不能为空：动作分量名是协议契约")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action_names 不能有重名")
        for field in ("cameras", "depth_cameras", "seg_cameras"):
            names = getattr(self, field)
            if len(set(names)) != len(names):
                raise ValueError(f"{field} 不能有重复相机")
        if self.chunk_per_action and not self.emit_chunk_end:
            raise ValueError(
                "chunk_per_action=True 需要 emit_chunk_end=True：前者只改变"
                "哨兵的覆盖范围（每个动作 vs 仅元组声明的末动作），不开后者"
                "等于不发哨兵，属于配置矛盾"
            )

    def required_inputs(self) -> frozenset[str]:
        """一帧完整观测需要收齐的 topic 集合。"""
        parts: set[str] = set(self.cameras)
        parts.update(self.depth_cameras)
        parts.update(self.seg_cameras)
        if self.scene_objects:
            parts.add("scene_objects")
        if self.camera_info:
            parts.add("camera_info")
        return frozenset(parts)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)
