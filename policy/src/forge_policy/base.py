"""策略抽象接口：算法实现这四个方法，Dora 收发全部由 PolicyRuntime 承担。

obs（一帧完整观测）的结构：
    {
        "state": np.ndarray[float64],          # 关节状态向量
        "state_names": list[str],              # 与 state 对齐的分量名
        "images": {cam: np.ndarray[H,W,3] uint8},   # cam = contract.cameras 中的名字
        "depths": {cam: np.ndarray[H,W] float32},   # meters，可选
        "segs":   {cam: np.ndarray[H,W] int},       # 实例标签，可选
    }
step 返回：
    np.ndarray[len(action_names)]  # 本帧动作（position 语义）
    None                           # 本帧不动作（如规划中），评测端 watchdog 会重发观测
抛异常：
    PolicyError                    # 结构化失败：runtime 停发动作并上报 policy_status=failed
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PolicyError(RuntimeError):
    """策略的结构化失败（不可恢复到本 episode 内），带稳定错误码。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Policy:
    """算法侧唯一需要实现的接口。"""

    def reset_episode(self) -> None:
        """评测端 reset_scene：清空 episode 内的计划/观测缓存/动作队列。
        不重载模型、不退出进程。"""

    def set_instruction(self, text: str) -> None:
        """接收本 episode 的自然语言任务指令；下一次 step 前应触发重规划。"""

    def step(self, obs: dict[str, Any]) -> "np.ndarray | tuple[np.ndarray, bool] | None":
        """消费一帧完整观测，返回一个动作向量或 None。

        返回值的三种形态（emit_chunk_end=True 时推荐元组形式）：
          None                     本帧不动作（如规划中），评测端 watchdog 会重发观测；
          np.ndarray               本帧动作，**非 chunk 末**（兼容旧用法）；
          (np.ndarray, bool)       (本帧动作, 是否本 chunk 的末动作)。

        元组形式的 is_last=True 动作由 PolicyRuntime 自动附 effort 哨兵
        （CHUNK_END_SENTINEL），时序与 RoboDojo 官方 rollout 逐字对齐——
        补查发生在 chunk 末动作之后、下一帧观测之前。这把"哨兵随末动作"
        由接口保证，不再依赖策略在正确时机手工调用 mark_chunk_end()。
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """stop/cancel 后的清理；进程即将退出。"""
