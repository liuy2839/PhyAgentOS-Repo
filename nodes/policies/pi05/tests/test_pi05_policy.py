"""π0.5 节点协议骨架的单测：chunk 节奏、哨兵、观测还原、WS 客户端调用序列。

推理走 WsModelClient（方案 a）——测试用 FakeClient 替身锁定节点侧语义；
算法/服务器到位后这些测试应依然全绿（它们锁的是协议，不是模型）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "policy" / "src"))
sys.path.insert(0, str(HERE.parent / "src"))


class _FakeDoraNode:
    def __init__(self):
        self.sent = []

    def send_output(self, output_id, value):
        self.sent.append((output_id, value))


_ = types.ModuleType("dora")
_.Node = _FakeDoraNode
sys.modules.setdefault("dora", _)

from forge_policy import PolicyContract, PolicyRuntime  # noqa: E402
from pi05_policy.policy import Pi05Policy, _action_names_for  # noqa: E402

ACTION_NAMES = _action_names_for()


class FakeWsClient:
    """WsModelClient 替身：记录调用序列，get_action 按脚本回放 chunk。"""

    def __init__(self, chunk_lens=(3,)):
        self.calls: list[tuple[str, Any]] = []
        self.chunk_lens = list(chunk_lens)
        self._infer_calls = 0

    def call(self, func_name, obs=None, **kw):
        self.calls.append((func_name, obs))
        if func_name == "get_action":
            n = self.chunk_lens[min(self._infer_calls, len(self.chunk_lens) - 1)]
            self._infer_calls += 1
            return [
                {
                    "left_arm_joint_state": np.zeros(6, dtype=np.float32),
                    "left_ee_joint_state": np.zeros(1, dtype=np.float32),
                    "right_arm_joint_state": np.zeros(6, dtype=np.float32),
                    "right_ee_joint_state": np.zeros(1, dtype=np.float32),
                }
                for _ in range(n)
            ]
        return None

    def close(self):
        pass


from typing import Any  # noqa: E402


def _mk_obs():
    names = (
        [f"left_arm_joint_state.{i}" for i in range(6)]
        + ["left_ee_joint_state.0"] + [f"left_ee_pose.{i}" for i in range(7)]
        + [f"right_arm_joint_state.{i}" for i in range(6)]
        + ["right_ee_joint_state.0"] + [f"right_ee_pose.{i}" for i in range(7)]
    )
    return {
        "state": np.zeros(OBS_DIM, dtype=np.float64),
        "state_names": names,
        "images": {"image/head": np.zeros((2, 2, 3), dtype=np.uint8)},
    }


OBS_DIM = 28


def _make_policy(chunk_lens=(3,)):
    client = FakeWsClient(chunk_lens)
    policy = Pi05Policy(client, server=None)
    return policy, client


def _wire(policy):
    contract = PolicyContract(
        action_names=ACTION_NAMES, cameras=("image/head",), emit_chunk_end=True
    )
    runtime = PolicyRuntime(policy, contract)
    return runtime


def test_step_calls_update_obs_then_get_action():
    """每帧：先 update_obs（无条件喂），chunk 用尽才 get_action。"""
    policy, client = _make_policy(chunk_lens=(3,))
    policy.step(_mk_obs())
    names = [c for c, _ in client.calls]
    assert names[0] == "update_obs"
    assert "get_action" in names
    # update_obs 出现 1 次（首帧），get_action 1 次
    assert names.count("update_obs") == 1
    assert names.count("get_action") == 1


def test_chunk_semantics_returns_tuples():
    policy, _ = _make_policy(chunk_lens=(3,))
    out = [policy.step(_mk_obs()) for _ in range(3)]
    # 元组形态：(vec, is_last)，只有第三个是末动作
    assert [o[1] for o in out] == [False, False, True]


def test_chunk_queue_no_reinfer_within_chunk():
    """chunk 内的后续帧：继续 update_obs，但不重新 get_action。"""
    policy, client = _make_policy(chunk_lens=(3,))
    for _ in range(3):
        policy.step(_mk_obs())
    assert [c for c, _ in client.calls].count("get_action") == 1
    assert [c for c, _ in client.calls].count("update_obs") == 3


def test_chunk_queue_refills_after_exhausted():
    policy, client = _make_policy(chunk_lens=(2,))
    for _ in range(3):
        policy.step(_mk_obs())
    assert client._infer_calls == 2      # 第 3 帧触发重推理


def test_empty_chunk_returns_none():
    policy, _ = _make_policy(chunk_lens=(0,))
    assert policy.step(_mk_obs()) is None


def test_reset_calls_ws_reset_twice():
    """官方每集双 reset 的节奏照抄（WS 客户端的 reset 也归零服务端步数）。"""
    policy, client = _make_policy()
    policy.reset_episode()
    assert [c for c, _ in client.calls] == ["reset", "reset"]


def test_flatten_dict_to_14d():
    policy, _ = _make_policy()
    action = {
        "left_arm_joint_state": np.arange(6, dtype=np.float32),
        "left_ee_joint_state": np.array([0.5], dtype=np.float32),
        "right_arm_joint_state": np.arange(6, dtype=np.float32),
        "right_ee_joint_state": np.array([0.1], dtype=np.float32),
    }
    vec = policy._flatten(action)
    assert vec.shape == (14,)
    assert vec[0] == 0.0 and vec[6] == 0.5 and vec[7] == 0.0 and vec[13] == 0.1


def test_action_names_shape():
    assert len(ACTION_NAMES) == 14
    assert ACTION_NAMES[0] == "left_arm_joint_state.0"
    assert ACTION_NAMES[-1] == "right_ee_joint_state.0"
