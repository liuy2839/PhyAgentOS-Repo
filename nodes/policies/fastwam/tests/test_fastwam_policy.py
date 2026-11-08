"""FastWAM 策略适配器的单测：四方法语义、chunk 节奏、chunk_end 哨兵时机。

不 import dora / XPolicyLab：model 用替身，contract 手工构造。
forge_policy SDK 走仓库内 policy/src（无 forge_msgs 时自动回退本地载体）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "policy" / "src"))
sys.path.insert(0, str(HERE.parent / "src"))

_ = __import__("types").ModuleType("dora")


class _FakeDoraNode:
    def __init__(self):
        self.sent = []

    def send_output(self, output_id, value):
        self.sent.append((output_id, value))


_.Node = _FakeDoraNode
sys.modules.setdefault("dora", _)

from forge_policy import PolicyContract, PolicyRuntime  # noqa: E402
from fastwam_policy.policy import (  # noqa: E402
    FastWAMPolicy,
    _build_obs_spec_from_dims,
    _build_state_spec_from_dims,
)

ENV_CFG = "arx_x5"
# ARX X5 双臂（与 XPolicyLab 的 get_robot_action_dim_info("arx_x5") 同构）
ARM_DIMS = [6, 6]
EE_DIMS = [1, 1]
OBS_SPEC = _build_obs_spec_from_dims(ARM_DIMS, EE_DIMS)
# 动作分量名：从被测函数本身推导（joint 模式 14 维），与评测侧同源
ACTION_NAMES = [
    f"{key}.{i}"
    for key, dim in _build_state_spec_from_dims(ARM_DIMS, EE_DIMS, "joint")
    for i in range(dim)
]


class FakeModel:
    """XPolicyLab model.Model 的替身：按脚本回放 action chunk。"""

    def __init__(self, chunks):
        self.chunks = [list(c) for c in chunks]   # 每个元素是一个 chunk（list[dict]）
        self.reset_calls = 0
        self.update_obs_calls = 0
        self.get_action_calls = 0
        self.last_obs = None

    def reset(self):
        self.reset_calls += 1

    def update_obs(self, obs):
        self.update_obs_calls += 1
        self.last_obs = obs

    def get_action(self):
        self.get_action_calls += 1
        if not self.chunks:
            return []
        return self.chunks.pop(0)


def _mk_action():
    """XPolicyLab 的动作 dict：键是 `left_arm_joint_state` 这类整键、值为 dim 维向量。"""
    return {
        key: np.zeros(dim, dtype=np.float32)
        for key, dim in _build_state_spec_from_dims(ARM_DIMS, EE_DIMS, "joint")
    }


def _mk_obs():
    return {
        "state": np.zeros(28, dtype=np.float64),
        "state_names": [f"{k}.{i}" for k, d in OBS_SPEC for i in range(d)],
        "images": {"image/head": np.zeros((2, 2, 3), dtype=np.uint8)},
    }


def _make_contract():
    return PolicyContract(
        action_names=tuple(ACTION_NAMES),
        cameras=("image/head",),
        emit_chunk_end=True,
    )


def _wire(policy, contract):
    runtime = PolicyRuntime(policy, contract)
    return runtime


def _make_policy(chunks):
    return FastWAMPolicy(FakeModel(chunks), ARM_DIMS, EE_DIMS, "joint", obs_spec=OBS_SPEC)


def test_reset_episode_double_reset():
    """官方每集 reset 两次的节奏照抄。"""
    policy = _make_policy([])
    policy.reset_episode()
    assert policy._model.reset_calls == 2


def test_set_instruction_feeds_model_obs():
    policy = _make_policy([])
    runtime = _wire(policy, _make_contract())
    runtime._phase = "running"
    policy.set_instruction("place each digit centered on its pad")
    policy._model.chunks = [[_mk_action()]]
    policy.step(_mk_obs())
    assert policy._instruction == "place each digit centered on its pad"
    assert policy._model.last_obs["instruction"] == "place each digit centered on its pad"


def test_chunk_rhythm_update_obs_every_frame():
    """chunk 执行期间每一步都要 update_obs（官方节奏坑 2）：
    chunk 长度 3、观测 3 帧 → update_obs 3 次、get_action 1 次。"""
    policy = _make_policy([[_mk_action(), _mk_action(), _mk_action()]])
    runtime = _wire(policy, _make_contract())
    runtime._phase = "running"

    for _ in range(3):
        policy.step(_mk_obs())
    assert policy._model.update_obs_calls == 3
    assert policy._model.get_action_calls == 1


def test_chunk_end_sentinel_on_last_action_of_chunk():
    """哨兵必须随**本 chunk 末动作**发出：chunk 长度 3 → 第 3 个 action 带
    effort 哨兵、前两个不带。元组返回 (vec, is_last) 的接口保证用法。"""
    policy = _make_policy([[_mk_action() for _ in range(3)]])
    runtime = _wire(policy, _make_contract())

    sent = []
    runtime.node.send_output = lambda output_id, value: sent.append((output_id, value))
    runtime._phase = "running"

    for _ in range(3):
        runtime._step_on(_mk_obs())

    assert len(sent) == 3
    # 用 forge_policy 的载体解析 effort 哨兵位置
    from forge_policy.msgs import JointCommand

    efforts = []
    for _, value in sent:
        jc = JointCommand.from_arrow(value)
        efforts.append(list(jc.effort or []))
    assert efforts[0] == [] and efforts[1] == []
    assert efforts[2] and efforts[2][0] == 1.0      # 末动作带哨兵


def test_empty_chunk_returns_none():
    policy = _make_policy([[]])
    assert policy.step(_mk_obs()) is None


def test_obs_spec_sorted_and_complete():
    assert OBS_SPEC == tuple(sorted(OBS_SPEC, key=lambda kv: kv[0]))
    keys = [k for k, _ in OBS_SPEC]
    assert keys == sorted(set(keys))
    # 与评测节点侧 robodojo_env.build_obs_spec 的算法逐字一致：
    # 6 个键 / 28 维（双臂 arm + ee_joint + ee_pose × 2）
    assert len(OBS_SPEC) == 6
    assert sum(d for _, d in OBS_SPEC) == 28
    assert len(ACTION_NAMES) == 14
