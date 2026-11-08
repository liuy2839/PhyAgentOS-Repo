from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# dora 不在开发机依赖里：fake 掉再导入 runtime
_ = ModuleType("dora")
_.Node = object
sys.modules.setdefault("dora", _)

from forge_policy import (  # noqa: E402
    CHUNK_END_SENTINEL,
    Policy,
    PolicyContract,
    PolicyError,
    PolicyRuntime,
)


class FakeNode:
    """按 robodojo tests/test_policy_shutdown.py 的手法伪造 dora.Node。"""

    def __init__(self, events: list[dict[str, Any]]):
        self.events = list(events)
        self.outputs: list[tuple[str, Any]] = []

    def __iter__(self):
        while self.events:
            yield self.events.pop(0)

    def send_output(self, output_id: str, value: Any, *args: Any, **kwargs: Any) -> None:
        self.outputs.append((output_id, value))


def _pc_event(command: str, inputs: dict | None = None) -> dict[str, Any]:
    from forge_policy.msgs import PolicyCommand

    msg = PolicyCommand.from_inputs(
        policy_id="test", command=command, inputs=inputs or {}
    )
    return {"type": "INPUT", "id": "policy_command", "value": msg.to_arrow()}


def _proprio_event(position: list[float], names: list[str]) -> dict[str, Any]:
    from forge_policy.msgs import JointState

    state = JointState(name=names, position=position)
    return {"type": "INPUT", "id": "proprio_state", "value": state.to_arrow()}


def _image_event(topic: str) -> dict[str, Any]:
    from forge_policy.msgs import Image

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image.from_numpy(frame, "rgb8")
    return {"type": "INPUT", "id": topic, "value": img.to_arrow()}


class ScriptedPolicy(Policy):
    def __init__(self) -> None:
        self.resets = 0
        self.instructions: list[str] = []
        self.steps = 0
        self.fail_next = False
        self.require_instruction = False  # 打开时模拟 NO_INSTRUCTION 可重试错误

    def reset_episode(self) -> None:
        self.resets += 1

    def set_instruction(self, text: str) -> None:
        self.instructions.append(text)

    def step(self, obs: dict[str, Any]) -> np.ndarray | None:
        self.steps += 1
        if self.fail_next:
            raise PolicyError("TEST_FAIL", "boom")
        if self.require_instruction and not self.instructions:
            raise PolicyError("NO_INSTRUCTION", "评测端未下发任务指令")
        # 输出维度 = 动作契约维度（state 3 维 + 相机 1 路 = 3，恰好同 action_names）
        return np.linspace(0.0, 1.0, 3, dtype=np.float64)


CONTRACT = PolicyContract(
    action_names=("j1", "j2", "j3"),
    cameras=("image/head",),
)


def test_contract_validation() -> None:
    with pytest.raises(ValueError):
        PolicyContract(action_names=())
    with pytest.raises(ValueError):
        PolicyContract(action_names=("a", "a"))
    with pytest.raises(ValueError):
        PolicyContract(action_names=("a",), cameras=("x", "x"))
    c = PolicyContract(action_names=("a", "b"), cameras=("image/h",), depth_cameras=("depth/h",))
    assert c.action_dim == 2
    assert c.required_inputs() == frozenset({"image/h", "depth/h"})


def test_lifecycle_and_lockstep() -> None:
    policy = ScriptedPolicy()
    node = FakeNode(
        [
            _pc_event("start"),
            _pc_event("set_instruction", {"instruction": "pick the cube"}),
            _pc_event("reset_scene"),
            _proprio_event([0.1, 0.2, 0.3], ["j1", "j2", "j3"]),
            _image_event("image/head"),  # 观测收齐 → 1 次 step + 1 个 action
            {"type": "INPUT", "id": "tick", "value": pa.array([1])},  # 心跳不推理
            _proprio_event([0.2, 0.2, 0.3], ["j1", "j2", "j3"]),  # 图像未刷新 → 不 step
            _image_event("image/head"),  # 收齐 → 第 2 次 step
            _pc_event("stop"),
        ]
    )
    runtime = PolicyRuntime(policy, CONTRACT, node=node, logger=lambda _: None)
    assert runtime.run() == 0
    assert policy.instructions == ["pick the cube"]
    assert policy.resets == 1
    assert policy.steps == 2  # tick 和半帧都不推理
    actions = [v for topic, v in node.outputs if topic == "action"]
    assert len(actions) == 2


def test_action_names_and_chunk_end() -> None:
    """deprecated 手工置位路径的兼容性测试：置位必须在 step 返回**末动作之前**
    才能得到正确语义（哨兵随末动作发出）。新代码请用元组返回
    （test_chunk_end_tuple_return）。"""
    policy = ScriptedPolicy()
    contract = PolicyContract(
        action_names=("j1", "j2", "j3"), cameras=("image/head",), emit_chunk_end=True
    )
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),
        ]
        + [{"type": "STOP"}]
    )
    runtime = PolicyRuntime(policy, contract, node=node, logger=lambda _: None)
    node_events = node.events
    # 在 step 前标记 chunk 结束，使该动作带上哨兵
    original_step = policy.step

    def step_with_mark(obs):
        result = original_step(obs)
        runtime.mark_chunk_end()
        return result

    policy.step = step_with_mark  # type: ignore[method-assign]
    runtime.run()
    from forge_policy.msgs import JointCommand

    actions = [v for topic, v in node.outputs if topic == "action"]
    assert len(actions) == 1
    cmd = JointCommand.from_arrow(actions[0])
    assert list(cmd.name) == ["j1", "j2", "j3"]
    assert cmd.mode == "position"
    assert cmd.effort == [CHUNK_END_SENTINEL] * 3
    assert cmd.position == [0.0, 0.5, 1.0]


def test_chunk_end_tuple_return() -> None:
    """元组返回 (vec, is_last)：哨兵必须只随本 chunk 末动作发出，且时机由
    接口保证——3 动作 chunk → 第 3 个 action 带 effort 哨兵、前两个不带。
    这是防回归的核心测试：任何把哨兵错位到下一 chunk 首动作的改动都会在此红。"""
    policy = ScriptedPolicy()
    contract = PolicyContract(
        action_names=("j1", "j2", "j3"), cameras=("image/head",), emit_chunk_end=True
    )
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),  # 3 帧完整观测 → 3 个动作（一个 chunk）
            _proprio_event([0.1, 0.1, 0.1], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            _proprio_event([0.2, 0.2, 0.2], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            {"type": "STOP"},
        ]
    )
    runtime = PolicyRuntime(policy, contract, node=node, logger=lambda _: None)

    calls = {"n": 0}

    def step_chunked(obs):
        vec = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        calls["n"] += 1
        return vec, (calls["n"] == 3)  # 唯一一个 chunk：第 3 个动作是末动作

    policy.step = step_chunked  # type: ignore[method-assign]
    runtime.run()
    from forge_policy.msgs import JointCommand

    actions = [v for topic, v in node.outputs if topic == "action"]
    assert len(actions) == 3
    efforts = [JointCommand.from_arrow(v).effort for v in actions]
    assert efforts[0] == []
    assert efforts[1] == []
    assert efforts[2] == [CHUNK_END_SENTINEL] * 3   # 末动作带哨兵


def test_chunk_per_action() -> None:
    """每帧制策略（N=1 chunk）：chunk_per_action=True 时每个动作都带哨兵。"""
    policy = ScriptedPolicy()
    contract = PolicyContract(
        action_names=("j1", "j2", "j3"),
        cameras=("image/head",),
        emit_chunk_end=True,
        chunk_per_action=True,
    )
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            _proprio_event([0.1, 0.1, 0.1], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            {"type": "STOP"},
        ]
    )
    PolicyRuntime(policy, contract, node=node, logger=lambda _: None).run()
    from forge_policy.msgs import JointCommand

    actions = [v for topic, v in node.outputs if topic == "action"]
    assert len(actions) == 2
    for v in actions:   # 每帧制：每个动作都是决策单元的末动作
        assert JointCommand.from_arrow(v).effort == [CHUNK_END_SENTINEL] * 3


def test_chunk_per_action_requires_emit_chunk_end() -> None:
    """配置矛盾必须构造即炸：per-action 哨兵依赖 emit_chunk_end 的语义。"""
    with pytest.raises(ValueError):
        PolicyContract(action_names=("j1",), chunk_per_action=True)


def test_invalid_tuple_return_fails_structured() -> None:
    """step 返回形态非法（第二元素非 bool）→ 结构化失败 + 无动作发出。"""
    policy = ScriptedPolicy()
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            {"type": "STOP"},
        ]
    )

    def bad_step(obs):
        return np.zeros(3), "yes"

    policy.step = bad_step  # type: ignore[method-assign]
    PolicyRuntime(policy, CONTRACT, node=node, logger=lambda _: None).run()
    statuses = [
        json.loads(bytes(v).decode("utf-8"))
        for topic, v in node.outputs
        if topic == "policy_status"
    ]
    failed = [s for s in statuses if s["status"] == "failed"]
    assert failed and failed[-1]["error_code"] == "invalid_step_return"
    assert not [v for topic, v in node.outputs if topic == "action"]


def test_legacy_ndarray_return_still_works() -> None:
    """旧用法（step 返回裸 ndarray）零改动兼容：动作正常发出、不带哨兵。"""
    policy = ScriptedPolicy()
    contract = PolicyContract(
        action_names=("j1", "j2", "j3"), cameras=("image/head",), emit_chunk_end=True
    )
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            {"type": "STOP"},
        ]
    )
    PolicyRuntime(policy, contract, node=node, logger=lambda _: None).run()
    from forge_policy.msgs import JointCommand

    actions = [v for topic, v in node.outputs if topic == "action"]
    assert len(actions) == 1
    assert JointCommand.from_arrow(actions[0]).effort == []


def test_policy_error_reports_failed_status() -> None:
    policy = ScriptedPolicy()
    policy.fail_next = True
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            {"type": "STOP"},
        ]
    )
    runtime = PolicyRuntime(policy, CONTRACT, node=node, logger=lambda _: None)
    runtime.run()
    statuses = [
        json.loads(bytes(v).decode("utf-8"))
        for topic, v in node.outputs
        if topic == "policy_status"
    ]
    failed = [s for s in statuses if s["status"] == "failed"]
    assert failed and failed[-1]["error_code"] == "TEST_FAIL"
    # 失败后即使再来完整观测也不再发动作
    actions = [v for topic, v in node.outputs if topic == "action"]
    assert actions == []


def test_dimension_and_finite_guard() -> None:
    class BadDim(ScriptedPolicy):
        def step(self, obs):  # noqa: ANN001
            return np.zeros(2)

    policy = BadDim()
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            {"type": "STOP"},
        ]
    )
    PolicyRuntime(policy, CONTRACT, node=node, logger=lambda _: None).run()
    assert not [v for t, v in node.outputs if t == "action"]
    statuses = [
        json.loads(bytes(v).decode("utf-8"))
        for t, v in node.outputs
        if t == "policy_status"
    ]
    assert any(s.get("error_code") == "action_dim_mismatch" for s in statuses)


def test_cancel_exits() -> None:
    policy = ScriptedPolicy()
    node = FakeNode(
        [
            _pc_event("start"),
            _pc_event("cancel"),
        ]
    )
    runtime = PolicyRuntime(policy, CONTRACT, node=node, logger=lambda _: None)
    assert runtime.run() == 0
    # cancel 之后不再消费事件（fake node 用 AssertionError 保护也可以，这里直接断言 step=0）
    assert policy.steps == 0


def test_reset_clears_partial_observation() -> None:
    policy = ScriptedPolicy()
    node = FakeNode(
        [
            _pc_event("start"),
            _image_event("image/head"),  # 图像先到
            _pc_event("reset_scene"),  # 清掉半帧
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),  # 仅 proprio：不齐
            {"type": "STOP"},
        ]
    )
    PolicyRuntime(policy, CONTRACT, node=node, logger=lambda _: None).run()
    assert policy.steps == 0
    assert not [v for t, v in node.outputs if t == "action"]


def test_instruction_arriving_after_first_frame_recovers() -> None:
    """跨通道乱序回归：首帧观测先于 set_instruction 到达时不得判死。

    真实 Dora smoke 实测场景（2026-10-28 zyserver）：proprio+image 与
    policy_command 是不同 dora 通道，首帧观测可能先到。策略报
    NO_INSTRUCTION 属暂时缺料；收到指令后必须用缓存的观测帧补推理。
    """
    policy = ScriptedPolicy()
    policy.require_instruction = True
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.1, 0.2, 0.3], ["j1", "j2", "j3"]),
            _image_event("image/head"),  # 观测齐了，但指令未到 → 暂不推理
            _pc_event("set_instruction", {"instruction": "pick"}),
            # 无需新观测：指令到达即用缓存帧重试 → 1 step + 1 action
            {"type": "STOP"},
        ]
    )
    runtime = PolicyRuntime(policy, CONTRACT, node=node, logger=lambda _: None)
    runtime.run()
    # 第 1 次 step 在指令前（报 NO_INSTRUCTION 不计数动作）；指令到达重试成功
    assert policy.steps == 2
    assert policy.instructions == ["pick"]
    actions = [v for t, v in node.outputs if t == "action"]
    assert len(actions) == 1
    statuses = [
        json.loads(bytes(v).decode("utf-8"))
        for t, v in node.outputs
        if t == "policy_status"
    ]
    assert not any(s["status"] == "failed" for s in statuses)


def test_camera_info_frame_and_obs_field() -> None:
    seen: list[dict[str, Any]] = []

    class InfoPolicy(ScriptedPolicy):
        def step(self, obs):  # noqa: ANN001
            seen.append({"camera_info": obs.get("camera_info"), "keys": sorted(obs)})
            return np.zeros(3)

    contract = PolicyContract(
        action_names=("j1", "j2", "j3"), cameras=("image/head",), camera_info=True
    )
    node = FakeNode(
        [
            _pc_event("start"),
            _proprio_event([0.0, 0.0, 0.0], ["j1", "j2", "j3"]),
            _image_event("image/head"),
            {"type": "INPUT", "id": "camera_info",
             "value": pa.array([json.dumps({"depth/head": {"intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                         "extrinsics": [[1, 0, 0, 0], [0, 1, 0, 0],
                                                                        [0, 0, 1, 0], [0, 0, 0, 1]]}})])},
            {"type": "STOP"},
        ]
    )
    PolicyRuntime(InfoPolicy(), contract, node=node, logger=lambda _: None).run()
    assert seen and seen[0]["camera_info"]["depth/head"]["intrinsics"][0][0] == 1
    assert "camera_info" in seen[0]["keys"]


def _unused(*_: Any) -> None:  # pragma: no cover
    _ = SimpleNamespace, ModuleType
