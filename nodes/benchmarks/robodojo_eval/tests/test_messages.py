"""消息层 round-trip：验证本目录 messages.py 的 wire 格式与 forge-msgs 1.x 一致。

历史注：本载体最初从旧 forge 流程的 adapters/messages.py 提取（两份曾逐字相同，
有互解单测）；旧流程已从本仓库移除，这里改为内嵌 forge-msgs 1.0.1 的字段布局
断言（字段名 + Arrow 类型），保证线上协议不变。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NODE_MESSAGES = _load(
    "robodojo_eval_messages", Path(__file__).resolve().parent.parent / "messages.py"
)

# forge-msgs 1.0.1 的 Arrow 字段布局（线上契约）
FORGE_MSGS_LAYOUT = {
    "PolicyCommand": {
        "fields": ["policy_id", "command", "request_id", "inputs_json"],
        "types": [pa.string()] * 4,
    },
    "JointState": {
        "fields": ["name", "position", "velocity", "effort"],
        "types": [pa.list_(pa.string())] + [pa.list_(pa.float64())] * 3,
    },
    "JointCommand": {
        "fields": ["name", "mode", "position", "velocity", "effort", "kp", "kd"],
        "types": [pa.list_(pa.string()), pa.string()] + [pa.list_(pa.float64())] * 5,
    },
    "Image": {
        "fields": ["height", "width", "encoding", "step", "data"],
        "types": [pa.uint32(), pa.uint32(), pa.string(), pa.uint32(), pa.large_binary()],
    },
}


def _assert_layout(cls, name):
    batch = None
    spec = FORGE_MSGS_LAYOUT[name]
    batch = cls.to_arrow(_example(cls, name))
    assert batch.schema.names == spec["fields"], f"{name} 字段序变动：{batch.schema.names}"
    for i, (field, want) in enumerate(zip(batch.schema, spec["types"])):
        assert field.type == want, f"{name}.{field.name} 类型变动：{field.type} != {want}"


def _example(cls, name):
    if name == "PolicyCommand":
        return cls.from_inputs(policy_id="x", command="start")
    if name == "JointState":
        return cls(name=["j"], position=[0.0])
    if name == "JointCommand":
        return cls(name=["j"], mode="position", position=[0.0])
    return cls(height=1, width=1, encoding="rgb8", step=3, data=b"\0\0\0")


def test_wire_layout_matches_forge_msgs_1x():
    """字段名与 Arrow 类型逐一对齐 forge-msgs 1.0.1（线上契约）。"""
    _assert_layout(NODE_MESSAGES.PolicyCommand, "PolicyCommand")
    _assert_layout(NODE_MESSAGES.JointState, "JointState")
    _assert_layout(NODE_MESSAGES.JointCommand, "JointCommand")
    _assert_layout(NODE_MESSAGES.Image, "Image")


def test_policy_command_roundtrip():
    msg = NODE_MESSAGES.PolicyCommand.from_inputs(
        policy_id="FastWAM_80k_forge", command="set_instruction",
        inputs={"instruction": "place each digit centered on its pad"},
    )
    back = NODE_MESSAGES.PolicyCommand.from_arrow(msg.to_arrow())
    assert back.policy_id == "FastWAM_80k_forge"
    assert back.command == "set_instruction"
    assert back.inputs_json == msg.inputs_json


def test_joint_state_roundtrip():
    msg = NODE_MESSAGES.JointState.from_np(
        np.arange(28, dtype=np.float64), [f"j{i}" for i in range(28)], "position"
    )
    back = NODE_MESSAGES.JointState.from_arrow(msg.to_arrow())
    assert back.name == msg.name
    assert back.position == msg.position


def test_joint_command_roundtrip_with_chunk_end():
    """chunk_end 哨兵走 effort 字段：末动作 effort 填满 1.0，其余留空。"""
    names = [f"comp.{i}" for i in range(14)]
    end = NODE_MESSAGES.JointCommand(
        name=names, mode="position", position=[0.1] * 14, effort=[1.0] * 14,
    )
    mid = NODE_MESSAGES.JointCommand(name=names, mode="position", position=[0.2] * 14)
    back_end = NODE_MESSAGES.JointCommand.from_arrow(end.to_arrow())
    back_mid = NODE_MESSAGES.JointCommand.from_arrow(mid.to_arrow())
    assert back_end.effort and back_end.effort[0] == 1.0
    assert back_mid.effort == []


def test_image_roundtrip_rgb8():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    msg = NODE_MESSAGES.Image.from_numpy(frame, "rgb8")
    back = NODE_MESSAGES.Image.from_arrow(msg.to_arrow())
    np.testing.assert_array_equal(back.to_numpy(), frame)
