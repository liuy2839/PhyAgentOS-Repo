"""RoboDojo's four Arrow message carriers, usable by its Python 3.10 policy.

The field names and Arrow types match forge-msgs 1.0.1. Both nodes in this
dataflow use this module; other benchmarks keep their installed forge-msgs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa


def _batch(value: pa.RecordBatch | pa.Table | pa.StructArray | bytes) -> pa.RecordBatch:
    if isinstance(value, pa.RecordBatch):
        return value
    if isinstance(value, pa.Table):
        batches = value.to_batches()
        if not batches:
            raise ValueError("Arrow table is empty")
        return batches[0]
    if isinstance(value, pa.StructArray):
        return pa.RecordBatch.from_arrays(
            [value.field(i) for i in range(value.type.num_fields)],
            names=[value.type.field(i).name for i in range(value.type.num_fields)],
        )
    if isinstance(value, bytes):
        return pa.ipc.open_stream(value).read_next_batch()
    raise TypeError(f"Expected an Arrow record batch, table, struct array, or bytes; got {type(value)}")


def _cell(batch: pa.RecordBatch, name: str):
    if batch.num_rows != 1:
        raise ValueError(f"Expected one {name} message, got {batch.num_rows} rows")
    return batch[name][0].as_py()


def _names(names: list[str]) -> None:
    if not names or len(set(names)) != len(names):
        raise ValueError("Joint names must be nonempty and unique")


def _vector(names: list[str], values: list[float], name: str) -> None:
    if values and len(values) != len(names):
        raise ValueError(f"{name} must be empty or match the number of joint names")


@dataclass
class PolicyCommand:
    policy_id: str
    command: str
    request_id: str = ""
    inputs_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.policy_id or not re.fullmatch(r"[a-z][a-z0-9_]*", self.command):
            raise ValueError("Policy command needs a policy ID and snake_case command")
        if not isinstance(json.loads(self.inputs_json), dict):
            raise ValueError("inputs_json must be a JSON object")

    @classmethod
    def from_inputs(
        cls,
        *,
        policy_id: str,
        command: str,
        inputs: dict | None = None,
        request_id: str = "",
    ):
        return cls(
            policy_id,
            command,
            request_id=request_id,
            inputs_json=json.dumps(inputs or {}, separators=(",", ":")),
        )

    def to_arrow(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_pydict(
            {
                "policy_id": pa.array([self.policy_id], type=pa.string()),
                "command": pa.array([self.command], type=pa.string()),
                "request_id": pa.array([self.request_id], type=pa.string()),
                "inputs_json": pa.array([self.inputs_json], type=pa.string()),
            }
        )

    @classmethod
    def from_arrow(cls, value: pa.RecordBatch | pa.Table | pa.StructArray | bytes):
        batch = _batch(value)
        return cls(*(_cell(batch, name) for name in ("policy_id", "command", "request_id", "inputs_json")))


@dataclass
class JointState:
    name: list[str]
    position: list[float] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        _names(self.name)
        for key in ("position", "velocity", "effort"):
            _vector(self.name, getattr(self, key), key)

    @classmethod
    def from_np(cls, values: np.ndarray, order: list[str], field_name: str = "position"):
        if field_name not in ("position", "velocity", "effort"):
            raise ValueError(f"Unsupported joint state field: {field_name}")
        if len(values) < len(order):
            raise ValueError("Joint state vector is shorter than its name list")
        return cls(name=list(order), **{field_name: [float(v) for v in values[: len(order)]]})

    def to_arrow(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_pydict(
            {
                "name": pa.array([self.name], type=pa.list_(pa.string())),
                "position": pa.array([self.position], type=pa.list_(pa.float64())),
                "velocity": pa.array([self.velocity], type=pa.list_(pa.float64())),
                "effort": pa.array([self.effort], type=pa.list_(pa.float64())),
            }
        )

    @classmethod
    def from_arrow(cls, value: pa.RecordBatch | pa.Table | pa.StructArray | bytes):
        batch = _batch(value)
        return cls(**{name: _cell(batch, name) or [] for name in ("name", "position", "velocity", "effort")})


@dataclass
class JointCommand:
    name: list[str]
    mode: str = "position"
    position: list[float] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)
    kp: list[float] = field(default_factory=list)
    kd: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        _names(self.name)
        if self.mode not in ("position", "velocity", "effort", "hybrid"):
            raise ValueError(f"Unsupported joint command mode: {self.mode}")
        for key in ("position", "velocity", "effort", "kp", "kd"):
            _vector(self.name, getattr(self, key), key)

    def to_arrow(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_pydict(
            {
                "name": pa.array([self.name], type=pa.list_(pa.string())),
                "mode": pa.array([self.mode], type=pa.string()),
                "position": pa.array([self.position], type=pa.list_(pa.float64())),
                "velocity": pa.array([self.velocity], type=pa.list_(pa.float64())),
                "effort": pa.array([self.effort], type=pa.list_(pa.float64())),
                "kp": pa.array([self.kp], type=pa.list_(pa.float64())),
                "kd": pa.array([self.kd], type=pa.list_(pa.float64())),
            }
        )

    @classmethod
    def from_arrow(cls, value: pa.RecordBatch | pa.Table | pa.StructArray | bytes):
        batch = _batch(value)
        return cls(
            name=_cell(batch, "name") or [],
            mode=_cell(batch, "mode") or "position",
            **{name: _cell(batch, name) or [] for name in ("position", "velocity", "effort", "kp", "kd")},
        )


@dataclass
class Image:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes

    def __post_init__(self) -> None:
        if self.height < 0 or self.width < 0:
            raise ValueError("RoboDojo images must have nonnegative dimensions")
        if self.encoding == "rgb8":
            if self.step < self.width * 3 or len(self.data) != self.step * self.height:
                raise ValueError("Image step or data length is inconsistent with its dimensions")
            return
        if self.encoding == "32FC1":
            expected = self.width * 4 * self.height
        elif self.encoding == "16UC1":
            expected = self.width * 2 * self.height
        else:
            raise ValueError(f"unsupported RoboDojo image encoding: {self.encoding}")
        if len(self.data) != expected:
            raise ValueError(
                f"{self.encoding} data length must equal width*itemsize*height "
                f"({len(self.data)} != {expected})"
            )

    @classmethod
    def from_numpy(cls, frame: np.ndarray, encoding: str = "rgb8"):
        if encoding == "32FC1":
            arr = np.ascontiguousarray(frame, dtype=np.float32)
            height, width = arr.shape[:2]
            return cls(height, width, encoding, width * 4, arr.tobytes())
        if encoding == "16UC1":
            arr = np.ascontiguousarray(frame, dtype=np.uint16)
            height, width = arr.shape[:2]
            return cls(height, width, encoding, width * 2, arr.tobytes())
        if encoding != "rgb8" or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("RoboDojo expects an HxWx3 uint8 rgb8 frame")
        height, width, _ = frame.shape
        return cls(height, width, encoding, width * 3, np.ascontiguousarray(frame).tobytes())

    def to_numpy(self) -> np.ndarray:
        if self.encoding == "32FC1":
            return np.frombuffer(self.data, dtype=np.float32).reshape(self.height, self.width).copy()
        if self.encoding == "16UC1":
            return np.frombuffer(self.data, dtype=np.uint16).reshape(self.height, self.width).copy()
        if self.step == self.width * 3:
            return np.frombuffer(self.data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.ndarray(
            (self.height, self.width, 3),
            dtype=np.uint8,
            buffer=self.data,
            strides=(self.step, 3, 1),
        ).copy()

    def to_arrow(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [
                pa.array([self.height], type=pa.uint32()),
                pa.array([self.width], type=pa.uint32()),
                pa.array([self.encoding], type=pa.string()),
                pa.array([self.step], type=pa.uint32()),
                pa.array([self.data], type=pa.large_binary()),
            ],
            names=["height", "width", "encoding", "step", "data"],
        )

    @classmethod
    def from_arrow(cls, value: pa.RecordBatch | pa.Table | pa.StructArray | bytes):
        batch = _batch(value)
        return cls(*(_cell(batch, name) for name in ("height", "width", "encoding", "step", "data")))
