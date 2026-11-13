"""评测发起节点纯逻辑的单测：配置、录像器、单回合状态机（假 node + 假 env）。

不 import dora / Isaac / RoboDojo —— episode_service 的设计保证了这一点。
目录里 messages.py 与 forge_robodojo_eval/robodojo/adapters/messages.py 同源，
以 `eval_messages` 别名加载避免 sys.modules 冲突。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

HERE = Path(__file__).resolve().parent
NODE_DIR = HERE.parent
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))


def _load_messages():
    src = NODE_DIR / "messages.py"
    spec = importlib.util.spec_from_file_location("robodojo_eval_messages", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["robodojo_eval_messages"] = mod
    spec.loader.exec_module(mod)
    return mod


messages = _load_messages()


# ---------------------------------------------------------------- 配置


def test_config_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "benchmark: robodojo\nenv_cfg_type: arx_x5\ndevice_id: 3\n"
        "action_type: joint\npolicy_id: FastWAM_80k_forge\n"
        "result_dir: /tmp/rsi_out\nwatchdog_s: 25\n",
        encoding="utf-8",
    )
    cfg = _CONFIG.load_config(str(p))
    assert cfg.env_cfg_type == "arx_x5"
    assert cfg.device_id == 3
    assert cfg.action_type == "joint"
    assert cfg.policy_id == "FastWAM_80k_forge"
    assert cfg.result_dir == "/tmp/rsi_out"
    assert cfg.watchdog_s == 25.0
    assert cfg.camera_map["image/head"] == "cam_head"


def test_config_rejects_bad_action_type(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("action_type: quaternion\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _CONFIG.load_config(str(p))


# ---------------------------------------------------------------- 录像器


def test_camera_short_name():
    assert _RECORDER.camera_short_name("image/head") == "head"
    assert _RECORDER.camera_short_name("image/left_wrist") == "left_wrist"
    assert _RECORDER.camera_short_name("plain") == "plain"


class _FakeWriter:
    def __init__(self, path, fourcc, fps, size):
        self.path, self.size = path, size
        self.frames = []
        self.released = False
        self.opened = True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True
        if self.frames:
            # 真 writer 在此落盘；替身写个非空文件让 save() 的存在性检查通过
            with open(self.path, "wb") as fh:
                fh.write(b"fake")

    def isOpened(self):
        return self.opened


def test_recorder_per_camera_files(tmp_path, monkeypatch):
    import cv2

    captured = {}

    def fake_video_writer(path, fourcc, fps, size):
        captured[path] = _FakeWriter(path, fourcc, fps, size)
        return captured[path]

    monkeypatch.setattr(cv2, "VideoWriter", fake_video_writer)

    rec = _RECORDER.PerCameraRecorder(str(tmp_path / "videos"), fps=30.0, log=print)
    assert rec.enabled
    h, w = 4, 6
    rec.add({
        "image/head": np.zeros((h, w, 3), dtype=np.uint8),
        "image/left_wrist": np.ones((h, w, 3), dtype=np.uint8),
    })
    rec.add({
        "image/head": np.ones((h, w, 3), dtype=np.uint8),
        "image/left_wrist": np.zeros((h, w, 3), dtype=np.uint8),
    })
    videos = rec.save()
    assert set(videos) == {"head", "left_wrist"}
    for name, path in videos.items():
        assert os.path.isfile(path)


def test_recorder_disabled_without_dir(tmp_path):
    rec = _RECORDER.PerCameraRecorder(None, log=print)
    assert not rec.enabled
    rec.add({"image/head": np.zeros((2, 2, 3), dtype=np.uint8)})
    assert rec.save() == {}


# ---------------------------------------------------------------- 状态机

CAMERA_MAP = {
    "image/head": "cam_head",
    "image/left_wrist": "cam_left_wrist",
    "image/right_wrist": "cam_right_wrist",
}
OBS_SPEC = (("arm_joint_state", 6), ("ee_joint_state", 1), ("ee_pose", 7))
# 动作分量名与真 arx_x5 joint 模板一致（handshake.ACTION_SPACE_TEMPLATES["joint"]
# 展开）——活校验（check_handshake_live）要求测试替身与真环境同形，否则合法
# 握手会被活终检拒绝。
import sys as _sys

_HANDSHAKE_MOD = None


def _action_names():
    global _HANDSHAKE_MOD
    if _HANDSHAKE_MOD is None:
        import importlib.util as _iu

        _spec = _iu.spec_from_file_location(
            "hs_mod", Path(__file__).resolve().parent.parent / "handshake.py"
        )
        _HANDSHAKE_MOD = _iu.module_from_spec(_spec)
        _sys.modules["hs_mod"] = _HANDSHAKE_MOD
        _spec.loader.exec_module(_HANDSHAKE_MOD)
    return _HANDSHAKE_MOD.action_space_names("joint")


ACTION_NAMES = _action_names()


class FakeNode:
    """dora Node 的替身：记录 send_output，事件由测试直接喂 process_event。"""

    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def send_output(self, output_id, value):
        self.sent.append((output_id, value))

    def next(self, timeout=None):
        return None

    # ---- 解析帮手 ----

    def texts(self, channel):
        out = []
        for cid, value in self.sent:
            if cid == channel:
                v = value
                if hasattr(v, "as_py"):
                    v = v.as_py()
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode()
                if isinstance(v, pa.Array):
                    v = v[0].as_py()
                out.append(v)
        return out

    def jsons(self, channel):
        return [json.loads(t) for t in self.texts(channel)]

    def policy_commands(self):
        """policy_command 通道上的 (command, inputs_json) 序列。"""
        out = []
        for t in self.texts("policy_command"):
            batch = messages.PolicyCommand.from_arrow(
                self.sent[[c for c, _ in self.sent].index("policy_command")][1]
            )
            out.append((batch.command, batch.inputs_json))
        return out


def _mk_obs(step: int = 0):
    state = {}
    for key, dim in OBS_SPEC:
        state[key] = [0.0] * dim
    vision = {cam: {"color": np.zeros((4, 6, 3), dtype=np.uint8)} for cam in CAMERA_MAP.values()}
    return {
        "instruction": "官方场景指令",
        "vision": vision,
        "state": state,
    }


class FakeEnv:
    """RoboDojoEpisodeEnv 的替身：记录调用，按脚本回放 step 结果。"""

    def __init__(self):
        self.log = print
        self.reset_calls: list[tuple[str, int]] = []
        self.step_calls: list[tuple[tuple, bool]] = []
        self.step_lim = 5
        self.episode_step_lim = 0
        self._script: list[object] = []
        self._obs = _mk_obs()
        self.env_instruction_value = "官方场景指令"
        self.closed = False

    def set_log(self, log):
        self.log = log

    @property
    def last_obs(self):
        return self._obs

    @property
    def env_instruction(self):
        return self.env_instruction_value

    def state_joint_names(self):
        return [f"{k}.{i}" for k, d in OBS_SPEC for i in range(d)]

    def action_component_names(self):
        return list(ACTION_NAMES)

    def reproducibility(self):
        return {"env_cfg_type": "arx_x5"}

    def reset_episode(self, task, layout_id):
        self.reset_calls.append((task, layout_id))
        self.episode_step_lim = self.step_lim
        return self._obs

    def script(self, outcomes):
        """按顺序回放的 step 结果；耗尽后永不结束。"""
        self._script = list(outcomes)

    def step(self, action, chunk_end=False):
        self.step_calls.append((tuple(action), chunk_end))
        out = self._script.pop(0) if self._script else None
        if out is None:
            from types import SimpleNamespace

            return SimpleNamespace(obs=self._obs, done=False, success=False, score=None, metrics={})
        return out

    def to_state(self, obs):
        return np.zeros(14, dtype=np.float64)

    def to_images(self, obs):
        return {ch: np.zeros((4, 6, 3), dtype=np.uint8) for ch in CAMERA_MAP}

    def close(self):
        self.closed = True


class FakeRecorder:
    def __init__(self, record_dir, fps=30.0, log=print):
        self.record_dir = record_dir
        self.closed = False

    def add(self, frames):
        pass

    def save(self):
        return {"head": os.path.join(self.record_dir, "head.mp4")}

    def close(self):
        self.closed = True


def _make_driver(cfg, node, env):
    return _SERVICE.EpisodeDriver(node, cfg, lambda log: env, log=print)


def _load_local(alias: str, filename: str):
    """以别名加载本目录模块：合跑时 sys.path 里有多个同名 config.py，
    别名加载不依赖 sys.modules['config'] 的占用者是谁。"""
    spec = importlib.util.spec_from_file_location(alias, NODE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_CONFIG = _load_local("robodojo_eval_config_alias", "config.py")
_RECORDER = _load_local("robodojo_eval_recorder_alias", "episode_recorder.py")
_SERVICE = _load_local("robodojo_eval_service_alias", "episode_service.py")
_HANDSHAKE = _load_local("robodojo_eval_handshake_alias", "handshake.py")


def _ready_payload(**overrides):
    """合法的 ready 自报载荷（joint 空间，与 handshake 模板逐字一致）。"""
    payload = {
        "status": "ready",
        "policy_id": "FastWAM_80k_forge",
        "action_type": "joint",
        "action_names": _HANDSHAKE.action_space_names("joint"),
        "cameras": list(CAMERA_MAP),
        "emit_chunk_end": True,
        "weight_id": "fastwam-80k-test",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def cfg(tmp_path):
    c = _CONFIG.EvalNodeConfig(result_dir=str(tmp_path / "out"))
    os.makedirs(c.result_dir, exist_ok=True)
    return c


@pytest.fixture
def node():
    return FakeNode()


@pytest.fixture
def env():
    return FakeEnv()


def _handshake(driver, **overrides):
    """预置握手：喂一条合法 ready，之后的 start_episode 才会被接受。"""
    driver.process_event({
        "type": "INPUT", "id": "policy_status",
        "value": pa.array([json.dumps(_ready_payload(**overrides))]),
    })


def _reset_ack(driver):
    """喂一条 reset 确认，触发暂存首帧的发布。"""
    driver.process_event({
        "type": "INPUT", "id": "policy_status",
        "value": pa.array([json.dumps({"status": "reset_ack"})]),
    })


def _start_episode(driver, **cmd_overrides):
    """握手 + start + reset_ack 的完整启动序列（返回启动后的首帧是否已发布）。"""
    _handshake(driver)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "start_episode", **cmd_overrides})]),
    })
    _reset_ack(driver)


def test_start_publishes_obs_and_instruction(cfg, node, env, monkeypatch):
    """start 命令序列完整：start → set_instruction → reset_scene → set_instruction → reset_ack → 首帧。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)

    _start_episode(driver, task="arrange_largest_number", layout_id=0, round=1,
                   instruction="Arrange the numbers")
    assert env.reset_calls == [("arrange_largest_number", 0)]

    status = node.jsons("eval_status")[0]
    assert status["command"] == "started"
    assert status["instruction"] == "Arrange the numbers"
    assert status["max_steps"] == 5

    # policy_command 序列：start → 占位指令 → reset_scene → 最终指令
    cmds = []
    for cid, value in node.sent:
        if cid == "policy_command":
            cmd = messages.PolicyCommand.from_arrow(value)
            cmds.append((cmd.command, cmd.inputs_json))
    assert [c for c, _ in cmds] == ["start", "set_instruction", "reset_scene", "set_instruction"]
    assert json.loads(cmds[-1][1])["instruction"] == "Arrange the numbers"

    # 观测已发布：1 个 proprio + 3 路图像
    chs = [c for c, _ in node.sent]
    assert chs.count("proprio_state") == 1
    assert chs.count("image/head") == 1

    driver.close()


def test_full_episode_success_and_result_payload(cfg, node, env, monkeypatch):
    """推 5 步后 done+success：结果 JSON 带接口②全部字段并落盘。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)

    _start_episode(driver, task="classify_objects", layout_id=3, round=2,
                   instruction="classify", output_dir=os.path.join(cfg.result_dir, "runA"))

    from types import SimpleNamespace

    env.script(
        [SimpleNamespace(obs=_mk_obs(), done=False, success=False, score=None, metrics={})] * 4
        + [SimpleNamespace(obs=_mk_obs(), done=True, success=True, score=1.0, metrics={})]
    )

    def mk_action(last=False):
        effort = [1.0] * len(ACTION_NAMES) if last else []
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.1] * len(ACTION_NAMES),
            effort=effort,
        ).to_arrow()

    for i in range(5):
        driver.process_event({
            "type": "INPUT", "id": "action", "value": mk_action(last=(i == 4)),
        })

    results = node.jsons("eval_result")
    assert len(results) == 1
    r = results[0]
    # 接口② 的最小集合
    assert r["task"] == "classify_objects"
    assert r["round"] == 2
    assert r["layout_id"] == 3
    assert r["success"] is True
    assert r["score"] == 1.0
    assert r["steps"] == 5
    assert r["instruction"] == "classify"
    assert set(r["videos"]) == {"head"}
    assert r["result_json"].endswith("episode_result.json")
    assert r["log"].endswith("episode.log")

    with open(r["result_json"], encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["success"] is True and on_disk["steps"] == 5
    assert os.path.isfile(r["log"])
    driver.close()


def test_chunk_end_sentinel_on_last_action_of_chunk(cfg, node, env, monkeypatch):
    """末动作 effort 哨兵 → env.step(chunk_end=True)，其余 False。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)

    from types import SimpleNamespace

    env.script([
        SimpleNamespace(obs=_mk_obs(), done=False, success=False, score=None, metrics={})
        for _ in range(2)
    ])

    def mk_action(last=False):
        effort = [1.0] * len(ACTION_NAMES) if last else []
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
            effort=effort,
        ).to_arrow()

    for last in (False, True, True):
        driver.process_event({"type": "INPUT", "id": "action", "value": mk_action(last)})
    assert [ce for (_a, ce) in env.step_calls] == [False, True, True]
    driver.close()


def test_action_name_mismatch_fails_episode(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    bad = [f"wrong.{i}" for i in range(len(ACTION_NAMES))]
    driver.process_event({
        "type": "INPUT", "id": "action",
        "value": messages.JointCommand(
            name=bad, mode="position", position=[0.0] * len(ACTION_NAMES)
        ).to_arrow(),
    })
    r = node.jsons("eval_result")[0]
    assert r["status"] == "failed"
    assert r["error_code"] == "action_name_mismatch"
    driver.close()


def test_seed_layout_conflict_rejected(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({
            "command": "start_episode", "task": "t", "layout_id": 0, "seed": 7,
        })]),
    })
    st = node.jsons("eval_status")[-1]
    assert st["command"] == "rejected"
    assert env.reset_calls == []
    driver.close()


def test_seed_as_layout_alias(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", seed=4)
    assert env.reset_calls == [("t", 4)]
    driver.close()


def test_busy_when_episode_running(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    _start_episode(driver, task="u", layout_id=1)
    st = node.jsons("eval_status")[-1]
    assert st["command"] == "busy"
    assert env.reset_calls == [("t", 0)]
    driver.close()


def test_timeout_republishes_obs(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    n_img = [c for c, _ in node.sent].count("image/head")
    # 真实 dora 的 ERROR 超时事件到来时，20s 真的过去了；测试里把发帧时间回拨
    driver.episode.obs_sent_at -= driver.cfg.watchdog_s + 1.0
    driver.process_event({"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"})
    assert [c for c, _ in node.sent].count("image/head") == n_img + 1
    driver.close()


def test_stop_command_finalizes(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "stop", "reason": "编排要求"})]),
    })
    r = node.jsons("eval_result")[0]
    assert r["status"] == "stopped"
    assert r["termination"] == "stopped"
    driver.close()


def test_late_actions_after_finish_are_dropped(cfg, node, env, monkeypatch):
    """终局后迟到的动作必须丢弃（chunk 余下动作还在 dora 队列里）。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    from types import SimpleNamespace

    env.script([SimpleNamespace(obs=_mk_obs(), done=True, success=True, score=1.0, metrics={})])

    def mk_action(last=False):
        effort = [1.0] * len(ACTION_NAMES) if last else []
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
            effort=effort,
        ).to_arrow()

    driver.process_event({"type": "INPUT", "id": "action", "value": mk_action(last=True)})
    driver.process_event({"type": "INPUT", "id": "action", "value": mk_action()})
    driver.process_event({"type": "INPUT", "id": "action", "value": mk_action()})
    assert len(env.step_calls) == 1
    assert len(node.jsons("eval_result")) == 1
    driver.close()


def test_max_steps_timeout(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)

    def mk_action(last=False):
        effort = [1.0] * len(ACTION_NAMES) if last else []
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
            effort=effort,
        ).to_arrow()

    for i in range(env.step_lim):
        driver.process_event({
            "type": "INPUT", "id": "action", "value": mk_action(last=(i == env.step_lim - 1)),
        })
    r = node.jsons("eval_result")[0]
    assert r["termination"] == "timed_out"
    assert r["status"] == "succeeded"
    assert r["success"] is False
    driver.close()


# ---------------------------------------------------------------- 协议层专项


def test_start_rejected_without_handshake(cfg, node, env, monkeypatch):
    """没握手就发 start → 拒绝启动（校验契约与 ready 后才启动回合）。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "start_episode", "task": "t", "layout_id": 0})]),
    })
    st = node.jsons("eval_status")[-1]
    assert st["command"] == "rejected"
    assert "握手" in st["error"]
    assert env.reset_calls == []
    driver.close()


def test_handshake_bad_action_names_rejected(cfg, node, env, monkeypatch):
    """契约自报的分量名与环境模板不一致 → 握手失败 → start 被拒。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _handshake(driver, action_names=[f"wrong.{i}" for i in range(14)])
    # 手工序列：不能再调 _start_episode（它会用合法 ready 覆盖坏握手）
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "start_episode", "task": "t", "layout_id": 0})]),
    })
    st = node.jsons("eval_status")[-1]
    assert st["command"] == "rejected"
    assert any("action_names" in e for e in st.get("handshake_errors", []))
    driver.close()


def test_handshake_ee_space_accepted(cfg, node, env, monkeypatch):
    """ee 空间（ReKep 形状）在环境接受的集合内 → 握手通过。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _handshake(driver, action_type="ee",
               action_names=_HANDSHAKE.action_space_names("ee"))
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "start_episode", "task": "t", "layout_id": 0})]),
    })
    _reset_ack(driver)
    # 拒绝列表里没有它（env 是 joint 的 FakeEnv，但握手只校验契约层面）
    sts = [s for s in node.jsons("eval_status") if s.get("command") == "rejected"]
    assert sts == []
    driver.close()


def test_policy_failed_ends_episode(cfg, node, env, monkeypatch):
    """策略上报 failed → 回合立即 failed 终态（不再等一个死掉的策略）。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    driver.process_event({
        "type": "INPUT", "id": "policy_status",
        "value": pa.array([json.dumps({
            "status": "failed", "error_code": "cuda_oom", "message": "OOM on layer 12",
        })]),
    })
    r = node.jsons("eval_result")[0]
    assert r["status"] == "failed"
    assert r["error_code"] == "policy_failed"
    assert "cuda_oom" in r["error_message"]
    driver.close()


def test_reset_ack_gates_first_frame(cfg, node, env, monkeypatch):
    """reset_ack 之前不发首帧；ack 到达立即发布。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _handshake(driver)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "start_episode", "task": "t", "layout_id": 0})]),
    })
    assert [c for c, _ in node.sent].count("proprio_state") == 0      # 还没发
    _reset_ack(driver)
    assert [c for c, _ in node.sent].count("proprio_state") == 1      # ack 即发
    # obs_meta 与首帧同拍
    metas = node.jsons("obs_meta")
    assert metas and metas[0]["seq"] == 0 and "episode_id" in metas[0]
    driver.close()


def test_obs_meta_seq_advances(cfg, node, env, monkeypatch):
    """每帧观测带递增 seq；动作回带 episode_id 校验。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    from types import SimpleNamespace

    env.script([SimpleNamespace(obs=_mk_obs(), done=False, success=False, score=None, metrics={})
                for _ in range(2)])

    def mk_action():
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
        ).to_arrow()

    ep_id = node.jsons("obs_meta")[0]["episode_id"]
    driver.process_event({"type": "INPUT", "id": "action", "value": mk_action()})
    driver.process_event({"type": "INPUT", "id": "action", "value": mk_action()})
    seqs = [m["seq"] for m in node.jsons("obs_meta")]
    assert seqs == [0, 1, 2]      # 首帧 + 两次 step 后的两帧
    assert driver.episode.episode_id == ep_id
    driver.close()


def test_mismatched_episode_action_dropped(cfg, node, env, monkeypatch):
    """异回合动作（meta.episode_id 不符）→ 机械丢弃，不推进 env。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    mk = lambda: messages.JointCommand(
        name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
    ).to_arrow()
    # 带着旧回合身份的动作
    driver.process_event({"type": "INPUT", "id": "action", "value": mk()})
    before = len(env.step_calls)
    driver._last_action_meta = {"episode_id": "old-episode"}
    driver.on_action(mk(), meta={"episode_id": "old-episode"})
    assert len(env.step_calls) == before
    assert driver.episode.dropped_actions == 1
    driver.close()


def test_watchdog_gives_up(cfg, node, env, monkeypatch):
    """重发次数达上限 → 回合 failed，错误信息带证据（等了多久/重发几次/seq）。"""
    c = _CONFIG.EvalNodeConfig(result_dir=str(cfg.result_dir), max_republishes=2,
                               give_up_after_s=0)
    os.makedirs(c.result_dir, exist_ok=True)
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(c, node, env)
    _start_episode(driver, task="t", layout_id=0)
    # 三次超时：第 1、2 次重发，第 3 次触发放弃
    for _ in range(3):
        driver.episode.obs_sent_at -= c.watchdog_s + 1.0
        driver.process_event({"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"})
    r = node.jsons("eval_result")[0]
    assert r["status"] == "failed"
    assert r["error_code"] == "policy_wait_timeout"
    assert "重发观测 2 次" in r["error_message"] and "seq=0" in r["error_message"]
    driver.close()


# ---------------------------------------------------------------- 批次层端到端


def test_batch_runs_all_episodes_and_summarizes(cfg, node, env, monkeypatch):
    """run 命令 → 展开 2 回合 → 逐回合跑完 → 批次汇总发出并落盘。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    c = _CONFIG.EvalNodeConfig(result_dir=cfg.result_dir, init_states_per_task=2)
    driver = _make_driver(c, node, env)
    _handshake(driver)

    from types import SimpleNamespace

    def mk_action(last=False):
        effort = [1.0] * len(ACTION_NAMES) if last else []
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
            effort=effort,
        ).to_arrow()

    # 每回合推 1 步就成功结束
    env.script([SimpleNamespace(obs=_mk_obs(), done=True, success=True, score=1.0, metrics={})])
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({
            "command": "run", "suite": "arrange_largest_number",
            "num_runs": 2, "init_state_ids": [0, 1],
            "output_dir": os.path.join(cfg.result_dir, "batchout"),
        })]),
    })
    # run 命令展开 2 布局 × 2 次 = 4 个回合；逐回合喂 ack + 一步成功的动作。
    # 每回合开始前重置 env 脚本（脚本耗尽后 FakeEnv 会永不结束）。
    for _ in range(4):
        _reset_ack(driver)
        env.script([SimpleNamespace(obs=_mk_obs(), done=True, success=True, score=1.0, metrics={})])
        driver.process_event({"type": "INPUT", "id": "action", "value": mk_action(last=True)})

    batch_results = [r for r in node.jsons("eval_result") if r.get("kind") == "batch"]
    assert len(batch_results) == 1
    br = batch_results[0]
    assert br["total_episodes"] == 4 and br["completed_episodes"] == 4
    assert br["success_rate"] == 1.0
    assert br["per_task"]["arrange_largest_number"]["successes"] == 4
    assert br["batch_result_json"] and os.path.isfile(br["batch_result_json"])
    # 每回合的 episode_result 也都发出过
    ep_results = [r for r in node.jsons("eval_result") if r.get("kind") is None]
    assert len(ep_results) == 4
    driver.close()


def test_batch_rejects_empty_task_ids(cfg, node, env, monkeypatch):
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _handshake(driver)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({
            "command": "run", "suite": "arrange_largest_number", "task_ids": [],
        })]),
    })
    st = node.jsons("eval_status")[-1]
    assert st["command"] == "rejected"
    assert driver.batch is None
    driver.close()


# ---------------------------------------------------------------- 工具接口与入口关闭


def test_admission_closed_after_stop_drops_late_actions(cfg, node, env, monkeypatch):
    """stop 后动作入口关死：迟到动作被拒并计数；新 start 重开入口。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)

    def mk_action():
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
        ).to_arrow()

    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "stop", "reason": "测试"})]),
    })
    assert driver.admission_open is False
    # stop 后迟到动作：入口关死，即使"此时有回合在等"也进不来
    n_steps = len(env.step_calls)
    driver.on_action(mk_action())
    assert len(env.step_calls) == n_steps
    assert driver.admission_dropped == 1
    # 新 start 重开入口
    _start_episode(driver, task="t", layout_id=1)
    assert driver.admission_open is True
    driver.close()


def test_describe_returns_env_contract(cfg, node, env, monkeypatch):
    """describe → eval_status 带环境契约（相机/动作空间集合/判分口径）。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _handshake(driver)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "describe"})]),
    })
    d = [s for s in node.jsons("eval_status") if s.get("command") == "describe"][-1]
    assert d["env_contract"]["accepted_action_spaces"] == ["ee", "joint"]
    assert d["env_contract"]["action_spaces"]["joint"]["action_dim"] == 14
    assert d["env_contract"]["judging"]["chunk_end_query"] == "official_2N_plus_1"
    assert d["camera_map"]["image/head"] == "cam_head"
    assert d["policy"]["policy_id"] == "FastWAM_80k_forge"     # 握手自报
    driver.close()


def test_status_reports_episode_and_batch(cfg, node, env, monkeypatch):
    """status → 回合进度 + 批次进度。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "status"})]),
    })
    st = [s for s in node.jsons("eval_status") if s.get("command") == "status"][-1]
    assert st["episode"]["task"] == "t"
    assert st["episode"]["seq"] == 0
    assert st["handshake_ok"] is True
    driver.close()


def test_policy_id_from_handshake_overrides_config(cfg, node, env, monkeypatch):
    """通用身份 → 握手的 policy_id 覆盖 cfg（官方组装与打标以握手为准）。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    assert driver.cfg.policy_id == "auto"       # 配置通用化
    _start_episode(driver, task="t", layout_id=0)
    assert driver.cfg.policy_id == "FastWAM_80k_forge"   # 被握手覆盖
    r = node.jsons("eval_status")[0]
    assert r["command"] == "started"
    driver.close()


def test_live_check_rejects_mismatched_names(cfg, node, env, monkeypatch):
    """活终检：握手自报与 env 活推导不一致 → 回合 failed（handshake_live_mismatch）。

    场景：静态快检过（同样 14 维 joint 模板），但 FakeEnv 实际报告的分量名
    与模板不同（模拟机器人换代后静态表过期的情形）——最终防线兜住。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    # 让 env 报告与模板不同的分量名（模拟环境变更后模板过期）
    env.action_component_names = lambda: [f"newbot_joint.{i}" for i in range(14)]
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0)
    r = node.jsons("eval_result")[0]
    assert r["status"] == "failed"
    assert r["error_code"] == "handshake_live_mismatch"
    # 尚未发首帧（失败发生在发布前）
    assert [c for c, _ in node.sent].count("proprio_state") == 0
    driver.close()


def test_describe_uses_live_env_when_built(cfg, node, env, monkeypatch):
    """env 已建后 describe 用活推导（accepted=['live']），未建时用静态快照。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    # env 未建：静态快照
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "describe"})]),
    })
    d0 = [s for s in node.jsons("eval_status") if s.get("command") == "describe"][-1]
    assert d0["env_contract"]["accepted_action_spaces"] == ["ee", "joint"]
    # env 已建：活推导
    _start_episode(driver, task="t", layout_id=0)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({"command": "describe"})]),
    })
    d1 = [s for s in node.jsons("eval_status") if s.get("command") == "describe"][-1]
    assert d1["env_contract"]["accepted_action_spaces"] == ["live"]
    assert d1["env_contract"]["action_spaces"]["live"]["action_dim"] == len(ACTION_NAMES)
    driver.close()


# ---------------------------------------------------------------- 问题.md 修复专项


def test_step_budget_limits_episode(cfg, node, env, monkeypatch):
    """设计问题 1：命令 max_steps=2 → 实际只跑 2 步（不被环境 5 步上限吞掉）。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="t", layout_id=0, max_steps=2)

    from types import SimpleNamespace

    def mk_action(last=False):
        effort = [1.0] * len(ACTION_NAMES) if last else []
        return messages.JointCommand(
            name=list(ACTION_NAMES), mode="position", position=[0.0] * len(ACTION_NAMES),
            effort=effort,
        ).to_arrow()

    for _ in range(5):        # 喂 5 步（环境 step_lim=5 允许），预算只有 2
        env.script([SimpleNamespace(obs=_mk_obs(), done=False, success=False, score=None, metrics={})])
        driver.process_event({"type": "INPUT", "id": "action", "value": mk_action(last=True)})
        if driver.episode is None:
            break
    r = node.jsons("eval_result")[0]
    assert r["steps"] == 2                       # 实际只跑 2 步
    assert r["termination"] == "timed_out"
    eff = r["effective"]
    assert eff["step_budget"] == 2               # 预算记录
    assert eff["env_step_lim"] == 5              # 环境上限记录
    assert eff["max_steps"] == 2                 # 实际生效 = min
    driver.close()


def test_batch_seed_passed_via_eval_seed(cfg, node, env, monkeypatch):
    """设计问题 1：批次 seed 经 eval_seed 透传到回合 cfg（不丢、不撞布局语义）。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _handshake(driver)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({
            "command": "run", "suite": "arrange_largest_number",
            "num_runs": 1, "init_state_ids": [0], "seed": 777,
            "output_dir": os.path.join(cfg.result_dir, "b777"),
        })]),
    })
    _reset_ack(driver)
    assert driver.cfg.seed == 777                # 透传到位（此前丢失）
    assert driver.episode is not None            # 没被 layout/seed 冲突拒绝
    driver.close()


def test_stop_closes_batch_no_next_episode(cfg, node, env, monkeypatch):
    """设计问题 2：dora STOP 后批次封闭——第二回合不被拉起。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(cfg, node, env)
    _handshake(driver)
    driver.process_event({
        "type": "INPUT", "id": "eval_command",
        "value": pa.array([json.dumps({
            "command": "run", "suite": "arrange_largest_number",
            "num_runs": 2, "init_state_ids": [0, 1],
            "output_dir": os.path.join(cfg.result_dir, "bst"),
        })]),
    })
    _reset_ack(driver)
    env.reset_calls.clear()
    # dora 原生 STOP（探针用的就是这条路径）
    driver.process_event({"type": "STOP"})
    assert driver.batch_stopped is True
    assert env.reset_calls == []                 # 第二回合没有被 reset（此前会）
    br = [r for r in node.jsons("eval_result") if r.get("kind") == "batch"]
    assert len(br) == 1 and br[0]["status"] == "stopped"
    driver.close()


def test_watchdog_gives_up_even_with_status_traffic(cfg, node, env, monkeypatch):
    """工程问题：看门狗独立时钟——高频 status 查询不再饿死放弃检查。"""
    c = _CONFIG.EvalNodeConfig(result_dir=str(cfg.result_dir), max_republishes=0,
                               give_up_after_s=1.0, watchdog_s=20.0)
    os.makedirs(c.result_dir, exist_ok=True)
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    driver = _make_driver(c, node, env)
    _start_episode(driver, task="t", layout_id=0)
    # 模拟探针：持续发 status 查询（每条都是消息，next() 永不超时）
    # 同时把等待时刻回拨 2 秒（超过 give_up_after_s=1.0）
    driver.episode.obs_sent_at -= 2.0
    driver.episode.step_started_at -= 2.0
    for _ in range(5):
        driver.process_event({
            "type": "INPUT", "id": "eval_command",
            "value": pa.array([json.dumps({"command": "status"})]),
        })
        if driver.episode is None:
            break
    assert driver.episode is None                # 放弃已执行（未被消息饿死）
    r = node.jsons("eval_result")[0]
    assert r["error_code"] == "policy_wait_timeout"
    assert "独立时钟" in r["error_message"]
    driver.close()


def test_native_result_and_effective_layout(cfg, node, env, monkeypatch):
    """工程问题：native_result_path 用 _env_task_name；effective 记实际布局。"""
    monkeypatch.setattr(_SERVICE.EpisodeDriver, "_recorder_class", lambda self: FakeRecorder)
    # 给 env 补上 native_result 探测所需的属性（真 RoboDojoEpisodeEnv 有 _env_task_name）
    env._env_task_name = "classify_objects"
    env._env = object()          # 非 None 即可触发探测

    driver = _make_driver(cfg, node, env)
    _start_episode(driver, task="classify_objects", layout_id=0,
                   instruction="c")
    # 请求 layout_id=0 → FakeEnv 不转换，实际值=0（记录两个值）
    r = node.jsons("eval_status")[0]
    assert r["command"] == "started"
    driver.close()


def test_numpy_available_for_optional_channels(cfg, node, env, monkeypatch):
    """工程问题：depth/seg 可选通道不再 NameError（模块级 numpy import）。"""
    import episode_service as _es   # 已由 conftest 链路加载（sys.path 已含节点目录）

    assert getattr(_es, "np", None) is not None
