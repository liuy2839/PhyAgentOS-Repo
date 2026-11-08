"""rekep_policy 节点入口：装配算法栈并用统一策略协议运行。

- 默认装配真实算法（rekep_alg 三模块 + IK），见 assembly.py；
- --self-test 用 fake 组件只验证协议链路（无需 torch/mujoco）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# 让同仓库的 forge_policy SDK 可导入（部署环境装包后可去掉）
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "policy" / "src"))

from forge_policy import PolicyContract, PolicyRuntime  # noqa: E402

from .core import ReKepPolicy  # noqa: E402
from .profile import RobotProfile  # noqa: E402


def load_robot_profile(path: str) -> RobotProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lower = data.get("arm_lower")
    upper = data.get("arm_upper")
    return RobotProfile(
        action_names=tuple(data["action_names"]),
        arm_action_names=tuple(data["arm_action_names"]),
        gripper_action_name=data["gripper_action_name"],
        action_space=data.get("action_space", "joint"),
        pose_action_names=tuple(data.get("pose_action_names", ())),
        gripper_open_value=float(data.get("gripper_open_value", 0.0)),
        gripper_close_value=float(data.get("gripper_close_value", 1.0)),
        arm_lower=tuple(lower) if lower is not None else None,
        arm_upper=tuple(upper) if upper is not None else None,
        hold_uncontrolled=bool(data.get("hold_uncontrolled", True)),
    )


# ------------------------------------------------------------- self-test fakes
class _FakePerception:
    """用 scene_objects（或合成物体）产出快照与世界状态。"""

    def __init__(self) -> None:
        self._names: list[str] = []

    def keypoints(self, obs: dict[str, Any]) -> dict[str, Any]:
        objs = obs.get("scene_objects") or [
            {"name": "cube", "type": "rigid", "position": [0.4, 0.0, 0.8], "quat_wxyz": [1, 0, 0, 0]},
            {"name": "zone", "type": "geometry", "position": [0.3, 0.2, 0.8], "quat_wxyz": [1, 0, 0, 0]},
        ]
        kps = [
            {
                "index": i,
                "position_m": list(o["position"][:3]),
                "object": o["name"],
                "kind": "center",
            }
            for i, o in enumerate(objs)
        ]
        self._names = [o["name"] for o in objs]
        return {
            "observation_id": "fake-obs",
            "keypoints": kps,
            "objects": [
                {"name": o["name"], "movable": o.get("type") == "rigid",
                 "region": o.get("type") != "rigid", "position_m": list(o["position"][:3])}
                for o in objs
            ],
        }

    def world_state(self, obs: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
        by_name = dict(zip(obs.get("state_names", []), obs.get("state", []), strict=False))
        ee = [by_name.get(f"left_ee_pose.{i}", 0.0) for i in range(7)]
        arm = [v for k, v in sorted(by_name.items()) if k.startswith("left_arm_joint_state.")]
        kps = np.asarray([k["position_m"] for k in snapshot["keypoints"]], dtype=float)
        return {"ee_pose": np.asarray(ee, dtype=float), "arm_q": np.asarray(arm, dtype=float),
                "keypoints": kps}


class _FakePlanner:
    def plan(self, instruction: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        from rekep_alg.planner import make_template_program

        return make_template_program(snapshot, instruction)


class _FakeExecutor:
    """不调求解器：给几个固定的 Cartisian 目标后结束，验证动作链路。"""

    def __init__(self) -> None:
        self._calls = 0

    def start_episode(self, plan: dict[str, Any], snapshot: dict[str, Any]) -> None:
        self._calls = 0

    def step(self, world_state: dict[str, Any]) -> dict[str, Any]:
        self._calls += 1
        if self._calls == 2:
            return {"type": "close_gripper"}
        if self._calls == 3:
            return {"type": "open_gripper"}
        if self._calls > 4:
            return {"type": "done"}
        return {
            "type": "cartesian",
            "pose7": [0.4, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
            "gripper": "open" if self._calls == 1 else "hold",
        }

    def on_grasp_result(self, ok: bool) -> dict[str, Any]:
        return {"type": "waiting"}

    def on_release_done(self) -> dict[str, Any]:
        return {"type": "waiting"}


def main() -> int:
    parser = argparse.ArgumentParser(description="ReKep 统一策略节点")
    parser.add_argument("--profile", help="robot profile JSON 路径")
    parser.add_argument(
        "--cameras", nargs="*", default=["image/head"],
        help="需要收齐的相机输入 topic",
    )
    parser.add_argument(
        "--scene-objects", action="store_true", default=True,
        help="向评测端声明需要 scene_objects 输入（默认开）",
    )
    parser.add_argument("--no-scene-objects", dest="scene_objects", action="store_false")
    parser.add_argument("--self-test", action="store_true", help="fake 组件协议自检")
    args = parser.parse_args()

    if args.self_test:
        profile = _self_test_profile()
        modules: dict[str, Any] = {
            "perception": _FakePerception(),
            "planner": _FakePlanner(),
            "executor": _FakeExecutor(),
        }
    else:
        if not args.profile:
            parser.error("需要 --profile（或 --self-test）")
        profile = load_robot_profile(args.profile)
        from .assembly import build_alg_modules

        modules = build_alg_modules(profile, log=lambda m: print(f"[rekep_policy] {m}", flush=True))

    use_dino = os.environ.get("REKEP_PERCEPTION", "scene_objects").lower() == "dino"
    # 每帧制策略（每帧重新决策 = 官方 N=1 chunk）：每个动作都是决策单元的
    # 末动作，必须带 effort 哨兵，评测端补查 is_episode_end() 的口径才与
    # RoboDojo 官方 rollout 对齐（此前 emit_chunk_end=False，每 chunk 少一次
    # 外层结束查询，成绩口径系统性偏低）。
    contract = PolicyContract(
        action_names=profile.action_names,
        cameras=tuple(args.cameras),
        depth_cameras=("depth/head",) if use_dino else (),
        scene_objects=bool(args.scene_objects),
        camera_info=use_dino,
        emit_chunk_end=True,
        chunk_per_action=True,
    )
    policy = ReKepPolicy(
        profile=profile,
        perception=modules["perception"],
        planner=modules["planner"],
        executor=modules["executor"],
        log=lambda m: print(f"[rekep_policy] {m}", flush=True),
    )
    return PolicyRuntime(policy, contract).run()


def _self_test_profile() -> RobotProfile:
    # 仿真 ARX X5 式命名：左臂 6 关节 + 左夹爪；右臂分量以 hold 跟随
    return RobotProfile(
        action_names=(
            "left_arm_joint_state.0", "left_arm_joint_state.1", "left_arm_joint_state.2",
            "left_arm_joint_state.3", "left_arm_joint_state.4", "left_arm_joint_state.5",
            "left_ee_joint_state.0",
            "right_arm_joint_state.0",
        ),
        arm_action_names=(
            "left_arm_joint_state.0", "left_arm_joint_state.1", "left_arm_joint_state.2",
            "left_arm_joint_state.3", "left_arm_joint_state.4", "left_arm_joint_state.5",
        ),
        gripper_action_name="left_ee_joint_state.0",
        action_space="joint",
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
