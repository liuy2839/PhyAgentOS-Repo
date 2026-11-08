"""评测发起节点的配置 —— 只管"怎么起一个回合"，不再有批次概念。

与原 forge 流程的差别：批次展开（task_ids × init_state_ids × num_runs）、
serve/护栏、describe 能力快照都是 `_core` + forge-tool 工具面的需求；解耦后
一次 eval_command = 一个回合，任务/布局/轮次/指令全在命令里，这里只剩
环境口径（env_cfg_type、动作语义、相机映射）和落盘位置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# dora 通道 → RoboDojo 相机名。原生相机名唯一，别自己发明别名。
CAMERA_MAP: dict[str, str] = {
    "image/head": "cam_head",
    "image/left_wrist": "cam_left_wrist",
    "image/right_wrist": "cam_right_wrist",
}


@dataclass
class EvalNodeConfig:
    benchmark: str = "robodojo"
    # env_cfg/<name>.yml，决定机器人/场景/相机/sim 四份子配置。ARX X5 双臂。
    env_cfg_type: str = "arx_x5"
    # 进程可见的 GPU 序号（外面用 CUDA_VISIBLE_DEVICES 选卡）。
    device_id: int = 0
    # 动作语义，必须与策略 deploy.yml 一致：joint → arm_joint_state；ee → ee_pose。
    action_type: str = "joint"
    # 仅用于官方配置组装与结果标注（FastWAM_80k_forge / G05），动作实际从 dora 来。
    policy_id: str = "FastWAM_80k_forge"
    additional_info: str = "forge"
    # 官方 eval_cfg["seed"]（影响过程随机化）；**不是布局 ID** —— 布局在
    # eval_command 的 layout_id 里逐回合指定。
    seed: int = 0
    # 可选观测通道（与原流程 adapter 同名开关，默认全关：FastWAM 只吃 RGB）。
    # scene_objects：环境无 depth/seg 时的物体真值位姿快照（ReKep 类策略依赖）；
    # camera_info：相机内/外参（配合 depth 反投影）。
    publish_scene_objects: bool = False
    publish_camera_info: bool = False
    camera_map: dict = field(default_factory=lambda: dict(CAMERA_MAP))
    # output_dir 未在命令里给出时的落盘根目录。
    result_dir: str = "./rsi_results"
    # 等 action 的看门狗周期（秒）。明显大于最慢策略的单步推理耗时
    # （G05/FastWAM 首帧秒级），又不至于把真死锁挂太久。
    watchdog_s: float = 20.0
    # 环境不暴露 step_lim 时的兜底 horizon（RoboDojo 各任务自带，正常用不到）。
    max_steps_default: int = 1050


def load_config(path: str | Path) -> EvalNodeConfig:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cfg = EvalNodeConfig()
    cfg.env_cfg_type = str(data.get("env_cfg_type", cfg.env_cfg_type))
    cfg.device_id = int(data.get("device_id", cfg.device_id))
    cfg.action_type = str(data.get("action_type", cfg.action_type))
    if cfg.action_type not in ("joint", "ee"):
        raise ValueError(f'action_type 必须是 "joint" 或 "ee"，收到 {cfg.action_type!r}')
    cfg.policy_id = str(data.get("policy_id", cfg.policy_id))
    cfg.additional_info = str(data.get("additional_info", cfg.additional_info))
    cfg.seed = int(data.get("seed", cfg.seed))
    cam = data.get("camera_map")
    if cam:
        if not isinstance(cam, dict):
            raise ValueError("camera_map 必须是 {dora 通道: 相机名} 的映射")
        cfg.camera_map = dict(cam)
    cfg.result_dir = str(data.get("result_dir", cfg.result_dir))
    cfg.publish_scene_objects = bool(data.get("publish_scene_objects", cfg.publish_scene_objects))
    cfg.publish_camera_info = bool(data.get("publish_camera_info", cfg.publish_camera_info))
    cfg.watchdog_s = float(data.get("watchdog_s", cfg.watchdog_s))
    if cfg.watchdog_s <= 0:
        raise ValueError("watchdog_s 必须 > 0")
    cfg.max_steps_default = int(data.get("max_steps_default", cfg.max_steps_default))
    if cfg.max_steps_default < 1:
        raise ValueError("max_steps_default 必须 ≥1")
    return cfg
