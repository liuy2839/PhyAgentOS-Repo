#!/usr/bin/env python3
"""评测发起节点（模块 A）—— 标准 dora 数据面节点，一次命令 = 一个回合。

与原 PhyAgentOS/forge 流程（forge_robodojo_eval/robodojo 的 gateway +
bench_tool + _core 批次调度 + skill 打包）的解耦边界：
  * 不再有 forge-tool 工具面 / session 租约 —— 编排方（模块 D，宇琛）经
    `eval_command`（JSON 文本通道）逐回合驱动；
  * 不再有 `_core` 批次调度 —— v2 文档接口① 一次一个回合，指令逐回合给定；
  * 本目录自包含：不 import `_core` / `bench_tool`，只依赖 dora/numpy/
    pyarrow/yaml/opencv 与 RoboDojo 官方仓库本身；
  * RoboDojo 官方链路的全部防坑逻辑原样保留（is_episode_end 调用次数协议、
    官方 run_eval 前置、WsModelClient stub、step_lim 跟环境走），成绩口径
    与官方 rollout 一致。

**启动顺序有硬约束**：Isaac Sim 的 `AppLauncher` 必须在 import 任何
isaaclab/omni 相关模块之前跑完，否则 Kit 起不来。所以本文件的 import 是
分段的，不要为了"整洁"把下半段的 import 提到顶上。

输入通道：
    eval_command   接口① 的 start_episode / stop（JSON 文本）
    action         策略节点的 JointCommand（1:1 锁步）

输出通道：
    proprio_state / image/head / image/left_wrist / image/right_wrist
    policy_command （start / set_instruction / reset_scene / stop）
    eval_status    （started / busy / rejected / 进度日志，JSON 文本）
    eval_result    接口② 的回合结果（JSON 文本）
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _log(msg: str) -> None:
    print(f"[robodojo_eval] {msg}", flush=True)


def main() -> int:
    # **先把配置路径定死成绝对路径**：下面会 os.chdir 到 RoboDojo 仓库根，
    # 之后任何相对路径都会解析到那边去。
    config_path = os.environ.get("EVAL_CONFIG", "config.yaml")
    if not os.path.isabs(config_path):
        config_path = os.path.join(_HERE, config_path)

    # RoboDojo 仓库根：env/ task/ utils/ src/ 都在那儿，必须进 sys.path。
    dojo_root = os.environ.get("ROBODOJO_ROOT")
    if not dojo_root or not os.path.isdir(dojo_root):
        raise RuntimeError("必须设置 ROBODOJO_ROOT 指向 RoboDojo 仓库根目录")
    if dojo_root not in sys.path:
        sys.path.insert(0, dojo_root)
    # 官方代码里到处用相对路径（eval_result/ 等），CWD 必须是仓库根。
    os.chdir(dojo_root)

    from config import load_config  # noqa: E402  (本目录的 config.py)

    cfg = load_config(config_path)
    # 输出路径：chdir 之后相对路径会落到 RoboDojo 仓库里去，先解析成绝对路径。
    if not os.path.isabs(cfg.result_dir):
        cfg.result_dir = os.path.normpath(os.path.join(_HERE, cfg.result_dir))

    # ---- 第一段结束：下面开始碰 Isaac Sim ----
    from isaaclab.app import AppLauncher  # noqa: E402

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.headless = True
    app_args.device = f"cuda:{cfg.device_id}"
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    simulation_app = AppLauncher(app_args).app
    _log(f"Isaac Sim 已启动 (device=cuda:{cfg.device_id}, headless)")

    # ---- 第二段：Kit 起来了，现在才能 import 依赖它的东西 ----
    from dora import Node  # noqa: E402

    from episode_service import EpisodeDriver  # noqa: E402
    from robodojo_env import RoboDojoEpisodeEnv  # noqa: E402

    def make_env(log):
        return RoboDojoEpisodeEnv(cfg, simulation_app, log=log)

    driver = EpisodeDriver(Node(), cfg, make_env, log=_log)
    _log(
        f"回合由 eval_command 逐个指定 | env_cfg={cfg.env_cfg_type} "
        f"action_type={cfg.action_type} | 结果落盘根 {cfg.result_dir}"
    )
    try:
        return driver.run()
    finally:
        driver.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
