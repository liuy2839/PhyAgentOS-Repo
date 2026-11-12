"""RoboDojo 官方 eval env 的封装 —— 评测发起节点的环境层。

从 `forge_robodojo_eval/robodojo/adapters/robodojo_adapter.py` 提取：保留全部
实测防坑逻辑（逐字不变），去掉的只有 `_core` 批次调度相关部分（suite 展开、
describe 能力快照、depth/seg/scene_objects 可选通道 —— 那些是 forge-tool 工具
面的需求，RSI 两节点流程不需要）。

本目录**不 import `_core`/`bench_tool`/forge-tool** —— 与原流程的解耦边界。
wire 协议（chunk_end 哨兵、官方判定序列）与原 adapter 完全一致。

三条不能动的理由（详细注释在对应位置）：

1. `is_episode_end()` 的调用次数影响成绩 → chunk_end 补发协议；
2. `run_reward()/get_score()`/外层首次 `is_episode_end()` 是官方 run_eval 的
   前置，漏一句就是"零动作满分"的假阳性；
3. `AppLauncher` 必须先于一切 isaaclab import 跑完 → 本文件顶层不 import
   任何 isaaclab/omni 模块，env 构造推迟到 main.py 起完 app 之后。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------- specs
# 键名取自 XPolicyLab 的标准观测/动作格式（README 的 Standard Data Formats），
# **不能自己起名**：这些键要原样进 take_action。
#
# **维度绝对不能写死。** ARX X5 是 6 DOF 手臂（arm_dim=[6,6], ee_dim=[1,1]，
# 总 14）。一律走官方的 get_robot_action_dim_info(env_cfg_type)，换机器人只换
# env_cfg_type。两边（本节点与策略节点）算出来必须完全一致：不一致时动作分量
# 名校验会直接把这一集判失败（宁可炸掉，也不要静默跑歪）。
def build_state_spec(env_cfg_type: str, action_type: str) -> tuple[tuple[str, int], ...]:
    """动作的 (键名, 维度) 序列。joint → arm_joint_state；ee → ee_pose(7)。"""
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
    if len(arm_dims) != len(ee_dims):
        raise ValueError(f"arm_dim 与 ee_dim 数量不一致：{info}")
    if len(arm_dims) == 1:
        prefixes = [""]
    elif len(arm_dims) == 2:
        prefixes = ["left_", "right_"]
    else:
        raise NotImplementedError(f"不支持 {len(arm_dims)} 条手臂")
    if action_type not in ("joint", "ee"):
        raise ValueError(f'action_type 必须是 "joint" 或 "ee"，收到 {action_type!r}')
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        if action_type == "joint":
            spec.append((f"{p}arm_joint_state", int(ad)))
        else:
            spec.append((f"{p}ee_pose", 7))
        spec.append((f"{p}ee_joint_state", int(ed)))
    return tuple(spec)


def build_obs_spec(env_cfg_type: str) -> tuple[tuple[str, int], ...]:
    """观测 state 的完整 (键名, 维度) 序列（含 ee_pose，28 维），按键名字典序。

    只把动作那 14 维塞进 proprio 通道，策略就再也看不到 ee_pose —— 而且不报错。
    首帧会拿真实观测校验键集，不一致直接炸。
    """
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(env_cfg_type)
    arm_dims, ee_dims = list(info["arm_dim"]), list(info["ee_dim"])
    prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
    spec: list[tuple[str, int]] = []
    for p, ad, ed in zip(prefixes, arm_dims, ee_dims):
        spec.append((f"{p}arm_joint_state", int(ad)))
        spec.append((f"{p}ee_joint_state", int(ed)))
        spec.append((f"{p}ee_pose", 7))
    return tuple(sorted(spec, key=lambda kv: kv[0]))


# ---------------------------------------------------------------- 守卫与 stub
def assert_real_robodojo() -> None:
    """确认 import 到的是真的 RoboDojo 包而不是同名目录造出的命名空间包。

    本仓库常被 clone 成 `robodojo`，与上游同名。父目录一旦进 sys.path，
    `import env` / `import task` 就会先撞上本仓库 → 变成没有 __init__ 的
    命名空间包 → 上游注册代码从不执行。症状是 "Task not found" 而不是
    ImportError，极难排查。
    """
    import env as _env

    if getattr(_env, "__file__", None) is None and not getattr(_env, "__path__", None):
        raise ImportError(
            "import 到的 `env` 是空命名空间包，多半是 sys.path 里混进了本仓库的父目录。"
            "只把 RoboDojo 仓库根挂上 sys.path。"
        )


def _stub_model_client() -> None:
    """把 eval_env 里的 WsModelClient 换成 stub。

    `create_eval_env()` 无条件构造 `WsModelClient` 且 `deploy_cfg["port"]` 必填，
    没有策略服务器就起不来。我们的动作从 dora 来，不 fork 上游代码，在调用前
    把该命名空间里的 WsModelClient 换掉。生命周期类调用放行（env.reset() 自己
    会调 model_client.call("reset")），取动作必须炸 —— 那说明有人误用了官方
    的 eval_one_episode()。
    """
    import src.eval_client.eval_env as _ee

    _LIFECYCLE = frozenset({"reset", "update_obs", "update_obs_batch",
                            "prepare_case", "trial_end"})

    class _NoPolicyClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def call(self, func_name: str | None = None, *a: Any, **k: Any) -> None:
            if func_name in _LIFECYCLE:
                return None
            raise RuntimeError(
                f"评测节点不经过 WS 策略服务器，但收到了 {func_name!r}。"
                "动作从 dora 的 action 通道来；走到这里说明有人调用了官方的 "
                "eval_one_episode()，那条路我们不用。"
            )

        def close(self) -> None:
            pass

    _ee.WsModelClient = _NoPolicyClient


# ---------------------------------------------------------------- 单步结果
@dataclass
class StepOutcome:
    """env.step() 的单步结果。

    `success` 的语义与官方一致：episode 结束且未被判失败。`score` 只在
    episode 结束时有意义（中途取到的是未结算的中间态）。
    """

    obs: dict[str, Any]
    done: bool
    success: bool
    score: float | None
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 环境封装
class RoboDojoEpisodeEnv:
    """一个实例管一个 Isaac Sim env（num_envs 恒为 1），一命令一回合。"""

    RECORD_FPS = 30.0

    def __init__(self, cfg: Any, simulation_app: Any, log: Callable[[str], None] = print) -> None:
        assert_real_robodojo()
        self._log = log
        self.cfg = cfg
        self._app = simulation_app
        self._env: Any = None
        self._env_task_name: str | None = None
        self._chunk_end_extra_calls = 0
        self._episode_end_calls = 0
        self._current_layout_id: int | None = None
        self._episode_started = False
        self._last_obs: dict[str, Any] | None = None

        from env.global_configs import BENCHMARK, ROOT_DIR

        self._benchmark = BENCHMARK
        self._root = ROOT_DIR
        self._state_spec = build_state_spec(cfg.env_cfg_type, cfg.action_type)   # 动作：14 维
        self._obs_spec = build_obs_spec(cfg.env_cfg_type)                        # 观测：28 维
        self._obs_spec_checked = False

    # ---- 元信息 ----

    def set_log(self, log: Callable[[str], None]) -> None:
        """回合开始时把日志切到该回合的 TeeLog（输出进 episode.log）。"""
        self._log = log

    def reproducibility(self) -> dict[str, Any]:
        """影响分数的口径，进结果 JSON。字段名与官方 _result.json 的 details 对齐。"""
        return {
            "env_cfg_type": self.cfg.env_cfg_type,
            "action_type": self.cfg.action_type,
            "seed_is_layout_id": True,
            "policy_id": self.cfg.policy_id,
        }

    def state_joint_names(self) -> list[str]:
        """观测 proprio 的分量名（28 个），不是动作的 14 个。"""
        return [f"{key}.{i}" for key, dim in self._obs_spec for i in range(dim)]

    def action_component_names(self) -> list[str]:
        """动作分量名，policy 必须逐字回填到 JointCommand.name。"""
        return [f"{key}.{i}" for key, dim in self._state_spec for i in range(dim)]

    # ---- env_cfg 组装 ----

    def _build_env_cfg(self, task_name: str) -> Any:
        """逐字照抄官方 src/eval_client/main.py 的组装顺序。

        任何一步顺序换了都可能改变随机化结果 —— 这里不做"看起来等价"的重排。
        """
        from omegaconf import OmegaConf

        from env.global_configs import ENV_CONFIG_PATH
        from utils.load_file import load_yaml
        from utils.pipeline_utils import (
            process_config,
            process_randomization,
            resolve_random_task_num_envs,
        )

        import importlib

        task_registry = importlib.import_module(f"task.{self._benchmark}.task_registry")
        bench_path = os.path.join(self._root, "task", self._benchmark)

        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, self.cfg.env_cfg_type + ".yml"))
        eval_cfg["task_name"] = task_name
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = int(self.cfg.device_id)
        eval_cfg["eval_batch"] = False          # 永远单环境
        eval_cfg["policy_name"] = self.cfg.policy_id
        eval_cfg["additional_info"] = self.cfg.additional_info
        eval_cfg["seed"] = int(self.cfg.seed)
        eval_cfg["physx_monitor_enabled"] = False

        # deploy_cfg 只为满足 create_eval_env 的必填校验；WsModelClient 已被
        # stub 掉，port 填什么都不会真的去连。
        deploy_cfg = {
            "policy_name": self.cfg.policy_id,
            "port": 0,
            "host": "localhost",
            "protocol": "ws",
            "policy_server_url": "ws://localhost:0",
            "evaluation_id": os.environ.get("ROBODOJO_RUN_ID", "rsi"),
            "trial_id": f"{task_name}-rsi",
            "action_case_id": f"{task_name}_case",
            "repeat_index": None,
        }

        env_cfg = OmegaConf.create(
            {
                "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
                "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
                "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
                "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
                "task_env": load_yaml(
                    task_registry.task_config_path(os.path.join(bench_path, "config"), task_name)
                ),
                "eval_cfg": eval_cfg,
                "deploy_cfg": deploy_cfg,
            }
        )
        num_envs = resolve_random_task_num_envs(task_name, 1, env_cfg.sim)
        OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        env_cfg.sim.seed = [0]
        return env_cfg

    # ---- episode 生命周期 ----

    def _ensure_env(self, task_name: str) -> Any:
        """换 task 必须重建 env：场景资产和 reward 规则都是 task 绑定的。

        同 task 的集与集之间照官方批次语义：`seed_manager.eval_step()` +
        `env.close()`（官方 main.py 的 while 体尾部）。close() 会
        `obs_manager.reset()` + 收掉 video writer；不调的话跨 episode 状态泄漏。
        标志必须**先复位再关**：reset_episode 里 ensure 之后还要读 seed_list，
        不复位就会连关两次。
        """
        if self._env is not None and self._env_task_name == task_name:
            if self._episode_started:
                self._episode_started = False
                self._env.seed_manager.eval_step()
                self._env.close()
            return self._env
        if self._env is not None:
            self._env.close()
            self._env = None
        _stub_model_client()
        from src.eval_client.eval_env import create_eval_env

        self._env = create_eval_env(self._build_env_cfg(task_name), self._app)
        self._env_task_name = task_name
        return self._env

    def reset_episode(self, task_name: str, layout_id: int) -> dict[str, Any]:
        """重置到指定布局，返回首帧观测。返回后可读 `episode_step_lim`。

        `layout_id` 必须是 `seed_manager.seed_list` 里的一个真实布局 ID
        （RoboDojo 的 "seed" 不是随机种子，是 `Assets/Eval_Layout/.../<seed>/`
        下的布局目录 ID；同一个布局 ID → 场景逐位确定）。
        """
        env = self._ensure_env(task_name)
        layout_ids = [int(x) for x in env.seed_manager.seed_list]
        if not layout_ids:
            raise RuntimeError(
                f"任务 {task_name} 在 Eval_Layout 下没有布局文件；"
                "多半是资产没下全或 seed 目录选错了"
            )
        if layout_id not in layout_ids:
            if 0 <= layout_id < len(layout_ids):
                resolved = layout_ids[layout_id]
                self._log(
                    f"layout_id {layout_id} 不在布局清单里，按第 {layout_id} 个布局解析为 {resolved}"
                )
                layout_id = resolved
            else:
                raise IndexError(
                    f"layout_id={layout_id} 越界：任务 {task_name} 的布局清单为 "
                    f"{layout_ids[:10]}{'...' if len(layout_ids) > 10 else ''}（共 {len(layout_ids)} 个）"
                )
        self._current_layout_id = int(layout_id)
        # 官方 reset 收的是一个列表，长度 = num_envs
        env.reset(seed=[int(layout_id)])

        # **官方 run_eval() 的前置，一句都不能漏**（eval_client/eval_env.py）：
        #
        #     def run_eval(self):
        #         self.run_reward()                      # ← 注册本 task 的成功判定条件
        #         if hasattr(self, "get_score"): self.get_score()
        #         ...
        #         self.eval_one_episode()
        #
        # `run_reward()` 把该任务的判定条件灌进 reward_manager（上游 RoboDojo
        # 仓库的 PR 检查项："run_reward() 必须有意义地调用
        # self.reward_manager.check(...)"）。不调它的后果：check_list /
        # final_check_list / trigger_check_list 全空 → `get_reward()` 直接返回
        # 1.0 → 第 1 步就判成功。零动作策略"一步搭好塔"，success_rate = 1.0。
        # **假阳性且分数报高。**
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()

        # 官方外层 `while not is_episode_end():` 的第一次求值 —— 它发生在
        # **首帧 get_obs 之前**，漏掉的话整条判定序列比官方少一次、从第一步就错位。
        env.is_episode_end()

        # **horizon 必须跟着环境走。** 每 task 的 step_lim 不同（build_tower 1050）。
        # 不同步的话会在 300 步截断，官方跑满 —— 两边不是同一件事，却不报错。
        # 从活的 env 读，不猜 step_lim 来自哪个配置文件。
        env_lim = int(getattr(env, "step_lim", 0) or 0)
        self.episode_step_lim = env_lim
        if env_lim:
            self._log(f"本任务 step_lim = {env_lim}")

        self._episode_end_calls = 0
        self._chunk_end_extra_calls = 0
        obs = env.get_obs()
        self._episode_started = True
        self._last_obs = obs
        return obs

    @property
    def env_instruction(self) -> str | None:
        """首帧观测里的官方指令（desc_manager 按当前场景生成，reset 前不存在）。"""
        obs = self._last_obs or {}
        v = obs.get("instruction")
        return str(v) if isinstance(v, str) and v else None

    @property
    def last_obs(self) -> dict[str, Any]:
        """最近一帧观测（看门狗重发用；语义见 episode_service._republish_if_stalled）。"""
        return self._last_obs or {}

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("env 还没建，reset_episode 必须先调用")
        return self._env

    def to_action_dict(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """扁平向量 → 官方动作 dict。切片边界与 state_spec 一致。"""
        out: dict[str, np.ndarray] = {}
        off = 0
        for key, dim in self._state_spec:
            out[key] = np.asarray(flat[off : off + dim], dtype=np.float32)
            off += dim
        return out

    def step(self, action: np.ndarray, chunk_end: bool = False) -> StepOutcome:
        env = self._require_env()
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = sum(d for _, d in self._state_spec)
        if act.size != expected:
            raise ValueError(f"动作维度应为 {expected}，收到 {act.size}")

        env.take_action(self.to_action_dict(act))

        # 内层那一次（每个动作后必调，包括 chunk 最后一个）
        done = bool(env.is_episode_end())
        self._episode_end_calls += 1
        # 外层那一次：官方在 chunk 走完、episode 还没结束时会再调一次（发生在
        # chunk 末动作之后、下一帧 get_obs 之前）。少调这一次会让 _final_check()
        # 的执行次数偏少 → 成绩系统性偏低。补发位置在 chunk 末尾，等价于官方
        # 下一 chunk 开头的那一次。
        if chunk_end and not done:
            done = bool(env.is_episode_end())
            self._episode_end_calls += 1
            self._chunk_end_extra_calls += 1

        # **`env.success` 的语义是"还没失败"，不是"任务已完成"。** 它在 reset 时
        # 被初始化成 True，由 reward_manager._mark_env_failed() 才翻成 False。
        # 每步直接读它，第 1 步就恒为 True —— 假阳性，且是把分数报高的方向。
        # 只有 episode 结束之后这个值才有意义。
        success = bool(done and env.success[0])
        # 分档得分，口径与官方 eval_env.run_eval 逐字一致：成功记 1.0，否则取
        # reward_manager 的过程分再除以 100。只在 episode 结束时取。
        episode_score = None
        if done:
            episode_score = 1.0 if success else 0.0
            if not success and hasattr(env, "get_score"):
                try:
                    process_scores = env.reward_manager.get_score()
                except Exception as exc:          # 不能因为取分失败就丢掉整集
                    self._log(f"取 process score 失败，本集按 0 记：{exc}")
                    process_scores = None
                if process_scores is not None:
                    episode_score = float(process_scores[0]) / 100.0
        # episode 已结束就不要再 get_obs()：官方在 break 之后不取观测，而
        # get_obs 内含 render() + _stream_vision()，多调一次就是多一帧。
        # （末帧由 is_episode_end() 内部的 get_obs_batch(last_frame=True) 取。）
        if not done:
            self._last_obs = env.get_obs()
        return StepOutcome(
            obs=self._last_obs or {},
            done=done,
            success=success,
            score=episode_score,
            metrics={
                "success": success,
                "episode_end_calls": self._episode_end_calls,
                # 官方 _result.json 的 details 里也是 layout_id，可直接对齐
                "layout_id": self._current_layout_id,
            },
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_task_name = None

    def native_result_path(self) -> str | None:
        """官方 eval 链路的 _result.json 路径（判分的原始证据）。

        RoboDojo 官方评测在 episode 结束时按 `eval_result/<benchmark>/<task>/
        <policy>/` 落盘原生结果。路径规则以官方 eval_client 实现为准，这里只做
        惰性探测（找不到就返回 None，不猜路径伪造证据）。
        """
        env = self._env
        if env is None:
            return None
        task = self.tasks[0] if getattr(self, "tasks", None) else None
        if not task:
            return None
        candidates = [
            os.path.join("eval_result", self._benchmark, task, self.cfg.policy_id),
            os.path.join("eval_result", self._benchmark, task),
        ]
        for base in candidates:
            if not os.path.isdir(base):
                continue
            # 目录内最新的 _result.json
            best = None
            for name in os.listdir(base):
                if name.endswith("_result.json"):
                    path = os.path.join(base, name)
                    if best is None or os.path.getmtime(path) > os.path.getmtime(best):
                        best = path
            if best is not None:
                return os.path.abspath(best)
        return None

    # ---- 观测转换 ----

    def to_state(self, obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("state") or {}
        if not self._obs_spec_checked:
            want, got = {k for k, _ in self._obs_spec}, set(state)
            if want != got:
                raise KeyError(
                    "观测 state 的键集与推导出来的不一致，说明机器人配置变了：\n"
                    f"  推导={sorted(want)}\n  实得={sorted(got)}"
                )
            self._obs_spec_checked = True
        parts: list[np.ndarray] = []
        for key, dim in self._obs_spec:
            v = state.get(key)
            if v is None:
                raise KeyError(f"观测缺少 state.{key}；拿到的键={sorted(state)}")
            # **用 float64，不要转 float32。** 中间过一道 float32 会在第 7 位
            # 有效数字上引入舍入，污染逐位等价比对。
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size != dim:
                raise ValueError(f"state.{key} 维度应为 {dim}，收到 {arr.size}")
            parts.append(arr)
        return np.concatenate(parts).astype(np.float64)

    def to_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """图像**已经是 RGB 且已解码**（XPolicyLab 的硬规矩），不做任何通道转换。

        任何 COLOR_BGR2RGB 都是 bug —— 上游 AGENTS.md 原话。
        """
        vision = obs.get("vision") or {}
        out: dict[str, np.ndarray] = {}
        for channel, cam in self.cfg.camera_map.items():
            entry = vision.get(cam)
            if entry is None:
                raise KeyError(f"观测里没有相机 {cam}；可用={sorted(vision)}")
            color = entry.get("color")
            if color is None:
                raise KeyError(f"相机 {cam} 没有 color")
            out[channel] = np.ascontiguousarray(np.asarray(color, dtype=np.uint8))
        return out

    # ---- 可选观测通道（cfg.publish_* 开启才输出；默认全关）----------
    # 提取自原 forge 流程 adapter 的同名实现（逐字保留）。ReKep 类策略在
    # 环境无 depth/seg 时用 scene_objects（物体真值位姿）构建感知。

    _DEPTH_KEY_CANDIDATES = ("depth", "depth_image", "point")
    _SEG_KEY_CANDIDATES = ("seg", "segmentation", "seg_id", "id")
    _SCENE_OBJECT_TYPES = ("rigid", "geometry", "articulation")

    def to_depths(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self._extract_per_camera(obs, self._DEPTH_KEY_CANDIDATES)

    def to_segs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        found = self._extract_per_camera(obs, self._SEG_KEY_CANDIDATES)
        return {ch.replace("depth/", "seg/") if ch.startswith("depth/") else ch: v for ch, v in found.items()}

    def _extract_per_camera(self, obs: dict[str, Any], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
        vision = obs.get("vision") or {}
        out: dict[str, np.ndarray] = {}
        for channel, cam in self.cfg.camera_map.items():
            entry = vision.get(cam)
            if not isinstance(entry, dict):
                continue
            for key in keys:
                value = entry.get(key)
                if value is not None:
                    out[channel.replace("image/", "depth/")] = np.asarray(value)
                    break
        return out

    def to_scene_objects(self, obs: dict[str, Any]) -> list[dict] | None:
        if not getattr(self.cfg, "publish_scene_objects", False):
            return None
        env = self._env
        if env is None:
            return None
        lm = getattr(env, "layout_manager", None)
        if lm is None:
            sm = getattr(env, "scene_manager", None)
            lm = getattr(sm, "layout_manager", None) if sm is not None else None
        if lm is None:
            rm = getattr(env, "reward_manager", None)
            lm = getattr(rm, "layout_manager", None) if rm is not None else None
        if lm is None:
            if not getattr(self, "_scene_warned", False):
                self._scene_warned = True
                self._log("scene_objects: 未找到 layout_manager（env/scene_manager/reward_manager 均无）")
            return None
        try:
            types_by_name = dict(lm.instance_type_by_env[0])
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_scene_warned", False):
                self._scene_warned = True
                self._log(f"scene_objects: instance_type_by_env 读取失败: {exc}")
            return None
        if not types_by_name and not getattr(self, "_scene_warned", False):
            self._scene_warned = True
            self._log("scene_objects: instance_type_by_env 为空（layout 尚未加载？）")
        out: list[dict] = []
        for inst_name, inst_type in types_by_name.items():
            if inst_type not in self._SCENE_OBJECT_TYPES:
                continue
            try:
                pos, quat = lm.get_instance_pose(
                    env_idx=0, inst_name=inst_name, relative=False
                )
            except Exception as exc:  # noqa: BLE001 单个实例取不到不影响整帧
                self._log(f"scene_objects: {inst_name} 位姿获取失败: {exc}")
                continue
            if pos is None or quat is None:
                continue
            pos = np.asarray(getattr(pos, "detach", lambda: pos)().cpu() if hasattr(pos, "detach") else pos, dtype=float).reshape(-1)
            quat = np.asarray(getattr(quat, "detach", lambda: quat)().cpu() if hasattr(quat, "detach") else quat, dtype=float).reshape(-1)
            out.append(
                {
                    "name": inst_name,
                    "type": inst_type,
                    "position": [float(x) for x in pos[:3]],
                    "quat_wxyz": [float(x) for x in quat[:4]],
                }
            )
        if out and not getattr(self, "_scene_logged", False):
            self._scene_logged = True
            self._log(f"scene_objects: {len(out)} 个实例，样例 {out[0]}")
        return out

    def to_camera_info(self, obs: dict[str, Any]) -> dict | None:
        if not getattr(self.cfg, "publish_camera_info", False):
            return None
        vision = obs.get("vision") or {}
        out: dict[str, dict] = {}
        for channel, cam in self.cfg.camera_map.items():
            entry = vision.get(cam)
            if not isinstance(entry, dict):
                continue
            k = entry.get("intrinsic_matrix")
            e = entry.get("extrinsic_matrix")
            if k is None or e is None:
                continue
            key = channel.replace("image/", "depth/") if channel.startswith("image/") else channel
            out[key] = {
                "intrinsics": np.asarray(k, dtype=float).reshape(3, 3).tolist(),
                "extrinsics": np.asarray(e, dtype=float).reshape(4, 4).tolist(),
            }
        return out or None
