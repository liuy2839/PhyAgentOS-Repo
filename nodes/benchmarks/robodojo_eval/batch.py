"""批次层（评测调度）—— run Session 的批次展开、回合排队与结果汇总。

对应《修改方案》内部模块表的"评测调度：参数校验、批次展开、回合管理、结果汇总"。
微信确认：批次展开**在 robodojo 节点内实现**（读法 A）——节点直接对外提供
run 工具语义，编排层一次命令可以跑一个批次。

与 benchmark-plan v0.2 第 5.3 节的统一工具参数对齐：
    suite / task_ids / init_state_ids（=布局 ID 列表）/ num_runs / max_steps /
    seed / options
约定沿用 plan：`task_ids=null` 表示全部，`[]` 报错。

批次与回合的关系：一条 run 命令展开成 N 个回合（task × layout × num_runs），
逐回合走 EpisodeDriver 的单回合流程（握手协议、身份标签、原子落盘全部复用）；
批次结束时汇总 per_task 成功率与总 success_rate，写 batch_result.json（原子写）。

纯逻辑模块：不 import dora / Isaac；env_factory 的任务清单解析由注入的
`task_inventory` 回调提供（有 Isaac 时来自 task_registry，测试里来自假清单）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class BatchValidationError(ValueError):
    """批次参数校验失败（run 命令被拒的理由）。"""


@dataclass
class BatchPlan:
    """校验后的批次计划。"""

    suite: str
    tasks: list[str]
    layout_ids: list[int]
    num_runs: int
    max_steps: int | None
    seed: int | None
    output_root: str
    # 展开后的回合队列：[(task, layout_id, run_index)]
    episodes: list[tuple[str, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.episodes = [
            (task, layout_id, run_index)
            for task in self.tasks
            for layout_id in self.layout_ids
            for run_index in range(self.num_runs)
        ]

    @property
    def total(self) -> int:
        return len(self.episodes)


def resolve_task_ids(spec: Any, all_tasks: list[str]) -> list[str]:
    """解析 task_ids：null/'all' → 全部；[] → 报错（plan 约定）；列表 → 成员校验。"""
    if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
        return list(all_tasks)
    if isinstance(spec, list):
        if not spec:
            raise BatchValidationError("task_ids=[] 不合法：要跑全部请传 null 或省略")
        unknown = [t for t in spec if t not in all_tasks]
        if unknown:
            raise BatchValidationError(
                f"task_ids 含未知任务：{unknown}；该 suite 可用：{all_tasks}"
            )
        return [str(t) for t in spec]
    raise BatchValidationError(f"task_ids 必须是 null、'all' 或列表，收到 {spec!r}")


def resolve_layout_ids(spec: Any, num_per_task: int) -> list[int]:
    """解析 init_state_ids（布局 ID）：null → range(num)；[] → 报错；'0-3'/'0,2' → 展开。"""
    if spec is None:
        return list(range(num_per_task))
    if isinstance(spec, list):
        if not spec:
            raise BatchValidationError("init_state_ids=[] 不合法：要跑全部请传 null 或省略")
        return [int(x) for x in spec]
    if isinstance(spec, str):
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        if not out:
            raise BatchValidationError("init_state_ids 解析后为空")
        return out
    if isinstance(spec, int):
        return [spec]
    raise BatchValidationError(f"init_state_ids 无法解析：{spec!r}")


def build_batch_plan(
    cmd: dict[str, Any],
    *,
    all_tasks: list[str],
    num_layouts_per_task: int,
    default_output_root: str,
    default_num_runs: int = 1,
) -> BatchPlan:
    """校验批次命令参数并展开成回合队列。参数错误抛 BatchValidationError。"""
    suite = str(cmd.get("suite") or "").strip()
    if not suite:
        raise BatchValidationError("run 命令缺少 suite")

    # suite 也可以就是单个任务名（单任务场景不必先查清单）
    known = list(all_tasks)
    if suite not in known and suite.lower() != "all":
        # suite 形如任务集合名时任务清单来自 registry；无法识别则按单任务处理
        known = known + [suite]

    tasks = resolve_task_ids(cmd.get("task_ids"), known)
    if suite.lower() != "all" and suite in all_tasks:
        tasks = [suite]          # suite 指定单任务时忽略 task_ids 里的其它成员
    if not tasks:
        raise BatchValidationError("批次为空：tasks 解析后没有任何元素")

    num_runs = int(cmd.get("num_runs", default_num_runs))
    if num_runs < 1:
        raise BatchValidationError(f"num_runs 必须 ≥1，收到 {num_runs}")

    layout_ids = resolve_layout_ids(cmd.get("init_state_ids"), num_layouts_per_task)

    max_steps = cmd.get("max_steps")
    if max_steps is not None and int(max_steps) < 1:
        raise BatchValidationError(f"max_steps 必须 ≥1，收到 {max_steps}")

    seed = cmd.get("seed")
    output_root = os.path.abspath(
        str(cmd.get("output_dir") or os.path.join(default_output_root, suite))
    )

    return BatchPlan(
        suite=suite,
        tasks=tasks,
        layout_ids=layout_ids,
        num_runs=num_runs,
        max_steps=int(max_steps) if max_steps is not None else None,
        seed=int(seed) if seed is not None else None,
        output_root=output_root,
    )


# ---------------------------------------------------------------- 批次执行状态


@dataclass
class BatchRunState:
    """一次批次的执行状态与结果汇总。"""

    plan: BatchPlan
    batch_id: str
    started_at: float = field(default_factory=time.time)
    next_index: int = 0                # 队列游标
    finished: bool = False
    status: str = "running"            # running / succeeded / stopped / failed
    # 每回合的摘要：{"task","layout_id","run","episode_id","status","success","score","result_json"}
    episode_summaries: list[dict[str, Any]] = field(default_factory=list)

    def current(self) -> tuple[str, int, int] | None:
        if self.next_index >= self.plan.total:
            return None
        return self.plan.episodes[self.next_index]

    def record_episode(self, summary: dict[str, Any]) -> None:
        self.episode_summaries.append(summary)
        self.next_index += 1

    # ---- 汇总（benchmark-plan 5.4 的字段子集：per_task / success_rate）----

    def summarize(self) -> dict[str, Any]:
        done = self.episode_summaries
        total = self.plan.total
        successes = sum(1 for s in done if s.get("success"))
        per_task: dict[str, dict[str, Any]] = {}
        for s in done:
            t = s["task"]
            slot = per_task.setdefault(
                t, {"total": 0, "successes": 0, "episodes": []}
            )
            slot["total"] += 1
            slot["successes"] += 1 if s.get("success") else 0
            slot["episodes"].append({
                "layout_id": s.get("layout_id"),
                "run": s.get("run"),
                "episode_id": s.get("episode_id"),
                "success": s.get("success"),
                "score": s.get("score"),
                "result_json": s.get("result_json"),
            })
        for slot in per_task.values():
            slot["success_rate"] = slot["successes"] / slot["total"] if slot["total"] else 0.0
        return {
            "batch_id": self.batch_id,
            "suite": self.plan.suite,
            "status": self.status,
            "total_episodes": total,
            "completed_episodes": len(done),
            "success_rate": successes / total if total else 0.0,
            "per_task": per_task,
            "num_runs": self.plan.num_runs,
            "layout_ids": self.plan.layout_ids,
            "max_steps": self.plan.max_steps,
            "seed": self.plan.seed,
            "elapsed_s": time.time() - self.started_at,
            "episodes": done,
        }


def write_batch_result(state: BatchRunState, path: str) -> str | None:
    """批次结果原子落盘。写失败返回 None（调用方决定如何上报）。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state.summarize(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path
    except Exception:  # noqa: BLE001
        return None


def episode_command_from_plan(
    plan: BatchPlan, task: str, layout_id: int, run_index: int, batch_id: str
) -> dict[str, Any]:
    """把批次里的一个回合转成 EpisodeDriver 认的单回合命令（接口① 形状）。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return {
        "command": "start_episode",
        "task": task,
        "layout_id": layout_id,
        "round": run_index + 1,
        "output_dir": os.path.join(
            plan.output_root, f"{batch_id}", f"{task}_L{layout_id}_r{run_index + 1}"
        ),
        # 批次参数透传（《问题.md》设计问题 1：seed 此前在计划→命令的传递中
        # 丢失）。注意语义分离：`eval_seed` 是随机化种子（benchmark-plan 的
        # seed，进 env_cfg），与接口① 的 `seed` 字段（布局 ID 同义）是两个
        # 旋钮——批次透传用 eval_seed，避免被当成布局触发不一致拒绝。
        **({"max_steps": plan.max_steps} if plan.max_steps else {}),
        **({"eval_seed": plan.seed} if plan.seed is not None else {}),
    }
