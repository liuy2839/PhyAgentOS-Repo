"""批次层（batch.py）单测：参数校验、队列展开、汇总、原子落盘。纯逻辑，无 dora/Isaac。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "robodojo_eval_batch", HERE.parent / "batch.py"
)
batch = importlib.util.module_from_spec(spec)
sys.modules["robodojo_eval_batch"] = batch
spec.loader.exec_module(batch)


ALL_TASKS = ["arrange_largest_number", "classify_objects", "classify_objects_by_language"]


def _plan(**overrides):
    cmd = {"suite": "arrange_largest_number", "num_runs": 2}
    cmd.update(overrides)
    return batch.build_batch_plan(
        cmd, all_tasks=ALL_TASKS, num_layouts_per_task=3,
        default_output_root="/tmp/out",
    )


def test_single_task_expands_cartesian():
    plan = _plan()
    assert plan.tasks == ["arrange_largest_number"]
    assert plan.layout_ids == [0, 1, 2]
    assert plan.total == 3 * 2
    # 队列顺序：task → layout → run
    assert plan.episodes[0] == ("arrange_largest_number", 0, 0)
    assert plan.episodes[1] == ("arrange_largest_number", 0, 1)
    assert plan.episodes[2] == ("arrange_largest_number", 1, 0)


def test_multi_task_via_task_ids():
    plan = _plan(suite="all", task_ids=["classify_objects", "arrange_largest_number"],
                 num_runs=1, init_state_ids=[0, 5])
    assert plan.tasks == ["classify_objects", "arrange_largest_number"]
    assert plan.total == 2 * 2 * 1


def test_null_means_all_and_empty_list_rejected():
    plan = _plan(suite="all", task_ids=None, init_state_ids=None, num_runs=1)
    assert plan.tasks == ALL_TASKS and plan.layout_ids == [0, 1, 2]
    with pytest.raises(batch.BatchValidationError, match="要跑全部"):
        _plan(task_ids=[])
    with pytest.raises(batch.BatchValidationError, match="要跑全部"):
        _plan(init_state_ids=[])


def test_unknown_task_rejected():
    with pytest.raises(batch.BatchValidationError, match="未知任务"):
        _plan(suite="all", task_ids=["no_such_task"])


def test_layout_range_string():
    plan = _plan(init_state_ids="0-2,7", num_runs=1)
    assert plan.layout_ids == [0, 1, 2, 7]


def test_invalid_num_runs_rejected():
    with pytest.raises(batch.BatchValidationError, match="num_runs"):
        _plan(num_runs=0)


# ---------------------------------------------------------------- 汇总


def _mk_state(total_layouts=2):
    plan = _plan(num_runs=1, init_state_ids=list(range(total_layouts)))
    state = batch.BatchRunState(plan=plan, batch_id="b1")
    return state


def test_summarize_per_task_and_rate():
    state = _mk_state()
    for layout, ok in [(0, True), (1, False)]:
        state.record_episode({
            "task": "arrange_largest_number", "layout_id": layout, "run": 1,
            "episode_id": f"ep{layout}", "status": "succeeded",
            "success": ok, "score": 1.0 if ok else 0.05, "result_json": f"/r/{layout}.json",
        })
    s = state.summarize()
    assert s["total_episodes"] == 2 and s["completed_episodes"] == 2
    assert s["success_rate"] == 0.5
    slot = s["per_task"]["arrange_largest_number"]
    assert slot["total"] == 2 and slot["successes"] == 1 and slot["success_rate"] == 0.5
    assert slot["episodes"][0]["score"] == 1.0


def test_summarize_multi_task_isolated():
    plan = _plan(suite="all", task_ids=["classify_objects", "arrange_largest_number"],
                 num_runs=1, init_state_ids=[0])
    state = batch.BatchRunState(plan=plan, batch_id="b2")
    state.record_episode({"task": "classify_objects", "layout_id": 0, "run": 1,
                          "episode_id": "e1", "status": "succeeded", "success": True,
                          "score": 1.0, "result_json": ""})
    state.record_episode({"task": "arrange_largest_number", "layout_id": 0, "run": 1,
                          "episode_id": "e2", "status": "succeeded", "success": False,
                          "score": 0.0, "result_json": ""})
    s = state.summarize()
    assert s["per_task"]["classify_objects"]["success_rate"] == 1.0
    assert s["per_task"]["arrange_largest_number"]["success_rate"] == 0.0
    assert s["success_rate"] == 0.5


def test_atomic_write(tmp_path):
    state = _mk_state(1)
    state.record_episode({"task": "arrange_largest_number", "layout_id": 0, "run": 1,
                          "episode_id": "e1", "status": "succeeded", "success": True,
                          "score": 1.0, "result_json": ""})
    path = str(tmp_path / "batch_result.json")
    out = batch.write_batch_result(state, path)
    assert out == path
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["batch_id"] == "b1" and data["success_rate"] == 1.0
    # 没有残留 .tmp
    assert not os.path.exists(path + ".tmp")


def test_episode_command_shape(tmp_path):
    plan = _plan(num_runs=1, init_state_ids=[3])
    cmd = batch.episode_command_from_plan(plan, plan.tasks[0], 3, 0, "b9")
    assert cmd["command"] == "start_episode"
    assert cmd["task"] == "arrange_largest_number"
    assert cmd["layout_id"] == 3
    assert cmd["round"] == 1
    assert "b9" in cmd["output_dir"]
    assert os.path.isabs(cmd["output_dir"])
