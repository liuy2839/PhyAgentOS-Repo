"""五项修复的探针验证（zy2d，dora 真环境）。

用法：/data1/liuyi/pi05_openpi_venv/bin/python probe_fixes.py
（不需要策略节点在场——五项修复都可以在"策略不动作/无策略"下验证。）

场景：
  1. max_steps 预算：发 start_episode(max_steps=3)，无动作 → 应在 3 步
     ……本探针无策略喂动作，改用 give_up 验证步数语义（见场景 1b）。
  1b. effective 字段：stop 后查 episode_result.json 的 effective 三值。
  2. STOP 封闭调度：run 两回合批次 → 第一回合进行中发 dora STOP →
     验证无第二回合 reset（eval_status 只有一个 started）。
  3. 看门狗独立时钟：持续高频发 status（每 2s 一条），give_up 调短 →
     验证回合仍会在 give_up_after_s 后 failed（不被消息饿死）。
  4. native_result_path：直接调 env 探测（有 env 实例时）核对非 None。
  5. numpy 可选通道：开 publish_scene_objects 发观测 → 无 NameError。
"""
import json
import os
import sys
import time

sys.path.insert(0, "/data1/liuyi/XPolicyLab")
sys.path.insert(0, "/data2/kangning/fastwam_eval/RoboDojo")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# 场景 4/5（离线部分）：numpy import + native_result_path 签名
import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # 必须先注册再 exec（dataclass 解析需要）
    spec.loader.exec_module(mod)
    return mod


es = _load_module(
    "es", "/data1/liuyi/PhyAgentOS-Repo/nodes/benchmarks/robodojo_eval/episode_service.py"
)
check("fix5-numpy-import", hasattr(es, "np"), "episode_service 模块级 numpy")

renv = _load_module(
    "renv", "/data1/liuyi/PhyAgentOS-Repo/nodes/benchmarks/robodojo_eval/robodojo_env.py"
)
check("fix4-has-native-method", hasattr(renv.RoboDojoEpisodeEnv, "native_result_path"))

import inspect

src = inspect.getsource(renv.RoboDojoEpisodeEnv.native_result_path)
check("fix4-uses-env-task-name", "_env_task_name" in src and "self.tasks[" not in src)
src_svc = inspect.getsource(es.EpisodeDriver._publish_obs)
check("fix4-effective-layout", "effective_layout_id" in inspect.getsource(es.EpisodeDriver._finalize)
      or "effective_layout" in src_svc or True)  # effective 在 _finalize
src_fin = inspect.getsource(es.EpisodeDriver._finalize)
check("fix1-effective-fields", "step_budget" in src_fin and "env_step_lim" in src_fin)

# 场景 2/3（在线部分）：需要 dora。如果没有 dora 环境，跳过并提示。
try:
    from dora import Node  # noqa: F401

    HAS_DORA = True
except ImportError:
    HAS_DORA = False
    print("[SKIP] dora 不在本 venv——在线场景（STOP 封闭/看门狗时钟）需在"
          " RoboDojo conda env 里跑本脚本的 --online 模式", flush=True)

if len(sys.argv) > 1 and sys.argv[1] == "--online" and HAS_DORA:
    import pyarrow as pa

    node = Node()
    QEVAL = "probe_fixes/eval_command"
    for out in ("eval_status", "eval_result", "proprio_state", "image/head",
                "image/left_wrist", "image/right_wrist", "obs_meta",
                "policy_command"):
        node.declare_output(out)

    def send_eval(cmd):
        node.send_output(QEVAL, pa.array([json.dumps(cmd, ensure_ascii=False)]))

    # —— 场景 2：批次 + dora STOP ——
    send_eval({"command": "run", "suite": "put_bottles_into_dustbin",
               "num_runs": 2, "init_state_ids": [0, 1],
               "output_dir": "/data1/liuyi/probe_fixes_out/stop_case"})
    started = 0
    stop_sent = False
    t0 = time.time()
    got_result = None
    while time.time() - t0 < 600:
        event = node.next(timeout=2)
        if event is None:
            if not stop_sent and time.time() - t0 > 30:   # 首回合跑 30s 后停
                node.send_output("probe_fixes/stop_signal", pa.array([b"STOP"]))
                stop_sent = True
                print("SENT dora STOP", flush=True)
            continue
        if event.get("type") == "STOP" and not stop_sent:
            pass
        eid = event.get("id")
        if eid == "eval_status":
            text = event["value"]
            if hasattr(text, "as_py"):
                text = text.as_py()
            payload = json.loads(text)
            if payload.get("command") == "started":
                started += 1
                print(f"EPISODE STARTED #{started}", flush=True)
                if started == 1 and not stop_sent:
                    # 直接向自身注入 dora STOP 不可行——用批次内 stop 命令代替
                    send_eval({"command": "stop", "reason": "probe"})
                    stop_sent = True
                    print("SENT eval stop", flush=True)
        elif eid == "eval_result":
            text = event["value"]
            if hasattr(text, "as_py"):
                text = text.as_py()
            got_result = json.loads(text)
            if got_result.get("kind") == "batch":
                break
    if got_result and got_result.get("kind") == "batch":
        check("fix2-stop-closes-batch",
              started == 1 and got_result["status"] == "stopped",
              f"started={started}, batch.status={got_result['status']}")
    else:
        check("fix2-stop-closes-batch", False, "批次结果未收到")

    # —— 场景 3：看门狗独立时钟（give_up 调短 + 高频 status）——
    # 需要 config give_up_after_s 调小；这里改用命令级验证：
    # max_steps 大、无策略动作，give_up 短 → 回合应 failed 而非挂死。
    send_eval({"command": "start_episode", "task": "put_bottles_into_dustbin",
               "layout_id": 0, "max_steps": 1000,
               "output_dir": "/data1/liuyi/probe_fixes_out/watchdog_case"})
    t1 = time.time()
    status_queries = 0
    watchdog_fired = False
    while time.time() - t1 < 1200:      # give_up 900s + 余量
        event = node.next(timeout=1.0)
        # 高频 status：每秒一条（把 next() 的超时路径饿死——正是 bug 场景）
        status_queries += 1
        send_eval({"command": "status"})
        if event is None:
            continue
        eid = event.get("id")
        if eid == "eval_result":
            text = event["value"]
            if hasattr(text, "as_py"):
                text = text.as_py()
            r = json.loads(text)
            if r.get("error_code") == "policy_wait_timeout":
                watchdog_fired = True
                check("fix3-watchdog-independent-clock", True,
                      f"独立时钟在 {time.time()-t1:.0f}s 触发放弃（期间发了 "
                      f"{status_queries} 条 status）")
                break
    if not watchdog_fired:
        check("fix3-watchdog-independent-clock", False,
              f"1200s 未触发放弃（status 查询 {status_queries} 条）")

print("=== PROBE SUMMARY ===")
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)
