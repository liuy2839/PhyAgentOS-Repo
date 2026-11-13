#!/usr/bin/env python3
"""五项修复的探针（dora dataflow 内节点版）——作为 dataflow 的"编排方"节点运行。

dataflow.yaml_probe：
    probe_orchestrator（本脚本）──eval_command──► robodojo_eval ──…──► fastwam_policy
         ▲──────────────── eval_status / eval_result ────────────┘

dora run 会给本节点注入 DORA_NODE_CONFIG，Node() 即可正常创建。
"""
from __future__ import annotations

import json
import sys
import time

from dora import Node


def main() -> int:
    node = Node()
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    def send(cmd: dict) -> None:
        import pyarrow as pa

        node.send_output("eval_command", pa.array([json.dumps(cmd, ensure_ascii=False)]))

    # ---------- 等评测节点就绪（dora 0.4.1 双向都不缓冲：双向竞态）----------
    # 被动等 node_ready 有竞态：若评测节点先上线、node_ready 发出时探针还没
    # 就绪，消息被丢，探针会傻等 900s。改为**主动轮询 describe**：周期性发
    # describe（Query），评测端每次都会回 describe 应答——只要应答到达，
    # 双向通道即证明已通，与上线先后无关。
    print("[PROBE] 轮询 describe 等评测节点通道打通 …", flush=True)
    t_ready = time.time()
    ready = False
    while time.time() - t_ready < 900:        # Isaac 冷启动窗口
        send({"command": "describe"})
        poll_t = time.time()
        while time.time() - poll_t < 5:       # 每轮 5s 窗口收应答
            event = node.next(timeout=1)
            if event is None:
                continue
            if event.get("id") == "eval_status":
                v = event["value"]
                if hasattr(v, "as_py"):
                    v = v.as_py()
                p = json.loads(v)
                if p.get("command") == "describe":
                    ready = True
                    print(f"[PROBE] 通道已通（等了 {time.time()-t_ready:.0f}s，"
                          f"轮询 {(time.time()-t_ready)//5+1} 次）", flush=True)
                    break
            if event.get("id") == "eval_status":
                v2 = event["value"]
                if hasattr(v2, "as_py"):
                    p2 = json.loads(v2.as_py())
                    if p2.get("command") == "node_ready":
                        ready = True
                        break
        if ready:
            break
    if not ready:
        check("node-ready", False, "900s 轮询 describe 无应答——评测节点没起来或通道不通")
        print("=== PROBE SUMMARY ===", flush=True)
        for name, ok, detail in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
        return 1
    check("node-ready", True)

    # ---------- 场景 1+2 合并：run 两回合批次 → 第一回合进行中 stop ----------
    # （需要 fastwam_policy 也在场才会真跑；无策略时回合会卡在等动作——
    #   stop 路径验证不依赖策略：只要看到"第二个 started 没有出现"即可。）
    print("[PROBE] 场景 2：批次 + stop → 验证 STOP 后第二回合不被拉起", flush=True)
    send({
        "command": "run", "suite": "put_bottles_into_dustbin",
        "num_runs": 2, "init_state_ids": [0, 1],
        "output_dir": "/data1/liuyi/probe_fixes_out/stop_case",
    })
    started_count = 0
    stop_sent = False
    batch_result = None
    t0 = time.time()
    while time.time() - t0 < 300 and batch_result is None:
        event = node.next(timeout=2)
        if event is None:
            # 首回合 started 后 20s 还没动作 → 发 stop（等动作挂死正是常态）
            if not stop_sent and started_count >= 1 and time.time() - t0 > 25:
                print("[PROBE] 发送 stop", flush=True)
                send({"command": "stop", "reason": "probe-stop-case"})
                stop_sent = True
            continue
        eid = event.get("id")
        if eid == "eval_status":
            v = event["value"]
            if hasattr(v, "as_py"):
                v = v.as_py()
            p = json.loads(v)
            if p.get("command") == "started":
                started_count += 1
                print(f"[PROBE] started #{started_count}", flush=True)
        elif eid == "eval_result":
            v = event["value"]
            if hasattr(v, "as_py"):
                v = v.as_py()
            p = json.loads(v)
            if p.get("kind") == "batch":
                batch_result = p

    if batch_result is None:
        check("fix2-stop-closes-batch", False, "300s 内未收到批次结果")
    else:
        check(
            "fix2-stop-closes-batch",
            started_count == 1 and batch_result.get("status") == "stopped",
            f"started 次数={started_count}（修复前会是 2），batch.status="
            f"{batch_result.get('status')}",
        )

    # ---------- 场景 3：看门狗独立时钟 ----------
    # give_up_after_s 需调小才能快速验证；若评测端 config 是 900s，本场景
    # 改为只验证"持续 status 查询下回合仍有终局"（事件循环没被饿死）。
    print("[PROBE] 场景 3：高频 status 查询 + 等待 → 验证事件循环存活与终局", flush=True)
    send({
        "command": "start_episode", "task": "put_bottles_into_dustbin",
        "layout_id": 0,
        "output_dir": "/data1/liuyi/probe_fixes_out/watchdog_case",
    })
    t1 = time.time()
    queries = 0
    episode_result = None
    while time.time() - t1 < 120 and episode_result is None:
        event = node.next(timeout=1)
        queries += 1
        if queries % 2 == 0:          # 高频 status：每个循环发一条查询
            send({"command": "status"})
            # 立刻可能来应答；next 上一行已经消费了一个事件，下轮继续
        if event is None:
            continue
        eid = event.get("id")
        if eid == "eval_status":
            v = event["value"]
            if hasattr(v, "as_py"):
                v = v.as_py()
            p = json.loads(v)
            if p.get("command") == "status":
                ep = p.get("episode")
                if ep:
                    print(f"[PROBE] status: steps={ep['steps']} seq={ep['seq']} "
                          f"awaiting={ep['awaiting_action']}", flush=True)
        elif eid == "eval_result":
            v = event["value"]
            if hasattr(v, "as_py"):
                v = v.as_py()
            p = json.loads(v)
            if p.get("kind") != "batch":
                episode_result = p

    if episode_result is None:
        # 120s 无终局 = 事件循环被 status 饿死 or 看门狗未触发——修复前会这样
        check("fix3-watchdog-independent-clock", False,
              f"120s 无终局（status 查询 {queries} 条）——事件循环可能被饿死")
    else:
        check("fix3-watchdog-independent-clock", True,
              f"有终局 termination={episode_result.get('termination')} "
              f"error={episode_result.get('error_code')}（查询 {queries} 条）")

    # ---------- 场景 1b：effective 字段 ----------
    if episode_result and episode_result.get("effective"):
        eff = episode_result["effective"]
        check("fix1-effective-fields",
              "step_budget" in eff and "env_step_lim" in eff and "max_steps" in eff,
              json.dumps({k: eff.get(k) for k in ("step_budget", "env_step_lim", "max_steps")},
                         ensure_ascii=False))
    else:
        check("fix1-effective-fields", False, "无回合结果可查")

    print("=== PROBE SUMMARY ===", flush=True)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
