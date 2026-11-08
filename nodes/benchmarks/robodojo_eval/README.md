# robodojo_eval —— RoboDojo 评测发起独立节点（FastWAM RSI 模块 A）

把 RoboDojo 评测发起从原 forge 评测流程（`forge_robodojo_eval/robodojo/` 的
gateway + bench_tool 工具面 + `_core` 批次调度 + skill 打包）解耦出来，
做成**自包含的标准 dora 数据面节点**：一次 `eval_command` = 一个回合，
任务/布局/轮次/**指令**全在命令里（`FastWAM_RSI_分工与接口_v2.md` 接口①），
回合结束按接口②交结果。本目录不 import `_core` / `bench_tool`。

## 文件

| 文件 | 职责 |
|---|---|
| `main.py` | 入口：Isaac Sim 启动（AppLauncher 必须先于一切 isaaclab import）+ 事件循环 |
| `episode_service.py` | 单回合状态机（纯逻辑，dora/Isaac 都不 import，可单测） |
| `robodojo_env.py` | RoboDojo 官方 env 封装（原 adapter 的全部防坑逻辑逐字保留） |
| `episode_recorder.py` | 逐相机录像（每路相机一个 mp4，反思节点按相机取关键帧方便） |
| `messages.py` | Arrow 消息载体（wire 格式对齐 forge-msgs 1.x，测试内嵌布局断言） |
| `config.py` / `config.yaml` | 环境口径（env_cfg_type/动作语义/相机映射）与落盘位置 |
| `dataflow.yaml` | 与 fastwam_policy 两节点的 dataflow（占位符按部署机补齐） |

## 保留不变的（线上协议与成绩口径）

- dora 通道与消息：`policy_command` / `proprio_state` / `image/*` / `action`
  （JointCommand + effort 哨兵），与原流程逐字一致（有单测对照）；
- **chunk_end 补发协议**：RoboDojo 的 `is_episode_end()` 调用次数影响成绩，
  官方每 chunk 在末动作之后、下一帧观测之前多查一次；
- **官方 run_eval 前置**：`run_reward()` + `get_score()` + 外层首次
  `is_episode_end()`（漏一句 = 零动作满分假阳性）；
- `env.success` 语义（"还没失败"≠"已完成"）、过程分 /100、step_lim 跟环境走、
  WsModelClient stub、布局 ID 语义（seed = Eval_Layout 目录 ID）。

## 通道协议

```text
编排方 ──eval_command──► robodojo_eval ──policy_command / proprio_state / image/*──► fastwam_policy
                                     ◄────────────── action ──────────────┘
robodojo_eval ──eval_status / eval_result──► 编排方（订阅这两个通道）
```

### eval_command（编排 → 本节点，JSON 文本通道）

```json
{"command": "start_episode", "task": "arrange_largest_number", "layout_id": 0,
 "round": 2, "instruction": "Arrange the numbers ... Note: place each digit centered on its pad.",
 "output_dir": "/data2/kangning/rsi/runs/arrange_r2"}
```

- `instruction` 可省：省略时用环境按场景生成的原指令（结果 JSON 的
  `env_instruction` 是环境原指令，`instruction` 是实际下发给策略的指令）。
- `layout_id` 是布局 ID；`seed` 同义（v2 接口① 两字段都出现，不一致拒绝启动）。
- `output_dir` 可省：省略时落 `<result_dir>/<task>_r<round>_layout<layout_id>/`。

`{"command": "stop", "reason": "..."}` 立即收尾当前回合（status=stopped）。

### eval_result（本节点 → 编排方，JSON 文本通道 + 落盘）

v2 接口② 最小集合 + 补充字段（同一份 JSON 落盘 `<output_dir>/episode_result.json`）：

```json
{"task": "...", "round": 2, "layout_id": 0, "success": false, "score": 0.05,
 "steps": 1050, "instruction": "（本轮实际使用的指令原文）",
 "videos": {"head": ".../videos/head.mp4", "left_wrist": "...", "right_wrist": "..."},
 "result_json": ".../episode_result.json", "log": ".../episode.log",
 "status": "succeeded", "termination": "done|timed_out|stopped|failed", "...": "..."}
```

### policy_command 序列（回合边界，与策略节点的契约）

`start` → `set_instruction`(占位) → `reset_scene` → `set_instruction`(最终)。
`start` 不能漏：策略端只在 running 相位推理（裸节点与 forge_policy SDK 皆然）。

## 启动（RoboDojo conda env；依赖 dora-rs==0.4.1 numpy pyarrow pyyaml opencv-python）

```bash
EVAL_CONFIG=/abs/path/config.yaml \
ROBODOJO_ROOT=/abs/path/RoboDojo \
PYTHONPATH=/abs/path/RoboDojo/XPolicyLab \
LD_LIBRARY_PATH=<ROBODOJO_ENV>/lib \
OMNI_KIT_ACCEPT_EULA=YES \
<ROBODOJO_ENV>/bin/python -u main.py
```

`TMPDIR` 在共享机上建议指到自己的目录（`/tmp/isaaclab` 属主冲突会两集全灭，
见原流程 launcher 的同款处理）。

## 单测

```bash
python -m pytest tests -q --basetemp=<tmp>
```

纯逻辑（状态机 / 消息 / 录像 / 配置），不需要 dora/Isaac/XPolicyLab。
`test_messages.py` 同时验证本目录载体与原流程 `adapters/messages.py` 的
wire 兼容（同数据互解）。

## 已知边界

- 不含反思（C）/编排（D）/编码（E）—— 那是宇琛的范围；编排方按上面的
  JSON 发 `eval_command` 即可。
- FastWAM_80k_forge 从未在此协议上实跑（此前只有 G05 在原流程跑通过）；
  首次上机先跑 v2 文档第 6 节的对照回合。
