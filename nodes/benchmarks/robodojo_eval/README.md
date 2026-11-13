# robodojo_eval —— RoboDojo 评测发起独立节点

把 RoboDojo 评测发起从原 forge 评测流程（gateway + bench_tool 工具面 +
`_core` 批次调度 + skill 打包）解耦出来，做成**自包含、策略无关**的标准
dora 数据面节点：`eval_command` 驱动（单回合或批次），任务/布局/轮次/
**指令**全在命令里，回合结束按接口②交结果。本目录不 import `_core` /
`bench_tool`，接入哪个策略由策略侧 dataflow 决定（见下）。

## 文件

| 文件 | 职责 |
|---|---|
| `main.py` | 入口：Isaac Sim 启动（AppLauncher 必须先于一切 isaaclab import）+ 事件循环 |
| `episode_service.py` | 单回合状态机 + 批次驱动（纯逻辑，dora/Isaac 都不 import，可单测） |
| `handshake.py` | 环境契约声明 + 策略握手校验（协议层） |
| `batch.py` | 批次展开 / 参数校验 / 汇总（评测调度） |
| `robodojo_env.py` | RoboDojo 官方 env 封装（原 adapter 的全部防坑逻辑逐字保留） |
| `episode_recorder.py` | 逐相机录像（每路相机一个 mp4，反思节点按相机取关键帧方便） |
| `messages.py` | Arrow 消息载体（wire 格式对齐 forge-msgs 1.x，测试内嵌布局断言） |
| `config.py` / `config.yaml` | 环境口径（env_cfg_type/动作语义/相机映射/放弃上限）与落盘位置 |

> 本目录**策略无关**：接入哪个策略由策略侧的 dataflow 决定（各策略目录下的
> `dataflow.example.yaml`，如 `nodes/policies/fastwam/dataflow.example.yaml`）——
> 评测节点代码与配置不随策略增减而改动。dataflow 里 policy 节点段的 env
> （权重路径等）由各策略自带；dora 端口名（action/policy_command/…）是
> 协议的一部分，保持不变。

## 统一协议（与策略节点共同遵守）

回合启动前必须完成**握手**：策略节点 ready 后经 `policy_status` 自报契约
（policy_id / action_type / action_names / cameras / weight_id / …），评测端
两级校验——

1. **启动期快检**（env 未建）：对照 arx_x5 静态快照（动作空间集合 joint 14 /
   ee 16、分量名模板、相机清单）——挡明显错配，秒级拒绝；
2. **活终检**（env 已建）：回合启动时对照 `env.action_component_names()` 的
   **活推导**逐字比对——机器人换代 / env_cfg_type 变更时评测端**零代码改动**
   （静态表过期不影响最终正确性，活推导是最后防线）。不通过 → 回合 failed
   （`handshake_live_mismatch`），此时尚未发首帧，成本只有一次 reset。

两级都通过才接受 start；失败秒级拒绝（附具体哪项不匹配）。
字段表见 `policy/src/forge_policy/HANDSHAKE_FIELDS.md`（协议规范，随 SDK）。

回合边界序列：`start` → `set_instruction`(占位) → `reset_scene` →
**等 `reset_ack`**（策略确认缓存已清；超时 5s 兜底按旧协议继续）→
`set_instruction`(最终) → 发首帧。reset 确认从结构上消除"观测先于 reset 被
消费/清空"的跨通道乱序死锁；看门狗重发降级为兜底。

运行中：每帧观测带 `obs_meta {episode_id, seq}`；动作须回带同回合身份
（`action_meta`），异回合动作机械丢弃并计数；chunk 末动作 effort 哨兵补查
`is_episode_end()`（官方口径）。异常收尾三出口：策略 `failed` 上报 → 回合
立即 failed；看门狗放弃上限（give_up_after_s / max_republishes，0=不限）→
failed 带证据（等了多久/重发几次/卡在哪个 seq）；stop → 回合 stopped。

结果（`eval_result` + 落盘）：实际生效参数（布局/step_lim/seed/策略身份/
weight_id）、官方原生 `_result.json` 路径、协议诊断（握手/确认/丢弃计数）、
唯一回合目录 + 原子写入（写失败 → 结果状态标 failed，不假交付）。

**停止语义**（收尾与调度分离）：stop（编排 stop 与 dora 原生 STOP 两条路径
等价）时 ① 关闭动作入口（此后迟到动作从状态上不可消费，拒绝计数进结果）
② 置 Driver 级 `batch_stopped` 标志 ③ 只做本回合收尾——**调度被结构性封锁**
（开下一回合的唯一入口 `_maybe_advance` 检查三重门：batch 活 / status
running / 停止标志未置），stop 后第二回合不可能被拉起；下一条 start/run
重置标志与入口。**失败分级**：动作级错误（invalid_action/step_error）回合
failed 后批次照常推进；策略已死级失败（policy_failed / 看门狗放弃 /
握手活校验失败）批次立即收尾（状态 failed）不再继续。

**看门狗独立时钟**：放弃检查（give_up_after_s / max_republishes）用墙钟
判断、任何事件处理后都执行——编排方高频发 status 查询**不会**饿死放弃
检查（防消息饥饿）；重发（republish）保留在空闲超时分支（被饿死无害）。

**步数三值与 seed 分离**：`effective` 记录 `step_budget`（命令预算）、
`env_step_lim`（任务上限）、`max_steps`（实际生效 = 两者取小）——请求
2 步就真的只跑 2 步；批次的 `seed` 经 `eval_seed` 字段透传到回合（与
接口① 的布局同义 `seed` 是两个旋钮）；`layout_id` 记实际生效值（附
`requested_layout_id` 对照）。

**策略身份通用化**：config 的 `policy_id` 是占位（默认 `auto`），回合启动时
被握手自报的 policy_id 覆盖——官方 env_cfg 组装、policy_command 打标、结果
标注全部以策略自报为准，配置不预判对面是谁。

## 工具接口（describe / status）

`eval_command` 另支持两条查询命令（不推进回合，随时可发）：

- `{"command": "describe"}` → 能力快照（Query 语义）：环境契约（相机、28 维
  观测键表、**动作空间集合 joint 14 / ee 16**、判分口径）、任务清单、默认
  参数、当前握手到的策略身份。编排层据此构造 run，策略端据此核对需求；
- `{"command": "status"}` → 进度查询：当前回合（episode_id/steps/seq/reset
  确认）+ 批次进度（batch_id/完成数/成功率）+ 动作入口状态 + 策略失败信息。

应答都经 `eval_status` 通道（command 字段为 describe / status）。

## 批次（run 命令）

`eval_command` 支持两种命令：单回合 `start_episode`（接口①）与批次 `run`。
run 参数对齐 benchmark-plan 5.3：`suite / task_ids / init_state_ids /
num_runs / max_steps / seed`；`task_ids=null` 跑全部、`[]` 报错。
批次在节点内展开成回合队列，逐回合走完整单回合流程（含握手/确认/哨兵），
结束发批次级 `eval_result`（kind=batch：总 success_rate + per_task 明细）
并原子落盘 `batch_result_<batch_id>.json`。批次中途 stop：当前回合照常收尾，
不再开下一回合，汇总标 stopped；第一回合启动被拒 → 批次立即 failed 收尾。

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
编排方 ──eval_command──► robodojo_eval ──policy_command / proprio_state / image/*──► policy 节点（任意）
                                     ◄────────────── action ──────────────┘
robodojo_eval ──eval_status / eval_result──► 编排方（订阅这两个通道）
```

（对端策略节点是可替换的：任何实现 forge_policy 协议的节点都接得上——
协议契约见 `policy/src/forge_policy/HANDSHAKE_FIELDS.md`。）

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
