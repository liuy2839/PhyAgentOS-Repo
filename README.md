# RoboDojo 评测独立节点

本仓库发布 **RoboDojo 操作策略评测的两个独立标准 dora 节点**：
评测发起节点（robodojo_eval）+ 策略节点（fastwam_policy / rekep_policy /
pi05_policy），以及它们共用的策略侧 SDK（forge_policy）。

- 评测环境：RoboDojo（Isaac Sim 仿真，ARX X5 双臂）
- 执行框架：dora 数据流（数据平面节点，观测→动作 1:1 锁步）
- 驱动方式：编排层经 `eval_command` JSON 通道驱动——单回合（接口①）或
  批次（run 命令，节点内展开汇总）；接口见 `FastWAM_RSI_分工与接口_v2.md`
- Skill profile（策略×环境的组合配置单）归控制层仓库维护，不在本仓库

> 历史注：本仓库此前包含完整 PAOS 评测流程（gateway + bench_tool 工具面 +
> `_core` 批次调度 + Skill Bundle 打包）。为避免与 agent 层重构产生绑定，
> 该流程已整体移除，仓库只保留两个独立节点作为发布主体；原防坑逻辑与成绩
> 口径在 robodojo_eval 内逐字保留。

## 整体架构

```text
编排方 ──eval_command──► robodojo_eval (RoboDojo conda env, Isaac Sim)
              │  JSON 文本通道：start_episode / stop，一次命令 = 一个回合
              ▼
robodojo_eval ──policy_command / proprio_state / image/* / depth/* / scene_objects──► policy 节点
policy 节点 ──action (JointCommand + chunk_end 哨兵)──► robodojo_eval
robodojo_eval ──eval_status / eval_result──► 编排方订阅（接口②）
```

**为什么是两个独立节点**：评测发起（Isaac Sim 侧）与策略执行（策略依赖侧）
必须分属不同 Python 环境（isaacsim 钉 py3.10、策略侧各自独立），dora 节点
是天然的进程边界；节点间只靠 wire 协议（forge-msgs 1.x 兼容的 Arrow 消息）
通信，编排层怎么重构都不影响节点本身。

## 仓库结构

```
nodes/benchmarks/robodojo_eval/    评测发起独立节点（模块 A，自包含）
├── main.py                        入口（Isaac Sim 启动 + 事件循环）
├── episode_service.py             单回合状态机 + 批次驱动（纯逻辑可单测）
├── handshake.py                   环境契约声明 + 策略握手校验（协议层）
├── batch.py                       批次展开/参数校验/汇总（评测调度）
├── robodojo_env.py                RoboDojo 官方 env 封装（防坑口径逐字保留）
├── episode_recorder.py            逐相机录像（每路相机一个 mp4）
├── messages.py / config.py / config.yaml
└── tests/                         状态机/协议/批次/消息 wire/录像单测
nodes/policies/fastwam/            FastWAM 策略独立节点（模块 B）
├── src/fastwam_policy/policy.py   四方法 + XPolicyLab 加载器（env POLICY_NAME 切模型）
└── tests/                         chunk 节奏 / 哨兵时机 / 观测还原单测
nodes/policies/rekep/              ReKep 策略节点（每帧制示例：chunk_per_action）
nodes/policies/pi05/               π0.5 策略节点（WS 客户端 + 本地服务器架构，
│                                  pro5000 冒烟通过：50 步 chunk / 首帧 14.1s）
policy/                            策略侧 SDK：forge_policy
├── src/forge_policy/base.py       Policy 四方法抽象（reset_episode/set_instruction/step/shutdown）
├── src/forge_policy/contract.py   策略线协议（PolicyContract：观测 topic、动作分量、哨兵）
├── src/forge_policy/runtime.py    PolicyRuntime：帧聚合、命令分发、动作发送、状态上报
├── src/forge_policy/HANDSHAKE_FIELDS.md 握手契约字段表（协议规范，随 SDK）
└── tests/                         协议与生命周期单测
scripts/deploy_rekep_policy.sh     策略侧部署（rsync + venv + 冒烟）
```

## 接口（eval_command / eval_result）

一次 `eval_command` = 一个完整回合（接口①，JSON 文本通道）：

```json
{"command": "start_episode", "task": "arrange_largest_number", "layout_id": 0,
 "round": 2, "instruction": "Arrange the numbers ... Note: place each digit centered on its pad.",
 "output_dir": "/data2/kangning/rsi/runs/arrange_r2"}
```

- `instruction` 可省：省略时用环境按场景生成的原指令（结果里 `env_instruction`
  是环境原指令，`instruction` 是实际下发的指令）。
- `layout_id` 是 RoboDojo 布局 ID（`Assets/Eval_Layout/.../<seed>/` 目录 ID）；
  `seed` 同义（两字段都给且不一致会拒绝启动）。
- `{"command": "stop"}` 立即收尾当前回合。

回合结束按接口②发 `eval_result`（同一份 JSON 落盘 `<output_dir>/episode_result.json`）：

```json
{"task": "...", "round": 2, "layout_id": 0, "success": false, "score": 0.05,
 "steps": 1050, "instruction": "（本轮实际使用的指令原文）",
 "videos": {"head": ".../videos/head.mp4", "left_wrist": "...", "right_wrist": "..."},
 "result_json": ".../episode_result.json", "log": ".../episode.log",
 "status": "succeeded", "termination": "done|timed_out|stopped|failed", "...": "..."}
```

回合边界的 policy_command 序列：`start` → `set_instruction`(占位) →
`reset_scene` → **等策略 reset_ack**（policy_status 通道，超时 5s 兜底）→
`set_instruction`(最终) → 首帧（带 obs_meta：episode_id + seq；动作须回带
同回合身份）。**握手两级校验**：启动期快检（静态快照挡明显错配）+ env 建好
后的**活终检**（对照 `env.action_component_names()` 活推导逐字比对，机器人
换代零代码改动）。运行中协议：策略 failed 上报 → 回合立即 failed；看门狗
放弃检查用**独立时钟**（give_up_after_s / max_republishes，高频消息不会饿
死放弃判定）→ failed 带证据；stop **关闭动作入口并置停止标志**（收尾与
调度分离——STOP 后下一回合结构性不可拉起）；结果含实际生效参数（步数
三值 step_budget/env_step_lim/max_steps、布局实际值+请求值）与官方原生
_result.json 路径，原子写入唯一回合目录。
**策略身份通用化**：配置的 `policy_id` 是占位（默认 `auto`），启动时被握手
自报覆盖——官方组装、打标、结果标注以策略自报为准。
dataflow 里 `policy_command` 必须 `queue_size: 32`（forge_policy contract
最低要求）。

### 查询（describe / status）

`eval_command` 另支持 `{"command": "describe"}`（能力快照：环境契约、任务
清单、动作空间集合、当前策略身份）与 `{"command": "status"}`（回合/批次
进度、动作入口状态），应答经 `eval_status`。

### 批次（run 命令，节点内展开汇总）

```json
{"command": "run", "suite": "arrange_largest_number", "task_ids": null,
 "init_state_ids": [0, 1], "num_runs": 2, "max_steps": null, "seed": 0}
```

节点内展开成回合队列逐回合执行，结束发 `eval_result`（kind=batch：总
success_rate + per_task 明细）并落盘 `batch_result_<batch_id>.json`。
`task_ids=null` 跑全部、`[]` 报错（benchmark-plan 约定）。

## 启动

两节点**必须用不同的 python env**（isaacsim 5.1 钉 py3.10，策略侧独立 env）。
依赖（**版本基线**——Isaac Sim 5.1 钉 numpy 1.x，pyarrow≥25 要 numpy≥2，
混装会让评测节点 import 直接炸，上机实测教训）：`dora-rs==0.4.1`、
`numpy>=1.24,<2.0`、`pyarrow>=14,<25`、`pyyaml>=6.0`（评测侧另需
`opencv-python` 录像；版本约束已写入 `nodes/benchmarks/robodojo_eval/
pyproject.toml`）。

评测节点（RoboDojo conda env；`TMPDIR` 共享机上指到自己的目录）：

```bash
EVAL_CONFIG=/abs/path/config.yaml \
ROBODOJO_ROOT=/abs/path/RoboDojo \
PYTHONPATH=/abs/path/RoboDojo/XPolicyLab \
LD_LIBRARY_PATH=<ROBODOJO_ENV>/lib \
OMNI_KIT_ACCEPT_EULA=YES \
<ROBODOJO_ENV>/bin/python -u nodes/benchmarks/robodojo_eval/main.py
```

策略节点（FastWAM/XPOLICYLAB env；权重路径在其 deploy.yml，dora 依赖见
`/data2/kangning/rsi/README.md` 第 2 节）：

```bash
PYTHONPATH=<REPO>/policy/src:<REPO>/nodes/policies/fastwam/src \
XPOLICYLAB_ROOT=/data2/kangning/fastwam_eval/RoboDojo/XPolicyLab \
POLICY_NAME=FastWAM_80k_forge \
ENV_CFG_TYPE=arx_x5 ACTION_TYPE=joint \
<FASTWAM_ENV>/bin/python -u -m fastwam_policy.main
```

ReKep 策略的对应 dataflow：`nodes/policies/rekep/dataflow.example.yaml`
（需评测侧 config 开 `publish_scene_objects: true`）。

## 新策略接入

实现 `forge_policy.Policy` 四方法 + 一份契约（与 ReKep / FastWAM 同构）：

```python
from forge_policy import Policy, PolicyContract, PolicyRuntime

class MyPolicy(Policy):
    def reset_episode(self): ...          # episode 边界：清缓存 / 重置内部状态
    def set_instruction(self, text): ...  # 接收任务指令
    def step(self, obs): ...              # 见下方三种返回形态
    def shutdown(self): ...

contract = PolicyContract(
    action_names=[...],          # 与 robodojo_env.action_component_names() 逐字一致（ARX X5 joint = 14 维）
    cameras=["image/head", "image/left_wrist", "image/right_wrist"],
    emit_chunk_end=True,         # RoboDojo 必须开
    # chunk_per_action=True,     # 每帧制策略（每帧重新决策 = N=1 chunk）加这行
)
PolicyRuntime(MyPolicy(), contract).run()
```

`step()` 的返回形态（`emit_chunk_end=True` 时推荐元组形式，哨兵时机由接口
保证，与 RoboDojo 官方「chunk 末动作之后、下一帧观测之前补查」的序列对齐）：

```python
def step(self, obs):
    ...
    return None                    # 本帧不动作（如规划中）
    return vec                     # 裸向量：非 chunk 末（兼容旧用法）
    return vec, True               # (向量, is_last)：runtime 自动给末动作附哨兵
```

策略代码放进 `nodes/policies/<你的策略>/`，准备独立 Python 环境（与评测侧
分开），参照 `nodes/policies/fastwam/` 与 `nodes/policies/rekep/`。

硬约束与常见坑（各节点源码内有展开注释）：观测驱动（tick 只当心跳）、
`policy_command` 通道 queue_size ≥ 32、动作分量名逐字匹配、图像必须 RGB、
动作维度走 `get_robot_action_dim_info(env_cfg_type)`、chunk 末动作 effort 填
`CHUNK_END_SENTINEL`。

## 工程注意事项

1. **dora 节点 env 是替换语义**：dataflow/launcher 必须自建完整环境
   （LD_LIBRARY_PATH / PYTHONPATH / CUDA），不能依赖父进程继承。
2. **`TMPDIR`**：isaaclab 的日志/缓存走 `tempfile.gettempdir()`；共享机
   `/tmp/isaaclab` 被他人占用会令 env 构造 `PermissionError`。
3. **`sys.path` 顺序**：RoboDojo 仓库常被 clone 成 `robodojo` 与上游同名；
   只把 RoboDojo 仓库根挂上 sys.path，父目录混入会得到空命名空间包
   （症状是 "Task not found" 而非 ImportError）。
4. **启动顺序**：Isaac Sim 的 `AppLauncher` 必须在 import 任何 isaaclab/omni
   模块之前跑完（robodojo_eval/main.py 的 import 分段不可重排）。
5. **chunk_end 哨兵**：RoboDojo 的 `is_episode_end()` 调用次数影响成绩；
   策略在 chunk 末动作返回前调 `mark_chunk_end()`，评测端在末动作之后补查
   一次，与官方 rollout 时机一致。

## 测试

```bash
python -m pytest nodes/benchmarks/robodojo_eval/tests -q --basetemp=<tmp>
python -m pytest nodes/policies/fastwam/tests -q --basetemp=<tmp>
python -m pytest nodes/policies/rekep/tests policy/tests -q --basetemp=<tmp>
```

全部为纯逻辑单测（假 node / 假 env / 消息 round-trip），不需要 dora / Isaac /
XPolicyLab / torch。合跑加 `--import-mode=importlib`。

## Node、Endpoint 与 Tool API

| 项目 | 当前实现 |
| --- | --- |
| 生产 Node | 4 个：`robodojo_eval`（评测发起，Isaac Sim 数据面）、`fastwam_policy`、`rekep_policy`、`pi05_policy`（策略执行）；`forge_policy` 是 SDK 库、`probe_node.py` 是探针工具，均不计入生产 Node |
| Tool 支持 | 数据平面 Node：编排层经 dora 通道（`eval_command` → `eval_status` / `eval_result`）直驱；无 HTTP Tool API、无 Gateway、无 Endpoint |
| Endpoint 数 | 0（数据平面节点，不注册 Forge Endpoint；见问题.md"控制面"层——可选的 Forge Endpoint 适配层属后续规划） |
| operation 数 | 0（逻辑操作经 `eval_command` 的 command 字段承载：`start_episode` / `run` / `describe`（Query）/ `status` / `stop`） |
| 直接 HTTP Tool API | 0（不监听 HTTP；调用方经 dora 通道） |
| Gateway Tool 数 | 0（编排层经 dora 通道直驱） |

**调用映射**：`eval_command`（JSON）→ robodojo_eval 内部 `EpisodeDriver`；命令
语义见上文"接口"节（单回合 `start_episode` / 批次 `run` / 查询 `describe`、
`status` / `stop`）。**协议契约**：消息 wire 格式对齐 forge-msgs 1.x（Arrow），
字段规范见 `policy/src/forge_policy/HANDSHAKE_FIELDS.md`。

**兼容性与构建**：`dora-rs==0.4.1`（版本基线）；`numpy>=1.24,<2.0`、
`pyarrow>=14,<25`（Isaac Sim 5.1 钉 numpy 1.x，pyarrow≥25 需 numpy≥2——混装
会让评测节点 import 直接炸，上机实测教训）；Python ≥3.10（评测侧）/ ≥3.11
（策略侧）。构建：源码直接运行（`python main.py` / `python -m fastwam_policy.main`），
无 PyInstaller 制品；依赖基线已写入
`nodes/benchmarks/robodojo_eval/pyproject.toml` 与各策略节点 pyproject。

## 已知边界

- FastWAM_80k_forge 从未在此协议实跑（此前只有 G05 在旧链路跑通过真实回合）；
  首次上机先跑同布局的对照回合（原指令 vs 原指令+经验，见分工文档第 6 节步 1）。
- ReKep 在 `arx_x5`（双臂四瓶交接）上只驱动左臂，分数不代表其真实能力
  （详见其 README 的实测记录）。
