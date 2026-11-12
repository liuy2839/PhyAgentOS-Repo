# fastwam_policy —— FastWAM 策略独立节点（FastWAM RSI 模块 B）

加载 `XPolicyLab/policy/<POLICY_NAME>/model.Model`（FastWAM_80k_forge 与 G05
同一入口），按本仓库的 `forge_policy` 统一策略协议（`policy/src/forge_policy/`，
与 ReKep 节点同构）经 dora 执行：算法实现 `Policy` 四方法，收发由
`PolicyRuntime` 承担。从原 forge 流程的裸 `policy_node.py` 解耦而来，官方
rollout 节奏逐字保留。

## 结构

```
src/fastwam_policy/
├── policy.py      FastWAMPolicy（Policy 四方法）+ XPolicyLab 加载器 + main
└── main.py        入口（python -m fastwam_policy.main）
```

## 与官方 rollout 的对应

```
model.reset()                    → reset_episode()（双 reset，官方每集两次）
model.update_obs(obs)            → step() 每帧无条件喂（坑 2：历史帧策略会静默劣化）
model.get_action()               → step() 仅在 chunk 用尽后调用，一次一个 chunk
chunk 末动作打标记               → step() 末动作后 mark_chunk_end()，哨兵随
                                   **本 chunk 末动作**发出（见下）
```

**chunk_end 哨兵时机**（RoboDojo 成绩口径的关键）：RoboDojo 的
`is_episode_end()` 有副作用，官方每 chunk 在末动作之后、下一帧观测之前多查
一次。本节点 `step()` 以元组 `return vec, is_last` 声明 chunk 边界——
`PolicyRuntime` 自动给 `is_last=True` 的动作附 effort 哨兵，评测端补查时机
与官方一致。**哨兵位置由接口保证**（元组跟着动作走），不再依赖手工置位的
调用时机；SDK 的 `mark_chunk_end()` 已标 deprecated（保留兼容，勿在新代码
使用——它在"step 返回后调用"的直觉用法下会把哨兵错位到下一 chunk 首动作）。

## 观测还原

XPolicyLab 的标准观测里，夹爪（`ee_joint_state`）是 Python list、其余是
float32 ndarray。传输层统一 float64，`_to_model_obs` 按官方类型拼回去
（类型错了会静默走进位姿还原的数学里）。图像已是 RGB，不做任何通道转换。

## 环境变量

| 变量 | 说明 |
|---|---|
| `XPOLICYLAB_ROOT` | XPolicyLab 检出目录（必填） |
| `POLICY_NAME` | `FastWAM_80k_forge`（默认 G05） |
| `ENV_CFG_TYPE` / `ACTION_TYPE` | `arx_x5` / `joint`（必须与评测节点一致） |
| `CKPT_NAME` | 覆盖 deploy.yml 的 ckpt_name（可选） |
| `G05_PROCESSOR_PATH` | 仅 G05 必填 |

FastWAM 权重路径已写进其 deploy.yml（`fastwam_eval/RoboDojo/XPolicyLab/
policy/FastWAM_80k_forge/`）；dora 依赖在 `/data2/kangning/rsi/pydeps_forge/`，
加载方式见 `/data2/kangning/rsi/README.md` 第 2 节。

## 启动（FastWAM/XPOLICYLAB env；依赖 dora-rs==0.4.1 numpy pyarrow + forge_policy SDK）

```bash
PYTHONPATH=<REPO>/policy/src:<REPO>/nodes/policies/fastwam/src \
XPOLICYLAB_ROOT=/data2/kangning/fastwam_eval/RoboDojo/XPolicyLab \
POLICY_NAME=FastWAM_80k_forge \
ENV_CFG_TYPE=arx_x5 \
ACTION_TYPE=joint \
<FASTWAM_ENV>/bin/python -u -m fastwam_policy.main
```

dataflow 里 `policy_command` 输入必须 `queue_size: 32`
（`forge_policy.contract.MIN_POLICY_COMMAND_QUEUE`：回合边界连发 4 条命令，
默认深度会挤掉前面的且不报错）。完整两节点拓扑见本目录的
`dataflow.example.yaml`（策略侧资产；评测节点本身策略无关）。

## 单测

```bash
python -m pytest tests -q --basetemp=<tmp>
```

覆盖：双 reset、每帧 update_obs 节奏、chunk 队列、**哨兵随本 chunk 末动作**、
空 chunk 返 None、观测规格（28 维 6 键）与评测侧同源。不 import dora /
XPolicyLab（model 用替身、规格注入维度）。

## 已知边界

- FastWAM_80k_forge 从未在此协议上实跑（只有 G05 在原流程跑通过真实回合）；
  首次上机先跑 v2 文档第 6 节步 1 的对照回合。
- 指令按轮传入：`set_instruction` 随时生效于下一次 chunk 推理；进程常驻，
  三轮 RSI 不重载权重。

## 协议 v2（评测端两级握手）

ready 时经 `runtime.set_handshake()` 上报契约自报（policy_id/action_type/weight_id/obs_state_dim）；reset_scene 后自动回 reset_ack；每帧观测随 obs_meta 带回合身份，动作经 action_meta 回带——异回合动作被评测端机械丢弃。字段表见 `policy/src/forge_policy/HANDSHAKE_FIELDS.md`。
