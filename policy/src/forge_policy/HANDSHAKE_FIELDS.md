# forge_policy 握手契约字段表（协议规范）

> **文档定位**：本表是 forge_policy 协议层的规范文档，属于 SDK（与本目录的
> `contract.py` / `runtime.py` 同级同源），**不隶属于任何具体策略**——现有
> 的 fastwam / rekep / pi05 只是当前使用者；后续任何策略节点经本 SDK 接入
> 评测，都遵循同一份契约。协议层的稳定性约定：字段变更必须同步本表、
> `contract.py`、评测端 `nodes/benchmarks/robodojo_eval/handshake.py` 三处，
> 且向下兼容（新字段可选，旧字段不改语义）。

策略节点启动、模型加载完成后，经 `policy_status` 通道上报 `ready` 载荷；
评测端（nodes/benchmarks/robodojo_eval/handshake.py）逐项校验，通过才允许
回合启动。**字段名是双方代码的契约**——改任何一边必须同步另一边。

## ready 载荷字段

| 字段 | 必填 | 含义 | 评测端校验 |
|---|---|---|---|
| `status` | ✅ | 恒为 `"ready"` | 识别消息类型 |
| `policy_id` | ✅ | 策略身份（结果 JSON 的身份标注来源） | 记录进结果 |
| `action_type` | ✅ | `joint` 或 `ee`（动作空间选择） | ∈ 环境接受的集合（arx_x5 两者都收） |
| `action_names` | ✅ | 动作分量名列表（发 JointCommand.name 的顺序） | 与该空间的模板逐字一致（14/16 维） |
| `cameras` | ✅ | 需要收齐的相机通道 | ⊆ 环境提供的 camera_map |
| `weight_id` | 建议 | 权重身份（如 `pi05-robodojo-arx-x5-joint`） | 缺失 → 警告（结果不可追溯） |
| `weight_sha256` | 建议 | 权重摘要 | 记录进结果 |
| `emit_chunk_end` | 建议 | 是否发 chunk 末哨兵（RoboDojo 必须 true） | 与 chunk_per_action 联合校验 |
| `chunk_per_action` | 建议 | 每帧制策略（ReKep）恒附哨兵 | 需搭配 emit_chunk_end=true |
| `obs_state_dim` | 可选 | 期望的观测状态维度（arx_x5=28） | 显式给出时必须等于 28 |
| `est_startup_s` | 可选 | 预计启动耗时（pi05 加载 7-8 分钟） | 记录（展示给编排层用） |

## 校验失败的表现

- **握手失败**：`eval_status` 收到 `{"command": "rejected", "handshake_errors": [...]}`
  （启动阶段秒级暴露，Isaac 冷启动成本不浪费）；
- **运行中失败**：策略发 `{"status": "failed", "error_code": ...}` → 评测端立即
  收尾回合为 failed（不再等一个死掉的策略）。

## 契约的两个自由维度（新策略按此选择，与具体节点无关）

| 维度 | 取值 | 语义 |
|---|---|---|
| `action_type` | `joint` / `ee` | 必须 ∈ 环境契约声明的 `accepted_action_spaces`（arx_x5 两者都收；ee 由环境内部解 IK） |
| chunk 语义 | chunk 制（元组声明末动作）/ 每帧制（`chunk_per_action=true`） | chunk 制：`step()` 返回 `(vec, is_last)`；每帧制：每个动作都是 N=1 chunk 的末动作 |

## 当前使用者的自报值（示例，非规范）

| 策略 | action_type | chunk 语义 | 说明 |
|---|---|---|---|
| fastwam | joint（14：arm 6+6 + 夹爪 1+1） | chunk 制 | VLA，一次推理产一段动作 |
| rekep | ee（16：ee_pose 7+7 + 夹爪 1+1） | 每帧制 | 每帧重新决策 |
| pi05 | joint（14，同 fastwam 模板） | chunk 制（实测 50 步/次） | openpi 架构，WS 服务器隔离依赖 |

**新策略接入**：按上表两个维度选择后填 `PolicyContract` + `set_handshake`，
评测端按环境契约自动校验——策略代码、评测节点代码都不用改。
