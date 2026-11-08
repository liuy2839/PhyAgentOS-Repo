# rekep_policy · 中文说明

[English](README.md) | [简体中文](README_zh.md)

把 ReKep（关系关键点约束操作）接入 Forge/RoboDojo 评测链路的**统一策略节点**：
对评测端它只是一个符合策略契约的节点（观测进、`JointCommand` 出），内部完成
感知 → 规划 → 分阶段求解执行。

## 当前状态

| 能力 | 状态 |
| --- | --- |
| 统一策略协议（生命周期 / 动作 / 状态） | ✅ 已接入，Dora 两进程 smoke 通过 |
| RoboDojo 单集评测 | ✅ 跑满 700 步，结果 JSON + 录像落盘（eval5） |
| ReKep 求解器（reference subgoal/path） | ✅ 真实运行；预热失败不致命，求解失败转"保持位姿" |
| 动作空间 | ✅ `ee`（评测端 cuRobo 解 IK）与 `joint` 两种 |
| 真实 X5 IK（求解器可达性评估） | ✅ `X5A.urdf` 加载，IK 误差 ~0mm |
| 真实视觉感知（深度 + DINOv2） | ⚠️ 代码就绪；本机 Isaac 5.1 深度注解器触发 CUDA 崩溃，默认不启用 |
| VLM 约束规划 | ⚠️ 代码就绪（`REKEP_PLAN_SOURCE=vlm`），需注入密钥；默认模板 |
| 模板规划模式 | ✅ 有 region 走"抓→放"3 段；无 region（如 `general_pickup`）走"抓→抬升 10cm"2 段，并按指令词选目标 |
| 动作安全护栏 | ✅ 位姿有限值/四元数归一化/工作空间裁剪；路径点避障半径 + 离地高度钳制 |

## 目录

```text
src/rekep_policy/
├── core.py                # ReKepPolicy：每集一次感知+规划，每帧推进一步执行
├── profile.py             # 动作命名映射 + joint/ee 两种动作空间
├── ik_adapters.py         # StubIK / X5IK（MuJoCo 数值 IK）
├── perception_adapters.py # SceneObjectPerception（真值）/ DepthDinoPerception（视觉）
├── assembly.py            # 真实算法装配（rekep_alg + IK + 规划器）
└── main.py                # 节点入口：PolicyRuntime 驱动
rekep_alg/                 # 算法核心（感知/规划+沙箱/执行状态机）
```

## 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `REKEP_PERCEPTION` | `scene_objects`（默认，稳定）或 `dino`（需深度） |
| `REKEP_IK` | `x5`（默认）或 `stub` |
| `REKEP_PLAN_SOURCE` | `template`（默认）或 `vlm` |
| `REKEP_X5_URDF` / `REKEP_X5_BASE_POS` / `REKEP_X5_BASE_QUAT_WXYZ` | X5 运动学 |
| `REKEP_REF_RUNTIME` | reference runtime 目录（打包后必填） |
| `REKEP_SUBGOAL_MAXFUN` / `REKEP_PATH_MAXFUN` / `REKEP_MINIMIZER_MAXITER` | 求解预算 |

## 运行

源码部署：`dora run dataflow.example.yaml`（路径为占位符，按部署机补齐；评测侧
是独立节点 `nodes/benchmarks/robodojo_eval/`，本策略需要其 `config.yaml` 开
`publish_scene_objects: true`）。

## 已知边界

- RoboDojo `arx_x5` 是双臂 4 瓶 + 交接任务；本策略只驱动左臂，右臂保持当前位姿，
  因此当前分数不代表 ReKep 的真实任务能力。
- 深度注解器 `distance_to_image_plane` 在该机器上不稳定（CUDA illegal address /
  device lost），`dino` 感知暂不可用。
- 夹爪抓取成功与否在评测观测里没有反馈，策略按"已抓取"推进（记录在计划阶段）。

## 实测记录（2026-10-29，zyserver）

- **稳定路径**：`put_bottles_into_dustbin` + `scene_objects` 感知 + 真实 X5 IK，
  连续两次跑满 700 步（`valid_episodes=1`、平均策略延迟 ~15ms、结果与录像落盘）。
- **诊断对照**：`general_pickup` 场景下，策略只保持位姿时 200 步无异常；一旦手臂
  开始运动，2–17 步内出现 `CUDA_ERROR_ILLEGAL_ADDRESS`（已排除干扰物数量因素：
  仅保留 1 个物体仍复现）。已加动作护栏（位姿校验、避障半径、离地钳制），
  仍需在环境/驱动稳定后复测。
- **并行开发注意（2026-10-29 记录，已被后续变更取代）**：当时的 PAOS `bench_tool`
  Skill 流程（`paos skill start robodojo-bench`）已从本仓库移除；现在的评测入口是
  独立节点 `robodojo_eval`，由 `eval_command` 逐回合驱动
  （见 `nodes/benchmarks/robodojo_eval/README.md`）。
