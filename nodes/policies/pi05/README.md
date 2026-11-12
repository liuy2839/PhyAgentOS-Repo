# pi05_policy —— π0.5 策略独立节点（WS 冒烟已通过）

按 `forge_policy` SDK 实现的 π0.5 适配节点。架构为**方案 a：WS 客户端 + 本地
策略服务器**——openpi 的依赖（jax 等）隔离在独立 venv，节点进程不加载模型，
而是拉起本地 Pi_05 服务器子进程、经 `WsModelClient` 驱动推理，退出时进程组
整组清理（不留占显存的孤儿）。

**状态：2026-11-12 zy2d 端到端冒烟通过**（拉服务器 → WS 握手 → 假观测
`get_action` → 50 步动作 chunk / 首帧 14.1s → 清理）。dora 两节点联调待做。

## 实测参数（zy2d，已预填进 policy.py，环境变量可覆盖）

| 参数 | 值 | 说明 |
|---|---|---|
| 权重 | `/data2/kangning/models/robotwin_pi05/RoboTwin-lerobot_v30-aloha_agilex-joint-0` | 目录名的 "RoboTwin/aloha" 是历史命名，训练配置是 arx-x5 |
| checkpoint | `29999` | deploy.yml 写 59999 但盘上只有 29999 |
| repo_id | `RoboTwin_lerobot_v30` | norm stats 在 `29999/assets/<repo_id>/norm_stats.json`（不是 deploy.yml 里的 arx_x5_sim） |
| XPolicyLab | `/data1/liuyi/XPolicyLab` | 可写副本（原目录只读建不了 egg-info；env_cfg 经软链指回仓库根） |
| openpi venv | `/data1/liuyi/pi05_openpi_venv` | uv sync 构建（jax 0.5.3 + openpi + xpolicylab） |
| 动作空间 | joint 14 维 | 带名 dict（left/right_arm_joint_state 6 + left/right_ee_joint_state 1），与 fastwam 同款展平 |
| chunk | **一次推理 50 步** | `step()` 返回 `(vec, is_last)`，末动作自动附 effort 哨兵 |
| 首帧推理 | ~14s | 评测端 watchdog 20s / give_up 900s 覆盖 |
| GPU | `PI05_GPU`（默认 2）+ `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15` | 共享卡必须限 jax 预留（默认 75% 会 OOM），跑前 nvidia-smi 挑空卡 |
| 端口 | `PI05_PORT`（默认 6008） | 官方默认 6000 易撞，避开 |

## 启动

```bash
PYTHONPATH=<REPO>/policy/src:<REPO>/nodes/policies/pi05/src \
XPOLICYLAB_ROOT=/data1/liuyi/XPolicyLab \
PI05_MODEL_DIR=/data2/kangning/models/robotwin_pi05/RoboTwin-lerobot_v30-aloha_agilex-joint-0 \
<pi05_openpi_venv 或独立 env>/bin/python -u -m pi05_policy.main
```

节点自己拉起/停止策略服务器子进程，编排层和评测端无感知。dataflow 接线与
fastwam 相同（policy_command/proprio_state/image/* 入，action + action_meta +
policy_status 出）——参照 `nodes/policies/fastwam/dataflow.example.yaml` 抄
一份 pi05 版即可（评测节点本身策略无关，只改 policy 节点段）。

## WS 冒烟复跑

```bash
# zy2d 上：
cd /data1/liuyi/pi05_ws_smoke && /data1/liuyi/pi05_openpi_venv/bin/python smoke.py
# 预期：SERVER READY → INFERENCE OK: 50 actions → SMOKE PASS
```

## 单测

```bash
python -m pytest tests -q --import-mode=importlib
```

8 个用例用 FakeWsClient 替身锁定节点侧语义（update_obs/get_action 调用序列、
chunk 元组与队列、空 chunk、双 reset、dict→14 维展平、分量名形状）——它们锁
的是协议，不依赖真权重/服务器。
