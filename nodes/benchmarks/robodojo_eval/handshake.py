"""评测侧的 policy 契约声明与握手校验（《修改方案》协议层 · Policy 协议模块）。

职责（对应方案内部模块表的 "Policy 协议：握手、身份与序号校验、错误处理"）：

1. **环境契约**（`env_contract`）：声明评测环境能提供什么——相机、观测状态
   维度与键表、**接受的动作空间集合**（及各空间的分量名表）、判分口径摘要。
2. **策略契约自报**（`PolicyHandshake`）：策略节点就绪时经 `policy_status` 上报
   的身份与需求（ready 载荷）。
3. **握手校验**（`check_handshake`）：两者逐项比对，任何一项不兼容都给出
   **具体哪一项不匹配**的结构化结论——错配在启动阶段秒级暴露，而不是等 Isaac
   冷启动几分钟后的第一步动作才炸。

**契约真源与稳定性**（协议层独立性的关键设计）：

* 契约的**真源是活环境**：env 已建时一律用 `env.action_component_names()` /
  `env.state_joint_names()` 的活推导校验（`check_handshake_live`）——机器人
  换代、env_cfg_type 变更，评测端**零代码改动**。
* `ACTION_SPACE_TEMPLATES` 等静态表只是 **arx_x5 的离线快照**，用于两个场景：
  env 未建时的启动期快检（ Isaac 还没起，先挡明显错配）和单测。它与
  `robodojo_env.build_state_spec()` 同源同构，**换环境时两者一起更新**（或
  直接依赖活推导，静态表过期只影响快检不影响正确性——活校验是最后防线）。
* 字段名与 policy 侧 `forge_policy.PolicyContract` 的自报载荷对齐
  （action_names / cameras / emit_chunk_end / chunk_per_action），双方代码
  没有共享模块，靠这些字段名契约保持一致——改任何一边必须同步另一边并在
  两侧测试中体现。字段规范见 `policy/src/forge_policy/HANDSHAKE_FIELDS.md`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# arx_x5 的动作空间快照（离线快检用；真源见 robodojo_env.build_state_spec——
# 两者同源同构，换 env_cfg_type 时一起更新）。
#   joint → {p}arm_joint_state（6 维）+ {p}ee_joint_state（1 维）
#   ee    → {p}ee_pose（7 维）+ {p}ee_joint_state（1 维）
ACTION_SPACE_TEMPLATES: dict[str, list[tuple[str, int]]] = {
    "joint": [
        ("left_arm_joint_state", 6),
        ("left_ee_joint_state", 1),
        ("right_arm_joint_state", 6),
        ("right_ee_joint_state", 1),
    ],
    "ee": [
        ("left_ee_pose", 7),
        ("left_ee_joint_state", 1),
        ("right_ee_pose", 7),
        ("right_ee_joint_state", 1),
    ],
}


def obs_state_keys() -> list[tuple[str, int]]:
    """观测 state 的 (键, 维度) 表快照：双臂 arm + ee_joint + ee_pose，字典序。

    与 robodojo_env.build_obs_spec 同源同构；活校验用 env.state_joint_names()。
    """
    keys: list[tuple[str, int]] = []
    for p in ("left_", "right_"):
        keys.append((f"{p}arm_joint_state", 6))
        keys.append((f"{p}ee_joint_state", 1))
        keys.append((f"{p}ee_pose", 7))
    return sorted(keys, key=lambda kv: kv[0])


def action_space_names(space: str) -> list[str]:
    """某动作空间的完整分量名列表（快照展开）。"""
    if space not in ACTION_SPACE_TEMPLATES:
        raise ValueError(f"未知动作空间 {space!r}；可选 {sorted(ACTION_SPACE_TEMPLATES)}")
    names: list[str] = []
    for key, dim in ACTION_SPACE_TEMPLATES[space]:
        names.extend(f"{key}.{i}" for i in range(dim))
    return names


def expected_action_dim(space: str) -> int:
    return sum(d for _, d in ACTION_SPACE_TEMPLATES[space])


def env_contract(
    camera_map: dict[str, str],
    *,
    live_action_names: list[str] | None = None,
    live_state_names: list[str] | None = None,
    env_cfg_type: str = "arx_x5",
    benchmark: str = "robodojo",
) -> dict[str, Any]:
    """环境契约声明。`live_*` 提供时以活推导为准（env 已建的正路）；
    未提供时用静态快照（env 未建的启动期）。"""
    if live_action_names is not None:
        action_spaces = {"live": {"action_names": list(live_action_names),
                                  "action_dim": len(live_action_names)}}
        accepted = ["live"]
        obs_keys = None
        obs_dim = len(live_state_names) if live_state_names is not None else None
    else:
        action_spaces = {
            space: {"action_names": action_space_names(space),
                    "action_dim": expected_action_dim(space)}
            for space in sorted(ACTION_SPACE_TEMPLATES)
        }
        accepted = sorted(ACTION_SPACE_TEMPLATES)
        obs_keys = [[k, d] for k, d in obs_state_keys()]
        obs_dim = sum(d for _, d in obs_state_keys())
    contract: dict[str, Any] = {
        "benchmark": benchmark,
        "env_cfg_type": env_cfg_type,
        "cameras": sorted(camera_map),
        "accepted_action_spaces": accepted,
        "action_spaces": action_spaces,
        "judging": {
            "success_semantics": "episode_end_and_not_failed",
            "process_score": "reward_manager.get_score()/100",
            "chunk_end_query": "official_2N_plus_1",
            "horizon": "task_step_lim",
        },
    }
    if obs_keys is not None:
        contract["obs_state_keys"] = obs_keys
    if obs_dim is not None:
        contract["obs_state_dim"] = obs_dim
    return contract


# ---------------------------------------------------------------- 策略自报


@dataclass(frozen=True)
class PolicyHandshake:
    """策略节点 ready 时自报的契约（policy_status 载荷）。

    必填：policy_id、action_type、action_names、cameras。
    可选：weight_id / weight_sha256（结果可追溯）、est_startup_s、
         emit_chunk_end / chunk_per_action（哨兵语义声明）、obs_state_dim。
    """

    policy_id: str
    action_type: str
    action_names: list[str]
    cameras: list[str]
    weight_id: str | None = None
    weight_sha256: str | None = None
    est_startup_s: float | None = None
    emit_chunk_end: bool = False
    chunk_per_action: bool = False
    obs_state_dim: int | None = None

    @classmethod
    def from_status_payload(cls, payload: dict[str, Any]) -> "PolicyHandshake":
        missing = [
            k for k in ("policy_id", "action_type", "action_names", "cameras")
            if not payload.get(k)
        ]
        if missing:
            raise ValueError(f"ready 载荷缺少必填字段：{missing}")
        return cls(
            policy_id=str(payload["policy_id"]),
            action_type=str(payload["action_type"]),
            action_names=[str(x) for x in payload["action_names"]],
            cameras=[str(x) for x in payload["cameras"]],
            weight_id=payload.get("weight_id"),
            weight_sha256=payload.get("weight_sha256"),
            est_startup_s=payload.get("est_startup_s"),
            emit_chunk_end=bool(payload.get("emit_chunk_end", False)),
            chunk_per_action=bool(payload.get("chunk_per_action", False)),
            obs_state_dim=payload.get("obs_state_dim"),
        )


# ---------------------------------------------------------------- 校验


@dataclass
class HandshakeResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _common_checks(
    hs: PolicyHandshake,
    camera_map: dict[str, str],
    expect_action_names: list[str] | None,
    expect_obs_dim: int | None,
    res: "HandshakeResult",
    action_space_accepted: bool,
) -> None:
    """与真源无关的公共校验：相机 ⊆ 提供、观测维度、哨兵语义、权重身份。"""
    provided = set(camera_map)
    missing = [c for c in hs.cameras if c not in provided]
    if missing:
        res.ok = False
        res.errors.append(
            f"策略需要的相机环境未提供：{missing}；环境可提供 {sorted(provided)}"
        )
    if expect_obs_dim is not None and hs.obs_state_dim is not None \
            and hs.obs_state_dim != expect_obs_dim:
        res.ok = False
        res.errors.append(f"obs_state_dim={hs.obs_state_dim}，环境提供 {expect_obs_dim}")
    if hs.chunk_per_action and not hs.emit_chunk_end:
        res.ok = False
        res.errors.append("chunk_per_action=True 需要 emit_chunk_end=True")
    if not hs.weight_id:
        res.warnings.append("未自报 weight_id：结果无法追溯到具体权重")
    if not action_space_accepted and expect_action_names is None:
        pass  # 由调用方处理"空间不在集合"的具体错误


def check_handshake(
    hs: PolicyHandshake, camera_map: dict[str, str]
) -> HandshakeResult:
    """启动期快检（env 未建）：对照 arx_x5 静态快照。

    注意：这是**前置快检**，挡明显错配；最终防线是 env 建好后的
    `check_handshake_live`（活推导）——静态快照过期只影响快检的覆盖面，
    不影响最终正确性。
    """
    res = HandshakeResult(ok=True)
    if hs.action_type not in ACTION_SPACE_TEMPLATES:
        res.ok = False
        res.errors.append(
            f"action_type={hs.action_type!r} 不在环境支持集合 "
            f"{sorted(ACTION_SPACE_TEMPLATES)} 内（arx_x5 接受 joint/ee）"
        )
        return res
    expect = action_space_names(hs.action_type)
    if hs.action_names != expect:
        res.ok = False
        res.errors.append(
            f"action_names 与 {hs.action_type} 模板不一致："
            f"期望 {len(expect)} 个 {expect[:3]}...，收到 {len(hs.action_names)} 个 "
            f"{hs.action_names[:3] if hs.action_names else []}..."
        )
    _common_checks(hs, camera_map, expect, sum(d for _, d in obs_state_keys()), res,
                   action_space_accepted=True)
    return res


def check_handshake_live(
    hs: PolicyHandshake,
    camera_map: dict[str, str],
    *,
    live_action_names: list[str],
    live_state_names: list[str],
) -> HandshakeResult:
    """最终防线（env 已建）：对照活推导的分量名表，逐字比对。

    不经过静态模板——机器人换代/env_cfg_type 变更时评测端零代码改动。
    """
    res = HandshakeResult(ok=True)
    if hs.action_names != list(live_action_names):
        res.ok = False
        res.errors.append(
            f"action_names 与环境活推导不一致：期望 {len(live_action_names)} 个 "
            f"{live_action_names[:3]}...，收到 {len(hs.action_names)} 个 "
            f"{hs.action_names[:3] if hs.action_names else []}..."
        )
    if hs.obs_state_dim is not None and hs.obs_state_dim != len(live_state_names):
        res.ok = False
        res.errors.append(
            f"obs_state_dim={hs.obs_state_dim}，环境活推导 {len(live_state_names)}"
        )
    _common_checks(hs, camera_map, None, len(live_state_names), res,
                   action_space_accepted=True)
    return res
