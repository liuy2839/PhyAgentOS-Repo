"""fastwam_policy：FastWAM/XPolicyLab 策略的 forge_policy 统一协议节点。"""

from .policy import FastWAMPolicy, build_model, main  # noqa: F401

__all__ = ["FastWAMPolicy", "build_model", "main"]
