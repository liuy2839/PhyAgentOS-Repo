"""fastwam_policy 节点入口：加载 XPolicyLab 模型并按统一策略协议运行。

用法（与 ReKep 节点同构，见 nodes/policies/rekep）：
    python -m fastwam_policy.main
环境变量见 policy.py 文件头。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 让同仓库的 forge_policy SDK 与本包可导入（部署环境装包后可去掉）
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "policy" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .policy import main  # noqa: E402

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
