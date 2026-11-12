"""pi05_policy 节点入口：python -m pi05_policy.main（协议骨架，推理待算法方）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "policy" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .policy import main  # noqa: E402

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
