"""Import ở đầu mọi script: thêm src/ vào sys.path và cho phép in tiếng Việt trên console Windows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8")
