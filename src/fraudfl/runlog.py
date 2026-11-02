"""Lưu trữ kết quả TỪNG LẦN CHẠY để sau này viết báo cáo cả quá trình (quy tắc của nhóm, 02/11/2026).

Thư mục kết quả chính (vd results/gnn_synthetic/) luôn là bản MỚI NHẤT và bị ghi đè khi chạy lại. Vì vậy cuối mỗi
script gọi `archive_run(...)`:
- chép toàn bộ thư mục kết quả vào results/_luu_tru/<YYYY-MM-DD_HHMM>_<tên thư mục>/ (không bao giờ ghi đè),
- ghi meta.json: thời điểm, lệnh chạy, commit git, có thay đổi chưa commit không, ghi chú,
- thêm một dòng vào results/NHAT_KY_CHAY.csv (sổ tổng hợp mọi lần chạy).
Thư mục _luu_tru và sổ nhật ký được commit lên git cùng kết quả.
"""

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fraudfl.config import RESULTS, ROOT

ARCHIVE = RESULTS / "_luu_tru"
JOURNAL = RESULTS / "NHAT_KY_CHAY.csv"
_FIELDS = ["thoi_diem", "script", "lenh", "commit", "co_thay_doi_chua_commit", "thu_muc_luu_tru", "ghi_chu"]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def archive_run(out_dir: Path, note: str = "", argv: list[str] | None = None) -> Path:
    """Chép `out_dir` vào kho lưu trữ, ghi meta và nhật ký. Trả về thư mục lưu trữ."""
    argv = list(sys.argv if argv is None else argv)
    now = datetime.now()
    dest = ARCHIVE / f"{now:%Y-%m-%d_%H%M}_{Path(out_dir).name}"
    i = 2
    while dest.exists():                                    # hai lần chạy cùng phút → thêm hậu tố, không ghi đè
        dest = ARCHIVE / f"{now:%Y-%m-%d_%H%M}_{Path(out_dir).name}_{i}"
        i += 1
    shutil.copytree(out_dir, dest)
    meta = {
        "thoi_diem": now.isoformat(timespec="seconds"),
        "script": Path(argv[0]).name if argv else "",
        "lenh": " ".join(argv),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "co_thay_doi_chua_commit": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "thu_muc_luu_tru": dest.relative_to(ROOT).as_posix(),
        "ghi_chu": note,
    }
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    new = not JOURNAL.exists()
    with open(JOURNAL, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new:
            w.writeheader()
        w.writerow(meta)
    return dest
