"""Bảng kết quả SO VỚI MỤC TIÊU (Slide 10) — để nhìn là biết tốt hay xấu.

Mục tiêu chính: PR-AUC của FL + DP >= 85% PR-AUC gộp chung (cùng fold), với ε <= 1.
Mỗi cách huấn luyện được quy về % gộp chung CÙNG FOLD, rồi lấy trung bình ± độ lệch qua các fold.
Cột "Đạt?" so % gộp chung với ngưỡng 85% (chỉ đánh giá cho các dòng có DP; dòng không DP là mốc tham chiếu).
Phụ: VDR@1% so với 0,80 (chỉ tham khảo — trên IBM dao động rất mạnh).

Chạy:  python scripts/11_so_voi_muc_tieu.py <thư mục kết quả 1> [<thư mục 2> ...] [--out ten]
       (các thư mục chứa per_fold.csv của 09_ibm_fl_dp.py; phải có dòng 'central' ở ít nhất một thư mục)
Ra:    results/so_voi_muc_tieu[_ten]/bang.md, bang.csv (+ lưu trữ)
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from fraudfl.config import RESULTS
from fraudfl.runlog import archive_run

TARGET_REL = 0.85        # FL + DP >= 85% gộp chung
TARGET_EPS = 1.0
TARGET_VDR = 0.80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(d / "per_fold.csv") for d in args.dirs], ignore_index=True)
    central = df[df["setting"] == "central"].set_index("fold")["pr_auc"]
    if central.empty:
        raise SystemExit("Không có dòng 'central' — cần kết quả gộp chung để làm mốc.")
    df = df[df["fold"].isin(central.index)].copy()
    df["pct_central"] = df["pr_auc"] / df["fold"].map(central)

    rows = []
    for s, g in df.groupby("setting", sort=False):
        eps = g["epsilon"].iloc[0] if "epsilon" in g and pd.notna(g["epsilon"].iloc[0]) else np.inf
        is_dp = np.isfinite(eps)
        pct = g["pct_central"]
        ok_rel = pct.mean() >= TARGET_REL
        rows.append({
            "Cách": s, "Số fold": len(g),
            "PR-AUC": f"{g['pr_auc'].mean():.4f} ± {g['pr_auc'].std(ddof=1) if len(g) > 1 else 0:.4f}",
            "% gộp chung (mục tiêu ≥ 85%)": f"{100 * pct.mean():.0f}% ± {100 * (pct.std(ddof=1) if len(g) > 1 else 0):.0f}",
            "ε (mục tiêu ≤ 1)": "—" if not is_dp else f"{eps:g}",
            "Đạt mục tiêu chính?": ("mốc" if not is_dp else
                                    ("✅" if ok_rel and eps <= TARGET_EPS else
                                     ("⚠️ đạt % nhưng ε > 1" if ok_rel else f"❌ thiếu {100 * (TARGET_REL - pct.mean()):.0f} điểm %"))),
            "VDR@1% (phụ, mục tiêu ≥ 0,80)": f"{g['vdr@1%'].mean():.2f} {'✅' if g['vdr@1%'].mean() >= TARGET_VDR else '❌'}",
            "_pct": pct.mean(),
        })
    tab = pd.DataFrame(rows)
    out = RESULTS / ("so_voi_muc_tieu" + (f"_{args.out}" if args.out else ""))
    out.mkdir(parents=True, exist_ok=True)
    tab.drop(columns="_pct").to_csv(out / "bang.csv", index=False)
    md = ["| " + " | ".join(c for c in tab.columns if c != "_pct") + " |",
          "|" + "---|" * (len(tab.columns) - 1)]
    md += ["| " + " | ".join(str(r[c]) for c in tab.columns if c != "_pct") + " |" for _, r in tab.iterrows()]
    note = (f"\nMốc gộp chung theo fold: " + ", ".join(f"fold {k}: {v:.4f}" for k, v in central.items())
            + "\nNguồn: " + ", ".join(str(d) for d in args.dirs))
    (out / "bang.md").write_text("\n".join(md) + "\n" + note + "\n", encoding="utf-8")
    print("\n".join(md) + note)
    print(f"Đã lưu trữ: {archive_run(out, note='Bảng so với mục tiêu: ' + ', '.join(d.name for d in args.dirs))}")


if __name__ == "__main__":
    main()
