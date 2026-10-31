"""Bước 2: tạo file chia fold cố định (seed=42). Chỉ chạy MỘT lần rồi commit file kết quả.

Chạy:  python scripts/02_make_splits.py
Ra:    data/splits/fold_assignments_synthetic.csv
"""

import sys

import _bootstrap  # noqa: F401

import pandas as pd

from fraudfl.config import DATA_PROCESSED, DATA_SPLITS
from fraudfl.data.splits import make_fold_assignments


def main():
    out_path = DATA_SPLITS / "fold_assignments_synthetic.csv"
    if out_path.exists() and "--force" not in sys.argv:
        sys.exit(f"{out_path} đã tồn tại. Không tạo lại để giữ kết quả tái lập được (dùng --force nếu thật sự cần).")

    tx = pd.read_parquet(DATA_PROCESSED / "synthetic_tx.parquet",
                         columns=["TRANSACTION_ID", "CLIENT_ID", "TX_FRAUD", "TX_TIME_DAYS"])
    folds = make_fold_assignments(tx)
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_path, index=False)

    check = folds.merge(tx[["TRANSACTION_ID", "TX_FRAUD"]], on="TRANSACTION_ID")
    print("Tỷ lệ gian lận theo (client × outer_fold):")
    print(check.pivot_table(index="CLIENT_ID", columns="outer_fold", values="TX_FRAUD").round(4).to_string())
    print(check["temporal_set"].value_counts().to_string())
    print(f"Đã lưu {out_path}")


if __name__ == "__main__":
    main()
