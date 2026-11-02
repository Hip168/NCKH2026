"""So điểm do consumer real-time ghi trong Postgres với điểm tính theo lô trên cùng mô hình.

Chạy:  python scripts/52_doi_chieu_real_time.py --mo-hinh artifacts/<run>/model.pt
"""

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import psycopg

from fraudfl.data.features import FEATURES
from fraudfl.stream.scorer import Scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mo-hinh", required=True)
    ap.add_argument("--postgres", default="postgresql://nckh:nckh@localhost:55432/nckh")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    with psycopg.connect(args.postgres) as db:
        rows = db.execute("SELECT tx_id, ngan_hang, diem, che_do, do_tre_ms FROM giao_dich").fetchall()
        n_alert, orphan = db.execute("""SELECT count(*), count(*) FILTER (WHERE g.tx_id IS NULL)
                                        FROM canh_bao c LEFT JOIN giao_dich g USING (tx_id)""").fetchone()
    live = pd.DataFrame(rows, columns=["TRANSACTION_ID", "bank", "diem", "che_do", "do_tre_ms"])
    tx = pd.read_parquet("data/processed/synthetic_tx.parquet", columns=["TRANSACTION_ID", "TX_DATETIME"] + FEATURES)
    m = live.merge(tx, on="TRANSACTION_ID")
    gaps = []
    for bank, g in m.groupby("bank"):
        s = Scorer(args.mo_hinh, "data/banks", int(bank), args.fold)
        gaps.append(pd.Series(np.abs(s.score(g[FEATURES].to_numpy(float)) - g["diem"].to_numpy()), index=g.index))
    gap = pd.concat(gaps)
    print(f"Giao dịch trong Postgres: {len(live):,}, không trùng: {live['TRANSACTION_ID'].is_unique}")
    print(f"Cảnh báo: {n_alert:,}, mồ côi: {orphan}")
    print(f"Chênh điểm online và theo lô: lớn nhất {gap.max():.2e}, số dòng lệch > 1e-4: {(gap > 1e-4).sum()}")
    print(live.groupby("che_do")["do_tre_ms"].describe(percentiles=[0.5, 0.95, 0.99]).round(1).to_string())


if __name__ == "__main__":
    main()
