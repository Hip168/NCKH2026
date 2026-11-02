"""Nạp lịch sử giao dịch trước thời điểm phát vào Redis, để đặc trưng cửa sổ 30 ngày đúng ngay từ giao dịch đầu."""

import argparse

import pandas as pd

from fraudfl.stream.online_features import HISTORY_SECONDS, Tx, keys
from fraudfl.stream.redis_store import RedisStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--du-lieu", default="data/processed/synthetic_tx.parquet")
    ap.add_argument("--redis", default="redis://localhost:6379/0")
    ap.add_argument("--banks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tu-ngay", type=int, default=60)
    args = ap.parse_args()

    tx = pd.read_parquet(args.du_lieu, columns=["TRANSACTION_ID", "CLIENT_ID", "CUSTOMER_ID", "TERMINAL_ID",
                                                "TX_AMOUNT", "TX_DATETIME", "TX_FRAUD", "TX_TIME_DAYS"])
    tx = tx[tx["CLIENT_ID"].isin(args.banks) & (tx["TX_TIME_DAYS"] < args.tu_ngay)]
    ts = tx["TX_DATETIME"].astype("int64").to_numpy() // 10**9 if str(tx["TX_DATETIME"].dtype).endswith("[ns]") \
        else (tx["TX_DATETIME"] - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64").to_numpy()
    tx = tx.assign(ts=ts)
    tx = tx[tx["ts"] >= tx["ts"].max() - HISTORY_SECONDS]
    rows = []
    for r in tx.itertuples():
        t = Tx(int(r.TRANSACTION_ID), int(r.CLIENT_ID), int(r.CUSTOMER_ID), int(r.TERMINAL_ID), float(r.TX_AMOUNT),
               int(r.ts), int(r.TX_FRAUD))
        k_c, k_t = keys(t)
        rows += [(k_c, t.tx_id, t.ts, t.amount), (k_t, t.tx_id, t.ts, t.fraud)]
    RedisStore(args.redis).add_many(rows)
    print(f"Đã nạp {len(tx):,} giao dịch lịch sử vào Redis")


if __name__ == "__main__":
    main()
