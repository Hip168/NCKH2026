"""Phát lại giao dịch lên Kafka theo đúng thứ tự thời gian. Mỗi ngân hàng là một partition của topic.

Bật idempotence để broker không ghi trùng khi producer gửi lại. `--toc-do 0` là phát nhanh nhất có thể.
"""

import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer

TOPIC = "giao_dich"


def message(r) -> bytes:
    return json.dumps({"tx_id": int(r.TRANSACTION_ID), "bank": int(r.CLIENT_ID), "customer": int(r.CUSTOMER_ID),
                       "terminal": int(r.TERMINAL_ID), "amount": float(r.TX_AMOUNT),
                       "ts": int(r.TX_DATETIME.timestamp()), "fraud": int(r.TX_FRAUD),
                       "sent": time.time()}).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--du-lieu", default="data/processed/synthetic_tx.parquet")
    ap.add_argument("--kafka", default="localhost:29092")
    ap.add_argument("--banks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tu-ngay", type=int, default=60)
    ap.add_argument("--den-ngay", type=int, default=62)
    ap.add_argument("--toc-do", type=float, default=0, help="giao dịch/giây, 0 = nhanh nhất có thể")
    args = ap.parse_args()

    tx = pd.read_parquet(args.du_lieu, columns=["TRANSACTION_ID", "CLIENT_ID", "CUSTOMER_ID", "TERMINAL_ID",
                                                "TX_AMOUNT", "TX_DATETIME", "TX_FRAUD", "TX_TIME_DAYS"])
    tx = tx[tx["CLIENT_ID"].isin(args.banks) & tx["TX_TIME_DAYS"].between(args.tu_ngay, args.den_ngay - 1)]
    tx = tx.sort_values(["TX_DATETIME", "TRANSACTION_ID"])
    p = Producer({"bootstrap.servers": args.kafka, "enable.idempotence": True, "acks": "all", "linger.ms": 5})
    t0 = time.time()
    for i, r in enumerate(tx.itertuples()):
        if args.toc_do > 0:
            wait = t0 + i / args.toc_do - time.time()
            if wait > 0:
                time.sleep(wait)
        while True:
            try:
                p.produce(TOPIC, partition=int(r.CLIENT_ID), key=str(r.CUSTOMER_ID), value=message(r))
                break
            except BufferError:
                p.poll(0.05)
        p.poll(0)
    p.flush()
    print(f"Đã phát {len(tx):,} giao dịch trong {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
