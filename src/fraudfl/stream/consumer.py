"""Consumer của một ngân hàng: đọc Kafka → tính đặc trưng → chấm điểm → ghi Postgres.

Hai chế độ dùng chung một luồng xử lý:
- real-time: mỗi lần xử lý 1 giao dịch
- theo-lo:   gom tối đa `--lo` giao dịch rồi chấm một lượt

Bảo đảm dữ liệu (at-least-once + idempotent):
1. Giao dịch và cảnh báo ghi trong CÙNG một transaction Postgres (Atomicity).
2. Chỉ commit offset Kafka SAU khi Postgres commit. Sập giữa chừng thì đọc lại từ offset cũ.
3. Đọc lại không sinh trùng: khóa chính tx_id ở Postgres, thành viên tx_id ở Redis.
"""

import argparse
import json
import os
import time

import numpy as np
import psycopg
from confluent_kafka import Consumer, TopicPartition

from fraudfl.stream.online_features import Tx, as_vector, features, remember
from fraudfl.stream.producer import TOPIC
from fraudfl.stream.redis_store import RedisStore
from fraudfl.stream.scorer import Scorer

SQL_TX = """INSERT INTO giao_dich (tx_id, ngan_hang, khach, terminal, so_tien, thoi_diem, diem, nhan_that,
                                   do_tre_ms, che_do, kafka_vi_tri)
            VALUES (%s, %s, %s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s) ON CONFLICT (tx_id) DO NOTHING"""
SQL_ALERT = "INSERT INTO canh_bao (tx_id, ngan_hang, diem, nguong) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING"


def process(msgs, store, scorer, db, mode):
    """Xử lý một lô tin nhắn. Trả về số giao dịch đã xử lý."""
    txs, vectors = [], []
    for m in msgs:
        d = json.loads(m.value())
        t = Tx(d["tx_id"], d["bank"], d["customer"], d["terminal"], d["amount"], d["ts"], d["fraud"])
        remember(store, t)
        txs.append((t, d["sent"], m.offset()))
        vectors.append(as_vector(features(store, t)))
    scores = scorer.score(np.asarray(vectors, dtype=np.float64))
    now = time.time()
    with db.transaction(), db.cursor() as cur:
        for (t, sent, offset), s in zip(txs, scores):
            cur.execute(SQL_TX, (t.tx_id, t.bank, t.customer, t.terminal, t.amount, t.ts, float(s), t.fraud,
                                 (now - sent) * 1000, mode, offset))
            if s >= scorer.threshold:
                cur.execute(SQL_ALERT, (t.tx_id, t.bank, float(s), scorer.threshold))
    return len(txs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, required=True)
    ap.add_argument("--che-do", choices=["real-time", "theo-lo"], default="real-time")
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--kafka", default="localhost:29092")
    ap.add_argument("--redis", default="redis://localhost:6379/0")
    ap.add_argument("--postgres", default="postgresql://nckh:nckh@localhost:55432/nckh")
    ap.add_argument("--mo-hinh", default="artifacts/tach_khong_dp/model.pt")
    ap.add_argument("--du-lieu", default="data/banks")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dung-sau-rong", type=float, default=15.0, help="dừng nếu không có tin mới sau N giây, 0 = chạy mãi")
    ap.add_argument("--sap-sau", type=int, default=0, help="kiểm thử: tự thoát đột ngột sau N giao dịch")
    args = ap.parse_args()

    scorer = Scorer(args.mo_hinh, args.du_lieu, args.bank, args.fold)
    store = RedisStore(args.redis)
    c = Consumer({"bootstrap.servers": args.kafka, "group.id": f"ngan-hang-{args.bank}",
                  "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    c.assign([TopicPartition(TOPIC, args.bank, -1001)])   # -1001: tiếp tục từ offset đã commit
    size = 1 if args.che_do == "real-time" else args.lo
    done, idle_since = 0, time.time()
    with psycopg.connect(args.postgres, autocommit=True) as db:
        while True:
            msgs = [m for m in c.consume(size, timeout=1.0) if m.error() is None]
            if not msgs:
                if args.dung_sau_rong and time.time() - idle_since > args.dung_sau_rong:
                    break
                continue
            idle_since = time.time()
            done += process(msgs, store, scorer, db, args.che_do)
            if args.sap_sau and done >= args.sap_sau:
                os._exit(9)      # kịch bản xấu nhất: Postgres đã commit nhưng Kafka chưa biết
            c.commit(offsets=[TopicPartition(TOPIC, args.bank, msgs[-1].offset() + 1)], asynchronous=False)
    c.close()
    print(f"Ngân hàng {args.bank}: đã xử lý {done:,} giao dịch ({args.che_do})")


if __name__ == "__main__":
    main()
