"""Kiểm tra luồng xử lý consumer: đọc lại cùng tin nhắn không làm sai trạng thái đặc trưng."""

import json

import numpy as np

from fraudfl.stream.consumer import process
from fraudfl.stream.online_features import MemoryStore


class Msg:
    def __init__(self, tx_id, offset):
        self.offset_ = offset
        self.body = json.dumps({"tx_id": tx_id, "bank": 0, "customer": 1, "terminal": 2, "amount": 10.0,
                                "ts": 10_000_000 + tx_id, "fraud": 0, "sent": 0.0}).encode()

    def value(self):
        return self.body

    def offset(self):
        return self.offset_


class Scorer:
    threshold = 0.5

    def __init__(self, s):
        self.s = s

    def score(self, rows):
        return np.full(len(rows), self.s)


class Cursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, values):
        self.log.append((sql.split()[2], values[0]))


class Tx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Db:
    def __init__(self):
        self.log = []

    def transaction(self):
        return Tx()

    def cursor(self):
        return Cursor(self.log)


def test_alert_written_only_above_threshold():
    db = Db()
    process([Msg(1, 0)], MemoryStore(), Scorer(0.1), db, "real-time")
    assert [t for t, _ in db.log] == ["giao_dich"]
    db = Db()
    process([Msg(1, 0)], MemoryStore(), Scorer(0.9), db, "real-time")
    assert [t for t, _ in db.log] == ["giao_dich", "canh_bao"]


def test_reprocessing_keeps_feature_state_unique():
    store = MemoryStore()
    for _ in range(3):
        process([Msg(1, 0), Msg(2, 1)], store, Scorer(0.1), Db(), "theo-lo")
    assert len(store.d["b0:c1"]) == 2
