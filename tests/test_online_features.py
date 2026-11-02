import numpy as np
import pandas as pd
import pytest

from fraudfl.config import BankProfile, SimulationConfig
from fraudfl.data.features import FEATURES, build_features
from fraudfl.data.simulate import generate
from fraudfl.stream.online_features import MemoryStore, Tx, as_vector, features, remember


@pytest.fixture(scope="module")
def tx():
    cfg = SimulationConfig(n_days=60, n_terminals=300, banks=[
        BankProfile("a", (50, 50), 20, (5, 60), (1, 3), target_n_tx=3000),
        BankProfile("b", (50, 50), 20, (5, 60), (1, 3), target_n_tx=3000),
    ])
    return build_features(generate(cfg)[2])


def to_tx(r) -> Tx:
    ts = int(pd.Timestamp(r.TX_DATETIME).timestamp())
    return Tx(int(r.TRANSACTION_ID), int(r.CLIENT_ID), int(r.CUSTOMER_ID), int(r.TERMINAL_ID), float(r.TX_AMOUNT),
              ts, int(r.TX_FRAUD))


def stream(tx):
    store = MemoryStore()
    rows = []
    for r in tx.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).itertuples():
        t = to_tx(r)
        remember(store, t)
        rows.append((r.Index, as_vector(features(store, t))))
    return rows


def ties(tx):
    """Giao dịch trùng giây trong cùng khách hoặc terminal: bản lô thấy cả giao dịch đến sau, real-time thì không."""
    d = pd.Series(False, index=tx.index)
    for key in (["CLIENT_ID", "CUSTOMER_ID", "TX_DATETIME"], ["CLIENT_ID", "TERMINAL_ID", "TX_DATETIME"]):
        d |= tx.duplicated(key, keep=False)
    return d


def test_streaming_matches_batch_on_every_transaction(tx):
    tie = ties(tx)
    batch = tx[FEATURES].to_numpy(float)
    bad = [(i, FEATURES[j]) for i, v in stream(tx) if not tie[i]
           for j in np.flatnonzero(~np.isclose(v, batch[i], rtol=1e-9, atol=1e-9))]
    assert not bad, f"{len(bad)} ô lệch, ví dụ {bad[:5]}"


def test_replaying_same_transaction_does_not_double_count(tx):
    store = MemoryStore()
    rows = list(tx.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).head(500).itertuples())
    for r in rows:
        remember(store, to_tx(r))
    last = to_tx(rows[-1])
    before = features(store, last)
    for r in rows:
        remember(store, to_tx(r))
    assert features(store, last) == before


def test_future_label_is_not_visible():
    store = MemoryStore()
    now = 100 * 86_400
    recent = Tx(1, 0, 1, 9, 10.0, now - 3 * 86_400, 1)     # gian lận mới 3 ngày: nhãn chưa được biết
    current = Tx(2, 0, 1, 9, 10.0, now, 0)
    remember(store, recent)
    remember(store, current)
    row = features(store, current)
    assert row["TERMINAL_ID_RISK_30DAY_WINDOW"] == 0.0 and row["TERMINAL_ID_NB_TX_30DAY_WINDOW"] == 0
