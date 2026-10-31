"""So sánh đặc trưng vector hóa với cách tính "thủ công" từng giao dịch trên một tập dữ liệu nhỏ."""

import numpy as np
import pandas as pd
import pytest

from fraudfl.config import BankProfile, SimulationConfig
from fraudfl.data.features import build_features
from fraudfl.data.simulate import generate


@pytest.fixture(scope="module")
def feats():
    cfg = SimulationConfig(n_days=40, n_terminals=300, banks=[
        BankProfile("a", (50, 50), 20, (5, 60), (1, 3), target_n_tx=3000),
        BankProfile("b", (40, 60), 20, (20, 90), (1, 3), target_n_tx=3000),
    ])
    _, _, tx = generate(cfg)
    return build_features(tx, delay=7)


def test_customer_window_matches_bruteforce(feats):
    rng = np.random.default_rng(0)
    for i in rng.choice(len(feats), 50, replace=False):
        row = feats.iloc[i]
        same = feats[(feats.CLIENT_ID == row.CLIENT_ID) & (feats.CUSTOMER_ID == row.CUSTOMER_ID)]
        win = same[(same.TX_DATETIME > row.TX_DATETIME - pd.Timedelta(days=7)) & (same.TX_DATETIME <= row.TX_DATETIME)]
        assert row.CUSTOMER_ID_NB_TX_7DAY_WINDOW == len(win)
        assert row.CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW == pytest.approx(win.TX_AMOUNT.mean())


def test_terminal_risk_uses_only_delayed_labels(feats):
    rng = np.random.default_rng(1)
    for i in rng.choice(len(feats), 50, replace=False):
        row = feats.iloc[i]
        same = feats[(feats.CLIENT_ID == row.CLIENT_ID) & (feats.TERMINAL_ID == row.TERMINAL_ID)]
        end = row.TX_DATETIME - pd.Timedelta(days=7)
        win = same[(same.TX_DATETIME > end - pd.Timedelta(days=7)) & (same.TX_DATETIME <= end)]
        assert row.TERMINAL_ID_NB_TX_7DAY_WINDOW == len(win)
        expected = win.TX_FRAUD.mean() if len(win) else 0.0
        assert row.TERMINAL_ID_RISK_7DAY_WINDOW == pytest.approx(expected)


def test_features_are_per_client(feats):
    # Đặc trưng ngân hàng a không được phụ thuộc dữ liệu ngân hàng b
    alone = build_features(feats[feats.CLIENT_ID == 0][feats.columns[:9]], delay=7)
    cols = ["TERMINAL_ID_RISK_30DAY_WINDOW", "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW"]
    np.testing.assert_allclose(alone[cols].to_numpy(),
                               feats[feats.CLIENT_ID == 0][cols].to_numpy())
