import numpy as np
import pandas as pd
import pytest

from fraudfl.data.ibm_features import build_ibm_features, feature_columns, usd_rates


def _df(rows):
    cols = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Received",
            "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
    df = pd.DataFrame(rows, columns=cols)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["TX_TIME_DAYS"] = (df["Timestamp"] - df["Timestamp"].min().normalize()).dt.days
    return df


ROWS = [
    # A (ngân hàng 010) gửi B; B gửi C; A gửi lại B; đổi tiền USD → Euro để suy tỷ giá
    ["2022-09-01 00:00", "010", "A", "020", "B", 100.0, "US Dollar", 100.0, "US Dollar", "ACH", 0],
    ["2022-09-01 05:00", "020", "B", "030", "C", 85.0, "Euro", 100.0, "US Dollar", "Wire", 1],
    ["2022-09-01 05:00", "010", "A", "020", "B", 50.0, "US Dollar", 50.0, "US Dollar", "ACH", 0],
    ["2022-09-03 00:00", "010", "A", "010", "A", 42.5, "Euro", 42.5, "Euro", "Cash", 0],
    ["2022-09-09 00:00", "010", "A", "020", "B", 10.0, "US Dollar", 10.0, "US Dollar", "Cheque", 0],
]


def test_usd_rates_from_data():
    r = usd_rates(_df(ROWS))
    assert r["Euro"] == pytest.approx(0.85) and r["US Dollar"] == 1.0


def test_windows_use_only_strict_past():
    f = build_ibm_features(_df(ROWS))
    # Giao dịch đầu của A: chưa có lịch sử
    assert f.loc[0, "SND_OUT_NB_LOG_1D"] == 0 and f.loc[0, "SND_AGE_DAYS"] == 0
    # Dòng 2 (A lúc 05:00): thấy giao dịch 00:00, KHÔNG thấy giao dịch cùng phút 05:00 của B
    assert f.loc[2, "SND_OUT_NB_LOG_1D"] == pytest.approx(np.log1p(1))
    # Dòng 1 (B gửi lúc 05:00): B đã nhận 1 lần trước đó (từ A lúc 00:00), lần nhận cùng phút không tính
    assert f.loc[1, "SND_IN_NB_LOG_1D"] == pytest.approx(np.log1p(1))
    # Dòng 3 (A ngày 3, 42,5 Euro = 50 USD): cửa sổ 1 ngày trống, cửa sổ 7 ngày có 2 giao dịch 100 + 50
    assert f.loc[3, "SND_OUT_NB_LOG_1D"] == 0 and f.loc[3, "SND_OUT_NB_LOG_7D"] == pytest.approx(np.log1p(2))
    assert f.loc[3, "AMOUNT_USD"] == pytest.approx(50.0)
    assert f.loc[3, "SND_AMOUNT_LOGRATIO_7D"] == pytest.approx(np.log1p(50) - np.log1p(75))
    assert f.loc[3, "SELF_TRANSFER"] == 1 and f.loc[3, "SAME_BANK"] == 1
    # Dòng 4 (ngày 9): giao dịch ngày 1 đã ra khỏi cửa sổ 7 ngày, còn giao dịch ngày 3
    assert f.loc[4, "SND_OUT_NB_LOG_7D"] == pytest.approx(np.log1p(1))
    assert f.loc[1, "CROSS_CURRENCY"] == 1 and f.loc[1, "PF_WIRE"] == 1


def test_bank_view_drops_receiver_features_and_no_label_in_features():
    g, b = build_ibm_features(_df(ROWS), "global"), build_ibm_features(_df(ROWS), "bank")
    assert any(c.startswith("RCV_") for c in feature_columns(g))
    assert not any(c.startswith("RCV_") for c in feature_columns(b))
    assert "LABEL" not in feature_columns(g)
    assert np.isfinite(g[feature_columns(g)].to_numpy(float)).all()
