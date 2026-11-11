import numpy as np
import pandas as pd
import pytest

from fraudfl.data.dac_trung_do_thi import (
    ALL_GRAPH_FEATURES,
    GRAPH_FEATURES_FL,
    GRAPH_FEATURES_RCV,
    build_graph_features,
)


def _df(rows):
    cols = [
        "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
        "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
        "Payment Format", "Is Laundering",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["TX_TIME_DAYS"] = (df["Timestamp"] - df["Timestamp"].min().normalize()).dt.days
    return df


# Kịch bản nhỏ có chủ đích:
# 0: 00:00  Bank1:A -> Bank2:B
# 1: 00:30  Bank2:B -> Bank3:C   (B vừa nhận từ A ở Bank1, gửi tiếp cho C ở Bank3 trong 1h)
# 2: 02:00  Bank3:C -> Bank1:A   (C gửi cho A; nên ở bước sau B->A sẽ là chu trình hoặc C->A)
# 3: 02:00  Bank1:A -> Bank2:B   (cùng phút với dòng 2: KHÔNG được thấy dòng 2)
# 4: 03:00  Bank2:B -> Bank1:A   (chu trình A->B->A trong 7 ngày! A từng gửi cho B ở dòng 0 và 3)
# 5: 10d    Bank1:A -> Bank2:B   (sau 10 ngày: cửa sổ 1d và 7d đã hết, nhưng appear tổng vẫn còn)
ROWS = [
    ["2022-09-01 00:00", "1", "A", "2", "B", 100.0, "US Dollar", 100.0, "US Dollar", "ACH", 0],
    ["2022-09-01 00:30", "2", "B", "3", "C", 90.0, "US Dollar", 90.0, "US Dollar", "Wire", 1],
    ["2022-09-01 02:00", "3", "C", "1", "A", 80.0, "US Dollar", 80.0, "US Dollar", "ACH", 0],
    ["2022-09-01 02:00", "1", "A", "2", "B", 50.0, "US Dollar", 50.0, "US Dollar", "ACH", 0],
    ["2022-09-01 03:00", "2", "B", "1", "A", 40.0, "US Dollar", 40.0, "US Dollar", "Wire", 1],
    ["2022-09-11 00:00", "1", "A", "2", "B", 10.0, "US Dollar", 10.0, "US Dollar", "Cash", 0],
]


def test_strict_past_and_graph_features_global():
    df = _df(ROWS)
    g = build_graph_features(df, scope="global")
    assert list(g.columns) == ["TX_ID"] + ALL_GRAPH_FEATURES
    assert np.isfinite(g[ALL_GRAPH_FEATURES].to_numpy(float)).all()

    # Dòng 0: hoàn toàn chưa có quá khứ
    assert (g.loc[0, ALL_GRAPH_FEATURES] == 0).all()

    # Dòng 1 (00:30, B->C): B đã nhận từ A lúc 00:00 (< 1h) -> pass-through 1h = 1, appear = log1p(1)
    assert g.loc[1, "G_SND_PASS_1H"] == 1.0
    assert g.loc[1, "G_SND_FANIN_LOG_1D"] == pytest.approx(np.log1p(1))
    assert g.loc[1, "G_SND_FANOUT_LOG_1D"] == 0.0

    # Dòng 3 (02:00, A->B): cùng phút với dòng 2 (C->A lúc 02:00), KHÔNG được thấy dòng 2!
    # A mới chỉ có dòng 0 (gửi B) -> fanin=0, pass_1h=0, appear=log1p(1), pair_past=log1p(1)
    assert g.loc[3, "G_SND_FANIN_LOG_1D"] == 0.0
    assert g.loc[3, "G_SND_PASS_1H"] == 0.0
    assert g.loc[3, "G_PAIR_PAST_LOG_7D"] == pytest.approx(np.log1p(1))
    # Tài khoản nhận B ở dòng 3: đã nhận 1 (dòng 0) và đã gửi 1 (dòng 1, lúc 00:30, trong vòng 1d nhưng > 1h so với 02:00)
    assert g.loc[3, "G_RCV_PASS_1H"] == 0.0
    assert g.loc[3, "G_RCV_PASS_1D"] == 1.0
    assert g.loc[3, "G_RCV_RECENT_OUT_1D"] == 1.0

    # Dòng 4 (03:00, B->A): A từng gửi cho B ở dòng 0 và dòng 3 -> chu trình A->B (dòng 0,3) -> A (dòng 4) = 2 lần
    assert g.loc[4, "G_CYCLE_ABA_LOG_7D"] == pytest.approx(np.log1p(2))
    # A (tài khoản nhận ở dòng 4) từng nhận từ C lúc 02:00 (1h trước) và gửi B lúc 02:00 -> G_RCV_PASS_1H = 1
    assert g.loc[4, "G_RCV_PASS_1H"] == 1.0
    assert g.loc[4, "G_RCV_FANIN_LOG_1D"] == pytest.approx(np.log1p(1))

    # Dòng 5 (ngày 11): ngoài cửa sổ 7 ngày so với ngày 1 -> fanout/cycle/pair = 0, nhưng appear vẫn > 0
    assert g.loc[5, "G_SND_FANOUT_LOG_7D"] == 0.0
    assert g.loc[5, "G_CYCLE_ABA_LOG_7D"] == 0.0
    assert g.loc[5, "G_SND_APPEAR_LOG"] > 0.0
    assert g.loc[5, "G_RCV_APPEAR_LOG"] > 0.0


def test_bank_scope_hides_third_party_bank_edges():
    df = _df(ROWS)
    g_glob = build_graph_features(df, scope="global")
    g_bank = build_graph_features(df, scope="bank")

    # Nhóm FL-safe (tài khoản gửi + chu trình vào/ra tài khoản gửi) PHẢI giống hệt global
    pd.testing.assert_frame_equal(g_glob[GRAPH_FEATURES_FL], g_bank[GRAPH_FEATURES_FL])

    # Dòng 3 (Bank1:A -> Bank2:B lúc 02:00):
    # Trên global, B đã gửi cho Bank3:C lúc 00:30 -> G_RCV_RECENT_OUT_1D = 1.0, G_RCV_APPEAR_LOG = log1p(2)
    # Trên bank scope, Bank1 KHÔNG thấy giao dịch Bank2:B -> Bank3:C! Nên Bank1 chỉ thấy B nhận 1 lần (dòng 0) và chưa gửi ai
    assert g_glob.loc[3, "G_RCV_RECENT_OUT_1D"] == 1.0
    assert g_bank.loc[3, "G_RCV_RECENT_OUT_1D"] == 0.0
    assert g_bank.loc[3, "G_RCV_FANOUT_LOG_1D"] == 0.0
    assert g_bank.loc[3, "G_RCV_APPEAR_LOG"] == pytest.approx(np.log1p(1))


def test_country_scope_shares_within_same_country():
    df = _df(ROWS)
    # Giả sử Bank 1 và Bank 2 cùng ở "USA", Bank 3 ở "Germany"
    b2c = {"1": "USA", "2": "USA", "3": "Germany"}
    g_glob = build_graph_features(df, scope="global")
    g_cnt = build_graph_features(df, scope="country", bank_to_country=b2c)

    pd.testing.assert_frame_equal(g_glob[GRAPH_FEATURES_FL], g_cnt[GRAPH_FEATURES_FL])
    # Dòng 3 (USA:A -> USA:B): vì Bank2:B -> Germany:C có đầu gửi ở USA, nên USA thấy được giao dịch đó!
    assert g_cnt.loc[3, "G_RCV_RECENT_OUT_1D"] == 1.0
    assert g_cnt.loc[3, "G_RCV_FANOUT_LOG_1D"] == pytest.approx(np.log1p(1))


def test_row_order_invariance():
    df = _df(ROWS)
    g_orig = build_graph_features(df, scope="global")
    perm = [4, 1, 5, 0, 3, 2]
    df_shuf = df.iloc[perm].reset_index(drop=True)
    g_shuf = build_graph_features(df_shuf, scope="global")

    for i, orig_i in enumerate(perm):
        np.testing.assert_allclose(
            g_shuf.loc[i, ALL_GRAPH_FEATURES].to_numpy(float),
            g_orig.loc[orig_i, ALL_GRAPH_FEATURES].to_numpy(float),
            rtol=1e-6,
            atol=1e-6,
        )
