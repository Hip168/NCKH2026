import numpy as np
import pytest

from fraudfl.eval.metrics import best_f1_threshold, evaluate, top_frac_daily


def test_top_frac_daily_hand_example():
    # 2 ngày, mỗi ngày gắn cờ ceil(0.4 × n): ngày 0 (2 giao dịch) → 1, ngày 1 (3 giao dịch) → 2
    y = [1, 0, 1, 0, 1]
    s = [0.9, 0.1, 0.2, 0.8, 0.3]
    a = [100, 5, 50, 5, 50]
    d = [0, 0, 1, 1, 1]
    r = top_frac_daily(y, s, a, d, frac=0.4)
    # gắn cờ: ngày 0 → tx0 (gian lận 100); ngày 1 → tx3 (không), tx4 (gian lận 50)
    assert r["recall@40%"] == pytest.approx(2 / 3)
    assert r["precision@40%"] == pytest.approx(2 / 3)
    assert r["vdr@40%"] == pytest.approx(150 / 200)


def test_perfect_scores():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=10000) < 0.005).astype(int)
    res = evaluate(y, y.astype(float), rng.uniform(1, 100, 10000), rng.integers(0, 10, 10000), threshold=0.5)
    assert res["pr_auc"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)
    assert res["recall@1%"] == pytest.approx(1.0)
    assert res["vdr@1%"] == pytest.approx(1.0)


def test_best_f1_threshold_separates():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    assert 0.3 < best_f1_threshold(y, s) <= 0.7


def _old_top_frac_daily(y, s, a, d, frac):
    """Bản không trọng số trước khi thêm `weight` — dùng để kiểm tra kết quả không đổi."""
    import pandas as pd
    df = pd.DataFrame({"y": y, "s": s, "a": np.asarray(a, dtype=float), "d": d})
    rank = df.groupby("d")["s"].rank(method="first", ascending=False)
    flagged = rank <= np.ceil(frac * df.groupby("d")["s"].transform("size"))
    fraud = df["y"] == 1
    return ((flagged & fraud).sum() / fraud.sum(), (flagged & fraud).sum() / flagged.sum(),
            df.loc[flagged & fraud, "a"].sum() / df.loc[fraud, "a"].sum())


def test_no_weight_matches_previous_definition():
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.uniform(size=n) < 0.02).astype(int)
    s = rng.uniform(size=n) + y * 0.5                      # không có điểm hòa → phải khớp định nghĩa cũ
    a, d = rng.uniform(1, 100, n), rng.integers(0, 20, n)
    r = top_frac_daily(y, s, a, d, 0.01)
    assert (r["recall@1%"], r["precision@1%"], r["vdr@1%"]) == pytest.approx(_old_top_frac_daily(y, s, a, d, 0.01))
    r1 = top_frac_daily(y, s, a, d, 0.01, weight=np.ones(n))
    assert r1 == pytest.approx(r)


def test_weighted_subsample_approximates_full_data():
    # Dữ liệu đầy đủ 0,2% gian lận; bốc 5% ca bình thường, trọng số 1/0,05 = 20 → chỉ số gần bằng bản đầy đủ
    rng = np.random.default_rng(2)
    n = 400_000
    y = (rng.uniform(size=n) < 0.002).astype(int)
    s = rng.normal(size=n) + 2.5 * y
    a, d = rng.uniform(1, 100, n), rng.integers(0, 30, n)
    full = evaluate(y, s, a, d)
    r = 0.05
    keep = (y == 1) | (rng.uniform(size=n) < r)
    w = np.where(y[keep] == 1, 1.0, 1 / r)
    sub = evaluate(y[keep], s[keep], a[keep], d[keep], weight=w)
    naive = evaluate(y[keep], s[keep], a[keep], d[keep])
    assert sub["pr_auc"] == pytest.approx(full["pr_auc"], abs=0.03)
    assert sub["recall@1%"] == pytest.approx(full["recall@1%"], abs=0.03)
    assert sub["fraud_rate"] == pytest.approx(full["fraud_rate"], rel=0.1)
    assert naive["pr_auc"] > full["pr_auc"] + 0.2          # không trọng số thì PR-AUC cao giả


def test_ties_do_not_depend_on_row_order():
    # Mô hình "sụp": mọi điểm bằng nhau, ca gian lận nằm hết ở đầu file → Recall@1% phải ≈ 1% (ngẫu nhiên),
    # không được ≈ 100% như khi phá hòa theo thứ tự dòng (#41)
    n = 100_000
    y = np.r_[np.ones(500, int), np.zeros(n - 500, int)]
    r = top_frac_daily(y, np.zeros(n), np.ones(n), np.zeros(n, int), 0.01)
    assert r["recall@1%"] < 0.05
