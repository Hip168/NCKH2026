import numpy as np
import pytest

from fraudfl.eval.metrics import best_f1_threshold, evaluate, top_k_daily


def test_top_k_daily_hand_example():
    # 2 ngày, k=1: ngày 0 gắn cờ giao dịch điểm 0.9 (gian lận, 100đ); ngày 1 gắn cờ 0.8 (không gian lận)
    y = [1, 0, 1, 0, 1]
    s = [0.9, 0.1, 0.2, 0.8, 0.3]
    a = [100, 5, 50, 5, 50]
    d = [0, 0, 1, 1, 1]
    r = top_k_daily(y, s, a, d, k=1)
    assert r["recall@1"] == pytest.approx(1 / 3)
    assert r["precision@1"] == pytest.approx(1 / 2)
    assert r["vdr@1"] == pytest.approx(100 / 200)


def test_perfect_scores():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=1000) < 0.03).astype(int)
    res = evaluate(y, y.astype(float), rng.uniform(1, 100, 1000), rng.integers(0, 10, 1000), threshold=0.5, k=100)
    assert res["pr_auc"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)
    assert res["recall@100"] == pytest.approx(1.0)
    assert res["vdr@100"] == pytest.approx(1.0)


def test_best_f1_threshold_separates():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    assert 0.3 < best_f1_threshold(y, s) <= 0.7
