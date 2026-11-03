import numpy as np
import pytest

from fraudfl.data.sampling import subsample_negatives


def _data(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    labels = (rng.uniform(size=n) < 0.01).astype(int)
    groups = rng.choice([3, 7, 11], size=n, p=[0.6, 0.3, 0.1])
    return np.arange(n) * 10, labels, groups


def test_keeps_all_positives_and_rate_per_group():
    ids, labels, groups = _data()
    s = subsample_negatives(ids, labels, groups, rate=0.1, seed=42)
    assert s["LABEL"].sum() == labels.sum()                       # giữ 100% ca gian lận
    for g in (3, 7, 11):
        n_neg = ((groups == g) & (labels == 0)).sum()
        kept = s[(s["GROUP"] == g) & (s["LABEL"] == 0)]
        assert len(kept) == round(0.1 * n_neg)                    # đúng tỷ lệ trong từng ngân hàng
        assert kept["WEIGHT"].sum() == pytest.approx(n_neg)        # trọng số bù đủ số ca bình thường gốc
    assert (s.loc[s["LABEL"] == 1, "WEIGHT"] == 1).all()
    assert set(s["TX_ID"]) <= set(ids)


def test_same_seed_same_sample_and_different_seed_differs():
    ids, labels, groups = _data()
    a = subsample_negatives(ids, labels, groups, 0.05, seed=1)
    b = subsample_negatives(ids, labels, groups, 0.05, seed=1)
    c = subsample_negatives(ids, labels, groups, 0.05, seed=2)
    assert a.equals(b)
    assert not a["TX_ID"].equals(c["TX_ID"])


def test_tiny_group_keeps_one_negative():
    s = subsample_negatives(np.arange(4), np.array([0, 0, 0, 1]), np.array([5, 5, 5, 5]), rate=0.01, seed=0)
    assert (s["LABEL"] == 0).sum() == 1 and s.loc[s["LABEL"] == 0, "WEIGHT"].item() == 3
