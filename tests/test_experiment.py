import numpy as np
import pandas as pd
import pytest

from fraudfl.experiment import per_client_metrics


def test_per_client_metrics_not_affected_by_score_scale_of_other_bank():
    # Hai ngân hàng, mỗi bên tự xếp hạng hoàn hảo nhưng thang điểm khác nhau → chỉ số từng bên = 1
    rng = np.random.default_rng(0)
    n = 4000
    tx = pd.DataFrame({"CLIENT_ID": np.repeat([0, 1], n // 2), "TX_FRAUD": (rng.uniform(size=n) < 0.05).astype(int),
                       "TX_AMOUNT": rng.uniform(1, 50, n), "TX_TIME_DAYS": rng.integers(0, 5, n)})
    score = tx["TX_FRAUD"].to_numpy() * 0.1 + np.where(tx["CLIENT_ID"] == 0, 0.0, 0.85)   # bank 1 điểm cao hơn hẳn
    masks = {"test": np.ones(n, dtype=bool)}
    r = per_client_metrics(tx, masks, score)
    assert r["c0_pr_auc"] == pytest.approx(1.0) and r["c1_pr_auc"] == pytest.approx(1.0)
    assert r["macro_pr_auc"] == pytest.approx(1.0)
