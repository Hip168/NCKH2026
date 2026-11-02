import numpy as np
import torch

from fraudfl.data.features import FEATURES
from fraudfl.fl.partition import Partition, fold_path, load_partition, save_partition, scaler
from fraudfl.fl.task import evaluate_split, local_train, new_model, train_plan


def _masks(n):
    fit = np.zeros(n, bool)
    fit[: n // 2] = True
    val = np.zeros(n, bool)
    val[n // 2: 3 * n // 4] = True
    return {"fit": fit, "val": val, "test": ~(fit | val)}


def _partition(n=400):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    y = (x[:, 0] > 1).astype(np.float32)
    m = _masks(n)
    return Partition(torch.from_numpy(x), torch.from_numpy(y), m["fit"], m["val"], m["test"])


def test_no_dp_means_no_noise():
    assert train_plan(4000, 0.0, 30)[2] == 0.0
    assert train_plan(4000, float("inf"), 30)[2] == 0.0


def test_smaller_epsilon_needs_more_noise():
    assert train_plan(40000, 0.5, 30)[2] > train_plan(40000, 2.0, 30)[2] > 0


def test_local_train_changes_weights_and_is_repeatable():
    p = _partition()
    a, b = new_model(p.x.shape[1]), new_model(p.x.shape[1])
    start = {k: v.clone() for k, v in a.state_dict().items()}
    local_train(a, p, 1.0, 3, seed=7)
    local_train(b, p, 1.0, 3, seed=7)
    assert any(not torch.equal(start[k], v) for k, v in a.state_dict().items())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_evaluate_reports_val_size():
    p = _partition()
    r = evaluate_split(new_model(p.x.shape[1]), p, "val")
    assert r["num-examples"] == int(p.val.sum()) and 0 <= r["pr-auc"] <= 1


def test_partition_round_trip(tmp_path):
    n = 20
    x, y = np.ones((n, 3)), np.zeros(n)
    path = fold_path(tmp_path, "bank_0", 0)
    save_partition(path, x, y, _masks(n), np.arange(3.0), np.ones(3))
    p = load_partition(path)
    mu, sd = scaler(path)
    assert p.x.shape == (n, 3) and p.n_fit == n // 2 and mu.tolist() == [0, 1, 2] and sd.tolist() == [1, 1, 1]
