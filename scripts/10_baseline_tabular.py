"""Baseline dạng bảng (Logistic Regression, Random Forest) để KIỂM TRA PIPELINE trước khi làm GNN/FL.

Chạy 2 chế độ trên 5 outer fold:
- centralized: gộp dữ liệu mọi ngân hàng → cận trên (tương tự E1)
- local:       mỗi ngân hàng tự train trên dữ liệu của mình → cận dưới (tương tự E2)
Ngưỡng F1 chọn trên phần validation (inner fold 0) của tập train, không chọn trên test.

Chạy:  python scripts/10_baseline_tabular.py [--folds 0 1] [--models lr rf]
Ra:    results/baseline_tabular/per_fold.csv, summary.csv
"""

import argparse
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudfl.config import DATA_PROCESSED, DATA_SPLITS, N_OUTER_FOLDS, RESULTS, SEED
from fraudfl.data.features import FEATURES
from fraudfl.eval.metrics import best_f1_threshold, evaluate

MODELS = {
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED,
                                         class_weight="balanced_subsample"),
}


def fit_predict(model_name, train, test):
    """Train trên train (trừ inner fold 0), chọn ngưỡng trên inner fold 0, dự đoán test."""
    fit, val = train[train["inner"] != 0], train[train["inner"] == 0]
    m = MODELS[model_name]().fit(fit[FEATURES], fit["TX_FRAUD"])
    thr = best_f1_threshold(val["TX_FRAUD"], m.predict_proba(val[FEATURES])[:, 1])
    return m.predict_proba(test[FEATURES])[:, 1], thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_OUTER_FOLDS)))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()

    tx = pd.read_parquet(DATA_PROCESSED / "synthetic_tx.parquet")
    folds = pd.read_csv(DATA_SPLITS / "fold_assignments_synthetic.csv").drop(columns="CLIENT_ID")
    tx = tx.merge(folds, on="TRANSACTION_ID")

    rows = []
    for k in args.folds:
        tx["inner"] = tx[f"inner_fold_o{k}"]
        train, test = tx[tx["outer_fold"] != k], tx[tx["outer_fold"] == k]
        for name in args.models:
            # centralized
            score, thr = fit_predict(name, train, test)
            rows.append({"fold": k, "model": name, "setting": "centralized",
                         **evaluate(test["TX_FRAUD"], score, test["TX_AMOUNT"], test["TX_TIME_DAYS"], thr)})
            # local-only: mỗi ngân hàng có model riêng; ghép điểm lại để so trên cùng tập test
            score = np.zeros(len(test))
            thr_test = np.zeros(len(test))
            for c in sorted(tx["CLIENT_ID"].unique()):
                tr_c, te_mask = train[train["CLIENT_ID"] == c], (test["CLIENT_ID"] == c).to_numpy()
                s, t = fit_predict(name, tr_c, test[te_mask])
                score[te_mask], thr_test[te_mask] = s, t
            # mỗi ngân hàng dùng ngưỡng F1 riêng của mình
            res = evaluate(test["TX_FRAUD"], score, test["TX_AMOUNT"], test["TX_TIME_DAYS"])
            res["f1"] = float(f1_score(test["TX_FRAUD"], score >= thr_test, zero_division=0))
            res["threshold"] = np.nan
            rows.append({"fold": k, "model": name, "setting": "local", **res})
            print(pd.DataFrame(rows[-2:]).drop(columns=["threshold"]).round(4).to_string(index=False))

    out = RESULTS / "baseline_tabular"
    out.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out / "per_fold.csv", index=False)
    metrics = ["pr_auc", "roc_auc", "f1", "recall@100", "precision@100", "vdr@100"]
    summary = per_fold.groupby(["model", "setting"])[metrics].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    print("\n=== Trung bình ± độ lệch chuẩn qua các fold ===")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
