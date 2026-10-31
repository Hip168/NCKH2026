"""Baseline dạng bảng (Logistic Regression, Random Forest) để KIỂM TRA PIPELINE trước khi làm GNN/FL.

Chạy 2 chế độ trên 5 outer fold:
- centralized: gộp dữ liệu mọi ngân hàng → cận trên (tương tự E1)
- local:       mỗi ngân hàng tự train trên dữ liệu của mình → cận dưới (tương tự E2)
và 2 phiên bản đặc trưng (để thấy tác dụng của quyết định D8):
- v0: đặc trưng gốc, không chuẩn hóa riêng từng client
- v1: thêm TX_AMOUNT_RATIO_30DAY + chuẩn hóa riêng từng client
Ngưỡng F1 chọn trên phần validation (inner fold 0) của tập train, không chọn trên test.

Chạy:  python scripts/10_baseline_tabular.py [--folds 0 1] [--models lr rf] [--variants v0 v1]
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

from fraudfl.config import N_OUTER_FOLDS, RESULTS, SEED
from fraudfl.data.features import FEATURES
from fraudfl.data.prep import fold_masks, load_synthetic, scale_per_client
from fraudfl.eval.metrics import MAIN_METRICS, best_f1_threshold, evaluate

MODELS = {
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED,
                                         class_weight="balanced_subsample"),
}
FEATURES_V0 = [f for f in FEATURES if f != "TX_AMOUNT_RATIO_30DAY"]


def fit_predict(model_name, X, y, fit, val, test):
    """Train trên fit, chọn ngưỡng F1 trên val, trả về (điểm trên test, ngưỡng)."""
    m = MODELS[model_name]().fit(X[fit], y[fit])
    thr = best_f1_threshold(y[val], m.predict_proba(X[val])[:, 1])
    return m.predict_proba(X[test])[:, 1], thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_OUTER_FOLDS)))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--variants", nargs="+", default=["v0", "v1"])
    args = ap.parse_args()

    tx = load_synthetic()
    y = tx["TX_FRAUD"].to_numpy()
    clients = tx["CLIENT_ID"].to_numpy()

    rows = []
    for k in args.folds:
        mk = fold_masks(tx, k)
        test = tx[mk["test"]]
        for variant in args.variants:
            X = (tx[FEATURES_V0].to_numpy(np.float32) if variant == "v0"
                 else scale_per_client(tx, mk["fit"]))
            for name in args.models:
                score, thr = fit_predict(name, X, y, mk["fit"], mk["val"], mk["test"])
                rows.append({"fold": k, "variant": variant, "model": name, "setting": "centralized",
                             **evaluate(test["TX_FRAUD"], score, test["TX_AMOUNT"], test["TX_TIME_DAYS"], thr)})

                # local-only: mỗi ngân hàng có model + ngưỡng riêng; ghép điểm lại để so trên cùng tập test
                score = np.zeros(len(test))
                thr_test = np.zeros(len(test))
                for c in np.unique(clients):
                    own = clients == c
                    s, t = fit_predict(name, X, y, mk["fit"] & own, mk["val"] & own, mk["test"] & own)
                    te = clients[mk["test"]] == c
                    score[te], thr_test[te] = s, t
                res = evaluate(test["TX_FRAUD"], score, test["TX_AMOUNT"], test["TX_TIME_DAYS"])
                res["f1"] = float(f1_score(test["TX_FRAUD"], score >= thr_test, zero_division=0))
                res["threshold"] = np.nan
                rows.append({"fold": k, "variant": variant, "model": name, "setting": "local", **res})
                print(pd.DataFrame(rows[-2:])[["fold", "variant", "model", "setting", "pr_auc", "recall@1%",
                                               "vdr@1%"]].round(4).to_string(index=False, header=False))

    out = RESULTS / "baseline_tabular"
    out.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out / "per_fold.csv", index=False)
    summary = per_fold.groupby(["variant", "model", "setting"])[MAIN_METRICS].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    print("\n=== Trung bình qua các fold ===")
    print(per_fold.groupby(["variant", "model", "setting"])[MAIN_METRICS].mean().round(4).to_string())


if __name__ == "__main__":
    main()
