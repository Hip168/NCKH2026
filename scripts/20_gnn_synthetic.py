"""E1 / E2 / E3 trên dữ liệu synthetic với GraphSAGE + MLP.

- E1 centralized: gộp mọi ngân hàng, đồ thị có cạnh terminal xuyên ngân hàng (cận trên)
- E2 local:       mỗi ngân hàng tự train trên đồ thị riêng (cận dưới)
- E3 fedavg:      FedAvg 5 client, không DP

Siêu tham số cố định theo Slide 6, trừ: full-batch thay cho batch 64 và lr = 5e-3 thay cho 1e-3
(xem docs/decisions.md D10). Val (inner fold 0) dùng cho dừng sớm và chọn ngưỡng F1.

Chạy:  python scripts/20_gnn_synthetic.py [--folds 0] [--settings centralized local fedavg]
Ra:    results/gnn_synthetic/per_fold.csv, summary.csv, fedavg_history.csv, log.txt
"""

import argparse
import time

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch

from fraudfl.config import N_OUTER_FOLDS, RESULTS, SEED
from fraudfl.data.prep import fold_masks, load_synthetic, scale_per_client
from fraudfl.eval.metrics import MAIN_METRICS, best_f1_threshold
from fraudfl.experiment import make_data, scores, test_metrics
from fraudfl.graph import build_edge_index
from fraudfl.models import SageMLP
from fraudfl.train import fedavg, predict, train_full_batch

LR = 5e-3
MAX_EPOCHS = 300
PATIENCE = 30
ROUNDS = 50                 # 30 vòng chưa hội tụ trên fold 0 (val PR-AUC vẫn tăng) → dùng bản dài 50 vòng
LOCAL_EPOCHS = 3

OUT = RESULTS / "gnn_synthetic"


def log(msg: str):
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def report(tx, masks, score, thr_per_row, fold, setting, extra=None):
    res = test_metrics(tx, masks, score, thr_per_row)
    row = {"fold": fold, "setting": setting, **res, **(extra or {})}
    log(f"  => {setting:12s} PR-AUC {res['pr_auc']:.4f}  Recall@1% {res['recall@1%']:.4f}  "
        f"VDR@1% {res['vdr@1%']:.4f}  F1 {res['f1']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_OUTER_FOLDS)))
    ap.add_argument("--settings", nargs="+", default=["centralized", "local", "fedavg"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "log.txt").write_text("", encoding="utf-8")

    tx = load_synthetic()
    y = tx["TX_FRAUD"].to_numpy()
    clients = tx["CLIENT_ID"].to_numpy()
    t0 = time.time()
    ei_central = build_edge_index(tx, cross_client=True)
    ei_local = build_edge_index(tx, cross_client=False)
    log(f"Đồ thị: {len(tx):,} nút; cạnh centralized {ei_central.shape[1]:,}, local {ei_local.shape[1]:,} "
        f"({time.time() - t0:.1f}s)")

    rows, history = [], []
    for k in args.folds:
        masks = fold_masks(tx, k)
        x = scale_per_client(tx, masks["fit"])
        per_client = [make_data(x, y, masks, ei_local, np.flatnonzero(clients == c)) for c in np.unique(clients)]
        log(f"\n=== Outer fold {k} ===")

        if "centralized" in args.settings:
            t0 = time.time()
            torch.manual_seed(SEED)
            d = make_data(x, y, masks, ei_central)
            model = train_full_batch(SageMLP(x.shape[1]), d, LR, MAX_EPOCHS, PATIENCE)
            score = scores(model, [d], len(tx))
            thr = best_f1_threshold(y[masks["val"]], score[masks["val"]])
            rows.append(report(tx, masks, score, np.full(len(tx), thr), k, "centralized",
                               {"seconds": time.time() - t0}))

        if "local" in args.settings:
            t0 = time.time()
            score, thr = np.full(len(tx), np.nan), np.zeros(len(tx))
            for c, d in enumerate(per_client):
                torch.manual_seed(SEED)
                model = train_full_batch(SageMLP(x.shape[1]), d, LR, MAX_EPOCHS, PATIENCE)
                s = predict(model, d)
                score[d.node_ids] = s
                v = d.val.numpy()
                thr[d.node_ids] = best_f1_threshold(d.y.numpy()[v], s[v])
            rows.append(report(tx, masks, score, thr, k, "local", {"seconds": time.time() - t0}))

        if "fedavg" in args.settings:
            t0 = time.time()
            torch.manual_seed(SEED)
            model, hist = fedavg(SageMLP(x.shape[1]), per_client, LR, ROUNDS, LOCAL_EPOCHS)
            history += [{"fold": k, **h} for h in hist]
            score = scores(model, per_client, len(tx))
            thr = best_f1_threshold(y[masks["val"]], score[masks["val"]])
            best_round = max(hist, key=lambda h: h["val_pr_auc"])["round"]
            rows.append(report(tx, masks, score, np.full(len(tx), thr), k, "fedavg",
                               {"seconds": time.time() - t0, "best_round": best_round}))

        pd.DataFrame(rows).to_csv(OUT / "per_fold.csv", index=False)
        if history:
            pd.DataFrame(history).to_csv(OUT / "fedavg_history.csv", index=False)

    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby("setting")[MAIN_METRICS].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")
    log("\n=== Trung bình qua các fold ===")
    log(per_fold.groupby("setting")[MAIN_METRICS].mean().round(4).to_string())


if __name__ == "__main__":
    main()
