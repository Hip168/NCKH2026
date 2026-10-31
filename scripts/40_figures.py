"""Biểu đồ cho bài báo từ kết quả E1–E5.

Chạy:  python scripts/40_figures.py   (sau 20_gnn_synthetic.py và 30_dp_fl.py)
Ra:    results/figures/*.png
"""

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudfl.config import RESULTS

LABELS = {"centralized": "E1 Centralized", "fedavg": "E3 FedAvg", "local": "E2 Local-only"}


def main():
    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    gnn = pd.read_csv(RESULTS / "gnn_synthetic" / "per_fold.csv")
    rec = pd.read_csv(RESULTS / "dp_record" / "per_fold.csv")
    cli = pd.read_csv(RESULTS / "dp_client" / "per_fold.csv")
    hist = pd.read_csv(RESULTS / "gnn_synthetic" / "fedavg_history.csv")

    # 1. So sánh E1/E2/E3 từng fold
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.27
    for i, s in enumerate(["centralized", "fedavg", "local"]):
        d = gnn[gnn.setting == s].sort_values("fold")
        ax.bar(d.fold + (i - 1) * w, d.pr_auc, w, label=LABELS[s])
    ax.set(xlabel="Outer fold", ylabel="PR-AUC", title="GraphSAGE: centralized vs FedAvg vs local-only",
           ylim=(0.65, 0.77))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "e1_e3_pr_auc_by_fold.png", dpi=150)

    # 2. Đánh đổi riêng tư – hiệu năng
    fig, ax = plt.subplots(figsize=(7, 4))
    r = rec[np.isfinite(rec.epsilon)].groupby("epsilon").pr_auc.agg(["mean", "std"])
    ax.errorbar(r.index, r["mean"], yerr=r["std"], marker="o", capsize=4, label="FL+DP record-level (MLP, DP-SGD)")
    c = cli.groupby("epsilon").pr_auc.mean()
    ax.plot(c.index, c.values, marker="x", ls="--", label="FL+DP client-level (GNN, Slide 4)")
    for s, ls in [("centralized", ":"), ("fedavg", "-.")]:
        ax.axhline(gnn[gnn.setting == s].pr_auc.mean(), color="gray", ls=ls, label=f"{LABELS[s]} (không DP)")
    ax.axhline(rec[~np.isfinite(rec.epsilon)].pr_auc.mean(), color="C0", ls=":", alpha=0.6,
               label="FedAvg MLP (không DP)")
    ax.set(xscale="log", xlabel="ε (δ = 1e-5)", ylabel="PR-AUC", title="Đánh đổi riêng tư – hiệu năng")
    ax.set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
    ax.minorticks_off()
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "privacy_utility.png", dpi=150)

    # 3. Hội tụ FedAvg
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, d in hist.groupby("fold"):
        ax.plot(d["round"], d.val_pr_auc, label=f"fold {k}")
    ax.set(xlabel="Vòng FL", ylabel="PR-AUC (validation)", title="Hội tụ FedAvg (GraphSAGE, 5 client)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fedavg_convergence.png", dpi=150)
    print(f"Đã lưu biểu đồ vào {out}")


if __name__ == "__main__":
    main()
