"""Biểu đồ cho bài báo — IBM HI-Small, 802 ngân hàng (Slide 10: VDR/PR-AUC theo ε, hội tụ, ...).

Đọc thẳng từ per_fold.csv / history.csv của các lần chạy chính (giai đoạn 23, 26–29). Màu: 3 slot đầu của bảng màu
chuẩn (xanh dương, cam, xanh ngọc — đạt kiểm tra phân biệt cho người mù màu với mọi cặp).

Ra: results/bieu_do_ibm/
  1_so_sanh_theo_ngan_hang.png   PR-AUC trung bình theo ngân hàng: tự học riêng → FL + DP → FL → gộp chung
  2_rieng_tu_hieu_nang.png       % giữ lại so với FL không DP theo ε (30 và 100 vòng)
  3_hoi_tu.png                   PR-AUC validation theo vòng (fold 0, 200 vòng)
  4_chi_phi_dp_theo_so_ngan_hang.png   DP ε = 1 giữ được bao nhiêu % FL không DP khi có 5 / 28 / 802 ngân hàng
"""

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudfl.config import RESULTS
from fraudfl.runlog import archive_run

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OUT = RESULTS / "bieu_do_ibm"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load(tag):
    return pd.read_csv(RESULTS / f"ibm_fl_dp_{tag}" / "per_fold.csv")


def fig1():
    moc, loc, f100 = load("CHINH_5fold_moc"), load("CHINH_5fold_local"), load("CHINH_5fold_100vong")
    rows = [("Tự học riêng", loc[loc.setting == "local"]),
            ("FL + DP ε = 1", f100[f100.setting == "dp ε=1"]), ("FL + DP ε = 2", f100[f100.setting == "dp ε=2"]),
            ("FL + DP ε = 8", f100[f100.setting == "dp ε=8"]), ("FL không DP\n(FedProx)", f100[f100.setting == "fedavg_deu"]),
            ("Gộp chung", moc[moc.setting == "central"])]
    names = [r[0] for r in rows]
    mean = np.array([r[1].macro_pr_auc.mean() for r in rows])
    sd = np.array([r[1].macro_pr_auc.std() for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, mean, xerr=sd, height=0.55, color=BLUE, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for yi, m, e in zip(y, mean, sd):
        ax.text(m + e + 0.008, yi, f"{m:.3f}  ({m / mean[0]:.2f}× tự học)" if m != mean[0] else f"{m:.3f}",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 0.50)
    ax.set_xlabel("PR-AUC trung bình theo ngân hàng (5 fold, ± độ lệch)")
    ax.set_title("FL + DP so với tự học riêng và gộp chung — IBM HI-Small, 802 ngân hàng", loc="left", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "1_so_sanh_theo_ngan_hang.png", dpi=200)


def fig2():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for tag, label, color, marker in [("CHINH_5fold_fedprox_dp", "30 vòng", ORANGE, "s"), ("CHINH_5fold_100vong", "100 vòng", BLUE, "o")]:
        d = load(tag)
        base = d[d.setting == "fedavg_deu"].set_index("fold").pr_auc
        eps = [0.5, 1, 2, 4, 8]
        extra = RESULTS / "ibm_fl_dp_CHINH_5fold_100vong_eps0.5" / "per_fold.csv"
        if tag == "CHINH_5fold_100vong" and extra.exists():
            d = pd.concat([d, pd.read_csv(extra)])
        xs, ms, ss = [], [], []
        for e in eps:
            s = d[d.setting == f"dp ε={e:g}"].set_index("fold").pr_auc
            if len(s):
                r = 100 * s / base.loc[s.index]
                xs.append(e); ms.append(r.mean()); ss.append(r.std())
        ax.errorbar(xs, ms, yerr=ss, color=color, marker=marker, markersize=7, linewidth=2, capsize=3, label=label)
        ax.annotate(label, (xs[-1], ms[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.axhline(85, color=INK2, linewidth=1, linestyle="--")
    ax.text(2.6, 80.5, "ngưỡng 85% (tiêu chí 2 tầng)", color=INK2, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0,5", "1", "2", "4", "8"])
    ax.set_xlabel("ε (δ = 1e-5) — nhỏ hơn = bảo mật chặt hơn")
    ax.set_ylabel("% PR-AUC so với FL không DP")
    ax.set_ylim(40, 110)
    ax.set_title("Đánh đổi riêng tư – hiệu năng (DP mức client, 5 fold)", loc="left", fontsize=10.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "2_rieng_tu_hieu_nang.png", dpi=200)


def fig3():
    h = pd.read_csv(RESULTS / "ibm_fl_dp_K802_200vong_prox_dp1" / "history.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for s, label, color in [("fedavg_deu", "FL không DP (FedProx)", BLUE), ("dp_eps1", "FL + DP ε = 1", ORANGE)]:
        g = h[h.setting == s]
        ax.plot(g["round"], g["val_pr_auc"], color=color, linewidth=2, label=label)
        b = g.loc[g.val_pr_auc.idxmax()]
        ax.plot(b["round"], b["val_pr_auc"], marker="o", markersize=7, color=color, markeredgecolor=SURF, markeredgewidth=2)
        ax.annotate(f"tốt nhất: vòng {int(b['round'])}", (b["round"], b["val_pr_auc"]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_ylim(0, 0.045)
    ax.axvline(100, color=INK2, linewidth=1, linestyle="--")
    ax.text(102, 0.042, "100 vòng (cấu hình chính)", color=INK2, fontsize=8.5)
    ax.set_xlabel("Vòng FL")
    ax.set_ylabel("PR-AUC trên validation")
    ax.set_title("Hội tụ — fold 0, 802 ngân hàng, 200 vòng", loc="left", fontsize=10.5, pad=10)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    fig.savefig(OUT / "3_hoi_tu.png", dpi=200)


def fig4():
    # Nguồn (docs/nhat_ky_ket_qua.md): K = 5 synthetic, DP mức client ε = 1 PR-AUC 0,028 so với FedAvg 0,715 (giai đoạn 2);
    # K = 28 IBM, DP ε = 1 0,0012 so với FedAvg cùng cấu hình 0,0578 (giai đoạn 14–15, fold 0);
    # K = 802 IBM, DP ε = 1 so với FedProx không DP, 100 vòng, 5 fold (giai đoạn 27).
    d = load("CHINH_5fold_100vong")
    k802 = 100 * (d[d.setting == "dp ε=1"].set_index("fold").pr_auc / d[d.setting == "fedavg_deu"].set_index("fold").pr_auc).mean()
    labels = ["5 ngân hàng\n(synthetic)", "28 ngân hàng\n(IBM)", "802 ngân hàng\n(IBM)"]
    vals = [100 * 0.028 / 0.715, 100 * 0.0012 / 0.0578, k802]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(3)
    ax.bar(x, vals, width=0.55, color=[GRID, GRID, BLUE], edgecolor=[INK2, INK2, BLUE])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("DP ε = 1 giữ được (% FL không DP)")
    ax.set_title("DP mức client cần nhiều bên tham gia", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "4_chi_phi_dp_theo_so_ngan_hang.png", dpi=200)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
        print("Đã vẽ", f.__name__)
    print("Lưu trữ:", archive_run(OUT, note="Biểu đồ IBM cho bài báo (giai đoạn 23, 26–29)"))


if __name__ == "__main__":
    main()
