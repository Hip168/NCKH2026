"""Kiểm định thống kê cho kết quả chính trên IBM (Slide 8b): Wilcoxon signed-rank, Holm, rank-biserial, bootstrap CI.

Ghép cặp theo (seed, fold) — cùng cách chia dữ liệu nên so sánh được từng cặp. Mỗi seed khai báo các thư mục kết quả
(per_fold.csv của 09_ibm_fl_dp.py) thuộc seed đó.

Chạy:  python scripts/15_thong_ke.py --nhom 42:results/ibm_fl_dp_CHINH_5fold_moc,results/ibm_fl_dp_CHINH_5fold_local,results/ibm_fl_dp_CHINH_5fold_100vong \
                                     --nhom 7:...  --nhom 2027:...  [--chi-so macro_pr_auc pr_auc] [--out ten]
Ra:    results/thong_ke[_ten]/so_sanh.csv, bang.md, mo_ta.csv (+ lưu trữ)

Cặp so sánh mặc định (chỉ những cặp có đủ dữ liệu):
  FL + DP ε=1 vs Tự học riêng      — RQ1: tham gia có lợi không
  FedProx vs Tự học riêng          — RQ1 không bảo mật
  Gộp chung vs FedProx             — chi phí của FL
  FedProx vs FL + DP ε=1 / ε=2     — chi phí của DP (RQ2)
  FedProx vs FedAvg thường         — lợi ích FedProx
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from fraudfl.config import RESULTS
from fraudfl.runlog import archive_run

NAMES = {"central": "Gộp chung", "local": "Tự học riêng", "fedavg": "FedAvg thường", "fedavg_deu": "FedProx (TB đều)"}
PAIRS = [("dp ε=1", "local"), ("fedavg_deu", "local"), ("central", "fedavg_deu"),
         ("fedavg_deu", "dp ε=2"), ("fedavg_deu", "dp ε=1"), ("fedavg_deu", "fedavg")]


def name(s):
    return NAMES.get(s, "FL + DP " + s.replace("dp ", "")) if s.startswith("dp") or s in NAMES else s


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_ci(d, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d), replace=True).mean() for _ in range(B)]
    return np.percentile(m, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhom", action="append", required=True, help="seed:thư_mục1,thư_mục2,...")
    ap.add_argument("--chi-so", nargs="+", default=["macro_pr_auc", "pr_auc"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    frames = []
    for g in args.nhom:
        seed, dirs = g.split(":", 1)
        for d in dirs.split(","):
            f = pd.read_csv(Path(d) / "per_fold.csv")
            f["seed"] = int(seed)
            frames.append(f)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["seed", "fold", "setting"], keep="last")

    out = RESULTS / ("thong_ke" + (f"_{args.out}" if args.out else ""))
    out.mkdir(parents=True, exist_ok=True)
    desc = df.groupby("setting")[args.chi_so].agg(["mean", "std", "count"])
    desc.to_csv(out / "mo_ta.csv", float_format="%.5f")

    rows = []
    for metric in args.chi_so:
        piv = df.pivot_table(index=["seed", "fold"], columns="setting", values=metric)
        for a, b in PAIRS:
            if a not in piv or b not in piv:
                continue
            x = piv[[a, b]].dropna()
            d = (x[a] - x[b]).to_numpy()
            if len(d) < 2:
                continue
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            lo, hi = boot_ci(d)
            rows.append({"chi_so": metric, "A": name(a), "B": name(b), "n_cap": len(d),
                         "TB_A": x[a].mean(), "TB_B": x[b].mean(), "chenh_TB": d.mean(),
                         "CI95_thap": lo, "CI95_cao": hi, "A_thang": int((d > 0).sum()),
                         "p_wilcoxon": p, "rank_biserial": rank_biserial(d)})
    res = pd.DataFrame(rows)
    for metric in args.chi_so:
        m = res["chi_so"] == metric
        res.loc[m, "p_holm"] = holm(res.loc[m, "p_wilcoxon"])
    res["co_y_nghia_0.05"] = res["p_holm"] < 0.05
    res.to_csv(out / "so_sanh.csv", index=False, float_format="%.5f")

    p_min = {n: 2 * 0.5 ** n for n in sorted(res["n_cap"].unique())}
    lines = [f"Ghép cặp theo (seed, fold). Seeds: {sorted(df['seed'].unique())}. "
             f"p nhỏ nhất có thể của Wilcoxon hai phía: " + ", ".join(f"n={n} → {v:.4f}" for n, v in p_min.items()), "",
             "| Chỉ số | A | B | n | TB A | TB B | Chênh (A−B) [CI 95%] | A thắng | p Wilcoxon | p Holm | r rank-biserial |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r.chi_so} | {r.A} | {r.B} | {r.n_cap} | {r.TB_A:.4f} | {r.TB_B:.4f} | "
                     f"{r.chenh_TB:+.4f} [{r.CI95_thap:+.4f}; {r.CI95_cao:+.4f}] | {r.A_thang}/{r.n_cap} | "
                     f"{r.p_wilcoxon:.4f} | {r.p_holm:.4f}{' ✅' if r.p_holm < 0.05 else ''} | {r.rank_biserial:+.2f} |")
    (out / "bang.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Đã lưu trữ: {archive_run(out, note='Kiểm định thống kê: ' + ' | '.join(args.nhom))}")


if __name__ == "__main__":
    main()
