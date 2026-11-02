"""FL + DP mức client trên IBM HI-Small với nhiều ngân hàng (D19). Mô hình MLP dạng bảng ("cách B").

Thiết lập:
- Dữ liệu: đặc trưng chế độ `bank` (06_ibm_features.py --view bank), ngày 0–9 (D18).
- Client: các ngân hàng có >= --min-cases ca rửa tiền trong ngày 0–9 (mặc định 20 → 27 ngân hàng) + 1 client gộp
  mọi ngân hàng còn lại ("_nho"). Lưu ý: danh sách client chọn theo số ca trên toàn bộ ngày 0–9 (chỉ để chọn bên tham
  gia, không dùng cho huấn luyện).
- Chia: 5 fold phân tầng theo (client × nhãn), seed 42. Trong phần train của fold: 20% làm VALIDATION ĐẦY ĐỦ, 80% còn
  lại lấy mẫu r = --rate ca bình thường trong từng client (D17). TEST giữ đầy đủ.
- Chuẩn hóa z-score riêng từng client trên phần fit (D8).
- Các cách huấn luyện:
    central   gộp chung mọi client (cận trên)
    local     mỗi client tự học (cận dưới)
    fedavg    FedAvg không DP (trung bình theo số mẫu)
    dp        FedAvg + DP mức client: client chỉ cắt Δ về C, server cộng nhiễu một lần (trung bình đều, std = zC/K);
              ε tính bằng RDP, mọi client tham gia mọi vòng (sample_rate = 1, steps = số vòng)
- Chỉ số trên test đầy đủ: PR-AUC, Recall@1%, VDR@1% tổng; và PR-AUC trung bình theo client (chỉ client có ca xấu
  trong test).

Chạy:  python scripts/09_ibm_fl_dp.py [--folds 0] [--eps 8 4 2 1] [--rounds 30] [--min-cases 20]
Ra:    results/ibm_fl_dp[_<tag>]/per_fold.csv, per_client.csv, history.csv, log.txt (+ lưu trữ)
"""

import argparse
import math
import time

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudfl.config import DATA_PROCESSED, IBM_MAX_DAY, RESULTS, SEED
from fraudfl.data.ibm_features import feature_columns
from fraudfl.data.sampling import SMALL_GROUP, subsample_negatives
from fraudfl.eval.metrics import evaluate
from fraudfl.models import TabMLP
from fraudfl.privacy import CLIP, DELTA, add_server_noise, noise_multiplier, privatize_update
from fraudfl.runlog import archive_run
from fraudfl.train import GraphData, fedavg, predict, train_full_batch

LR, LOCAL_EPOCHS, MAX_EPOCHS, PATIENCE = 5e-3, 3, 300, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--eps", type=float, nargs="+", default=[8.0, 4.0, 2.0, 1.0])
    ap.add_argument("--settings", nargs="+", default=["central", "local", "fedavg", "dp"])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--rate", type=float, default=0.02)
    ap.add_argument("--min-cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch-size", type=int, default=None, help="mini-batch cho MLP (mặc định full-batch như bản đầu)")
    ap.add_argument("--pos-weight", type=float, default=None, help="trọng số lớp rửa tiền (mặc định = neg/pos)")
    ap.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    out = RESULTS / ("ibm_fl_dp" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text("", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        with open(out / "log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    f = pd.read_parquet(DATA_PROCESSED / f"{args.stem}_features_bank.parquet")
    f = f[f["TX_TIME_DAYS"] <= IBM_MAX_DAY].reset_index(drop=True)
    fc = feature_columns(f)
    X = f[fc].to_numpy(np.float32)
    y = f["LABEL"].to_numpy().astype(np.int64)
    amt, day = f["AMOUNT_USD"].to_numpy(), f["TX_TIME_DAYS"].to_numpy()

    cases = f.groupby("FROM_BANK")["LABEL"].sum()
    big = sorted(cases[cases >= args.min_cases].index, key=lambda b: -cases[b])
    client_name = np.where(f["FROM_BANK"].isin(big), f["FROM_BANK"], SMALL_GROUP)
    names = big + [SMALL_GROUP]
    cid = pd.Series(range(len(names)), index=names)[client_name].to_numpy()
    K = len(names)
    log(f"{args.stem} ngày 0–{IBM_MAX_DAY}: {len(f):,} gd, {y.sum():,} ca; {len(fc)} đặc trưng (view bank); "
        f"K = {K} client ({len(big)} ngân hàng >= {args.min_cases} ca + '{SMALL_GROUP}'); r = {args.rate}; "
        f"{args.rounds} vòng × {args.local_epochs} epoch; lr {args.lr}; batch {args.batch_size}; pos_weight {args.pos_weight}; "
        f"ε {args.eps}; seed {args.seed}")
    log("Ca rửa tiền theo client: " + ", ".join(f"{n}:{int((y[cid == i]).sum())}" for i, n in enumerate(names)))

    rows, per_client_rows, history = [], [], []
    skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    for k, (tr_all, te) in enumerate(skf.split(X, cid * 2 + y)):
        if k not in args.folds:
            continue
        fit_all, val = train_test_split(tr_all, test_size=0.2, stratify=cid[tr_all] * 2 + y[tr_all],
                                        random_state=args.seed + k)
        s = subsample_negatives(fit_all, y[fit_all], cid[fit_all], args.rate, args.seed + k)
        fit = s["TX_ID"].to_numpy()
        log(f"\n=== Fold {k}: fit (lấy mẫu) {len(fit):,} | val {len(val):,} | test {len(te):,} ===")

        # chuẩn hóa riêng từng client trên phần fit
        Xs = np.empty_like(X)
        used = np.concatenate([fit, val, te])
        for i in range(K):
            m_fit = fit[cid[fit] == i]
            mu, sd = X[m_fit].mean(0), X[m_fit].std(0)
            rows_i = used[cid[used] == i]
            Xs[rows_i] = (X[rows_i] - mu) / np.where(sd > 0, sd, 1.0)

        def client_data(i):
            nodes = np.concatenate([fit[cid[fit] == i], val[cid[val] == i], te[cid[te] == i]])
            nf, nv = (cid[fit] == i).sum(), (cid[val] == i).sum()
            mask = lambda a, b: torch.from_numpy(np.r_[np.zeros(a, bool), np.ones(b - a, bool), np.zeros(len(nodes) - b, bool)])
            return GraphData(torch.from_numpy(Xs[nodes]), None, torch.from_numpy(y[nodes].astype(np.float32)),
                             mask(0, nf), mask(nf, nf + nv), mask(nf + nv, len(nodes)), nodes)
        clients = [client_data(i) for i in range(K)]

        def report(setting, score, extra=None):
            r = evaluate(y[te], score[te], amt[te], day[te])
            pc = []
            for i, n in enumerate(names):
                t = te[cid[te] == i]
                if y[t].sum() > 0:
                    ap_i = float(average_precision_score(y[t], score[t]))
                    pc.append(ap_i)
                    per_client_rows.append({"fold": k, "setting": setting, "client": n, "n_test": len(t),
                                            "ca_xau_test": int(y[t].sum()), "pr_auc": ap_i})
            row = {"fold": k, "setting": setting, "pr_auc": r["pr_auc"], "macro_pr_auc": float(np.mean(pc)),
                   "recall@1%": r["recall@1%"], "vdr@1%": r["vdr@1%"], **(extra or {})}
            rows.append(row)
            log(f"  {setting:14s} PR-AUC {row['pr_auc']:.4f} | TB theo client {row['macro_pr_auc']:.4f} | "
                f"Recall@1% {row['recall@1%']:.4f} | VDR@1% {row['vdr@1%']:.4f}"
                + (f" | z={extra['z']:.2f}" if extra and "z" in extra else ""))
            pd.DataFrame(rows).to_csv(out / "per_fold.csv", index=False, float_format="%.5f")
            pd.DataFrame(per_client_rows).to_csv(out / "per_client.csv", index=False, float_format="%.5f")

        def full_score(model, datas):
            sc = np.full(len(y), np.nan)
            for d in datas:
                sc[d.node_ids] = predict(model, d)
            return sc

        if "central" in args.settings:
            t0 = time.time()
            nodes = np.concatenate([d.node_ids for d in clients])
            cat = lambda a: torch.cat([getattr(d, a) for d in clients])
            d_all = GraphData(cat("x"), None, cat("y"), cat("fit"), cat("val"), cat("test"), nodes)
            torch.manual_seed(args.seed)
            m = train_full_batch(TabMLP(len(fc)), d_all, args.lr, MAX_EPOCHS, PATIENCE, batch_size=args.batch_size,
                                 pos_weight=args.pos_weight)
            report("central", full_score(m, [d_all]), {"seconds": round(time.time() - t0)})

        if "local" in args.settings:
            t0 = time.time()
            sc = np.full(len(y), np.nan)
            for d in clients:
                torch.manual_seed(args.seed)
                m = train_full_batch(TabMLP(len(fc)), d, args.lr, MAX_EPOCHS, PATIENCE, batch_size=args.batch_size,
                                     pos_weight=args.pos_weight)
                sc[d.node_ids] = predict(m, d)
            report("local", sc, {"seconds": round(time.time() - t0)})

        if "fedavg" in args.settings:
            t0 = time.time()
            torch.manual_seed(args.seed)
            m, hist = fedavg(TabMLP(len(fc)), clients, args.lr, args.rounds, args.local_epochs,
                             batch_size=args.batch_size, pos_weight=args.pos_weight)
            history += [{"fold": k, "setting": "fedavg", **h} for h in hist]
            report("fedavg", full_score(m, clients), {"seconds": round(time.time() - t0)})

        if "dp" in args.settings:
            for eps in args.eps:
                t0 = time.time()
                gen = torch.Generator().manual_seed(args.seed + k)
                z = noise_multiplier(eps, sample_rate=1.0, steps=args.rounds)
                torch.manual_seed(args.seed)
                m, hist = fedavg(TabMLP(len(fc)), clients, args.lr, args.rounds, args.local_epochs,
                                 batch_size=args.batch_size, pos_weight=args.pos_weight,
                                 privatize=lambda loc, glo: privatize_update(loc, glo, CLIP, 0.0, gen),
                                 server_privatize=lambda avg, glo: add_server_noise(avg, glo, z * CLIP / K, gen),
                                 uniform_weights=True)
                history += [{"fold": k, "setting": f"dp_eps{eps:g}", **h} for h in hist]
                report(f"dp ε={eps:g}", full_score(m, clients),
                       {"epsilon": eps, "z": z, "noise_std": z * CLIP / K, "seconds": round(time.time() - t0)})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False, float_format="%.5f")

    df = pd.DataFrame(rows)
    log("\n=== Trung bình qua các fold ===")
    log(df.groupby("setting", sort=False)[["pr_auc", "macro_pr_auc", "recall@1%", "vdr@1%"]].mean().round(4).to_string())
    log(f"Đã lưu trữ: {archive_run(out, note=f'FL+DP mức client IBM {args.stem}, K={K}, folds {args.folds}, ε {args.eps}, {args.rounds} vòng')}")


if __name__ == "__main__":
    main()
