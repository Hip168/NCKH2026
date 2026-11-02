"""Bằng chứng cho cách lấy mẫu IBM (D17): lấy mẫu cho TRAIN được không, và TEST lấy mẫu có đo đúng không.

Trên dữ liệu ngày 0–9 (D18), chia 5 fold phân tầng theo nhãn (seed 42). Với mỗi fold k:
- Phần train của fold: tách 20% làm VALIDATION giữ ĐỦ ca bình thường (chọn ngưỡng F1 — tránh lỗi validation lấy mẫu nhỏ).
- 80% còn lại: lấy mẫu r_train ∈ --rates (giữ hết ca rửa tiền; ngân hàng nhỏ gộp nhóm như 05) → HistGradientBoosting.
- Chấm trên TEST ĐẦY ĐỦ (mọi ca bình thường của fold) = con số chuẩn.
- Cùng mô hình, chấm trên test LẤY MẪU r_test ∈ {0,02; 0,1} có trọng số 1/r → so với con số chuẩn.
Chỉ số: PR-AUC, F1 lớp rửa tiền (ngưỡng chọn trên validation), Recall@1%, VDR@1%.

Chạy:  python scripts/08_ibm_danh_gia_lay_mau.py [--rates 0.02 0.05 0.1 0.2] [--folds 0 1 2 3 4]
Ra:    results/ibm_danh_gia_lay_mau/per_fold.csv, tom_tat.csv, log.txt (+ lưu trữ)
"""

import argparse
import time

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudfl.config import DATA_PROCESSED, IBM_MAX_DAY, RESULTS, SEED
from fraudfl.data.ibm_features import feature_columns
from fraudfl.data.sampling import sampling_groups, subsample_negatives
from fraudfl.eval.metrics import best_f1_threshold, evaluate
from fraudfl.runlog import archive_run

OUT = RESULTS / "ibm_danh_gia_lay_mau"


def log(msg):
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def metrics(y, p, amount, day, thr, w=None):
    r = evaluate(y, p, amount, day, weight=w)
    return {"pr_auc": r["pr_auc"], "f1": float(f1_score(y, p >= thr, sample_weight=w, zero_division=0)),
            "recall@1%": r["recall@1%"], "vdr@1%": r["vdr@1%"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--test-rates", type=float, nargs="+", default=[0.02, 0.1])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="balanced: trọng số lớp cân bằng khi huấn luyện (tách ảnh hưởng của r và của mất cân bằng lớp)")
    args = ap.parse_args()
    global OUT
    if args.class_weight != "none":
        OUT = RESULTS / f"ibm_danh_gia_lay_mau_cw_{args.class_weight}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "log.txt").write_text("", encoding="utf-8")

    g = pd.read_parquet(DATA_PROCESSED / f"{args.stem}_features_global.parquet")
    g = g[g["TX_TIME_DAYS"] <= IBM_MAX_DAY].reset_index(drop=True)
    fc = feature_columns(g)
    X = g[fc].to_numpy(np.float32)
    y = g["LABEL"].to_numpy()
    amt, day = g["AMOUNT_USD"].to_numpy(), g["TX_TIME_DAYS"].to_numpy()
    grp = sampling_groups(g["FROM_BANK"].to_numpy(), y)
    log(f"{args.stem} ngày 0–{IBM_MAX_DAY}: {len(g):,} gd, {y.sum():,} ca rửa tiền; {len(fc)} đặc trưng; "
        f"r_train {args.rates}; r_test {args.test_rates}; class_weight {args.class_weight}")

    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for k, (tr_all, te) in enumerate(skf.split(X, y)):
        if k not in args.folds:
            continue
        fit, val = train_test_split(tr_all, test_size=0.2, stratify=y[tr_all], random_state=SEED + k)
        for r in args.rates:
            t0 = time.time()
            s = subsample_negatives(fit, y[fit], grp[fit], r, SEED + k)
            idx = s["TX_ID"].to_numpy()
            m = HistGradientBoostingClassifier(max_iter=300, random_state=SEED,
                                               class_weight=None if args.class_weight == "none" else args.class_weight).fit(X[idx], y[idx])
            thr = best_f1_threshold(y[val], m.predict_proba(X[val])[:, 1])     # validation đầy đủ
            p = m.predict_proba(X[te])[:, 1]
            base = {"fold": k, "r_train": r, "n_train": len(idx), "seconds": round(time.time() - t0)}
            rows.append({**base, "test": "day_du", **metrics(y[te], p, amt[te], day[te], thr)})
            for rt in args.test_rates:
                st = subsample_negatives(np.arange(len(te)), y[te], grp[te], rt, SEED + 100 + k)
                j = st["TX_ID"].to_numpy()
                rows.append({**base, "test": f"lay_mau_{rt}",
                             **metrics(y[te][j], p[j], amt[te][j], day[te][j], thr, st["WEIGHT"].to_numpy())})
            r_full = rows[-1 - len(args.test_rates)]
            log(f"fold {k} r_train={r:<5} n_train={len(idx):>7,}  test đầy đủ: PR-AUC {r_full['pr_auc']:.4f} "
                f"F1 {r_full['f1']:.4f} VDR@1% {r_full['vdr@1%']:.4f} | test lấy mẫu: "
                + ", ".join(f"r={rt}: {row['pr_auc']:.4f}" for rt, row in zip(args.test_rates, rows[-len(args.test_rates):]))
                + f"  ({base['seconds']}s)")
            pd.DataFrame(rows).to_csv(OUT / "per_fold.csv", index=False, float_format="%.5f")

    df = pd.DataFrame(rows)
    summ = df.groupby(["r_train", "test"])[["pr_auc", "f1", "recall@1%", "vdr@1%"]].agg(["mean", "std"])
    summ.to_csv(OUT / "tom_tat.csv", float_format="%.4f")
    log("\n=== Trung bình ± độ lệch qua các fold ===")
    log(summ.round(4).to_string())
    log(f"Đã lưu trữ: {archive_run(OUT, note=f'Bằng chứng lấy mẫu IBM: r_train {args.rates}, test đầy đủ vs lấy mẫu, folds {args.folds}, class_weight {args.class_weight}')}")


if __name__ == "__main__":
    main()
