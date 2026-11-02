"""Kiểm tra nhanh chất lượng đặc trưng IBM (sau 06_ibm_features.py). KHÔNG phải thí nghiệm chính thức.

1. Sạch dữ liệu: thiếu, vô cực, khoảng giá trị.
2. Sức phân biệt từng đặc trưng trên toàn bộ dữ liệu: ROC-AUC đơn biến (đảo chiều nếu < 0,5) và AP đơn biến.
3. Mô hình thử: HistGradientBoosting trên mẫu tạm (giữ hết ca rửa tiền, r ca bình thường — mẫu này KHÔNG phải mẫu
   chính thức D17), chia 80/20 phân tầng, chỉ số trên test CÓ TRỌNG SỐ 1/r. So sánh view global và bank.

Chạy:  python scripts/07_ibm_kiem_tra_dac_trung.py [--stem HI-Small] [--rate 0.02]
Ra:    results/ibm_kiem_tra_dac_trung/ (+ lưu trữ)
"""

import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fraudfl.config import DATA_PROCESSED, RESULTS, SEED
from fraudfl.data.ibm_features import feature_columns
from fraudfl.data.sampling import subsample_negatives
from fraudfl.eval.metrics import evaluate
from fraudfl.runlog import archive_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--rate", type=float, default=0.02)
    ap.add_argument("--max-day", type=int, default=None, help="chỉ giữ TX_TIME_DAYS <= giá trị này (đuôi HI-Small từ ngày 10 bất thường)")
    ap.add_argument("--drop", nargs="*", default=[], help="bỏ các đặc trưng này")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = RESULTS / ("ibm_kiem_tra_dac_trung" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    def load(view):
        df = pd.read_parquet(DATA_PROCESSED / f"{args.stem}_features_{view}.parquet").drop(columns=args.drop, errors="ignore")
        return df if args.max_day is None else df[df["TX_TIME_DAYS"] <= args.max_day].reset_index(drop=True)
    g = load("global")
    cols = feature_columns(g)
    y = g["LABEL"].to_numpy()

    # 1. Sạch dữ liệu
    X = g[cols].to_numpy(np.float64)
    clean = pd.DataFrame({"thieu": np.isnan(X).sum(0), "vo_cuc": np.isinf(X).sum(0),
                          "min": np.nanmin(X, 0), "max": np.nanmax(X, 0)}, index=cols)

    # 2. Đơn biến
    uni = []
    for i, c in enumerate(cols):
        auc = roc_auc_score(y, X[:, i])
        sign = 1 if auc >= 0.5 else -1
        uni.append({"dac_trung": c, "roc_auc": max(auc, 1 - auc), "chieu": "+" if sign > 0 else "−",
                    "ap": average_precision_score(y, sign * X[:, i])})
    uni = pd.DataFrame(uni).sort_values("roc_auc", ascending=False)
    uni.to_csv(out / "don_bien.csv", index=False, float_format="%.4f")
    clean.join(uni.set_index("dac_trung")).to_csv(out / "sach_du_lieu.csv", float_format="%.4f")

    # 3. Mô hình thử
    s = subsample_negatives(np.arange(len(g)), y, g["FROM_BANK"].to_numpy(), args.rate, SEED)   # chỉ số dòng sau khi lọc
    b = load("bank")
    tr, te = train_test_split(np.arange(len(s)), test_size=0.2, stratify=s["LABEL"], random_state=SEED)
    rows = []
    for view, df in [("global", g), ("bank", b)]:
        fc = feature_columns(df)
        d = df.iloc[s["TX_ID"].to_numpy()]
        Xs, ys, w = d[fc].to_numpy(np.float32), d["LABEL"].to_numpy(), s["WEIGHT"].to_numpy()
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
        m.fit(Xs[tr], ys[tr], sample_weight=np.where(ys[tr] == 1, 1.0, 1.0))   # huấn luyện trên tỷ lệ mẫu
        p = m.predict_proba(Xs[te])[:, 1]
        dt = d.iloc[te]
        r_w = evaluate(ys[te], p, dt["AMOUNT_USD"], dt["TX_TIME_DAYS"], weight=w[te])
        r_n = evaluate(ys[te], p, dt["AMOUNT_USD"], dt["TX_TIME_DAYS"])
        rows.append({"view": view, "n_dac_trung": len(fc), "pr_auc_co_trong_so": r_w["pr_auc"],
                     "pr_auc_khong_trong_so": r_n["pr_auc"], "roc_auc": r_w["roc_auc"],
                     "recall@1%": r_w["recall@1%"], "vdr@1%": r_w["vdr@1%"], "ty_le_ngau_nhien": r_w["fraud_rate"]})
    model = pd.DataFrame(rows)
    model.to_csv(out / "mo_hinh_thu.csv", index=False, float_format="%.4f")
    meta = {"stem": args.stem, "rate_tam": args.rate, "max_day": args.max_day, "drop": args.drop, "n_mau": len(s), "n_test": len(te),
            "ghi_chu": "Mẫu tạm, chia ngẫu nhiên 80/20 — chỉ để kiểm tra đặc trưng, không dùng làm kết quả bài báo"}
    (out / "tom_tat.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("== Sạch dữ liệu: thiếu", int(clean["thieu"].sum()), "| vô cực", int(clean["vo_cuc"].sum()))
    print("== Đơn biến (top / cuối):\n", uni.round(4).to_string(index=False))
    print("== Mô hình thử:\n", model.round(4).to_string(index=False))
    print("Lưu trữ:", archive_run(out, note=f"Kiểm tra đặc trưng IBM {args.stem}, mẫu tạm r={args.rate}, max_day={args.max_day}, bỏ {args.drop}"))


if __name__ == "__main__":
    main()
