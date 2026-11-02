"""Lấy mẫu con IBM AML (D17) rồi chia nested stratified k-fold trên mẫu. Chạy MỘT lần, commit kết quả.

Đầu vào: bản gốc Kaggle có cột From Bank, ví dụ HI-Small_Trans.csv (đọc bằng ibm_aml.load_raw).
Đầu ra (trong data/splits/, commit lên git để cả nhóm dùng chung một mẫu):
  ibm_sample_ids.csv           TX_ID (số dòng trong file gốc, bắt đầu từ 0), GROUP (nhóm lấy mẫu), BANK (From Bank),
                               LABEL, WEIGHT
  ibm_sample_meta.json         file nguồn, tỷ lệ, seed, số lượng trước / sau
  fold_assignments_ibm.csv     cùng định dạng fold_assignments_synthetic.csv (TRANSACTION_ID = TX_ID)

D18: chỉ lấy giao dịch ngày <= --max-day (mặc định config.IBM_MAX_DAY = 9).
Nhóm lấy mẫu: ngân hàng có >= --min-bank-neg ca bình thường được lấy mẫu riêng; các ngân hàng nhỏ hơn GỘP thành một
nhóm "_nho" rồi lấy mẫu chung (tránh việc mỗi ngân hàng tí hon bị ép giữ 1 ca làm phình mẫu; khớp thiết kế client
"top K ngân hàng + 1 client gộp"). Nhóm có < --min-bank-tx giao dịch trong mẫu cũng gộp vào "_nho" khi chia fold.

Chạy:  python scripts/05_ibm_sample.py --input data/raw/ibm_aml/HI-Small_Trans.csv --rate 0.02
       (thêm --force để ghi đè — chỉ khi cả nhóm đã đồng ý đổi mẫu)
"""

import argparse
import json
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from fraudfl.config import DATA_SPLITS, IBM_MAX_DAY, SEED
from fraudfl.data.ibm_aml import load_raw
from fraudfl.data.sampling import subsample_negatives
from fraudfl.data.splits import stratified_nested, temporal
from fraudfl.runlog import archive_run
from fraudfl.config import RESULTS
import warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--rate", type=float, required=True, help="tỷ lệ ca bình thường được giữ, ví dụ 0.02")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-bank-tx", type=int, default=50)
    ap.add_argument("--min-bank-neg", type=int, default=1000, help="ngân hàng có ít ca bình thường hơn → nhóm gộp")
    ap.add_argument("--max-day", type=int, default=IBM_MAX_DAY, help="D18: chỉ lấy TX_TIME_DAYS <= giá trị này")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_ids = DATA_SPLITS / "ibm_sample_ids.csv"
    out_meta = DATA_SPLITS / "ibm_sample_meta.json"
    out_folds = DATA_SPLITS / "fold_assignments_ibm.csv"
    if out_ids.exists() and not args.force:
        raise SystemExit(f"{out_ids} đã tồn tại — mẫu đã cố định. Dùng --force chỉ khi cả nhóm đồng ý đổi mẫu.")

    df = load_raw(args.input)
    labels = df["Is Laundering"].to_numpy()
    # D18: bỏ đuôi dữ liệu bất thường (HI-Small ngày 10–17: ~60% rửa tiền). TX_ID vẫn là số dòng trong file gốc.
    keep = df["TX_TIME_DAYS"].to_numpy() <= args.max_day
    SMALL = "_nho"                                                # mã ngân hàng là chuỗi → dùng nhãn chuỗi
    bank = df["From Bank"].to_numpy()[keep]
    n_neg = pd.Series(labels[keep] == 0).groupby(bank).transform("sum").to_numpy()
    samp_group = np.where(n_neg >= args.min_bank_neg, bank, SMALL)
    s = subsample_negatives(np.flatnonzero(keep), labels[keep], samp_group, args.rate, args.seed)
    s.insert(2, "BANK", df["From Bank"].to_numpy()[s["TX_ID"]])

    # Chia fold trên mẫu: phân tầng theo (nhãn × nhóm); nhóm quá nhỏ trong mẫu gộp vào SMALL
    counts = s["GROUP"].map(s["GROUP"].value_counts())
    split_group = np.where(counts >= args.min_bank_tx, s["GROUP"].astype(str), SMALL).astype(object)
    if 0 < (split_group == SMALL).sum() < args.min_bank_tx:       # nhóm gộp vẫn quá nhỏ → nhập vào ngân hàng lớn nhất
        split_group[split_group == SMALL] = str(s["GROUP"].value_counts().idxmax())
    sub = pd.DataFrame({"TRANSACTION_ID": s["TX_ID"], "CLIENT_ID": split_group, "TX_FRAUD": s["LABEL"],
                        "TX_TIME_DAYS": df["TX_TIME_DAYS"].to_numpy()[s["TX_ID"]]})
    with warnings.catch_warnings():                               # nhóm ít ca rửa tiền → sklearn cảnh báo, chấp nhận
        warnings.simplefilter("ignore", UserWarning)
        folds = stratified_nested(sub, seed=args.seed)
    folds["temporal_set"] = temporal(sub).to_numpy()

    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    s.to_csv(out_ids, index=False)
    folds.to_csv(out_folds, index=False)
    meta = {
        "source_file": args.input.name, "created": date.today().isoformat(), "seed": args.seed,
        "rate_normal": args.rate, "min_bank_tx": args.min_bank_tx, "min_bank_neg": args.min_bank_neg,
        "max_day": args.max_day,
        "n_full": int(len(df)), "n_fraud_full": int(labels.sum()), "fraud_rate_full": float(labels.mean()),
        "n_after_max_day": int(keep.sum()), "n_fraud_after_max_day": int(labels[keep].sum()),
        "fraud_rate_after_max_day": float(labels[keep].mean()),
        "n_sample": int(len(s)), "n_fraud_sample": int(s["LABEL"].sum()),
        "fraud_rate_sample": float(s["LABEL"].mean()),
        "n_banks": int(s["BANK"].nunique()), "n_sampling_groups": int(s["GROUP"].nunique()),
        "n_banks_in_small_group": int(s["BANK"][split_group == SMALL].nunique()),
        "n_split_groups": int(pd.Series(split_group).nunique()),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Đã ghi {out_ids}, {out_folds}, {out_meta}. Nhớ commit cả ba file.")
    rec = RESULTS / "ibm_sample"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "ibm_sample_meta.json").write_text(out_meta.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Đã lưu trữ: {archive_run(rec, note=f'Lấy mẫu IBM r={args.rate}, seed {args.seed}')}")


if __name__ == "__main__":
    main()
