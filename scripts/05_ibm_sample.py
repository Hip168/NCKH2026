"""Lấy mẫu con IBM AML (D17) rồi chia nested stratified k-fold trên mẫu. Chạy MỘT lần, commit kết quả.

Đầu vào: bản gốc Kaggle có cột From Bank, ví dụ HI-Small_Trans.csv (đọc bằng ibm_aml.load_raw).
Đầu ra (trong data/splits/, commit lên git để cả nhóm dùng chung một mẫu):
  ibm_sample_ids.csv           TX_ID (số dòng trong file gốc, bắt đầu từ 0), GROUP (From Bank), LABEL, WEIGHT
  ibm_sample_meta.json         file nguồn, tỷ lệ, seed, số lượng trước / sau
  fold_assignments_ibm.csv     cùng định dạng fold_assignments_synthetic.csv (TRANSACTION_ID = TX_ID)

Ngân hàng có ít hơn --min-bank-tx giao dịch trong mẫu được gộp vào nhóm -1 khi chia fold (StratifiedKFold cần đủ
mẫu mỗi nhóm); nếu nhóm -1 vẫn quá nhỏ thì nhập vào ngân hàng lớn nhất. Cột GROUP vẫn giữ mã ngân hàng gốc để chia
client cho FL sau này.

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

from fraudfl.config import DATA_SPLITS, SEED
from fraudfl.data.ibm_aml import load_raw
from fraudfl.data.sampling import subsample_negatives
from fraudfl.data.splits import stratified_nested, temporal
from fraudfl.runlog import archive_run
from fraudfl.config import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--rate", type=float, required=True, help="tỷ lệ ca bình thường được giữ, ví dụ 0.02")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-bank-tx", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_ids = DATA_SPLITS / "ibm_sample_ids.csv"
    out_meta = DATA_SPLITS / "ibm_sample_meta.json"
    out_folds = DATA_SPLITS / "fold_assignments_ibm.csv"
    if out_ids.exists() and not args.force:
        raise SystemExit(f"{out_ids} đã tồn tại — mẫu đã cố định. Dùng --force chỉ khi cả nhóm đồng ý đổi mẫu.")

    df = load_raw(args.input)
    labels = df["Is Laundering"].to_numpy()
    s = subsample_negatives(np.arange(len(df)), labels, df["From Bank"].to_numpy(), args.rate, args.seed)

    # Chia fold trên mẫu: phân tầng theo (nhãn × ngân hàng); ngân hàng nhỏ gộp chung nhóm -1
    counts = s["GROUP"].map(s["GROUP"].value_counts())
    split_group = np.where(counts >= args.min_bank_tx, s["GROUP"], -1)
    if 0 < (split_group == -1).sum() < args.min_bank_tx:          # nhóm gộp vẫn quá nhỏ → nhập vào ngân hàng lớn nhất
        split_group[split_group == -1] = s["GROUP"].value_counts().idxmax()
    sub = pd.DataFrame({"TRANSACTION_ID": s["TX_ID"], "CLIENT_ID": split_group, "TX_FRAUD": s["LABEL"],
                        "TX_TIME_DAYS": df["TX_TIME_DAYS"].to_numpy()[s["TX_ID"]]})
    folds = stratified_nested(sub, seed=args.seed)
    folds["temporal_set"] = temporal(sub).to_numpy()

    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    s.to_csv(out_ids, index=False)
    folds.to_csv(out_folds, index=False)
    meta = {
        "source_file": args.input.name, "created": date.today().isoformat(), "seed": args.seed,
        "rate_normal": args.rate, "min_bank_tx": args.min_bank_tx,
        "n_full": int(len(df)), "n_fraud_full": int(labels.sum()), "fraud_rate_full": float(labels.mean()),
        "n_sample": int(len(s)), "n_fraud_sample": int(s["LABEL"].sum()),
        "fraud_rate_sample": float(s["LABEL"].mean()),
        "n_banks": int(s["GROUP"].nunique()), "n_banks_in_small_group": int(pd.Series(s["GROUP"][split_group == -1]).nunique()),
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
