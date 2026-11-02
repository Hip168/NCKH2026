"""Đánh giá mô hình Flower đã lưu trên tập test, cùng bộ chỉ số với các thí nghiệm khác.

Chạy:  python scripts/51_danh_gia_mo_hinh.py artifacts/<run>/model.pt [artifacts/<run2>/model.pt ...]
Dùng ở máy nghiên cứu vì cần toàn bộ dữ liệu; trong triển khai thật mỗi ngân hàng chỉ tự đánh giá phần của mình.
"""

import argparse

import _bootstrap  # noqa: F401

import numpy as np
import torch

from fraudfl.data.prep import fold_masks, load_synthetic, scale_per_client
from fraudfl.eval.metrics import evaluate
from fraudfl.fl.task import new_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--banks", type=int, nargs="+", default=None)
    args = ap.parse_args()
    tx = load_synthetic()
    if args.banks:
        tx = tx[tx["CLIENT_ID"].isin(args.banks)].reset_index(drop=True)
    masks = fold_masks(tx, args.fold)
    x = torch.from_numpy(scale_per_client(tx, masks["fit"]))
    test = tx[masks["test"]]
    for path in args.models:
        model = new_model(x.shape[1])
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(x)).numpy()
        r = evaluate(test["TX_FRAUD"], s[masks["test"]], test["TX_AMOUNT"], test["TX_TIME_DAYS"])
        print(f"{path}: PR-AUC {r['pr_auc']:.4f}  Recall@1% {r['recall@1%']:.4f}  VDR@1% {r['vdr@1%']:.4f}  "
              f"trên {np.count_nonzero(masks['test']):,} giao dịch")


if __name__ == "__main__":
    main()
