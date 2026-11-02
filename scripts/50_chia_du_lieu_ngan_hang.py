"""Xuất dữ liệu riêng cho từng ngân hàng để chạy Flower và real-time.

Chạy:  python scripts/50_chia_du_lieu_ngan_hang.py [--folds 0] [--banks 0 1 2 3 4]
Ra:    data/banks/bank_<i>/fold<k>.npz và data/banks/gop/fold<k>.npz (chế độ gộp các ngân hàng đã chọn)
Chuẩn hóa z-score tính riêng trên phần fit của từng ngân hàng, giống prep.scale_per_client (D8).
"""

import argparse

import _bootstrap  # noqa: F401

import numpy as np

from fraudfl.data.features import FEATURES
from fraudfl.data.prep import fold_masks, load_synthetic, scale_per_client
from fraudfl.fl.partition import BANKS_DIR, fold_path, save_partition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--banks", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    tx = load_synthetic()
    tx = tx[tx["CLIENT_ID"].isin(args.banks)].reset_index(drop=True)
    y = tx["TX_FRAUD"].to_numpy()
    clients = tx["CLIENT_ID"].to_numpy()
    X_raw = tx[FEATURES].to_numpy(np.float64)

    for k in args.folds:
        masks = fold_masks(tx, k)
        x = scale_per_client(tx, masks["fit"])
        keep_all = masks["fit"] | masks["val"] | masks["test"]
        mus, sds = {}, {}
        for c in args.banks:
            own = clients == c
            ref = X_raw[own & masks["fit"]]
            mus[c], sds[c] = ref.mean(0), np.where(ref.std(0) > 0, ref.std(0), 1.0)
            m = own & keep_all
            save_partition(fold_path(BANKS_DIR, f"bank_{c}", k), x[m], y[m],
                           {n: v[m] for n, v in masks.items()}, mus[c], sds[c])
        save_partition(fold_path(BANKS_DIR, "gop", k), x[keep_all], y[keep_all],
                       {n: v[keep_all] for n, v in masks.items()}, np.stack(list(mus.values())),
                       np.stack(list(sds.values())))
        print(f"fold {k}: {int(keep_all.sum()):,} giao dịch, {len(args.banks)} ngân hàng + bản gộp → {BANKS_DIR}")


if __name__ == "__main__":
    main()
