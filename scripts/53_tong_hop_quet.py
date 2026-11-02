"""Tổng hợp kết quả quét thông số của infra/quet_thong_so.sh thành một bảng.

Chạy:  python scripts/53_tong_hop_quet.py [--fold 0] [--banks 0 1 2]
Ra:    artifacts/tn/tong_hop.csv và in bảng ra màn hình.
Điểm test tính trên toàn bộ giao dịch test của các ngân hàng đã chọn. Điểm val chỉ dùng để chọn vòng tốt nhất.
"""

import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch

from fraudfl.config import ROOT
from fraudfl.data.prep import fold_masks, load_synthetic, scale_per_client
from fraudfl.eval.metrics import evaluate
from fraudfl.fl.task import new_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--banks", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    tx = load_synthetic()
    tx = tx[tx["CLIENT_ID"].isin(args.banks)].reset_index(drop=True)
    masks = fold_masks(tx, args.fold)
    x = torch.from_numpy(scale_per_client(tx, masks["fit"]))
    test = tx[masks["test"]]

    rows = []
    root = ROOT / "artifacts" / "tn"
    for run in sorted(p for p in root.iterdir() if (p / "tom_tat.json").exists()):
        info = json.loads((run / "tom_tat.json").read_text())
        curve = pd.read_csv(run / "rounds.csv")
        model = new_model(x.shape[1])
        model.load_state_dict(torch.load(run / "model.pt", weights_only=True))
        model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(x)).numpy()
        r = evaluate(test["TX_FRAUD"], score[masks["test"]], test["TX_AMOUNT"], test["TX_TIME_DAYS"])
        rows.append({"cau_hinh": run.name, "so_vong": len(curve), "vong_tot_nhat": info["vong_tot_nhat"],
                     "val_pr_auc": info["val_pr_auc"], "val_vong_cuoi": curve["val_pr_auc"].iloc[-1],
                     "test_pr_auc": r["pr_auc"], "recall_1pct": r["recall@1%"], "vdr_1pct": r["vdr@1%"],
                     "giay": info["tong_giay"], "mb": info["tong_mb"]})
    table = pd.DataFrame(rows)
    table.to_csv(root / "tong_hop.csv", index=False)
    print(f"{np.count_nonzero(masks['test']):,} giao dịch test")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
