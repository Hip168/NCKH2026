"""Hàm dùng chung cho các script thí nghiệm: tạo GraphData, ghép điểm về bảng gốc, tính chỉ số trên test."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch_geometric.utils import subgraph

from fraudfl.eval.metrics import evaluate
from fraudfl.train import GraphData, predict


def make_data(x: np.ndarray, y: np.ndarray, masks: dict, edge_index: torch.Tensor | None,
              nodes: np.ndarray | None = None) -> GraphData:
    """GraphData cho một bên. `nodes` = chỉ số giao dịch của bên đó (None = toàn bộ).
    `edge_index` = None cho mô hình dạng bảng."""
    if nodes is None:
        nodes = np.arange(len(y))
        ei = edge_index
    elif edge_index is not None:
        ei, _ = subgraph(torch.from_numpy(nodes), edge_index, relabel_nodes=True, num_nodes=len(y))
    else:
        ei = None
    t = lambda a: torch.from_numpy(a[nodes])
    return GraphData(t(x), ei, t(y.astype(np.float32)), t(masks["fit"]), t(masks["val"]), t(masks["test"]), nodes)


def scores(model, datas: list[GraphData], n: int) -> np.ndarray:
    """Điểm của mô hình trên mọi nút của các bên, ghép về chỉ số giao dịch gốc."""
    out = np.full(n, np.nan)
    for d in datas:
        out[d.node_ids] = predict(model, d)
    return out


def test_metrics(tx: pd.DataFrame, masks: dict, score: np.ndarray, thr_per_row: np.ndarray) -> dict:
    """Chỉ số trên tập test; F1 dùng ngưỡng theo từng hàng (cho phép mỗi ngân hàng một ngưỡng)."""
    te = masks["test"]
    test = tx[te]
    res = evaluate(test["TX_FRAUD"], score[te], test["TX_AMOUNT"], test["TX_TIME_DAYS"])
    res["f1"] = float(f1_score(test["TX_FRAUD"], score[te] >= thr_per_row[te], zero_division=0))
    res.pop("threshold")
    return res
