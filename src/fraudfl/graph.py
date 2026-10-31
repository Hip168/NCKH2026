"""Đồ thị giao dịch: mỗi giao dịch là một nút; cạnh nối giao dịch với L giao dịch LIỀN TRƯỚC
của cùng khách hàng và cùng terminal.

- Cạnh chỉ có chiều quá khứ → hiện tại: khi chấm điểm một giao dịch, mô hình chỉ nhìn giao dịch đã xảy ra
  (không dùng thông tin tương lai, tránh rò rỉ theo thời gian).
- Local / FL: cạnh terminal chỉ nối giao dịch CÙNG ngân hàng (mỗi ngân hàng chỉ thấy dữ liệu mình).
- Centralized: dữ liệu đã gộp nên cạnh terminal nối xuyên ngân hàng — đây là lợi thế thông tin của
  centralized mà FL không có.
"""

import numpy as np
import pandas as pd
import torch

LAGS = 3


def _lag_edges(df: pd.DataFrame, group_cols: list[str], lags: int) -> np.ndarray:
    order = df.sort_values(group_cols + ["TX_DATETIME", "TRANSACTION_ID"], kind="stable").index.to_numpy()
    g = df.loc[order, group_cols].to_numpy()
    edges = []
    for lag in range(1, lags + 1):
        same = np.all(g[lag:] == g[:-lag], axis=1)
        edges.append(np.stack([order[:-lag][same], order[lag:][same]]))
    return np.concatenate(edges, axis=1)


def build_edge_index(tx: pd.DataFrame, cross_client: bool, lags: int = LAGS) -> torch.Tensor:
    """tx phải có index 0..n-1 trùng với chỉ số nút. Trả về edge_index [2, E] (nguồn = quá khứ, đích = hiện tại)."""
    terminal_key = ["TERMINAL_ID"] if cross_client else ["CLIENT_ID", "TERMINAL_ID"]
    e = np.concatenate([_lag_edges(tx, ["CLIENT_ID", "CUSTOMER_ID"], lags),
                        _lag_edges(tx, terminal_key, lags)], axis=1)
    return torch.from_numpy(np.unique(e, axis=1)).long()
