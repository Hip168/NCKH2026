"""Nạp dữ liệu synthetic đã xử lý, ghép fold, cắt giai đoạn khởi động, chuẩn hóa riêng từng client.

Dùng chung cho mọi thí nghiệm (baseline bảng, GNN, FL) để các thí nghiệm so sánh được với nhau.
"""

import numpy as np
import pandas as pd

from fraudfl.config import BURN_IN_DAYS, DATA_PROCESSED, DATA_SPLITS
from fraudfl.data.features import FEATURES


def load_synthetic() -> pd.DataFrame:
    """Toàn bộ giao dịch (kể cả giai đoạn khởi động) kèm cột fold. Sắp theo TRANSACTION_ID."""
    tx = pd.read_parquet(DATA_PROCESSED / "synthetic_tx.parquet")
    folds = pd.read_csv(DATA_SPLITS / "fold_assignments_synthetic.csv").drop(columns="CLIENT_ID")
    tx = tx.merge(folds, on="TRANSACTION_ID").sort_values("TRANSACTION_ID").reset_index(drop=True)
    tx["in_scope"] = tx["TX_TIME_DAYS"] >= BURN_IN_DAYS
    return tx


def fold_masks(tx: pd.DataFrame, k: int) -> dict[str, np.ndarray]:
    """Mặt nạ boolean cho outer fold k: fit (train trừ inner 0), val (inner 0), test. Đã bỏ giai đoạn khởi động."""
    scope = tx["in_scope"].to_numpy()
    test = (tx["outer_fold"] == k).to_numpy() & scope
    inner = tx[f"inner_fold_o{k}"].to_numpy()
    return {"fit": scope & (inner > 0), "val": scope & (inner == 0), "test": test}


def scale_per_client(tx: pd.DataFrame, fit_mask: np.ndarray, features=FEATURES) -> np.ndarray:
    """Chuẩn hóa (z-score) đặc trưng, thống kê tính trên phần fit của TỪNG client (D8).

    Mỗi ngân hàng tự chuẩn hóa dữ liệu của mình → tương thích FL (không cần chia sẻ thống kê).
    """
    X = tx[features].to_numpy(dtype=np.float64)
    out = np.empty_like(X, dtype=np.float32)
    clients = tx["CLIENT_ID"].to_numpy()
    for c in np.unique(clients):
        m = clients == c
        ref = X[m & fit_mask]
        mu, sd = ref.mean(0), ref.std(0)
        out[m] = (X[m] - mu) / np.where(sd > 0, sd, 1.0)
    return out
