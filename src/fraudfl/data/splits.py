"""Chia fold cố định cho cả nhóm. Tạo MỘT lần, lưu `fold_assignments.csv`, mọi thí nghiệm đọc lại file này.

Hai giao thức:
- stratified: Nested Stratified K-Fold (Slide 8) — phân tầng theo (nhãn × client), chia bên trong từng client.
  Cột `outer_fold` ∈ {0..K-1}; cột `inner_fold_o{k}` là fold trong (0..J-1) của hàng đó khi outer fold k
  làm tập test, = -1 nếu hàng thuộc tập test của outer fold k.
- temporal: train trên giai đoạn đầu, bỏ một khoảng đệm bằng độ trễ nhãn, test trên giai đoạn sau.
  Dùng cho dữ liệu đồ thị / đánh giá thực tế để tránh rò rỉ thông tin theo thời gian.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fraudfl.config import N_INNER_FOLDS, N_OUTER_FOLDS, SEED, TERMINAL_DELAY_DAYS


def stratified_nested(tx: pd.DataFrame, n_outer: int = N_OUTER_FOLDS, n_inner: int = N_INNER_FOLDS,
                      seed: int = SEED) -> pd.DataFrame:
    out = pd.DataFrame({"TRANSACTION_ID": tx["TRANSACTION_ID"].to_numpy(),
                        "CLIENT_ID": tx["CLIENT_ID"].to_numpy(),
                        "TX_FRAUD": tx["TX_FRAUD"].to_numpy()})
    out["outer_fold"] = -1
    for k in range(n_outer):
        out[f"inner_fold_o{k}"] = -1

    for _, g in out.groupby("CLIENT_ID"):
        idx, y = g.index.to_numpy(), g["TX_FRAUD"].to_numpy()
        outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(outer.split(idx, y)):
            out.loc[idx[te], "outer_fold"] = k
            inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + 1 + k)
            for j, (_, va) in enumerate(inner.split(idx[tr], y[tr])):
                out.loc[idx[tr][va], f"inner_fold_o{k}"] = j
    return out.drop(columns="TX_FRAUD")


def temporal(tx: pd.DataFrame, train_frac: float = 0.6, gap_days: int = TERMINAL_DELAY_DAYS) -> pd.Series:
    """Trả về nhãn 'train' / 'gap' / 'test' cho từng giao dịch."""
    days = tx["TX_TIME_DAYS"].to_numpy()
    cut = int(np.floor(days.max() * train_frac))
    return pd.Series(np.where(days < cut, "train", np.where(days < cut + gap_days, "gap", "test")),
                     index=tx.index, name="temporal_set")


def make_fold_assignments(tx: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    folds = stratified_nested(tx, seed=seed)
    folds["temporal_set"] = temporal(tx).to_numpy()
    return folds
