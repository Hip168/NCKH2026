"""Lấy mẫu con ca bình thường cho bộ dữ liệu lớn (IBM AML), theo quyết định D17.

- Giữ 100% ca gian lận / rửa tiền.
- Trong TỪNG nhóm (ngân hàng), bốc ngẫu nhiên đúng round(rate × số ca bình thường) ca bình thường, seed cố định.
- Mỗi ca bình thường được giữ mang trọng số = số ca bình thường của nhóm / số ca được bốc (≈ 1/rate);
  ca gian lận có trọng số 1. Dùng trọng số này khi tính chỉ số trên test (`evaluate(..., weight=)`)
  để con số tương đương với tỷ lệ gian lận thật.

Quan trọng: đặc trưng và đồ thị phải tính trên DỮ LIỆU ĐẦY ĐỦ trước, rồi mới lọc theo danh sách ID này.
"""

import numpy as np
import pandas as pd


def subsample_negatives(ids: np.ndarray, labels: np.ndarray, groups: np.ndarray, rate: float,
                        seed: int) -> pd.DataFrame:
    """Trả về bảng các giao dịch được giữ: cột TX_ID, GROUP, LABEL, WEIGHT (sắp theo TX_ID)."""
    if not 0 < rate <= 1:
        raise ValueError("rate phải trong (0, 1]")
    ids, labels, groups = np.asarray(ids), np.asarray(labels).astype(int), np.asarray(groups)
    rng = np.random.default_rng(seed)
    keep_idx, weight = [], []
    order = np.argsort(groups, kind="stable")                    # duyệt nhóm theo thứ tự xác định
    uniq, starts = np.unique(groups[order], return_index=True)
    for g, idx in zip(uniq, np.split(order, starts[1:])):
        pos = idx[labels[idx] == 1]
        neg = idx[labels[idx] == 0]
        n_take = int(round(rate * len(neg)))
        if len(neg) and n_take == 0:
            n_take = 1                                            # nhóm nhỏ vẫn giữ ít nhất 1 ca bình thường
        take = np.sort(rng.choice(neg, size=n_take, replace=False)) if n_take else neg[:0]
        keep_idx += [pos, take]
        weight += [np.ones(len(pos)), np.full(len(take), len(neg) / n_take if n_take else 1.0)]
    keep_idx, weight = np.concatenate(keep_idx), np.concatenate(weight)
    out = pd.DataFrame({"TX_ID": ids[keep_idx], "GROUP": groups[keep_idx], "LABEL": labels[keep_idx],
                        "WEIGHT": weight})
    return out.sort_values("TX_ID").reset_index(drop=True)


SMALL_GROUP = "_nho"


def sampling_groups(banks: np.ndarray, labels: np.ndarray, min_neg: int = 1000) -> np.ndarray:
    """Nhóm lấy mẫu (D17): ngân hàng có >= min_neg ca bình thường giữ mã riêng; còn lại gộp thành SMALL_GROUP."""
    banks = np.asarray(banks).astype(str)
    n_neg = pd.Series(np.asarray(labels) == 0).groupby(banks).transform("sum").to_numpy()
    return np.where(n_neg >= min_neg, banks, SMALL_GROUP)
