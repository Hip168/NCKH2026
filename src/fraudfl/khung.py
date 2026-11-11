"""Khung B0: cách chia dữ liệu IBM HI-Small CỐ ĐỊNH cho mọi bước B1–B6 (D25).

Mỗi giao dịch (ngày 0–9) được gán vào 1 trong 5 fold, riêng cho từng seed. Với fold k:
    test = fold k,   val = fold (k + 1) mod 5,   fit = 3 fold còn lại   (60 / 20 / 20)
Vai trò của một dòng chỉ phụ thuộc (seed, k), nên mọi người, mọi bước, mọi máy (kể cả Kaggle) dùng đúng một cách chia.

Phân tầng: (ngân hàng gửi, nhãn) nếu ngân hàng có >= 10 ca rửa tiền, ngược lại chỉ theo nhãn → mỗi fold có ~1/5 số ca
của từng ngân hàng lớn, và ~1/5 tổng số ca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fraudfl.config import DATA_SPLITS

SEEDS = (42, 7, 2027)
N_FOLD = 5
FILE_CHIA = DATA_SPLITS / "ibm_hi_small_chia.parquet"


def tang_phan_tang(label: np.ndarray, nhom: np.ndarray, min_ca: int = 10) -> np.ndarray:
    """Mã tầng: nhóm*2 + nhãn với nhóm có >= min_ca ca; còn lại -(nhãn + 1)."""
    label = np.asarray(label).astype(np.int64)
    ma, nhom_id = np.unique(np.asarray(nhom), return_inverse=True)
    ca = np.bincount(nhom_id, weights=label, minlength=len(ma))
    return np.where(ca[nhom_id] >= min_ca, nhom_id * 2 + label, -(label + 1))


def tao_chia(tx_id: np.ndarray, label: np.ndarray, nhom: np.ndarray, seeds=SEEDS, n_fold: int = N_FOLD) -> pd.DataFrame:
    """Bảng TX_ID + cột fold_s<seed> (int8, 0..n_fold-1), sắp theo TX_ID."""
    tx_id = np.asarray(tx_id)
    if len(np.unique(tx_id)) != len(tx_id):
        raise ValueError("TX_ID bị trùng")
    strat = tang_phan_tang(label, nhom)
    out = {"TX_ID": tx_id}
    for s in seeds:
        fold = np.full(len(tx_id), -1, dtype=np.int8)
        for k, (_, te) in enumerate(StratifiedKFold(n_fold, shuffle=True, random_state=s).split(tx_id, strat)):
            fold[te] = k
        out[f"fold_s{s}"] = fold
    return pd.DataFrame(out).sort_values("TX_ID", kind="stable").reset_index(drop=True)


def vai_tro(fold: np.ndarray, k: int, n_fold: int = N_FOLD) -> dict[str, np.ndarray]:
    """Mặt nạ fit / val / test cho fold k."""
    fold = np.asarray(fold)
    te = fold == k
    va = fold == (k + 1) % n_fold
    return {"fit": ~(te | va), "val": va, "test": te}


def doc_chia(seed: int, path=FILE_CHIA) -> pd.Series:
    """fold của từng TX_ID cho một seed (index = TX_ID)."""
    d = pd.read_parquet(path, columns=["TX_ID", f"fold_s{seed}"])
    return d.set_index("TX_ID")[f"fold_s{seed}"]


def gan_fold(tx_id: np.ndarray, seed: int, path=FILE_CHIA) -> np.ndarray:
    """fold cho dãy TX_ID bất kỳ (lỗi nếu có TX_ID không nằm trong file chia)."""
    f = doc_chia(seed, path).reindex(np.asarray(tx_id))
    if f.isna().any():
        raise KeyError(f"{int(f.isna().sum())} TX_ID không có trong {path.name} (ngoài ngày 0–9?)")
    return f.to_numpy(np.int8)
