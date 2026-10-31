"""Nạp IBM AML (Altman et al., NeurIPS 2023 Datasets & Benchmarks).

Hai dạng file:
- Bản gốc Kaggle `*_Trans.csv` (cột: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, ...).
  Có mã ngân hàng → chia được client theo ngân hàng cho FL (E8). Dùng `load_raw`.
- Bản đã định dạng `formatted_transactions.csv` (sinh bởi format_kaggle_files.py của IBM/Multi-GNN).
  Mã ngân hàng đã bị gộp vào mã tài khoản → CHỈ dùng được cho centralized (E7). Dùng `load_formatted`.
"""

from pathlib import Path

import pandas as pd

FORMATTED_COLUMNS = {
    "EdgeID": "int64", "from_id": "int64", "to_id": "int64", "Timestamp": "int64",
    "Amount Sent": "float64", "Sent Currency": "int16", "Amount Received": "float64",
    "Received Currency": "int16", "Payment Format": "int16", "Is Laundering": "int8",
}


def load_formatted(csv_path: Path, cache: Path | None = None) -> pd.DataFrame:
    """Đọc bản đã định dạng; lưu cache parquet để lần sau đọc nhanh."""
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(csv_path, dtype=FORMATTED_COLUMNS, engine="pyarrow")
    df["TX_TIME_DAYS"] = (df["Timestamp"] // 86400).astype("int32")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Đọc bản gốc Kaggle; tài khoản được định danh bằng (ngân hàng, số tài khoản)."""
    df = pd.read_csv(csv_path, engine="pyarrow")
    df = df.rename(columns={"Account": "From Account", "Account.1": "To Account"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df["TX_TIME_DAYS"] = (df["Timestamp"] - df["Timestamp"].min().normalize()).dt.days
    return df
