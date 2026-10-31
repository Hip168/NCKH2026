"""Biến đổi đặc trưng dạng bảng (theo Le Borgne et al., chương 3, viết lại dạng vector hóa).

Quan trọng cho FL: đặc trưng được tính RIÊNG trong từng ngân hàng — mỗi ngân hàng chỉ dùng giao dịch
của chính mình, không nhìn thấy dữ liệu ngân hàng khác.
"""

import pandas as pd

from fraudfl.config import TERMINAL_DELAY_DAYS

WINDOWS = (1, 7, 30)

FEATURES = (
    ["TX_AMOUNT", "TX_DURING_WEEKEND", "TX_DURING_NIGHT", "TX_AMOUNT_RATIO_30DAY"]
    + [f"CUSTOMER_ID_NB_TX_{w}DAY_WINDOW" for w in WINDOWS]
    + [f"CUSTOMER_ID_AVG_AMOUNT_{w}DAY_WINDOW" for w in WINDOWS]
    + [f"TERMINAL_ID_NB_TX_{w}DAY_WINDOW" for w in WINDOWS]
    + [f"TERMINAL_ID_RISK_{w}DAY_WINDOW" for w in WINDOWS]
)


def _rolling(df: pd.DataFrame, key: str, col: str, days: int, how: str) -> pd.Series:
    """Tổng/đếm trượt theo thời gian trong cửa sổ `days` ngày (tính cả giao dịch hiện tại), theo nhóm `key`."""
    # Sắp theo (key, thời gian) để thứ tự kết quả groupby-rolling trùng với thứ tự hàng, rồi gán theo vị trí
    s = df.sort_values([key, "TX_DATETIME"], kind="stable")
    r = s.groupby(key, sort=True).rolling(f"{days}D", on="TX_DATETIME")[col]
    out = r.sum() if how == "sum" else r.count()
    return pd.Series(out.to_numpy(), index=s.index).reindex(df.index)


def _client_features(df: pd.DataFrame, delay: int) -> pd.DataFrame:
    df = df.sort_values("TX_DATETIME", kind="stable").copy()

    for w in WINDOWS:
        nb = _rolling(df, "CUSTOMER_ID", "TX_AMOUNT", w, "count")
        total = _rolling(df, "CUSTOMER_ID", "TX_AMOUNT", w, "sum")
        df[f"CUSTOMER_ID_NB_TX_{w}DAY_WINDOW"] = nb
        df[f"CUSTOMER_ID_AVG_AMOUNT_{w}DAY_WINDOW"] = total / nb
    # Giá trị tương đối so với thói quen chi tiêu của chính khách → không phụ thuộc quy mô ngân hàng (D8)
    df["TX_AMOUNT_RATIO_30DAY"] = df["TX_AMOUNT"] / df["CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW"].clip(lower=0.01)

    # Rủi ro terminal: chỉ dùng nhãn đã biết, tức giao dịch trong [t - delay - w, t - delay)
    nb_delay = _rolling(df, "TERMINAL_ID", "TX_FRAUD", delay, "count")
    fr_delay = _rolling(df, "TERMINAL_ID", "TX_FRAUD", delay, "sum")
    for w in WINDOWS:
        nb = _rolling(df, "TERMINAL_ID", "TX_FRAUD", delay + w, "count") - nb_delay
        fr = _rolling(df, "TERMINAL_ID", "TX_FRAUD", delay + w, "sum") - fr_delay
        df[f"TERMINAL_ID_NB_TX_{w}DAY_WINDOW"] = nb
        df[f"TERMINAL_ID_RISK_{w}DAY_WINDOW"] = (fr / nb.where(nb > 0)).fillna(0.0)
    return df


def build_features(tx: pd.DataFrame, delay: int = TERMINAL_DELAY_DAYS) -> pd.DataFrame:
    tx = tx.copy()
    tx["TX_DURING_WEEKEND"] = (tx["TX_DATETIME"].dt.weekday >= 5).astype(int)
    tx["TX_DURING_NIGHT"] = (tx["TX_DATETIME"].dt.hour <= 6).astype(int)
    parts = [_client_features(g, delay) for _, g in tx.groupby("CLIENT_ID", sort=True)]
    return pd.concat(parts).sort_values("TRANSACTION_ID").reset_index(drop=True)
