"""Ngân hàng → quốc gia cho IBM AML (dùng để chia client theo nước, B2).

HI-Small_accounts.csv đặt tên ngân hàng dạng "Germany Bank #123" → nước = phần trước "Bank #".
Ngân hàng có tên riêng ("National Bank of Cleveland", "Oasis Thrift", ...) không có tiền tố nước; tên toàn là thành phố
Mỹ nên tạm gán "USA" (cần xác nhận lại với tài liệu AMLworld). "Crytpo" là lỗi chính tả có sẵn trong dữ liệu → "Crypto".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KHONG_TIEN_TO = "USA"
SUA_TEN = {"Crytpo": "Crypto"}


def nuoc_tu_ten(ten: pd.Series) -> pd.Series:
    nuoc = ten.str.extract(r"^(.*?)\s*Bank\s*#\d+$")[0].fillna(KHONG_TIEN_TO).str.strip()
    return nuoc.replace(SUA_TEN)


def bang_ngan_hang_quoc_gia(accounts_csv: Path) -> pd.DataFrame:
    """Bảng BANK (chuỗi, đã bỏ số 0 ở đầu như ibm_aml.load_raw), TEN_NGAN_HANG, NUOC; mỗi ngân hàng một dòng."""
    a = pd.read_csv(accounts_csv, usecols=["Bank Name", "Bank ID"], dtype=str).drop_duplicates("Bank ID")
    bank = a["Bank ID"].str.lstrip("0").replace("", "0")
    out = pd.DataFrame({"BANK": bank, "TEN_NGAN_HANG": a["Bank Name"], "NUOC": nuoc_tu_ten(a["Bank Name"])})
    if out["BANK"].duplicated().any():
        raise ValueError("Mã ngân hàng trùng sau khi bỏ số 0 ở đầu")
    return out.sort_values("BANK").reset_index(drop=True)


GOP_NUOC_NHO = 10_000
TEN_GOP = "_khac"


def client_theo_nuoc(nuoc_giao_dich: pd.Series, gop_nho: int = GOP_NUOC_NHO) -> pd.DataFrame:
    """Bảng nước → client (B2). Nước có < gop_nho giao dịch gộp chung một client TEN_GOP.

    Vào: nước của ngân hàng gửi, mỗi giao dịch một dòng. Ra: NUOC, SO_GIAO_DICH, CLIENT, MA_CLIENT
    (MA_CLIENT 0..K-1, client nhiều giao dịch nhất mã 0; TEN_GOP luôn ở cuối). Script 09 (--client nuoc) và
    scripts/b2_client_theo_nuoc.py cùng dùng hàm này nên luôn ra cùng một bảng.
    """
    if nuoc_giao_dich.isna().any():
        raise ValueError(f"{int(nuoc_giao_dich.isna().sum())} giao dịch không tra được nước")
    size = nuoc_giao_dich.value_counts()
    bang = pd.DataFrame({"NUOC": size.index.astype(str), "SO_GIAO_DICH": size.to_numpy()})
    bang["CLIENT"] = np.where(bang["SO_GIAO_DICH"] < gop_nho, TEN_GOP, bang["NUOC"])
    tong = bang.groupby("CLIENT")["SO_GIAO_DICH"].sum()
    thu_tu = sorted(tong.index, key=lambda c: (c == TEN_GOP, -tong[c], c))
    bang["MA_CLIENT"] = bang["CLIENT"].map({c: i for i, c in enumerate(thu_tu)}).astype(int)
    return bang.sort_values(["MA_CLIENT", "SO_GIAO_DICH"], ascending=[True, False]).reset_index(drop=True)
