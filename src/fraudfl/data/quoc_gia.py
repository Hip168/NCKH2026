"""Ngân hàng → quốc gia cho IBM AML (dùng để chia client theo nước, B2).

HI-Small_accounts.csv đặt tên ngân hàng dạng "Germany Bank #123" → nước = phần trước "Bank #".
Ngân hàng có tên riêng ("National Bank of Cleveland", "Oasis Thrift", ...) không có tiền tố nước; tên toàn là thành phố
Mỹ nên tạm gán "USA" (cần xác nhận lại với tài liệu AMLworld). "Crytpo" là lỗi chính tả có sẵn trong dữ liệu → "Crypto".
"""

from __future__ import annotations

from pathlib import Path

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
