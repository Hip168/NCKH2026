"""Kiểm tra giả thuyết "FL thấp vì 802 ngân hàng khác nhau quá (non-IID)": tạo bản IBM có NGÂN HÀNG NGẪU NHIÊN.

Cách làm: trong các giao dịch ngày 0–9, xáo trộn cột FROM_BANK giữa các dòng (seed cố định). Kết quả:
- mỗi ngân hàng giữ ĐÚNG số giao dịch như thật → vẫn 802 ngân hàng >= 1.000 giao dịch, cùng kích cỡ;
- nhưng giao dịch (và ca rửa tiền) trong mỗi ngân hàng là mẫu ngẫu nhiên của toàn bộ → mọi ngân hàng giống nhau (IID);
- đặc trưng không đổi (tính theo lịch sử tài khoản, không phụ thuộc nhãn ngân hàng).
Chạy 09_ibm_fl_dp.py với --stem HI-Small-IID và đúng cấu hình chính của Đạt rồi so với bản thật:
  FL trên ngân hàng ngẫu nhiên gần bằng gộp chung → "khác nhau" là nguyên nhân chính.
  FL trên ngân hàng ngẫu nhiên vẫn thấp           → nguyên nhân nằm ở cách huấn luyện FL (số vòng, lấy mẫu ...).

Chạy:  python scripts/14_ibm_ngan_hang_ngau_nhien.py
Ra:    data/processed/HI-Small-IID_features_bank.parquet
"""

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from fraudfl.config import DATA_PROCESSED, IBM_MAX_DAY


def main():
    f = pd.read_parquet(DATA_PROCESSED / "HI-Small_features_bank.parquet")
    m = (f["TX_TIME_DAYS"] <= IBM_MAX_DAY).to_numpy()
    banks = f["FROM_BANK"].to_numpy().copy()
    banks[m] = np.random.default_rng(0).permutation(banks[m])
    f["FROM_BANK"] = banks
    f.to_parquet(DATA_PROCESSED / "HI-Small-IID_features_bank.parquet", index=False)
    g = f[m].groupby("FROM_BANK")["LABEL"].agg(["size", "sum"])
    lon = g[g["size"] >= 1000]
    print(f"{len(lon)} ngân hàng >= 1.000 gd; ca rửa tiền mỗi ngân hàng: trung vị {lon['sum'].median():.0f}, "
          f"không có ca: {(lon['sum'] == 0).sum()}")


if __name__ == "__main__":
    main()
