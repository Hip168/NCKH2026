"""EDA cho IBM AML LI-Small (bản đã định dạng).

Chạy:  python scripts/04_ibm_aml_eda.py
Ra:    data/processed/ibm_li_small.parquet (cache), results/eda_ibm/summary.csv, daily.png
"""

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fraudfl.config import DATA_PROCESSED, IBM_LI_SMALL_FORMATTED, RESULTS
from fraudfl.data.ibm_aml import load_formatted


def main():
    out = RESULTS / "eda_ibm"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Đọc {IBM_LI_SMALL_FORMATTED}")
    df = load_formatted(IBM_LI_SMALL_FORMATTED, cache=DATA_PROCESSED / "ibm_li_small.parquet")

    accounts = pd.concat([df["from_id"], df["to_id"]]).nunique()
    illicit = df["Is Laundering"] == 1
    summary = pd.Series({
        "n_transactions": len(df),
        "n_accounts": accounts,
        "n_days": df["TX_TIME_DAYS"].nunique(),
        "laundering_rate": illicit.mean(),
        "n_laundering": int(illicit.sum()),
        # Không tính tỷ trọng giá trị: "Amount Sent" ở 15 loại tiền tệ khác nhau, cộng thẳng là vô nghĩa.
        # VDR trên IBM AML cần quy đổi về một đồng tiền trước (xem docs/decisions.md Q6).
        "n_currencies": df["Sent Currency"].nunique(),
        "n_payment_formats": df["Payment Format"].nunique(),
    })
    summary.to_csv(out / "summary.csv", header=["value"])
    print(summary.to_string())

    by_format = df.groupby("Payment Format")["Is Laundering"].agg(["size", "mean"])
    by_format.to_csv(out / "by_payment_format.csv")
    print(by_format.to_string())

    daily = df.groupby("TX_TIME_DAYS").agg(n=("Is Laundering", "size"), illicit=("Is Laundering", "sum"))
    daily.to_csv(out / "daily.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    daily["n"].plot(ax=ax[0], title="Số giao dịch mỗi ngày")
    daily["illicit"].plot(ax=ax[1], title="Số giao dịch rửa tiền mỗi ngày")
    fig.tight_layout()
    fig.savefig(out / "daily.png", dpi=130)
    print(f"Đã lưu vào {out}")


if __name__ == "__main__":
    main()
