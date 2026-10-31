"""Bước 3: EDA — thống kê và biểu đồ mô tả dữ liệu từng ngân hàng (minh chứng non-IID cho bài báo).

Chạy:  python scripts/03_eda.py
Ra:    results/eda/*.csv, results/eda/*.png
"""

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudfl.config import DATA_PROCESSED, RESULTS, SimulationConfig


def main():
    out = RESULTS / "eda"
    out.mkdir(parents=True, exist_ok=True)
    tx = pd.read_parquet(DATA_PROCESSED / "synthetic_tx.parquet")
    names = {i: b.name for i, b in enumerate(SimulationConfig().banks)}
    tx["bank"] = tx["CLIENT_ID"].map(names)

    per_bank = tx.groupby("bank").agg(
        n_tx=("TX_FRAUD", "size"), n_customers=("CUSTOMER_ID", "nunique"),
        n_terminals=("TERMINAL_ID", "nunique"), fraud_rate=("TX_FRAUD", "mean"),
        amount_mean=("TX_AMOUNT", "mean"), amount_median=("TX_AMOUNT", "median"),
        fraud_value_share=("TX_AMOUNT", lambda a: a[tx.loc[a.index, "TX_FRAUD"] == 1].sum() / a.sum()),
    )
    per_bank.to_csv(out / "per_bank_summary.csv")
    print(per_bank.round(4).to_string())

    scen = pd.crosstab(tx["bank"], tx["TX_FRAUD_SCENARIO"], normalize="index")
    scen.columns = ["genuine", "s1_amount", "s2_terminal", "s3_card"][: len(scen.columns)]
    scen.to_csv(out / "fraud_scenarios_by_bank.csv")
    print(scen.round(4).to_string())

    # Terminal chung: bao nhiêu terminal bị gian lận kịch bản 2 xuất hiện ở >1 ngân hàng
    s2 = tx[tx["TX_FRAUD_SCENARIO"] == 2].groupby("TERMINAL_ID")["CLIENT_ID"].nunique()
    print(f"Terminal gian lận KB2 xuất hiện ở >1 ngân hàng: {(s2 > 1).mean():.1%} ({(s2 > 1).sum()}/{len(s2)})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for bank, g in tx.groupby("bank"):
        axes[0].hist(np.log10(g["TX_AMOUNT"].clip(lower=0.01)), bins=60, histtype="step", label=bank, density=True)
    axes[0].set(title="Phân phối log10(giá trị giao dịch)", xlabel="log10(amount)")
    axes[0].legend(fontsize=8)
    per_bank["fraud_rate"].plot.bar(ax=axes[1], title="Tỷ lệ gian lận theo ngân hàng")
    daily = tx.groupby(["TX_TIME_DAYS", "bank"])["TX_FRAUD"].mean().unstack()
    daily.rolling(7).mean().plot(ax=axes[2], title="Tỷ lệ gian lận theo ngày (TB trượt 7 ngày)", legend=False)
    fig.tight_layout()
    fig.savefig(out / "non_iid_overview.png", dpi=150)
    print(f"Đã lưu biểu đồ và bảng vào {out}")


if __name__ == "__main__":
    main()
