"""Bước 1: sinh dữ liệu synthetic 5 ngân hàng + tính đặc trưng.

Chạy:  python scripts/01_generate_synthetic.py
Ra:    data/processed/synthetic_tx.parquet, data/processed/synthetic_customers.parquet
"""

import time

import _bootstrap  # noqa: F401

from fraudfl.config import DATA_PROCESSED, SimulationConfig
from fraudfl.data.features import build_features
from fraudfl.data.simulate import generate


def main():
    cfg = SimulationConfig()
    t0 = time.time()
    customers, terminals, tx = generate(cfg)
    print(f"Sinh {len(tx):,} giao dịch, {len(customers):,} khách, {len(terminals):,} terminal "
          f"trong {time.time() - t0:.1f}s")

    t0 = time.time()
    tx = build_features(tx)
    print(f"Tính đặc trưng trong {time.time() - t0:.1f}s")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(DATA_PROCESSED / "synthetic_tx.parquet", index=False)
    customers.to_parquet(DATA_PROCESSED / "synthetic_customers.parquet", index=False)
    terminals.to_parquet(DATA_PROCESSED / "synthetic_terminals.parquet", index=False)

    summary = tx.groupby("CLIENT_ID").agg(n_tx=("TX_FRAUD", "size"), fraud_rate=("TX_FRAUD", "mean"),
                                          mean_amount=("TX_AMOUNT", "mean"))
    summary.insert(0, "bank", [b.name for b in cfg.banks])
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"Tỷ lệ gian lận chung: {tx['TX_FRAUD'].mean():.4f}")


if __name__ == "__main__":
    main()
