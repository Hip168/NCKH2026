"""Sinh dữ liệu giao dịch tổng hợp cho nhiều ngân hàng (client), có tính non-IID.

Phương pháp dựa trên bộ sinh dữ liệu của Le Borgne et al., "Reproducible Machine Learning for
Credit Card Fraud Detection — Practical Handbook" (2022), chương 3 (mã nguồn GPL-3.0), được viết lại
dạng vector hóa và mở rộng cho bối cảnh nhiều ngân hàng:

- Mỗi ngân hàng có tập khách hàng riêng với phân phối địa lý, giá trị và tần suất giao dịch khác nhau.
- Terminal (điểm bán) dùng chung giữa các ngân hàng.
- Kịch bản 1: giao dịch vượt ngưỡng giá trị (ngưỡng riêng từng ngân hàng).
- Kịch bản 2: terminal bị xâm nhập → mọi giao dịch tại đó trong 28 ngày là gian lận. Vì terminal dùng
  chung, mẫu gian lận này trải trên nhiều ngân hàng; mỗi ngân hàng chỉ thấy một phần — động lực cho FL.
- Kịch bản 3: thông tin thẻ khách bị lộ → 1/3 giao dịch trong 14 ngày bị nhân giá trị x5.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fraudfl.config import BankProfile, SimulationConfig


def _terminals(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    xy = rng.uniform(0, 100, size=(cfg.n_terminals, 2))
    return pd.DataFrame({"TERMINAL_ID": np.arange(cfg.n_terminals), "x": xy[:, 0], "y": xy[:, 1]})


def _customers(bank: BankProfile, client_id: int, n_days: int, id_offset: int,
               rng: np.random.Generator) -> pd.DataFrame:
    mean_rate = np.mean(bank.rate_range)
    # ~5% giao dịch bị loại do thời điểm rơi ngoài [0, 86400) → bù vào số khách
    n = int(round(bank.target_n_tx / (n_days * mean_rate * 0.97)))
    xy = np.clip(rng.normal(bank.center, bank.spread, size=(n, 2)), 0, 100)
    mean_amount = rng.uniform(*bank.amount_range, size=n)
    return pd.DataFrame({
        "CUSTOMER_ID": id_offset + np.arange(n),
        "CLIENT_ID": client_id,
        "x": xy[:, 0], "y": xy[:, 1],
        "mean_amount": mean_amount,
        "std_amount": mean_amount / 2,
        "mean_nb_tx_per_day": rng.uniform(*bank.rate_range, size=n),
    })


def _transactions(customers: pd.DataFrame, terminals: pd.DataFrame, cfg: SimulationConfig,
                  rng: np.random.Generator) -> pd.DataFrame:
    # Terminal trong bán kính r của mỗi khách, lưu dạng CSR (offset + danh sách phẳng)
    tree = cKDTree(terminals[["x", "y"]].to_numpy())
    near = tree.query_ball_point(customers[["x", "y"]].to_numpy(), r=cfg.terminal_radius)
    n_near = np.array([len(t) for t in near])
    offsets = np.concatenate([[0], np.cumsum(n_near)[:-1]])
    flat = np.concatenate([np.asarray(t, dtype=np.int64) for t in near]) if n_near.sum() else np.array([], np.int64)

    counts = rng.poisson(customers["mean_nb_tx_per_day"].to_numpy()[:, None],
                         size=(len(customers), cfg.n_days))
    cust_idx, day = np.nonzero(counts)
    reps = counts[cust_idx, day]
    cust_idx, day = np.repeat(cust_idx, reps), np.repeat(day, reps)

    t = rng.normal(86400 / 2, 20000, size=len(cust_idx)).astype(np.int64)
    keep = (t > 0) & (t < 86400) & (n_near[cust_idx] > 0)
    cust_idx, day, t = cust_idx[keep], day[keep], t[keep]

    mu = customers["mean_amount"].to_numpy()[cust_idx]
    sd = customers["std_amount"].to_numpy()[cust_idx]
    amount = rng.normal(mu, sd)
    neg = amount < 0
    amount[neg] = rng.uniform(0, 2 * mu[neg])

    pick = np.floor(rng.uniform(size=len(cust_idx)) * n_near[cust_idx]).astype(np.int64)
    terminal = flat[offsets[cust_idx] + pick]

    seconds = t + day * 86400
    return pd.DataFrame({
        "TX_DATETIME": pd.to_datetime(seconds, unit="s", origin=cfg.start_date),
        "CUSTOMER_ID": customers["CUSTOMER_ID"].to_numpy()[cust_idx],
        "CLIENT_ID": customers["CLIENT_ID"].to_numpy()[cust_idx],
        "TERMINAL_ID": terminal,
        "TX_AMOUNT": np.round(amount, 2),
        "TX_TIME_SECONDS": seconds,
        "TX_TIME_DAYS": day,
    })


def _add_frauds(tx: pd.DataFrame, customers: pd.DataFrame, cfg: SimulationConfig,
                rng: np.random.Generator) -> pd.DataFrame:
    tx["TX_FRAUD"] = 0
    tx["TX_FRAUD_SCENARIO"] = 0
    days = tx["TX_TIME_DAYS"].to_numpy()

    # Kịch bản 1: vượt ngưỡng giá trị riêng của từng ngân hàng
    thresholds = np.array([b.s1_threshold for b in cfg.banks])[tx["CLIENT_ID"].to_numpy()]
    s1 = tx["TX_AMOUNT"].to_numpy() > thresholds
    tx.loc[s1, ["TX_FRAUD", "TX_FRAUD_SCENARIO"]] = [1, 1]

    # Kịch bản 2: terminal bị xâm nhập (chung cho mọi ngân hàng)
    n_term = tx["TERMINAL_ID"].max() + 1
    for d in range(cfg.n_days):
        comp = rng.choice(n_term, size=cfg.s2_terminals_per_day, replace=False)
        m = (days >= d) & (days < d + cfg.s2_duration_days) & tx["TERMINAL_ID"].isin(comp).to_numpy()
        tx.loc[m, ["TX_FRAUD", "TX_FRAUD_SCENARIO"]] = [1, 2]

    # Kịch bản 3: thẻ bị lộ, mỗi ngân hàng có cường độ khác nhau
    for client_id, bank in enumerate(cfg.banks):
        cust_ids = customers.loc[customers["CLIENT_ID"] == client_id, "CUSTOMER_ID"].to_numpy()
        for d in range(cfg.n_days):
            comp = rng.choice(cust_ids, size=bank.s3_customers_per_day, replace=False)
            idx = tx.index[(days >= d) & (days < d + cfg.s3_duration_days)
                           & tx["CUSTOMER_ID"].isin(comp).to_numpy()]
            hit = rng.choice(idx, size=len(idx) // 3, replace=False) if len(idx) else idx
            tx.loc[hit, "TX_AMOUNT"] = tx.loc[hit, "TX_AMOUNT"] * 5
            tx.loc[hit, ["TX_FRAUD", "TX_FRAUD_SCENARIO"]] = [1, 3]
    return tx


def generate(cfg: SimulationConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trả về (customers, terminals, transactions). transactions đã sắp theo thời gian, có TRANSACTION_ID."""
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    terminals = _terminals(cfg, rng)

    customers, offset = [], 0
    for client_id, bank in enumerate(cfg.banks):
        c = _customers(bank, client_id, cfg.n_days, offset, rng)
        customers.append(c)
        offset += len(c)
    customers = pd.concat(customers, ignore_index=True)

    tx = _transactions(customers, terminals, cfg, rng)
    tx = tx.sort_values("TX_DATETIME", kind="stable").reset_index(drop=True)
    tx = _add_frauds(tx, customers, cfg, rng)
    tx.insert(0, "TRANSACTION_ID", np.arange(len(tx)))
    return customers, terminals, tx
