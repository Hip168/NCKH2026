"""Các chỉ số đánh giá đã chốt (Slide 6, docs/decisions.md D1–D3, D7). Cả nhóm dùng chung module này.

Định nghĩa dùng trong bài báo:
- PR-AUC: average precision trên toàn tập test.
- Recall@p, Precision@p, VDR@p: mỗi NGÀY, gắn cờ ceil(p × số giao dịch trong ngày) giao dịch có điểm cao
  nhất (ngân sách kiểm tra của chuyên viên, p = 1% chính, 2% phân tích độ nhạy); cộng dồn trên mọi ngày.
    Recall@p    = số gian lận bị gắn cờ / tổng số gian lận
    Precision@p = số gian lận bị gắn cờ / tổng số giao dịch bị gắn cờ
    VDR@p       = tổng giá trị gian lận bị gắn cờ / tổng giá trị gian lận
  Dùng tỷ lệ thay vì k tuyệt đối để chỉ số không phụ thuộc kích thước fold hay bộ dữ liệu.
- F1: tại một ngưỡng xác suất; ngưỡng phải chọn trên tập validation (best_f1_threshold), không chọn trên test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from fraudfl.config import ALERT_FRACS


def _pct(frac: float) -> str:
    return f"{frac * 100:g}%"


def top_frac_daily(y_true, y_score, amount, day, frac: float) -> dict:
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score),
                       "a": np.asarray(amount, dtype=float), "d": np.asarray(day)})
    # rank trong từng ngày, điểm cao nhất = 1; hòa điểm xử lý theo thứ tự xuất hiện để kết quả xác định
    rank = df.groupby("d")["s"].rank(method="first", ascending=False)
    k_day = np.ceil(frac * df.groupby("d")["s"].transform("size"))
    flagged = rank <= k_day
    fraud = df["y"] == 1
    n_fraud = fraud.sum()
    fraud_value = df.loc[fraud, "a"].sum()
    p = _pct(frac)
    return {
        f"recall@{p}": float((flagged & fraud).sum() / n_fraud) if n_fraud else np.nan,
        f"precision@{p}": float((flagged & fraud).sum() / flagged.sum()) if flagged.sum() else np.nan,
        f"vdr@{p}": float(df.loc[flagged & fraud, "a"].sum() / fraud_value) if fraud_value else np.nan,
    }


def best_f1_threshold(y_true, y_score) -> float:
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def evaluate(y_true, y_score, amount, day, threshold: float = 0.5, fracs=ALERT_FRACS) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    res = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_score >= threshold, zero_division=0)),
        "threshold": float(threshold),
    }
    for frac in fracs:
        res.update(top_frac_daily(y_true, y_score, amount, day, frac))
    res["fraud_rate"] = float(y_true.mean())
    return res


MAIN_METRICS = ["pr_auc", "roc_auc", "f1"] + [f"{m}@{_pct(f)}" for f in ALERT_FRACS
                                              for m in ("recall", "precision", "vdr")]
