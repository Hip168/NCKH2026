"""Các chỉ số đánh giá đã chốt (Slide 6). Cả nhóm dùng chung module này — không tự tính lại ở chỗ khác.

Định nghĩa dùng trong bài báo:
- PR-AUC: average precision trên toàn tập test.
- Recall@k, Precision@k, VDR@k: mỗi NGÀY, gắn cờ k giao dịch có điểm cao nhất (ngân sách kiểm tra của
  chuyên viên); cộng dồn trên mọi ngày.
    Recall@k    = số gian lận bị gắn cờ / tổng số gian lận
    Precision@k = số gian lận bị gắn cờ / tổng số giao dịch bị gắn cờ
    VDR@k       = tổng giá trị gian lận bị gắn cờ / tổng giá trị gian lận
  Lưu ý: nếu tính Recall@100 trên toàn tập test (không chia theo ngày) thì với ~2.000 ca gian lận,
  Recall tối đa chỉ 100/2000 = 5% — vì vậy bắt buộc tính theo ngày.
- F1: tại một ngưỡng xác suất; ngưỡng phải chọn trên tập validation (best_f1_threshold), không chọn trên test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from fraudfl.config import TOP_K_PER_DAY


def top_k_daily(y_true, y_score, amount, day, k: int = TOP_K_PER_DAY) -> dict:
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score),
                       "a": np.asarray(amount, dtype=float), "d": np.asarray(day)})
    # rank trong từng ngày, điểm cao nhất = 1; hòa điểm xử lý theo thứ tự xuất hiện để kết quả xác định
    rank = df.groupby("d")["s"].rank(method="first", ascending=False)
    flagged = rank <= k
    fraud = df["y"] == 1
    n_fraud = fraud.sum()
    fraud_value = df.loc[fraud, "a"].sum()
    return {
        f"recall@{k}": float((flagged & fraud).sum() / n_fraud) if n_fraud else np.nan,
        f"precision@{k}": float((flagged & fraud).sum() / flagged.sum()) if flagged.sum() else np.nan,
        f"vdr@{k}": float(df.loc[flagged & fraud, "a"].sum() / fraud_value) if fraud_value else np.nan,
    }


def best_f1_threshold(y_true, y_score) -> float:
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def evaluate(y_true, y_score, amount, day, threshold: float = 0.5, k: int = TOP_K_PER_DAY) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    res = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_score >= threshold, zero_division=0)),
        "threshold": float(threshold),
    }
    res.update(top_k_daily(y_true, y_score, amount, day, k))
    res["fraud_rate"] = float(y_true.mean())
    return res
