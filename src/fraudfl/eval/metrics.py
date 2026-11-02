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

Trọng số (`weight`, tùy chọn): dùng khi dữ liệu đã lấy mẫu con (D17). Mỗi ca bình thường được bốc với tỷ lệ r thì
mang trọng số 1/r, ca gian lận giữ trọng số 1. Khi đó mọi chỉ số ước lượng đúng giá trị trên dữ liệu đầy đủ:
PR-AUC/ROC-AUC/F1 tính có trọng số; ngân sách cảnh báo mỗi ngày = ceil(p × tổng trọng số của ngày), giao dịch được
gắn cờ theo thứ tự điểm cho tới khi hết ngân sách. Không truyền `weight` thì kết quả giống hệt bản không trọng số.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from fraudfl.config import ALERT_FRACS


def _pct(frac: float) -> str:
    return f"{frac * 100:g}%"


def top_frac_daily(y_true, y_score, amount, day, frac: float, weight=None) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    a = np.asarray(amount, dtype=float)
    d = np.asarray(day)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    # Sắp theo (ngày, điểm giảm dần). Hòa điểm phá bằng hoán vị ngẫu nhiên seed cố định (#41): phá theo thứ tự dòng
    # làm mô hình đã sụp (điểm gần như bằng nhau) vẫn có Recall@p cao giả nếu thứ tự dòng tương quan với nhãn.
    tie = np.random.default_rng(0).permutation(len(y))
    order = np.lexsort((tie, -s, d))
    d_o, w_o = d[order], w[order]
    cum = pd.Series(w_o).groupby(d_o).cumsum().to_numpy()             # trọng số cộng dồn trong ngày
    budget = np.ceil(frac * pd.Series(w_o).groupby(d_o).transform("sum").to_numpy())
    flagged = np.empty(len(y), dtype=bool)
    flagged[order] = cum - w_o < budget                                # gắn cờ khi còn ngân sách lúc tới lượt
    fraud = y == 1
    fw = w * fraud
    n_fraud, n_flag = fw.sum(), (w * flagged).sum()
    fraud_value = (a * fw).sum()
    p = _pct(frac)
    return {
        f"recall@{p}": float((fw * flagged).sum() / n_fraud) if n_fraud else np.nan,
        f"precision@{p}": float((fw * flagged).sum() / n_flag) if n_flag else np.nan,
        f"vdr@{p}": float((a * fw * flagged).sum() / fraud_value) if fraud_value else np.nan,
    }


def best_f1_threshold(y_true, y_score) -> float:
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def evaluate(y_true, y_score, amount, day, threshold: float = 0.5, fracs=ALERT_FRACS, weight=None) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    w = None if weight is None else np.asarray(weight, dtype=float)
    res = {
        "pr_auc": float(average_precision_score(y_true, y_score, sample_weight=w)),
        "roc_auc": float(roc_auc_score(y_true, y_score, sample_weight=w)),
        "f1": float(f1_score(y_true, y_score >= threshold, zero_division=0, sample_weight=w)),
        "threshold": float(threshold),
    }
    for frac in fracs:
        res.update(top_frac_daily(y_true, y_score, amount, day, frac, weight=w))
    res["fraud_rate"] = float(np.average(y_true, weights=w))
    return res


MAIN_METRICS = ["pr_auc", "roc_auc", "f1"] + [f"{m}@{_pct(f)}" for f in ALERT_FRACS
                                              for m in ("recall", "precision", "vdr")]
