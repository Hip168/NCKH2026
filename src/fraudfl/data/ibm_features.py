"""Đặc trưng dạng bảng cho IBM AML (bản gốc có From Bank / To Bank), tính trên DỮ LIỆU ĐẦY ĐỦ trước khi lấy mẫu (D17).

Nguyên tắc:
- Chỉ dùng giao dịch QUÁ KHỨ: cửa sổ [t − w, t), không gồm giao dịch cùng thời điểm t (file gốc chỉ có độ phân giải
  phút) → không nhìn trước tương lai.
- Không dùng nhãn → không rò rỉ nhãn.
- Số đếm và tổng tiền ở thang log1p (phân phối rất lệch: có tài khoản gửi hàng nghìn giao dịch/tuần).
- Tài khoản định danh bằng (mã ngân hàng, số tài khoản), mã ngân hàng là chuỗi.
- Quy đổi USD bằng tỷ giá suy ra từ chính dữ liệu (giao dịch đổi tiền có Amount Paid và Amount Received ở hai loại
  tiền; độ lệch chuẩn tỷ giá ≈ 0 vì IBM sinh dữ liệu với tỷ giá cố định) → giải quyết Q6.

Hai chế độ `view`:
- "global": dùng cho E7 (gộp chung). Có thêm đặc trưng của tài khoản NHẬN (RCV_*).
- "bank":   dùng cho FL. Chỉ những gì ngân hàng GỬI tự biết: lịch sử gửi và nhận của chính tài khoản gửi (tài khoản này
            thuộc ngân hàng gửi). Bỏ RCV_* vì tài khoản nhận có thể ở ngân hàng khác.
"""

import numpy as np
import pandas as pd

WINDOWS = (1, 7)
PAYMENT_FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire")
_STRIDE = 10**9                      # khóa ghép = mã tài khoản × _STRIDE + giây; 18 ngày ≈ 1,6e6 giây ≪ _STRIDE


def usd_rates(df: pd.DataFrame) -> pd.Series:
    """Số đơn vị tiền tệ cho 1 USD, suy từ giao dịch trả USD nhận tiền khác (trung vị)."""
    x = df[(df["Payment Currency"] == "US Dollar") & (df["Receiving Currency"] != "US Dollar")]
    rates = (x["Amount Received"] / x["Amount Paid"]).groupby(x["Receiving Currency"]).median()
    rates["US Dollar"] = 1.0
    missing = set(df["Payment Currency"]) - set(rates.index)
    if missing:
        raise ValueError(f"Không suy được tỷ giá cho: {sorted(missing)}")
    return rates


def _count_sum_past(ev_key, ev_t, ev_v, q_key, q_t, w_sec):
    """Với mỗi truy vấn (khóa, t): số sự kiện và tổng giá trị của cùng khóa trong [t − w, t)."""
    order = np.lexsort((ev_t, ev_key))
    comp = ev_key[order] * _STRIDE + ev_t[order]
    cs = np.concatenate([[0.0], np.cumsum(ev_v[order])])
    q = q_key * _STRIDE + q_t
    hi = np.searchsorted(comp, q, side="left")
    lo = np.searchsorted(comp, q - w_sec, side="left")
    return (hi - lo).astype(np.float32), (cs[hi] - cs[lo]).astype(np.float64)


def build_ibm_features(df: pd.DataFrame, view: str = "global") -> pd.DataFrame:
    """df: kết quả `ibm_aml.load_raw`. Trả về DataFrame cùng thứ tự dòng, gồm cột khóa + cột đặc trưng."""
    if view not in ("global", "bank"):
        raise ValueError("view phải là 'global' hoặc 'bank'")
    rates = usd_rates(df)
    amount = (df["Amount Paid"] / df["Payment Currency"].map(rates)).to_numpy(np.float64)
    t = ((df["Timestamp"] - df["Timestamp"].min()).dt.total_seconds()).to_numpy(np.int64)

    acct_from = df["From Bank"] + "|" + df["From Account"]
    acct_to = df["To Bank"] + "|" + df["To Account"]
    codes, _ = pd.factorize(pd.concat([acct_from, acct_to], ignore_index=True))
    a_from, a_to = codes[: len(df)].astype(np.int64), codes[len(df):].astype(np.int64)

    out = pd.DataFrame({
        "TX_ID": np.arange(len(df)), "FROM_BANK": df["From Bank"].to_numpy(), "TO_BANK": df["To Bank"].to_numpy(),
        "TX_TIME_DAYS": df["TX_TIME_DAYS"].to_numpy(), "LABEL": df["Is Laundering"].to_numpy().astype(np.int8),
        "AMOUNT_USD": amount,
    })
    out["AMOUNT_USD_LOG"] = np.log1p(amount)
    out["CROSS_CURRENCY"] = (df["Payment Currency"] != df["Receiving Currency"]).to_numpy(np.int8)
    out["SAME_BANK"] = (df["From Bank"] == df["To Bank"]).to_numpy(np.int8)
    out["SELF_TRANSFER"] = (a_from == a_to).astype(np.int8)
    for pf in PAYMENT_FORMATS:
        out[f"PF_{pf.upper().replace(' ', '_')}"] = (df["Payment Format"] == pf).to_numpy(np.int8)
    out["HOUR"] = df["Timestamp"].dt.hour.to_numpy(np.int8)
    out["WEEKEND"] = (df["Timestamp"].dt.weekday >= 5).to_numpy(np.int8)

    # Tuổi tài khoản gửi: thời gian từ lần xuất hiện đầu tiên (gửi hoặc nhận) tới t; lần đầu luôn ≤ t nên không nhìn trước
    first = pd.Series(np.concatenate([t, t])).groupby(codes).min().to_numpy()
    out["SND_AGE_DAYS"] = (t - first[a_from]) / 86400.0

    for w in WINDOWS:
        w_sec = w * 86400
        nb, s = _count_sum_past(a_from, t, amount, a_from, t, w_sec)            # tài khoản gửi đã GỬI
        out[f"SND_OUT_NB_LOG_{w}D"], out[f"SND_OUT_SUM_LOG_{w}D"] = np.log1p(nb), np.log1p(s)
        if w == max(WINDOWS):   # log(giá trị / trung bình gửi 7 ngày); chưa có lịch sử → 0
            avg = s / np.maximum(nb, 1)
            out[f"SND_AMOUNT_LOGRATIO_{w}D"] = np.where(nb > 0, np.log1p(amount) - np.log1p(avg), 0.0)
        nb, s = _count_sum_past(a_to, t, amount, a_from, t, w_sec)              # tài khoản gửi đã NHẬN
        out[f"SND_IN_NB_LOG_{w}D"], out[f"SND_IN_SUM_LOG_{w}D"] = np.log1p(nb), np.log1p(s)
        if view == "global":
            nb, s = _count_sum_past(a_to, t, amount, a_to, t, w_sec)            # tài khoản nhận đã NHẬN
            out[f"RCV_IN_NB_LOG_{w}D"], out[f"RCV_IN_SUM_LOG_{w}D"] = np.log1p(nb), np.log1p(s)
            nb, _ = _count_sum_past(a_from, t, amount, a_to, t, w_sec)          # tài khoản nhận đã GỬI
            out[f"RCV_OUT_NB_LOG_{w}D"] = np.log1p(nb)
    return out


KEY_COLUMNS = ["TX_ID", "FROM_BANK", "TO_BANK", "TX_TIME_DAYS", "LABEL", "AMOUNT_USD"]


# D18: tuổi tài khoản gắn chặt với thời gian trong dữ liệu (proxy của "ngày") → vẫn tính và lưu, nhưng KHÔNG đưa vào
# mô hình. Bỏ chỉ giảm ~0,01 PR-AUC (docs/nhat_ky_ket_qua.md giai đoạn 6).
EXCLUDED_FEATURES = ["SND_AGE_DAYS"]


def feature_columns(feat: pd.DataFrame, include_excluded: bool = False) -> list[str]:
    skip = KEY_COLUMNS + ([] if include_excluded else EXCLUDED_FEATURES)
    return [c for c in feat.columns if c not in skip]
