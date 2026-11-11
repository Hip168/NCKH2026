"""Đặc trưng đồ thị từ dữ liệu giao dịch IBM AML (B3 — chỉ dùng quá khứ, không rò rỉ tương lai).

Nguyên tắc bắt buộc (docs/giao_viec/EM2_B3_mo_hinh.md):
- Chỉ dùng giao dịch quá khứ trong cửa sổ [t - w, t), không gồm giao dịch tại thời điểm t.
- Tách rõ 2 nhóm đặc trưng:
  1. GRAPH_FEATURES_FL: tính thuần trên lịch sử tài khoản gửi và các giao dịch vào/ra tài khoản gửi
     → ngân hàng gửi tự tính được 100% trong Học liên kết (FL), KHÔNG cần thấy ngân hàng khác.
  2. GRAPH_FEATURES_RCV: lịch sử của tài khoản nhận (fan-in, fan-out, chuyển tiếp, số lần xuất hiện)
     → nếu tài khoản nhận thuộc ngân hàng/nước khác, muốn thấy đầy đủ thì PHẢI thấy dữ liệu của
       client khác (không dùng được đầy đủ trong FL thuần túy).

Ba phạm vi (`scope`) để so sánh công bằng:
- "bank":    ngân hàng gửi chỉ thấy giao dịch có ít nhất một đầu thuộc ngân hàng mình (FL ngân hàng).
- "country": nước gửi chỉ thấy giao dịch có ít nhất một đầu thuộc nước mình (FL theo nước).
- "global":  thấy toàn bộ đồ thị (gộp chung / cận trên tuyệt đối).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STRIDE = 10**9
SEC_1H = 3600
SEC_1D = 86400
SEC_7D = 7 * 86400

GRAPH_FEATURES_FL = [
    "G_SND_FANOUT_LOG_1D",
    "G_SND_FANOUT_LOG_7D",
    "G_SND_FANIN_LOG_1D",
    "G_SND_FANIN_LOG_7D",
    "G_SND_PASS_1H",
    "G_SND_PASS_1D",
    "G_CYCLE_ABA_LOG_7D",
    "G_PAIR_PAST_LOG_7D",
    "G_SND_APPEAR_LOG",
]

GRAPH_FEATURES_RCV = [
    "G_RCV_FANIN_LOG_1D",
    "G_RCV_FANIN_LOG_7D",
    "G_RCV_FANOUT_LOG_1D",
    "G_RCV_FANOUT_LOG_7D",
    "G_RCV_PASS_1H",
    "G_RCV_PASS_1D",
    "G_RCV_RECENT_OUT_1H",
    "G_RCV_RECENT_OUT_1D",
    "G_RCV_APPEAR_LOG",
]

ALL_GRAPH_FEATURES = GRAPH_FEATURES_FL + GRAPH_FEATURES_RCV


def _count_past(
    ev_key: np.ndarray,
    ev_t: np.ndarray,
    q_key: np.ndarray,
    q_t: np.ndarray,
    w_sec: int | None = None,
) -> np.ndarray:
    """Số sự kiện của cùng khóa trong [t - w_sec, t) (nếu w_sec=None thì [0, t))."""
    if len(ev_key) == 0:
        return np.zeros(len(q_key), dtype=np.float32)
    comp = np.sort(ev_key.astype(np.int64) * _STRIDE + ev_t.astype(np.int64))
    q_k = q_key.astype(np.int64)
    q_t64 = q_t.astype(np.int64)
    hi = np.searchsorted(comp, q_k * _STRIDE + q_t64, side="left")
    t_lo = np.zeros_like(q_t64) if w_sec is None else np.maximum(q_t64 - w_sec, 0)
    lo = np.searchsorted(comp, q_k * _STRIDE + t_lo, side="left")
    return (hi - lo).astype(np.float32)


def _prepare_distinct_sweepline(
    ev_key: np.ndarray,
    ev_partner: np.ndarray,
    ev_t: np.ndarray,
    w_sec: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo mảng sweep-line (comp, cumsum) cho đếm số đối tác phân biệt trong [t - w_sec, t)."""
    if len(ev_key) == 0:
        return np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64)

    order = np.lexsort((ev_t, ev_partner, ev_key))
    ek = ev_key[order].astype(np.int64)
    ep = ev_partner[order].astype(np.int64)
    et = ev_t[order].astype(np.int64)

    same_prev = np.zeros(len(ek), dtype=bool)
    same_prev[1:] = (ek[1:] == ek[:-1]) & (ep[1:] == ep[:-1]) & (et[1:] == et[:-1])
    keep = ~same_prev
    ek, ep, et = ek[keep], ep[keep], et[keep]

    m = len(ek)
    same_pair = np.zeros(m, dtype=bool)
    same_pair[1:] = (ek[1:] == ek[:-1]) & (ep[1:] == ep[:-1])

    is_start = np.ones(m, dtype=bool)
    is_start[1:] = (~same_pair[1:]) | ((et[1:] - et[:-1]) > w_sec)

    is_end = np.ones(m, dtype=bool)
    is_end[:-1] = (~same_pair[1:]) | ((et[1:] - et[:-1]) > w_sec)

    sw_key = np.concatenate([ek[is_start], ek[is_end]])
    sw_t = np.concatenate([et[is_start] + 1, et[is_end] + w_sec + 1])
    sw_delta = np.concatenate([
        np.ones(int(is_start.sum()), dtype=np.int32),
        -np.ones(int(is_end.sum()), dtype=np.int32),
    ])
    sw_comp = sw_key * _STRIDE + sw_t
    sw_order = np.argsort(sw_comp, kind="mergesort")
    sw_comp = sw_comp[sw_order]
    cs = np.concatenate([[0], np.cumsum(sw_delta[sw_order], dtype=np.int64)])
    return sw_comp, cs


def _query_sweepline(sw_comp: np.ndarray, cs: np.ndarray, q_key: np.ndarray, q_t: np.ndarray) -> np.ndarray:
    if len(sw_comp) == 0:
        return np.zeros(len(q_key), dtype=np.float32)
    q_comp = q_key.astype(np.int64) * _STRIDE + q_t.astype(np.int64)
    pos = np.searchsorted(sw_comp, q_comp, side="right")
    return cs[pos].astype(np.float32)


def _count_distinct_past(
    ev_key: np.ndarray,
    ev_partner: np.ndarray,
    ev_t: np.ndarray,
    q_key: np.ndarray,
    q_t: np.ndarray,
    w_sec: int,
) -> np.ndarray:
    """Số đối tác phân biệt của cùng khóa trong [t - w_sec, t)."""
    sw_comp, cs = _prepare_distinct_sweepline(ev_key, ev_partner, ev_t, w_sec)
    return _query_sweepline(sw_comp, cs, q_key, q_t)


def _encode_observer_accounts(
    obs_from: np.ndarray,
    obs_to: np.ndarray,
    a_from: np.ndarray,
    a_to: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tạo sự kiện theo góc nhìn từng client quan sát (bank hoặc country).

    Giao dịch (a_from tại obs_from -> a_to tại obs_to) xuất hiện trong sổ của obs_from
    và (nếu khác client) cũng xuất hiện trong sổ của obs_to.
    Trả về (ev_obs_from_acc, ev_obs_to_acc, ev_t, q_obs_send_acc, q_obs_rcv_acc).
    """
    n = len(t)
    diff = obs_from != obs_to
    n_acc = int(max(a_from.max(), a_to.max())) + 1

    # Gán mã số duy nhất cho cặp (observer, account)
    # Luôn cần khóa truy vấn của observer gửi cho cả a_from và a_to, plus bản sao tại observer nhận khi diff
    k_of_f = obs_from * n_acc + a_from
    k_of_t = obs_from * n_acc + a_to
    k_ot_f = obs_to[diff] * n_acc + a_from[diff]
    k_ot_t = obs_to[diff] * n_acc + a_to[diff]

    all_keys = np.concatenate([k_of_f, k_of_t, k_ot_f, k_ot_t])
    codes, _ = pd.factorize(all_keys, sort=False)
    q_of_f = codes[:n].astype(np.int64)
    q_of_t = codes[n : 2 * n].astype(np.int64)
    m = int(diff.sum())
    ev_ot_f = codes[2 * n : 2 * n + m].astype(np.int64)
    ev_ot_t = codes[2 * n + m :].astype(np.int64)

    ev_from = np.concatenate([q_of_f, ev_ot_f])
    ev_to = np.concatenate([q_of_t, ev_ot_t])
    ev_t = np.concatenate([t, t[diff]])
    return ev_from, ev_to, ev_t, q_of_f, q_of_t


def build_graph_features(
    df: pd.DataFrame,
    scope: str = "global",
    bank_to_country: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Tính 18 đặc trưng đồ thị cho dữ liệu IBM AML (df từ `ibm_aml.load_raw`).

    Args:
        df: DataFrame gốc từ `load_raw`.
        scope: "global" (toàn mạng), "bank" (FL theo ngân hàng), "country" (FL theo nước).
        bank_to_country: ánh xạ mã ngân hàng -> tên nước (bắt buộc khi scope="country").
    """
    if scope not in ("global", "bank", "country"):
        raise ValueError("scope phải là 'global', 'bank' hoặc 'country'")
    if len(df) == 0:
        return pd.DataFrame({"TX_ID": np.empty(0, dtype=np.int64), **{c: np.empty(0, dtype=np.float32) for c in ALL_GRAPH_FEATURES}})

    t = ((df["Timestamp"] - df["Timestamp"].min()).dt.total_seconds()).to_numpy(np.int64)
    if int(t.max()) + SEC_7D + 10 >= _STRIDE:
        raise ValueError("Khoảng thời gian vượt quá giới hạn _STRIDE")

    acct_from = df["From Bank"].astype(str) + "|" + df["From Account"].astype(str)
    acct_to = df["To Bank"].astype(str) + "|" + df["To Account"].astype(str)
    codes, _ = pd.factorize(pd.concat([acct_from, acct_to], ignore_index=True))
    n = len(df)
    a_from = codes[:n].astype(np.int64)
    a_to = codes[n:].astype(np.int64)

    out = pd.DataFrame({"TX_ID": np.arange(n, dtype=np.int64)})

    # ---- Nhóm 1: Đặc trưng phía tài khoản gửi + cặp với tài khoản gửi (FL-safe) ----
    # Khi không phải global, ngân hàng/nước gửi vẫn luôn thấy toàn bộ giao dịch gửi và nhận của a_from.
    for w_label, w_sec in (("1D", SEC_1D), ("7D", SEC_7D)):
        sw_out, cs_out = _prepare_distinct_sweepline(a_from, a_to, t, w_sec)
        out[f"G_SND_FANOUT_LOG_{w_label}"] = np.log1p(_query_sweepline(sw_out, cs_out, a_from, t)).astype(np.float32)

        sw_in, cs_in = _prepare_distinct_sweepline(a_to, a_from, t, w_sec)
        out[f"G_SND_FANIN_LOG_{w_label}"] = np.log1p(_query_sweepline(sw_in, cs_in, a_from, t)).astype(np.float32)

        if scope == "global":
            out[f"G_RCV_FANOUT_LOG_{w_label}"] = np.log1p(_query_sweepline(sw_out, cs_out, a_to, t)).astype(np.float32)
            out[f"G_RCV_FANIN_LOG_{w_label}"] = np.log1p(_query_sweepline(sw_in, cs_in, a_to, t)).astype(np.float32)

    # Tài khoản gửi vừa nhận tiền trước khi gửi tiếp (pass-through)
    out["G_SND_PASS_1H"] = (_count_past(a_to, t, a_from, t, SEC_1H) > 0).astype(np.float32)
    out["G_SND_PASS_1D"] = (_count_past(a_to, t, a_from, t, SEC_1D) > 0).astype(np.float32)

    # Chu trình ngắn A -> B -> A và lặp cặp A -> B trong 7 ngày
    n_acc = int(max(a_from.max(), a_to.max())) + 1
    pair_ab_raw = a_from * n_acc + a_to
    pair_ba_raw = a_to * n_acc + a_from
    pair_codes, _ = pd.factorize(np.concatenate([pair_ab_raw, pair_ba_raw]))
    p_ab, p_ba = pair_codes[:n].astype(np.int64), pair_codes[n:].astype(np.int64)
    out["G_CYCLE_ABA_LOG_7D"] = np.log1p(_count_past(p_ab, t, p_ba, t, SEC_7D)).astype(np.float32)
    out["G_PAIR_PAST_LOG_7D"] = np.log1p(_count_past(p_ab, t, p_ab, t, SEC_7D)).astype(np.float32)

    # Tổng số lần xuất hiện trước đó của tài khoản gửi
    both_acc = np.concatenate([a_from, a_to])
    both_t = np.concatenate([t, t])
    out["G_SND_APPEAR_LOG"] = np.log1p(_count_past(both_acc, both_t, a_from, t, None)).astype(np.float32)

    # ---- Nhóm 2: Đặc trưng tài khoản nhận ----
    if scope == "global":
        rcv_in_1h = _count_past(a_to, t, a_to, t, SEC_1H)
        rcv_out_1h = _count_past(a_from, t, a_to, t, SEC_1H)
        rcv_in_1d = _count_past(a_to, t, a_to, t, SEC_1D)
        rcv_out_1d = _count_past(a_from, t, a_to, t, SEC_1D)
        out["G_RCV_PASS_1H"] = ((rcv_in_1h > 0) & (rcv_out_1h > 0)).astype(np.float32)
        out["G_RCV_PASS_1D"] = ((rcv_in_1d > 0) & (rcv_out_1d > 0)).astype(np.float32)
        out["G_RCV_RECENT_OUT_1H"] = (rcv_out_1h > 0).astype(np.float32)
        out["G_RCV_RECENT_OUT_1D"] = (rcv_out_1d > 0).astype(np.float32)
        out["G_RCV_APPEAR_LOG"] = np.log1p(_count_past(both_acc, both_t, a_to, t, None)).astype(np.float32)
    else:
        if scope == "bank":
            obs_series = pd.concat([df["From Bank"].astype(str), df["To Bank"].astype(str)], ignore_index=True)
        else:
            if bank_to_country is None:
                raise ValueError("Cần truyền bank_to_country khi scope='country'")
            fb_c = df["From Bank"].astype(str).map(bank_to_country)
            tb_c = df["To Bank"].astype(str).map(bank_to_country)
            if fb_c.isna().any() or tb_c.isna().any():
                raise KeyError("Có mã ngân hàng không tra được nước trong bank_to_country")
            obs_series = pd.concat([fb_c, tb_c], ignore_index=True)
        obs_codes, _ = pd.factorize(obs_series)
        obs_from, obs_to = obs_codes[:n].astype(np.int64), obs_codes[n:].astype(np.int64)

        ev_from, ev_to, ev_t, _, q_rcv = _encode_observer_accounts(obs_from, obs_to, a_from, a_to, t)
        # Lưu ý: giữ partner ở mã tài khoản gốc (a_from/a_to) để đếm đối tác khác nhau chính xác
        diff = obs_from != obs_to
        ev_partner_to = np.concatenate([a_to, a_to[diff]])
        ev_partner_from = np.concatenate([a_from, a_from[diff]])

        for w_label, w_sec in (("1D", SEC_1D), ("7D", SEC_7D)):
            out[f"G_RCV_FANIN_LOG_{w_label}"] = np.log1p(
                _count_distinct_past(ev_to, ev_partner_from, ev_t, q_rcv, t, w_sec)
            ).astype(np.float32)
            out[f"G_RCV_FANOUT_LOG_{w_label}"] = np.log1p(
                _count_distinct_past(ev_from, ev_partner_to, ev_t, q_rcv, t, w_sec)
            ).astype(np.float32)

        rcv_in_1h = _count_past(ev_to, ev_t, q_rcv, t, SEC_1H)
        rcv_out_1h = _count_past(ev_from, ev_t, q_rcv, t, SEC_1H)
        rcv_in_1d = _count_past(ev_to, ev_t, q_rcv, t, SEC_1D)
        rcv_out_1d = _count_past(ev_from, ev_t, q_rcv, t, SEC_1D)
        out["G_RCV_PASS_1H"] = ((rcv_in_1h > 0) & (rcv_out_1h > 0)).astype(np.float32)
        out["G_RCV_PASS_1D"] = ((rcv_in_1d > 0) & (rcv_out_1d > 0)).astype(np.float32)
        out["G_RCV_RECENT_OUT_1H"] = (rcv_out_1h > 0).astype(np.float32)
        out["G_RCV_RECENT_OUT_1D"] = (rcv_out_1d > 0).astype(np.float32)
        ev_both_k = np.concatenate([ev_from, ev_to])
        ev_both_t = np.concatenate([ev_t, ev_t])
        out["G_RCV_APPEAR_LOG"] = np.log1p(_count_past(ev_both_k, ev_both_t, q_rcv, t, None)).astype(np.float32)

    return out[["TX_ID"] + ALL_GRAPH_FEATURES]
