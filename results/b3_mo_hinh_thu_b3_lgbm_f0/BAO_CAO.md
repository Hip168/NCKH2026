# Báo cáo thử nghiệm B3 — LightGBM, MLP & Đặc trưng đồ thị

## Tổng hợp kết quả (mean ± std qua các seed/fold đã chạy)

| Mô hình | Bộ đặc trưng | PR-AUC | Recall@1% | VDR@1% | Macro PR-AUC | Thời gian (s) |
|---|---|---:|---:|---:|---:|---:|
| lgbm | co_ban | 0.2174 | 0.6011 | 0.0336 | 0.4993 | 814.5 |
| lgbm | do_thi_fl | 0.4183 | 0.6983 | 0.0735 | 0.6191 | 1033.5 |
| lgbm | do_thi_nuoc | 0.4508 | 0.7215 | 0.9913 | 0.6606 | 1243.5 |
| lgbm | do_thi_global | 0.4846 | 0.7204 | 0.9950 | 0.6654 | 1221.0 |

## Ghi chú tính khả dụng trong Học liên kết (FL)

- **FL-safe 100% (14 cột):** `G_SND_FANOUT_LOG_1D, G_SND_FANOUT_LOG_7D, G_SND_FANIN_LOG_1D, G_SND_FANIN_LOG_7D, G_SND_PASS_1H, G_SND_PASS_1D, G_CYCLE_ABA_LOG_7D, G_PAIR_PAST_LOG_7D, G_SND_APPEAR_LOG, G_SND_PASS_RATIO_1H, G_SND_PASS_RATIO_1D, G_SND_IN_OUT_LATENCY_LOG, G_SND_BURST_1D_7D, G_CYCLE_ABA_DIFF_LOG_7D` — ngân hàng gửi tự tính hoàn toàn từ tài khoản mình quản lý.
- **Phụ thuộc phạm vi client (12 cột):** `G_RCV_FANIN_LOG_1D, G_RCV_FANIN_LOG_7D, G_RCV_FANOUT_LOG_1D, G_RCV_FANOUT_LOG_7D, G_RCV_PASS_1H, G_RCV_PASS_1D, G_RCV_RECENT_OUT_1H, G_RCV_RECENT_OUT_1D, G_RCV_APPEAR_LOG, G_RCV_PASS_RATIO_1H, G_RCV_PASS_RATIO_1D, G_RCV_IN_OUT_LATENCY_LOG` — trong `do_thi_bank`/`do_thi_nuoc` chỉ dùng giao dịch liên quan đến client đang xét; trong `do_thi_global` thấy cả giao dịch của tài khoản nhận ở ngân hàng khác (KHÔNG dùng nguyên bản được trong FL thuần túy).
