# Kết quả thí nghiệm — bản 2026-11-01

Dữ liệu synthetic 5 ngân hàng (D5), 5 outer fold, seed 42, **1 seed** (chưa đủ 3 seed theo Slide 8b).
Chỉ số theo D1, D2, D7. Số liệu gốc: `results/*/per_fold.csv`. Biểu đồ: `results/figures/`.

## RQ1 — FL so với centralized và local-only (E1, E2, E3; GraphSAGE + MLP)

| Thí nghiệm | PR-AUC | Recall@1% | VDR@1% | F1 | % của centralized (PR-AUC) |
|---|---|---|---|---|---|
| E1 Centralized | 0,733 ± 0,014 | 0,548 ± 0,018 | 0,799 ± 0,015 | 0,714 ± 0,017 | 100% |
| E3 FedAvg | 0,715 ± 0,013 | 0,539 ± 0,017 | 0,793 ± 0,014 | 0,702 ± 0,017 | 97,6% (97,1–98,0%) |
| E2 Local-only | 0,706 ± 0,011 | 0,535 ± 0,013 | 0,786 ± 0,012 | 0,693 ± 0,011 | 96,3% (95,4–96,7%) |

- Thứ tự Centralized > FedAvg > Local-only đúng ở **5/5 fold**.
- Khoảng cách nhỏ (1–3 điểm PR-AUC). Với 5 cặp, Wilcoxon hai phía không thể đạt p < 0,05 (p nhỏ nhất = 0,0625)
  → cần thêm 2 seed (15 cặp) trước khi khẳng định ý nghĩa thống kê.
- FedAvg chưa bão hòa hẳn ở vòng 50 (`results/figures/fedavg_convergence.png`) → con số FedAvg có thể hơi thấp
  hơn khả năng thật.

## RQ2 — Đánh đổi riêng tư – hiệu năng (E4, E5)

### Record-level DP (DP-SGD trong từng ngân hàng, MLP dạng bảng, FedAvg 30 vòng)

| ε (δ = 1e-5) | Hệ số nhiễu z | PR-AUC | Recall@1% | VDR@1% | % so với không DP | % của E1 centralized |
|---|---|---|---|---|---|---|
| ∞ (không DP) | 0 | 0,708 ± 0,011 | 0,539 | 0,802 | 100% | 96,6% |
| 2 | 2,07 | 0,672 ± 0,014 | 0,510 | 0,787 | 94,9% | 91,7% |
| **1** | **3,70** | **0,657 ± 0,014** | **0,503** | **0,780** | **92,7%** | **89,6% ± 0,4%** |
| 0,5 | 6,90 | 0,625 ± 0,018 | 0,489 | 0,778 | 88,2% | 85,3% |

→ Đạt tiêu chí Slide 10 ("FL+DP ≥ 85% centralized", ε ≤ 1, δ ≤ 1e-5). VDR gần như giữ nguyên (0,80 → 0,78):
các vụ gian lận giá trị lớn vẫn bị bắt.

### Client-level DP (đúng thiết kế Slide 4, GraphSAGE, fold 0)

| ε | Hệ số nhiễu z | PR-AUC |
|---|---|---|
| 2 | 15,2 | 0,028 |
| 1 | 28,8 | 0,028 |
| 0,5 | 55,0 | 0,028 |

→ Mô hình **sụp hoàn toàn** (tỷ lệ gian lận ~1,5%; PR-AUC 0,028 ≈ đoán mò). Với 5 client, bảo vệ toàn bộ dữ liệu
một ngân hàng ở ε ≤ 2 là không khả thi. Đây là kết quả có giá trị cho bài báo: lý giải vì sao chọn DP mức bản ghi.

## So với giả thuyết ở Slide 7

| | Slide 7 (kỳ vọng) | Thực tế |
|---|---|---|
| Thứ tự | Centralized > FedAvg > FL+DP > Local | Centralized > FedAvg > Local > FL+DP (ε=1) |
| FL+DP ε=1 / centralized | 85–90% | 89,6% |
| Local / centralized | ~60% | 96% |

Local-only mạnh hơn kỳ vọng nhiều vì mỗi ngân hàng có ~100k giao dịch — đủ để tự học tốt. Lợi ích của FL sẽ
rõ hơn khi có ngân hàng nhỏ (ít dữ liệu). Gợi ý thí nghiệm bổ sung: giảm dữ liệu 1–2 ngân hàng xuống 5–10k.

## Việc còn thiếu trước khi viết bài

1. Thêm 2 seed cho E1–E4 → Wilcoxon / Friedman + Holm, effect size (Slide 8b).
2. FedAvg 80–100 vòng để chắc đã hội tụ.
3. E6 FedProx; thí nghiệm ngân hàng nhỏ.
4. E7 trên IBM AML (cần chốt Q5, Q6).
