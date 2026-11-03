| Cách | Số fold | PR-AUC | % gộp chung (mục tiêu ≥ 85%) | ε (mục tiêu ≤ 1) | Đạt mục tiêu chính? | VDR@1% (phụ, mục tiêu ≥ 0,80) |
|---|---|---|---|---|---|---|
| central | 4 | 0.0967 ± 0.0098 | 100% ± 0 | — | mốc | 0.35 ❌ |
| fedavg | 4 | 0.0246 ± 0.0005 | 26% ± 3 | — | mốc | 0.00 ❌ |
| dp ε=1 | 1 | 0.0233 ± 0.0000 | 23% ± 0 | 1 | ❌ thiếu 62 điểm % | 0.53 ❌ |
| fedavg_deu | 1 | 0.0282 ± 0.0000 | 28% ± 0 | — | mốc | 0.54 ❌ |

Mốc gộp chung theo fold: fold 0: 0.1006, fold 1: 0.1050, fold 2: 0.0825, fold 3: 0.0985
Nguồn: results\ibm_fl_dp_CHINH_5fold_moc, results\ibm_fl_dp_K802_dp1_prox0.1_C0.35, results\ibm_fl_dp_K802_deu_P2_prox0.1
