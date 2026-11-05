Ghép cặp theo (seed, fold). Seeds: [np.int64(7), np.int64(42)]. p nhỏ nhất có thể của Wilcoxon hai phía: n=8 → 0.0078, n=9 → 0.0039

| Chỉ số | A | B | n | TB A | TB B | Chênh (A−B) [CI 95%] | A thắng | p Wilcoxon | p Holm | r rank-biserial |
|---|---|---|---|---|---|---|---|---|---|---|
| macro_pr_auc | FL + DP ε=1 | Tự học riêng | 8 | 0.2222 | 0.1257 | +0.0964 [+0.0861; +0.1046] | 8/8 | 0.0078 | 0.0234 ✅ | +1.00 |
| macro_pr_auc | FedProx (TB đều) | Tự học riêng | 9 | 0.2374 | 0.1254 | +0.1120 [+0.0980; +0.1242] | 9/9 | 0.0039 | 0.0234 ✅ | +1.00 |
| macro_pr_auc | Gộp chung | FedProx (TB đều) | 9 | 0.3444 | 0.2374 | +0.1070 [+0.0962; +0.1194] | 9/9 | 0.0039 | 0.0234 ✅ | +1.00 |
| macro_pr_auc | FedProx (TB đều) | FL + DP ε=2 | 8 | 0.2395 | 0.2304 | +0.0091 [+0.0025; +0.0155] | 7/8 | 0.0547 | 0.1094 | +0.78 |
| macro_pr_auc | FedProx (TB đều) | FL + DP ε=1 | 8 | 0.2395 | 0.2222 | +0.0174 [-0.0002; +0.0321] | 6/8 | 0.1953 | 0.1953 | +0.56 |
| macro_pr_auc | FedProx (TB đều) | FedAvg thường | 9 | 0.2374 | 0.2027 | +0.0347 [+0.0219; +0.0454] | 9/9 | 0.0039 | 0.0234 ✅ | +1.00 |
