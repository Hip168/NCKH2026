# Nhật ký kết quả theo giai đoạn

Tài liệu dùng để viết **báo cáo cả quá trình** lúc kết thúc đề tài. Mỗi giai đoạn ghi: đã làm gì, kết quả chính, kết
quả lưu ở đâu, và rút ra điều gì.

## Quy tắc lưu trữ (từ 02/11/2026)

1. **Không xóa kết quả cũ.** Thư mục kết quả chính (`results/<tên>/`) là bản mới nhất. Mỗi lần chạy script tự chép
   một bản vào `results/_luu_tru/<YYYY-MM-DD_HHMM>_<tên>/`, kèm `meta.json` (lệnh chạy, commit git, ghi chú).
2. **Sổ `results/NHAT_KY_CHAY.csv`** có một dòng cho mỗi lần chạy. Không sửa tay các dòng cũ.
3. **Commit `results/` (gồm `_luu_tru/` và sổ nhật ký) lên git** sau mỗi đợt chạy. Dữ liệu lớn (`data/raw`,
   `data/processed`) không commit.
4. **Kết thúc mỗi giai đoạn, thêm một mục vào tài liệu này**, kể cả kết quả xấu hoặc thí nghiệm thất bại.
5. Script mới phải gọi `fraudfl.runlog.archive_run(thu_muc_ket_qua, note=...)` ở cuối.

Kết quả chạy **trước ngày 02/11** chưa có kho lưu trữ; bản gốc nằm trong lịch sử git ở commit ghi trong từng mục.

---

## Giai đoạn 1 — Pipeline nền (31/10/2026)

- **Làm:** sinh dữ liệu synthetic 5 ngân hàng × khoảng 100 nghìn giao dịch, 15 đặc trưng, chia fold nested 5×3,
  EDA, module chỉ số, baseline RF/LR. Áp đề xuất D7–D9.
- **Kết quả:** RF với đặc trưng v1 (D8): PR-AUC 0,694 gộp chung, 0,690 tự học riêng.
- **Lưu ở:** commit `b62a631` và `97be662` (`results/baseline_tabular/`, `results/eda/`).
- **Rút ra:** đặc trưng tương đối (D8) gần như xóa khoảng cách gộp chung – tự học riêng.

## Giai đoạn 2 — E1–E5 (01/11/2026)

- **Làm:** GraphSAGE gộp chung / tự học riêng / FedAvg; DP mức client và mức bản ghi.
- **Kết quả (5 fold, 1 seed):** E1 gộp chung 0,733; E3 FedAvg 0,715; E2 tự học riêng 0,706. DP mức bản ghi ở
  ε = 1: 0,657 (89,6% so với gộp chung). DP mức client: 0,028 ở mọi ε ≤ 2.
- **Lưu ở:** commit `97be662` (`results/gnn_synthetic/`, `results/dp_record/`, `results/dp_client/`), tổng hợp ở
  `docs/results.md`.
- **Rút ra:** DP mức client với 5 ngân hàng làm mô hình sụp. Lợi ích của FL so với tự học riêng nhỏ.

## Giai đoạn 3 — Hạ tầng FL + real-time (02/11/2026, Đạo)

- **Làm:** Flower 1.42, Kafka, Redis, Postgres, Docker. Merge vào `main` qua PR #1 (`a06295e`).
- **Kết quả:** Flower khớp vòng lặp mô phỏng (PR-AUC 0,7188 so với 0,7168). Không mất hay trùng giao dịch khi tắt đột
  ngột consumer. Độ trễ chấm điểm trung vị 9,4 ms.
- **Lưu ở:** `docs/ha_tang.md`.

## Giai đoạn 4 — Chốt DP mức client, rà soát đánh giá (02/11/2026)

- **Quyết định:** giữ DP mức client (D12, `docs/luu_y_dp_muc_client.md`); lấy mẫu con IBM (D17, anh Hùng duyệt).
- **Rà soát code** phát hiện thêm: fold chia ngẫu nhiên thay vì theo thời gian (#32); điểm tự học riêng bị xếp hạng
  chung (#33); ngân sách cảnh báo tính chung 5 ngân hàng (#34). Đã thêm chỉ số theo từng ngân hàng (`macro_*`,
  `c<id>_*`) và chỉ số có trọng số cho dữ liệu lấy mẫu.
- **Thử nhanh DP mức client** (fold 0, 10 vòng, 5 ngân hàng, PR-AUC):

  | ε | A: nhiễu ở server, GNN nhỏ | B: nhiễu ở từng ngân hàng, GNN nhỏ | C: nhiễu ở server, GNN gốc |
  |---|---|---|---|
  | ∞ | 0,587 | 0,587 | 0,647 |
  | 8 | 0,069 | 0,019 | 0,021 |
  | 4 | 0,022 | 0,018 | 0,020 |
  | 1 | 0,018 | 0,019 | 0,020 |

  Lưu ở `results/_luu_tru/2026-11-02_2225_dp_client_nhanh_*`.
- **Rút ra:** với 5 ngân hàng, đổi chỗ cộng nhiễu, giảm vòng hay thu nhỏ mô hình đều không đủ — sụp cả ở ε = 8.
  Cần tăng nhiều số ngân hàng tham gia.

## Giai đoạn 5 — Dữ liệu IBM AML HI-Small (02/11/2026)

- **Dữ liệu:** 5.078.345 giao dịch, 5.177 ca rửa tiền (0,102%), 18 ngày, ~~30.528~~ 30.470 ngân hàng gửi (đính chính giai đoạn 12). Top 5 ngân hàng giữ
  13,6% giao dịch; chỉ 34 ngân hàng có ≥ 20 ca rửa tiền. Ca rửa tiền dồn vào 10 ngày đầu.
- ~~**Sửa lỗi:** `load_raw` đọc mã ngân hàng thành số làm gộp nhầm 58 ngân hàng (`"010"` = `"10"`). Đã sửa, commit
  `cabad04`.~~ **ĐÍNH CHÍNH ở giai đoạn 12:** đây KHÔNG phải lỗi — `"010"` và `"0010"` là cùng một ngân hàng; số đúng
  là 30.470 ngân hàng.
- **Tỷ giá (Q6):** suy từ chính dữ liệu, độ lệch chuẩn ≈ 0 — `results/ibm_features_global/ty_gia.csv`.
- **Đặc trưng:** 29 đặc trưng (chế độ global, cho E7), 23 đặc trưng (chế độ bank, cho FL); chỉ dùng quá khứ, không dùng
  nhãn. Đặc trưng mạnh: hình thức ACH (87% ca rửa tiền so với 12% ca thường), cuối tuần, mức hoạt động của tài khoản
  gửi; rửa tiền gần như không có ở Reinvestment, Wire, đổi tiền, tự chuyển cho mình.
- **Lưu ở:** `results/_luu_tru/2026-11-02_2224_ibm_features_global`, `..._2225_ibm_features_bank` (bản dùng);
  `..._2221_...`, `..._2222_...` là bản đầu trước khi chuyển số đếm sang thang log.
- **Chờ:** chốt r với anh Hùng rồi chạy `05_ibm_sample.py`.

## Giai đoạn 6 — Đánh giá đặc trưng IBM HI-Small (02/11/2026)

- **Làm:** `scripts/07_ibm_kiem_tra_dac_trung.py` — kiểm tra sạch dữ liệu, sức phân biệt đơn biến trên 5,08 triệu
  giao dịch, mô hình thử HistGradientBoosting trên mẫu TẠM (r = 0,02, chia ngẫu nhiên 80/20; không phải mẫu D17).
- **Sạch dữ liệu:** 0 giá trị thiếu, 0 vô cực.
- **Phát hiện lỗi dữ liệu:** từ ngày 10 đến 17 chỉ còn khoảng 1.100 giao dịch nhưng khoảng 60% là rửa tiền (các ngày
  trước: 0,03–0,21%). 715 ca rửa tiền (13,8%) nằm ở đuôi này. Đặc trưng gắn với thời gian (tuổi tài khoản, AP đơn biến
  0,072 so với mức ngẫu nhiên 0,001) bị rò rỉ. → đề xuất D18: chỉ dùng ngày 0–9.
- **Mô hình thử (PR-AUC trên test có trọng số 1/r, tức tỷ lệ thật khoảng 0,09–0,1%):**

  | Dữ liệu | View | PR-AUC có trọng số | PR-AUC không trọng số | ROC-AUC | Recall@1% | VDR@1% |
  |---|---|---|---|---|---|---|
  | Cả 18 ngày | global | 0,347 | 0,821 | 0,981 | 0,565 | 0,545 |
  | Cả 18 ngày | bank | 0,285 | 0,781 | 0,978 | 0,489 | 0,518 |
  | Ngày 0–9 | global | 0,304 | 0,774 | 0,977 | 0,653 | 0,102 |
  | Ngày 0–9 | bank | 0,224 | 0,729 | 0,974 | 0,566 | 0,043 |
  | Ngày 0–9, bỏ tuổi TK | global | 0,293 | 0,770 | 0,977 | 0,645 | 0,077 |
  | Ngày 0–9, bỏ tuổi TK | bank | 0,223 | 0,716 | 0,973 | 0,551 | 0,043 |

- **Lưu ở:** `results/_luu_tru/2026-11-02_2231_ibm_kiem_tra_dac_trung`, `..._2232_..._ngay0_9`,
  `..._2233_..._ngay0_9_bo_tuoi`.
- **Rút ra:**
  1. Đặc trưng có tín hiệu thật. ~~PR-AUC khoảng 0,29~~ — **ĐÍNH CHÍNH ở giai đoạn 9:** con số đúng (5 fold, test
     đầy đủ) là 0,209 ± 0,018.
  2. PR-AUC không trọng số cao giả khoảng 2,5 lần → bắt buộc đánh giá có trọng số (D17).
  3. Đuôi dữ liệu thổi phồng VDR (0,10 → 0,55) vì ca rửa tiền ở đuôi có giá trị lớn.
  4. Chế độ bank (cho FL) kém global khoảng 0,07 PR-AUC — đây là giá của việc không thấy thông tin tài khoản nhận.
  5. VDR thấp (0,04–0,10) dù Recall khoảng 0,6: mô hình bắt được nhiều ca nhưng bỏ sót ca giá trị lớn.

## Giai đoạn 7 — Chốt D18 và chạy thử lấy mẫu IBM (02/11/2026)

- **Quyết định:** D18 — IBM HI-Small chỉ dùng ngày 0–9; bỏ `SND_AGE_DAYS` khỏi đầu vào mô hình. Đạt chốt, cần báo
  lại anh Hùng.
- **Sửa `05_ibm_sample.py`:**
  1. Lọc ngày ≤ 9 trước khi lấy mẫu; `TX_ID` vẫn là số dòng trong file gốc.
  2. Lỗi: sau khi mã ngân hàng thành chuỗi, nhóm "ngân hàng nhỏ" (-1) không còn so khớp được → đổi sang nhãn `_nho`.
  3. Lỗi thiết kế: quy tắc "mỗi ngân hàng giữ ít nhất 1 ca bình thường" áp lên ~30 nghìn ngân hàng tí hon làm mẫu phình
     từ ~106 nghìn lên 130 nghìn. Sửa: ngân hàng có ≥ 1.000 ca bình thường lấy mẫu riêng, còn lại gộp nhóm `_nho`.
- **Chạy thử trên HI-Small thật, r = 0,02, ghi ra thư mục tạm (CHƯA phải mẫu chính thức vì r chưa chốt):**
  105.973 giao dịch, 4.522 ca rửa tiền (4,27% trong mẫu; 0,089% thật), 806 ngân hàng lấy mẫu riêng + nhóm `_nho`
  (4.905 ngân hàng); mỗi fold test 895–913 ca rửa tiền; tổng trọng số ca bình thường = 5.072.715, khớp đúng số thật.
- **Chờ:** chốt r, rồi chạy chính thức vào `data/splits/` và commit.

## Giai đoạn 8 — Đường học theo r và phát hiện: tập test lấy mẫu nhỏ cho PR-AUC lạc quan (02/11/2026)

- **Làm:** `07_ibm_kiem_tra_dac_trung.py` với r ∈ {0,01; 0,02; 0,05; 0,1} (ngày 0–9, bỏ tuổi tài khoản, cách gộp
  ngân hàng nhỏ giống `05`). Lưu ở `results/_luu_tru/2026-11-02_*_ibm_kiem_tra_dac_trung_duong_hoc_r*`.

  | r | PR-AUC có trọng số (global) | (bank) | VDR@1% (global) |
  |---|---|---|---|
  | 0,01 | 0,186 | 0,097 | 0,895 |
  | 0,02 | 0,294 | 0,189 | 0,728 |
  | 0,05 | 0,088 | 0,079 | 0,977 |
  | 0,10 | 0,095 | 0,081 | 0,029 |

  r lớn hơn mà PR-AUC giảm, VDR nhảy loạn → bất thường.
- **Chẩn đoán (chạy trực tiếp, không lưu trữ thành thư mục; số liệu ghi ở đây):**
  - Đổi seed chia test (42, 7) và bật/tắt early stopping: r = 0,02 cho 0,25–0,29; r = 0,1 cho 0,07–0,10 → không phải
    do một lần chia may rủi.
  - **Chấm chéo trên CÙNG tập test r = 0,1** (bỏ giao dịch đã dùng để train): mô hình học với r = 0,02 đạt **0,104**,
    mô hình học với r = 0,1 đạt **0,095**.
  - Các giao dịch bình thường bị chấm điểm cao nhất: không phải ACH, tài khoản gửi gần như chưa có lịch sử, số tiền lớn
    (1.700–44.000 USD) — "trông giống rửa tiền".
- **⚠ Kết luận dưới đây đã được ĐÍNH CHÍNH ở giai đoạn 9** (so sánh ở đây lẫn hai yếu tố: khác mô hình và khác tập
  test; một lần chia ngẫu nhiên). Giữ nguyên để lưu vết.
- **Rút ra (cũ):**
  1. **Để HUẤN LUYỆN, r = 0,02 là đủ**: mô hình tốt ngang r = 0,1.
  2. **Để ĐÁNH GIÁ, tập test lấy mẫu r = 0,02 KHÔNG tin được**: PR-AUC phụ thuộc vào vài ca bình thường "khó", với
     r nhỏ phần lớn các ca này không lọt vào test → con số lạc quan khoảng 3 lần (0,29 so với ~0,10). Trọng số 1/r
     sửa được tỷ lệ trung bình nhưng không sửa được độ dao động này.
  3. → Đề xuất sửa D17: **lấy mẫu chỉ cho train/validation; tập test giữ TOÀN BỘ ca bình thường của fold đó**
     (hoặc r_test lớn, ≥ 0,2). Mô hình dạng bảng chấm 1 triệu dòng mất vài giây.
  4. VDR@1% dao động rất mạnh giữa các lần (0,03–0,98) vì VDR bị chi phối bởi một vài ca rửa tiền giá trị rất lớn →
     cần báo cáo VDR kèm khoảng dao động, và cũng cần tập test đầy đủ.

## Giai đoạn 9 — Bằng chứng 5 fold cho cách lấy mẫu IBM (02/11/2026)

- **Làm:** `scripts/08_ibm_danh_gia_lay_mau.py` — ngày 0–9, 5 fold phân tầng (seed 42). Mỗi fold: 20% phần train làm
  validation ĐẦY ĐỦ (chọn ngưỡng F1); 80% còn lại lấy mẫu r_train ∈ {0,02; 0,05; 0,1; 0,2} → HistGradientBoosting;
  chấm trên TEST ĐẦY ĐỦ và trên test lấy mẫu r_test ∈ {0,02; 0,1} có trọng số. Hai vòng: không trọng số lớp và trọng số
  lớp cân bằng. Lưu ở `results/_luu_tru/2026-11-02_2308_ibm_danh_gia_lay_mau`,
  `..._2310_ibm_danh_gia_lay_mau_cw_balanced`.
- **PR-AUC trên test đầy đủ (trung bình ± độ lệch 5 fold):**

  | r_train | Không trọng số lớp | Trọng số lớp cân bằng |
  |---|---|---|
  | 0,02 | **0,209 ± 0,018** | 0,165 ± 0,017 |
  | 0,05 | 0,118 ± 0,017 | 0,168 ± 0,008 |
  | 0,10 | 0,111 ± 0,010 | 0,173 ± 0,009 |
  | 0,20 | 0,113 ± 0,028 | 0,161 ± 0,009 |

  F1 lớp rửa tiền tốt nhất 0,284 (r = 0,02, không trọng số lớp); Recall@1% 0,63.
- **Test lấy mẫu so với test đầy đủ (cùng mô hình):** trung bình chỉ lệch +0,005 đến +0,02 (hơi lạc quan), nhưng độ lệch
  chuẩn giữa các fold lớn gấp 2–4 lần (vd r = 0,02, cân bằng: 0,045 so với 0,017).
- **Rút ra (thay cho kết luận cũ ở giai đoạn 8):**
  1. **Thêm dữ liệu bình thường không giúp:** khi đã kiểm soát mất cân bằng lớp (cột cân bằng), PR-AUC gần như
     không đổi từ r = 0,02 đến 0,2 (0,161–0,173). Vòng không trọng số lớp tụt ở r lớn là do mất cân bằng (0,35% rửa tiền
     ở r = 0,2), không phải do dữ liệu.
  2. **r = 0,02 không trọng số lớp là cấu hình tốt nhất** (0,209): tỷ lệ ~4% rửa tiền trong mẫu tốt hơn cả cân bằng hoàn
     toàn. → Chọn r = 0,02 có căn cứ: đủ dữ liệu, nhẹ máy, tỷ lệ lớp tốt.
  3. **Test lấy mẫu KHÔNG lạc quan gấp 3 lần** như giai đoạn 8 nói — sai số chủ yếu là dao động. Vẫn nên giữ test đầy
     đủ để con số ổn định (và validation đầy đủ để chọn ngưỡng).
  4. **VDR@1% trên IBM dao động quá mạnh** (độ lệch chuẩn 0,27–0,40) → chỉ báo cáo làm chỉ số phụ kèm khoảng dao động.
  5. Mốc tham chiếu cho E7: mô hình cây dạng bảng đạt PR-AUC ~0,21, F1 ~0,28 (mức ngẫu nhiên 0,00089).

## Giai đoạn 10 — Ghi chú cách A / cách B, chuẩn bị chạy lại (02/11/2026)

- **Làm:** thêm `--model mlp` cho `30_dp_fl.py --mode client` (cách B trên synthetic: TabMLP, bỏ cạnh đồ thị); viết
  [ghi_chu_cach_A_B.md](ghi_chu_cach_A_B.md) gồm định nghĩa, những gì đã biết, lệnh chạy lại, cách đọc kết quả.
- **Đính chính:** câu "cách B chịu DP tốt hơn" (trả lời trong buổi trao đổi) chưa có căn cứ — GraphSAGE nhỏ cỡ
  3,2 nghìn tham số (bằng MLP) vẫn sụp; nguyên nhân chính là ít ngân hàng.
- **Kế tiếp:** Đạt chạy lại lần lượt theo mục 3 của ghi chú; nếu cách B cũng sụp → họp lập kế hoạch tăng số ngân hàng.

## Giai đoạn 11 — Chạy cách B và thử ε lớn: DP mức client với 5 ngân hàng KHÔNG cứu được (02–03/11/2026)

- **Làm:** theo mục 3 của [ghi_chu_cach_A_B.md](ghi_chu_cach_A_B.md); synthetic, fold 0, 5 ngân hàng.
- **Kết quả PR-AUC** (mức ngẫu nhiên ≈ 0,015–0,02; gộp chung không DP = 0,733):

  | ε | B: MLP, nhiễu server, 10 vòng | B: MLP, nhiễu từng NH, 10 vòng | B: MLP, nhiễu server, 50 vòng | A: GNN nhỏ, nhiễu server, 10 vòng |
  |---|---|---|---|---|
  | ∞ | 0,653 | 0,652 | 0,700 | 0,587 |
  | 64 | 0,604 | — | — | — |
  | 32 | 0,447 | — | — | 0,303 |
  | 16 | 0,123 | — | — | 0,159 |
  | 8 | 0,058 | 0,026 | 0,043 | 0,069 |
  | 4 | 0,023 | 0,023 | 0,044 | 0,022 |
  | 1 | 0,020 | 0,021 | 0,042 | 0,018 |

  Lưu ở `results/_luu_tru/2026-11-02_2357_dp_client_B_*`, `..._2359_..._B_server_mlp_50v`,
  `2026-11-03_0000_dp_client_B_server_mlp_eps_lon`, `2026-11-03_0001_dp_client_A_server_nho_eps_lon`.
- **Rút ra:**
  1. Cách B cũng sụp ở mọi ε ≤ 8 — đúng dự đoán; đổi mô hình không cứu được.
  2. Mô hình chỉ bắt đầu học được từ **ε ≈ 16–32**, đạt ≥ 85% so với gộp chung chỉ ở ε ≈ 64 (cách B: 0,604 = 82%).
     ε cỡ đó gần như không còn ý nghĩa bảo mật → **với 5 ngân hàng, DP mức client không khả thi ở mức ε có ý nghĩa**.
  3. Không DP: MLP (cách B) 10 vòng đạt 0,653, cao hơn GNN nhỏ 0,587; 50 vòng đạt 0,700 (gần FedAvg GNN 0,715).
  4. → Theo tiêu chí đã thống nhất ở ghi chú A/B: **bắt buộc tăng số ngân hàng → họp lập kế hoạch mới.**

## Giai đoạn 12 — Đính chính mã ngân hàng IBM (03/11/2026)

- **Bối cảnh:** bạn trong nhóm hỏi "HI-Small có 10 ngân hàng?". Kiểm tra `HI-Small_accounts.csv`: **30.470 mã ngân
  hàng (Bank ID), 20.053 tên ngân hàng** (một tên có tới 48 mã). Không phải 10 ngân hàng.
- **Phát hiện:** 58 cặp mã chỉ khác số 0 ở đầu (`"010"` / `"0010"`) đều trỏ tới CÙNG Bank ID và CÙNG tên ngân hàng
  (58/58 cặp; vd Bank ID 10 = National Bank of Laramie). → "Sửa lỗi" ở giai đoạn 5 (commit `cabad04`, đọc mã dạng chuỗi
  giữ số 0) là **sai**: nó tách một ngân hàng thành hai, và tách lịch sử của tài khoản ở các ngân hàng đó.
- **Sửa:** `load_raw` đọc dạng chuỗi rồi bỏ số 0 ở đầu → 30.470 ngân hàng, khớp `accounts.csv`. Test cập nhật.
  Tính lại đặc trưng: `results/_luu_tru/2026-11-03_0007_ibm_features_global`, `..._0008_ibm_features_bank`.
- **Ảnh hưởng — chạy lại đánh giá 5 fold** (`results/_luu_tru/2026-11-03_0010_ibm_danh_gia_lay_mau`), PR-AUC test đầy
  đủ, không trọng số lớp:

  | r_train | Trước khi sửa (giai đoạn 9) | Sau khi sửa |
  |---|---|---|
  | 0,02 | 0,209 ± 0,018 | **0,219 ± 0,016** |
  | 0,05 | 0,118 ± 0,017 | 0,153 ± 0,032 |
  | 0,10 | 0,111 ± 0,010 | 0,108 ± 0,020 |
  | 0,20 | 0,113 ± 0,028 | 0,101 ± 0,010 |

  F1 lớp rửa tiền (r = 0,02): 0,276; Recall@1%: 0,633. Kết luận giai đoạn 9 **không đổi**: r = 0,02 vẫn tốt nhất;
  chênh lệch nằm trong mức dao động giữa các fold. Mốc baseline dạng bảng cập nhật: PR-AUC ~0,22.
- **Số ngân hàng dùng được làm client (ngày 0–9):** ≥ 1.000 giao dịch: 802; ≥ 10.000: 91; có ≥ 1 ca rửa tiền: 987;
  ≥ 5 ca: 223; **≥ 20 ca: 27**; ≥ 50 ca: 3.
- **Bài học:** đối chiếu với file `accounts.csv` (nguồn chuẩn) trước khi kết luận về định danh.

## Giai đoạn 13 — FL + DP mức client trên IBM, 27 ngân hàng + 1 client gộp (03/11/2026)

- **Làm:** `scripts/09_ibm_fl_dp.py` (D19), fold 0, K = 28, MLP (cách B), đặc trưng view bank, train r = 0,02,
  val/test đầy đủ, 30 vòng. Lưu ở `results/_luu_tru/2026-11-03_0027_ibm_fl_dp`.
- **Kết quả** (mức ngẫu nhiên PR-AUC tổng ≈ 0,0009):

  | Cách | PR-AUC tổng | PR-AUC TB theo client | Recall@1% | z |
  |---|---|---|---|---|
  | Gộp chung | 0,055 | 0,231 | 0,454 | |
  | Tự học riêng | 0,041 | 0,182 | 0,439 | |
  | FedAvg | 0,033 | 0,127 | 0,343 | |
  | DP ε = 8 | 0,011 | 0,104 | 0,155 | 3,50 |
  | DP ε = 4 | 0,005 | 0,039 | 0,083 | 6,35 |
  | DP ε = 2 | 0,002 | 0,017 | (0,372)* | 11,80 |
  | DP ε = 1 | 0,002 | 0,025 | (0,381)* | 22,19 |

- **Rút ra:**
  1. Tăng K từ 5 lên 28 có tác dụng: ε = 8 không còn sụp hoàn toàn (PR-AUC gấp ~12 lần ngẫu nhiên). Nhưng ε ≤ 4 vẫn
     gần như sụp; ε ≤ 2 sụp.
  2. **MLP yếu:** gộp chung chỉ 0,055, trong khi mô hình cây (giai đoạn 9/12) đạt ~0,22 → mọi con số đang bị mô hình
     yếu kéo xuống. FedAvg còn đang tăng ở vòng 30 (val 0,014 → 0,020 → 0,026 → 0,028 ở vòng 5/10/20/30) → học thiếu.
  3. FedAvg (0,033) kém tự học riêng (0,041) — có thể do học thiếu và do client gộp `_nho` (68% ca rửa tiền) khác hẳn
     các ngân hàng lẻ.
  4. *Lỗi đo phát hiện thêm (#41):* Recall@1% = 0,37–0,38 ở ε = 2 và 1 dù PR-AUC ≈ ngẫu nhiên. Khi mô hình sụp, điểm gần
     như bằng nhau; `top_frac_daily` phá hòa theo thứ tự xuất hiện → thứ tự dòng trong file (không ngẫu nhiên) quyết
     định ai bị gắn cờ. Cần phá hòa ngẫu nhiên (seed cố định) hoặc theo trung bình hạng.
- **Kế tiếp đề xuất:** (1) làm MLP mạnh lên trước (nhiều vòng/epoch hơn, chỉnh lr) để FedAvg không DP tiến gần gộp
  chung; (2) sửa #41; (3) mới chạy lại DP và 5 fold.
