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

- **Dữ liệu:** 5.078.345 giao dịch, 5.177 ca rửa tiền (0,102%), 18 ngày, 30.528 ngân hàng gửi. Top 5 ngân hàng giữ
  13,6% giao dịch; chỉ 34 ngân hàng có ≥ 20 ca rửa tiền. Ca rửa tiền dồn vào 10 ngày đầu.
- **Sửa lỗi:** `load_raw` đọc mã ngân hàng thành số làm gộp nhầm 58 ngân hàng (`"010"` = `"10"`). Đã sửa, commit
  `cabad04`.
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
  1. Đặc trưng có tín hiệu thật. ~~PR-AUC khoảng 0,29~~ — **ĐÍNH CHÍNH ở giai đoạn 8:** con số này quá lạc quan do
     tập test r = 0,02 quá nhỏ; ước lượng đúng hơn khoảng 0,10 (vẫn gấp khoảng 100 lần mức ngẫu nhiên 0,001).
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
- **Rút ra:**
  1. **Để HUẤN LUYỆN, r = 0,02 là đủ**: mô hình tốt ngang r = 0,1.
  2. **Để ĐÁNH GIÁ, tập test lấy mẫu r = 0,02 KHÔNG tin được**: PR-AUC phụ thuộc vào vài ca bình thường "khó", với
     r nhỏ phần lớn các ca này không lọt vào test → con số lạc quan khoảng 3 lần (0,29 so với ~0,10). Trọng số 1/r
     sửa được tỷ lệ trung bình nhưng không sửa được độ dao động này.
  3. → Đề xuất sửa D17: **lấy mẫu chỉ cho train/validation; tập test giữ TOÀN BỘ ca bình thường của fold đó**
     (hoặc r_test lớn, ≥ 0,2). Mô hình dạng bảng chấm 1 triệu dòng mất vài giây.
  4. VDR@1% dao động rất mạnh giữa các lần (0,03–0,98) vì VDR bị chi phối bởi một vài ca rửa tiền giá trị rất lớn →
     cần báo cáo VDR kèm khoảng dao động, và cũng cần tập test đầy đủ.
