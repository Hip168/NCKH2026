# Sổ quyết định

Ghi mọi quyết định ảnh hưởng tới kết quả **trước khi** chạy thí nghiệm chính. Mỗi mục: ngày, nội dung, lý do.
Trạng thái "đề xuất" = đã áp vào code để chạy thử, **chờ cả nhóm duyệt** trong buổi họp.

## Đã chốt

| # | Ngày | Quyết định | Lý do |
|---|---|---|---|
| D1 | 2026-10-31 | Chỉ số chính: PR-AUC, Recall@p, VDR@p. Phụ: F1, ROC-AUC, Precision@p, chi phí truyền thông, độ trễ | Theo Slide 6. Chốt trước để tránh chọn chỉ số theo kết quả |
| D2 | 2026-10-31 | Recall/Precision/VDR tính **theo ngày**: mỗi ngày gắn cờ phần đầu bảng xếp hạng, cộng dồn trên mọi ngày | Tính trên toàn tập test thì Recall@100 tối đa ~5%, vô nghĩa |
| D3 | 2026-10-31 | Ngưỡng F1 chọn trên inner fold 0 của tập train | Không chọn ngưỡng trên test |
| D4 | 2026-10-31 | Đặc trưng tính riêng trong từng ngân hàng; rủi ro terminal chỉ dùng nhãn có độ trễ 7 ngày | Đúng tinh thần FL; tránh rò rỉ nhãn |
| D5 | 2026-10-31 | Synthetic: 5 ngân hàng × ~100k giao dịch, 120 ngày, tỷ lệ gian lận 1,0–1,8%, terminal dùng chung | Theo Slide 5. 81% terminal bị xâm nhập xuất hiện ở >1 ngân hàng → động lực cho FL |
| D6 | 2026-10-31 | Chia fold: nested stratified 5×3 theo (nhãn × client), seed 42; thêm cột chia theo thời gian | Theo Slide 8 |
| D24 | 2026-11-05 | **100 vòng** (thay 30) cho cấu hình chính FL + DP trên IBM; đang chạy lại 5 fold (`CHINH_5fold_100vong`). Không lên 500 vòng | Giai đoạn 26: DP ε = 1 fold 0 tăng 0,0233 → 0,0304 (+30%) dù nhiễu tăng 1,8×. Tính toán: 500 vòng → nhiễu 0,040/tham số, vùng kết quả tụt mạnh |
| D23 | 2026-11-05 | **Giữ trung bình đều** cho DP; không dùng trọng số bị chặn | Giai đoạn 25: trọng số bị chặn không giúp khi có FedProx (0,0275–0,0281 so với 0,0282) và làm DP kém hơn (nhiễu tăng tới 6×) |
| D22 | 2026-11-03 | **Cắt C = 0,35** (trung vị độ lớn cập nhật khi có FedProx) | Giai đoạn 21–22: FedProx làm cập nhật nhỏ đi ~3,5×; C = 0,35 tốt nhất trong {0,35; 0,5; 0,67; 1,6}. Hạn chế: chọn ngoài ngân sách ε |
| D21 | 2026-11-03 | **FedProx μ = 0,1** cho FL và FL + DP trên IBM; MLP học theo lô nhỏ (lô 512, lr 5e-3, 5 epoch cục bộ, pos_weight 1) | Giai đoạn 14, 17, 20–21: μ = 0,1 tốt nhất trong {0,01; 0,1; 0,3; 1}; full-batch làm MLP học thiếu |
| D20 | 2026-11-03 | **DP mức client trên IBM với 802 ngân hàng** (Đạt quyết định): client = mọi ngân hàng có ≥ 1.000 giao dịch trong ngày 0–9 (gồm cả ngân hàng không có ca rửa tiền); ngân hàng nhỏ hơn BỎ khỏi FL (không gộp) để mỗi client là một ngân hàng thật. Mỗi vòng chọn ngẫu nhiên 200 ngân hàng (q = 0,25), ε tính với sample_rate = q; server cộng nhiễu std = z·C/m. Train lấy mẫu r = 0,1; val/test đầy đủ. Sau đó quét K ∈ {28, 100, 200, 400, 802} | K = 28 sụp ở mọi ε ≤ 8 (giai đoạn 15). 802 ngân hàng bao phủ 90% giao dịch, 89% ca rửa tiền; nhiễu mỗi tham số ở ε = 1 giảm từ 0,79 (K = 28) xuống 0,029 |
| D19 | 2026-11-03 | **IBM HI-Small là dữ liệu chính** (Đạt quyết định). FL + DP mức client chạy trên **27 ngân hàng có ≥ 20 ca rửa tiền (ngày 0–9) + 1 client gộp phần còn lại** (K = 28), mô hình MLP dạng bảng (cách B), script `09_ibm_fl_dp.py`. Kết quả synthetic (E1–E5, thử nhanh DP) giữ làm phần thử nghiệm ban đầu | Synthetic chỉ có 5 ngân hàng → DP mức client sụp ở mọi ε ≤ 8 (giai đoạn 11); IBM có đủ ngân hàng để tăng K. Lưu ý: client gộp `_nho` chứa ~68% ca rửa tiền |
| D18 | 2026-11-02 | **IBM HI-Small chỉ dùng ngày 0–9** (`config.IBM_MAX_DAY = 9`, áp trong `05_ibm_sample.py` trước khi lấy mẫu). Đặc trưng `SND_AGE_DAYS` vẫn tính nhưng không đưa vào mô hình (`ibm_features.EXCLUDED_FEATURES`). Chi tiết ở mục D18 bên dưới | Ngày 10–17 chỉ ~1.100 gd nhưng ~60% rửa tiền (lỗi bộ sinh dữ liệu); bỏ chỉ mất 0,02% giao dịch, còn 4.522/5.177 ca rửa tiền; giữ lại thì đặc trưng thời gian rò rỉ và VDR bị thổi phồng 0,10 → 0,55 |
| D17 | 2026-11-02 | **Lấy mẫu con IBM AML** (anh Hùng duyệt): giữ 100% ca rửa tiền, bốc tỷ lệ r ca bình thường **trong từng ngân hàng** có ≥ 1.000 ca bình thường, các ngân hàng nhỏ hơn gộp thành một nhóm `_nho` rồi bốc chung (tránh phình mẫu), seed 42; lưu cố định `data/splits/ibm_sample_ids.csv` + `ibm_sample_meta.json` (script `05_ibm_sample.py`, từ chối ghi đè); sau đó nested stratified 5×3 trên mẫu → `fold_assignments_ibm.csv`. Đặc trưng và đồ thị tính trên dữ liệu đầy đủ trước khi lọc. Trong báo cáo chỉ nhắc ngắn | 5 triệu giao dịch không chạy được trên máy nhóm; gian lận chỉ ~0,1% nên bỏ bớt ca bình thường ít mất thông tin. **Chờ chốt:** giá trị r (theo đường học) và việc tính chỉ số trên test có trọng số 1/r (`evaluate(..., weight=)`) để PR-AUC đúng tỷ lệ thật |
| D12 | 2026-11-02 | **Giữ DP mức client** (đúng Slide 4) làm cơ chế chính, không chuyển sang mức bản ghi. Đọc [luu_y_dp_muc_client.md](luu_y_dp_muc_client.md) | Giữ bài toán tổng quát: che được cả một ngân hàng, áp dụng được cho GNN. Với 5 ngân hàng, ε ≤ 2 hiện làm mô hình sụp (PR-AUC 0,028); cần chốt lại tiêu chí Slide 10 và thêm thí nghiệm quét số ngân hàng |


> Ghi chú: D14–D16 không tồn tại — là các đề xuất trong kế hoạch 02/11 (thí nghiệm ngân hàng nhỏ, đơn vị riêng tư, quét K) chưa từng được chốt. Quét K được thay bằng D19–D20.

## Đề xuất — đã áp vào code, chờ nhóm duyệt

| # | Ngày | Đề xuất | Lý do | Thay cho |
|---|---|---|---|---|
| D7 | 2026-10-31 | Ngân sách cảnh báo **p = 1%** giao dịch mỗi ngày (chỉ số chính), **2%** (độ nhạy). `config.ALERT_FRACS` | Giả định năng lực kiểm tra của chuyên viên tỷ lệ với lưu lượng; không phụ thuộc kích thước fold/bộ dữ liệu | Q1: k = 100/ngày gắn cờ ~12% giao dịch → VDR ≈ 0,94 với mọi mô hình |
| D8 | 2026-10-31 | Thêm `TX_AMOUNT_RATIO_30DAY` (giá trị / chi tiêu TB 30 ngày của khách) + chuẩn hóa z-score **riêng từng client** (`prep.scale_per_client`) | Giá trị tiền khác quy mô giữa ngân hàng; chuẩn hóa cục bộ không cần chia sẻ thống kê → tương thích FL | Q2 |
| D9 | 2026-10-31 | Bỏ 30 ngày đầu khỏi fit/val/test (`config.BURN_IN_DAYS`); vẫn dùng làm lịch sử cho đặc trưng và nút đồ thị | Kịch bản gian lận và cửa sổ 30 ngày chưa ổn định | Q3 |
| D10 | 2026-10-31 | GNN: huấn luyện **full-batch**, lr = 5e-3 (Slide 6: batch 64, lr = 1e-3); dừng sớm theo PR-AUC val, patience 30 epoch, tối đa 300; FedAvg **50 vòng** × 3 epoch cục bộ | Đồ thị liên kết nên mini-batch cần thư viện lấy mẫu láng giềng (pyg-lib) chưa có trên Windows; lr 1e-3 full-batch hội tụ quá chậm. FedAvg 30 vòng chưa hội tụ ở fold 0 (val PR-AUC còn tăng ở vòng 28–30) | Slide 6 |
| ~~D12~~ | 2026-11-01 | **Đã thay bằng D12 ngày 2026-11-02 ở mục "Đã chốt".** Đề xuất cũ: DP có **2 mức** (`src/fraudfl/privacy.py`, `scripts/30_dp_fl.py`): (a) **client-level** đúng Slide 4 — cắt Δ của cả ngân hàng, cộng nhiễu, GNN; (b) **record-level** — DP-SGD trong từng ngân hàng (cắt gradient từng giao dịch, lấy mẫu Poisson, batch kỳ vọng 1024), MLP dạng bảng. ε tính bằng RDP accountant (Opacus), δ = 1e-5, C = 1 | Với 5 client, client-level ε = 1 cần hệ số nhiễu z ≈ 29 (nhiễu lớn hơn tín hiệu hàng nghìn lần). Record-level ε = 1 chỉ cần z ≈ 3 nhờ khuếch đại lấy mẫu con. Bài toán ngân hàng cần bảo vệ **từng khách/giao dịch** → record-level là đơn vị riêng tư đúng | Slide 4 |
| D13 | 2026-11-01 | Record-level dùng SGD momentum 0,9, lr 0,5 (không phải Adam); pos_weight = 10 cố định | Adam khuếch đại nhiễu DP (thử trên val: 0,14 so với 0,67). pos_weight tính từ dữ liệu sẽ tốn ngân sách riêng tư | Slide 6 |
| D11 | 2026-10-31 | Đồ thị: nút = giao dịch; cạnh từ 3 giao dịch liền trước của cùng khách và cùng terminal, **chỉ chiều quá khứ → hiện tại**. Local/FL: cạnh trong cùng ngân hàng. Centralized: cạnh terminal xuyên ngân hàng | Không dùng thông tin tương lai; centralized có lợi thế thông tin thật sự so với FL | — |

### Kết quả sau khi áp D7–D9 (baseline dạng bảng, trung bình 5 fold, `results/baseline_tabular/`)

| Đặc trưng | Mô hình | Cách train | PR-AUC | Recall@1% | VDR@1% |
|---|---|---|---|---|---|
| v0 (gốc) | RF | centralized | 0,632 | 0,488 | 0,741 |
| v0 (gốc) | RF | local | 0,670 | 0,521 | 0,784 |
| v1 (D8) | RF | centralized | 0,694 | 0,529 | 0,769 |
| v1 (D8) | RF | local | 0,690 | 0,538 | 0,790 |
| v0 (gốc) | LR | centralized | 0,459 | 0,366 | 0,660 |
| v1 (D8) | LR | centralized | 0,549 | 0,436 | 0,679 |

D8 xóa gần hết khoảng cách centralized–local (RF: 0,694 so với 0,690). Recall@1% giờ phân biệt rõ mô hình
(0,37 → 0,54), không còn bão hòa như k = 100.

### D18 (ĐÃ CHỐT 2026-11-02, Đạt quyết định — báo lại anh Hùng): IBM HI-Small chỉ dùng ngày 0–9

Từ ngày 10 đến 17 chỉ còn khoảng 1.100 giao dịch nhưng khoảng 60% là rửa tiền (trước đó 0,03–0,21%) — lỗi của bộ sinh
dữ liệu. Giữ lại thì mọi đặc trưng gắn với thời gian bị rò rỉ và VDR bị thổi phồng (0,10 → 0,55). Đề xuất: lọc
`TX_TIME_DAYS <= 9` TRƯỚC khi lấy mẫu (D17); đặc trưng vẫn tính trên toàn bộ dữ liệu (chỉ nhìn quá khứ nên không bị
ảnh hưởng bởi đuôi). Cân nhắc bỏ `SND_AGE_DAYS` (chênh PR-AUC 0,01). Bằng chứng: `docs/nhat_ky_ket_qua.md` giai đoạn 6.

## Câu hỏi còn mở

### Q7. Giới hạn của DP record-level cần nêu trong bài báo

- GNN không dùng được DP-SGD thông thường (gradient một nút phụ thuộc láng giềng) → kết quả DP dùng MLP
  dạng bảng. DP cho GNN (node-level DP, ví dụ GAP — Sajadmanesh et al., USENIX Security 2023) là hướng mở rộng.
- Đặc trưng cửa sổ (chi tiêu TB của khách, rủi ro terminal) được tính từ nhiều giao dịch → một giao dịch ảnh
  hưởng tới đặc trưng của giao dịch khác. Đảm bảo ε chỉ đúng khi coi mỗi hàng đặc trưng là một bản ghi;
  bảo đảm chặt hơn cần tính đặc trưng có DP — nêu là hạn chế.
- Không có SecAgg: server thấy cập nhật đã thêm nhiễu của từng client (đúng phạm vi phiên bản 1.0.0).

### Q4. Hiệu năng kỳ vọng ở Slide 7 cần sửa theo số thật

Đã có kết quả E1–E5 (`docs/results.md`). Slide 7 cần thay bảng "kết quả kỳ vọng" bằng số thật, và Slide 4 cần
đổi cơ chế DP sang mức bản ghi (client-level sụp ở mọi ε ≤ 2). Local-only mạnh hơn kỳ vọng (96% centralized)
→ cân nhắc thêm kịch bản ngân hàng nhỏ để thể hiện lợi ích FL.

### Q5. IBM AML: dùng LI-Small hay HI-Small?

Máy dev có sẵn **LI-Small** (bản đã định dạng, `Test/TH1/data-raw`): 6,92 triệu giao dịch, 706k tài khoản,
17 ngày, tỷ lệ rửa tiền **0,05%** (3.565 ca). Slide 5 ghi "~5 triệu giao dịch, ~300k tài khoản" — gần với
**HI-Small** (5,08 triệu giao dịch, tỷ lệ ~0,1%). Cần chốt dùng bộ nào và sửa slide cho khớp.

Bản đã định dạng **mất mã ngân hàng** (bị gộp vào mã tài khoản) → chỉ dùng được cho E7 (centralized).
Muốn làm E8 (FL theo ngân hàng) phải tải bản gốc `*_Trans.csv` từ Kaggle (có cột From Bank / To Bank) và
đọc bằng `ibm_aml.load_raw`.

### Q6. VDR trên IBM AML cần quy đổi tiền tệ

Giao dịch IBM AML ở 15 loại tiền tệ. Tính VDR phải quy đổi về một đồng tiền (ví dụ USD theo tỷ giá cố định)
— cần chốt bảng tỷ giá. Elliptic không có giá trị tiền → không tính VDR trên Elliptic, chỉ báo PR-AUC, Recall@p.
