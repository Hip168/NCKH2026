# Sổ quyết định

Ghi mọi quyết định ảnh hưởng tới kết quả **trước khi** chạy thí nghiệm chính. Mỗi mục: ngày, nội dung, lý do, người quyết.

## Đã chốt

| # | Ngày | Quyết định | Lý do |
|---|---|---|---|
| D1 | 2026-10-31 | Chỉ số chính: PR-AUC, Recall@k, VDR@k. Phụ: F1, ROC-AUC, Precision@k, chi phí truyền thông, độ trễ | Theo Slide 6. Chốt trước để tránh chọn chỉ số theo kết quả |
| D2 | 2026-10-31 | Recall@k, Precision@k, VDR@k tính **theo ngày**: mỗi ngày gắn cờ top-k, cộng dồn trên mọi ngày | Tính trên toàn tập test thì Recall@100 tối đa ~5%, vô nghĩa |
| D3 | 2026-10-31 | Ngưỡng F1 chọn trên inner fold 0 của tập train | Không chọn ngưỡng trên test |
| D4 | 2026-10-31 | Đặc trưng tính riêng trong từng ngân hàng; rủi ro terminal chỉ dùng nhãn có độ trễ 7 ngày | Đúng tinh thần FL; tránh rò rỉ nhãn |
| D5 | 2026-10-31 | Synthetic: 5 ngân hàng × ~100k giao dịch, 120 ngày, tỷ lệ gian lận 1,0–1,8%, terminal dùng chung | Theo Slide 5. Terminal chung → 81% terminal bị xâm nhập xuất hiện ở >1 ngân hàng, tạo động lực cho FL |
| D6 | 2026-10-31 | Chia fold: nested stratified 5×3 theo (nhãn × client), seed 42; thêm cột chia theo thời gian (60% train / 7 ngày đệm / còn lại test) | Theo Slide 8; chia theo thời gian dùng cho dữ liệu đồ thị |

## Cần nhóm quyết định (phát hiện từ baseline 2026-10-31)

### Q1. Giá trị k của Recall@k / VDR@k — **k = 100/ngày quá rộng**

Mỗi outer fold có ~99k giao dịch / 120 ngày ≈ 830 giao dịch/ngày, trong đó ~13 ca gian lận.
k = 100 nghĩa là gắn cờ ~12% giao dịch mỗi ngày → VDR@100 ≈ 0,94 với **mọi** mô hình, kể cả Logistic Regression.
Chỉ số không còn phân biệt được mô hình tốt và kém, và tiêu chí "VDR ≥ 0,80" trở nên quá dễ đạt.

Đề xuất: định nghĩa k theo **tỷ lệ** lưu lượng mỗi ngày (ví dụ 1% hoặc 2%) để không phụ thuộc kích thước fold hay
bộ dữ liệu, hoặc chọn k tuyệt đối nhỏ (ví dụ 10–20/ngày/fold). Phải chốt **trước** khi chạy E1–E4,
và lý do chọn phải dựa trên năng lực kiểm tra giả định của chuyên viên, không dựa trên kết quả.

### Q2. Local-only đang **tốt hơn** centralized trên baseline dạng bảng

| Mô hình | Cách train | PR-AUC (5 fold) |
|---|---|---|
| Random Forest | centralized | 0,610 ± 0,011 |
| Random Forest | local-only | 0,661 ± 0,014 |
| Logistic Regression | centralized | 0,445 ± 0,019 |
| Logistic Regression | local-only | 0,562 ± 0,022 |

Nguyên nhân có thể: giá trị giao dịch và ngưỡng gian lận khác nhau mạnh giữa các ngân hàng (non-IID). Một mô hình
chung chỉ dùng giá trị tiền tuyệt đối không học được ngưỡng riêng của từng ngân hàng.
Đây là phát hiện thật, ngược với giả thuyết "centralized > local" ở Slide 7, và FL nhiều khả năng gặp cùng vấn đề.

Hướng xử lý (chọn trước khi làm E1–E3):
- Thêm đặc trưng tương đối, ví dụ `TX_AMOUNT / CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW`, để giá trị không phụ thuộc quy mô ngân hàng.
- Chuẩn hóa đặc trưng riêng trong từng client (local scaler) trước khi train FL.
- Cân nhắc FL cá nhân hóa (FedPer / fine-tune cục bộ) như một thí nghiệm bổ sung.
- Báo cáo trung thực trong bài báo nếu hiện tượng vẫn còn: đây là lập luận cho việc cần xử lý non-IID.

### Q3. 2–3 tuần đầu của dữ liệu là giai đoạn "khởi động"

Tỷ lệ gian lận tăng dần trong ~20 ngày đầu (biểu đồ `results/eda/non_iid_overview.png`), vì các kịch bản gian lận
cộng dồn theo thời gian, và đặc trưng cửa sổ 30 ngày chưa đủ dữ liệu. Có thể bỏ 30 ngày đầu khỏi train/test.
