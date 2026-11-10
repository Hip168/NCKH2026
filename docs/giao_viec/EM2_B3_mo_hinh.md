# Giao việc Em 2 — B3: mô hình gộp chung (cận trên)

Người duyệt: Đạo. Nhánh làm việc: `em2/b3-mo-hinh`. Dùng trợ lý AI thoải mái, nhưng **mọi con số trong báo cáo
phải do script chạy ra**. Chạy trên Kaggle (`notebooks/kaggle_mau.ipynb`) nếu máy yếu.

## Đọc trước (30 phút)

1. `docs/HUONG_DAN_CHAY.md` mục 1–2, `docs/KHUNG_CHOT_BUOC.md`.
2. Bài gốc của bộ dữ liệu: Altman et al., NeurIPS 2023, https://arxiv.org/abs/2306.16424 (đọc phần thí nghiệm:
   GNN và LightGBM + đặc trưng đồ thị).
3. Mốc hiện tại để so: MLP 22 đặc trưng, học đủ dữ liệu, **PR-AUC ≈ 0,130** (802 ngân hàng, cách chia cũ).

## Câu hỏi

Khi gộp chung dữ liệu, mô hình nào tốt nhất **mà vẫn chạy được kiểu học liên kết** (mỗi client chỉ thấy dữ liệu và
đồ thị con của mình)? Điểm của mô hình đó là **cận trên** cho mọi bước sau.

## Luật bắt buộc

- Chia dữ liệu: `fraudfl.khung.gan_fold(tx_id, seed)` + `vai_tro(fold, k)` (B0). Không tự chia.
- Chọn siêu tham số và dừng sớm **chỉ trên val**. Test chấm 1 lần.
- Chấm bằng `fraudfl.eval.metrics.evaluate` (PR-AUC, Recall@1%, VDR@1%).
- Đặc trưng đồ thị **chỉ được dùng thông tin quá khứ** (giao dịch trước thời điểm đang xét) và **trong phạm vi một client**
  (một ngân hàng hoặc một nước). Ghi rõ đặc trưng nào cần thấy ngân hàng khác (không dùng được trong FL).

## Việc (bậc thang, rẻ → đắt; dừng khi bậc sau không giúp nữa)

| Bậc | Việc | Thời gian ước |
|---|---|---|
| 1 | **LightGBM** trên 22 đặc trưng hiện có (`fraudfl.data.ibm_features.feature_columns`). `scale_pos_weight` chọn trên val. | 1 ngày |
| 2 | Thêm **đặc trưng đồ thị** tính trong client: số đối tác gửi/nhận khác nhau (fan-out/fan-in) trong 1 ngày và 7 ngày; tài khoản nhận có vừa nhận vừa gửi tiếp trong 1 giờ / 1 ngày không (chuyển tiếp); chu trình ngắn A → B → A trong 7 ngày; số lần tài khoản nhận xuất hiện. So LightGBM và MLP có / không có đặc trưng mới. | 2–3 ngày |
| 3 | (Chỉ khi bậc 2 tăng rõ) GNN nhỏ (GraphSAGE / GIN có đặc trưng cạnh) trên đồ thị con từng nước, GPU Kaggle. Bàn với Đạo trước khi làm. | 1 tuần |

Bắt đầu với 1 seed (42), 5 fold, `--tag thu_...`. Khi đã chọn được mô hình: chạy 3 seed × 5 fold để chốt.

## Đầu ra

- `scripts/b3_lightgbm.py`, `src/fraudfl/data/dac_trung_do_thi.py` (+ test trong `tests/`).
- `results/b3_mo_hinh/bang.csv`: mô hình, đặc trưng, seed, fold, PR-AUC, Recall@1%, VDR@1%, thời gian.
- `results/b3_mo_hinh/BAO_CAO.md`: bảng so sánh + đề xuất D28 (mô hình cận trên + mô hình dùng cho FL).

Lưu ý: LightGBM khó làm FL + DP hơn mạng nơ-ron. Nếu LightGBM thắng thì nó là **cận trên**; mô hình dùng cho FL vẫn là
mạng nơ-ron (MLP + đặc trưng đồ thị). Vì vậy luôn báo cả MLP + đặc trưng mới.

## Bàn giao

PR từ `em2/b3-mo-hinh` vào `main`, CI xanh, Đạo duyệt.
