# Khung đi từng bước, chốt từng bước (từ 11/11/2026)

Mục đích: đi đến đâu chốt đến đó, bước sau chỉ dùng kết quả đã chốt của bước trước, về sau không phải sửa lại.
Kế hoạch đầy đủ và lý do: `../docs/Ke-hoach-7-Khung-tung-buoc-va-phan-cong-2026-11-11.md` (ngoài repo, thư mục `NCKH/docs`).

## Trạng thái

| Bước | Câu hỏi | Người | Trạng thái | Quyết định | Đầu ra đã chốt |
|---|---|---|---|---|---|
| **B0** Nền đo | Chia dữ liệu và chấm điểm thế nào? | Đạo | ✅ chốt 11/11 | D25 | `data/splits/ibm_hi_small_chia.parquet`, `ibm_ngan_hang_quoc_gia.csv`, `b0_tom_tat.json` |
| **B1** Dữ liệu | Giữ bao nhiêu bản ghi (~1 triệu, để máy nhẹ)? | Em 1 | đang làm | D26 | danh sách TX_ID phần fit được giữ |
| **B2** Client | Chia theo ngân hàng hay theo nước? | Em 1 + Đạo | đang làm | D27 | bảng client |
| **B3** Gộp chung | Mô hình nào làm cận trên? | Em 2 | đang làm | D28 | mô hình + siêu tham số |
| **B4** Tự học riêng | Từng client tự học được bao nhiêu? | Đạo (thay Đạt) | chờ B3 | D29 | điểm cận dưới |
| **B5** FL không DP | FL lấy lại được bao nhiêu? | Đạo (thay Đạt) | chờ B4 | D30 | cấu hình FL |
| **B6** FL + DP | Bảo mật tốn bao nhiêu? | Đạo (thay Đạt) | chờ B5 | D31 | bảng kết quả chính |
| **B7** Đóng gói | Người khác chạy lại được không? | Đạo | notebook mẫu | — | notebook Kaggle |

Mục tiêu tham khảo (gợi ý của anh Hùng, 11/11): **tự học riêng ≈ 50%** cận trên B3, **FL + DP ≈ 60%** trở lên.
Nghĩa là: cận dưới phải là một mốc tự học *mạnh, công bằng* (không phải mốc yếu cho FL dễ thắng), và FL + DP phải hơn
nó rõ ràng.

## Quy tắc

1. Mỗi bước trả lời **một câu hỏi**. Tiêu chí chốt viết vào bảng trên **trước khi** chạy.
2. Đầu mỗi bước chạy `python scripts/b0_chia_co_dinh.py --kiem-tra` → phải ra `KHỚP` cho cả 2 file.
3. Mọi số **chính thức** dùng chia của B0: `fraudfl.khung.gan_fold(tx_id, seed)` rồi `vai_tro(fold, k)`
   → fit / val / test. Không tự gọi `StratifiedKFold` trong script mới.
4. Chọn siêu tham số, dừng sớm, chọn vòng: **chỉ trên val**. Test chỉ chấm một lần, sau khi đã chọn xong.
5. Số chính thức: 3 seed × 5 fold. Số thử: đặt `--tag thu_...`, không đưa vào báo cáo.
6. **Chốt một bước** = cả 3 việc:
   - thêm dòng `Dxx` vào `docs/decisions.md` (quyết định + lý do + số liệu dẫn chứng);
   - kết quả nằm trong `results/b<k>_*` (và bản lưu trữ tự động trong `results/_luu_tru/`);
   - tag git: `git tag b<k>-chot && git push origin b<k>-chot` (sau khi PR của bước đó đã merge vào `main`).
7. Mở lại bước đã chốt: phải ghi lý do vào `decisions.md`, và **chạy lại mọi bước sau nó**. Tránh nếu có thể.
8. Làm trên nhánh riêng (`dao/...`, `em1/...`, `em2/...`), PR vào `main`, CI phải xanh mới merge.

## B0 chốt gì (D25)

- Dữ liệu: IBM HI-Small, **ngày 0–9** (D18), đủ **30.470 ngân hàng**: 5.077.237 giao dịch, 4.522 ca (0,089%).
- Fold: 3 seed (42, 7, 2027) × 5 fold, phân tầng theo (ngân hàng gửi × nhãn) với ngân hàng có ≥ 10 ca, còn lại theo nhãn.
- Fold k: **test = fold k, val = fold (k+1) mod 5, fit = 3 fold còn lại** → ~3,05 triệu / 1,02 triệu / 1,02 triệu dòng,
  ~2.713 / 904 / 905 ca.
- Kịch bản 802 ngân hàng = lọc dòng của các ngân hàng ≥ 1.000 giao dịch (vai trò từng dòng không đổi).
- Ngân hàng → nước: tên "X Bank #n" → nước X; ngân hàng có tên riêng (thành phố Mỹ) → USA (chưa xác nhận);
  "Crytpo" trong dữ liệu → "Crypto". 34 nước, mọi giao dịch đều tra được nước.
- Chỉ số: PR-AUC (chính), macro PR-AUC theo client (so với tự học riêng), Recall@1% theo ngày;
  tính bằng `fraudfl.eval.metrics` (D1, D2, D7).
- Lấy mẫu con (B1) chỉ áp vào phần **fit**. Val và test luôn giữ tỷ lệ thật.

Khác với các lần chạy trước 11/11: trước đây test = 20%, val = 16%, fit = 64% (val tách ngẫu nhiên từ phần train),
và tập dòng phụ thuộc `--min-tx`. Số cũ vẫn giữ trong `results/` để tham khảo, nhưng **không trộn** với số chạy theo khung mới.
