# Giao việc Em 1 — B1 (giữ bao nhiêu dữ liệu) và B2 (chia client theo nước)

Người duyệt: Đạo. Nhánh làm việc: `em1/b1-du-lieu`. Hỏi gì cứ hỏi Đạo; dùng trợ lý AI thoải mái, nhưng **mọi con số
trong báo cáo phải do script chạy ra**, không chép số AI tự nói.

## Đọc trước (30 phút)

1. `docs/HUONG_DAN_CHAY.md` mục 1–2: cài đặt, tải dữ liệu, tính đặc trưng.
2. `docs/KHUNG_CHOT_BUOC.md`: quy tắc khung. Đặc biệt quy tắc 2–5.
3. Bối cảnh: IBM HI-Small, chỉ dùng ngày 0–9, ~5,08 triệu giao dịch, 4.522 ca rửa tiền (0,09%).

Máy Windows: dùng PowerShell; hoặc chạy hết trên Kaggle bằng `notebooks/kaggle_mau.ipynb` (không cần cài gì).

## Việc 1 — Thống kê phân bổ dữ liệu (không cần mô hình)

Viết `scripts/b1_thong_ke.py`, đọc `data/processed/HI-Small_features_bank.parquet` (lọc `TX_TIME_DAYS <= 9`) và
`data/splits/ibm_ngan_hang_quoc_gia.csv`. Ra `results/b1_thong_ke/`:

| Bảng | Cột |
|---|---|
| `theo_ngay.csv` | ngày, số giao dịch, số ca, tỷ lệ % |
| `theo_nuoc.csv` | nước, số giao dịch, số ca, tỷ lệ %, số ngân hàng |
| `theo_ngan_hang_nhom.csv` | nhóm cỡ ngân hàng (< 1k, 1k–10k, 10k–100k, ≥ 100k giao dịch), số ngân hàng, số giao dịch, số ca |
| `theo_loai_thanh_toan.csv` | 7 cột `PF_*` (ACH, Bitcoin, …), số giao dịch, số ca, tỷ lệ % |

Kèm 2–3 biểu đồ (tỷ lệ gian lận theo nước; phân bố cỡ ngân hàng thang log).
Câu cần trả lời trong báo cáo ngắn: gian lận phân bố đều hay dồn vào vài nước / loại thanh toán?

## Việc 2 — Đường học: giữ bao nhiêu bản ghi là đủ (B1)

Mục đích: **giảm dữ liệu để máy chạy nhẹ hơn** (~1 triệu bản ghi), xem mất bao nhiêu điểm.

- Giữ **hết** ca rửa tiền trong phần fit, bốc ngẫu nhiên ca bình thường theo từng ngân hàng, **có bù trọng số**.
  Val và test **không** lấy mẫu.
- Cỡ phần fit: 100 nghìn, 250 nghìn, 500 nghìn, **1 triệu**, đủ (~3,05 triệu).
- Mô hình: MLP hiện tại (`scripts/12_ibm_cach1_lo_nho.py` có sẵn `--so-mau N --bu-trong-so`). Lệnh mẫu:

  ```bash
  python scripts/12_ibm_cach1_lo_nho.py --folds 0 1 2 3 4 --min-tx 0 --batch-size 512 --lr 0.001 \
      --max-epochs 300 --patience 30 --so-mau 1000000 --bu-trong-so --tag b1_n1000000_s42
  ```

- ⚠️ Script 12 **đang tự chia fold** (cách cũ). Việc đầu tiên: sửa script 12 thêm tùy chọn `--chia b0` để lấy fit/val/test
  từ `fraudfl.khung.gan_fold(tx_id, seed)` + `vai_tro(fold, k)`. Viết test cho phần sửa. Đạo duyệt phần này trước khi
  chạy số chính thức.
- Chạy chính thức: 3 seed (42, 7, 2027) × 5 fold cho mỗi cỡ.

Tiêu chí chốt (viết sẵn): chọn **N nhỏ nhất đạt ≥ 95%** PR-AUC của "đủ". Nếu 1 triệu chưa đạt 95% thì vẫn có thể chọn
1 triệu vì lý do máy, nhưng phải ghi rõ mất bao nhiêu % vào `decisions.md` (D26).

Đầu ra: `results/b1_duong_hoc/bang.csv` (cỡ, seed, fold, PR-AUC, Recall@1%, thời gian, RAM) + 1 biểu đồ đường học.

## Việc 3 — Bảng client theo nước (B2)

Từ `theo_nuoc.csv`: đề xuất danh sách client theo nước (gộp các nước < 10.000 giao dịch thành một client "Khác"
hoặc bỏ, tự đề xuất và giải thích). Ra `results/b2_client/client_theo_nuoc.csv` (nước → mã client).
Đạo dùng bảng này cho FL theo nước.

## Bàn giao

- PR từ `em1/b1-du-lieu` vào `main`, CI xanh.
- Báo cáo ngắn `results/b1_thong_ke/BAO_CAO.md`: 5–10 dòng kết luận + bảng chính, kèm đề xuất D26 và D27.
