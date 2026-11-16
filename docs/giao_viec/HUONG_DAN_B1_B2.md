# Hướng dẫn chạy B1 và B2: thống kê dữ liệu, đường học tới 1 triệu bản ghi, bảng client theo nước

Dành cho Em 1 (phiếu việc: `EM1_B1_B2_du_lieu.md`). **Script đã có sẵn**, em chỉ cần chạy, kiểm tra số, viết báo cáo.
Mọi con số trong báo cáo phải lấy từ file kết quả script ghi ra, không chép số do AI tự nói.

| Việc | Script | Thời gian | Ra |
|---|---|---|---|
| 1. Thống kê phân bổ dữ liệu | `scripts/b1_thong_ke.py` | ~10 giây | `results/b1_thong_ke/` |
| 2. Đường học 100k → 1 triệu → đủ | `scripts/b1_duong_hoc.py` (gọi script 12) | ~8 giờ CPU cho cả 3 seed | `results/b1_duong_hoc/` |
| 3. Bảng client theo nước | `scripts/b2_client_theo_nuoc.py` | ~5 giây | `results/b2_client/` |

## 0. Chuẩn bị (làm một lần)

Chọn **một** trong hai cách:

- **Trên máy:** cài theo `docs/HUONG_DAN_CHAY.md` mục 1–2 (venv, tải dữ liệu IBM HI-Small, chạy
  `scripts/06_ibm_features.py`). Máy Windows dùng PowerShell, lệnh giống hệt, chỉ đổi `/` thành `\` ở đường dẫn
  nếu cần.
- **Trên Kaggle (khuyên dùng cho việc 2 vì nặng):** làm theo `docs/HUONG_DAN_KAGGLE.md` đến hết ô 5.

Sau đó tạo nhánh làm việc:

```bash
git checkout main && git pull
git checkout -b em1/b1-du-lieu
```

Rồi **luôn** kiểm tra B0 trước khi chạy gì:

```bash
python scripts/b0_chia_co_dinh.py --kiem-tra
```

Phải ra đủ 3 dòng `KHỚP`, dòng cuối là `(5,077,237 giao dịch ngày 0–9, 4,522 ca)`. Không khớp → dừng, báo Đạo.

## Việc 1 — Thống kê phân bổ dữ liệu

```bash
python scripts/b1_thong_ke.py
```

Script đọc đặc trưng (ngày 0–9) và bảng ngân hàng → nước, ghi ra `results/b1_thong_ke/`:

| File | Nội dung |
|---|---|
| `theo_ngay.csv` | mỗi ngày: số giao dịch, số ca, tỷ lệ %, % giao dịch, % ca |
| `theo_nuoc.csv` | mỗi nước (của ngân hàng gửi): như trên + số ngân hàng |
| `theo_ngan_hang_nhom.csv` | 4 nhóm cỡ ngân hàng (< 1k, 1k–10k, 10k–100k, ≥ 100k giao dịch) |
| `theo_loai_thanh_toan.csv` | 7 loại thanh toán (ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment, Wire) |
| `tom_tat.json` | các số chính để viết báo cáo |
| `1_ty_le_theo_nuoc.png`, `2_co_ngan_hang.png`, `3_ty_le_theo_loai_thanh_toan.png` | biểu đồ |

**Kiểm tra số:** dòng đầu màn hình phải là
`5,077,237 giao dịch, 4,522 ca (0.089%), 30,470 ngân hàng, 34 nước`. Khác → báo Đạo.

**Câu hỏi báo cáo phải trả lời** (mỗi câu 1–2 dòng, kèm số từ file csv):

1. Gian lận dồn vào loại thanh toán nào? (xem `theo_loai_thanh_toan.csv`, cột `pct_ca`)
2. Gian lận có dồn vào vài nước không? Nước nào tỷ lệ cao bất thường, và nước đó có bao nhiêu ca (ít ca thì tỷ lệ
   dễ dao động)?
3. Tỷ lệ gian lận theo ngày có đều không? Ngày nào khác hẳn?
4. Ngân hàng nhỏ (< 1k giao dịch) chiếm bao nhiêu % giao dịch và bao nhiêu % ca?

## Việc 2 — Đường học: giữ bao nhiêu bản ghi là đủ

Mục đích: **giảm dữ liệu để máy chạy nhẹ hơn** (khoảng 1 triệu bản ghi) và đo xem mất bao nhiêu điểm PR-AUC.

Cách làm (script tự làm hết):

- Cách chia B0: fit 60% (~3,05 triệu dòng), val 20%, test 20%.
- Mỗi cỡ N: giữ **hết** ca rửa tiền trong phần fit, bốc ngẫu nhiên ca bình thường theo từng ngân hàng cho đủ N dòng,
  **bù trọng số** cho phần ca bình thường bị bỏ. Val và test **không** lấy mẫu (giữ tỷ lệ thật).
- Cỡ: 100 nghìn, 250 nghìn, 500 nghìn, **1 triệu**, đủ (`0` nghĩa là đủ).
- Mô hình MLP, lô 512, lr 0,001, tối đa 300 lượt, dừng nếu 30 lượt liền không tăng (giống kết quả 0,130 hiện có).
- 3 seed (42, 7, 2027) × 5 fold = 15 lần cho mỗi cỡ.

### Bước 2a — chạy thử (5 phút) để chắc máy chạy được

```bash
python scripts/b1_duong_hoc.py --co 100000 --seeds 42 --folds 0 --max-epochs 20 --patience 5 --ten thu
```

Cuối màn hình phải có bảng tóm tắt và dòng `Đã ghi .../results/b1_duong_hoc_thu`. Chạy thử xong thì xoá kết quả
thử cho gọn: các thư mục `results/ibm_cach1_thu_*` và `results/b1_duong_hoc_thu`.

### Bước 2b — chạy chính thức

Chạy cả 3 seed trong một lệnh (~8 giờ):

```bash
python scripts/b1_duong_hoc.py --co 100000 250000 500000 1000000 0 --seeds 42 7 2027
```

Hoặc **chia theo seed** để chạy song song trên 3 notebook Kaggle, mỗi notebook ~3 giờ
(dùng **Save Version → Save & Run All** để chạy nền, xem `HUONG_DAN_KAGGLE.md` mục 8). Ở ô 6 của notebook:

```python
!python scripts/b1_duong_hoc.py --co 100000 250000 500000 1000000 0 --seeds 42
```

(notebook thứ hai `--seeds 7`, thứ ba `--seeds 2027`).

Lưu ý:

- **Bị ngắt giữa chừng thì chạy lại đúng lệnh cũ:** cấu hình nào xong rồi script tự bỏ qua (`[bỏ qua] …`).
- Mỗi cấu hình tạo cache 2–3 GB trong `data/processed/cach1/`, script tự xoá sau khi chạy xong. Muốn giữ thì thêm
  `--giu-cache`.
- Thời gian ước tính mỗi seed (5 fold, CPU Kaggle): đủ ~1,5 giờ; 1 triệu ~35 phút; 500k ~20 phút; 250k ~12 phút;
  100k ~8 phút.

### Bước 2c — tổng hợp

Lệnh chạy ở 2b tự tổng hợp khi xong. Chạy ở nhiều nơi (3 notebook Kaggle) thì giải nén cả 3 file `ket_qua.zip` vào
`results/` trên máy rồi chạy:

```bash
python scripts/b1_duong_hoc.py --chi-tong-hop
```

Ra `results/b1_duong_hoc/`:

| File | Nội dung |
|---|---|
| `bang.csv` | mỗi dòng một lần chạy: cỡ, seed, fold, số dòng fit, số ca fit, PR-AUC, Recall@1%, VDR@1%, val PR-AUC, lượt tốt nhất, thời gian, RAM |
| `tom_tat.csv` | mỗi cỡ: số lần chạy, PR-AUC trung bình ± độ lệch chuẩn, Recall@1%, thời gian trung bình, RAM đỉnh, **% so với đủ** |
| `duong_hoc.png` | biểu đồ đường học, có vạch 95% của "đủ" |
| `de_xuat.txt` | kết luận theo tiêu chí chốt |

**Kiểm tra:** mỗi cỡ trong `tom_tat.csv` phải có `so_lan = 15`. Thiếu → `de_xuat.txt` ghi "CHƯA chính thức", chạy
bổ sung seed/fold còn thiếu rồi tổng hợp lại.

**Tiêu chí chốt B1 (đã viết sẵn, không đổi sau khi thấy số):** chọn **N nhỏ nhất đạt ≥ 95%** PR-AUC của "đủ".
Nếu 1 triệu chưa đạt 95% thì vẫn có thể chọn 1 triệu vì lý do máy, nhưng phải ghi rõ mất bao nhiêu % vào
`docs/decisions.md` (D26).

## Việc 3 — Bảng client theo nước

```bash
python scripts/b2_client_theo_nuoc.py
```

Quy tắc: client = nước của ngân hàng gửi; nước có < 10.000 giao dịch gộp chung một client `_khac`.
Script 09 (`--client nuoc`) dùng **cùng một hàm** (`fraudfl.data.quoc_gia.client_theo_nuoc`) nên FL theo nước luôn
dùng đúng bảng này.

Ra `results/b2_client/`:

| File | Nội dung |
|---|---|
| `client_theo_nuoc.csv` | mỗi nước: số giao dịch, số ca, số ngân hàng, tên client, mã client |
| `client.csv` | mỗi client: số nước, số giao dịch, % giao dịch, số ca, tỷ lệ %, số ca trong fit / val / test của fold 0 |
| `tom_tat.json` | số client, nước bị gộp, client lớn nhất, các client quá ít ca |

**Kiểm tra số:** phải ra `27 client từ 34 nước`, 8 nước gộp vào `_khac`
(Croatia, Lithuania, Slovenia, Estonia, Latvia, Cyprus, Malta, Luxembourg).

Muốn thử ngưỡng gộp khác (để đề xuất), ví dụ 50.000 giao dịch:

```bash
python scripts/b2_client_theo_nuoc.py --gop-nho 50000
```

Kết quả ghi vào thư mục riêng `results/b2_client_gop50000/`, không đè bảng chính.

**Câu hỏi báo cáo phải trả lời:**

1. Client lớn nhất (USA) chiếm bao nhiêu % giao dịch? Có nên tách USA thành nhiều client không, vì sao?
2. Script cảnh báo những client có < 10 ca trong test fold 0. Với các client đó, PR-AUC riêng sẽ dao động mạnh:
   đề xuất giữ nguyên, gộp thêm, hay chỉ báo PR-AUC chung?
3. Đề xuất ngưỡng gộp cuối cùng (10.000 hay khác) và lý do → ghi D27.

## Bàn giao

1. Commit trên nhánh `em1/b1-du-lieu`:
   - `results/b1_thong_ke/`, `results/b1_duong_hoc/`, `results/b2_client/`;
   - các thư mục `results/ibm_cach1_b1_*` (kết quả từng lần chạy);
   - báo cáo ngắn `results/b1_thong_ke/BAO_CAO.md`: 5–10 dòng kết luận + bảng chính của 3 việc + đề xuất D26 (cỡ dữ
     liệu) và D27 (bảng client).
2. Đẩy nhánh, mở PR vào `main`, chờ CI xanh, nhờ Đạo duyệt.
3. **Không** sửa `data/splits/` và các script b0/b1/b2. Thấy script sai thì báo Đạo, đừng tự sửa cho ra số.
