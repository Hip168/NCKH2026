# Hướng dẫn chạy thí nghiệm trên Kaggle

Dùng khi máy yếu (Windows, RAM < 8 GB). Kaggle cho máy ảo miễn phí: 4 nhân CPU, 30 GB RAM, tối đa 12 giờ mỗi phiên.
Đã thử 11/11/2026: Cách 1 fold 0 seed 42 trên Kaggle ra PR-AUC test 0,1384, máy Đạo 0,1402, dừng cùng lượt 59/89 → kết quả tái lập được.

## 1. Chuẩn bị tài khoản (làm một lần)

1. Đăng ký tài khoản ở kaggle.com.
2. Vào ảnh đại diện → **Settings** → mục **Phone verification** → xác minh số điện thoại.
   Không xác minh thì **không bật được Internet** cho notebook, mà notebook cần Internet để tải code từ GitHub.

## 2. Tạo notebook

1. Tải file `notebooks/kaggle_mau.ipynb` trong repo về máy
   (trên GitHub: mở file → nút **Download raw file**).
2. Trên Kaggle: **Create → New Notebook**.
3. Trong notebook mới: **File → Import Notebook** → chọn file `kaggle_mau.ipynb` vừa tải → **Import**.
4. Đổi tên notebook cho dễ nhận (bấm vào tên ở góc trên), ví dụ `B3_lightgbm_em2`.
   Notebook mặc định ở chế độ **Private**, giữ nguyên như vậy.

## 3. Thêm dữ liệu và bật Internet

1. Cột bên phải, mục **Input** → **Add Input**.
2. Gõ tìm `IBM Transactions for Anti Money Laundering` → chọn bộ của tác giả **ealtman2019** → bấm dấu **+**.
3. Kiểm tra: dưới **Input → DATASETS** thấy có `HI-Small_Trans.csv` và `HI-Small_accounts.csv` là đúng.
4. Cột bên phải, mục **Session options** (hoặc menu **Settings**):
   - **Internet: On**.
   - **Accelerator: None** (CPU). Chỉ chọn GPU khi chạy mô hình đồ thị (GNN).

## 4. Sửa ô đầu tiên: chọn nhánh code

```python
NHANH = 'main'
```

- Muốn chạy code chung đã chốt: để `'main'`.
- Muốn chạy code trên nhánh riêng của mình (chưa merge): đổi thành tên nhánh, ví dụ `'em2/lightgbm'`.
  Nhánh phải **đã push lên GitHub** thì Kaggle mới thấy.

## 5. Chạy

Bấm **Run All** (hoặc chạy từng ô bằng **Shift + Enter**). Lần đầu Kaggle mất 1–3 phút để bật máy ảo,
lúc đó dưới cùng hiện "Draft Session Waiting for Jupyter", cứ chờ.

Từng ô làm gì và phải thấy gì:

| Ô | Làm gì | Thời gian | Phải thấy |
|---|---|---|---|
| 1 | Tải code từ GitHub về máy ảo | vài giây | dòng cuối là một commit, ví dụ `dff2247 Phiếu giao việc…` |
| 2 | Cài thư viện thiếu (opacus, torch-geometric) và chạy test | 1–2 phút | dòng cuối `… passed`, không có `failed` |
| 3 | Tìm đường dẫn file dữ liệu IBM | vài giây | 2 đường dẫn `/kaggle/input/…/HI-Small_Trans.csv` và `…accounts.csv` |
| 4 | Tính đặc trưng từ dữ liệu gốc | 3–5 phút | không báo lỗi; tạo `data/processed/HI-Small_features_bank.parquet` |
| 5 | **Kiểm tra B0** (xem mục 6) | ~1 phút | đủ 3 dòng `KHỚP` |
| 6 | Chạy thí nghiệm | tuỳ thí nghiệm; mẫu Cách 1 fold 0 ~18 phút | dòng `TEST PR-AUC …` ở cuối |
| 7 | Nén kết quả mới thành `ket_qua.zip` | vài giây | danh sách file trong zip |

## 6. Kiểm tra B0 là gì, đọc thế nào

Cả nhóm phải dùng **đúng một cách chia** train/val/test (đã chốt ở bước B0), thì số của người này mới so được với số
của người kia. Ô 5 kiểm tra việc đó. Kết quả đúng phải như sau:

```
ibm_hi_small_chia.parquet: KHỚP
ibm_ngan_hang_quoc_gia.csv: KHỚP
HI-Small_features_bank.parquet: KHỚP (5,077,237 giao dịch ngày 0–9, 4,522 ca)
```

- Dòng 1: file chia fold trong repo không bị ai sửa.
- Dòng 2: bảng ngân hàng → quốc gia trong repo không bị ai sửa.
- Dòng 3: dữ liệu vừa tính trên Kaggle có cùng số giao dịch, cùng số ca gian lận và cùng danh sách mã giao dịch
  với cách chia đã chốt.

**Có dòng nào ghi `KHÁC` hoặc thiếu dòng → DỪNG, không chạy tiếp, chụp màn hình gửi Đạo.**
Kết quả kiểm tra cũng được lưu vào `results/kaggle_b0_kiem_tra.txt` trong file zip.

## 7. Đổi thí nghiệm

Chỉ sửa **ô 6**, các ô khác giữ nguyên. Ví dụ:

```python
!python scripts/12_ibm_cach1_lo_nho.py --folds 0 1 2 3 4 --seed 7 --max-epochs 300 --patience 30 --tag ten_thi_nghiem
```

- Luôn đặt `--tag` rõ nghĩa, kết quả sẽ nằm ở `results/<tên script>_<tag>/`.
- Lệnh của từng bước xem phiếu giao việc trong `docs/giao_viec/` và `docs/HUONG_DAN_CHAY.md`.
- Mỗi bước B1–B6 nên là một notebook riêng: mở notebook mẫu → **File → Save a copy**/**Copy & Edit**, rồi chỉ sửa ô 6.

## 8. Chạy lâu: dùng Save Version để chạy nền

Chạy kiểu thường (Run All) thì **đóng tab là phiên tắt sau một lúc** và mất kết quả. Thí nghiệm dài (> 30 phút):

1. Bấm **Save Version** (góc trên phải) → chọn **Save & Run All (Commit)** → **Save**.
2. Kaggle chạy toàn bộ notebook ở chế độ nền, tối đa 12 giờ. Có thể tắt máy.
3. Xem tiến độ: trang notebook → tab **Versions** hoặc biểu tượng chuông thông báo.
4. Xong thì mở phiên bản đó → tab **Output** → tải `ket_qua.zip`.

## 9. Lấy kết quả về

- Chạy kiểu thường: cột bên phải **Output → /kaggle/working** → bấm dấu ⋮ cạnh `ket_qua.zip` → **Download**.
- Chạy kiểu Save Version: xem mục 8.
- Giải nén vào thư mục `results/` của repo trên máy, rồi commit trên nhánh của mình và mở PR như bình thường.
  **Không** sửa file trong repo trực tiếp trên Kaggle (thay đổi trên Kaggle không tự lên GitHub).

## 10. Lỗi hay gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `Could not resolve host: github.com` | Internet đang tắt | Settings → Internet: On, chạy lại ô 1 |
| Không bật được Internet | Tài khoản chưa xác minh số điện thoại | Làm mục 1 |
| `Remote branch … not found` | Sai tên nhánh hoặc nhánh chưa push | Kiểm tra tên ở ô 1, push nhánh lên GitHub |
| `getcwd: cannot access parent directories` | Notebook bản cũ, chạy lại ô 1 | Lấy notebook mới nhất trong repo, hoặc thêm `%cd /kaggle/working` trước dòng `!rm -rf` |
| `IndexError: list index out of range` ở ô 3 | Chưa thêm bộ dữ liệu IBM | Làm mục 3 |
| Ô 5 có dòng `KHÁC` | Dữ liệu hoặc code đặc trưng khác bản đã chốt | Dừng, báo Đạo |
| Phiên tự tắt giữa chừng | Đóng tab hoặc để quá lâu không thao tác | Dùng Save Version (mục 8) |
| `Killed` / hết RAM | Thí nghiệm quá nặng | Giảm `--rate` hoặc chạy ít fold mỗi lần |

## 11. Giới hạn miễn phí

- CPU: mỗi phiên tối đa 12 giờ, chạy được nhiều phiên.
- GPU: khoảng 30 giờ mỗi tuần, chỉ dùng khi chạy mô hình đồ thị.
- Bộ nhớ đĩa `/kaggle/working`: khoảng 20 GB; dữ liệu đặc trưng và cache tự sinh lại nên không cần giữ.
