# Hướng dẫn chạy repo (cập nhật 05/11/2026)

Thời gian ghi trong bảng là đo trên MacBook M4 16 GB; máy Windows của Đạt chậm hơn khoảng 1,5–2 lần.
Mọi lệnh chạy từ thư mục gốc của repo (thư mục có `README.md`).

## 1. Cài đặt (làm 1 lần)

```bash
git clone https://github.com/Hip168/NCKH2026.git
cd NCKH2026
git checkout main                       # hoặc nhánh của mình
```

Cần Python **3.11**.

| | Mac / Linux | Windows (PowerShell) |
|---|---|---|
| Tạo môi trường | `python3.11 -m venv .venv` | `py -3.11 -m venv .venv` |
| Bật môi trường | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| Cài thư viện | `pip install -r requirements.txt -r requirements-ha-tang.txt` | như bên trái |

Kiểm tra cài đúng chưa: `python -m pytest -q` → mọi test phải qua (`passed`, không có `failed`), khoảng 10 giây.

## 2. Dữ liệu IBM HI-Small (không có trên git)

Tải bộ "IBM Transactions for Anti Money Laundering" trên Kaggle, lấy file `HI-Small_Trans.csv` (454 MB),
đặt vào `data/raw/ibm_aml/` (hoặc để chỗ khác rồi trỏ `--input` tới đó).
Máy Đạo đang có sẵn ở `~/Documents/NCKH/AI/aml_fraud_detection/archive/HI-Small_Trans.csv`.

Tính đặc trưng (1 lần, khoảng 1 phút, RAM ~2,4 GB):

```bash
python scripts/06_ibm_features.py --input data/raw/ibm_aml/HI-Small_Trans.csv --view bank
```

Ra `data/processed/HI-Small_features_bank.parquet` (280 MB). Mọi bước dưới đọc file này.

## 3. Các thí nghiệm chính

**Từ 11/11 làm theo khung chốt từng bước: `docs/KHUNG_CHOT_BUOC.md`.** Đầu mỗi bước chạy
`python scripts/b0_chia_co_dinh.py --kiem-tra` (phải ra 3 dòng `KHỚP`). Chạy trên Kaggle: `notebooks/kaggle_mau.ipynb`, hướng dẫn từng bước ở `docs/HUONG_DAN_KAGGLE.md`.
Các lệnh dưới đây là cách chạy cũ (mỗi script tự chia fold), giữ để tham khảo.

Thử nhanh bất kỳ lệnh nào: đổi `--folds 0 1 2 3 4` thành `--folds 0` (nhanh gấp 5).
Muốn seed khác: thêm `--seed 7` và đổi `--tag`.

### 3.1 Học gộp chung đủ dữ liệu (Cách 1, kết quả tốt nhất: PR-AUC 0,130)

```bash
python scripts/12_ibm_cach1_lo_nho.py --folds 0 1 2 3 4 --batch-size 512 --lr 0.001 \
    --max-epochs 300 --patience 30 --tag du_300luot
```

Khoảng 25 phút. Ra `results/ibm_cach1_du_300luot/`. Lần đầu tự tạo bộ đệm ở `data/processed/cach1/`
(~2 GB mỗi seed, xoá được, chạy lại tự sinh).

Biến thể:
- Lấy mẫu con 100 nghìn dòng + bù trọng số: thêm `--so-mau 100000 --bu-trong-so`.
- Đủ 30.470 ngân hàng (không bỏ ngân hàng nhỏ): thêm `--min-tx 0`.
- Đối chứng không xáo trộn: thêm `--thu-tu co_dinh`.

### 3.2 Học liên kết + DP (thí nghiệm chính của Đạt)

Phải chạy bằng **bash** (zsh trên Mac không tách `$B` thành nhiều tham số → gõ `bash` trước).

```bash
B="--min-tx 1000 --clients-per-round 200 --rate 0.1 --batch-size 512 --lr 5e-3 --local-epochs 5 --pos-weight 1 --folds 0 1 2 3 4"
# A. Mốc: gộp chung + FedAvg thường (30 vòng)
python scripts/09_ibm_fl_dp.py $B --rounds 30 --settings central fedavg --tag CHINH_5fold_moc
# B. FedProx + DP mức client, quét ε, 100 vòng (cấu hình chính hiện tại, giai đoạn 27)
python scripts/09_ibm_fl_dp.py $B --rounds 100 --settings fedavg_deu dp --prox-mu 0.1 --clip 0.35 --eps 8 4 2 1 --tag CHINH_5fold_100vong
# C. Tự học riêng từng ngân hàng (mốc dưới)
python scripts/09_ibm_fl_dp.py $B --settings local --tag CHINH_5fold_local
```

Thời gian: A khoảng 15 phút, B khoảng 50 phút.

Sau khi chạy:

| Lệnh | Ra |
|---|---|
| `python scripts/11_so_voi_muc_tieu.py results/ibm_fl_dp_CHINH_5fold_moc results/ibm_fl_dp_CHINH_5fold_100vong --out CHINH_5fold_100vong` | bảng % so với mục tiêu |
| `python scripts/15_thong_ke.py --nhom 42:results/ibm_fl_dp_CHINH_5fold_moc,results/ibm_fl_dp_CHINH_5fold_local,results/ibm_fl_dp_CHINH_5fold_100vong` | kiểm định Wilcoxon, Holm, khoảng tin cậy |
| `python scripts/16_bieu_do_ibm.py` | 4 biểu đồ cho bài báo ở `results/bieu_do_ibm/` |

### 3.3 Kiểm tra "ngân hàng khác nhau" (non-IID)

```bash
python scripts/14_ibm_ngan_hang_ngau_nhien.py        # tạo bản ngân hàng ngẫu nhiên, vài giây
# rồi chạy lại lệnh 3.2 với thêm --stem HI-Small-IID và --tag khác, so với bản thật
```

## 4. Xem kết quả

- Mỗi lần chạy ghi vào `results/<tên>/` (`per_fold.csv`, `log.txt`) và tự lưu bản sao ở `results/_luu_tru/`,
  thêm 1 dòng vào `results/NHAT_KY_CHAY.csv`. Không xoá kết quả cũ.
- Xem log khi đang chạy: `tail -f results/<tên>/log.txt`.
- Giao diện web MLflow (tuỳ chọn, môi trường riêng): xem đầu file `scripts/13_dong_bo_mlflow.py`.
- Diễn giải kết quả: `docs/nhat_ky_ket_qua.md` (theo giai đoạn), `docs/BAO_CAO_CACH1_VA_NON_IID_2026-11-04.md`,
  `docs/BAO_CAO_BAN_GIAO_2026-11-03.md`, quyết định ở `docs/decisions.md`.

## 5. Hệ thống real-time bằng Docker (synthetic, không cần Python)

Mở Docker Desktop rồi: Mac/Linux `sh chay.sh` (tắt: `sh tat.sh`), Windows `chay.bat` (tắt: `tat.bat`).
Lần đầu build khoảng 5–15 phút. Chi tiết: `docs/huong_dan_docker.md`.

## 6. Quy trình đẩy code

1. Làm trên nhánh riêng (`dao/...`, `dat/...`), không làm thẳng trên `main`.
2. Trước khi đẩy: `python -m pytest -q` phải qua hết.
3. `git push origin <nhánh>` → GitHub tự chạy test (tab **Actions**, file `.github/workflows/test.yml`, khoảng 1 phút).
4. Mở PR vào `main`; chỉ merge khi dấu tích xanh.
5. Kết quả mới: thêm mục vào `docs/nhat_ky_ket_qua.md`. Nếu bị xung đột ở `results/NHAT_KY_CHAY.csv`
   thì giữ cả hai bên và xếp theo thời gian (file chỉ ghi thêm dòng).
