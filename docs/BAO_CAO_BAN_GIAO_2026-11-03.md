# Báo cáo bàn giao — cập nhật 03/11/2026

Dành cho cả nhóm sau khi pull về. Đọc khoảng 20 phút. Sau khi đọc, mỗi người biết: đề tài đang hướng tới gì, đã làm
được gì, kết quả so với mục tiêu ra sao, cách chạy lại, và **việc tiếp theo của mình**.

Tổng hợp từ: [nhat_ky_ket_qua.md](nhat_ky_ket_qua.md) (23 giai đoạn, có đủ số liệu và đính chính),
[decisions.md](decisions.md) (sổ quyết định D1–D20), [luu_y_dp_muc_client.md](luu_y_dp_muc_client.md),
[ghi_chu_cach_A_B.md](ghi_chu_cach_A_B.md). Báo cáo trước đó: [BAO_CAO_TONG_HOP.md](BAO_CAO_TONG_HOP.md) (01/11).

---

## 0. Lấy code về

Code mới nằm trên nhánh **`dat/dp-client-lay-mau-ibm`** (25 commit, **chưa merge vào `main`**).

```bash
cd project
git fetch
git switch dat/dp-client-lay-mau-ibm
pip install -r requirements.txt          # Python 3.11
python -m pytest -q tests --ignore=tests/test_consumer.py   # phải đạt 36/36
```

`tests/test_consumer.py` cần thêm gói hạ tầng của Đạo (`pip install -r requirements-ha-tang.txt`). Pull request:
https://github.com/trantatdat06/NCKH2026/pull/new/dat/dp-client-lay-mau-ibm — **B review trước khi merge.**

---

## 1. Tóm tắt trong 6 dòng

1. Đề tài đã chuyển dữ liệu chính sang **IBM AML HI-Small** (D19), DP giữ ở **mức client** (D12, đúng Slide 4).
2. Synthetic 5 ngân hàng: FL rất tốt (98% gộp chung) nhưng **DP mức client sụp** ở mọi ε ≤ 8 vì chỉ có 5 ngân hàng.
3. IBM với **802 ngân hàng** (D20): DP mức client ở **ε = 1 học được**, giữ **85% ± 8** so với FL không DP.
4. Nhưng FL trên 802 ngân hàng rất khác nhau chỉ đạt **24–30% gộp chung** → FL + DP = **25%** gộp chung.
5. Theo tiêu chí Slide 10 (FL + DP ≥ 85% gộp chung): **chưa đạt**. ε ≤ 1: **đạt**. Phần hụt do FL, không do DP.
6. Việc gấp nhất: họp chốt lại tiêu chí; chạy 3 seed + kiểm định thống kê; E9/E10 chưa làm.

---

## 2. Đề tài hướng tới gì (Slide 2, 3, 10)

**Câu hỏi nghiên cứu:**
- RQ1: FL + DP hiệu quả thế nào so với gộp chung / tự học riêng?
- RQ2: Đánh đổi riêng tư – hiệu năng khi quét ε?
- RQ3: Khái quát chéo bộ dữ liệu (train IBM → đánh giá synthetic / Elliptic)?

**Tiêu chí thành công (Slide 10):**

| Tiêu chí | Ngưỡng |
|---|---|
| Prototype | FL + DP trên ≥ 3 client, xử lý non-IID |
| **Hiệu năng chính** | **FL + DP ≥ 85% gộp chung** (PR-AUC) trên cả 2 bộ dữ liệu |
| Bảo mật | ε ≤ 1, δ ≤ 1e-5 |
| Phụ | VDR ≥ 0,80; PR-AUC ≥ 0,70 |
| Bài báo | ≥ 3 biểu đồ (VDR theo ε, độ trễ, hội tụ) |

Lưu ý: "PR-AUC ≥ 0,70" chỉ hợp với synthetic (1,5% gian lận). IBM có 0,089% rửa tiền → mức ngẫu nhiên PR-AUC chỉ
0,0009, nên không dùng ngưỡng 0,70 cho IBM. Mọi bảng kết quả mới đều quy về **% gộp chung cùng fold**
(`scripts/11_so_voi_muc_tieu.py`).

---

## 3. Kết quả so với mục tiêu — tách riêng 2 bộ dữ liệu

### 3.1 Synthetic (5 ngân hàng, ~500 nghìn giao dịch, 5 fold, 1 seed)

| Cách | PR-AUC | % gộp chung (mục tiêu ≥ 85%) | Slide 7 kỳ vọng |
|---|---|---|---|
| E1 Gộp chung | 0,733 | 100% | 0,88 |
| E2 Tự học riêng | 0,706 | 96% | 0,55 (~60%) |
| E3 FedAvg | 0,715 | 98% | 0,78 |
| E4 FL + DP ε = 1, **mức bản ghi** | 0,657 | 90% ✅ | 0,72 |
| E4 FL + DP ε = 1, **mức client** (đã chốt) | 0,028 | **4% ❌** | 0,72 |

- DP mức client thử thêm (fold 0): đổi chỗ cộng nhiễu, thu nhỏ GNN, đổi sang MLP, giảm vòng — đều sụp ở ε ≤ 8; chỉ học
  được từ ε ≈ 16–32 (giai đoạn 4, 11). → **5 ngân hàng là quá ít cho DP mức client.**

### 3.2 IBM HI-Small (802 ngân hàng, 4,56 triệu giao dịch, ngày 0–9) — KẾT QUẢ CHÍNH, 5 fold, 1 seed

| Cách | PR-AUC | % gộp chung (mục tiêu ≥ 85%) | % FedProx không DP | Đạt? |
|---|---|---|---|---|
| E7 Gộp chung | 0,0979 ± 0,0089 | 100% | | mốc |
| FedAvg thường | 0,0236 ± 0,0022 | 24% ± 4 | | mốc |
| E6 FedProx μ = 0,1 (không DP) | 0,0291 ± 0,0019 | 30% ± 3 | 100% | mốc |
| DP ε = 8 | 0,0260 ± 0,0014 | 27% ± 3 | 90% ± 2 | ❌ |
| DP ε = 4 | 0,0256 ± 0,0012 | 26% ± 3 | 88% ± 3 | ❌ |
| DP ε = 2 | 0,0255 ± 0,0016 | 26% ± 4 | 88% ± 7 | ❌ |
| **E8 DP ε = 1** | **0,0246 ± 0,0017** | **25% ± 4** | **85% ± 8** | ❌ thiếu 60 điểm % |

Slide 7 kỳ vọng IBM: gộp chung PR-AUC ~0,82, FL + DP ~0,68 (~83%). Thực tế khó hơn nhiều.

### 3.3 Đọc kết quả

| | Synthetic | IBM |
|---|---|---|
| Số ngân hàng | 5 | 802 |
| FL / gộp chung | **98%** ✅ | **24–30%** ❌ |
| DP ε = 1 / FL | **sụp** ❌ | **85%** ✅ |
| Vấn đề chính | quá ít ngân hàng cho DP | ngân hàng quá khác nhau cho FL |

**Luận điểm cho bài báo:** DP mức client cần nhiều ngân hàng tham gia (5 và 28 ngân hàng đều sụp; 802 thì không), nhưng
nhiều ngân hàng thì dữ liệu khác nhau hơn và FL khó học hơn. Chi phí của DP (~15%) nhỏ hơn nhiều chi phí của FL (~70%).

### 3.4 Đề xuất chỉnh tiêu chí (CHỜ NHÓM DUYỆT)

Tách "FL + DP ≥ 85% gộp chung" thành 2 tầng:
- **Chi phí DP:** DP / FL không DP ≥ 85% → **đạt (85% ± 8)**.
- **Chi phí FL:** FL / gộp chung → báo cáo trung thực (24–30%) kèm phân tích nguyên nhân (non-IID).

---

## 4. Diễn biến chính (chi tiết ở nhật ký)

| Giai đoạn | Việc | Kết quả / bài học |
|---|---|---|
| 1–3 | Pipeline synthetic, E1–E5, hạ tầng Flower/Kafka (Đạo) | FL tốt; DP mức client sụp |
| 4 | Chốt DP mức client; rà code | Phát hiện #32 fold ngẫu nhiên, #33 điểm tự học riêng bị xếp chung, #34 ngân sách cảnh báo chung |
| 5–7 | Đưa IBM HI-Small vào, đặc trưng, lấy mẫu | Đuôi dữ liệu ngày 10–17 lỗi (~60% rửa tiền) → D18 |
| 8–9 | Chứng minh cách lấy mẫu bằng 5 fold | Train r nhỏ là đủ; test phải giữ đầy đủ |
| 10–11 | Cách A (GraphSAGE) / B (MLP) trên synthetic | Cả hai sụp → bắt buộc tăng số ngân hàng |
| 12 | Đính chính mã ngân hàng IBM | "010" = "0010" → 30.470 ngân hàng (không phải 30.528) |
| 13–15 | IBM 27 ngân hàng + 1 client gộp; sửa MLP (mini-batch); sửa lỗi đo #41 | K = 28 vẫn sụp |
| 16–17 | IBM 802 ngân hàng; làm FedAvg học nhanh hơn | Nhiễu đủ nhỏ; FedAvg 5 epoch, lr 5e-3 |
| 18–19 | Tách nguyên nhân DP tụt | 75% do trung bình đều; nhiễu gần như không tốn |
| 20–22 | FedProx; chọn lại C theo độ lớn cập nhật | FedProx μ = 0,1 tốt nhất; C = 0,35 (trung vị ‖Δ‖) |
| 23 | **Chạy chính 5 fold** | Bảng mục 3.2 |

Các lần **đính chính** (đều giữ lại trong nhật ký): giai đoạn 6 → 9 (PR-AUC test lấy mẫu), 5 → 12 (mã ngân hàng),
10 (câu "MLP chịu DP tốt hơn"), 13 → 14 (Recall@1% ảo khi mô hình sụp, lỗi #41).

---

## 5. Quyết định đã chốt (D12–D20)

| # | Nội dung ngắn |
|---|---|
| D12 | Giữ **DP mức client** (đúng Slide 4) |
| D17 | Lấy mẫu con ca bình thường cho train (anh Hùng duyệt). **Chờ chốt:** r và việc giữ val/test đầy đủ |
| D18 | IBM chỉ dùng **ngày 0–9**; bỏ đặc trưng tuổi tài khoản khỏi mô hình |
| D19 | **IBM HI-Small là dữ liệu chính**; synthetic giữ làm thử nghiệm ban đầu |
| D20 | Client = **802 ngân hàng** có ≥ 1.000 giao dịch (bỏ ngân hàng nhỏ hơn); 200 ngân hàng/vòng |

D7–D11, D13 vẫn ở mục "đề xuất" trong decisions.md, **chờ nhóm duyệt**.

**Cấu hình chính hiện tại** (giai đoạn 22): MLP 22 đặc trưng (view bank), lô 512, lr 5e-3, 5 epoch cục bộ, 30 vòng,
pos_weight 1, train r = 0,1, FedProx μ = 0,1, trung bình đều, DP: cắt C = 0,35 + nhiễu ở server, δ = 1e-5.

---

## 6. Chạy lại từ đầu

### 6.1 Dữ liệu IBM (không có trên git — mỗi người tự tải)

Tải IBM AML từ Kaggle ("IBM Transactions for Anti Money Laundering"), đặt vào `project/data/raw/ibm_aml/`:
`HI-Small_Trans.csv` (454 MB), `HI-Small_accounts.csv`. RAM cần ~2 GB cho mỗi lần chạy.

### 6.2 Các bước (từ thư mục `project/`)

| # | Lệnh | Thời gian | Ra |
|---|---|---|---|
| 1 | `python scripts/06_ibm_features.py --input data/raw/ibm_aml/HI-Small_Trans.csv --view global` | ~3 phút | đặc trưng 28 cột (E7) |
| 2 | `python scripts/06_ibm_features.py --input data/raw/ibm_aml/HI-Small_Trans.csv --view bank` | ~2 phút | đặc trưng 22 cột (FL) |
| 3 | Mốc (5 fold): xem 6.3 lệnh A | ~35 phút | `results/ibm_fl_dp_CHINH_5fold_moc/` |
| 4 | FedProx + DP (5 fold): xem 6.3 lệnh B | ~50 phút | `results/ibm_fl_dp_CHINH_5fold_fedprox_dp/` |
| 5 | `python scripts/11_so_voi_muc_tieu.py results/ibm_fl_dp_CHINH_5fold_moc results/ibm_fl_dp_CHINH_5fold_fedprox_dp --out CHINH_5fold` | vài giây | bảng so với mục tiêu |

### 6.3 Lệnh chính

```bash
B="--min-tx 1000 --clients-per-round 200 --rate 0.1 --rounds 30 --batch-size 512 --lr 5e-3 --local-epochs 5 --pos-weight 1 --folds 0 1 2 3 4"
# A. mốc: gộp chung + FedAvg thường
python scripts/09_ibm_fl_dp.py $B --settings central fedavg --tag CHINH_5fold_moc
# B. FedProx + DP mức client, quét ε
python scripts/09_ibm_fl_dp.py $B --settings fedavg_deu dp --prox-mu 0.1 --clip 0.35 --eps 8 4 2 1 --tag CHINH_5fold_fedprox_dp
```

Muốn đổi seed: thêm `--seed S` và đổi `--tag`. Thử nhanh: `--folds 0`.

### 6.4 Bản đồ script

| Script | Việc |
|---|---|
| `01`–`03`, `10`, `20`, `30` | Synthetic: sinh dữ liệu, chia fold, EDA, baseline, GNN E1–E3, DP (`30 --mode client/record`) |
| `05_ibm_sample.py` | Lấy mẫu IBM cố định theo D17 (lưu `data/splits/ibm_sample_ids.csv`). **Chưa chạy chính thức** — chờ chốt r |
| `06_ibm_features.py` | Đặc trưng IBM (chỉ quá khứ, không nhãn; tỷ giá suy từ dữ liệu) |
| `07_ibm_kiem_tra_dac_trung.py` | Kiểm tra nhanh đặc trưng bằng mô hình cây |
| `08_ibm_danh_gia_lay_mau.py` | Bằng chứng 5 fold cho cách lấy mẫu |
| `09_ibm_fl_dp.py` | **Thí nghiệm chính FL + DP trên IBM** |
| `11_so_voi_muc_tieu.py` | Bảng % so với mục tiêu Slide 10 |
| `50`–`53` | Hạ tầng Flower / real-time của Đạo (đang dùng synthetic) |

### 6.5 Quy tắc lưu trữ (bắt buộc)

- Mọi script tự chép kết quả vào `results/_luu_tru/<ngày-giờ>_<tên>/` + ghi `results/NHAT_KY_CHAY.csv`. Không xóa,
  không ghi đè kết quả cũ.
- Hết mỗi giai đoạn: thêm mục vào `docs/nhat_ky_ket_qua.md` (kể cả kết quả xấu, đính chính).
- Mọi bảng kết quả kèm cột **% so với mục tiêu**.
- Đổi quyết định quan trọng → ghi `docs/decisions.md` TRƯỚC khi chạy.

---

## 7. Hạn chế phải ghi trong bài báo

1. Chỉ **1 seed**; chưa có Wilcoxon / bootstrap CI (Slide 8b yêu cầu 3 seed).
2. IBM dùng **MLP**, không phải GraphSAGE (GNN trên đồ thị 5 triệu nút quá nặng; xem ghi_chu_cach_A_B.md).
3. Ngưỡng cắt C = 0,35 chọn từ lần chạy không DP (ngoài ngân sách ε) → nên dùng cắt thích ứng (Andrew et al., 2021).
4. Siêu tham số chọn trên fold 0 (validation); số lần thử chưa tính vào ε.
5. Danh sách 802 ngân hàng chọn theo số giao dịch trên toàn ngày 0–9 (chỉ để chọn bên tham gia).
6. View bank (FL) thiếu thông tin tài khoản nhận; gộp chung trong bảng cũng dùng view bank để so công bằng.
7. VDR@1% trên IBM dao động rất mạnh — chỉ báo cáo làm chỉ số phụ.
8. Không có SecAgg; Flower chạy `--insecure`.
9. Synthetic: fold chia ngẫu nhiên (không theo thời gian, #32) — có thể lạc quan.

---

## 8. Việc tiếp theo — đề xuất phân công

Thứ tự ưu tiên: (1) họp chốt → (2) độ tin cậy thống kê → (3) lấp mục tiêu còn thiếu → (4) viết.

### Cả nhóm — họp tuần này
- Duyệt **tiêu chí 2 tầng** (mục 3.4) hoặc giữ nguyên tiêu chí cũ.
- Anh Hùng chốt **D17** (r cho train; val/test giữ đầy đủ) và xác nhận **D18**.
- Duyệt D7–D11, D13.
- Quyết định có làm E9/E10 (điểm "độc đáo nhất" theo Slide 3) hay ghi là hướng mở.

### A — BA/PO
- Sửa Slide 4 (DP: nhiễu ở server, chọn ngẫu nhiên ngân hàng, C = 0,35), Slide 5 (IBM chia theo ngân hàng, ngày 0–9),
  Slide 6 (MLP cho IBM, siêu tham số mới), Slide 7 (thay kết quả kỳ vọng bằng số thật), Slide 10 (tiêu chí 2 tầng).
- Dàn ý bài báo theo luận điểm mục 3.3.

### B — Tech lead
- **Review và merge** PR nhánh `dat/dp-client-lay-mau-ibm`.
- Thu hẹp khoảng cách FL – gộp chung (nút thắt lớn nhất): thử SCAFFOLD, nhiều vòng hơn, phân cụm ngân hàng.
- Cắt thích ứng thay cho C cố định (hạn chế 3).

### C — Research lead
- Tổng quan: DP mức client cần nhiều bên tham gia (McMahan et al., 2018); FedProx (Li et al., 2020); non-IID trong
  cross-silo FL; cắt thích ứng (Andrew et al., 2021). Kiểm tra lại mọi trích dẫn trong docs.
- Viết Related Work + Limitations (mục 7).

### D — Data engineer
- Chạy `05_ibm_sample.py` chính thức khi D17 được chốt; đồng bộ `09` dùng file mẫu cố định đó.
- Data card cho IBM HI-Small (30.470 ngân hàng, 0,089% rửa tiền, đuôi lỗi ngày 10–17).
- Chuẩn bị Elliptic nếu làm E10.

### E — Experiment runner
- **3 seed** cho thí nghiệm chính (mục 6.3, thêm `--seed 7`, `--seed 2027`; ~1,5 giờ/seed).
- Script thống kê (`scripts/60_stats.py`): Wilcoxon, bootstrap CI, effect size trên `per_fold.csv`.
- Thêm **ε = 0,5** (`--eps 0.5`), và **tự học riêng** với 802 ngân hàng (nặng — có thể lấy mẫu ngân hàng).
- Biểu đồ: PR-AUC theo ε (IBM), PR-AUC theo số ngân hàng (5 / 28 / 802), đường hội tụ.

### Đạo — hạ tầng
- Biểu đồ độ trễ (tách phần mạng / khởi động).
- Cân nhắc chuyển demo real-time sang dữ liệu IBM (hiện dùng synthetic).

---

## 9. Câu hỏi khó hội đồng có thể hỏi

| Câu hỏi | Trả lời ngắn |
|---|---|
| Sao không đạt 85%? | Phần hụt do FL với 802 ngân hàng non-IID (24–30%), DP chỉ tốn ~15%. Có bảng phân rã. |
| Sao không dùng DP mức bản ghi (đạt 90% trên synthetic)? | Nhóm chọn mức client để che được cả một ngân hàng (D12); kết quả mức bản ghi nêu làm so sánh. |
| Sao bỏ ngày 10–17 của IBM? | ~60% là rửa tiền — lỗi bộ sinh dữ liệu; bỏ chỉ mất 0,02% giao dịch. |
| Lấy mẫu có làm PR-AUC đẹp giả? | Chỉ lấy mẫu train; val/test giữ đầy đủ (giai đoạn 9). |
| PR-AUC 0,025 có quá thấp? | Mức ngẫu nhiên 0,0009 → gấp ~27 lần; gộp chung 0,098 (gấp ~110 lần). |
| Sao IBM không dùng GNN? | Đồ thị 5 triệu nút quá nặng cho máy nhóm; MLP + đặc trưng cửa sổ; GNN là hướng mở. |
