# Báo cáo 04/11/2026: Cách 1 (lô nhỏ, đủ dữ liệu) và kiểm tra giả thuyết "ngân hàng khác nhau quá"

Người chạy: Đạo. Dữ liệu: IBM HI-Small, ngày 0–9. Máy: MacBook Pro M4, 16 GB RAM, 4 luồng CPU cho PyTorch.
Mọi số là PR-AUC trên **test đầy đủ**, trung bình ± độ lệch chuẩn qua 5 fold (mức đoán ngẫu nhiên ≈ 0,0009).
Kết quả nằm trong `results/ibm_cach1_*`, `results/ibm_fl_dp_IID_*`, `results/ibm_fl_dp_THAT_*` và đã lưu trữ ở `results/_luu_tru/`.
Xem bằng MLflow: `scripts/13_dong_bo_mlflow.py` (hướng dẫn ở đầu file).

## Tóm tắt

1. **Cách 1 (học theo lô nhỏ trên toàn bộ train, không lấy mẫu) tốt hơn gộp chung lấy mẫu 10%:**
   0,1118 ± 0,0102 (3 seed × 5 fold) so với 0,0979 của Đạt. Cùng vòng học của mình, dùng đủ dữ liệu hơn lấy mẫu 10%
   ở cả 5/5 fold (0,1125 so với 0,1023).
2. **Không xáo trộn thì mô hình hỏng:** đọc theo thứ tự thời gian chỉ còn 0,0189 (khoảng 17%).
3. **Cỡ khối 10 nghìn đến 3 triệu dòng không đổi thời gian** (khoảng 2,0–2,6 giây mỗi lượt) và RAM chỉ 0,8–1,3 GB.
   Với MLP trên HI-Small, máy nhóm không tràn RAM ở bất kỳ cỡ nào đã thử. Chọn 100.000 dòng là hợp lý nhưng không quan trọng.
4. **Đủ 30.470 ngân hàng** (thêm 11% ca rửa tiền của ngân hàng nhỏ): 0,1090 ± 0,0104; mô hình bắt được rửa tiền ở
   ngân hàng nhỏ ngang ngân hàng lớn (0,112 so với 0,110).
5. **Câu "chỉ số FL thấp vì ngân hàng khác nhau quá" chỉ đúng một phần.** Cho mọi ngân hàng giống hệt nhau (chia ngẫu
   nhiên) thì:
   - FedAvg theo số giao dịch tăng mạnh: 24% → 57% gộp chung. Ở đây khác biệt giữa ngân hàng có tác động lớn.
   - **Nhưng với trung bình đều, đúng cách DP mức client bắt buộc, chỉ tăng 21% → 24% (FedProx: 30% → 32%).**
     Cấu hình FL + DP chính của nhóm đang bị giới hạn bởi **trung bình đều trên 802 ngân hàng rất lệch về kích cỡ**,
     không phải bởi việc ngân hàng khác nhau. Tiền xử lý cho ngân hàng "giống nhau hơn" sẽ không cứu được kết quả DP.
   - FL còn **chưa học xong ở vòng 30**: tăng lên 100 vòng (fold 0) thì FedAvg từ 0,0248 lên 0,0318, vòng tốt nhất là 97.

## 1. Cách 1 làm gì

`scripts/12_ibm_cach1_lo_nho.py`. Mỗi lượt: xáo trộn toàn bộ chỉ số train → đọc từng khối 100.000 dòng từ ổ cứng
(memmap) → trong khối xáo lại, chia lô → mỗi lô một bước Adam. Hết lượt chấm PR-AUC trên val; 3 lượt liền không tăng thì
dừng, giữ mô hình tốt nhất. Tối đa 20 lượt.

Giữ nguyên của Đạt (để so được): 802 ngân hàng ≥ 1.000 giao dịch, cách chia 5 fold và 20% val (đã đối chiếu: số dòng
fit/val/test cả 5 fold khớp log của Đạt), chuẩn hóa theo ngân hàng, 22 đặc trưng view bank, MLP, hàm đánh giá.
Khác: train dùng 100% ca bình thường (Đạt lấy 10%), dừng theo lượt (Đạt chấm 5 epoch/lần, dừng sau 30 epoch).

Mỗi fold: train 2.916.621 dòng (2.574 ca), val 729.156, test 911.445. Một lượt = 30 khối, 5.717 bước (lô 512),
khoảng 2–3 giây. Mỗi fold khoảng 50 giây.

## 2. Chọn cấu hình (fold 0, chỉ dựa vào val)

Lưới: lô {256, 512, 1024, 2048, 4096} × pos_weight {1, 10} × lr {0,001; 0,005}. Tốt nhất theo val: **lô 512,
pos_weight 1, lr 0,001** (val 0,1236). Ba cấu hình tiếp theo sát nút (0,1225; 0,1186; 0,1136). lr 0,005 kém hơn ở hầu
hết cỡ lô. Bảng đủ 20 cấu hình: `results/ibm_cach1_chon_tong_hop/bang.csv`.

Hạn chế: chọn trên fold 0, và fold 0 cũng nằm trong 5 fold báo cáo (Đạt chọn cùng kiểu).

## 3. Kết quả chính (802 ngân hàng)

| Cách | PR-AUC | Recall@1% | TB theo ngân hàng |
|---|---|---|---|
| Gộp chung lấy mẫu 10% (Đạt, `CHINH_5fold_moc`) | 0,0979 ± 0,0089 | 0,498 | 0,347 |
| Vòng học Cách 1 nhưng lấy mẫu 10% | 0,1023 ± 0,0067 | 0,508 | 0,353 |
| **Cách 1, đủ dữ liệu, seed 42** | **0,1125 ± 0,0068** | 0,514 | 0,377 |
| Cách 1, seed 7 | 0,1150 ± 0,0151 | 0,519 | 0,375 |
| Cách 1, seed 2027 | 0,1080 ± 0,0074 | 0,517 | 0,369 |
| **Cách 1, 3 seed × 5 fold** | **0,1118 ± 0,0102** | 0,517 | 0,374 |
| Cách 1 nhưng **không xáo trộn** (đọc theo thời gian) | 0,0189 ± 0,0044 | 0,233 | 0,193 |

- Cách 1 hơn Đạt ở 5/5 fold (+0,0146; 104–151%); hơn lấy mẫu 10% cùng vòng học ở 5/5 fold (+0,0102; 103–120%).
  Wilcoxon p = 0,0625 cho cả hai, đây là mức nhỏ nhất có thể với 5 cặp, nên chưa thể đạt p < 0,05 chỉ với 5 fold.
- Tách phần tăng: đổi vòng học +0,0044; dùng đủ dữ liệu thay vì 10% +0,0102.
- Điểm tốt nhất trung bình ở lượt 13; 1/15 lần chạm trần 20 lượt (seed 42, fold 2), có thể còn tăng nhẹ.
- VDR@1% dao động rất mạnh giữa các fold (0,02–0,98), giống nhận xét của Đạt; chỉ dùng làm chỉ số phụ.

## 4. Bằng chứng chọn cỡ khối (yêu cầu của anh Hùng)

Fold 0, 1 lượt, mỗi cỡ chạy trong tiến trình riêng, đo 2 lần. "RAM khối" là riêng phần dữ liệu của một khối.

| Khối (dòng) | Giây/lượt (TB 2 lần) | RAM đỉnh tiến trình (MB) | RAM khối (MB) |
|---|---|---|---|
| 10.000 | 2,59 | 814 / 828 | 0,8 |
| 25.000 | 2,22 | 871 / 779 | 2,1 |
| 50.000 | 2,20 | 878 / 799 | 4,2 |
| 90.000 | 2,34 | 891 / 769 | 7,6 |
| **100.000** | **2,04** | **933 / 916** | **8,4** |
| 110.000 | 2,06 | 950 / 925 | 9,2 |
| 200.000 | 2,22 | 990 / 900 | 16,8 |
| 500.000 | 2,02 | 1.298 / 1.189 | 42,0 |
| 1.000.000 | 2,17 | 1.159 / 1.191 | 83,9 |
| 3.000.000 (cả train) | 2,16 | 998 / 1.132 | 251,8 |

Đọc: thời gian gần như không đổi (chênh nằm trong dao động giữa 2 lần đo). RAM đỉnh chủ yếu là Python + PyTorch
(~0,75 GB) và phần file macOS giữ trong bộ đệm; dữ liệu khối chỉ vài MB. Bước tốn RAM nhất là **tính đặc trưng**
(2,4 GB, 49 giây) và **chuẩn bị fold** (2,1 GB với 802 ngân hàng, 2,6–2,8 GB với đủ ngân hàng), không phải bước học.
Kết luận: với MLP trên HI-Small, chọn 100.000 dòng hoặc lớn hơn đều được. Cỡ **lô** (số dòng mỗi bước sửa) mới ảnh hưởng
kết quả (mục 2). Cần đo lại nếu chuyển sang GNN hoặc HI-Large.

## 5. Đủ 30.470 ngân hàng

`--min-tx 0`. Ngày 0–9: 5.077.237 giao dịch, 4.522 ca; train mỗi fold 3.249.431 dòng. Ngân hàng có < 30 dòng train
(28.700 ngân hàng, phần lớn có dưới 5 giao dịch) dùng chuẩn hóa chung thay vì tự chuẩn hóa. Cùng cấu hình mục 2.

| | PR-AUC | Ngân hàng lớn (≥ 1.000 gd) | Ngân hàng nhỏ | TB theo ngân hàng |
|---|---|---|---|---|
| seed 42 | 0,1086 ± 0,0016 | 0,1110 | 0,1028 | 0,411 |
| seed 7 | 0,1040 ± 0,0126 | 0,1057 | 0,1025 | 0,405 |
| seed 2027 | 0,1144 ± 0,0122 | 0,1132 | 0,1320 | 0,412 |
| **3 seed × 5 fold** | **0,1090 ± 0,0104** | **0,110** | **0,112** | 0,410 |

- Phần ngân hàng nhỏ trong test có khoảng 100 ca mỗi fold; mô hình chấm chúng tốt ngang ngân hàng lớn.
- So với bản 802 ngân hàng (0,1118): gần bằng, nhưng không so thẳng từng fold vì thêm dữ liệu thì cách chia fold đổi.
- Đây là **gộp chung**. Học liên kết với 30.470 client là bài toán khác, chưa làm.

## 6. Kiểm tra giả thuyết "chỉ số FL thấp vì ngân hàng khác nhau quá"

**Cách làm** (`scripts/14_ibm_ngan_hang_ngau_nhien.py`): trong ngày 0–9, xáo trộn cột ngân hàng gửi giữa các giao dịch.
Mỗi ngân hàng giữ đúng số giao dịch như thật (vẫn 802 ngân hàng ≥ 1.000 giao dịch, cùng kích cỡ), nhưng giao dịch bên
trong là mẫu ngẫu nhiên nên mọi ngân hàng giống nhau. Sau đó chạy **đúng lệnh chính của Đạt** (BAO_CAO_BAN_GIAO mục 6.3:
200 ngân hàng/vòng, r = 0,1, 30 vòng, lô 512, lr 5e-3, 5 epoch, pos_weight 1, 5 fold, seed 42), chỉ đổi `--stem`.

Lưu ý: ngay cả khi ngân hàng giống nhau, mỗi ngân hàng vẫn có trung vị 3 ca rửa tiền và 113 ngân hàng không có ca nào
(bản thật: trung vị 2, 94 ngân hàng). Tức là "mỗi ngân hàng rất ít ca" là do rửa tiền hiếm (0,09%) và phần lớn ngân hàng
nhỏ, không phải do ngân hàng khác nhau.

**Kết quả** (% so với gộp chung của chính bộ đó):

| Cách FL | Ngân hàng thật | Ngân hàng giống nhau | Ngân hàng giống nhau tốt hơn |
|---|---|---|---|
| Gộp chung (mốc) | 0,0979 (100%) | 0,1025 (100%) | |
| FedAvg trung bình theo số giao dịch | 0,0236 (24% ± 4) | 0,0582 (**57%** ± 5) | 5/5 fold |
| Trung bình đều, không FedProx | 0,0203 (21% ± 3) | 0,0241 (**24%** ± 3) | 5/5 fold |
| Trung bình đều + FedProx μ = 0,1 (mốc của DP) | 0,0291 (30% ± 3) | 0,0329 (**32%** ± 7) | 4/5 fold |

**Thêm vòng** (ngân hàng thật, FedAvg, fold 0): 30 vòng → test 0,0248; **100 vòng → 0,0318** (+28%), val vẫn tăng tới
vòng 97. Trong lần chạy chính của Đạt, vòng tốt nhất của FedProx là vòng 30 ở cả 5 fold, của DP ε = 1 là 27–30.

**Đọc kết quả:**
1. Với FedAvg trung bình theo số giao dịch, khác biệt giữa ngân hàng giải thích khoảng 40% khoảng hụt
   ((57 − 24) / (100 − 24)). Phần còn lại (43 điểm %) vẫn còn khi ngân hàng giống hệt nhau.
2. **Với trung bình đều, ngân hàng giống nhau gần như không giúp (+3 điểm %).** DP mức client chuẩn bắt buộc trung bình
   đều, nên cho FL + DP (kết quả chính 25%), nút thắt là trung bình đều trên các ngân hàng rất lệch kích cỡ: ngân hàng
   1.000 giao dịch có cùng trọng số với ngân hàng 070 (449.859 giao dịch), trong khi bản cập nhật của ngân hàng nhỏ
   gần như chỉ học từ 0–2 ca.
3. FL còn chưa học xong ở vòng 30.

**Trả lời từng ý trong tin nhắn của Đạt:**

| Ý | Đánh giá |
|---|---|
| "Không bị sập nữa" | **Đúng.** DP ε = 1 với 802 ngân hàng đạt 0,0246, gấp ~27 lần ngẫu nhiên |
| "Dữ liệu ngân hàng khác nhau quá nên chỉ số thấp" | **Đúng một phần, và không đúng cho cấu hình DP.** Khác biệt giải thích ~40% khoảng hụt của FedAvg thường; với trung bình đều (DP) chỉ ~3 điểm %. Nhật ký giai đoạn 17 cũng chỉ ghi "nhiều khả năng" |
| "Làm sao để ngân hàng không khác nhau thì FL mới hiệu quả" | **Không đủ.** Kể cả khi giống hệt nhau, FedAvg chỉ đạt 57% và FL + trung bình đều chỉ 24–32% |
| "Chắc phải tiền xử lý một tí" | **Chưa có cơ sở.** Ưu tiên trước: trọng số bị chặn tương thích DP (McMahan et al., 2018, Đạt đã đề xuất ở giai đoạn 19), tăng số vòng, gộp ngân hàng nhỏ thành client lớn hơn (đánh đổi với số client cần cho DP) |

**Hạn chế:** 1 seed (42) và 1 cách xáo cho phép thử; FedProx μ, lr, số epoch được chọn trên ngân hàng thật nên có thể
chưa tối ưu cho ngân hàng giống nhau; thử 100 vòng chỉ có fold 0, không DP.

## 7. Việc tiếp theo đề xuất

1. Chạy FL + DP (FedProx, C = 0,35, ε = 1) với 100 vòng, 5 fold, xem chi phí FL còn bao nhiêu khi học đủ.
2. Thử trọng số bị chặn tương thích DP thay cho trung bình đều.
3. Thêm 2 seed cho phép thử ngân hàng giống nhau để có thống kê chắc hơn.
4. Đưa bảng mục 4 (cỡ khối) và mục 3 (không xáo trộn) vào báo cáo thứ 5.

## Cách chạy lại

```bash
cd dat-ai && source .venv/bin/activate
python scripts/06_ibm_features.py --input <...>/HI-Small_Trans.csv --view bank          # 49 giây, 2,4 GB RAM
python scripts/12_ibm_cach1_lo_nho.py --folds 0 1 2 3 4 --batch-size 512 --pos-weight 1 --lr 0.001 --tag CHINH_5fold
python scripts/12_ibm_cach1_lo_nho.py --folds 0 1 2 3 4 --batch-size 512 --pos-weight 1 --lr 0.001 --min-tx 0 --tag DU_NH_5fold_seed42
python scripts/12_ibm_cach1_lo_nho.py --folds 0 --batch-size 512 --pos-weight 1 --lr 0.001 \
       --quet-khoi 10000 25000 50000 90000 100000 110000 200000 500000 1000000 3000000 --tag do_khoi
python scripts/14_ibm_ngan_hang_ngau_nhien.py
B="--min-tx 1000 --clients-per-round 200 --rate 0.1 --rounds 30 --batch-size 512 --lr 5e-3 --local-epochs 5 --pos-weight 1 --folds 0 1 2 3 4"
python scripts/09_ibm_fl_dp.py $B --stem HI-Small-IID --settings central fedavg --tag IID_5fold_moc
python scripts/09_ibm_fl_dp.py $B --stem HI-Small-IID --settings fedavg_deu --prox-mu 0.1 --tag IID_5fold_fedprox
python scripts/09_ibm_fl_dp.py $B --stem HI-Small-IID --settings fedavg_deu --tag IID_5fold_deu
python scripts/09_ibm_fl_dp.py $B --settings fedavg_deu --tag THAT_5fold_deu
python scripts/09_ibm_fl_dp.py ${B/--rounds 30/--rounds 100} --folds 0 --settings fedavg --tag THAT_100vong_f0
```
Lệnh dùng `$B` phải chạy bằng bash (zsh không tách chuỗi `$B` thành nhiều tham số).
