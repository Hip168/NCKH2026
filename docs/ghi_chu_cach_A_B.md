# Ghi chú: cách A và cách B cho DP mức client

Ngày 02/11/2026. Mục đích: chạy lại lần lượt từng cách để xem DP mức client với 5 ngân hàng có "cứu" được không. Nếu
cả hai đều sụp → **bắt buộc tăng số ngân hàng** → nhóm họp lập kế hoạch mới.

Đọc kèm: [luu_y_dp_muc_client.md](luu_y_dp_muc_client.md), nhật ký giai đoạn 4 trong
[nhat_ky_ket_qua.md](nhat_ky_ket_qua.md).

## 1. Hai cách là gì

Cả hai đều có **MLP ở cuối**. Khác nhau ở phần xử lý đồ thị:

```
Cách A (GraphSAGE):  [2 lớp GraphSAGE — ĐƯỢC HỌC] → [MLP — được học]
Cách B (MLP):        [tổng hợp láng giềng — TÍNH SẴN, cố định, hoặc không dùng] → [MLP — được học]
```

| | Cách A | Cách B |
|---|---|---|
| Phần đồ thị | Mô hình tự học cách lấy thông tin láng giềng | Nhóm quyết định trước công thức (vd trung bình đặc trưng láng giềng) và lưu thành cột; trên synthetic hiện **chưa có** cột láng giềng → chỉ dùng 16 đặc trưng dạng bảng |
| Mô hình trong code | `SageMLP` (gốc 128 → 64, khoảng 23 nghìn tham số; nhỏ `--small` 32 → 16, khoảng 3,2 nghìn) | `TabMLP` 16 → 64 → 32 → 16 → 1, khoảng 3,7 nghìn tham số |
| Trong bài báo gọi là | GNN (GraphSAGE) | GNN không học tham số đồ thị (precomputed aggregation, kiểu SGC / SIGN) |
| Ưu | Mạnh, linh hoạt | Nhẹ, chạy nhanh (vài giây mỗi lần), dễ chạy trên IBM 5 triệu giao dịch |
| Nhược | Nặng; trên IBM cần lấy mẫu láng giềng (`pyg-lib`, chỉ có qua WSL/Docker) | Kém linh hoạt hơn A |

## 2. Đã biết gì (đến 02/11)

- **Mọi lần DP mức client bị sụp đều dùng cách A** (GraphSAGE): E4 ban đầu (PR-AUC 0,028), thử nhanh cấu hình A, B, C
  (sụp cả ở ε = 8). Lưu ý: chữ A/B/C của lần thử nhanh là **tên cấu hình** (nhiễu ở đâu, mô hình to/nhỏ), KHÔNG phải
  cách A/B ở tài liệu này.
- **GraphSAGE nhỏ (khoảng 3,2 nghìn tham số, cỡ bằng MLP) vẫn sụp** → mô hình nhỏ đi không đủ.
- **Dự đoán:** cách B cũng sụp với 5 ngân hàng, vì nguyên nhân chính là **ít ngân hàng**: nhiễu cộng vào từng tham số
  mỗi vòng (cỡ 0,4 ở ε = 8) lớn hơn nhiều so với thay đổi thật của mỗi tham số (cỡ 0,02). **Chưa có số liệu** cho cách
  B — đây là việc cần chạy.
- DP **mức bản ghi** (MLP, PR-AUC 0,657 ở ε = 1) không sụp vì là mức DP khác, nhiễu nhỏ hơn nhiều — không phải vì MLP.
- Lần chạy kiểm tra lệnh (2 vòng, thư mục tạm, KHÔNG phải kết quả): cách B, nhiễu ở server, ε = 8 → PR-AUC 0,115;
  không DP → 0,582. Cần chạy đủ 10 vòng như dưới đây mới so được với cách A.

## 3. Chạy lại lần lượt

Cùng điều kiện với lần thử nhanh: synthetic, fold 0, 10 vòng, 5 ngân hàng, ε ∈ {∞, 8, 4, 1}. Mỗi lệnh tự lưu trữ vào
`results/_luu_tru/` và ghi `results/NHAT_KY_CHAY.csv`.

```bash
cd project

# Cách A — đã có kết quả (lưu ở results/_luu_tru/2026-11-02_2225_dp_client_nhanh_*). Code của cách A KHÔNG đổi từ
# lúc đó và seed cố định → chạy lại lệnh cũ ra ĐÚNG số cũ (chỉ để kiểm tra tái lập, không có thông tin mới):
python scripts/30_dp_fl.py --mode client --folds 0 --rounds 10 --noise server --small --eps inf 8 4 1 --tag A_server_nho    # ~3 phút
python scripts/30_dp_fl.py --mode client --folds 0 --rounds 10 --noise server --eps inf 8 4 1 --tag A_server_goc            # ~10 phút

# Cách B — CHƯA chạy (vài chục giây mỗi lệnh):
python scripts/30_dp_fl.py --mode client --folds 0 --rounds 10 --noise server --model mlp --eps inf 8 4 1 --tag B_server_mlp
python scripts/30_dp_fl.py --mode client --folds 0 --rounds 10 --noise client --model mlp --eps inf 8 4 1 --tag B_client_mlp
# Thêm: nhiều vòng hơn cho cách B (MLP nhanh nên chạy được)
python scripts/30_dp_fl.py --mode client --folds 0 --rounds 50 --noise server --model mlp --eps inf 8 4 1 --tag B_server_mlp_50v

# Nếu cách B cũng sụp — cách A với ε lớn (CHƯA chạy) để biết từ ε nào mô hình bắt đầu học được:
python scripts/30_dp_fl.py --mode client --folds 0 --rounds 10 --noise server --small --eps 32 16 --tag A_server_nho_eps_lon
```

Các hướng khác cho cách A chưa thử (cần sửa code): ngưỡng cắt C lớn hơn (5, 10); learning rate hoặc momentum ở server
(McMahan et al. 2018); nhiều seed để xem 0,069 ở ε = 8 có ổn định không.

Xem kết quả: dòng `ε=...` trong `results/dp_client_<tag>/log.txt`.

## 4. Đọc kết quả thế nào

Mức đoán ngẫu nhiên trên synthetic: PR-AUC khoảng 0,015–0,02. Mốc: FedAvg không DP 50 vòng GraphSAGE đạt 0,715.

| Nếu ở ε = 1 (hoặc ε ≤ 2) | Kết luận |
|---|---|
| PR-AUC ≥ khoảng 0,6 (≥ 85% so với gộp chung 0,733) | Cứu được với 5 ngân hàng → giữ thiết kế, chạy đủ 5 fold |
| 0,1 – 0,6 | Cứu được một phần → cân nhắc nới ε (cần chốt tiêu chí Slide 10 trước) |
| ≈ 0,02 – 0,1 (như cách A) | **Không cứu được** → bắt buộc tăng số ngân hàng → họp lập kế hoạch mới |

## 5. Nếu phải tăng số ngân hàng (chuẩn bị cho buổi họp)

- **Synthetic:** sinh thêm cấu hình 20 / 50 / 100 ngân hàng nhỏ (`config.py`), quét K. Cần thêm `--profile` cho
  `01_generate_synthetic.py`.
- **IBM HI-Small:** có 30.528 ngân hàng, nhưng ngày 0–9 chỉ 34 ngân hàng có ≥ 20 ca rửa tiền; top 20 ngân hàng có ngân
  hàng chỉ 2 ca. → client = top K ngân hàng + 1 client gộp; nhiều client gần như không có ca xấu.
- **IBM HI-Medium** (khoảng 32 triệu giao dịch, file có sẵn trong Downloads) để mỗi ngân hàng nhiều ca hơn; chưa kiểm
  tra lỗi đuôi dữ liệu như HI-Small.
- Khi có nhiều ngân hàng, chọn ngẫu nhiên một phần mỗi vòng để được khuếch đại nhờ lấy mẫu con (sửa `sample_rate`
  trong cách tính ε).
