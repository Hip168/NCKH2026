# Hạ tầng FL + real-time

Cách chạy: [huong_dan_docker.md](huong_dan_docker.md). Số liệu dưới đây đo ngày 2026-11-01 trên MacBook 10 nhân, 16 GB,
fold 0, dữ liệu synthetic, mô hình `TabMLP`.

## 1. Thành phần

| Yêu cầu | Làm bằng | File |
|---|---|---|
| Ngân hàng mô phỏng | Flower 1.42, mỗi ngân hàng một container, chỉ mount dữ liệu của mình | `fl_app/`, `src/fraudfl/fl/` |
| Tách / gộp | Cùng code, đổi tên partition: `bank_i` hoặc `gop` | `scripts/50_chia_du_lieu_ngan_hang.py` |
| Real-time / theo lô | Cùng hàm `consumer.process`, khác kích thước lô | `src/fraudfl/stream/consumer.py` |
| Đặc trưng real-time | Redis sorted set, phải khớp `data/features.py` | `src/fraudfl/stream/online_features.py` |
| Kafka | Mỗi ngân hàng một partition, producer idempotent | `src/fraudfl/stream/producer.py` |
| ACID | Giao dịch và cảnh báo trong cùng transaction Postgres | `infra/init.sql` |

**Slide ghi Flower ≥ 2.0 nhưng bản đó không tồn tại.** Bản mới nhất trên PyPI là 1.42.0. API 1.42 khác nhiều hướng
dẫn cũ: một cổng HTTP duy nhất, `ServerApp` và `ClientApp`, chạy bằng `flwr run`.

## 2. Bảo đảm dữ liệu

Thứ tự trong consumer: chấm điểm → commit Postgres → commit offset Kafka.

| Sự cố | Điều xảy ra |
|---|---|
| Sập trước commit Postgres | Chưa ghi gì, đọc lại từ offset cũ |
| Sập sau commit Postgres, trước commit Kafka | Đọc lại; khóa `tx_id` ở Postgres và Redis chặn trùng |

Đây là **at-least-once cộng idempotent**, kết quả cuối như xử lý đúng một lần.

**CAP:** hệ thống hiện chọn nhất quán (CP) ở Postgres. Mỗi dịch vụ chỉ có một bản sao nên chưa có thí nghiệm phân
vùng mạng thật. Đây là việc còn lại.

## 3. Số đo

| Kiểm tra | Kết quả |
|---|---|
| Test tự động | 23/23 đạt |
| Tách 5 ngân hàng qua Flower so với vòng lặp mô phỏng | PR-AUC 0,7188 so với 0,7168 |
| Tách + DP ε = 1 qua Flower so với vòng lặp mô phỏng | PR-AUC 0,6651 so với 0,6654 |
| Gộp 1 bên qua Flower | PR-AUC 0,7180 |
| Dung lượng mỗi vòng | Mô hình 15,9 KB; 95 KB (3 ngân hàng); 159 KB (5 ngân hàng) |
| Thời gian mỗi vòng | 13,9 giây (3 ngân hàng); 16,8 giây (5 ngân hàng); huấn luyện chỉ 0,7–1 giây |
| Consumer bị `docker kill` giữa luồng | 2.555/2.555 giao dịch, không trùng, 0 cảnh báo mồ côi |
| Điểm online so với điểm theo lô | Lệch lớn nhất 8,9e-8 |
| Độ trễ real-time ở 50 giao dịch/giây | Trung vị 9,4 ms; p95 12,8 ms; sau khởi động p99 13,8 ms, lớn nhất 60,6 ms |
| Cảnh báo trong 1 ngày mô phỏng | 32 cảnh báo, 29 đúng. Mô hình lúc đó là bản 3 ngân hàng |

Thời gian mỗi vòng phần lớn **không phải huấn luyện**. Chưa tách được phần do mạng và phần do khởi động tiến trình,
nên **chưa dùng số này làm độ trễ mạng trong bài báo**.

## 4. Giới hạn đã biết

- Flower mới chạy `TabMLP`, chưa chạy GraphSAGE.
- Chọn vòng tốt nhất dùng trung bình PR-AUC của từng ngân hàng, không gộp điểm như `train.fedavg`, nên vòng được chọn
  có thể khác.
- **Consumer 3 ngân hàng dùng chung một Redis và một Postgres** để tiết kiệm RAM. Huấn luyện thì cách ly thật, phần
  real-time mới là demo. Triển khai thật mỗi ngân hàng phải có kho riêng.
- Flower chạy `--insecure`, không mã hóa đường truyền. Chỉ dùng trong mạng thử nghiệm.
- Nhãn đi kèm tin nhắn thay vì đến trễ qua kênh riêng. Đặc trưng chỉ đọc nhãn cũ hơn 7 ngày nên không rò rỉ, nhưng chưa
  mô phỏng đúng luồng nhãn trễ.
- Kafka có một broker, chưa thử broker sập.
- Dữ liệu real-time mới là synthetic, chưa chạy IBM AML.
- Mới chạy trên macOS, chưa thử Windows.
