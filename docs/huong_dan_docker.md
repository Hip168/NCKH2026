# Hướng dẫn Docker cho cả nhóm

Làm theo từ trên xuống. **Máy không cần cài Python**, chỉ cần Docker và Git.
Đã chạy trọn vẹn trên macOS (chip Apple). **Chưa thử trên Windows**, ai chạy Windows gặp lỗi thì báo Đạo.

## 1. Docker là gì?

Docker giúp mọi người chạy cùng một môi trường. Không còn cảnh "máy tao chạy được mà máy mày lỗi".

| Từ | Nghĩa | Ví dụ trong đề tài |
|---|---|---|
| **Image** | Hộp đóng sẵn: hệ điều hành nhỏ + Python + thư viện + code | `nckh-fraudfl` |
| **Container** | Một "máy tính nhỏ" chạy từ image. Một image chạy được nhiều container | `bank-0`, `bank-1`, `bank-2` dùng chung một image |
| **Volume** | Thư mục trên máy thật cho container dùng chung | `data/banks/bank_0` |
| **Docker Compose** | File `docker-compose.yml` liệt kê tất cả container, một lệnh bật cả hệ thống | `infra/docker-compose.yml` |

## 2. Hệ thống của mình gồm những gì?

Mỗi dòng là một container. Hệ thống mô phỏng 3 ngân hàng cùng học một mô hình mà không chia sẻ dữ liệu.

| Container | Đóng vai | Việc làm |
|---|---|---|
| `superlink` | Người điều phối | Gửi mô hình xuống ngân hàng, gom trọng số về, lấy trung bình |
| `bank-0`, `bank-1`, `bank-2` | 3 ngân hàng | Tự học trên dữ liệu của mình. **Mỗi container chỉ thấy dữ liệu của chính nó** |
| `kafka` | Băng chuyền giao dịch | Giao dịch mới chạy qua đây theo đúng thứ tự, không mất khi hệ thống gặp sự cố |
| `redis` | Sổ tay ghi nhanh | Nhớ lịch sử gần đây của khách để tính đặc trưng trong vài mili giây |
| `postgres` | Sổ cái | Lưu giao dịch đã chấm điểm và cảnh báo |
| `cham-diem-0/1/2` | Nhân viên soát | Đọc Kafka, chấm điểm bằng mô hình, ghi Postgres |

**Hai luồng chính:**

```
Huấn luyện (chạy theo lô):
  superlink ──mô hình──> bank-0, bank-1, bank-2 (tự học)
  superlink <──trọng số── (chỉ trọng số đi qua mạng, dữ liệu ở lại)

Chấm điểm (real-time):
  giao dịch → kafka → cham-diem → redis (lấy lịch sử) → mô hình → postgres (lưu, cảnh báo)
```

## 3. Cần tải những gì?

| Thứ cần có | Tải ở đâu | Dung lượng |
|---|---|---|
| **Docker Desktop** | https://www.docker.com/products/docker-desktop/ (chọn đúng Windows, hoặc Mac chip Apple/Intel) | Khoảng 1–2 GB |
| **Git** | https://git-scm.com/downloads | Khoảng 300 MB |
| **Code của nhóm** | Nhờ Đạt mời vào repo trên GitHub, repo đang private | Khoảng 20 MB |

Các thứ còn lại Docker **tự tải** ở bước 4.2: image của nhóm (2,2 GB), Kafka, Redis, Postgres (khoảng 1 GB).

**Yêu cầu máy:** còn trống ít nhất **10 GB** ổ cứng, RAM từ **8 GB**. Cả hệ thống đo được dùng khoảng 2,4 GB RAM.

**Riêng Windows:** Docker Desktop cần WSL2. Mở PowerShell bằng quyền Administrator, chạy `wsl --install`, rồi khởi động lại máy.

**Kiểm tra đã cài xong:** mở Docker Desktop, đợi biểu tượng báo đang chạy, rồi gõ:

```bash
docker --version
docker compose version
```

Thấy số phiên bản là được. Docker Desktop phải **luôn mở** khi làm các bước dưới.

## 4. Chạy lần đầu

### 4.1. Lấy code

```bash
git clone https://github.com/trantatdat06/NCKH2026.git
cd NCKH2026
git checkout dao/ha-tang-fl-kafka
cd infra
```

Từ đây trở đi mọi lệnh đều chạy trong thư mục `infra`.

### 4.2. Build image (làm 1 lần, 5–15 phút tùy mạng)

```bash
docker compose build
```

### 4.3. Sinh dữ liệu (khoảng 1 phút)

```bash
docker compose run --rm chuan-bi
```

Việc này sinh ~496 nghìn giao dịch rồi chia cho từng ngân hàng vào `data/banks/`. Dữ liệu sinh ra giống hệt nhau trên mọi máy.

### 4.4. Bật hệ thống

```bash
docker compose up -d
docker compose ps
```

Thấy `kafka`, `redis`, `postgres`, `superlink` báo `healthy` và 3 `bank` báo `Up` là được.

### 4.5. Huấn luyện học liên kết (khoảng 7 phút)

```bash
docker compose run --rm huan-luyen
```

Thành công khi thấy dòng `Xong. Vòng tốt nhất ...`. Mô hình được lưu ở `artifacts/tach_khong_dp/`. File `rounds.csv` ghi thời gian và dung lượng từng vòng.

### 4.6. Chạy real-time

```bash
docker compose run --rm tao-topic
docker compose run --rm nap-lich-su
docker compose --profile cham-diem up -d cham-diem-0 cham-diem-1 cham-diem-2
docker compose run --rm phat-giao-dich
```

Lệnh cuối phát 2.555 giao dịch, mất khoảng 1 phút.

### 4.7. Xem kết quả

```bash
docker compose exec postgres psql -U nckh -d nckh -c "select count(*) from giao_dich"
docker compose exec postgres psql -U nckh -d nckh -c "select count(*) as canh_bao, sum(nhan_that) as dung from giao_dich join canh_bao using (tx_id, ngan_hang)"
```

Kết quả tham khảo: 2.555 giao dịch, 32 cảnh báo, 29 cảnh báo đúng.

### 4.8. Tắt khi xong

```bash
docker compose --profile cham-diem down
```

Lệnh này xóa container nhưng giữ lại image và dữ liệu, nên lần sau chỉ cần làm từ bước 4.4.

## 5. Lệnh hay dùng

| Muốn làm gì | Lệnh |
|---|---|
| Xem container nào đang chạy | `docker compose ps` |
| Xem log một container | `docker compose logs -f bank-0` (Ctrl+C để thoát) |
| Xem RAM đang dùng | `docker stats` (Ctrl+C để thoát) |
| Vào bên trong container | `docker compose exec bank-0 bash` |
| Sửa code xong, build lại | `docker compose build` |
| Thử sập hệ thống | `docker kill nckh-cham-diem-0-1`, rồi bật lại bằng lệnh ở 4.6 |

## 6. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop chưa mở. Mở lên rồi đợi 1 phút |
| `port is already allocated` | Có chương trình khác đang dùng cổng đó (29092, 6379 hoặc 55432). Đổi số **bên trái** dấu `:` ở dòng `ports` trong `docker-compose.yml`, hoặc tắt chương trình kia |
| Container tự tắt, trạng thái `Exited (137)` | Thiếu RAM. Vào Docker Desktop, Settings, Resources, tăng Memory lên 4–6 GB |
| `no space left on device` | Hết ổ cứng. Chạy `docker system prune` để dọn. Đọc kỹ cảnh báo trước khi gõ `y` vì lệnh áp dụng cho mọi dự án Docker trên máy |
| `huan-luyen` đứng rất lâu ở vòng đầu | Thường do thiếu ngân hàng. Chạy `docker compose ps` xem đủ 3 `bank` chưa |
| Sửa code nhưng chạy vẫn như cũ | Code nằm trong image, phải `docker compose build` lại |
