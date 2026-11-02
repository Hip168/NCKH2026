#!/bin/sh
# Chạy toàn bộ hệ thống từ đầu. Dùng trên Mac/Linux:  sh chay.sh
# Chạy lại lúc nào cũng được: script tự dọn lần chạy trước.
set -e
cd "$(dirname "$0")/infra"

docker info >/dev/null 2>&1 || { echo "LOI: Docker chua chay. Mo Docker Desktop, doi 1 phut roi chay lai."; exit 1; }

echo "[1/7] Dung lan chay truoc (neu co)"
docker compose --profile cham-diem down --remove-orphans
rm -f ../artifacts/tach_khong_dp/model.pt

echo "[2/7] Build image. Lan dau mat 5-15 phut, tai khoang 3 GB"
docker compose build

echo "[3/7] Sinh du lieu va chia cho tung ngan hang"
docker compose run --rm chuan-bi

echo "[4/7] Bat Kafka, Redis, Postgres, Flower va 3 ngan hang"
docker compose up -d --wait

echo "[5/7] Huan luyen hoc lien ket 30 vong, khoang 7-15 phut"
docker compose run --rm huan-luyen
[ -f ../artifacts/tach_khong_dp/model.pt ] || { echo "LOI: huan luyen khong tao ra mo hinh."; exit 1; }

echo "[6/7] Chay real-time: phat 2.555 giao dich"
docker compose run --rm tao-topic
docker compose run --rm nap-lich-su
docker compose --profile cham-diem up -d cham-diem-0 cham-diem-1 cham-diem-2
docker compose run --rm phat-giao-dich
sleep 15

echo "[7/7] Ket qua"
docker compose exec -T postgres psql -U nckh -d nckh -c "select count(*) as so_giao_dich from giao_dich"
docker compose exec -T postgres psql -U nckh -d nckh -c "select count(*) as so_canh_bao, sum(nhan_that) as canh_bao_dung from giao_dich join canh_bao using (tx_id, ngan_hang)"
echo
echo "THANH CONG. He thong van dang chay. Xong thi chay: sh tat.sh"
