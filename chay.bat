@echo off
rem Chay toan bo he thong tu dau. Dung tren Windows: bam dup vao file nay.
rem Chay lai luc nao cung duoc: script tu don lan chay truoc.
cd /d "%~dp0infra"

docker info >nul 2>&1
if errorlevel 1 (
  echo LOI: Docker chua chay. Mo Docker Desktop, doi 1 phut roi chay lai.
  goto loi
)

echo [1/7] Dung lan chay truoc (neu co)
docker compose --profile cham-diem down --remove-orphans
if exist ..\artifacts\tach_khong_dp\model.pt del ..\artifacts\tach_khong_dp\model.pt

echo [2/7] Build image. Lan dau mat 5-15 phut, tai khoang 3 GB
docker compose build
if errorlevel 1 goto loi

echo [3/7] Sinh du lieu va chia cho tung ngan hang
docker compose run --rm chuan-bi
if errorlevel 1 goto loi

echo [4/7] Bat Kafka, Redis, Postgres, Flower va 3 ngan hang
docker compose up -d --wait
if errorlevel 1 goto loi

echo [5/7] Huan luyen hoc lien ket 30 vong, khoang 7-15 phut
docker compose run --rm huan-luyen
if not exist ..\artifacts\tach_khong_dp\model.pt (
  echo LOI: huan luyen khong tao ra mo hinh.
  goto loi
)

echo [6/7] Chay real-time: phat 2.555 giao dich
docker compose run --rm tao-topic
if errorlevel 1 goto loi
docker compose run --rm nap-lich-su
if errorlevel 1 goto loi
docker compose --profile cham-diem up -d cham-diem-0 cham-diem-1 cham-diem-2
if errorlevel 1 goto loi
docker compose run --rm phat-giao-dich
if errorlevel 1 goto loi
ping -n 16 127.0.0.1 >nul

echo [7/7] Ket qua
docker compose exec -T postgres psql -U nckh -d nckh -c "select count(*) as so_giao_dich from giao_dich"
docker compose exec -T postgres psql -U nckh -d nckh -c "select count(*) as so_canh_bao, sum(nhan_that) as canh_bao_dung from giao_dich join canh_bao using (tx_id, ngan_hang)"
echo.
echo THANH CONG. He thong van dang chay. Xong thi bam dup vao tat.bat
pause
exit /b 0

:loi
echo.
echo CO LOI. Chup man hinh doan phia tren gui cho Dao.
pause
exit /b 1
