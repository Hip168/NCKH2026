#!/bin/sh
# Đổi từng thông số một, các thông số còn lại giữ nguyên, để xem ảnh hưởng của từng cái.
# Chạy:  sh infra/quet_thong_so.sh      (cần: docker compose up -d --wait superlink bank-0 bank-1 bank-2)
# Kết quả: artifacts/tn/<tên>/, tổng hợp bằng scripts/53_tong_hop_quet.py
# Chạy lại được: lần nào đã có tom_tat.json thì bỏ qua.
cd "$(dirname "$0")" || exit 1

chay() {  # chay <tên> <thông số đổi so với mức nền>
    ten=$1; shift
    if [ -f "../artifacts/tn/$ten/tom_tat.json" ]; then echo "[bo qua] $ten"; return; fi
    echo "[$(date +%H:%M:%S)] bat dau $ten: $*"
    if docker compose run --rm -T huan-luyen flwr run fl_app docker --stream --run-config \
        "data-root=\"/app/data/banks\" out-dir=\"/app/artifacts\" run-name=\"tn/$ten\" $*" > "/tmp/tn_$ten.log" 2>&1 \
        && [ -f "../artifacts/tn/$ten/tom_tat.json" ]; then
        echo "[$(date +%H:%M:%S)] xong $ten"
    else
        echo "[$(date +%H:%M:%S)] LOI $ten, xem /tmp/tn_$ten.log"
    fi
}

# Mức nền: không DP, 30 vòng, lr 0.5, batch 1024, pos-weight 10
chay nen ""

# Riêng tư: ε càng nhỏ càng nhiều nhiễu
chay eps_0_5 "epsilon=0.5"
chay eps_1 "epsilon=1.0"
chay eps_2 "epsilon=2.0"
chay eps_5 "epsilon=5.0"

# Số vòng khi bật DP: nhiều vòng thì nhiễu mỗi vòng phải lớn hơn để giữ nguyên ε
chay eps_1_vong_10 "epsilon=1.0 num-rounds=10"
chay eps_1_vong_60 "epsilon=1.0 num-rounds=60"

# Trọng số phạt khi bỏ sót gian lận
chay pos_1 "pos-weight=1.0"
chay pos_50 "pos-weight=50.0"

# Tốc độ học
chay lr_0_05 "lr=0.05"
chay lr_2 "lr=2.0"

# Cỡ batch
chay batch_256 "batch=256"
chay batch_4096 "batch=4096"

# Ngưỡng cắt, chỉ có tác dụng khi bật DP nên so với eps_1
chay eps_1_clip_0_2 "epsilon=1.0 clip=0.2"
chay eps_1_clip_5 "epsilon=1.0 clip=5.0"

echo "HET"
