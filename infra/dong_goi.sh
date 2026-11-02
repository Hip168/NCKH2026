#!/bin/sh
# Đóng gói code thành 1 file zip để gửi cho thành viên nhóm. Chạy lại sau mỗi lần sửa code.
# Dùng:  sh infra/dong_goi.sh [đường_dẫn_file_zip]
set -e
cd "$(dirname "$0")/.."
OUT="${1:-$PWD/../goi_gui_em/NCKH_docker.zip}"
mkdir -p "$(dirname "$OUT")"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
rm -f "$OUT"
git ls-files -co --exclude-standard | grep -v -E '^results/' | while read -r f; do [ -f "$f" ] && echo "$f"; done \
  | zip -q -X "$OUT" -@
echo "Da tao $OUT ($(du -h "$OUT" | cut -f1))"
