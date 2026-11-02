#!/bin/sh
# Tắt toàn bộ hệ thống, giữ lại image và dữ liệu.
cd "$(dirname "$0")/infra"
docker compose --profile cham-diem down --remove-orphans
