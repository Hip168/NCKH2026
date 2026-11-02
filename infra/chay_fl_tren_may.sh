#!/usr/bin/env bash
# Chạy Flower qua mạng localhost, không cần Docker: 1 SuperLink + N SuperNode (mỗi ngân hàng một tiến trình).
# Dùng:  bash infra/chay_fl_tren_may.sh "bank_0 bank_1 bank_2" "num-rounds=3 run-name=\"thu\" epsilon=0.0"
#        Chế độ gộp: bash infra/chay_fl_tren_may.sh "gop" "min-nodes=1 run-name=\"gop\""
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARTS="${1:-bank_0 bank_1 bank_2}"
CFG="${2:-}"
BIN="$ROOT/.venv/bin"
export PATH="$BIN:$PATH"
export FLWR_HOME="$(mktemp -d)"
export PYTHONPATH="$ROOT/src"
LOGS="$FLWR_HOME/logs"; mkdir -p "$LOGS"
cat > "$FLWR_HOME/config.toml" <<TOML
[superlink]
default = "tren-may"
[superlink.tren-may]
address = "127.0.0.1:19091"
insecure = true
TOML

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  # SuperExec do SuperLink/SuperNode sinh ra chạy tách nhóm tiến trình, nhận diện theo cổng của lần chạy này
  pkill -f "flower-superexec.*127.0.0.1:19[01]" 2>/dev/null || true
  wait 2>/dev/null || true
  rm -rf "$FLWR_HOME"
}
trap cleanup EXIT

cd "$ROOT"
"$BIN/flower-superlink" --insecure --host 127.0.0.1 --port 19091 --database "$FLWR_HOME/state.db" >"$LOGS/superlink.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 60); do nc -z 127.0.0.1 19091 2>/dev/null && break; sleep 1; done
nc -z 127.0.0.1 19091 || { echo "SuperLink không khởi động được:"; tail -20 "$LOGS/superlink.log"; exit 1; }
port=19100
for p in $PARTS; do
  "$BIN/flower-supernode" --insecure --superlink 127.0.0.1:19091 --port "$port" \
      --node-config "partition=\"$p\"" >"$LOGS/$p.log" 2>&1 &
  PIDS+=($!)
  port=$((port + 1))
done
sleep 8
OUT=$("$BIN/flwr" run fl_app tren-may --format json --run-config "data-root=\"$ROOT/data/banks\" out-dir=\"$ROOT/artifacts\" $CFG")
echo "$OUT"
RUN=$(echo "$OUT" | "$BIN/python" -c "import json,sys; print(json.load(sys.stdin).get('run-id',''))" 2>/dev/null)
[ -n "$RUN" ] || { echo "Không tạo được run"; tail -5 "$LOGS"/*.log; exit 1; }
"$BIN/flwr" log "$RUN" tren-may --stream 2>&1 | grep -E "Xong|ERROR|Error|Traceback|rror:" | tail -30 || true
