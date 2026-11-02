"""Phía điều phối. Chạy FedAvg qua mạng, ghi độ trễ và dung lượng truyền từng vòng, lưu mô hình tốt nhất."""

# =====================================================================================
# FILE NÀY CHẠY Ở ĐÂU: trong container superlink (người điều phối).
# VIỆC CỦA NÓ: lo VÒNG LẶP NGOÀI CÙNG của học liên kết.
#
#   Lặp 30 vòng:
#     1. Gửi mô hình chung cho 3 ngân hàng
#     2. Nhận 3 bộ số đã sửa về
#     3. Gộp thành 1 mô hình mới
#     4. Nhờ các ngân hàng chấm thử trên phần val
#   Hết 30 vòng: lưu lại mô hình của vòng có điểm val cao nhất.
#
# Vòng lặp này nằm trong thư viện Flower (strategy.start). File này chỉ cấu hình và ghi số liệu.
# File này KHÔNG đọc dữ liệu giao dịch nào: container superlink không được gắn thư mục ngân hàng.
# =====================================================================================

import json
import time
from pathlib import Path

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fraudfl.data.features import FEATURES
from fraudfl.fl.task import new_model

app = ServerApp()                           # báo cho Flower biết đây là chương trình phía điều phối


def _model_bytes(arrays: ArrayRecord) -> int:
    # Tính dung lượng của một bộ số: mô hình này khoảng 16 KB
    return sum(len(a.data) for a in arrays.values())


class FedAvgDo(FedAvg):
    """FedAvg có đo thời gian huấn luyện của ngân hàng và đếm số ngân hàng trả lời mỗi vòng."""

    # Kế thừa FedAvg của Flower, chỉ thêm phần ghi số đo. Công thức gộp vẫn là của Flower.

    def __init__(self, **kw):
        super().__init__(**kw)
        self.train_log = {}                 # sổ ghi: vòng nào có mấy ngân hàng trả lời, chậm nhất bao lâu

    def aggregate_train(self, server_round, replies):
        # Flower gọi hàm này mỗi vòng, khi đã nhận bộ số của các ngân hàng
        replies = list(replies)
        secs = [m.content["metrics"]["train-seconds"] for m in replies if m.has_content()]   # thời gian học của từng ngân hàng
        self.train_log[server_round] = {"so_ngan_hang": len(secs), "train_max_s": max(secs, default=0.0)}
        # ĐÂY LÀ BƯỚC GỘP 3 SỔ. Với từng con số, Flower tính trung bình có trọng số:
        #   số mới = (số_0 × n_0 + số_1 × n_1 + số_2 × n_2) / (n_0 + n_1 + n_2)
        # n là "num-examples" mỗi ngân hàng báo lên, tức số giao dịch đã học.
        return super().aggregate_train(server_round, replies)


@app.main()                                 # Flower gọi hàm này khi nhận lệnh flwr run
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config                # thông số của lần chạy: pyproject.toml + --run-config
    rounds, min_nodes = cfg["num-rounds"], cfg["min-nodes"]   # số vòng, số ngân hàng tối thiểu
    model = new_model(len(FEATURES))        # tạo mô hình ban đầu với 16 đặc trưng, các con số còn ngẫu nhiên
    arrays = ArrayRecord(model.state_dict())   # đổi mô hình thành dạng gửi được qua mạng
    size = _model_bytes(arrays)             # dung lượng một bộ số

    marks, snapshots = {}, {}               # marks: thời điểm xong mỗi vòng; snapshots: mô hình sau mỗi vòng

    def on_round_end(server_round: int, arr: ArrayRecord):
        # Flower gọi hàm này sau mỗi lần có mô hình gộp, chỉ dùng để ghi lại
        marks[server_round] = time.perf_counter()
        snapshots[server_round] = arr       # giữ mô hình của vòng này, cuối cùng chọn vòng tốt nhất
        return None                         # phía điều phối không tự chấm điểm vì không có dữ liệu

    # fraction_train=1.0, fraction_evaluate=1.0: mọi ngân hàng đều tham gia học và chấm ở mọi vòng
    # min_available_nodes: chưa đủ số ngân hàng thì đứng đợi, chưa bắt đầu
    strategy = FedAvgDo(fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=min_nodes,
                        min_evaluate_nodes=min_nodes, min_available_nodes=min_nodes)
    # VÒNG LẶP CHÍNH: chạy đủ num_rounds vòng, mỗi vòng gồm gửi, học, gộp, chấm.
    # Lệnh này chạy tới khi xong hết rồi mới đi tiếp xuống dưới.
    result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=rounds, evaluate_fn=on_round_end,
                            train_config=ConfigRecord({"lr": 0.5}))

    # Lấy điểm PR-AUC trên val của từng vòng, rồi chọn vòng có điểm cao nhất
    val = {r: float(m["pr-auc"]) for r, m in result.evaluate_metrics_clientapp.items()}
    best = max(val, key=val.get)

    # Gom số liệu từng vòng thành bảng để ghi ra rounds.csv
    rows = []
    for r in range(1, rounds + 1):
        info = strategy.train_log.get(r, {"so_ngan_hang": 0, "train_max_s": 0.0})
        wall = marks[r] - marks[r - 1]      # thời gian thực của vòng này
        n = info["so_ngan_hang"]
        # bytes_ca_vong = 2 × số ngân hàng × dung lượng: mỗi ngân hàng nhận 1 lần và gửi 1 lần
        rows.append({"round": r, "val_pr_auc": val.get(r), "so_ngan_hang": n, "wall_s": wall,
                     "train_max_s": info["train_max_s"], "bytes_mot_chieu": size, "bytes_ca_vong": 2 * n * size})
    out = Path(cfg["out-dir"]) / cfg["run-name"]   # thư mục kết quả, ví dụ artifacts/tn/nen
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "rounds.csv", index=False)   # điểm và thời gian từng vòng
    torch.save(snapshots[best].to_torch_state_dict(), out / "model.pt")   # SẢN PHẨM CUỐI: mô hình của vòng tốt nhất
    (out / "tom_tat.json").write_text(json.dumps({   # tóm tắt lần chạy
        "vong_tot_nhat": best, "val_pr_auc": val[best], "epsilon": cfg["epsilon"], "fold": cfg["fold"],
        "tong_mb": sum(x["bytes_ca_vong"] for x in rows) / 1e6, "tong_giay": marks[rounds] - marks[0]}, indent=2))
    print(f"Xong. Vòng tốt nhất {best}, val PR-AUC {val[best]:.4f}, lưu ở {out}")
