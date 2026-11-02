"""Phía ngân hàng. Dữ liệu ở lại máy, chỉ trọng số mô hình đi qua mạng."""

# =====================================================================================
# FILE NÀY CHẠY Ở ĐÂU: trong từng container bank-0, bank-1, bank-2.
# VIỆC CỦA NÓ: trả lời 2 yêu cầu từ superlink trong mỗi vòng.
#
#   train()    : nhận mô hình chung, học trên dữ liệu của mình, gửi bộ số mới về
#   evaluate() : nhận mô hình đã gộp, chấm trên phần val của mình, gửi điểm về
#
# Dữ liệu chỉ được đọc ở đây. Trong tin gửi đi chỉ có bộ số của mô hình và vài con số thống kê.
# =====================================================================================

import time

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fraudfl.config import SEED
from fraudfl.fl.partition import fold_path, load_partition
from fraudfl.fl.task import evaluate_split, local_train, new_model

app = ClientApp()                           # báo cho Flower biết đây là chương trình phía ngân hàng


def _data(context: Context):
    # Tìm và đọc file dữ liệu của ngân hàng này.
    # node_config["partition"] là tên ngân hàng, đặt trong docker-compose.yml: partition="bank_0"
    # Kết quả là đường dẫn dạng /app/data/banks/bank_0/fold0.npz
    path = fold_path(context.run_config["data-root"], context.node_config["partition"],
                     context.run_config["fold"])
    return load_partition(path)


@app.train()                                # Flower gọi hàm này khi superlink yêu cầu học
def train(msg: Message, context: Context) -> Message:
    p = _data(context)                      # đọc dữ liệu riêng của ngân hàng này
    model = new_model(p.x.shape[1])         # tạo khung mô hình; p.x.shape[1] = 16 đặc trưng
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())   # chép bộ số vừa nhận vào khung
    server_round = msg.content["config"]["server-round"]   # đây là vòng thứ mấy
    # Hạt giống ngẫu nhiên: khác nhau theo vòng và theo ngân hàng, chạy lại vẫn ra đúng kết quả cũ
    bank_seed = SEED + 1000 * server_round + sum(map(ord, context.node_config["partition"]))
    t0 = time.perf_counter()                # bắt đầu bấm giờ
    cfg = context.run_config                # thông số của lần chạy
    # HỌC Ở ĐÂY: sửa các con số trong model. Chi tiết ở src/fraudfl/fl/task.py
    info = local_train(model, p, cfg["epsilon"], cfg["num-rounds"], bank_seed, lr=cfg["lr"], batch=cfg["batch"],
                       pos_weight=cfg["pos-weight"], clip=cfg["clip"])
    # num-examples: số giao dịch đã học, superlink dùng làm trọng số khi gộp
    metrics = {"num-examples": p.n_fit, "train-seconds": time.perf_counter() - t0, **info}
    # GỬI VỀ: bộ số mới của mô hình và các số thống kê. Không có dòng giao dịch nào trong tin này.
    return Message(RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)}),
                   reply_to=msg)


@app.evaluate()                             # Flower gọi hàm này khi superlink yêu cầu chấm thử
def evaluate(msg: Message, context: Context) -> Message:
    p = _data(context)                      # đọc dữ liệu riêng của ngân hàng này
    model = new_model(p.x.shape[1])         # tạo khung mô hình
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())   # chép mô hình đã gộp vào khung
    # Chấm trên phần val của ngân hàng, chỉ gửi điểm PR-AUC và số giao dịch đã chấm
    return Message(RecordDict({"metrics": MetricRecord(evaluate_split(model, p, "val"))}), reply_to=msg)
