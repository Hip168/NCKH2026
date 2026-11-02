"""Logic phía ngân hàng cho Flower, tách khỏi Flower để test và dùng lại được.

Dùng lại nguyên DP-SGD của `fraudfl.privacy`. Mỗi ngân hàng tự giữ ngân sách ε và tự cộng nhiễu trên máy mình.
PR-AUC không lấy trung bình cộng đúng nghĩa được, nhưng dữ liệu không rời ngân hàng nên server chỉ nhận trung
bình có trọng số của PR-AUC từng ngân hàng. Đây là khác biệt so với `train.fedavg` (gộp điểm val về một chỗ).
"""

# =====================================================================================
# FILE NÀY LÀ GÌ: phần việc của một ngân hàng trong một vòng, không dính tới Flower.
#
#   new_model()      : tạo khung mô hình
#   train_plan()     : tính kế hoạch học của vòng này: bốc bao nhiêu, học mấy bước, nhiễu bao nhiêu
#   local_train()    : học thật, phần sửa số nằm trong src/fraudfl/privacy.py
#   evaluate_split() : chấm điểm mô hình trên một phần dữ liệu
# =====================================================================================

import math

import torch
from sklearn.metrics import average_precision_score

from fraudfl.config import SEED
from fraudfl.fl.partition import Partition
from fraudfl.models import TabMLP
from fraudfl.privacy import CLIP, dpsgd_steps, noise_multiplier

# Giá trị mặc định. Đổi được cho từng lần chạy qua fl_app/pyproject.toml hoặc --run-config.
LR, MOMENTUM = 0.5, 0.9             # đúng scripts/30_dp_fl.py (D13)
                                    # LR: mỗi lần sửa mạnh bao nhiêu; MOMENTUM: giữ lại bao nhiêu đà sửa của các bước trước
EXPECTED_BATCH = 1024               # trung bình mỗi bước xem 1.024 giao dịch
POS_WEIGHT = 10.0                   # chấm sai một ca gian lận bị phạt nặng gấp 10 lần ca thường


def new_model(in_dim: int) -> TabMLP:
    torch.manual_seed(SEED)         # cố định ngẫu nhiên: lần nào tạo cũng ra đúng bộ số ban đầu
    return TabMLP(in_dim)           # in_dim = số đặc trưng đầu vào = 16


def train_plan(n_fit: int, epsilon: float, rounds: int, batch: int = EXPECTED_BATCH) -> tuple[float, int, float]:
    """(tỷ lệ lấy mẫu q, số bước mỗi vòng, hệ số nhiễu z). epsilon <= 0 hoặc inf nghĩa là không DP."""
    q = min(1.0, batch / n_fit)     # q: xác suất mỗi giao dịch được bốc. Ví dụ 1024 / 40.000 ≈ 2,6%
    steps = max(1, round(1 / q))    # số bước mỗi vòng ≈ 39, tức đi qua dữ liệu khoảng 1 lần
    use_dp = epsilon > 0 and not math.isinf(epsilon)   # có bật DP không
    # z: nhiễu cộng vào mạnh bao nhiêu. Tính theo TỔNG số bước của cả lần chạy (steps × rounds),
    # vì ε là ngân sách cho toàn bộ quá trình. Học càng nhiều vòng thì z phải càng lớn.
    z = noise_multiplier(epsilon, sample_rate=q, steps=steps * rounds) if use_dp else 0.0
    return q, steps, z


def local_train(model: TabMLP, p: Partition, epsilon: float, rounds: int, seed: int, lr: float = LR,
                batch: int = EXPECTED_BATCH, pos_weight: float = POS_WEIGHT, clip: float = CLIP) -> dict:
    # Một ngân hàng học trong một vòng. Hàm này sửa trực tiếp các con số trong model.
    q, steps, z = train_plan(p.n_fit, epsilon, rounds, batch)   # lập kế hoạch
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)   # bộ sửa số; tạo mới mỗi vòng nên đà sửa bắt đầu lại từ 0
    gen = torch.Generator().manual_seed(seed)   # bộ sinh ngẫu nhiên riêng của ngân hàng và vòng này
    # VÀO VÒNG LẶP HỌC. p.x[p.fit], p.y[p.fit]: chỉ lấy phần fit để học, không đụng val và test.
    # Không bật DP thì z = 0 và clip = vô cực, tức không thêm nhiễu và không cắt gradient.
    dpsgd_steps(model, p.x[p.fit], p.y[p.fit], opt, steps, q, clip=clip if z > 0 else float("inf"), z=z,
                pos_weight=pos_weight, gen=gen)
    return {"noise-multiplier": z, "dp-steps": steps}   # báo lại z và số bước đã chạy


@torch.no_grad()                    # chỉ chấm điểm, không học nên không cần tính hướng sửa
def evaluate_split(model: TabMLP, p: Partition, split: str = "val") -> dict:
    mask = getattr(p, split)        # lấy cột đánh dấu giao dịch thuộc phần "val" hoặc "test"
    model.eval()                    # chuyển sang chế độ chấm điểm
    score = torch.sigmoid(model(p.x[mask])).numpy()   # chấm từng giao dịch; sigmoid đưa kết quả về khoảng 0 đến 1
    y = p.y[mask].numpy()           # đáp án thật
    # PR-AUC: càng gần 1 thì mô hình càng xếp được giao dịch gian lận lên trên giao dịch thường
    return {"pr-auc": float(average_precision_score(y, score)), "num-examples": int(mask.sum())}
