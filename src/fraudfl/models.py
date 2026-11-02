"""Mô hình GraphSAGE + MLP theo Slide 6."""

# =====================================================================================
# FILE NÀY LÀ GÌ: hình dạng của "cuốn sổ tay".
#   TabMLP  : mô hình bản Flower đang dùng, chấm từng giao dịch dựa trên 16 đặc trưng
#   SageMLP : mô hình đồ thị của thí nghiệm E1 đến E3, bản Flower không dùng
# =====================================================================================

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class SageMLP(nn.Module):
    """GraphSAGE 2 lớp (128 → 64) → MLP 64 → 32 → 16 → 1 (logit gian lận)."""

    def __init__(self, in_dim: int, hidden: int = 128, emb: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)   # lớp đồ thị 1: lấy thông tin từ các giao dịch liên quan
        self.conv2 = SAGEConv(hidden, emb)      # lớp đồ thị 2
        self.dropout = nn.Dropout(dropout)      # khi học, tắt ngẫu nhiên 30% số nút để bớt học vẹt
        self.head = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.conv1(x, edge_index).relu())
        h = self.dropout(self.conv2(h, edge_index).relu())
        return self.head(h).squeeze(-1)


class TabMLP(nn.Module):
    """MLP dạng bảng cho DP mức bản ghi (in → 64 → 32 → 16 → 1), cùng phần đầu với SageMLP.

    Không dùng dropout: DP-SGD tính gradient từng mẫu bằng vmap, cần mô hình tất định.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        # Dữ liệu đi qua 4 lớp, từ 16 đặc trưng thu dần về 1 con số là điểm nghi ngờ:
        #   16 → 64 : 16×64 + 64 = 1.088 con số
        #   64 → 32 : 64×32 + 32 = 2.080 con số
        #   32 → 16 : 32×16 + 16 =   528 con số
        #   16 → 1  : 16×1  + 1  =    17 con số
        #   Tổng 3.713 con số: đây chính là bộ số được gửi qua mạng mỗi vòng.
        # ReLU đặt giữa các lớp để mô hình học được quan hệ phức tạp hơn phép cộng đơn giản.
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        # edge_index bỏ qua — giữ cùng chữ ký với SageMLP để dùng chung hàm train/predict
        # Kết quả chưa nằm trong khoảng 0 đến 1. Muốn ra điểm 0 đến 1 thì qua sigmoid, như ở evaluate_split.
        return self.net(x).squeeze(-1)
