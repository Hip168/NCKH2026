"""Mô hình GraphSAGE + MLP theo Slide 6."""

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class SageMLP(nn.Module):
    """GraphSAGE 2 lớp (128 → 64) → MLP 64 → 32 → 16 → 1 (logit gian lận)."""

    def __init__(self, in_dim: int, hidden: int = 128, emb: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, emb)
        self.dropout = nn.Dropout(dropout)
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
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        # edge_index bỏ qua — giữ cùng chữ ký với SageMLP để dùng chung hàm train/predict
        return self.net(x).squeeze(-1)
