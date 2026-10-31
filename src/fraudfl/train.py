"""Huấn luyện GNN full-batch (centralized / local) và FedAvg mô phỏng nhiều client.

Một `GraphData` là đồ thị của một bên (cả hệ thống, hoặc một ngân hàng) cùng các mặt nạ fit/val/test.
Dừng sớm theo PR-AUC trên val; giữ trọng số tốt nhất.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    fit: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    node_ids: np.ndarray        # chỉ số nút trong bảng giao dịch gốc

    @property
    def n_fit(self) -> int:
        return int(self.fit.sum())


def _pos_weight(datas: list[GraphData]) -> torch.Tensor:
    pos = sum(float(d.y[d.fit].sum()) for d in datas)
    neg = sum(d.n_fit for d in datas) - pos
    return torch.tensor(neg / max(pos, 1.0))


@torch.no_grad()
def predict(model: nn.Module, d: GraphData) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(d.x, d.edge_index)).numpy()


def _local_epochs(model, d: GraphData, opt, loss_fn, epochs: int):
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(d.x, d.edge_index)[d.fit], d.y[d.fit])
        loss.backward()
        opt.step()


def _val_ap(model, datas: list[GraphData]) -> float:
    y = np.concatenate([d.y[d.val].numpy() for d in datas])
    s = np.concatenate([predict(model, d)[d.val.numpy()] for d in datas])
    return float(average_precision_score(y, s))


def train_full_batch(model: nn.Module, d: GraphData, lr: float, max_epochs: int, patience: int,
                     eval_every: int = 5, log=None) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight([d]))
    best, best_state, bad = -1.0, None, 0
    for epoch in range(eval_every, max_epochs + 1, eval_every):
        _local_epochs(model, d, opt, loss_fn, eval_every)
        ap = _val_ap(model, [d])
        if ap > best:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += eval_every
        if log:
            log(f"    epoch {epoch:3d}  val PR-AUC {ap:.4f}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model


def fedavg(model: nn.Module, clients: list[GraphData], lr: float, rounds: int, local_epochs: int,
           log=None, privatize=None, local_train=None) -> tuple[nn.Module, list[dict]]:
    """FedAvg (McMahan et al., 2017): trung bình trọng số theo số mẫu fit của từng client.

    Mỗi vòng: client nhận mô hình chung → train `local_epochs` epoch trên dữ liệu riêng (Adam mới mỗi vòng)
    → gửi trọng số → server lấy trung bình có trọng số. Chọn vòng có PR-AUC val (gộp) cao nhất.
    `privatize(local_state, global_state) -> state`: nếu có, client áp DP lên cập nhật trước khi gửi.
    `local_train(model, client)`: thay cho bước train Adam mặc định (ví dụ DP-SGD).
    Trả về (mô hình tốt nhất, lịch sử từng vòng).
    """
    pos_weight = _pos_weight(clients)   # chỉ là một số vô hướng; thực tế có thể để từng client tự tính
    weights = np.array([c.n_fit for c in clients], dtype=np.float64)
    weights /= weights.sum()
    best, best_state, history = -1.0, None, []
    for r in range(1, rounds + 1):
        global_state = copy.deepcopy(model.state_dict())
        states = []
        for c in clients:
            model.load_state_dict(global_state)
            if local_train:
                local_train(model, c)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                _local_epochs(model, c, opt, nn.BCEWithLogitsLoss(pos_weight=pos_weight), local_epochs)
            state = copy.deepcopy(model.state_dict())
            states.append(privatize(state, global_state) if privatize else state)
        model.load_state_dict({k: sum(w * s[k] for w, s in zip(weights, states)) for k in global_state})
        ap = _val_ap(model, clients)
        history.append({"round": r, "val_pr_auc": ap})
        if ap > best:
            best, best_state = ap, copy.deepcopy(model.state_dict())
        if log:
            log(f"    round {r:3d}  val PR-AUC {ap:.4f}")
    model.load_state_dict(best_state)
    return model, history
