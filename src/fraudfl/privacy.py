"""Differential Privacy cho FL. Hai mức bảo vệ (xem docs/decisions.md D12):

1. Client-level (đúng Slide 4): mỗi ngân hàng cắt cập nhật Δ = w_local − w_global về chuẩn L2 ≤ C rồi cộng
   nhiễu N(0, (zC)²I) trước khi gửi. Bảo vệ TOÀN BỘ dữ liệu của một ngân hàng trước server
   (quan hệ lân cận: thêm/bớt một client, độ nhạy C). Mỗi vòng là một cơ chế Gauss không lấy mẫu con.
2. Record-level (DP-SGD, Abadi et al. 2016): trong mỗi ngân hàng, lấy mẫu Poisson tỷ lệ q, cắt gradient
   TỪNG giao dịch về ≤ C, cộng nhiễu N(0, (zC)²I) vào tổng gradient. Bảo vệ từng giao dịch; được khuếch đại
   nhờ lấy mẫu con nên cần ít nhiễu hơn nhiều. Chỉ áp dụng cho mô hình mà mỗi mẫu có gradient độc lập
   (MLP dạng bảng) — GNN không dùng được vì gradient của một nút phụ thuộc các nút láng giềng.

Ngân sách ε tính bằng RDP accountant của Opacus.
"""

import warnings

import torch
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.func import functional_call, grad, vmap

DELTA = 1e-5
CLIP = 1.0


def noise_multiplier(epsilon: float, sample_rate: float, steps: int, delta: float = DELTA) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                          sample_rate=sample_rate, steps=steps, accountant="rdp"))


def privatize_update(local: dict, global_: dict, clip: float, z: float, gen: torch.Generator) -> dict:
    """Client-level: cắt Δ về chuẩn ≤ clip (trên toàn bộ tham số) rồi cộng nhiễu Gauss. Trả về state mới."""
    keys = list(global_)
    delta = torch.cat([(local[k] - global_[k]).flatten().float() for k in keys])
    delta = delta * min(1.0, clip / (delta.norm().item() + 1e-12))
    delta = delta + torch.normal(0.0, z * clip, size=delta.shape, generator=gen)
    out, i = {}, 0
    for k in keys:
        n = global_[k].numel()
        out[k] = global_[k] + delta[i:i + n].view_as(global_[k])
        i += n
    return out


def dpsgd_steps(model: nn.Module, X: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer, steps: int,
                sample_rate: float, clip: float, z: float, pos_weight: float, gen: torch.Generator):
    """Record-level DP-SGD: lấy mẫu Poisson, gradient từng mẫu (torch.func), cắt, cộng nhiễu, chia cho batch kỳ vọng."""
    names = [n for n, _ in model.named_parameters()]
    pw = torch.tensor(pos_weight)

    def loss_one(params, xb, yb):
        out = functional_call(model, params, (xb.unsqueeze(0),)).squeeze()
        return nn.functional.binary_cross_entropy_with_logits(out, yb, pos_weight=pw)

    per_sample_grad = vmap(grad(loss_one), in_dims=(None, 0, 0))
    expected_batch = sample_rate * len(X)
    model.train()
    for _ in range(steps):
        idx = torch.nonzero(torch.rand(len(X), generator=gen) < sample_rate).squeeze(1)
        params = {n: p.detach() for n, p in model.named_parameters()}
        if len(idx):
            g = per_sample_grad(params, X[idx], y[idx])
            norms = torch.sqrt(sum(g[n].flatten(1).pow(2).sum(1) for n in names))
            scale = (clip / (norms + 1e-12)).clamp(max=1.0)
            summed = {n: (g[n] * scale.view(-1, *[1] * (g[n].dim() - 1))).sum(0) for n in names}
        else:
            summed = {n: torch.zeros_like(params[n]) for n in names}
        opt.zero_grad()
        for n, p in model.named_parameters():
            noise = torch.normal(0.0, z * clip, size=p.shape, generator=gen) if z > 0 else 0.0
            p.grad = (summed[n] + noise) / expected_batch
        opt.step()
