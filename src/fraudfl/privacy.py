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

# =====================================================================================
# FILE NÀY LÀ GÌ: nơi các con số của mô hình THỰC SỰ được sửa.
#
#   dpsgd_steps()      : VÒNG LẶP HỌC, phần quan trọng nhất. Flower gọi tới hàm này.
#                        Không bật DP vẫn chạy qua đây, chỉ là tắt nhiễu và tắt cắt gradient.
#   noise_multiplier() : đổi ngân sách ε thành mức nhiễu z.
#   privatize_update() : cách DP khác, dùng cho thí nghiệm cũ. Bản Flower không gọi hàm này.
#
# Từ cần biết:
#   gradient : với mỗi con số, cho biết nên tăng hay giảm để bớt chấm sai
#   clip     : ngưỡng cắt, không cho một giao dịch nào ảnh hưởng quá mạnh
#   noise    : nhiễu ngẫu nhiên, che dấu vết của từng giao dịch
# =====================================================================================

import warnings

import torch
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.func import functional_call, grad, vmap

DELTA = 1e-5                        # δ: xác suất rất nhỏ cho phép lời hứa ε bị vi phạm
CLIP = 1.0                          # ngưỡng cắt gradient mặc định


def noise_multiplier(epsilon: float, sample_rate: float, steps: int, delta: float = DELTA) -> float:
    # Hỏi Opacus: muốn đạt ε này với tỷ lệ bốc và số bước này thì nhiễu z phải bằng bao nhiêu?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                          sample_rate=sample_rate, steps=steps, accountant="rdp"))


def privatize_update(local: dict, global_: dict, clip: float, z: float, gen: torch.Generator) -> dict:
    """Client-level: cắt Δ về chuẩn ≤ clip (trên toàn bộ tham số) rồi cộng nhiễu Gauss. Trả về state mới."""
    # Bản Flower không gọi hàm này. Hàm này thêm nhiễu vào cả bản cập nhật của ngân hàng,
    # còn dpsgd_steps bên dưới thêm nhiễu ở từng bước học.
    keys = list(global_)
    delta = torch.cat([(local[k] - global_[k]).flatten().float() for k in keys])   # phần ngân hàng đã sửa
    delta = delta * min(1.0, clip / (delta.norm().item() + 1e-12))   # cắt nếu sửa quá nhiều
    delta = delta + torch.normal(0.0, z * clip, size=delta.shape, generator=gen)   # cộng nhiễu
    out, i = {}, 0
    for k in keys:                  # trả phần sửa về đúng từng lớp của mô hình
        n = global_[k].numel()
        out[k] = global_[k] + delta[i:i + n].view_as(global_[k])
        i += n
    return out


def add_server_noise(avg: dict, global_: dict, std: float, gen: torch.Generator) -> dict:
    """Server cộng nhiễu Gauss N(0, std²) MỘT lần vào phần thay đổi của mô hình trung bình.

    Dùng cho DP mức client khi server được tin cậy (hoặc có SecAgg): mỗi ngân hàng chỉ cắt Δ về ≤ C
    (privatize_update với z = 0), trung bình đều K ngân hàng có độ nhạy C/K, nên std = z·C/K.
    So với mỗi ngân hàng tự cộng nhiễu z·C, tổng nhiễu nhỏ hơn √K lần với cùng ε.
    """
    return {k: avg[k] + torch.normal(0.0, std, size=avg[k].shape, generator=gen) if avg[k].is_floating_point()
            else avg[k] for k in global_}


def _mlp_tuyen_tinh(model: nn.Module):
    """model.net là chuỗi chỉ gồm Linear / ReLU và chứa mọi tham số của model → trả về (net, tên tham số), ngược lại None."""
    net = getattr(model, "net", None)
    if not isinstance(net, nn.Sequential) or not all(isinstance(m, (nn.Linear, nn.ReLU)) for m in net):
        return None
    lin = [(i, m) for i, m in enumerate(net) if isinstance(m, nn.Linear)]
    if any(m.bias is None for _, m in lin) or \
            sum(p.numel() for p in model.parameters()) != sum(p.numel() for _, m in lin for p in m.parameters()):
        return None
    ten = {id(p): n for n, p in model.named_parameters()}
    return net, [(m, ten[id(m.weight)], ten[id(m.bias)]) for _, m in lin]


def _tong_gradient_da_cat_ghost(net, lin, xb, yb, clip, pw) -> dict:
    """Ghost clipping cho MLP (Goodfellow 2015; Li et al. 2022): ở lớp tuyến tính, gradient của mẫu i là g_i·a_iᵀ (+ g_i cho
    bias) nên ‖·‖² = ‖g_i‖²(‖a_i‖² + 1). Tính chuẩn từng mẫu và tổng đã cắt Σ c_i g_i a_iᵀ bằng vài phép nhân ma trận,
    không cần dựng gradient từng mẫu. Kết quả bằng đúng đường vmap (test_privacy.py)."""
    h, vao, ra = xb, [], []
    for m in net:
        if isinstance(m, nn.Linear):
            vao.append(h.detach())
            h = m(h)
            ra.append(h)
        else:
            h = m(h)
    loss = nn.functional.binary_cross_entropy_with_logits(h.squeeze(-1), yb, pos_weight=pw, reduction="sum")
    G = torch.autograd.grad(loss, ra)            # tổng loss → hàng i chỉ phụ thuộc mẫu i
    sq = sum(g.pow(2).sum(1) * (a.pow(2).sum(1) + 1.0) for g, a in zip(G, vao))
    c = (clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
    out = {}
    for (m, tw, tb), g, a in zip(lin, G, vao):
        gc = g * c[:, None]
        out[tw], out[tb] = gc.T @ a, gc.sum(0)
    return out


def dpsgd_steps(model: nn.Module, X: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer, steps: int,
                sample_rate: float, clip: float, z: float, pos_weight: float, gen: torch.Generator,
                ghost: bool | None = None):
    """Record-level DP-SGD: lấy mẫu Poisson, gradient từng mẫu (torch.func), cắt, cộng nhiễu, chia cho batch kỳ vọng.

    `ghost`: None = tự dùng ghost clipping nếu model là MLP chỉ gồm Linear/ReLU (nhanh hơn nhiều, cùng kết quả);
    False = luôn dùng vmap."""
    # X: 16 đặc trưng của các giao dịch phần fit. y: đáp án, 1 là gian lận, 0 là thường.
    names = [n for n, _ in model.named_parameters()]   # tên các nhóm số trong mô hình
    pw = torch.tensor(pos_weight)   # mức phạt khi chấm sai ca gian lận, mặc định 10

    def loss_one(params, xb, yb):
        # Đo mức sai trên MỘT giao dịch: chấm sai càng nặng thì số trả về càng lớn
        out = functional_call(model, params, (xb.unsqueeze(0),)).squeeze()   # mô hình chấm giao dịch này
        return nn.functional.binary_cross_entropy_with_logits(out, yb, pos_weight=pw)   # so với đáp án

    # grad(loss_one): với mỗi con số, tính hướng sửa để bớt sai
    # vmap: làm việc đó cho từng giao dịch trong lô, không gộp chung, vì DP cần cắt riêng từng giao dịch
    per_sample_grad = vmap(grad(loss_one), in_dims=(None, 0, 0))
    expected_batch = sample_rate * len(X)   # cỡ lô trung bình, khoảng 1.024
    model.train()                   # chuyển sang chế độ học
    if z == 0 and clip == float("inf"):
        # KHÔNG DP (không cắt, không nhiễu): tổng gradient từng mẫu = gradient của tổng loss → tính một lần cho cả lô,
        # nhanh hơn vmap nhiều lần, cùng kết quả (test_privacy.py so hai đường). Vẫn lấy mẫu Poisson như bản DP.
        for _ in range(steps):
            idx = torch.nonzero(torch.rand(len(X), generator=gen) < sample_rate).squeeze(1)
            opt.zero_grad()
            if len(idx):
                out = model(X[idx]).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(out, y[idx], pos_weight=pw, reduction="sum")
                (loss / expected_batch).backward()
            else:
                for p in model.parameters():
                    p.grad = torch.zeros_like(p)
            opt.step()
        return
    gh = _mlp_tuyen_tinh(model) if ghost is not False else None
    if ghost and gh is None:
        raise ValueError("ghost clipping chỉ dùng cho MLP gồm Linear/ReLU")
    for _ in range(steps):          # VÒNG LẶP HỌC, khoảng 39 bước mỗi vòng
        # Bước 1, BỐC LÔ: mỗi giao dịch có xác suất sample_rate được chọn, nên cỡ lô dao động quanh 1.024
        idx = torch.nonzero(torch.rand(len(X), generator=gen) < sample_rate).squeeze(1)
        params = {n: p.detach() for n, p in model.named_parameters()}   # các con số hiện tại
        if len(idx) and gh is not None:   # ghost clipping: cùng kết quả bước 2–4, không dựng gradient từng mẫu
            with torch.enable_grad():
                summed = _tong_gradient_da_cat_ghost(gh[0], gh[1], X[idx], y[idx], clip, pw)
        elif len(idx):              # lô có ít nhất một giao dịch
            # Bước 2, CHẤM VÀ SO ĐÁP ÁN: tính hướng sửa của từng giao dịch
            g = per_sample_grad(params, X[idx], y[idx])
            # Bước 3, CẮT: đo độ lớn hướng sửa của từng giao dịch, cái nào vượt clip thì thu nhỏ về clip.
            # Không bật DP thì clip = vô cực, scale luôn bằng 1, tức không cắt gì.
            norms = torch.sqrt(sum(g[n].flatten(1).pow(2).sum(1) for n in names))
            scale = (clip / (norms + 1e-12)).clamp(max=1.0)
            # Bước 4, CỘNG LẠI: tổng hướng sửa của cả lô
            summed = {n: (g[n] * scale.view(-1, *[1] * (g[n].dim() - 1))).sum(0) for n in names}
        else:                       # hiếm khi lô rỗng, khi đó tổng bằng 0
            summed = {n: torch.zeros_like(params[n]) for n in names}
        opt.zero_grad()             # xóa hướng sửa của bước trước
        for n, p in model.named_parameters():
            # Bước 5, THÊM NHIỄU: không bật DP thì z = 0 nên không có nhiễu
            noise = torch.normal(0.0, z * clip, size=p.shape, generator=gen) if z > 0 else 0.0
            # Bước 6, LẤY TRUNG BÌNH: chia cho cỡ lô trung bình
            p.grad = (summed[n] + noise) / expected_batch
        # Bước 7, SỬA SỐ: 3.713 con số thay đổi ở dòng này, theo hướng sửa nhân với lr
        opt.step()
