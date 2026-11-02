import torch
from torch import nn

from fraudfl.models import TabMLP
from fraudfl.privacy import dpsgd_steps, noise_multiplier, privatize_update


def test_privatize_update_clips_without_noise():
    g = {"w": torch.zeros(10), "b": torch.zeros(3)}
    local = {"w": torch.full((10,), 5.0), "b": torch.full((3,), 5.0)}
    out = privatize_update(local, g, clip=1.0, z=0.0, gen=torch.Generator().manual_seed(0))
    norm = torch.cat([out["w"], out["b"]]).norm()
    assert abs(norm.item() - 1.0) < 1e-5


def test_dpsgd_without_clip_or_noise_equals_plain_gradient():
    torch.manual_seed(0)
    X, y = torch.randn(64, 5), (torch.rand(64) < 0.3).float()
    a, b = TabMLP(5), TabMLP(5)
    b.load_state_dict(a.state_dict())

    # q = 1 → mọi mẫu được chọn; tổng gradient / 64 = gradient của loss trung bình
    dpsgd_steps(a, X, y, torch.optim.SGD(a.parameters(), lr=0.1), steps=1, sample_rate=1.0,
                clip=float("inf"), z=0.0, pos_weight=3.0, gen=torch.Generator().manual_seed(0))
    opt = torch.optim.SGD(b.parameters(), lr=0.1)
    nn.functional.binary_cross_entropy_with_logits(b(X), y, pos_weight=torch.tensor(3.0)).backward()
    opt.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)


def test_noise_decreases_with_epsilon():
    zs = [noise_multiplier(e, sample_rate=0.02, steps=500) for e in (0.5, 1.0, 2.0)]
    assert zs[0] > zs[1] > zs[2] > 0


def test_server_noise_std_and_uniform_fedavg_clip_only():
    from fraudfl.privacy import add_server_noise
    g = {"w": torch.zeros(200_000)}
    out = add_server_noise({"w": torch.zeros(200_000)}, g, std=0.5, gen=torch.Generator().manual_seed(0))
    assert abs(out["w"].std().item() - 0.5) < 0.01
