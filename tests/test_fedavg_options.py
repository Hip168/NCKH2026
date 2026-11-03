import numpy as np
import torch

from fraudfl.models import TabMLP
from fraudfl.train import GraphData, fedavg


def _clients(n_clients=3, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for c in range(n_clients):
        x = torch.randn(n, 5, generator=g)
        y = (x[:, 0] + 0.5 * torch.randn(n, generator=g) > 1).float()
        fit = torch.zeros(n, dtype=torch.bool); fit[:120] = True
        val = torch.zeros(n, dtype=torch.bool); val[120:160] = True
        out.append(GraphData(x, None, y, fit, val, ~(fit | val), np.arange(n) + c * n))
    return out


def _run(**kw):
    torch.manual_seed(0)
    m, _ = fedavg(TabMLP(5), _clients(), lr=1e-2, rounds=3, local_epochs=1, batch_size=32, pos_weight=1.0, **kw)
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def test_defaults_equal_explicit_plain_fedavg():
    torch.manual_seed(0); a = _run()
    torch.manual_seed(0); b = _run(prox_mu=0.0, server_lr=1.0, server_momentum=0.0)
    torch.testing.assert_close(a, b)


def test_large_prox_mu_keeps_model_close_to_start():
    torch.manual_seed(0)
    start = torch.cat([p.detach().flatten() for p in TabMLP(5).parameters()])
    torch.manual_seed(0); plain = _run()
    torch.manual_seed(0); prox = _run(prox_mu=100.0)
    assert (prox - start).norm() < (plain - start).norm()
