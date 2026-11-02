"""Chấm điểm bằng mô hình toàn cục từ Flower, chuẩn hóa bằng thống kê riêng của từng ngân hàng."""

from pathlib import Path

import numpy as np
import torch

from fraudfl.data.features import FEATURES
from fraudfl.fl.partition import fold_path, scaler
from fraudfl.fl.task import new_model


class Scorer:
    def __init__(self, model_path: Path, data_root: Path, bank: int, fold: int, alert_frac: float = 0.01):
        self.model = new_model(len(FEATURES))
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        path = fold_path(data_root, f"bank_{bank}", fold)
        self.mu, self.sd = scaler(path)
        self.threshold = self._alert_threshold(path, alert_frac)

    @torch.no_grad()
    def _alert_threshold(self, path, frac):
        """Ngưỡng để khoảng `frac` giao dịch bị gắn cờ, chọn trên val của chính ngân hàng (D3, D7)."""
        with np.load(path, allow_pickle=False) as z:
            x = torch.from_numpy(z["x"][z["val"]])
        return float(np.quantile(torch.sigmoid(self.model(x)).numpy(), 1 - frac))

    @torch.no_grad()
    def score(self, rows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(((rows - self.mu) / self.sd).astype(np.float32))
        return torch.sigmoid(self.model(x)).numpy()
