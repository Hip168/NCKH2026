"""Dữ liệu riêng của từng ngân hàng cho Flower và real-time.

Mỗi ngân hàng có một thư mục `data/banks/<tên>/`, trong container chỉ mount đúng thư mục đó, nên không
ngân hàng nào đọc được dữ liệu của bên khác. Cùng một code chạy được 2 chế độ:
- tách: bank_0 ... bank_4, mỗi bên một phần
- gộp:  một bên duy nhất tên `gop` chứa toàn bộ dữ liệu
"""

# =====================================================================================
# FILE NÀY LÀ GÌ: đọc và ghi file dữ liệu fold0.npz của từng ngân hàng.
# =====================================================================================

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fraudfl.config import ROOT
from fraudfl.data.features import FEATURES

BANKS_DIR = ROOT / "data" / "banks"


@dataclass
class Partition:
    # Toàn bộ dữ liệu của một ngân hàng sau khi đọc vào
    x: torch.Tensor     # bảng đặc trưng: mỗi dòng một giao dịch, 16 cột, đã chuẩn hóa
    y: torch.Tensor     # đáp án: 1 là gian lận, 0 là thường
    fit: np.ndarray     # cột Đúng/Sai: giao dịch này thuộc phần dùng để học
    val: np.ndarray     # cột Đúng/Sai: giao dịch này thuộc phần chọn vòng tốt nhất
    test: np.ndarray    # cột Đúng/Sai: giao dịch này thuộc phần chấm điểm cuối

    @property
    def n_fit(self) -> int:
        # Số giao dịch dùng để học. Khi gộp, ngân hàng có n_fit lớn hơn được tính nặng hơn.
        return int(self.fit.sum())


def fold_path(root: Path, name: str, fold: int) -> Path:
    # Ghép đường dẫn, ví dụ: data/banks/bank_0/fold0.npz
    return Path(root) / name / f"fold{fold}.npz"


def save_partition(path: Path, x: np.ndarray, y: np.ndarray, masks: dict, mu: np.ndarray, sd: np.ndarray):
    # Ghi dữ liệu của một ngân hàng ra file nén. Script 50_chia_du_lieu_ngan_hang.py gọi hàm này.
    # mu, sd: trung bình và độ lệch chuẩn của ngân hàng, dùng lại để chuẩn hóa khi chấm real-time.
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x.astype(np.float32), y=y.astype(np.float32), fit=masks["fit"], val=masks["val"],
                        test=masks["test"], mu=mu, sd=sd, features=np.array(FEATURES))


def load_partition(path: Path) -> Partition:
    # Đọc file fold0.npz thành một Partition
    with np.load(path, allow_pickle=False) as z:
        return Partition(torch.from_numpy(z["x"]), torch.from_numpy(z["y"]), z["fit"], z["val"], z["test"])


def scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Thống kê chuẩn hóa của ngân hàng, dùng lại khi chấm điểm real-time."""
    with np.load(path, allow_pickle=False) as z:
        return z["mu"], z["sd"]
