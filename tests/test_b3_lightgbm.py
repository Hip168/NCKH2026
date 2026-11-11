import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import b3_lightgbm as b3  # noqa: E402


def test_chuan_hoa_theo_ngan_hang_va_nho():
    rng = np.random.default_rng(0)
    # Ngân hàng 0 có 50 dòng fit (đủ >= 30 tự chuẩn hóa), ngân hàng 1 chỉ có 5 dòng fit (dùng chuẩn hóa chung)
    X = rng.normal(10.0, 2.0, size=(100, 4)).astype(np.float32)
    X[:70] += 100.0
    cid = np.array([0] * 70 + [1] * 30)
    fit_idx = np.concatenate([np.arange(50), np.arange(70, 75)])
    val_idx = np.concatenate([np.arange(50, 60), np.arange(75, 85)])
    test_idx = np.concatenate([np.arange(60, 70), np.arange(85, 100)])

    Xf, Xv, Xt = b3._chuan_hoa_theo_ngan_hang(X, cid, 2, fit_idx, val_idx, test_idx, min_fit=30)
    assert Xf.shape == (len(fit_idx), 4)
    assert np.isfinite(Xf).all() and np.isfinite(Xv).all() and np.isfinite(Xt).all()
    # Ngân hàng 0 tự chuẩn hóa → trung bình phần fit xấp xỉ 0
    assert np.allclose(Xf[:50].mean(0), 0.0, atol=1e-5)


def test_chay_lightgbm_fold_chon_spw_tren_val():
    rng = np.random.default_rng(42)
    n_fit, n_val, n_test = 600, 200, 200
    X_fit = rng.normal(size=(n_fit, 6)).astype(np.float32)
    y_fit = (X_fit[:, 0] + 0.5 * X_fit[:, 1] > 1.2).astype(np.int64)
    X_val = rng.normal(size=(n_val, 6)).astype(np.float32)
    y_val = (X_val[:, 0] + 0.5 * X_val[:, 1] > 1.2).astype(np.int64)
    X_test = rng.normal(size=(n_test, 6)).astype(np.float32)

    val_sc, test_sc, info = b3.chay_lightgbm_fold(
        X_fit, y_fit, None, X_val, y_val, X_test,
        spw_grid=[1.0, 5.0], seed=42, n_estimators=40, lr=0.1, num_leaves=15, early_stopping=10, n_jobs=1,
        log=lambda m: None,
    )
    assert len(val_sc) == n_val and len(test_sc) == n_test
    assert np.all((test_sc >= 0.0) & (test_sc <= 1.0))
    assert info["scale_pos_weight"] in (1.0, 5.0)
    assert info["val_pr_auc"] > 0.5


def test_viet_bao_cao(tmp_path):
    df = pd.DataFrame([
        {"mo_hinh": "lgbm", "dac_trung": "co_ban", "seed": 42, "fold": 0, "pr_auc": 0.25, "recall@1%": 0.4, "vdr@1%": 0.5, "macro_pr_auc": 0.2, "thoi_gian": 12.0},
        {"mo_hinh": "lgbm", "dac_trung": "do_thi_fl", "seed": 42, "fold": 0, "pr_auc": 0.35, "recall@1%": 0.5, "vdr@1%": 0.6, "macro_pr_auc": 0.3, "thoi_gian": 15.0},
    ])
    b3.viet_bao_cao(df, tmp_path)
    md = (tmp_path / "BAO_CAO.md").read_text(encoding="utf-8")
    assert "do_thi_fl" in md and "0.3500" in md and "FL-safe" in md
