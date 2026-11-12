"""B3 — Mô hình gộp chung (cận trên) trên IBM HI-Small với khung B0 (D25, D28).

So sánh LightGBM và MLP trên các bộ đặc trưng:
  - co_ban:       22 đặc trưng dạng bảng hiện có (ibm_features, view bank)
  - do_thi_fl:    22 đặc trưng cơ bản + 9 đặc trưng đồ thị phía gửi / chu trình A->B->A (FL-safe 100%)
  - do_thi_bank:  22 đặc trưng cơ bản + 18 đặc trưng đồ thị trong phạm vi từng ngân hàng
  - do_thi_nuoc:  22 đặc trưng cơ bản + 18 đặc trưng đồ thị trong phạm vi từng nước
  - do_thi_global:22 đặc trưng cơ bản + 18 đặc trưng đồ thị toàn mạng (cận trên tuyệt đối)

Luật bắt buộc (docs/giao_viec/EM2_B3_mo_hinh.md):
  - Chia dữ liệu cố định từ fraudfl.khung (gan_fold + vai_tro).
  - Chọn scale_pos_weight / early stopping / ngưỡng F1 chỉ trên val; test chấm 1 lần.
  - Chấm bằng fraudfl.eval.metrics.evaluate.

Chạy mẫu (trên Kaggle hoặc máy có đủ dữ liệu):
  python scripts/b3_lightgbm.py --raw /path/to/HI-Small_Trans.csv --folds 0 --seeds 42 --tag thu_f0
  python scripts/b3_lightgbm.py --raw /path/to/HI-Small_Trans.csv --folds 0 1 2 3 4 --seeds 42 7 2027 --tag chot
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudfl.config import DATA_PROCESSED, DATA_SPLITS, IBM_MAX_DAY, RESULTS
from fraudfl.data.dac_trung_do_thi import (
    ALL_GRAPH_FEATURES,
    GRAPH_FEATURES_FL,
    GRAPH_FEATURES_RCV,
    build_graph_features,
)
from fraudfl.data.ibm_aml import load_raw
from fraudfl.data.ibm_features import feature_columns
from fraudfl.data.sampling import subsample_negatives
from fraudfl.eval.metrics import best_f1_threshold, evaluate
from fraudfl.khung import FILE_CHIA, SEEDS as B0_SEEDS, gan_fold, vai_tro
from fraudfl.runlog import archive_run

BO_DAC_TRUNG = ("co_ban", "do_thi_fl", "do_thi_bank", "do_thi_nuoc", "do_thi_global")
DEFAULT_SPW_GRID = [1.0, 3.0, 5.0, 10.0, 20.0]


def ram_dinh_mb() -> float:
    """RAM đỉnh của tiến trình hiện tại (Linux/macOS qua resource, Windows trả 0 nếu không có psutil)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == "darwin" else r / 2**10
    except ImportError:
        try:
            import psutil

            return psutil.Process().memory_info().rss / 2**20
        except ImportError:
            return 0.0


def _dam_bao_do_thi(stem: str, scope: str, raw_path: Path | None, log) -> Path:
    """Đảm bảo file parquet đặc trưng đồ thị cho `scope` đã tồn tại; nếu chưa thì tính từ `--raw`."""
    p = DATA_PROCESSED / f"{stem}_graph_{scope}.parquet"
    if p.exists():
        return p
    if raw_path is None or not raw_path.exists():
        raise SystemExit(
            f"Chưa có {p} và không tìm thấy file gốc --raw={raw_path}. "
            "Truyền --raw <đường_dẫn/HI-Small_Trans.csv> để tự tính đặc trưng đồ thị."
        )
    t0 = time.time()
    log(f"Đang tính đặc trưng đồ thị (scope={scope}) từ {raw_path.name} ...")
    df = load_raw(raw_path)
    b2c = None
    if scope == "country":
        file_nuoc = DATA_SPLITS / "ibm_ngan_hang_quoc_gia.csv"
        if not file_nuoc.exists():
            raise SystemExit(f"Thiếu bảng ngân hàng → nước: {file_nuoc}")
        b2c = pd.read_csv(file_nuoc, dtype=str).set_index("BANK")["NUOC"]
    g = build_graph_features(df, scope=scope, bank_to_country=b2c)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    g.to_parquet(p, index=False)
    log(f"  Đã lưu {p.name} ({len(g):,} dòng, {time.time() - t0:.1f}s)")
    del df, g
    gc.collect()
    return p


def chuan_bi_bang_du_lieu(stem: str, feature_sets: list[str], raw_path: Path | None, log) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Đọc bảng đặc trưng cơ bản và ghép thêm các cột đồ thị theo từng bộ yêu cầu."""
    base_path = DATA_PROCESSED / f"{stem}_features_bank.parquet"
    if not base_path.exists():
        raise SystemExit(
            f"Chưa có {base_path}. Hãy chạy trước: python scripts/06_ibm_features.py --input <HI-Small_Trans.csv> --view bank"
        )
    t0 = time.time()
    feat = pd.read_parquet(base_path)
    feat = feat[feat["TX_TIME_DAYS"] <= IBM_MAX_DAY].reset_index(drop=True)
    base_cols = feature_columns(feat)
    log(f"Nạp {base_path.name}: {len(feat):,} dòng ngày 0–{IBM_MAX_DAY}, {int(feat['LABEL'].sum()):,} ca, "
        f"{len(base_cols)} đặc trưng cơ bản ({time.time() - t0:.1f}s)")

    scope_map = {
        "do_thi_fl": ("global", GRAPH_FEATURES_FL),       # FL-safe giống hệt ở mọi scope, dùng global cho nhanh
        "do_thi_bank": ("bank", ALL_GRAPH_FEATURES),
        "do_thi_nuoc": ("country", ALL_GRAPH_FEATURES),
        "do_thi_global": ("global", ALL_GRAPH_FEATURES),
    }

    col_sets: dict[str, list[str]] = {}
    loaded_scopes: dict[str, pd.DataFrame] = {}

    for fs in feature_sets:
        if fs == "co_ban":
            col_sets[fs] = list(base_cols)
            continue
        scope, g_cols = scope_map[fs]
        if scope not in loaded_scopes:
            gp = _dam_bao_do_thi(stem, scope, raw_path, log)
            gdf = pd.read_parquet(gp)
            gdf = gdf.set_index("TX_ID").reindex(feat["TX_ID"]).reset_index(drop=True)
            loaded_scopes[scope] = gdf
        gdf = loaded_scopes[scope]
        prefix = "" if scope == "global" else f"{scope}__"
        added_cols = []
        for c in g_cols:
            col_name = c if c in GRAPH_FEATURES_FL or scope == "global" else f"{prefix}{c}"
            if col_name not in feat.columns:
                feat[col_name] = gdf[c].to_numpy(np.float32)
            added_cols.append(col_name)
        col_sets[fs] = list(base_cols) + added_cols

    del loaded_scopes
    gc.collect()
    return feat, col_sets


def chay_lightgbm_fold(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    spw_grid: list[float],
    seed: int,
    n_estimators: int,
    lr: float,
    num_leaves: int,
    early_stopping: int,
    n_jobs: int,
    log,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Huấn luyện LightGBM; chọn scale_pos_weight tốt nhất dựa trên val PR-AUC."""
    import lightgbm as lgb

    best_val_ap = -1.0
    best_model = None
    best_spw = spw_grid[0]
    best_iter = 0
    best_val_score = None

    min_child_samples = min(100, max(20, len(y_fit) // 50))
    for spw in spw_grid:
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            min_child_weight=5.0,
            reg_lambda=10.0,
            max_delta_step=1.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=float(spw),
            metric="average_precision",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
            model.fit(
                X_fit,
                y_fit,
                sample_weight=w_fit,
                eval_set=[(X_val, y_val)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping, first_metric_only=True, verbose=False)],
            )
            val_pred = model.predict_proba(X_val)[:, 1]
        val_ap = float(average_precision_score(y_val, val_pred))
        b_iter = int(model.best_iteration_ or n_estimators)
        log(f"    [LGBM spw={spw:g}] val PR-AUC={val_ap:.4f} (best_iter={b_iter})")
        if np.isfinite(val_ap) and val_ap > best_val_ap:
            best_val_ap = val_ap
            best_model = model
            best_spw = float(spw)
            best_iter = b_iter
            best_val_score = val_pred

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
        test_score = best_model.predict_proba(X_test)[:, 1]
    return best_val_score, test_score, {"scale_pos_weight": best_spw, "best_iter": best_iter, "val_pr_auc": best_val_ap}


def _chuan_hoa_theo_ngan_hang(
    X: np.ndarray,
    cid: np.ndarray,
    K: int,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    min_fit: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chuẩn hóa z-score từng ngân hàng trên tập fit (ngân hàng ít mẫu dùng chuẩn hóa toàn cục)."""
    mu_c, sd_c = X[fit_idx].mean(0), X[fit_idx].std(0)
    mu = np.tile(mu_c, (K, 1)).astype(np.float32)
    sd = np.tile(np.where(sd_c > 0, sd_c, 1.0), (K, 1)).astype(np.float32)
    g = pd.DataFrame(X[fit_idx]).groupby(cid[fit_idx])
    du = g.size()
    du = du[du >= min_fit].index.to_numpy()
    if len(du):
        m_, s_ = g.mean().loc[du].to_numpy(np.float32), g.std(ddof=0).loc[du].to_numpy(np.float32)
        mu[du] = m_
        sd[du] = np.where(s_ > 0, s_, 1.0)

    def _norm(idx):
        return ((X[idx] - mu[cid[idx]]) / sd[cid[idx]]).astype(np.float32)

    return _norm(fit_idx), _norm(val_idx), _norm(test_idx)


def chay_mlp_fold(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    log,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Huấn luyện TabMLP với lô nhỏ và dừng sớm theo val PR-AUC (hỗ trợ GPU nếu có)."""
    import torch
    from torch import nn

    try:
        from fraudfl.models import TabMLP
    except ImportError:
        class TabMLP(nn.Module):  # dự phòng khi chạy offline trên Kaggle không có torch_geometric
            def __init__(self, in_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
                return self.net(x).squeeze(-1)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabMLP(X_fit.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")

    def _predict(arr: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), 32768):
                xb = torch.from_numpy(arr[i : i + 32768]).to(device)
                out.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.nan_to_num(np.concatenate(out), nan=0.0)

    best_val_ap = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad = 0
    n = len(y_fit)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            xb = torch.from_numpy(X_fit[b]).to(device)
            yb = torch.from_numpy(y_fit[b].astype(np.float32)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            if w_fit is not None:
                wb = torch.from_numpy(w_fit[b].astype(np.float32)).to(device)
                loss = (loss * wb).sum() / wb.sum()
            else:
                loss = loss.mean()
            loss.backward()
            opt.step()

        val_pred = _predict(X_val)
        val_ap = float(average_precision_score(y_val, val_pred))
        if np.isfinite(val_ap) and val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if epoch == 1 or epoch % 10 == 0 or bad == 0 and epoch >= 5:
            log(f"    [MLP epoch {epoch:3d}] val PR-AUC={val_ap:.4f} (best={best_val_ap:.4f} @{best_epoch})")

    model.load_state_dict(best_state)
    val_score = _predict(X_val)
    test_score = _predict(X_test)
    return val_score, test_score, {"scale_pos_weight": pos_weight, "best_iter": best_epoch, "val_pr_auc": best_val_ap}


def viet_bao_cao(df: pd.DataFrame, out: Path) -> None:
    """Sinh file tóm tắt kết quả BAO_CAO.md cho bước B3."""
    if df.empty:
        return
    grp = df.groupby(["mo_hinh", "dac_trung"], sort=False)[["pr_auc", "recall@1%", "vdr@1%", "macro_pr_auc", "thoi_gian"]]
    agg = grp.agg(["mean", "std"]).round(4)

    lines = [
        "# Báo cáo thử nghiệm B3 — LightGBM, MLP & Đặc trưng đồ thị",
        "",
        "## Tổng hợp kết quả (mean ± std qua các seed/fold đã chạy)",
        "",
        "| Mô hình | Bộ đặc trưng | PR-AUC | Recall@1% | VDR@1% | Macro PR-AUC | Thời gian (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (m, fs), row in agg.iterrows():
        def _fmt(col):
            mu = row[(col, "mean")]
            sd = row[(col, "std")]
            return f"{mu:.4f}" if pd.isna(sd) else f"{mu:.4f} ± {sd:.4f}"

        lines.append(
            f"| {m} | {fs} | {_fmt('pr_auc')} | {_fmt('recall@1%')} | {_fmt('vdr@1%')} | "
            f"{_fmt('macro_pr_auc')} | {row[('thoi_gian', 'mean')]:.1f} |"
        )

    lines += [
        "",
        "## Ghi chú tính khả dụng trong Học liên kết (FL)",
        "",
        f"- **FL-safe 100% ({len(GRAPH_FEATURES_FL)} cột):** `{', '.join(GRAPH_FEATURES_FL)}` — "
        "ngân hàng gửi tự tính hoàn toàn từ tài khoản mình quản lý.",
        f"- **Phụ thuộc phạm vi client ({len(GRAPH_FEATURES_RCV)} cột):** `{', '.join(GRAPH_FEATURES_RCV)}` — "
        "trong `do_thi_bank`/`do_thi_nuoc` chỉ dùng giao dịch liên quan đến client đang xét; trong `do_thi_global` "
        "thấy cả giao dịch của tài khoản nhận ở ngân hàng khác (KHÔNG dùng nguyên bản được trong FL thuần túy).",
        "",
    ]
    (out / "BAO_CAO.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--raw", type=Path, default=None, help="Đường dẫn HI-Small_Trans.csv để tự tính đặc trưng đồ thị nếu chưa có")
    ap.add_argument("--mo-hinh", nargs="+", choices=["lgbm", "mlp"], default=["lgbm"])
    ap.add_argument("--bo-dac-trung", nargs="+", choices=list(BO_DAC_TRUNG), default=["co_ban", "do_thi_fl", "do_thi_nuoc", "do_thi_global"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[B0_SEEDS[0]])
    ap.add_argument("--so-mau", type=int, default=0, help=">0: lấy mẫu con phần fit còn N dòng (giữ hết ca rửa tiền, bù trọng số)")
    ap.add_argument("--spw-grid", type=float, nargs="+", default=DEFAULT_SPW_GRID, help="Danh sách scale_pos_weight dò trên val cho LightGBM")
    ap.add_argument("--lgbm-estimators", type=int, default=1500)
    ap.add_argument("--lgbm-lr", type=float, default=0.05)
    ap.add_argument("--lgbm-leaves", type=int, default=31)
    ap.add_argument("--lgbm-patience", type=int, default=50)
    ap.add_argument("--mlp-epochs", type=int, default=120)
    ap.add_argument("--mlp-patience", type=int, default=15)
    ap.add_argument("--mlp-batch", type=int, default=2048)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-pos-weight", type=float, default=1.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = RESULTS / ("b3_mo_hinh" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text("", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        with open(out / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    if not FILE_CHIA.exists():
        raise SystemExit(f"Thiếu file chia cố định B0: {FILE_CHIA}")

    log(
        f"B3 | mô hình={args.mo_hinh} | đặc trưng={args.bo_dac_trung} | seeds={args.seeds} | folds={args.folds} | "
        f"so_mau={args.so_mau or 'đủ'} | spw_grid={args.spw_grid}"
    )

    feat, col_sets = chuan_bi_bang_du_lieu(args.stem, args.bo_dac_trung, args.raw, log)
    tx_id = feat["TX_ID"].to_numpy()
    y_all = feat["LABEL"].to_numpy().astype(np.int64)
    amt_all = feat["AMOUNT_USD"].to_numpy(np.float64)
    day_all = feat["TX_TIME_DAYS"].to_numpy()

    bank_names = feat["FROM_BANK"].unique()
    cid_all = pd.Series(range(len(bank_names)), index=bank_names)[feat["FROM_BANK"].to_numpy()].to_numpy()
    K_banks = len(bank_names)

    rows = []
    for seed in args.seeds:
        fold_b0 = gan_fold(tx_id, seed)
        for k in args.folds:
            m = vai_tro(fold_b0, k)
            fit_raw = np.flatnonzero(m["fit"])
            val_idx = np.flatnonzero(m["val"])
            test_idx = np.flatnonzero(m["test"])

            if args.so_mau > 0 and args.so_mau < len(fit_raw):
                am = int((y_all[fit_raw] == 0).sum())
                rate = min(1.0, (args.so_mau - int(y_all[fit_raw].sum())) / max(am, 1))
                bang = subsample_negatives(fit_raw, y_all[fit_raw], cid_all[fit_raw], rate, seed + k)
                fit_idx = bang["TX_ID"].to_numpy()
                w_fit = bang["WEIGHT"].to_numpy(np.float32)
            else:
                fit_idx = fit_raw
                w_fit = None

            y_fit, y_val, y_test = y_all[fit_idx], y_all[val_idx], y_all[test_idx]
            amt_test, day_test, cid_test = amt_all[test_idx], day_all[test_idx], cid_all[test_idx]
            nhom_test = pd.Series(np.arange(len(y_test))).groupby(cid_test).agg(list)
            nhom_test_idx = [np.asarray(ix) for ix in nhom_test if y_test[ix].sum() > 0]

            for fs in args.bo_dac_trung:
                cols = col_sets[fs]
                X_mat = feat[cols].to_numpy(np.float32)

                for mh in args.mo_hinh:
                    t0 = time.time()
                    log(f"\n=== seed={seed} fold={k} | {mh} | {fs} ({len(cols)} cột) ===")
                    if mh == "lgbm":
                        val_sc, test_sc, info = chay_lightgbm_fold(
                            X_mat[fit_idx], y_fit, w_fit,
                            X_mat[val_idx], y_val,
                            X_mat[test_idx],
                            args.spw_grid, seed + k,
                            args.lgbm_estimators, args.lgbm_lr, args.lgbm_leaves, args.lgbm_patience,
                            args.n_jobs, log,
                        )
                    else:
                        Xf_n, Xv_n, Xt_n = _chuan_hoa_theo_ngan_hang(X_mat, cid_all, K_banks, fit_idx, val_idx, test_idx)
                        val_sc, test_sc, info = chay_mlp_fold(
                            Xf_n, y_fit, w_fit, Xv_n, y_val, Xt_n,
                            seed + k, args.mlp_epochs, args.mlp_patience, args.mlp_batch,
                            args.mlp_lr, args.mlp_pos_weight, log,
                        )
                        del Xf_n, Xv_n, Xt_n

                    thr = best_f1_threshold(y_val, val_sc)
                    res = evaluate(y_test, test_sc, amt_test, day_test, threshold=thr)
                    macro_ap = float(np.mean([average_precision_score(y_test[ix], test_sc[ix]) for ix in nhom_test_idx]))
                    dt = time.time() - t0
                    ram = ram_dinh_mb()
                    row = {
                        "mo_hinh": mh,
                        "dac_trung": fs,
                        "so_cot": len(cols),
                        "seed": seed,
                        "fold": k,
                        "pr_auc": res["pr_auc"],
                        "recall@1%": res["recall@1%"],
                        "vdr@1%": res["vdr@1%"],
                        "macro_pr_auc": macro_ap,
                        "roc_auc": res["roc_auc"],
                        "f1": res["f1"],
                        "val_pr_auc": info["val_pr_auc"],
                        "scale_pos_weight": info["scale_pos_weight"],
                        "best_iter": info["best_iter"],
                        "thoi_gian": round(dt, 1),
                        "ram_dinh_mb": round(ram),
                    }
                    rows.append(row)
                    log(
                        f"  TEST PR-AUC {row['pr_auc']:.4f} | Recall@1% {row['recall@1%']:.4f} | "
                        f"VDR@1% {row['vdr@1%']:.4f} | macro {row['macro_pr_auc']:.4f} | {row['thoi_gian']:.1f}s"
                    )
                    df_out = pd.DataFrame(rows)
                    df_out.to_csv(out / "bang.csv", index=False, float_format="%.5f")
                    viet_bao_cao(df_out, out)

                del X_mat
                gc.collect()

    log(f"\nĐã lưu trữ: {archive_run(out, note=f'B3 {args.mo_hinh} {args.bo_dac_trung} seeds={args.seeds} folds={args.folds}')}")


if __name__ == "__main__":
    main()

