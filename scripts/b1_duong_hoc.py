"""B1 — Đường học: giữ bao nhiêu bản ghi là đủ. Gọi script 12 cho từng (cỡ, seed) rồi tổng hợp.

Mỗi cỡ N: giữ HẾT ca rửa tiền trong phần fit, bốc ca bình thường theo từng ngân hàng cho đủ N dòng, bù trọng số;
val và test giữ nguyên tỷ lệ thật. Cách chia: B0 (cố định, D25). Cỡ 0 = đủ dữ liệu (~3,05 triệu dòng fit).

    # chạy (bỏ qua cấu hình đã chạy xong; chạy lại được sau khi bị ngắt)
    python scripts/b1_duong_hoc.py --co 100000 250000 500000 1000000 0 --seeds 42 7 2027
    # thử nhanh: 1 cỡ, 1 seed, 1 fold, 20 lượt
    python scripts/b1_duong_hoc.py --co 100000 --seeds 42 --folds 0 --max-epochs 20 --patience 5 --ten thu
    # chỉ tổng hợp lại từ kết quả đã có
    python scripts/b1_duong_hoc.py --chi-tong-hop

Ra   results/ibm_cach1_b1_n<N>_s<seed>/       kết quả từng lần chạy script 12
     results/b1_duong_hoc/bang.csv            cỡ, seed, fold, PR-AUC, Recall@1%, VDR@1%, thời gian, RAM
     results/b1_duong_hoc/tom_tat.csv         trung bình ± độ lệch chuẩn theo cỡ, % so với đủ
     results/b1_duong_hoc/duong_hoc.png       biểu đồ
     results/b1_duong_hoc/de_xuat.txt         N nhỏ nhất đạt >= 95% "đủ" (tiêu chí chốt B1)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudfl.config import DATA_PROCESSED, RESULTS
from fraudfl.khung import SEEDS

S12 = Path(__file__).with_name("12_ibm_cach1_lo_nho.py")
CACHE = DATA_PROCESSED / "cach1"
MAU, CHU, LUOI = "#2a78d6", "#52514e", "#e6e5e1"
NGUONG = 0.95


def ten_chay(ten: str, n: int, seed: int) -> str:
    return f"{ten}_n{n}_s{seed}" if n else f"{ten}_du_s{seed}"


def thu_muc_cache(stem: str, n: int, seed: int) -> Path:       # khớp thu_muc_fold() của script 12 với --min-tx 0 --chia b0
    mau = f"_n{n}" if n else "_r1"
    return CACHE / f"{stem}_mintx0{mau}_b0_seed{seed}"


def da_xong(out: Path, folds: list[int]) -> bool:
    p = out / "per_fold.csv"
    return p.exists() and set(folds) <= set(pd.read_csv(p)["fold"])


def chay(args):
    for n in args.co:
        for s in args.seeds:
            tag = ten_chay(args.ten, n, s)
            out = RESULTS / f"ibm_cach1_{tag}"
            if da_xong(out, args.folds):
                print(f"[bỏ qua] {tag}: đã có đủ fold {args.folds}", flush=True)
                continue
            cmd = [sys.executable, str(S12), "--chia", "b0", "--min-tx", "0", "--seed", str(s),
                   "--folds", *map(str, args.folds), "--batch-size", "512", "--lr", "0.001",
                   "--max-epochs", str(args.max_epochs), "--patience", str(args.patience), "--tag", tag]
            if n:
                cmd += ["--so-mau", str(n), "--bu-trong-so"]
            print(f"\n[chạy] {tag}: {' '.join(cmd[1:])}", flush=True)
            r = subprocess.run(cmd)
            if r.returncode:
                raise SystemExit(f"Script 12 lỗi ở {tag} (mã {r.returncode}) — xem {out / 'log.txt'}")
            if not args.giu_cache:                  # cache của mỗi cấu hình 2–3 GB, không xoá thì đầy đĩa
                shutil.rmtree(thu_muc_cache(args.stem, n, s), ignore_errors=True)


def tong_hop(args):
    rows = []
    for p in sorted(RESULTS.glob(f"ibm_cach1_{args.ten}_*_s*/per_fold.csv")):
        d = pd.read_csv(p)
        if "chia" in d and (d["chia"] != "b0").any():
            continue
        d["co"] = d["so_mau"].where(d["so_mau"] > 0, d["n_fit"])
        d["du"] = d["so_mau"] == 0
        d["seed"] = d["seed"] if "seed" in d else int(p.parent.name.rsplit("_s", 1)[1])
        rows.append(d)
    if not rows:
        raise SystemExit(f"Chưa có kết quả nào dạng results/ibm_cach1_{args.ten}_*_s*/")
    bang = pd.concat(rows, ignore_index=True)
    cot = ["co", "du", "seed", "fold", "n_fit", "ca_fit", "pr_auc", "recall@1%", "vdr@1%", "val_pr_auc", "luot_tot",
           "so_luot", "giay", "ram_dinh_mb"]
    bang = bang[cot].sort_values(["du", "co", "seed", "fold"])
    out = RESULTS / ("b1_duong_hoc" if args.ten == "b1" else f"b1_duong_hoc_{args.ten}")
    out.mkdir(parents=True, exist_ok=True)
    bang.to_csv(out / "bang.csv", index=False, float_format="%.5f")

    bang["nhan"] = np.where(bang["du"], "đủ", bang["co"].map(lambda v: f"{v:,}"))
    tt = (bang.groupby(["du", "nhan"], sort=False)
          .agg(co=("co", "mean"), so_lan=("pr_auc", "size"), pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
               recall_1=("recall@1%", "mean"), giay_tb=("giay", "mean"), ram_mb=("ram_dinh_mb", "max"))
          .reset_index().sort_values(["du", "co"]))
    du = tt.loc[tt["du"], "pr_auc"]
    moc = float(du.iloc[0]) if len(du) else np.nan
    tt["pct_so_voi_du"] = 100 * tt["pr_auc"] / moc
    tt.drop(columns="du").to_csv(out / "tom_tat.csv", index=False, float_format="%.4f")

    # tiêu chí chốt B1: N nhỏ nhất đạt >= 95% "đủ"; phải có đủ 3 seed × 5 fold mới tính là chính thức
    dat = tt[(~tt["du"]) & (tt["pct_so_voi_du"] >= 100 * NGUONG)]
    chinh_thuc = bool(len(tt)) and (tt["so_lan"] >= len(SEEDS) * 5).all()
    if np.isnan(moc):
        de_xuat = "Chưa có kết quả 'đủ' (cỡ 0) → chưa so được. Chạy thêm --co 0."
    elif len(dat):
        r = dat.iloc[0]
        de_xuat = (f"N nhỏ nhất đạt >= {NGUONG:.0%} 'đủ': {int(r['co']):,} dòng fit "
                   f"(PR-AUC {r['pr_auc']:.4f} = {r['pct_so_voi_du']:.1f}% của {moc:.4f}).")
    else:
        r = tt[~tt["du"]].iloc[-1] if (~tt["du"]).any() else None
        de_xuat = (f"Chưa cỡ nào đạt {NGUONG:.0%}. Cỡ lớn nhất đã chạy {int(r['co']):,} đạt {r['pct_so_voi_du']:.1f}%."
                   " Nếu vẫn chọn 1 triệu vì lý do máy, ghi rõ % mất vào decisions.md (D26)." if r is not None else "Chưa có cỡ nào.")
    if not chinh_thuc:
        de_xuat += "\n(CHƯA chính thức: có cỡ chưa đủ 3 seed × 5 fold.)"
    (out / "de_xuat.txt").write_text(de_xuat + "\n", encoding="utf-8")

    # biểu đồ: một đường (trung bình qua seed × fold) + dải ± 1 độ lệch chuẩn; vạch ngang = 95% "đủ"
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x, yv, sd = tt["co"].to_numpy(), tt["pr_auc"].to_numpy(), tt["pr_auc_sd"].fillna(0).to_numpy()
    ax.fill_between(x, yv - sd, yv + sd, color=MAU, alpha=0.15, linewidth=0)
    ax.plot(x, yv, color=MAU, linewidth=2, marker="o", markersize=6)
    for xi, yi, nh in zip(x, yv, tt["nhan"]):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=CHU)
    if not np.isnan(moc):
        ax.axhline(NGUONG * moc, color=CHU, linewidth=1, linestyle="--")
        ax.text(x.min(), NGUONG * moc, f" 95% của đủ = {NGUONG * moc:.3f}", color=CHU, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tt["nhan"], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("số dòng fit giữ lại (giữ hết ca rửa tiền, có bù trọng số)", color=CHU)
    ax.set_ylabel("PR-AUC test", color=CHU)
    ax.set_title("Đường học B1 — trung bình qua seed × fold, dải = ± 1 độ lệch chuẩn", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LUOI)
    ax.tick_params(colors=CHU, labelsize=8)
    ax.grid(axis="y", color=LUOI, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out / "duong_hoc.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 160)
    print(tt.drop(columns="du").round(4).to_string(index=False))
    print("\n" + de_xuat)
    print(f"Đã ghi {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--co", type=int, nargs="+", default=[100_000, 250_000, 500_000, 1_000_000, 0],
                    help="cỡ phần fit (số dòng); 0 = đủ")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--ten", default="b1", help="tiền tố tên kết quả; đổi khi chạy thử để không lẫn với số chính thức")
    ap.add_argument("--giu-cache", action="store_true", help="không xoá data/processed/cach1/... sau mỗi cấu hình")
    ap.add_argument("--chi-tong-hop", action="store_true")
    args = ap.parse_args()
    if not args.chi_tong_hop:
        chay(args)
    tong_hop(args)


if __name__ == "__main__":
    main()
