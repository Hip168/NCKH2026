"""B1 — Thống kê phân bổ dữ liệu IBM HI-Small (ngày 0–9). Không cần mô hình, chạy ~1 phút, RAM ~2 GB.

Đọc  data/processed/HI-Small_features_bank.parquet  (tính bằng scripts/06_ibm_features.py)
     data/splits/ibm_ngan_hang_quoc_gia.csv          (có sẵn trong repo, B0)
Ra   results/b1_thong_ke/
     theo_ngay.csv, theo_nuoc.csv, theo_ngan_hang_nhom.csv, theo_loai_thanh_toan.csv, tom_tat.json
     1_ty_le_theo_nuoc.png, 2_co_ngan_hang.png, 3_ty_le_theo_loai_thanh_toan.png

    python scripts/b1_thong_ke.py
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fraudfl.config import DATA_PROCESSED, DATA_SPLITS, IBM_MAX_DAY, RESULTS

MAU, CHU, LUOI = "#2a78d6", "#52514e", "#e6e5e1"
NHOM_CO = [(0, 1_000, "< 1k"), (1_000, 10_000, "1k–10k"), (10_000, 100_000, "10k–100k"), (100_000, np.inf, "≥ 100k")]


def bang_dem(df: pd.DataFrame, theo, ten: str) -> pd.DataFrame:
    g = df.groupby(theo)["LABEL"].agg(so_giao_dich="size", so_ca="sum").reset_index().rename(columns={theo: ten})
    g["ty_le_pct"] = 100 * g["so_ca"] / g["so_giao_dich"]
    g["pct_giao_dich"] = 100 * g["so_giao_dich"] / g["so_giao_dich"].sum()
    g["pct_ca"] = 100 * g["so_ca"] / g["so_ca"].sum()
    return g


def kieu_truc(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LUOI)
    ax.tick_params(colors=CHU, labelsize=9)
    ax.grid(axis="x", color=LUOI, linewidth=0.8)
    ax.set_axisbelow(True)


def bieu_do_thanh(bang: pd.DataFrame, nhan: str, gia_tri: str, tieu_de: str, truc_x: str, file, moc=None):
    b = bang.sort_values(gia_tri)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(b) + 1)))
    ax.barh(b[nhan].astype(str), b[gia_tri], color=MAU, height=0.7)
    if moc is not None:
        ax.axvline(moc, color=CHU, linewidth=1, linestyle="--")
        ax.text(moc, len(b) - 0.4, f" toàn bộ {moc:.3f}%", color=CHU, fontsize=8, va="bottom")
    for y_, v in enumerate(b[gia_tri]):
        ax.text(v, y_, f" {v:.3f}", va="center", fontsize=7, color=CHU)
    ax.set_title(tieu_de, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_xlabel(truc_x, color=CHU)
    kieu_truc(ax)
    fig.tight_layout()
    fig.savefig(file, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    args = ap.parse_args()
    out = RESULTS / "b1_thong_ke"
    out.mkdir(parents=True, exist_ok=True)

    file = DATA_PROCESSED / f"{args.stem}_features_bank.parquet"
    pf = [c for c in pq.read_schema(file).names if c.startswith("PF_")]
    f = pd.read_parquet(file,
                        columns=["TX_ID", "FROM_BANK", "TX_TIME_DAYS", "LABEL", *pf])
    f = f[f["TX_TIME_DAYS"] <= IBM_MAX_DAY].reset_index(drop=True)
    nuoc = pd.read_csv(DATA_SPLITS / "ibm_ngan_hang_quoc_gia.csv", dtype=str).set_index("BANK")["NUOC"]
    f["NUOC"] = nuoc.reindex(f["FROM_BANK"].astype(str)).to_numpy()
    if f["NUOC"].isna().any():
        raise SystemExit(f"{int(f['NUOC'].isna().sum())} giao dịch không tra được nước — kiểm tra lại B0")
    f["NGAY"] = np.floor(f["TX_TIME_DAYS"]).astype(int)
    toan_bo = 100 * f["LABEL"].mean()

    # 1. theo ngày
    ngay = bang_dem(f, "NGAY", "ngay")
    ngay.to_csv(out / "theo_ngay.csv", index=False, float_format="%.4f")

    # 2. theo nước (của ngân hàng gửi)
    nuoc_b = bang_dem(f, "NUOC", "nuoc")
    nuoc_b["so_ngan_hang"] = nuoc_b["nuoc"].map(f.groupby("NUOC")["FROM_BANK"].nunique())
    nuoc_b = nuoc_b.sort_values("so_giao_dich", ascending=False)
    nuoc_b.to_csv(out / "theo_nuoc.csv", index=False, float_format="%.4f")

    # 3. theo nhóm cỡ ngân hàng gửi
    co = f.groupby("FROM_BANK")["LABEL"].agg(n="size", ca="sum")
    rows = []
    for lo, hi, ten in NHOM_CO:
        m = (co["n"] >= lo) & (co["n"] < hi)
        rows.append({"nhom_co": ten, "so_ngan_hang": int(m.sum()), "so_giao_dich": int(co.loc[m, "n"].sum()),
                     "so_ca": int(co.loc[m, "ca"].sum()), "so_nh_co_ca": int((co.loc[m, "ca"] > 0).sum())})
    nh = pd.DataFrame(rows)
    nh["ty_le_pct"] = 100 * nh["so_ca"] / nh["so_giao_dich"]
    nh["pct_giao_dich"] = 100 * nh["so_giao_dich"] / nh["so_giao_dich"].sum()
    nh["pct_ca"] = 100 * nh["so_ca"] / nh["so_ca"].sum()
    nh.to_csv(out / "theo_ngan_hang_nhom.csv", index=False, float_format="%.4f")

    # 4. theo loại thanh toán (cột one-hot PF_*)
    loai = pd.Series(np.array(pf)[f[pf].to_numpy().argmax(1)]).str.replace("PF_", "", regex=False)
    if not (f[pf].sum(1) == 1).all():
        raise SystemExit("Cột PF_* không phải one-hot — kiểm tra lại scripts/06")
    f["LOAI"] = loai.to_numpy()
    lt = bang_dem(f, "LOAI", "loai_thanh_toan").sort_values("so_giao_dich", ascending=False)
    lt.to_csv(out / "theo_loai_thanh_toan.csv", index=False, float_format="%.4f")

    # biểu đồ
    bieu_do_thanh(nuoc_b, "nuoc", "ty_le_pct", "Tỷ lệ rửa tiền theo nước của ngân hàng gửi (%)", "% giao dịch là rửa tiền",
                  out / "1_ty_le_theo_nuoc.png", moc=toan_bo)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(np.log10(co["n"]), bins=40, color=MAU, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Cỡ ngân hàng gửi: {len(co):,} ngân hàng", loc="left", fontsize=11)
    ax.set_xlabel("số giao dịch (thang log10: 1 = 10, 3 = 1.000, 5 = 100.000)", color=CHU)
    ax.set_ylabel("số ngân hàng", color=CHU)
    kieu_truc(ax)
    ax.grid(axis="y", color=LUOI, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "2_co_ngan_hang.png", dpi=150)
    plt.close(fig)
    bieu_do_thanh(lt, "loai_thanh_toan", "ty_le_pct", "Tỷ lệ rửa tiền theo loại thanh toán (%)", "% giao dịch là rửa tiền",
                  out / "3_ty_le_theo_loai_thanh_toan.png", moc=toan_bo)

    # tóm tắt để viết báo cáo
    top3 = nuoc_b.nlargest(3, "so_ca")
    tt = {"so_giao_dich": int(len(f)), "so_ca": int(f["LABEL"].sum()), "ty_le_pct": round(toan_bo, 4),
          "so_ngan_hang": int(len(co)), "so_nuoc": int(len(nuoc_b)),
          "nuoc_nhieu_ca_nhat": top3[["nuoc", "so_ca", "pct_ca"]].round(2).to_dict("records"),
          "pct_ca_3_nuoc_dau": round(float(top3["pct_ca"].sum()), 2),
          "loai_ty_le_cao_nhat": lt.nlargest(1, "ty_le_pct")[["loai_thanh_toan", "ty_le_pct"]].round(4).to_dict("records")[0],
          "ty_le_ngay_min_max_pct": [round(float(ngay["ty_le_pct"].min()), 4), round(float(ngay["ty_le_pct"].max()), 4)]}
    (out / "tom_tat.json").write_text(json.dumps(tt, ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 160)
    print(f"{tt['so_giao_dich']:,} giao dịch, {tt['so_ca']:,} ca ({toan_bo:.3f}%), {tt['so_ngan_hang']:,} ngân hàng, {tt['so_nuoc']} nước\n")
    print("Theo ngày:\n" + ngay.round(3).to_string(index=False))
    print("\nTheo nước (10 nước nhiều giao dịch nhất):\n" + nuoc_b.head(10).round(3).to_string(index=False))
    print("\nTheo cỡ ngân hàng:\n" + nh.round(3).to_string(index=False))
    print("\nTheo loại thanh toán:\n" + lt.round(3).to_string(index=False))
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
