"""B2 — Bảng client theo nước cho FL: nước của ngân hàng gửi → mã client. Chạy vài giây.

Quy tắc (mặc định, giống script 09 --client nuoc vì cùng dùng fraudfl.data.quoc_gia.client_theo_nuoc):
nước có < --gop-nho giao dịch (ngày 0–9) gộp chung một client '_khac'.

Ra   results/b2_client/client_theo_nuoc.csv   NUOC, SO_GIAO_DICH, SO_CA, SO_NGAN_HANG, CLIENT, MA_CLIENT
     results/b2_client/client.csv             mỗi client một dòng: MA_CLIENT, CLIENT, số nước, giao dịch, ca, ca fit/val/test fold 0
     results/b2_client/tom_tat.json

    python scripts/b2_client_theo_nuoc.py                 # gộp nước < 10.000 giao dịch
    python scripts/b2_client_theo_nuoc.py --gop-nho 50000 # thử ngưỡng khác (ghi vào thư mục riêng)
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from fraudfl.config import DATA_PROCESSED, DATA_SPLITS, IBM_MAX_DAY, RESULTS
from fraudfl.data.quoc_gia import GOP_NUOC_NHO, TEN_GOP, client_theo_nuoc
from fraudfl.khung import SEEDS, gan_fold, vai_tro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--gop-nho", type=int, default=GOP_NUOC_NHO, help="nước ít giao dịch hơn → gộp vào '_khac'")
    ap.add_argument("--min-ca-test", type=int, default=10, help="cảnh báo nếu client có ít ca hơn trong test fold 0")
    args = ap.parse_args()
    out = RESULTS / ("b2_client" if args.gop_nho == GOP_NUOC_NHO else f"b2_client_gop{args.gop_nho}")
    out.mkdir(parents=True, exist_ok=True)

    f = pd.read_parquet(DATA_PROCESSED / f"{args.stem}_features_bank.parquet", columns=["TX_ID", "FROM_BANK", "TX_TIME_DAYS", "LABEL"])
    f = f[f["TX_TIME_DAYS"] <= IBM_MAX_DAY].reset_index(drop=True)
    nuoc = pd.read_csv(DATA_SPLITS / "ibm_ngan_hang_quoc_gia.csv", dtype=str).set_index("BANK")["NUOC"]
    f["NUOC"] = nuoc.reindex(f["FROM_BANK"].astype(str)).to_numpy()

    bang = client_theo_nuoc(f["NUOC"], args.gop_nho)
    them = f.groupby("NUOC").agg(SO_CA=("LABEL", "sum"), SO_NGAN_HANG=("FROM_BANK", "nunique"))
    bang = bang.join(them, on="NUOC")[["NUOC", "SO_GIAO_DICH", "SO_CA", "SO_NGAN_HANG", "CLIENT", "MA_CLIENT"]]
    bang.to_csv(out / "client_theo_nuoc.csv", index=False)

    f["CLIENT"] = f["NUOC"].map(bang.set_index("NUOC")["CLIENT"])
    vt = vai_tro(gan_fold(f["TX_ID"].to_numpy(), SEEDS[0]), 0)
    for ten, m in vt.items():
        f[f"ca_{ten}"] = np.where(m, f["LABEL"], 0)
    cl = (f.groupby("CLIENT").agg(so_giao_dich=("LABEL", "size"), so_ca=("LABEL", "sum"),
                                  ca_fit_f0=("ca_fit", "sum"), ca_val_f0=("ca_val", "sum"), ca_test_f0=("ca_test", "sum"))
          .join(bang.groupby("CLIENT").agg(MA_CLIENT=("MA_CLIENT", "first"), so_nuoc=("NUOC", "size")))
          .reset_index().sort_values("MA_CLIENT"))
    cl["ty_le_pct"] = 100 * cl["so_ca"] / cl["so_giao_dich"]
    cl["pct_giao_dich"] = 100 * cl["so_giao_dich"] / cl["so_giao_dich"].sum()
    cl = cl[["MA_CLIENT", "CLIENT", "so_nuoc", "so_giao_dich", "pct_giao_dich", "so_ca", "ty_le_pct",
             "ca_fit_f0", "ca_val_f0", "ca_test_f0"]]
    cl.to_csv(out / "client.csv", index=False, float_format="%.4f")

    it_ca = cl.loc[cl["ca_test_f0"] < args.min_ca_test, "CLIENT"].tolist()
    lon = cl.iloc[0]
    tt = {"gop_nho": args.gop_nho, "so_client": int(len(cl)), "so_nuoc": int(len(bang)),
          "nuoc_gop_vao_khac": bang.loc[bang["CLIENT"] == TEN_GOP, "NUOC"].tolist(),
          "client_lon_nhat": {"ten": lon["CLIENT"], "pct_giao_dich": round(float(lon["pct_giao_dich"]), 2)},
          "client_it_hon_min_ca_test_f0": it_ca}
    (out / "tom_tat.json").write_text(json.dumps(tt, ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 160)
    print(cl.round(3).to_string(index=False))
    print(f"\n{len(cl)} client từ {len(bang)} nước; gộp vào '{TEN_GOP}': {', '.join(tt['nuoc_gop_vao_khac']) or '(không)'}")
    print(f"Client lớn nhất: {lon['CLIENT']} ({lon['pct_giao_dich']:.1f}% giao dịch)")
    if it_ca:
        print(f"⚠ {len(it_ca)} client có < {args.min_ca_test} ca trong test fold 0 (PR-AUC riêng của client đó sẽ dao động mạnh): {', '.join(it_ca)}")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
