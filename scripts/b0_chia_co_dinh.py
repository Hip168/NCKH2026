"""B0 — tạo các file nền CỐ ĐỊNH cho mọi bước sau (D25). Chạy 1 lần, commit kết quả, rồi không chạy lại.

Ra (commit vào git):
    data/splits/ibm_hi_small_chia.parquet       TX_ID + fold_s42 / fold_s7 / fold_s2027 (ngày 0–9, đủ 30.470 ngân hàng)
    data/splits/ibm_ngan_hang_quoc_gia.csv      ngân hàng → nước (nếu có --accounts)
    data/splits/b0_tom_tat.json                 số dòng / số ca từng fold, mã băm để kiểm tra file không bị đổi

Chạy:
    python scripts/b0_chia_co_dinh.py --accounts <...>/HI-Small_accounts.csv
    python scripts/b0_chia_co_dinh.py --kiem-tra      # chỉ so mã băm với b0_tom_tat.json (dùng ở đầu mỗi bước)
"""

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from fraudfl.config import DATA_PROCESSED, DATA_SPLITS, IBM_MAX_DAY
from fraudfl.data.quoc_gia import bang_ngan_hang_quoc_gia
from fraudfl.khung import FILE_CHIA, N_FOLD, SEEDS, tao_chia, vai_tro

TOM_TAT = DATA_SPLITS / "b0_tom_tat.json"
FILE_NUOC = DATA_SPLITS / "ibm_ngan_hang_quoc_gia.csv"


def ma_bam(df: pd.DataFrame) -> str:
    """Băm theo nội dung (không theo byte file parquet, vì byte có thể khác giữa phiên bản pyarrow)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def kiem_tra() -> int:
    tt = json.loads(TOM_TAT.read_text(encoding="utf-8"))
    ok = True
    for ten, path in (("chia", FILE_CHIA), ("nuoc", FILE_NUOC)):
        if ten not in tt["ma_bam"]:
            continue
        d = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str)
        dung = ma_bam(d) == tt["ma_bam"][ten]
        ok &= dung
        print(f"{path.name}: {'KHỚP' if dung else 'KHÁC — file nền đã bị đổi!'}")
    dt = DATA_PROCESSED / f"{tt['stem']}_features_bank.parquet"
    if dt.exists():                   # đặc trưng tính lại ở máy khác (vd. Kaggle) phải ra đúng TX_ID và nhãn như lúc chốt
        f = pd.read_parquet(dt, columns=["TX_ID", "TX_TIME_DAYS", "LABEL"])
        f = f[f["TX_TIME_DAYS"] <= IBM_MAX_DAY]
        c = pd.read_parquet(FILE_CHIA, columns=["TX_ID"])["TX_ID"]
        dung = len(f) == tt["so_dong"] and int(f["LABEL"].sum()) == tt["so_ca"] and f["TX_ID"].sort_values().reset_index(drop=True).equals(c)
        ok &= dung
        print(f"{dt.name}: {'KHỚP' if dung else 'KHÁC'} ({len(f):,} giao dịch ngày 0–{IBM_MAX_DAY}, {int(f['LABEL'].sum()):,} ca)")
    else:
        print(f"(chưa có {dt.name} — tính bằng scripts/06_ibm_features.py rồi kiểm tra lại)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--accounts", default="", help="HI-Small_accounts.csv để tạo bảng ngân hàng → nước")
    ap.add_argument("--kiem-tra", action="store_true")
    ap.add_argument("--ghi-de", action="store_true", help="cho phép ghi đè file đã chốt (phải ghi lý do vào decisions.md)")
    args = ap.parse_args()
    if args.kiem_tra:
        sys.exit(kiem_tra())
    if FILE_CHIA.exists() and not args.ghi_de:
        sys.exit(f"{FILE_CHIA} đã có (B0 đã chốt). Kiểm tra: --kiem-tra. Ghi đè: --ghi-de (ghi lý do vào decisions.md).")

    f = pd.read_parquet(DATA_PROCESSED / f"{args.stem}_features_bank.parquet",
                        columns=["TX_ID", "FROM_BANK", "TX_TIME_DAYS", "LABEL"])
    f = f[f["TX_TIME_DAYS"] <= IBM_MAX_DAY]
    chia = tao_chia(f["TX_ID"].to_numpy(), f["LABEL"].to_numpy(), f["FROM_BANK"].to_numpy())
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    chia.to_parquet(FILE_CHIA, index=False, compression="zstd", compression_level=19)   # ~10 MB thay vì 27 MB

    y = f.set_index("TX_ID")["LABEL"].reindex(chia["TX_ID"]).to_numpy()
    tt = {"stem": args.stem, "ngay": f"0-{IBM_MAX_DAY}", "so_dong": len(chia), "so_ca": int(y.sum()),
          "so_ngan_hang": int(f["FROM_BANK"].nunique()), "seeds": list(SEEDS),
          "quy_tac": "fold k: test = k, val = (k+1) mod 5, fit = còn lại", "theo_seed": {}, "ma_bam": {"chia": ma_bam(chia)}}
    for s in SEEDS:
        fold = chia[f"fold_s{s}"].to_numpy()
        tt["theo_seed"][str(s)] = [{"fold": k, **{f"{p}_dong": int(m.sum()) for p, m in vai_tro(fold, k).items()},
                                    **{f"{p}_ca": int(y[m].sum()) for p, m in vai_tro(fold, k).items()}}
                                   for k in range(N_FOLD)]
    print(f"Chia: {len(chia):,} giao dịch, {int(y.sum()):,} ca, {tt['so_ngan_hang']:,} ngân hàng → {FILE_CHIA}")
    for r in tt["theo_seed"][str(SEEDS[0])]:
        print(f"  seed {SEEDS[0]} fold {r['fold']}: fit {r['fit_dong']:,} ({r['fit_ca']} ca) | "
              f"val {r['val_dong']:,} ({r['val_ca']} ca) | test {r['test_dong']:,} ({r['test_ca']} ca)")

    if args.accounts:
        nuoc = bang_ngan_hang_quoc_gia(args.accounts)
        nuoc.to_csv(FILE_NUOC, index=False)
        tt["ma_bam"]["nuoc"] = ma_bam(pd.read_csv(FILE_NUOC, dtype=str))
        g = f.merge(nuoc[["BANK", "NUOC"]], left_on="FROM_BANK", right_on="BANK", how="left")
        thieu = int(g["NUOC"].isna().sum())
        tt["so_nuoc"] = int(g["NUOC"].nunique())
        tt["dong_khong_co_nuoc"] = thieu
        print(f"Nước: {tt['so_nuoc']} nước, {thieu} giao dịch không tra được nước → {FILE_NUOC}")
    TOM_TAT.write_text(json.dumps(tt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Tóm tắt + mã băm → {TOM_TAT}")


if __name__ == "__main__":
    main()
