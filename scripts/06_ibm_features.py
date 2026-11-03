"""Tính đặc trưng IBM AML trên DỮ LIỆU ĐẦY ĐỦ (trước khi lấy mẫu, D17). Xem src/fraudfl/data/ibm_features.py.

Chạy:  python scripts/06_ibm_features.py --input data/raw/ibm_aml/HI-Small_Trans.csv [--view global|bank]
Ra:    data/processed/<tên file>_features_<view>.parquet   (không commit — sinh lại được)
       results/ibm_features_<view>/thong_ke.csv, tom_tat.json, ty_gia.csv   (commit + lưu trữ)
"""

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from fraudfl.config import DATA_PROCESSED, RESULTS
from fraudfl.data.ibm_aml import load_raw
from fraudfl.data.ibm_features import build_ibm_features, feature_columns, usd_rates
from fraudfl.runlog import archive_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--view", choices=["global", "bank"], default="global")
    args = ap.parse_args()

    t0 = time.time()
    df = load_raw(args.input)
    t_load = time.time() - t0
    feat = build_ibm_features(df, args.view)
    t_feat = time.time() - t0 - t_load

    stem = args.input.stem.replace("_Trans", "")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_parquet = DATA_PROCESSED / f"{stem}_features_{args.view}.parquet"
    feat.to_parquet(out_parquet, index=False)

    out = RESULTS / f"ibm_features_{args.view}"
    out.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(feat)
    feat.groupby("LABEL")[cols].mean().T.rename(columns={0: "TB_binh_thuong", 1: "TB_rua_tien"}).to_csv(
        out / "thong_ke.csv", float_format="%.4f")
    usd_rates(df).rename("don_vi_moi_USD").to_csv(out / "ty_gia.csv", float_format="%.6f")
    summary = {"input": args.input.name, "view": args.view, "n_tx": len(feat), "n_features": len(cols),
               "features": cols, "n_laundering": int(feat["LABEL"].sum()),
               "giay_doc": round(t_load), "giay_tinh_dac_trung": round(t_feat), "parquet": out_parquet.name}
    (out / "tom_tat.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "features"}, ensure_ascii=False, indent=2))
    dest = archive_run(out, note=f"Đặc trưng IBM {stem}, view={args.view}, {len(cols)} đặc trưng")
    print(f"Đã lưu trữ: {dest}")


if __name__ == "__main__":
    main()
