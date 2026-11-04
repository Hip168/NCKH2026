"""Cách 1 trên IBM HI-Small: học GỘP CHUNG bằng lô nhỏ trên TOÀN BỘ phần train, không lấy mẫu con.

Ý tưởng (đúng như mô tả gửi anh Hùng):
    Mỗi lượt (epoch): xáo trộn toàn bộ chỉ số train → đọc từng KHỐI (mặc định 100.000 dòng) từ ổ cứng vào RAM
    → trong khối, xáo lại và chia thành LÔ (mặc định 1.024 dòng) → mỗi lô sửa mô hình một bước (Adam).
    Hết lượt: chấm PR-AUC trên validation. Dừng sớm khi --patience lượt liền không tăng; giữ mô hình tốt nhất.
    Một mô hình duy nhất được sửa dần qua mọi lô. RAM khi học chỉ cần một khối + mô hình.

Để so được với kết quả chính của Đạt (results/ibm_fl_dp_CHINH_5fold_moc, "central", lấy mẫu r = 0,1):
    cùng 802 ngân hàng (--min-tx 1000), cùng chia 5 fold, cùng 20% validation, cùng chuẩn hóa theo ngân hàng,
    cùng MLP 22 đặc trưng view bank, cùng lr 5e-3. Khác: train dùng 100% ca bình thường (--rate 1) và dừng theo lượt.

Hai bước, tách riêng để đo RAM của bước học cho trung thực:
    1. Chuẩn bị (lần đầu của mỗi fold): đọc parquet đặc trưng, chia fold, chuẩn hóa, ghi ma trận ra .npy
       (data/processed/cach1/...). Bước này tốn RAM nhiều nhất vì đọc cả bảng.
    2. Học: mở .npy bằng memmap (không nạp cả file), chỉ chép từng khối vào RAM.

Chạy (từ thư mục gốc repo):
    python scripts/06_ibm_features.py --input <...>/HI-Small_Trans.csv --view bank    # một lần, nếu chưa có parquet
    python scripts/12_ibm_cach1_lo_nho.py --folds 0                        # 1 fold
    python scripts/12_ibm_cach1_lo_nho.py --folds 0 1 2 3 4 --tag 5fold    # đủ 5 fold
    python scripts/12_ibm_cach1_lo_nho.py --folds 0 --quet-khoi 50000 90000 100000 110000 200000 --tag do_khoi
        # bảng bằng chứng cỡ khối: mỗi cỡ chạy 1 lượt trong tiến trình riêng, đo giây và RAM đỉnh
    python scripts/12_ibm_cach1_lo_nho.py --folds 0 --thu-tu co_dinh --tag co_dinh
        # đối chứng: đọc khối theo thứ tự thời gian, không xáo (cách KHÔNG nên làm)
Ra:     results/ibm_cach1[_<tag>]/per_fold.csv, history.csv, log.txt (+ do_khoi.csv khi quét) và bản lưu trữ.
"""

import argparse
import copy
import json
import resource
import subprocess
import sys
import time

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from fraudfl.config import DATA_PROCESSED, IBM_MAX_DAY, RESULTS, SEED
from fraudfl.data.ibm_features import feature_columns
from fraudfl.data.sampling import subsample_negatives
from fraudfl.eval.metrics import evaluate
from fraudfl.models import TabMLP
from fraudfl.runlog import archive_run

CACHE = DATA_PROCESSED / "cach1"
MOC = RESULTS / "ibm_fl_dp_CHINH_5fold_moc" / "per_fold.csv"     # kết quả gộp chung có lấy mẫu của Đạt


def ram_dinh_mb() -> float:
    """RAM đỉnh của tiến trình này từ lúc chạy (macOS trả byte, Linux trả KB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def thu_muc_fold(args, k: int):
    return CACHE / f"{args.stem}_mintx{args.min_tx}_r{args.rate:g}_seed{args.seed}" / f"fold{k}"


# ---------------------------------------------------------------- bước 1: chuẩn bị (giống hệt cách chia của 09)
def chuan_bi(args, folds: list[int], log):
    can = [k for k in folds if not (thu_muc_fold(args, k) / "xong.json").exists()]
    if not can:
        return
    t0 = time.time()
    f = pd.read_parquet(DATA_PROCESSED / f"{args.stem}_features_bank.parquet")
    f = f[f["TX_TIME_DAYS"] <= IBM_MAX_DAY].reset_index(drop=True)
    size = f.groupby("FROM_BANK").size().sort_values(ascending=False, kind="stable")
    names = list(size[size >= args.min_tx].index)          # --min-tx 0: đủ mọi ngân hàng
    f = f[f["FROM_BANK"].isin(names)].reset_index(drop=True)
    lon = (size[f["FROM_BANK"]].to_numpy() >= 1000)          # ngân hàng lớn theo D20, để chấm riêng
    fc = feature_columns(f)
    X = f[fc].to_numpy(np.float32)
    y = f["LABEL"].to_numpy().astype(np.int64)
    amt, day = f["AMOUNT_USD"].to_numpy(), f["TX_TIME_DAYS"].to_numpy()
    cid = pd.Series(range(len(names)), index=names)[f["FROM_BANK"].to_numpy()].to_numpy()
    K = len(names)
    del f
    log(f"Chuẩn bị: {len(y):,} giao dịch, {int(y.sum()):,} ca, {K} ngân hàng, {len(fc)} đặc trưng "
        f"({time.time() - t0:.0f} giây đọc)")

    skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    cc_all = np.bincount(cid, weights=y, minlength=K)
    strat = np.where(cc_all[cid] >= 10, cid * 2 + y, -(y + 1))
    for k, (tr_all, te) in enumerate(skf.split(X, strat)):
        if k not in can:
            continue
        fit_all, val = train_test_split(tr_all, test_size=0.2, stratify=strat[tr_all], random_state=args.seed + k)
        if args.rate < 1:
            fit = subsample_negatives(fit_all, y[fit_all], cid[fit_all], args.rate, args.seed + k)["TX_ID"].to_numpy()
        else:
            fit = np.sort(fit_all)                 # giữ thứ tự thời gian trên đĩa; xáo trộn làm lúc học
        val, te = np.sort(val), np.sort(te)
        # chuẩn hóa z-score riêng từng ngân hàng trên phần fit (như 09). Ngân hàng có < --min-fit-chuan-hoa dòng fit
        # (hoặc không có dòng nào) không đủ để tự chuẩn hóa → dùng trung bình / độ lệch chuẩn chung của toàn bộ fit.
        mu_c, sd_c = X[fit].mean(0), X[fit].std(0)
        mu = np.tile(mu_c, (K, 1)).astype(np.float32)
        sd = np.tile(np.where(sd_c > 0, sd_c, 1.0), (K, 1)).astype(np.float32)
        g = pd.DataFrame(X[fit]).groupby(cid[fit])
        du = g.size()
        du = du[du >= args.min_fit_chuan_hoa].index.to_numpy()
        m_, s_ = g.mean().loc[du].to_numpy(np.float32), g.std(ddof=0).loc[du].to_numpy(np.float32)
        mu[du], sd[du] = m_, np.where(s_ > 0, s_, 1.0)
        d = thu_muc_fold(args, k)
        d.mkdir(parents=True, exist_ok=True)
        for ten, idx in (("fit", fit), ("val", val), ("test", te)):
            np.save(d / f"X_{ten}.npy", (X[idx] - mu[cid[idx]]) / sd[cid[idx]])
            np.save(d / f"y_{ten}.npy", y[idx].astype(np.float32))
        np.savez(d / "test_kem.npz", amt=amt[te], day=day[te], cid=cid[te], lon=lon[te])
        info = {"n_fit": len(fit), "n_val": len(val), "n_test": len(te), "ca_fit": int(y[fit].sum()),
                "ca_val": int(y[val].sum()), "ca_test": int(y[te].sum()), "dac_trung": fc, "K": K,
                "nh_tu_chuan_hoa": int(len(du))}
        (d / "xong.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  fold {k}: fit {len(fit):,} ({info['ca_fit']} ca) | val {len(val):,} ({info['ca_val']} ca) | "
            f"test {len(te):,} ({info['ca_test']} ca); {len(du):,}/{K:,} ngân hàng tự chuẩn hóa → {d}")
    log(f"Chuẩn bị xong sau {time.time() - t0:.0f} giây, RAM đỉnh {ram_dinh_mb():,.0f} MB")


# ---------------------------------------------------------------- bước 2: học theo khối + lô
@torch.no_grad()
def cham(model, Xm, khoi: int) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(np.array(Xm[i:i + khoi])))).numpy()
           for i in range(0, len(Xm), khoi)]
    return np.nan_to_num(np.concatenate(out), nan=0.0)


def mot_luot(model, opt, loss_fn, Xf, yf, args, rng) -> tuple[float, int]:
    """Một lượt qua toàn bộ phần fit. Trả về (loss trung bình, số bước cập nhật)."""
    model.train()
    n = len(yf)
    if args.thu_tu == "xao":
        thu_tu = rng.permutation(n)                # xáo toàn bộ, rồi mới chia khối
    else:
        thu_tu = np.arange(n)                      # đối chứng: theo thời gian, không xáo
    tong, buoc = 0.0, 0
    for j in range(0, n, args.khoi):
        idx = thu_tu[j:j + args.khoi]
        doc = np.sort(idx)                         # đọc đĩa theo thứ tự tăng cho nhanh ...
        xb = torch.from_numpy(np.asarray(Xf[doc]))  # ... chỉ khối này nằm trong RAM
        yb = torch.from_numpy(np.asarray(yf[doc]))
        trong = torch.from_numpy(rng.permutation(len(doc))) if args.thu_tu == "xao" else torch.arange(len(doc))
        for b in trong.split(args.batch_size):     # ... rồi xáo lại trong khối trước khi chia lô
            opt.zero_grad()
            loss = loss_fn(model(xb[b]), yb[b])
            loss.backward()
            opt.step()
            tong += loss.item()
            buoc += 1
    return tong / max(buoc, 1), buoc


def hoc_mot_fold(args, k: int, log, max_luot: int | None = None) -> tuple[dict, list[dict]]:
    d = thu_muc_fold(args, k)
    info = json.loads((d / "xong.json").read_text(encoding="utf-8"))
    mm = lambda ten: np.load(d / f"{ten}.npy", mmap_mode="r")
    Xf, yf, Xv, yv = mm("X_fit"), mm("y_fit"), mm("X_val"), mm("y_val")
    torch.manual_seed(args.seed + k)
    rng = np.random.default_rng(args.seed + k)
    model = TabMLP(Xf.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
    yv_np = np.asarray(yv)
    best, best_state, best_luot, xau, hist = -1.0, copy.deepcopy(model.state_dict()), 0, 0, []
    t_dau = time.time()
    for luot in range(1, (max_luot or args.max_epochs) + 1):
        t0 = time.time()
        loss, buoc = mot_luot(model, opt, loss_fn, Xf, yf, args, rng)
        t_hoc = time.time() - t0
        ap = float(average_precision_score(yv_np, cham(model, Xv, args.khoi)))
        hist.append({"fold": k, "luot": luot, "loss": loss, "buoc": buoc, "giay_hoc": t_hoc,
                     "giay_tong": time.time() - t0, "val_pr_auc": ap, "ram_dinh_mb": ram_dinh_mb()})
        log(f"    lượt {luot:2d}  loss {loss:.4f}  val PR-AUC {ap:.4f}  {buoc} bước  {t_hoc:.0f}s học  "
            f"RAM đỉnh {ram_dinh_mb():,.0f} MB")
        if np.isfinite(ap) and ap > best:
            best, best_state, best_luot, xau = ap, copy.deepcopy(model.state_dict()), luot, 0
        else:
            xau += 1
            if xau >= args.patience:
                break
    model.load_state_dict(best_state)
    kem = np.load(d / "test_kem.npz")
    yt = np.asarray(mm("y_test"))
    st = cham(model, mm("X_test"), args.khoi)
    r = evaluate(yt, st, kem["amt"], kem["day"])
    nhom = pd.Series(np.arange(len(yt))).groupby(kem["cid"]).agg(list)      # chỉ số test theo ngân hàng
    pc = [float(average_precision_score(yt[ix], st[ix])) for ix in map(np.array, nhom) if yt[ix].sum() > 0]
    row = {"fold": k, "setting": "cach1", "rate": args.rate, "thu_tu": args.thu_tu, "khoi": args.khoi,
           "batch": args.batch_size, "pos_weight": args.pos_weight, "lr": args.lr,
           "pr_auc": r["pr_auc"], "macro_pr_auc": float(np.mean(pc)), "recall@1%": r["recall@1%"],
           "vdr@1%": r["vdr@1%"], "val_pr_auc": best, "luot_tot": best_luot, "so_luot": len(hist),
           "n_fit": info["n_fit"], "ca_fit": info["ca_fit"], "giay": round(time.time() - t_dau),
           "ram_dinh_mb": round(ram_dinh_mb()), "so_nh": info["K"], "so_nh_co_ca_test": len(pc)}
    if "lon" in kem.files and (~kem["lon"]).any():           # chấm riêng ngân hàng lớn (>= 1.000 gd) và nhỏ
        for ten, m in (("nh_lon", kem["lon"]), ("nh_nho", ~kem["lon"])):
            if yt[m].sum() > 0:
                row[f"pr_auc_{ten}"] = float(average_precision_score(yt[m], st[m]))
                row[f"ca_test_{ten}"] = int(yt[m].sum())
    if MOC.exists() and args.min_tx == 1000:                 # chỉ so thẳng với Đạt khi cùng 802 ngân hàng
        moc = pd.read_csv(MOC)
        moc = moc[(moc["fold"] == k) & (moc["setting"] == "central")]
        if len(moc):
            row["central_lay_mau"] = float(moc["pr_auc"].iloc[0])
            row["pct_so_voi_lay_mau"] = 100 * r["pr_auc"] / row["central_lay_mau"]
    return row, hist


def quet_khoi(args, out, log):
    """Mỗi cỡ khối chạy 1 lượt trong tiến trình riêng (RAM đỉnh không bị lẫn với bước chuẩn bị)."""
    rows = []
    for kh in args.quet_khoi:
        cmd = [sys.executable, __file__, "--folds", str(args.folds[0]), "--khoi", str(kh), "--max-epochs", "1",
               "--batch-size", str(args.batch_size), "--rate", str(args.rate), "--min-tx", str(args.min_tx),
               "--seed", str(args.seed), "--pos-weight", str(args.pos_weight), "--lr", str(args.lr),
               "--_con", str(out / f"_con_{kh}.json")]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        log(f"  khối {kh:,}: chạy 1 lượt ...")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            rows.append({"khoi": kh, "loi": p.stderr.strip().splitlines()[-1] if p.stderr else f"mã {p.returncode}"})
            log(f"    LỖI: {rows[-1]['loi']}")
            continue
        r = json.loads((out / f"_con_{kh}.json").read_text(encoding="utf-8"))
        (out / f"_con_{kh}.json").unlink()
        r["ram_khoi_mb"] = kh * r["so_cot"] * 4 / 2**20
        rows.append(r)
        log(f"    {r['giay_hoc']:.1f}s học, {r['buoc']} bước, RAM đỉnh {r['ram_dinh_mb']:,.0f} MB "
            f"(riêng khối {r['ram_khoi_mb']:.1f} MB), val PR-AUC sau 1 lượt {r['val_pr_auc']:.4f}")
        pd.DataFrame(rows).to_csv(out / "do_khoi.csv", index=False, float_format="%.4f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="HI-Small")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-tx", type=int, default=1000, help="như D20: ngân hàng >= 1.000 giao dịch; 0 = đủ mọi ngân hàng")
    ap.add_argument("--min-fit-chuan-hoa", type=int, default=30, help="ngân hàng ít dòng fit hơn → dùng chuẩn hóa chung")
    ap.add_argument("--rate", type=float, default=1.0, help="1 = toàn bộ (Cách 1); < 1 = lấy mẫu con như Đạt, để đối chứng")
    ap.add_argument("--khoi", type=int, default=100_000, help="số dòng đọc vào RAM một lần")
    ap.add_argument("--batch-size", type=int, default=1024, help="số dòng mỗi bước sửa mô hình")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--thu-tu", choices=["xao", "co_dinh"], default="xao")
    ap.add_argument("--threads", type=int, default=0, help="số luồng CPU cho torch (0 = mặc định)")
    ap.add_argument("--quet-khoi", type=int, nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--_con", default="", help=argparse.SUPPRESS)      # dùng nội bộ khi quét khối
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args._con:                                  # tiến trình con của --quet-khoi: chỉ học 1 lượt rồi báo số
        row, hist = hoc_mot_fold(args, args.folds[0], lambda m: None, max_luot=1)
        h = hist[0]
        so_cot = np.load(thu_muc_fold(args, args.folds[0]) / "X_fit.npy", mmap_mode="r").shape[1]
        out = {"khoi": args.khoi, "batch": args.batch_size, "buoc": h["buoc"], "giay_hoc": h["giay_hoc"],
               "giay_tong": h["giay_tong"], "ram_dinh_mb": h["ram_dinh_mb"], "val_pr_auc": h["val_pr_auc"],
               "so_cot": so_cot, "luong": torch.get_num_threads()}
        with open(args._con, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        return

    out = RESULTS / ("ibm_cach1" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text("", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        with open(out / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    log(f"Cách 1 | {args.stem} ngày 0–{IBM_MAX_DAY} | ngân hàng >= {args.min_tx} gd | rate {args.rate:g} | "
        f"khối {args.khoi:,} | lô {args.batch_size} | lr {args.lr} | pos_weight {args.pos_weight} | "
        f"tối đa {args.max_epochs} lượt, dừng sau {args.patience} lượt không tăng | thứ tự {args.thu_tu} | "
        f"seed {args.seed} | {torch.get_num_threads()} luồng | ngân hàng < {args.min_fit_chuan_hoa} dòng fit dùng chuẩn hóa chung")
    chuan_bi(args, args.folds, log)

    if args.quet_khoi:
        log(f"\n=== Quét cỡ khối trên fold {args.folds[0]} ===")
        quet_khoi(args, out, log)
        log(f"Đã lưu trữ: {archive_run(out, note=f'Cách 1: quét cỡ khối {args.quet_khoi}, fold {args.folds[0]}')}")
        return

    rows, history = [], []
    for k in args.folds:
        log(f"\n=== Fold {k} ===")
        row, hist = hoc_mot_fold(args, k, log)
        rows.append(row)
        history += hist
        log(f"  TEST  PR-AUC {row['pr_auc']:.4f} | TB theo ngân hàng {row['macro_pr_auc']:.4f} | "
            f"Recall@1% {row['recall@1%']:.4f} | VDR@1% {row['vdr@1%']:.4f} | tốt nhất ở lượt {row['luot_tot']}/"
            f"{row['so_luot']} | {row['giay']}s"
            + (f" | gộp chung lấy mẫu {row['central_lay_mau']:.4f} → {row['pct_so_voi_lay_mau']:.0f}%"
               if "central_lay_mau" in row else ""))
        pd.DataFrame(rows).to_csv(out / "per_fold.csv", index=False, float_format="%.5f")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False, float_format="%.5f")

    df = pd.DataFrame(rows)
    cols = [c for c in ["pr_auc", "pr_auc_nh_lon", "pr_auc_nh_nho", "macro_pr_auc", "recall@1%", "vdr@1%",
                        "central_lay_mau", "pct_so_voi_lay_mau",
                        "luot_tot", "giay"] if c in df]
    log("\n=== Trung bình qua các fold ===")
    log(df[cols].agg(["mean", "std"]).round(4).to_string())
    log(f"Đã lưu trữ: {archive_run(out, note=f'Cách 1 lô nhỏ, rate {args.rate:g}, khối {args.khoi}, lô {args.batch_size}, pos_weight {args.pos_weight}, thứ tự {args.thu_tu}, folds {args.folds}')}")


if __name__ == "__main__":
    main()
