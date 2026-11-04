"""So sánh chạy full bộ dữ liệu với chia nhiều máy trên Elliptic và ULB Credit Card, có và không có DP.

Chạy:  python scripts/60_thi_nghiem_bo_cong_khai.py --bo elliptic --lap 0
       python scripts/60_thi_nghiem_bo_cong_khai.py --bo ulb --lap 0 --cau-hinh gop lk3
Ra:    artifacts/cong_khai/<bo>_lap<k>.csv

Hai bộ này không có ngân hàng, nên "máy" là chia giả định, theo 2 cách:
- đều: chia ngẫu nhiên giữ tỷ lệ gian lận, giống chia dữ liệu cho máy chạy nổi.
- lệch: giao dịch thường chia đều, ca gian lận chia theo Dirichlet alpha = 1, giống các ngân hàng khác nhau.
Chạy trong một tiến trình bằng đúng local_train và evaluate_split của bản Flower, chưa đi qua Docker.
- Elliptic: chỉ dùng 46.564 dòng có nhãn, chia theo thời gian. Mỗi lần lặp đổi seed.
- ULB: chia ngẫu nhiên giữ tỷ lệ 60/20/20. Mỗi lần lặp là một fold khác.
"""

import argparse
import time
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudfl.config import ROOT, SEED
from fraudfl.eval.metrics import evaluate
from fraudfl.fl.partition import Partition
from fraudfl.fl.task import evaluate_split, local_train, new_model

RAW = ROOT / "data" / "raw"
OUT = ROOT / "artifacts" / "cong_khai"
ROUNDS = 30

# tên: (kiểu, số client, epsilon, cách chia). epsilon = 0 là không DP.
CAU_HINH = {
    "gop": ("lk", 1, 0.0, "deu"), "gop_e1": ("lk", 1, 1.0, "deu"),
    "tu3": ("tu", 3, 0.0, "deu"), "tu3_e1": ("tu", 3, 1.0, "deu"),
    "lk3": ("lk", 3, 0.0, "deu"), "lk3_e2": ("lk", 3, 2.0, "deu"), "lk3_e1": ("lk", 3, 1.0, "deu"),
    "lk3_e05": ("lk", 3, 0.5, "deu"), "lk5": ("lk", 5, 0.0, "deu"), "lk5_e1": ("lk", 5, 1.0, "deu"),
    "tu3l": ("tu", 3, 0.0, "lech"), "lk3l": ("lk", 3, 0.0, "lech"), "lk3l_e1": ("lk", 3, 1.0, "lech"),
}


def load_elliptic() -> dict:
    cache = RAW / "elliptic" / "co_nhan.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    cls = pd.read_csv(RAW / "elliptic" / "elliptic_txs_classes.csv.zip")
    f = pd.read_csv(RAW / "elliptic" / "elliptic_txs_features.csv.zip", header=None)
    f = f.rename(columns={0: "txId", 1: "t"}).merge(cls, on="txId")
    f = f[f["class"] != "unknown"]
    out = {"x": f.drop(columns=["txId", "t", "class"]).to_numpy(np.float32),
           "y": (f["class"] == "1").to_numpy(np.float32), "t": f["t"].to_numpy(np.int32)}
    np.savez_compressed(cache, **out)
    return out


def load_ulb() -> dict:
    cache = RAW / "ulb" / "bang.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    path = RAW / "ulb" / "creditcard.arff"
    with open(path, encoding="utf-8") as fh:
        skip = next(i for i, line in enumerate(fh) if line.strip().lower() == "@data") + 1
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = pd.read_csv(path, skiprows=skip, header=None, names=cols, quotechar="'")
    x = df[[f"V{i}" for i in range(1, 29)]].to_numpy(np.float32)
    x = np.column_stack([x, np.log1p(df["Amount"].to_numpy(np.float32))])
    out = {"x": x, "y": df["Class"].astype(int).to_numpy(np.float32), "t": df["Time"].to_numpy(np.float32),
           "tien": df["Amount"].to_numpy(np.float32)}
    np.savez_compressed(cache, **out)
    return out


def chia_ba_phan(d: dict, bo: str, lap: int) -> np.ndarray:
    """0 = fit, 1 = val, 2 = test."""
    y = d["y"]
    if bo == "elliptic":
        return np.select([d["t"] <= 26, d["t"] <= 34], [0, 1], 2)
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, y))[lap]
    fit, val = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=SEED)
    out = np.empty(len(y), dtype=int)
    out[fit], out[val], out[te] = 0, 1, 2
    return out


def chia_client(y: np.ndarray, k: int, seed: int, cach: str = "deu") -> np.ndarray:
    rng = np.random.default_rng(seed)
    who = np.zeros(len(y), dtype=int)
    if k == 1:
        return who
    for c in (0.0, 1.0):
        idx = rng.permutation(np.nonzero(y == c)[0])
        ty_le = rng.dirichlet([1.0] * k) if (cach == "lech" and c == 1.0) else np.full(k, 1 / k)
        cuts = (np.cumsum(ty_le)[:-1] * len(idx)).astype(int)
        for i, part in enumerate(np.split(idx, cuts)):
            who[part] = i
    return who


def tao_client(d: dict, phan: np.ndarray, who: np.ndarray, k: int):
    parts, chi_so = [], []
    for c in range(k):
        i = np.nonzero(who == c)[0]
        x, ph = d["x"][i], phan[i]
        mu, sd = x[ph == 0].mean(0), x[ph == 0].std(0)
        x = (x - mu) / np.where(sd > 0, sd, 1.0)
        parts.append(Partition(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(d["y"][i]),
                               ph == 0, ph == 1, ph == 2))
        chi_so.append(i)
    return parts, chi_so


def diem_val(model, p) -> tuple[float, int]:
    n = int(p.val.sum())
    if n == 0 or p.y[p.val].sum() == 0:
        return 0.0, 0
    return evaluate_split(model, p, "val")["pr-auc"], n


def lk(parts, eps, lr, seed):
    """FedAvg theo số giao dịch đã học. Một client thì chính là chạy full."""
    d = parts[0].x.shape[1]
    st = new_model(d).state_dict()
    tong = sum(p.n_fit for p in parts)
    best, best_st, best_r = -1.0, st, 0
    for r in range(1, ROUNDS + 1):
        outs = []
        for i, p in enumerate(parts):
            m = new_model(d)
            m.load_state_dict(st)
            local_train(m, p, eps, ROUNDS, seed + 1000 * r + i, lr=lr)
            outs.append({k: v.clone() for k, v in m.state_dict().items()})
        st = {k: sum(o[k] * p.n_fit for o, p in zip(outs, parts)) / tong for k in st}
        m = new_model(d)
        m.load_state_dict(st)
        pairs = [diem_val(m, p) for p in parts]
        n = sum(b for _, b in pairs)
        v = sum(a * b for a, b in pairs) / max(n, 1)
        if np.isfinite(v) and v > best:
            best, best_st, best_r = v, st, r
    m = new_model(d)
    m.load_state_dict(best_st)
    return [m] * len(parts), best, best_r


def tu_hoc(parts, eps, lr, seed):
    """Mỗi client tự học, không gộp. Chọn vòng tốt nhất theo val của chính nó."""
    models, vals, rs = [], [], []
    for i, p in enumerate(parts):
        d = p.x.shape[1]
        m = new_model(d)
        best, best_st, best_r = -1.0, {k: v.clone() for k, v in m.state_dict().items()}, 0
        for r in range(1, ROUNDS + 1):
            local_train(m, p, eps, ROUNDS, seed + 1000 * r + i, lr=lr)
            v, _ = diem_val(m, p)
            if np.isfinite(v) and v > best:
                best, best_st, best_r = v, {k: x.clone() for k, x in m.state_dict().items()}, r
        k = new_model(d)
        k.load_state_dict(best_st)
        models.append(k)
        vals.append(best)
        rs.append(best_r)
    return models, float(np.mean(vals)), int(np.mean(rs))


@torch.no_grad()
def cham_test(models, parts, chi_so, d, bo) -> dict:
    ys, ss, ids = [], [], []
    for m, p, i in zip(models, parts, chi_so):
        m.eval()
        s = torch.nan_to_num(torch.sigmoid(m(p.x[p.test])), nan=0.0).numpy()
        ys.append(p.y[p.test].numpy())
        ss.append(s)
        ids.append(i[p.test])
    y, s, ids = np.concatenate(ys), np.concatenate(ss), np.concatenate(ids)
    tien = d["tien"][ids] if bo == "ulb" else np.ones(len(y))
    ngay = (d["t"][ids] // 86400).astype(int) if bo == "ulb" else d["t"][ids]
    r = evaluate(y, s, tien, ngay)
    return {"pr_auc": r["pr_auc"], "recall_1": r["recall@1%"], "vdr_1": r["vdr@1%"], "ty_le": r["fraud_rate"], "n_test": len(y)}


def do_tai(d, phan, k, lr):
    """Đo một máy: học 3 vòng trên phần dữ liệu lớn nhất của nó."""
    who = chia_client(d["y"], k, SEED)
    parts, _ = tao_client(d, phan, who, k)
    p = max(parts, key=lambda q: q.n_fit)
    m = new_model(p.x.shape[1])
    t0 = time.perf_counter()
    for r in range(3):
        local_train(m, p, 0.0, ROUNDS, SEED + r, lr=lr)
    print(f"k={k} n_fit={p.n_fit} giay_moi_vong={(time.perf_counter() - t0) / 3:.2f}")


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo", choices=["elliptic", "ulb"], required=True)
    ap.add_argument("--lap", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05, help="không DP")
    ap.add_argument("--lr-dp", type=float, default=0.5, help="có DP, chọn bằng điểm val")
    ap.add_argument("--cau-hinh", nargs="+", default=list(CAU_HINH))
    ap.add_argument("--do-tai", type=int, default=0, help="chỉ đo thời gian của một máy, với số máy này")
    ap.add_argument("--luu", default="")
    args = ap.parse_args()
    torch.set_num_threads(2)

    d = load_elliptic() if args.bo == "elliptic" else load_ulb()
    phan = chia_ba_phan(d, args.bo, args.lap)
    if args.do_tai:
        do_tai(d, phan, args.do_tai, args.lr)
        return
    seed = SEED + args.lap
    rows = []
    for ten in args.cau_hinh:
        kieu, k, eps, cach = CAU_HINH[ten]
        parts, chi_so = tao_client(d, phan, chia_client(d["y"], k, seed, cach), k)
        t0 = time.perf_counter()
        lr = args.lr_dp if eps > 0 else args.lr
        models, val, vong = (lk if kieu == "lk" else tu_hoc)(parts, eps, lr, seed)
        giay = time.perf_counter() - t0
        r = {"bo": args.bo, "lap": args.lap, "cau_hinh": ten, "lr": lr, "val": val, "vong_tot": vong, "giay": giay,
             "gian_lan_fit_it_nhat": int(min(p.y[p.fit].sum() for p in parts)),
             **cham_test(models, parts, chi_so, d, args.bo)}
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    if args.luu:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / args.luu, index=False)


if __name__ == "__main__":
    main()
