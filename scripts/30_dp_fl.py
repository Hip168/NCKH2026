"""E4 / E5: FL + Differential Privacy trên dữ liệu synthetic, quét ε (δ = 1e-5).

Hai chế độ (docs/decisions.md D12):
- client: GraphSAGE + FedAvg, mỗi ngân hàng cắt Δ về C = 1 và cộng nhiễu Gauss (đúng thiết kế Slide 4).
          Bảo vệ toàn bộ dữ liệu một ngân hàng. Chỉ có 5 client nên nhiễu rất lớn.
- record: MLP dạng bảng + FedAvg, mỗi ngân hàng train bằng DP-SGD (cắt gradient từng giao dịch, lấy mẫu
          Poisson). Bảo vệ từng giao dịch. ε = inf là cùng mô hình không DP để so sánh.

Chạy:  python scripts/30_dp_fl.py --mode record [--folds 0 1 2 3 4] [--eps inf 2 1 0.5]
       python scripts/30_dp_fl.py --mode client [--folds 0] [--eps 2 1 0.5]
Tùy chọn thêm (02/11, xem docs/luu_y_dp_muc_client.md):
       --seed S          seed huấn luyện (mặc định config.SEED); kết quả ghi vào thư mục có hậu tố _seedS
       --rounds R        số vòng FedAvg (mặc định 50 cho client, 30 cho record)
       --local-epochs E  số epoch cục bộ mỗi vòng ở chế độ client (mặc định 3)
       --noise server    chế độ client: ngân hàng chỉ cắt, server cộng nhiễu một lần (trung bình đều, std = zC/K)
       --small           chế độ client: GraphSAGE nhỏ 32 → 16 thay cho 128 → 64
       --model mlp       chế độ client: MLP dạng bảng (TabMLP, không dùng đồ thị) thay cho GraphSAGE — "cách B" trên
                         synthetic (docs/ghi_chu_cach_A_B.md)
       --tag T           hậu tố thư mục kết quả, ví dụ results/dp_client_T/
Ra:    results/dp_<mode>[_<tag>][_seed<S>]/per_fold.csv, summary.csv, history.csv, log.txt
"""

import argparse
import math
import time

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch

from fraudfl.config import RESULTS, SEED
from fraudfl.data.prep import fold_masks, load_synthetic, scale_per_client
from fraudfl.eval.metrics import MAIN_METRICS, best_f1_threshold
from fraudfl.experiment import make_data, scores, test_metrics
from fraudfl.graph import build_edge_index
from fraudfl.models import SageMLP, TabMLP
from fraudfl.privacy import CLIP, DELTA, add_server_noise, dpsgd_steps, noise_multiplier, privatize_update
from fraudfl.runlog import archive_run
from fraudfl.train import fedavg

# Client-level (GNN): giống E3 để so sánh trực tiếp
GNN_LR, GNN_ROUNDS, GNN_LOCAL_EPOCHS = 5e-3, 50, 3
# Record-level (MLP + DP-SGD). SGD momentum thay Adam: Adam chuẩn hóa theo phương sai gradient nên khuếch đại
# nhiễu DP (thử trên val fold 0, client 0: Adam + nhiễu PR-AUC 0,14 sau 3 epoch; SGD momentum 0,67)
MLP_LR, MLP_MOMENTUM, MLP_ROUNDS, MLP_LOCAL_EPOCHS = 0.5, 0.9, 30, 1
EXPECTED_BATCH = 1024
POS_WEIGHT = 10.0   # hằng số công khai; tính từ dữ liệu sẽ tốn thêm ngân sách riêng tư


def run_client_level(tx, x, y, masks, ei_local, clients, eps, opt):
    per_client = [make_data(x, y, masks, ei_local, np.flatnonzero(clients == c)) for c in np.unique(clients)]
    K = len(per_client)
    gen = torch.Generator().manual_seed(opt.seed)
    privatize = server_privatize = None
    if math.isinf(eps):
        z = 0.0
    else:
        # mỗi vòng = 1 cơ chế Gauss (không lấy mẫu con client) → sample_rate = 1, steps = số vòng
        z = noise_multiplier(eps, sample_rate=1.0, steps=opt.rounds)
        if opt.noise == "client":
            privatize = lambda local, global_: privatize_update(local, global_, CLIP, z, gen)
        else:
            privatize = lambda local, global_: privatize_update(local, global_, CLIP, 0.0, gen)   # chỉ cắt
            server_privatize = lambda avg, global_: add_server_noise(avg, global_, z * CLIP / K, gen)
    torch.manual_seed(opt.seed)
    if opt.model == "mlp":
        model = TabMLP(x.shape[1])
        per_client = [make_data(x, y, masks, None, d.node_ids) for d in per_client]   # bỏ cạnh đồ thị
    else:
        model = SageMLP(x.shape[1], hidden=32, emb=16) if opt.small else SageMLP(x.shape[1])
    model, hist = fedavg(model, per_client, GNN_LR, opt.rounds, opt.local_epochs, privatize=privatize,
                         server_privatize=server_privatize, uniform_weights=opt.noise == "server")
    return scores(model, per_client, len(tx)), hist, {"noise_multiplier": z}


def run_record_level(tx, x, y, masks, clients, eps, opt):
    MLP_ROUNDS = opt.rounds
    SEED = opt.seed
    per_client = [make_data(x, y, masks, None, np.flatnonzero(clients == c)) for c in np.unique(clients)]
    gen = torch.Generator().manual_seed(SEED)
    plan = {}
    for i, d in enumerate(per_client):
        q = min(1.0, EXPECTED_BATCH / d.n_fit)
        steps_round = max(1, round(MLP_LOCAL_EPOCHS / q))
        z = 0.0 if math.isinf(eps) else noise_multiplier(eps, sample_rate=q, steps=steps_round * MLP_ROUNDS)
        plan[i] = (q, steps_round, z)
    client_index = {id(d): i for i, d in enumerate(per_client)}

    def local_train(model, d):
        q, steps, z = plan[client_index[id(d)]]
        opt = torch.optim.SGD(model.parameters(), lr=MLP_LR, momentum=MLP_MOMENTUM)
        dpsgd_steps(model, d.x[d.fit], d.y[d.fit], opt, steps, q,
                    clip=CLIP if z > 0 else float("inf"), z=z, pos_weight=POS_WEIGHT, gen=gen)

    torch.manual_seed(SEED)
    model, hist = fedavg(TabMLP(x.shape[1]), per_client, MLP_LR, MLP_ROUNDS, MLP_LOCAL_EPOCHS,
                         local_train=local_train)
    zs = [p[2] for p in plan.values()]
    return scores(model, per_client, len(tx)), hist, {"noise_multiplier": float(np.mean(zs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["client", "record"], required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--eps", type=float, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--local-epochs", type=int, default=GNN_LOCAL_EPOCHS)
    ap.add_argument("--noise", choices=["client", "server"], default="client")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--model", choices=["sage", "mlp"], default="sage")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.rounds is None:
        args.rounds = GNN_ROUNDS if args.mode == "client" else MLP_ROUNDS
    folds = args.folds if args.folds is not None else ([0] if args.mode == "client" else [0, 1, 2, 3, 4])
    eps_list = args.eps if args.eps is not None else (
        [2.0, 1.0, 0.5] if args.mode == "client" else [math.inf, 2.0, 1.0, 0.5])

    name = f"dp_{args.mode}" + (f"_{args.tag}" if args.tag else "") + (f"_seed{args.seed}" if args.seed != SEED else "")
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text("", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        with open(out / "log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    tx = load_synthetic()
    y = tx["TX_FRAUD"].to_numpy()
    clients = tx["CLIENT_ID"].to_numpy()
    ei_local = build_edge_index(tx, cross_client=False) if args.mode == "client" else None
    log(f"Chế độ {args.mode}-level DP, δ = {DELTA}, C = {CLIP}, folds {folds}, ε {eps_list}, seed {args.seed}, "
        f"{args.rounds} vòng" + (f", {args.local_epochs} epoch cục bộ, nhiễu ở {args.noise}"
                                 f"{', GNN nhỏ' if args.small else ''}{', MLP (cách B)' if args.model == 'mlp' else ''}" if args.mode == "client" else ""))

    rows, history = [], []
    for k in folds:
        masks = fold_masks(tx, k)
        x = scale_per_client(tx, masks["fit"])
        log(f"\n=== Outer fold {k} ===")
        for eps in eps_list:
            t0 = time.time()
            if args.mode == "client":
                score, hist, extra = run_client_level(tx, x, y, masks, ei_local, clients, eps, args)
            else:
                score, hist, extra = run_record_level(tx, x, y, masks, clients, eps, args)
            thr = best_f1_threshold(y[masks["val"]], score[masks["val"]])
            res = test_metrics(tx, masks, score, np.full(len(tx), thr))
            best_round = max(hist, key=lambda h: h["val_pr_auc"])["round"]
            rows.append({"fold": k, "epsilon": eps, **res, **extra, "best_round": best_round,
                         "seconds": time.time() - t0})
            history += [{"fold": k, "epsilon": eps, **h} for h in hist]
            log(f"  ε={eps:<4} z={extra['noise_multiplier']:6.2f}  PR-AUC {res['pr_auc']:.4f} "
                f"(TB theo ngân hàng {res['macro_pr_auc']:.4f})  "
                f"Recall@1% {res['recall@1%']:.4f}  VDR@1% {res['vdr@1%']:.4f}  "
                f"(vòng tốt nhất {best_round}, {time.time() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(out / "per_fold.csv", index=False)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    per_fold = pd.DataFrame(rows)
    per_fold.groupby("epsilon")[MAIN_METRICS].agg(["mean", "std"]).to_csv(out / "summary.csv")
    log("\n=== Trung bình qua các fold ===")
    log(per_fold.groupby("epsilon")[MAIN_METRICS + ["noise_multiplier"]].mean().round(4).to_string())
    log(f"Đã lưu trữ: {archive_run(out, note=f'DP {args.mode}, folds {folds}, eps {eps_list}, seed {args.seed}')}")


if __name__ == "__main__":
    main()
