"""Đẩy MỌI kết quả đã có của nhóm vào MLflow để xem bằng giao diện web. Chỉ ĐỌC file kết quả, không chạy mô hình.

Không script thí nghiệm nào gọi MLflow; script này đọc csv/json đã lưu rồi ghi sang MLflow, nên không ảnh hưởng
thí nghiệm đang chạy. Chạy lại bao nhiêu lần cũng được:
- file kết quả nào đổi (theo thời điểm sửa file) thì các run của file đó được tạo lại;
- Cách 1 (results/ibm_cach1_*/history.csv) được ghi thêm từng lượt mới, xem được khi đang chạy.

Các nhóm (experiment) trong MLflow:
    1  IBM - Cach 1 lo nho (Dao)            results/ibm_cach1_*
    2  IBM - FL + DP muc client (Dat)       results/ibm_fl_dp*
    3  IBM - lay mau va dac trung (Dat)     results/ibm_danh_gia_lay_mau*, ibm_kiem_tra_dac_trung*
    4  Bang so voi muc tieu (Dat)           results/so_voi_muc_tieu*
    5  Synthetic - GNN, FL, DP (Dat)        results/gnn_synthetic, baseline_tabular, dp_client*, dp_record
    6  Synthetic - Flower Docker (Dao)      artifacts/tn/*, artifacts/<lần chạy Flower>/
    7  Elliptic va ULB (Dao)                artifacts/cong_khai/*.csv
    8  Hinh va EDA                          results/figures, eda, eda_ibm (ảnh + bảng, xem ở tab Artifacts)

Cần MLflow ở môi trường riêng (không cài vào .venv của repo):
    ~/.venvs/mlflow_nckh/bin/python scripts/13_dong_bo_mlflow.py --lap 15     # đồng bộ mỗi 15 giây
    ~/.venvs/mlflow_nckh/bin/mlflow ui --backend-store-uri sqlite:///$PWD/artifacts/mlflow/mlflow.db --port 5050
    mở http://127.0.0.1:5050 → bấm "Model training" → Experiments
"""

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "artifacts" / "mlflow"
DB = STORE / "mlflow.db"

E1 = "1. IBM - Cach 1 lo nho (Dao)"
E2 = "2. IBM - FL + DP muc client (Dat)"
E3 = "3. IBM - lay mau va dac trung (Dat)"
E4 = "4. Bang so voi muc tieu (Dat)"
E5 = "5. Synthetic - GNN, FL, DP (Dat)"
E6 = "6. Synthetic - Flower Docker (Dao)"
E7 = "7. Elliptic va ULB (Dao)"
E8 = "8. Hinh va EDA"
STEP_COLS = ("round", "luot")
BANG_UU_TIEN = ("per_fold.csv", "mo_hinh_thu.csv", "bang.csv", "do_khoi.csv")


def ten(s) -> str:
    """Tên hợp lệ cho MLflow: bỏ dấu tiếng Việt, @ → _at_, % → pct."""
    s = str(s).replace("@", "_at_").replace("%", "pct").replace("đ", "d").replace("Đ", "D").replace("ε", "eps")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^0-9A-Za-z_\-. /]", "_", s).strip("_") or "x"


def la_so(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and np.isfinite(v)


class DongBo:
    def __init__(self):
        STORE.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{DB}")
        self.c = MlflowClient()
        self.exp = {}
        self.runs = {}

    def exp_id(self, name: str) -> str:
        if name not in self.exp:
            e = self.c.get_experiment_by_name(name)
            self.exp[name] = e.experiment_id if e else self.c.create_experiment(
                name, artifact_location=(STORE / "artifacts").as_uri())
            self.runs[name] = {r.data.tags.get("nguon"): r for r in
                               self.c.search_runs([self.exp[name]], max_results=50000)}
        return self.exp[name]

    def tao_run(self, exp: str, nguon: str, ten_run: str, row: dict, mtime: float, history=None,
                artifacts=(), xong=True, tags=None) -> str:
        """Tạo một run: số → metric, chữ → param; history (DataFrame có cột bước) → metric theo bước."""
        eid = self.exp_id(exp)
        run = self.c.create_run(eid, run_name=ten_run,
                                tags={"nguon": nguon, "mtime": str(mtime), **(tags or {})})
        rid = run.info.run_id
        now = int(time.time() * 1000)
        ms, ps = [], []
        for k, v in row.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if la_so(v):
                ms.append(Metric(ten(k), float(v), now, 0))
            else:
                ps.append(Param(ten(k), str(v)[:500]))
        if history is not None and len(history):
            step = next(c for c in STEP_COLS if c in history)
            for _, h in history.iterrows():
                for k, v in h.items():
                    if k != step and la_so(v) and k not in ("fold",):
                        ms.append(Metric("theo_vong/" + ten(k), float(v), now, int(h[step])))
        uniq = {}
        for p in ps:
            uniq.setdefault(p.key, p)
        ps = list(uniq.values())
        for i in range(0, max(len(ms), len(ps), 1), 900):
            self.c.log_batch(rid, metrics=ms[i:i + 900], params=ps[i:i + 100] if i == 0 else [])
        if len(ps) > 100:
            for i in range(100, len(ps), 100):
                self.c.log_batch(rid, params=ps[i:i + 100])
        for a in artifacts:
            if Path(a).exists():
                self.c.log_artifact(rid, str(a))
        if xong:
            self.c.set_terminated(rid, "FINISHED")
        self.runs[exp][nguon] = self.c.get_run(rid)
        return rid

    def can_lam(self, exp: str, prefix: str, mtime: float) -> bool:
        """True nếu chưa có run của nguồn này hoặc file đã đổi (xóa run cũ)."""
        self.exp_id(exp)
        cu = [r for n, r in self.runs[exp].items() if n and n.startswith(prefix)]
        if cu and all(r.data.tags.get("mtime") == str(mtime) for r in cu):
            return False
        for r in cu:
            self.c.delete_run(r.info.run_id)
            del self.runs[exp][r.data.tags["nguon"]]
        return True

    # ------------------------------------------------------------ kết quả dạng bảng trong results/
    def bang_results(self, d: Path, exp: str):
        bang = next((d / b for b in BANG_UU_TIEN if (d / b).exists()), None)
        if bang is None:
            return 0
        mtime = bang.stat().st_mtime
        prefix = f"{d.relative_to(ROOT)}/{bang.name}#"
        if not self.can_lam(exp, prefix, mtime):
            return 0
        t = pd.read_csv(bang)
        hp = next((d / h for h in ("history.csv", "fedavg_history.csv") if (d / h).exists()), None)
        hist = pd.read_csv(hp) if hp else None
        tag = d.name
        n = 0
        for i, r in t.iterrows():
            row = {"thu_muc": tag, **r.to_dict()}
            h = None
            if hist is not None:
                step = next((c for c in STEP_COLS if c in hist), None)
                key = [c for c in hist.columns if c in t.columns and c not in (step, "val_pr_auc")]
                if step and key:
                    m = np.ones(len(hist), bool)
                    for c in key:
                        m &= hist[c].astype(str).to_numpy() == str(r[c])
                    if "setting" in t.columns and "setting" not in hist.columns:
                        m &= "fedavg" in str(r["setting"])
                    h = hist[m] if m.any() else None
            nhan = " ".join(str(r[c]) for c in ("setting", "variant", "model", "epsilon", "r_train", "test", "view",
                                                 "Cách", "khoi") if c in t.columns and pd.notna(r[c]))
            fold = f" f{int(r['fold'])}" if "fold" in t.columns and pd.notna(r["fold"]) else ""
            self.tao_run(exp, f"{prefix}{i}", f"{tag} | {nhan}{fold}".strip(" |"), row, mtime, h,
                         artifacts=[d / "log.txt"] if i == 0 else [], tags={"thu_muc": tag})
            n += 1
        return n

    # ------------------------------------------------------------ Cách 1: ghi dần từng lượt
    def cach1(self, d: Path):
        h_path, pf_path = d / "history.csv", d / "per_fold.csv"
        if not h_path.exists():
            return self.bang_results(d, E1) if (d / "do_khoi.csv").exists() else 0
        try:
            h = pd.read_csv(h_path)
        except Exception:
            return 0
        pf = pd.read_csv(pf_path) if pf_path.exists() else pd.DataFrame()
        dau = (d / "log.txt").read_text(encoding="utf-8").splitlines()[:1]
        exp = E1
        self.exp_id(exp)
        n = 0
        for k, hk in h.groupby("fold"):
            nguon = f"{d.relative_to(ROOT)}#fold{k}"
            run = self.runs[exp].get(nguon)
            if run is None:
                run = self.c.create_run(self.exp[exp], run_name=f"{d.name.removeprefix('ibm_cach1_')} f{k}",
                                        tags={"nguon": nguon, "thu_muc": d.name})
                if dau:
                    self.c.log_param(run.info.run_id, "cau_hinh", dau[0][:500])
                self.runs[exp][nguon] = run
            rid = run.info.run_id
            if run.info.status == "FINISHED":
                continue
            da = int(run.data.metrics.get("_luot_da_ghi", 0))
            now = int(time.time() * 1000)
            ms = []
            for _, r in hk[hk["luot"] > da].iterrows():
                for c in ["val_pr_auc", "loss", "giay_hoc", "ram_dinh_mb", "buoc"]:
                    ms.append(Metric("theo_vong/" + c, float(r[c]), now, int(r["luot"])))
                ms.append(Metric("_luot_da_ghi", float(r["luot"]), now, 0))
                n += 1
            if ms:
                self.c.log_batch(rid, metrics=ms)
                self.c.log_artifact(rid, str(d / "log.txt"))      # log đang chạy: mở tab Artifacts để đọc
            xong = pf[pf["fold"] == k] if len(pf) else pf
            if len(xong):
                r = xong.iloc[0].to_dict()
                self.c.log_batch(rid, metrics=[Metric(ten(a), float(v), now, 0) for a, v in r.items() if la_so(v)],
                                 params=[Param(ten(a), str(v)) for a, v in r.items()
                                         if not la_so(v) and pd.notna(v)][:100])
                self.c.log_artifact(rid, str(d / "log.txt"))
                self.c.set_terminated(rid, "FINISHED")
            self.runs[exp][nguon] = self.c.get_run(rid)
        return n

    # ------------------------------------------------------------ Flower Docker trên synthetic (artifacts/)
    def flower(self):
        n = 0
        tong = ROOT / "artifacts" / "tn" / "tong_hop.csv"
        th = pd.read_csv(tong).set_index("cau_hinh") if tong.exists() else pd.DataFrame()
        for tt in sorted(ROOT.glob("artifacts/**/tom_tat.json")):
            d = tt.parent
            if "mlflow" in d.parts or "cong_khai" in d.parts:
                continue
            mtime = max(p.stat().st_mtime for p in [tt, *([tong] if d.parent.name == "tn" else [])])
            nguon = f"{d.relative_to(ROOT)}#"
            if not self.can_lam(E6, nguon, mtime):
                continue
            row = {"lan_chay": str(d.relative_to(ROOT / "artifacts")), **json.loads(tt.read_text(encoding="utf-8"))}
            if d.parent.name == "tn" and d.name in th.index:
                row.update({f"bang_{k}": v for k, v in th.loc[d.name].to_dict().items()})
            h = pd.read_csv(d / "rounds.csv") if (d / "rounds.csv").exists() else None
            self.tao_run(E6, nguon, row["lan_chay"], row, mtime, h)
            n += 1
        return n

    def cong_khai(self):
        n = 0
        for f in sorted(ROOT.glob("artifacts/cong_khai/*.csv")):
            mtime = f.stat().st_mtime
            prefix = f"{f.relative_to(ROOT)}#"
            if not self.can_lam(E7, prefix, mtime):
                continue
            for i, r in pd.read_csv(f).iterrows():
                self.tao_run(E7, f"{prefix}{i}", f"{r['bo']} {r['cau_hinh']} lap{r['lap']}", r.to_dict(), mtime)
                n += 1
        return n

    def hinh(self):
        n = 0
        for d in ("figures", "eda", "eda_ibm"):
            p = ROOT / "results" / d
            if not p.exists():
                continue
            files = sorted(x for x in p.iterdir() if x.suffix in (".png", ".csv"))
            mtime = max(x.stat().st_mtime for x in files)
            if not self.can_lam(E8, f"results/{d}#", mtime):
                continue
            self.tao_run(E8, f"results/{d}#", d, {"so_file": len(files)}, mtime, artifacts=files)
            n += 1
        return n

    def tat_ca(self) -> dict:
        dem = {"cach1": 0, "bang": 0}
        for d in sorted(ROOT.glob("results/*/")):
            name = d.name
            try:
                if name.startswith("ibm_cach1"):
                    dem["cach1"] += self.cach1(d)
                elif name.startswith("ibm_fl_dp"):
                    dem["bang"] += self.bang_results(d, E2)
                elif name.startswith(("ibm_danh_gia_lay_mau", "ibm_kiem_tra_dac_trung")):
                    dem["bang"] += self.bang_results(d, E3)
                elif name.startswith("so_voi_muc_tieu"):
                    dem["bang"] += self.bang_results(d, E4)
                elif name.startswith(("gnn_synthetic", "baseline_tabular", "dp_client", "dp_record")):
                    dem["bang"] += self.bang_results(d, E5)
            except Exception as e:                     # một thư mục lỗi không làm dừng cả lượt
                print(f"  bỏ qua {name}: {e}", flush=True)
        dem["flower"] = self.flower()
        dem["cong_khai"] = self.cong_khai()
        dem["hinh"] = self.hinh()
        return dem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lap", type=float, default=0, help="> 0: đồng bộ lặp lại mỗi N giây")
    args = ap.parse_args()
    db = DongBo()
    while True:
        print(f"{time.strftime('%H:%M:%S')} {db.tat_ca()}", flush=True)
        if args.lap <= 0:
            break
        time.sleep(args.lap)


if __name__ == "__main__":
    main()
