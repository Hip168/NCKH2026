# fraudfl — FL + DP cho phát hiện gian lận giao dịch ngân hàng

Mã nguồn prototype của đề tài NCKH 2026–2027. Kế hoạch tổng thể: [`../Báo/Map/presentation-plan_nckh.html`](../Báo/Map/presentation-plan_nckh.html).

## Cài đặt

```bash
pip install -r requirements.txt
```

Python 3.11. Đã kiểm tra trên Windows 11 với các phiên bản ghi trong `requirements.txt`.

## Chạy theo thứ tự

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Sinh dữ liệu synthetic 5 ngân hàng + đặc trưng | `python scripts/01_generate_synthetic.py` | `data/processed/*.parquet` (~15 giây) |
| 2. Chia fold cố định (chỉ chạy 1 lần) | `python scripts/02_make_splits.py` | `data/splits/fold_assignments_synthetic.csv` |
| 3. EDA | `python scripts/03_eda.py` | `results/eda/` |
| 3b. EDA IBM AML LI-Small | `python scripts/04_ibm_aml_eda.py` | `results/eda_ibm/` |
| 4. Baseline dạng bảng (kiểm tra pipeline) | `python scripts/10_baseline_tabular.py` | `results/baseline_tabular/` (~2 phút) |
| 5. GNN: E1 centralized, E2 local, E3 FedAvg | `python scripts/20_gnn_synthetic.py` | `results/gnn_synthetic/` (~2 giờ CPU; thử nhanh: `--folds 0`) |
| 6. E4/E5 DP mức bản ghi (MLP + DP-SGD), quét ε | `python scripts/30_dp_fl.py --mode record` | `results/dp_record/` (~1,5 giờ) |
| 7. DP mức client đúng Slide 4 (GNN), quét ε | `python scripts/30_dp_fl.py --mode client` | `results/dp_client/` (~25 phút, fold 0) |
| 8. Biểu đồ cho bài báo | `python scripts/40_figures.py` | `results/figures/` |
| Kiểm thử | `python -m pytest -q tests` | |

**Người mới vào nhóm: đọc [docs/BAO_CAO_TONG_HOP.md](docs/BAO_CAO_TONG_HOP.md) trước** (tiến độ, kết quả, phân công).
Chạy bằng Docker, không cần cài Python: [docs/huong_dan_docker.md](docs/huong_dan_docker.md).
Bảng số đầy đủ: [docs/results.md](docs/results.md).

Dữ liệu sinh ra là tất định (seed 42): ai chạy lại bước 1 cũng được đúng bộ dữ liệu đó.
File chia fold được commit vào git để mọi người dùng chung đúng một cách chia.

## Cấu trúc

```
project/
├── src/fraudfl/
│   ├── config.py          # đường dẫn, seed, hồ sơ 5 ngân hàng, k, số fold — sửa tham số ở đây
│   ├── data/
│   │   ├── simulate.py    # bộ sinh giao dịch non-IID nhiều ngân hàng
│   │   ├── features.py    # 15 đặc trưng dạng bảng, tính riêng trong từng ngân hàng
│   │   ├── splits.py      # nested stratified K-fold + chia theo thời gian
│   │   ├── prep.py        # nạp dữ liệu + fold, bỏ giai đoạn khởi động, chuẩn hóa riêng từng client
│   │   └── ibm_aml.py     # loader IBM AML (bản gốc Kaggle và bản đã định dạng)
│   ├── graph.py           # đồ thị giao dịch (cạnh quá khứ → hiện tại, cùng khách / cùng terminal)
│   ├── models.py          # GraphSAGE + MLP
│   ├── train.py           # huấn luyện full-batch + FedAvg
│   ├── privacy.py         # DP: client-level (cắt Δ + nhiễu), record-level (DP-SGD), tính ε
│   ├── experiment.py      # hàm dùng chung cho script thí nghiệm
│   └── eval/metrics.py    # PR-AUC, Recall@p, Precision@p, VDR@p, F1 — dùng chung, không tự tính lại
├── scripts/               # các bước chạy, đánh số theo thứ tự
├── tests/                 # kiểm thử đặc trưng và chỉ số
├── data/                  # raw/ và processed/ không commit; splits/ có commit
├── results/               # bảng và biểu đồ đầu ra
├── notebooks/             # notebook khám phá cá nhân
└── docs/decisions.md      # SỔ QUYẾT ĐỊNH — đọc trước khi làm
```

## Quy ước làm việc

- Mọi chỉ số tính qua `fraudfl.eval.metrics`. Mọi cách chia dữ liệu đọc từ `data/splits/`.
- Không chọn siêu tham số, ngưỡng hay chỉ số dựa trên tập test. Ngưỡng F1 chọn trên tập validation (inner fold).
- Thay đổi quyết định quan trọng (chỉ số, k, cách chia, tham số dữ liệu) → ghi vào `docs/decisions.md` trước.
- Code mới trong `src/` cần có test tương ứng trong `tests/`.
- **Lưu trữ kết quả từng giai đoạn** để viết báo cáo cả quá trình: không xóa kết quả cũ; mọi script gọi
  `fraudfl.runlog.archive_run` ở cuối (tự chép vào `results/_luu_tru/` và ghi `results/NHAT_KY_CHAY.csv`);
  hết mỗi giai đoạn thêm một mục vào [docs/nhat_ky_ket_qua.md](docs/nhat_ky_ket_qua.md). Chi tiết ở đầu tài liệu đó.

## Lộ trình

- [x] Tuần 1–2: dữ liệu synthetic, đặc trưng, chia fold, EDA, module chỉ số, baseline dạng bảng
- [x] Áp đề xuất D7–D9 (ngân sách cảnh báo 1%, đặc trưng tương đối + chuẩn hóa cục bộ, bỏ 30 ngày đầu)
- [x] Loader + EDA IBM AML LI-Small (bản đã định dạng)
- [x] Đồ thị giao dịch + GraphSAGE/MLP: E1 centralized, E2 local-only, E3 FedAvg
- [ ] Tải IBM AML bản gốc (có mã ngân hàng) + Elliptic; E7 centralized trên IBM
- [x] DP: clip + nhiễu Gauss + RDP accountant, 2 mức client / record (E4, E5)
- [ ] Chạy thêm 2 seed cho E1–E4 + kiểm định Wilcoxon, effect size (Slide 8b)
- [ ] FedProx / SCAFFOLD (E6)

## Nguồn tham khảo và giấy phép

Bộ sinh dữ liệu và đặc trưng dựa trên Le Borgne et al., *Reproducible Machine Learning for Credit Card
Fraud Detection — Practical Handbook* (2022), mã nguồn GPL-3.0 (bản sao ở `../Báo/Báo 1/fraud-detection-handbook`).
Vì mã ở đây phái sinh từ mã GPL-3.0, nếu công bố mã nguồn thì nên phát hành dưới GPL-3.0 và trích dẫn handbook.
