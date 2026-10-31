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
| 4. Baseline dạng bảng (kiểm tra pipeline) | `python scripts/10_baseline_tabular.py` | `results/baseline_tabular/` (~1 phút) |
| Kiểm thử | `python -m pytest -q tests` | |

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
│   │   └── splits.py      # nested stratified K-fold + chia theo thời gian
│   └── eval/metrics.py    # PR-AUC, Recall@k, Precision@k, VDR@k, F1 — dùng chung, không tự tính lại
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

## Lộ trình

- [x] Tuần 1–2: dữ liệu synthetic, đặc trưng, chia fold, EDA, module chỉ số, baseline dạng bảng
- [ ] Tải IBM AML Small + Elliptic vào `data/raw/`, viết loader
- [ ] Dựng đồ thị giao dịch + GraphSAGE/MLP centralized (E1), local-only (E2)
- [ ] FedAvg mô phỏng 5 client (E3)
- [ ] DP: clip + nhiễu Gauss + RDP accountant (E4, E5)

## Nguồn tham khảo và giấy phép

Bộ sinh dữ liệu và đặc trưng dựa trên Le Borgne et al., *Reproducible Machine Learning for Credit Card
Fraud Detection — Practical Handbook* (2022), mã nguồn GPL-3.0 (bản sao ở `../Báo/Báo 1/fraud-detection-handbook`).
Vì mã ở đây phái sinh từ mã GPL-3.0, nếu công bố mã nguồn thì nên phát hành dưới GPL-3.0 và trích dẫn handbook.
