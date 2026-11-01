# Báo cáo tổng hợp & bàn giao — cập nhật 2026-11-01

Tài liệu này dành cho cả nhóm (A–E). Đọc hết trong khoảng 15 phút. Sau khi đọc, mỗi người biết: dự án đang ở
đâu, kết quả ra sao, chạy lại thế nào, và **việc của mình là gì**.

Tài liệu liên quan:
- [results.md](results.md): bảng số đầy đủ.
- [decisions.md](decisions.md): sổ quyết định, có các đề xuất D7–D13 **chờ duyệt**.
- [../README.md](../README.md): cài đặt và lệnh chạy.

---

## 1. Tóm tắt trong 5 dòng

1. Pipeline đã chạy trọn: sinh dữ liệu 5 ngân hàng, tạo đặc trưng, chia fold, rồi GNN với 3 chế độ (centralized,
   local, FedAvg) và DP. Cả nhóm dùng chung một bộ chỉ số.
2. **FedAvg đạt 97,6% hiệu năng centralized** (PR-AUC 0,715 so với 0,733) và hơn local-only ở cả 5/5 fold.
3. **FL + DP mức bản ghi ở ε = 1 đạt 89,6% centralized**, đạt tiêu chí "≥ 85%" của Slide 10.
4. **Thiết kế DP ở Slide 4 (DP mức client) làm mô hình sụp** (PR-AUC 0,028). Vì vậy đề xuất đổi sang DP mức bản ghi;
   cần cả nhóm duyệt (D12).
5. Còn thiếu:
   - 2 seed nữa và kiểm định thống kê.
   - E6 FedProx.
   - Thí nghiệm trên dữ liệu thật IBM AML / Elliptic (E7–E10).

## 2. Tiến độ so với kế hoạch (Slide 9)

| Thí nghiệm | Nội dung | Trạng thái |
|---|---|---|
| E1 | Centralized, synthetic | ✅ Xong (5 fold, 1 seed) |
| E2 | Local-only | ✅ Xong |
| E3 | FedAvg không DP | ✅ Xong |
| E4 | FL + DP ε = 1 | ✅ Xong, dùng DP mức bản ghi, **mô hình MLP** (xem mục 4.3) |
| E5 | Quét ε ∈ {∞; 2; 1; 0,5} | ✅ Xong |
| E6 | FedProx / SCAFFOLD | ⬜ Chưa làm |
| E7 | Centralized trên IBM AML | ⬜ Đã có loader và EDA cho LI-Small, chưa train |
| E8 | FL + DP trên IBM AML | ⬜ Cần bản gốc có mã ngân hàng (Q5) |
| E9 | Cross-dataset | ⬜ |
| E10 | Elliptic | ⬜ Chưa tải dữ liệu |

Các thí nghiệm bắt buộc theo Slide 10 (E1–E4) đã có kết quả. Phần còn lại (E5–E10) có thể cắt nếu thiếu thời gian.

## 3. Kết quả chính

Điều kiện: dữ liệu synthetic, 5 ngân hàng × khoảng 100 nghìn giao dịch, 5 outer fold, seed 42.
Chỉ số chính là PR-AUC. Recall@1% và VDR@1% nghĩa là chuyên viên kiểm tra 1% giao dịch mỗi ngày.

### RQ1 — FL so với centralized và local-only (GraphSAGE + MLP)

| | PR-AUC | Recall@1% | VDR@1% | % centralized |
|---|---|---|---|---|
| E1 Centralized | 0,733 ± 0,014 | 0,548 | 0,799 | 100% |
| E3 FedAvg | 0,715 ± 0,013 | 0,539 | 0,793 | 97,6% |
| E2 Local-only | 0,706 ± 0,011 | 0,535 | 0,786 | 96,3% |

### RQ2 — Đánh đổi riêng tư – hiệu năng (DP-SGD mức bản ghi, MLP, FedAvg)

| ε (δ = 1e-5) | PR-AUC | VDR@1% | % centralized |
|---|---|---|---|
| ∞ (không DP) | 0,708 | 0,802 | 96,6% |
| 2 | 0,672 | 0,787 | 91,7% |
| **1** | **0,657** | **0,780** | **89,6%** |
| 0,5 | 0,625 | 0,778 | 85,3% |

Với DP mức client (đúng thiết kế Slide 4), PR-AUC chỉ đạt 0,028 ở mọi ε ≤ 2, tức là mô hình không học được gì.

### Đối chiếu tiêu chí thành công (Slide 10)

| Tiêu chí | Yêu cầu | Hiện tại | |
|---|---|---|---|
| Số client | ≥ 3 | 5 | ✅ |
| PR-AUC | ≥ 0,70 | 0,733 (E1); 0,715 (FedAvg) | ✅ |
| VDR | ≥ 0,80 | 0,799 (E1); 0,78 (FL+DP) | ⚠️ sát ngưỡng |
| FL+DP / centralized | ≥ 85% | 89,6% (synthetic) | ✅ synthetic, ⬜ IBM |
| Bảo mật | ε ≤ 1; δ ≤ 1e-5 | ε = 1; δ = 1e-5 | ✅ |
| Biểu đồ | ≥ 3 | 3 (`results/figures/`) | ✅ (latency chưa có) |

Biểu đồ đã có trong [../results/figures/](../results/figures/):
- `e1_e3_pr_auc_by_fold.png`: so sánh E1, E2, E3 theo từng fold.
- `privacy_utility.png`: PR-AUC theo ε.
- `fedavg_convergence.png`: đường hội tụ FedAvg.

## 4. Những điều cả nhóm phải biết (khác với slide)

### 4.1 Đã thay đổi so với kế hoạch, chờ duyệt

Chi tiết và lý do của từng mục nằm trong [decisions.md](decisions.md).

| # | Thay đổi | Vì sao |
|---|---|---|
| D7 | Thay k = 100 cảnh báo/ngày bằng **1% giao dịch/ngày** | Với k = 100, mô hình nào cũng có VDR ≈ 0,94 nên không phân biệt được |
| D8 | Thêm đặc trưng tỷ lệ tiền / chi tiêu 30 ngày; chuẩn hóa riêng từng ngân hàng | Quy mô tiền của các ngân hàng khác nhau |
| D9 | Bỏ 30 ngày đầu khỏi đánh giá | Đặc trưng cửa sổ 30 ngày chưa ổn định |
| D10 | GNN huấn luyện full-batch, lr = 5e-3, FedAvg 50 vòng | Trên Windows thiếu thư viện lấy mẫu láng giềng; lr = 1e-3 hội tụ quá chậm |
| D11 | Đồ thị: nút là giao dịch; cạnh nối cùng khách / cùng terminal, chỉ theo chiều quá khứ → hiện tại | Không dùng thông tin tương lai |
| D12 | **DP mức bản ghi** thay cho DP mức client | DP mức client với 5 ngân hàng làm mô hình sụp |
| D13 | DP-SGD dùng SGD momentum, không dùng Adam | Adam khuếch đại nhiễu DP |

### 4.2 Slide cần sửa

- **Slide 4**: đổi cơ chế DP sang mức bản ghi (D12).
- **Slide 5**: số liệu IBM AML chưa khớp với bộ đang có (Q5).
- **Slide 6**: k = 100 → 1%; lr và batch theo D10.
- **Slide 7**: thay "kết quả kỳ vọng" bằng số thật. Local-only đạt 96% centralized, không phải khoảng 60% như dự kiến.

### 4.3 Hạn chế phải ghi trong bài báo

- Kết quả DP dùng **MLP dạng bảng, không phải GNN**. DP-SGD không áp được trực tiếp cho GNN vì gradient của một
  nút phụ thuộc vào các nút láng giềng. Có thể nêu node-level DP cho GNN là hướng mở rộng.
- Đặc trưng cửa sổ trộn thông tin của nhiều giao dịch, nên ε chỉ đúng khi coi mỗi hàng đặc trưng là một bản ghi.
- Chưa có SecAgg (secure aggregation, cơ chế để server không đọc được cập nhật riêng của từng ngân hàng).
- Mới chạy 1 seed. Với 5 cặp, kiểm định Wilcoxon không thể đạt p < 0,05.
- FedAvg chưa bão hòa hẳn ở vòng 50.
- Bộ sinh dữ liệu phái sinh từ mã GPL-3.0 (Le Borgne et al.). Nếu công bố mã nguồn thì phải dùng GPL-3.0 và trích dẫn.

## 5. Bắt đầu làm việc (cho người mới vào repo)

```bash
# 1. Lấy mã. Repo private: nhờ chủ repo (trantatdat06) mời làm collaborator trên GitHub trước
git clone https://github.com/trantatdat06/NCKH2026.git
cd NCKH2026

# 2. Môi trường Python 3.11
pip install -r requirements.txt

# 3. Sinh lại dữ liệu (tất định, seed 42, ~15 giây). data/ không có trên git
python scripts/01_generate_synthetic.py

# 4. Kiểm tra (phải đạt 12/12)
python -m pytest -q tests

# 5. Thử nhanh GNN trên 1 fold (~25 phút CPU)
python scripts/20_gnn_synthetic.py --folds 0
```

Thời gian chạy đầy đủ trên laptop CPU:

| Script | Thời gian |
|---|---|
| `10_baseline_tabular.py` | ~2 phút |
| `20_gnn_synthetic.py` | ~2 giờ |
| `30_dp_fl.py --mode record` | ~1,5 giờ |
| `30_dp_fl.py --mode client` | ~25 phút |
| `40_figures.py` | vài giây |

Mỗi script ghi tiến độ vào `results/<tên>/log.txt` khi đang chạy, có thể mở file đó để theo dõi.

**Không chạy lại** `02_make_splits.py`. File chia fold đã commit để cả nhóm dùng chung; script cũng từ chối ghi đè.

### Quy tắc bắt buộc

1. Mọi chỉ số đều tính qua `src/fraudfl/eval/metrics.py`, không tự viết lại.
2. Không chọn siêu tham số hay ngưỡng dựa trên tập **test**. Chỉ dùng validation (inner fold 0).
3. Muốn đổi chỉ số, k, cách chia fold hay tham số dữ liệu: ghi vào `docs/decisions.md` **trước**, rồi mới chạy.
4. Code mới trong `src/` phải có test trong `tests/`.
5. Làm việc trên nhánh riêng (`git checkout -b ten-viec`), mở Pull Request và để B review trước khi gộp vào `main`.

## 6. Việc tiếp theo — phân công đề xuất

Thứ tự ưu tiên: (1) họp duyệt quyết định → (2) seed và thống kê → (3) dữ liệu thật → (4) mở rộng.

### Cả nhóm — họp tuần này (khoảng 1 giờ)

- Duyệt hoặc sửa D7–D13 trong `decisions.md`. Sau khi chốt, chuyển các mục từ "Đề xuất" sang "Đã chốt".
- Chốt Q5 (IBM LI-Small hay HI-Small) và Q6 (bảng tỷ giá để tính VDR).
- Quyết định có làm thí nghiệm "ngân hàng nhỏ" hay không (mục 6, việc của E).

### A — BA/PO

- Sửa Slide 4, 5, 6, 7 theo mục 4.2.
- Viết dàn ý bài báo: Introduction, Problem, Method. Phần Results lấy số từ `results.md`.
- Theo dõi tiến độ; cập nhật bảng ở mục 2 của tài liệu này.

### B — Tech lead

- Review code hiện có, nhất là các file sau:
  - `src/fraudfl/privacy.py`: kiểm tra cách tính ε, cắt gradient, nhiễu.
  - `src/fraudfl/train.py`: logic FedAvg.
- **E6 FedProx**: thêm số hạng µ/2 · ‖w − w_global‖² vào `local_train` của `fedavg()` trong `train.py`.
  Thử µ ∈ {0,01; 0,1}.
- FedAvg 80–100 vòng: sửa `ROUNDS` trong `scripts/20_gnn_synthetic.py`, chạy `--folds 0` và xem đường hội tụ.
- Thêm tham số `--seed` cho các script 20 và 30. Hiện seed cố định trong `config.SEED`.

### C — Research lead

- Tổng quan tài liệu cho 3 điểm lệch kế hoạch:
  - Client-level và record-level DP trong cross-silo FL.
  - Lý do Adam kém khi có nhiễu DP.
  - Node-level DP cho GNN (ví dụ GAP, Sajadmanesh et al., 2023).
- Viết mục Related Work và Limitations (dựa trên mục 4.3).

### D — Data engineer

- Tải từ Kaggle vào `data/raw/`:
  - IBM AML bản gốc `HI-Small_Trans.csv` (có cột From Bank / To Bank) → `data/raw/ibm_aml/`.
  - Elliptic → `data/raw/elliptic/`.
- Đọc IBM bằng `fraudfl.data.ibm_aml.load_raw()`. Viết hàm chia theo ngân hàng thành 5–10 client cho E8.
- Viết data card (nguồn, giấy phép, thống kê, tỷ lệ nhãn) cho cả 3 bộ dữ liệu.
- Bảng tỷ giá cố định cho 15 loại tiền (Q6).

### E — Experiment runner

- **Chạy thêm 2 seed** cho E1–E5. Chờ B thêm `--seed`; mỗi seed mất khoảng 3,5 giờ, có thể chạy qua đêm.
- Viết `scripts/50_stats.py`:
  - Wilcoxon cho từng cặp: centralized–FedAvg, FedAvg–local, FedAvg–DP.
  - Friedman + hiệu chỉnh Holm.
  - Effect size.
  - Dùng `scipy.stats`, đọc dữ liệu từ `results/*/per_fold.csv`.
- **Thí nghiệm ngân hàng nhỏ**: giảm `target_n_tx` của 1–2 ngân hàng trong `config.py` xuống 5–10 nghìn và lưu
  thành cấu hình riêng (không sửa cấu hình gốc). Mục đích là kiểm tra local-only có tụt mạnh không, để thấy rõ lợi
  ích của FL.
- Đo latency và chi phí truyền thông cho mỗi vòng FL để vẽ biểu đồ latency mà Slide 10 yêu cầu.

## 7. Bản đồ mã nguồn — muốn sửa X thì mở file nào

| Muốn… | Mở file |
|---|---|
| Đổi số ngân hàng, tỷ lệ gian lận, seed, k | `src/fraudfl/config.py` |
| Đổi cách sinh giao dịch / kịch bản gian lận | `src/fraudfl/data/simulate.py` |
| Thêm đặc trưng | `src/fraudfl/data/features.py` (+ test trong `tests/test_features.py`) |
| Đổi cách dựng đồ thị | `src/fraudfl/graph.py` |
| Đổi kiến trúc mô hình | `src/fraudfl/models.py` |
| Đổi FedAvg / thêm FedProx | `src/fraudfl/train.py` |
| Đổi DP / cách tính ε | `src/fraudfl/privacy.py` |
| Thêm chỉ số | `src/fraudfl/eval/metrics.py` |
| Vẽ thêm biểu đồ | `scripts/40_figures.py` |

## 8. Câu hỏi thường gặp

- **Sao không thấy thư mục `data/`?** Dữ liệu không đưa lên git. Chạy `01_generate_synthetic.py` để sinh lại;
  kết quả giống hệt nhau trên mọi máy.
- **Lỗi font tiếng Việt trên console Windows?** Script nào cũng `import _bootstrap`, file này đã chuyển console
  sang UTF-8. Script mới nhớ thêm dòng này.
- **torch-geometric lỗi?** Dùng đúng phiên bản trong `requirements.txt`. Không cần `pyg-lib` vì GNN chạy full-batch.
- **Kết quả của tôi khác bảng một chút?** Có thể do khác phiên bản thư viện hoặc khác CPU/GPU. Lệch dưới 0,005 PR-AUC
  thì chấp nhận được. Lệch lớn hơn: báo B.
