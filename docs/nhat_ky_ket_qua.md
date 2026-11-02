# Nhật ký kết quả theo giai đoạn

Tài liệu dùng để viết **báo cáo cả quá trình** lúc kết thúc đề tài. Mỗi giai đoạn ghi: đã làm gì, kết quả chính, kết
quả lưu ở đâu, và rút ra điều gì.

## Quy tắc lưu trữ (từ 02/11/2026)

1. **Không xóa kết quả cũ.** Thư mục kết quả chính (`results/<tên>/`) là bản mới nhất. Mỗi lần chạy script tự chép
   một bản vào `results/_luu_tru/<YYYY-MM-DD_HHMM>_<tên>/`, kèm `meta.json` (lệnh chạy, commit git, ghi chú).
2. **Sổ `results/NHAT_KY_CHAY.csv`** có một dòng cho mỗi lần chạy. Không sửa tay các dòng cũ.
3. **Commit `results/` (gồm `_luu_tru/` và sổ nhật ký) lên git** sau mỗi đợt chạy. Dữ liệu lớn (`data/raw`,
   `data/processed`) không commit.
4. **Kết thúc mỗi giai đoạn, thêm một mục vào tài liệu này**, kể cả kết quả xấu hoặc thí nghiệm thất bại.
5. Script mới phải gọi `fraudfl.runlog.archive_run(thu_muc_ket_qua, note=...)` ở cuối.

Kết quả chạy **trước ngày 02/11** chưa có kho lưu trữ; bản gốc nằm trong lịch sử git ở commit ghi trong từng mục.

---

## Giai đoạn 1 — Pipeline nền (31/10/2026)

- **Làm:** sinh dữ liệu synthetic 5 ngân hàng × khoảng 100 nghìn giao dịch, 15 đặc trưng, chia fold nested 5×3,
  EDA, module chỉ số, baseline RF/LR. Áp đề xuất D7–D9.
- **Kết quả:** RF với đặc trưng v1 (D8): PR-AUC 0,694 gộp chung, 0,690 tự học riêng.
- **Lưu ở:** commit `b62a631` và `97be662` (`results/baseline_tabular/`, `results/eda/`).
- **Rút ra:** đặc trưng tương đối (D8) gần như xóa khoảng cách gộp chung – tự học riêng.

## Giai đoạn 2 — E1–E5 (01/11/2026)

- **Làm:** GraphSAGE gộp chung / tự học riêng / FedAvg; DP mức client và mức bản ghi.
- **Kết quả (5 fold, 1 seed):** E1 gộp chung 0,733; E3 FedAvg 0,715; E2 tự học riêng 0,706. DP mức bản ghi ở
  ε = 1: 0,657 (89,6% so với gộp chung). DP mức client: 0,028 ở mọi ε ≤ 2.
- **Lưu ở:** commit `97be662` (`results/gnn_synthetic/`, `results/dp_record/`, `results/dp_client/`), tổng hợp ở
  `docs/results.md`.
- **Rút ra:** DP mức client với 5 ngân hàng làm mô hình sụp. Lợi ích của FL so với tự học riêng nhỏ.

## Giai đoạn 3 — Hạ tầng FL + real-time (02/11/2026, Đạo)

- **Làm:** Flower 1.42, Kafka, Redis, Postgres, Docker. Merge vào `main` qua PR #1 (`a06295e`).
- **Kết quả:** Flower khớp vòng lặp mô phỏng (PR-AUC 0,7188 so với 0,7168). Không mất hay trùng giao dịch khi tắt đột
  ngột consumer. Độ trễ chấm điểm trung vị 9,4 ms.
- **Lưu ở:** `docs/ha_tang.md`.

## Giai đoạn 4 — Chốt DP mức client, rà soát đánh giá (02/11/2026)

- **Quyết định:** giữ DP mức client (D12, `docs/luu_y_dp_muc_client.md`); lấy mẫu con IBM (D17, anh Hùng duyệt).
- **Rà soát code** phát hiện thêm: fold chia ngẫu nhiên thay vì theo thời gian (#32); điểm tự học riêng bị xếp hạng
  chung (#33); ngân sách cảnh báo tính chung 5 ngân hàng (#34). Đã thêm chỉ số theo từng ngân hàng (`macro_*`,
  `c<id>_*`) và chỉ số có trọng số cho dữ liệu lấy mẫu.
- **Thử nhanh DP mức client** (fold 0, 10 vòng, 5 ngân hàng, PR-AUC):

  | ε | A: nhiễu ở server, GNN nhỏ | B: nhiễu ở từng ngân hàng, GNN nhỏ | C: nhiễu ở server, GNN gốc |
  |---|---|---|---|
  | ∞ | 0,587 | 0,587 | 0,647 |
  | 8 | 0,069 | 0,019 | 0,021 |
  | 4 | 0,022 | 0,018 | 0,020 |
  | 1 | 0,018 | 0,019 | 0,020 |

  Lưu ở `results/_luu_tru/2026-11-02_2225_dp_client_nhanh_*`.
- **Rút ra:** với 5 ngân hàng, đổi chỗ cộng nhiễu, giảm vòng hay thu nhỏ mô hình đều không đủ — sụp cả ở ε = 8.
  Cần tăng nhiều số ngân hàng tham gia.

## Giai đoạn 5 — Dữ liệu IBM AML HI-Small (02/11/2026)

- **Dữ liệu:** 5.078.345 giao dịch, 5.177 ca rửa tiền (0,102%), 18 ngày, 30.528 ngân hàng gửi. Top 5 ngân hàng giữ
  13,6% giao dịch; chỉ 34 ngân hàng có ≥ 20 ca rửa tiền. Ca rửa tiền dồn vào 10 ngày đầu.
- **Sửa lỗi:** `load_raw` đọc mã ngân hàng thành số làm gộp nhầm 58 ngân hàng (`"010"` = `"10"`). Đã sửa, commit
  `cabad04`.
- **Tỷ giá (Q6):** suy từ chính dữ liệu, độ lệch chuẩn ≈ 0 — `results/ibm_features_global/ty_gia.csv`.
- **Đặc trưng:** 29 đặc trưng (chế độ global, cho E7), 23 đặc trưng (chế độ bank, cho FL); chỉ dùng quá khứ, không dùng
  nhãn. Đặc trưng mạnh: hình thức ACH (87% ca rửa tiền so với 12% ca thường), cuối tuần, mức hoạt động của tài khoản
  gửi; rửa tiền gần như không có ở Reinvestment, Wire, đổi tiền, tự chuyển cho mình.
- **Lưu ở:** `results/_luu_tru/2026-11-02_2224_ibm_features_global`, `..._2225_ibm_features_bank` (bản dùng);
  `..._2221_...`, `..._2222_...` là bản đầu trước khi chuyển số đếm sang thang log.
- **Chờ:** chốt r với anh Hùng rồi chạy `05_ibm_sample.py`.
