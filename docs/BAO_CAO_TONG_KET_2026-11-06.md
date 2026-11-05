# Báo cáo tổng kết — 06/11/2026 (kết quả chính thức 3 seed)

Dành cho cả nhóm sau khi pull `main`. Thay cho [BAO_CAO_BAN_GIAO_2026-11-03.md](BAO_CAO_BAN_GIAO_2026-11-03.md) (đã cũ
ở mốc gộp chung và nhận định "non-IID là nguyên nhân chính"). Số liệu chi tiết từng bước: [nhat_ky_ket_qua.md](nhat_ky_ket_qua.md)
(giai đoạn 1–32); quyết định: [decisions.md](decisions.md) (D12, D17–D24).

---

## 1. Mục tiêu (Slide 10) và cách đọc kết quả

- **Mục tiêu gốc:** FL + DP đạt **≥ 85% hiệu năng của mô hình gộp chung**, với **ε ≤ 1** (δ = 1e-5), DP **mức client**
  (quyết định của nhóm — [luu_y_dp_muc_client.md](luu_y_dp_muc_client.md)).
- **Tiêu chí 2 tầng** (đề xuất, chưa chốt): (a) FL + DP ≥ 85% FL không DP; (b) FL + DP hơn tự học riêng.
- Chỉ số chính: **PR-AUC** trên tập test (cả hệ thống) và **PR-AUC trung bình theo ngân hàng** (công bằng khi so với
  tự học riêng). VDR@1% không ổn định trên IBM nên chỉ để tham khảo.

## 2. Thiết lập chính

| Mục | Giá trị |
|---|---|
| Dữ liệu | IBM AML HI-Small, ngày 0–9 (D18), **802 ngân hàng** có ≥ 1.000 giao dịch (D20) |
| Lấy mẫu | Train giữ 100% ca rửa tiền + 10% ca bình thường mỗi ngân hàng (r = 0,1); val/test giữ đủ |
| Đặc trưng | 22 đặc trưng "góc nhìn ngân hàng", chỉ dùng quá khứ |
| Mô hình | MLP 22→64→32→16→1 (4.097 tham số) |
| FL | FedProx μ = 0,1, trung bình đều, 200/802 ngân hàng mỗi vòng (q = 0,25), 5 epoch cục bộ, batch 512, lr 5e-3, **100 vòng** (D24) |
| DP | Mức client: cắt cập nhật ‖Δ‖ ≤ C = 0,35, máy chủ thêm nhiễu Gauss; ε tính bằng RDP (Opacus) |
| Đánh giá | 5 fold × **3 seed (42, 7, 2027)** = 15 cặp; Wilcoxon ghép cặp + Holm + bootstrap CI |

Lệnh chạy lại (một seed): xem mục "Cách chạy" trong [nhat_ky_ket_qua.md](nhat_ky_ket_qua.md) giai đoạn 27–31; script
chính `scripts/09_ibm_fl_dp.py`, kiểm định `scripts/15_thong_ke.py`, biểu đồ `scripts/16_bieu_do_ibm.py`.

## 3. Kết quả chính (TB 15 cặp)

| Cách | PR-AUC | % gộp chung (mục tiêu ≥ 85%) | % FL không DP (2 tầng, ≥ 85%) | Gấp tự học riêng |
|---|---|---|---|---|
| Gộp chung | 0,0929 | 100% | — | 2,77× |
| FL không DP (FedProx) | 0,0333 | 36% ❌ | 100% | 1,96× ✅ |
| FedAvg thường | 0,0242 | 26% ❌ | 74% | 1,63× ✅ |
| FL + DP ε = 2 | 0,0308 | 34% ❌ | 94% ✅ | 1,89× ✅ |
| **FL + DP ε = 1** | **0,0279** | **30% ❌** (36% chặng đường tới mục tiêu) | **86% ✅** (theo seed: 78 / 105 / 75%) | **1,77× ✅** (15/15) |
| FL + DP ε = 0,5 (chỉ seed 42) | 0,0209 | 22% ❌ | 59% ❌ | 1,48× ✅ |

Kiểm định ([results/thong_ke_3seed/bang.md](../results/thong_ke_3seed/bang.md), PR-AUC TB theo ngân hàng, Holm):

| So sánh | Chênh TB [CI 95%] | Thắng | p Holm |
|---|---|---|---|
| FL + DP ε = 1 > tự học riêng | +0,095 [+0,088; +0,101] | 15/15 | 0,0004 ✅ |
| FedProx > tự học riêng | +0,118 [+0,108; +0,130] | 15/15 | 0,0004 ✅ |
| Gộp chung > FedProx | +0,102 [+0,091; +0,112] | 15/15 | 0,0004 ✅ |
| FedProx > FedAvg thường | +0,041 [+0,031; +0,050] | 15/15 | 0,0004 ✅ |
| FedProx > FL + DP ε = 1 (chi phí DP) | +0,023 [+0,010; +0,036] | 12/15 | 0,025 ✅ |
| FedProx > FL + DP ε = 2 | +0,009 [+0,001; +0,016] | 12/15 | 0,073 |

## 4. Kết luận

1. **Tham gia liên kết có lợi kể cả khi có DP:** FL + DP ε = 1 hơn tự học riêng 1,77× ở mọi cặp (seed, fold).
2. **Mục tiêu gốc chưa đạt:** FL + DP ε = 1 mới bằng 30% gộp chung. Khoảng cách **chủ yếu do FL** (FL không DP chỉ 36%);
   DP lấy thêm khoảng 6 điểm %.
3. **DP ε = 1 tốn khoảng 14% so với FL không DP** — có ý nghĩa thống kê với 15 cặp (giai đoạn 31 với 2 seed chưa thấy được).
   Tiêu chí 2 tầng đạt nhưng sát ngưỡng và dao động mạnh theo seed.
4. DP mức client cần nhiều bên: ε = 1 giữ 4% FL khi có 5 ngân hàng, 2% khi 28, 78–86% khi 802 (1 seed / 3 seed).

## 5. Vấn đề còn tồn tại

**Nghiêm trọng**

1. **Khoảng cách FL ↔ gộp chung lớn** (36%). Nghi do: mỗi ngân hàng ít dữ liệu (r = 0,1; đường học của Đạo cho thấy MLP
   chưa bão hoà), kích cỡ 802 ngân hàng chênh nhau rất lớn, mô hình nhỏ.
2. **Chưa thống nhất mốc gộp chung:** 0,0929 (cùng lượng dữ liệu r = 0,1) hay 0,130 (đủ dữ liệu, kết quả của Đạo). Với
   0,130, FL + DP ε = 1 chỉ còn khoảng 21%.
3. **Chọn vòng tốt nhất dùng validation gộp của cả 802 ngân hàng** ([train.py:164-170](../src/fraudfl/train.py#L164-L170)):
   máy chủ thực tế không có dữ liệu này, và bước chọn này không được tính vào ε. Ngưỡng cắt C = 0,35 cũng chọn từ lần chạy
   không DP. Hội đồng có thể hỏi.
4. **ε = 1 sát ngưỡng 85% và dao động 75–105% theo seed** → khi báo cáo phải ghi cả khoảng.

**Trung bình**

5. PR-AUC tuyệt đối thấp (khoảng 0,03) — kết quả mạnh ở so sánh tương đối, khó nói là dùng được ngay trong thực tế.
6. VDR@1% (nêu trên Slide 10) không ổn định trên IBM → cần thống nhất PR-AUC là chỉ số chính.
7. Mới có 1 bộ dữ liệu (IBM HI-Small, 10 ngày) — chưa có E9/E10.
8. Trên IBM chỉ dùng MLP; GNN mới chạy trên synthetic.
9. ε = 0,5 / 4 / 8 mới có 1 seed; biểu đồ trong `results/bieu_do_ibm/` vẫn vẽ từ seed 42.

**Nhỏ**

10. Wilcoxon coi 15 cặp là độc lập trong khi các fold dùng chung dữ liệu → ghi chú hoặc hiệu chỉnh Nadeau–Bengio.

## 6. Việc tiếp theo (theo thứ tự ưu tiên)

| # | Việc | Ai | Ghi chú |
|---|---|---|---|
| 1 | Họp với anh Hùng: chốt **tiêu chí** (2 tầng hay 85% gộp chung) và **mốc gộp chung** (0,0929 hay 0,130) | Cả nhóm | Quyết định đề tài "đạt" hay "chưa đạt" |
| 2 | Sửa vấn đề 3: thêm tuỳ chọn **lấy mô hình vòng cuối** (không chọn vòng theo val gộp) rồi chạy lại FedProx + DP ε = 1, 2 | Đạo | Khoảng 3 giờ / seed; ghi vào nhật ký giai đoạn mới |
| 3 | Thu hẹp khoảng cách FL: thử r = 0,3 và FL cá nhân hoá (mỗi ngân hàng tinh chỉnh thêm cục bộ) | Đạo | Chạy trước 1 seed, fold 0 để xem có đáng không |
| 4 | Vẽ lại biểu đồ theo 3 seed; chạy thêm seed cho ε = 0,5 / 4 / 8 | Đạo / Đạt | `scripts/16_bieu_do_ibm.py` |
| 5 | E9: hỏi về script 60 (Elliptic/ULB); không kịp thì ghi là hướng phát triển | Đạo | |
| 6 | Cập nhật slide 4–7, 10 theo bảng ở mục 3 | Đạt | |

**Luật của nhóm (giữ nguyên):** mọi lần chạy lưu trữ bằng `archive_run` (không ghi đè, `results/_luu_tru/` +
`results/NHAT_KY_CHAY.csv`); mỗi giai đoạn ghi vào `docs/nhat_ky_ket_qua.md`, quyết định vào `docs/decisions.md`; mọi kết
quả báo kèm **% so với mục tiêu**.
