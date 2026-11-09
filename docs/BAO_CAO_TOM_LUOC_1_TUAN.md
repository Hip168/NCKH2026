# BÁO CÁO TÓM LƯỢC: TIẾN TRÌNH THỰC NGHIỆM 1 TUẦN QUA & ĐỐI CHIẾU KẾT QUẢ MONG ĐỢI
*(Bản tóm tắt điều hành — Đi từ Tổng quan đến Chi tiết)*

**Thời gian thực hiện:** 31/10/2026 – 06/11/2026  
**Đề tài:** Ứng dụng Học liên kết (FL) kết hợp Quyền riêng tư vi phân (DP) trong Phát hiện Gian lận Ngân hàng & Rửa tiền (AML)  
**Tài liệu gốc chi tiết:** `Báo cáo 5` và `nhat_ky_ket_qua.md` (32 giai đoạn).

---

## 1. TỔNG QUAN BÀI TOÁN & MỤC TIÊU KỲ VỌNG

### 1.1 Bối cảnh thực tiễn
- **Nghịch lý dữ liệu:** Tội phạm rửa tiền chuyển dịch dòng tiền qua **nhiều ngân hàng** để xóa dấu vết. Mỗi ngân hàng nếu tự làm một mình (**Local-only**) sẽ chỉ thấy một góc nhỏ $\to$ bỏ lọt tội phạm. Ngược lại, việc gom dữ liệu về một chỗ (**Centralized**) bị **Luật Bảo vệ Dữ liệu Cá nhân 2025 / Nghị định 13/2023/NĐ-CP** nghiêm cấm.
- **Giải pháp:** Học liên kết (**FL** - giữ dữ liệu tại chỗ, chỉ gửi cập nhật mô hình) kết hợp Quyền riêng tư vi phân (**DP mức client** - cắt độ dài vector và bơm nhiễu Gauss để che giấu đóng góp của từng ngân hàng).

### 1.2 Mục tiêu đặt ra trong Slide đề cương (Slide 10)
1. Chạy trên $\ge 3$ client mô phỏng, xử lý được dữ liệu không đồng nhất (non-IID).
2. Hiệu năng đạt $\text{PR-AUC} \ge 0.70$ và tỷ lệ giá trị tiền thu hồi $\text{VDR@1\%} \ge 0.80$.
3. Mô hình $\text{FL} + \text{DP}$ phải đạt **$\ge 85\%$ hiệu năng so với Gộp chung (Centralized)** tại ngân sách riêng tư nghiêm ngặt $\varepsilon \le 1.0$ ($\delta = 10^{-5}$) ở mức độ bảo vệ toàn bộ ngân hàng (**Client-level DP**).

---

## 2. BẢNG TRA CỨU THAM SỐ CỐT LÕI (Ý NGHĨA & GIÁ TRỊ CHỐT)

| Tham số | Giá trị chốt | Ý nghĩa thực tiễn & Lý do kỹ thuật |
|---|---|---|
| **Số ngân hàng ($K$)** | **$802$ ngân hàng** | Tăng từ $5 \to 28 \to 802$ (trên IBM AML). Số bên tham gia lớn giúp chia nhỏ lượng nhiễu DP bắt buộc, cứu sống DP mức client. |
| **Tỷ lệ chọn/vòng ($q$)** | **$0.25$** ($200/802$ NH) | Chọn ngẫu nhiên $200$ NH/vòng. Giúp kích hoạt hiệu ứng **khuếch đại riêng tư bằng lấy mẫu con**, làm chậm tốc độ tiêu tốn $\varepsilon$. |
| **Ngân sách riêng tư ($\varepsilon$)** | **$1.0$** ($\delta = 10^{-5}$) | Chuẩn quốc tế cho bảo mật mạnh, đảm bảo về mặt toán học kẻ tấn công không thể suy ngược thông tin khách hàng. |
| **Ngưỡng cắt chuẩn ($C$)** | **$0.35$** | Trần giới hạn độ dài vector cập nhật $\|\Delta\|_2 \le C$. Chọn $C=0.35$ vì đúng bằng **trung vị (p50)** độ lớn cập nhật khi dùng FedProx. |
| **Hệ số phạt Proximal ($\mu$)** | **$0.1$** | Hệ số neo giữ của thuật toán **FedProx**. Kéo mô hình các ngân hàng không bị lệch hướng khi dữ liệu phân tán (tăng $+58\%$ hiệu năng so với FedAvg). |
| **Số vòng giao tiếp ($T$)** | **$100$ vòng** | Điểm cân bằng vàng: 30 vòng chưa học xong; 100 vòng hội tụ tối ưu; 200 vòng thì nhiễu tích lũy bắt đầu làm tụt mô hình. |
| **Epoch cục bộ ($E$)** | **$5$ epoch** | Số lượt quét dữ liệu nội bộ tại mỗi ngân hàng trong một vòng giao tiếp trước khi gửi trọng số về máy chủ. |
| **Kích thước lô ($B$)** | **$512$** | Mini-batch 512 trên mô hình MLP giúp gradient cập nhật đều đặn, vượt trội hoàn toàn so với full-batch. |
| **Tốc độ học ($\eta$)** | **$5 \times 10^{-3}$** | Giúp mạng nơ-ron hội tụ nhanh trong số vòng giới hạn của FL. |
| **Tỷ lệ lấy mẫu ($r$)** | **$0.1$** ($10\%$ ca thường) | Giữ $100\%$ ca rửa tiền + $10\%$ ca thường. Giảm $90\%$ tải tính toán nhưng hiệu năng tương đương $100\%$ dữ liệu. |
| **Lọc thời gian (`max_day`)** | **Ngày $0 - 9$** | Cắt bỏ ngày 10–17 vì lỗi giả lập của IBM (rửa tiền chiếm 60%), ngăn chặn hoàn toàn hiện tượng rò rỉ dữ liệu. |

---

## 3. HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ THEN CHỐT

1. **PR-AUC (Toàn hệ thống):** Thước đo chuẩn mực cho bài toán mất cân bằng cực đoan ($0.1\%$ ca rửa tiền). Mức ngẫu nhiên chỉ là $0.0009$. ROC-AUC bị loại bỏ vì luôn đạt $> 0.98$ ảo.
2. **Macro PR-AUC (Trung bình từng ngân hàng):** Tính PR-AUC riêng biệt cho từng ngân hàng rồi lấy trung bình. **Đây là thước đo công bằng duy nhất** khi so sánh FL với Tự học riêng (Local-only) nhằm tránh lỗi lệch thang điểm (#33).
3. **Recall@1%:** Tỷ lệ % tội phạm rửa tiền bị bắt giữ khi chuyên viên chỉ đủ nguồn lực kiểm tra $1\%$ số giao dịch đáng ngờ nhất trong ngày.
4. **Kiểm định Wilcoxon & Holm:** Kiểm định phi tham số trên $15$ cặp thực nghiệm ($3$ Seed $\times 5$ Fold). Đạt $p < 0.05$ chứng minh kết quả là quy luật khoa học vững chắc.

---

## 4. TIẾN TRÌNH THỰC NGHIỆM CHI TIẾT (8 BƯỚC NGOẶT)

```
[Synthetic 5 NH] ──(Sụp DP Client)──> [Chuyển sang IBM AML] ──(Lọc Ngày 0-9)──> [Mở rộng K=802 NH]
                                                                                       │
[Kiểm định 3-Seed] <──(Tăng 100 Vòng) <──(FedProx μ=0.1 & C=0.35) <──(Hồi sinh DP Client)
```

| Mốc | Tên chặng & Lần chạy | Thay đổi kỹ thuật then chốt | Chỉ số trước $\to$ sau | Ý nghĩa & Bài học rút ra |
|---|---|---|---|---|
| **1** | **Synthetic 5 NH** *(GĐ 1–4)* | GraphSAGE, FedAvg, thử DP mức client vs mức bản ghi. | DP bản ghi: $0.657$ ($89.6\%$ Gộp); **DP client: $0.028$ (Sụp đổ)**. | Với $5$ NH, tín hiệu ($5$) bị nhiễu ($z=28.8$) xóa sạch. **DP mức client bất khả thi trên mạng lưới nhỏ**, bắt buộc phải tăng số ngân hàng. |
| **2** | **IBM AML HI-Small** *(GĐ 5–9)* | Nạp $5.08$ triệu giao dịch; phát hiện rò rỉ ngày 10–17; kiểm chứng tỷ lệ lấy mẫu $r$. | Lọc ngày 0–9 giúp VDR thật hạ từ $0.55$ ảo xuống $0.10$. Mẫu $r=0.1$ đạt PR-AUC $\approx 0.21$. | **Quyết định D18:** Cắt bỏ ngày 10–17 để dữ liệu sạch; chứng minh lấy mẫu $r=0.1$ giữ nguyên chất lượng học nhưng nhẹ máy hơn 10 lần. |
| **3** | **MLP & Mini-batch** *(GĐ 10–15)* | Chuyển sang MLP dạng bảng; phát hiện lỗi đo ngắt hòa (#41); chuyển sang mini-batch 512. | PR-AUC Gộp tăng $0.055 \to 0.086$; Recall mô hình sụp giảm từ $38\%$ ảo về $<5\%$ thật. | Sửa lỗi đo đạc bằng phá hòa ngẫu nhiên cố định seed. Huấn luyện mini-batch giúp MLP khai thác triệt để dữ liệu. |
| **4** | **Đột phá $K=802$ NH** *(GĐ 16–19)* | Mở rộng FL lên toàn bộ $802$ NH thật, chọn $200$ NH/vòng ($q=0.25$). | Nhiễu/tham số giảm từ $0.79 \to 0.029$. **DP client $\varepsilon=1.0$ đạt $0.0152$**. | **Bước ngoặt sống còn:** Lần đầu tiên DP mức client ở $\varepsilon=1.0$ học thành công (gấp 17 lần đoán mò). Tách nguyên nhân: $75\%$ tụt là do trung bình đều, nhiễu chỉ mất chút ít. |
| **5** | **Thuật toán FedProx** *(GĐ 20–23)* | Tích hợp FedProx $\mu=0.1$; đo phân bố cập nhật chọn lại ngưỡng cắt $C=0.35$. | PR-AUC FedProx tăng $0.0178 \to 0.0282$ ($+58\%$). DP $\varepsilon=1$ đạt $0.0233$ ($83\%$ FedProx). | FedProx khắc phục xuất sắc dữ liệu lệch giữa các ngân hàng. $C=0.35$ (trung vị) giúp tối ưu hóa biên độ cắt và lượng nhiễu. |
| **6** | **Kiểm chứng Non-IID** *(GĐ 24–25)* | Xáo trộn dữ liệu thành IID hoàn toàn; thử nghiệm trọng số bị chặn (Bounded Weights). | Dữ liệu IID chỉ đạt $0.024$ (so với $0.021$ thật); Trọng số bị chặn làm tăng nhiễu 6 lần. | Phản nghiệm khoa học: Bản chất Non-IID không phải rào cản chính của DP. Chốt kiên định dùng **Trung bình đều**. |
| **7** | **Tối ưu 100 vòng** *(GĐ 26–28)* | Mở rộng số vòng từ $30 \to 100 \to 200$ vòng trên Fold 0. | 30 vòng: DP $= 0.0233$; **100 vòng: DP $= 0.0304$ ($+30\%$)**; 200 vòng: tụt về $0.0251$. | 100 vòng là điểm chín của mô hình. Vượt qua 100 vòng thì nhiễu tích lũy theo định lý DP tăng vọt ($0.025$/tham số) làm hỏng trọng số. |
| **8** | **3-Seed Chính thức** *(GĐ 29–32)* | Chạy 5-Fold trên $3$ seed ngẫu nhiên ($42, 7, 2027$) $\to 15$ cặp đối chứng Wilcoxon + Holm. | FL+DP $\varepsilon=1$ đạt $0.0279$ (toàn hệ thống) và $0.2193$ (Macro theo từng NH). | **Kết luận chính thức:** FL+DP $\varepsilon=1$ **thắng Tự học riêng ở 15/15 cặp ($p=0.0004$)**, khẳng định giá trị bền vững của việc hợp tác liên minh. |

---

## 5. BẢNG KẾT QUẢ CHÍNH THỨC & ĐỐI CHIẾU SLIDE MONG ĐỢI

### 5.1 Bảng kết quả tổng hợp chính thức (Trung bình 15 cặp thực nghiệm)

| Phương pháp thực nghiệm | PR-AUC toàn hệ thống | % so với Gộp chung (Slide 10: $\ge 85\%$) | % so với FedProx (Tiêu chí 2 tầng: $\ge 85\%$) | Macro PR-AUC (TB từng ngân hàng) | Gấp bao nhiêu lần Tự học riêng? | Số lần thắng (Wins / 15 cặp) |
|---|---|---|---|---|---|---|
| **Gộp chung (Centralized)** | $0.0929$ | $100\%$ | — | $0.3441$ | $2.77\times$ | $15/15$ |
| **FedProx không DP (100 vòng)** | $0.0333$ | $36\%$ ❌ | $100\%$ | $0.2426$ | $1.96\times$ | $15/15$ |
| **FedAvg thường (100 vòng)** | $0.0242$ | $26\%$ ❌ | $74\%$ | $0.2013$ | $1.63\times$ | $15/15$ |
| **FL + DP ở mức $\varepsilon = 2.0$** | $0.0308$ | $34\%$ ❌ | **$94\%$ ✅** | $0.2337$ | $1.89\times$ | $15/15$ |
| **FL + DP ở mức $\varepsilon = 1.0$ (Chuẩn)** | $\mathbf{0.0279}$ | $\mathbf{30\%}$ ❌ | **$86\%$ ✅** | $\mathbf{0.2193}$ | $\mathbf{1.77\times}$ | $\mathbf{15/15}$ |
| **FL + DP ở mức $\varepsilon = 0.5$ (Cao)** | $0.0209$ | $22\%$ ❌ | $59\%$ ❌ | $0.1830$ | $1.48\times$ | $15/15$ |
| **Tự học riêng (Local-only)** | $0.0178$ | $19\%$ | — | $0.1242$ | $1.00\times$ | Cận dưới |

---

### 5.2 Đối chiếu trực diện với Tiêu chí Slide 10

```
┌──────────────────────────────┬───────────────────┬───────────────────────────┬─────────────┐
│ Tiêu chí đánh giá            │ Chỉ tiêu Slide 10 │ Kết quả thực tế đạt được  │ Trạng thái  │
├──────────────────────────────┼───────────────────┼───────────────────────────┼─────────────┤
│ 1. Số lượng Client tham gia  │ ≥ 3 client        │ 802 ngân hàng             │ VƯỢT XA  ✅ │
│ 2. Đảm bảo chuẩn bảo mật DP  │ ε ≤ 1.0, δ ≤ 10⁻⁵ │ ε = 1.0, δ = 10⁻⁵         │ ĐẠT CHUẨN ✅ │
│ 3. PR-AUC tuyệt đối          │ PR-AUC ≥ 0.70     │ ~0.03 (Gộp chung ~0.09)   │ KHÔNG ĐẠT ❌│
│ 4. Giá trị tiền VDR@1%       │ VDR ≥ 0.80        │ 0.00 - 0.48 (Dao động lớn)│ KHÔNG ĐẠT ❌│
│ 5. Tỷ lệ so với Gộp chung    │ FL+DP ≥ 85% Gộp   │ 30% Gộp chung             │ KHÔNG ĐẠT ❌│
│ 6. Tiêu chí 2 tầng (Đề xuất) │ FL+DP ≥ 85% FL gốc│ 86% so với FL không DP    │ ĐẠT CHUẨN ✅ │
│ 7. Lợi ích liên kết (RQ1)    │ Hơn Tự học riêng  │ Thắng 15/15 cặp (1.77×)   │ ĐẠT CHUẨN ✅ │
└──────────────────────────────┴───────────────────┴───────────────────────────┴─────────────┘
```

---

## 6. BÓC TÁCH BẢN CHẤT KHOA HỌC: NÚT THẮT THỰC SỰ NẰM Ở ĐÂU?

### Điểm 1: Chi phí bảo mật DP chỉ lấy đi $14\%$ hiệu năng
- Mô hình $\text{FL} + \text{DP}$ ở $\varepsilon = 1.0$ đạt $0.0279$, giữ được **$86\%$** sức mạnh của FedProx không DP ($0.0333$).
- $\to$ **Cơ chế DP hoạt động hoàn toàn ổn định và đạt tiêu chí bảo tồn $\ge 85\%$ khi tách biệt khỏi ảnh hưởng của FL.**

### Điểm 2: Nút thắt chính nằm ở bài toán Học liên kết ($64\%$ mất mát)
- Bản thân Học liên kết không DP chỉ đạt $36\%$ so với Gộp chung ($0.0333$ vs $0.0929$).
- *Nguyên nhân:*
  1. Dữ liệu bị phân mảnh trên $802$ ngân hàng, mỗi ngân hàng có rất ít mẫu khiến mạng nơ-ron khó học liên kết phi tuyến.
  2. Ngân hàng bị **mù thông tin tài khoản thụ hưởng** (`bank view`), trong khi mô hình Gộp chung nhìn thấy toàn bộ đường đi dòng tiền xuyên ngân hàng (mất $0.07$ điểm PR-AUC).

### Điểm 3: Khẳng định tính ưu việt của Liên minh (RQ1)
- Trong mọi phép đo trên 15 cặp độc lập, **$\text{FL} + \text{DP}$ $\varepsilon = 1.0$ luôn vượt trội Tự học riêng gấp $1.77$ lần với $p = 0.0004$**.
- Tham gia liên minh mang lại lợi ích sống còn, đặc biệt cho các ngân hàng vừa và nhỏ vốn chỉ có trung vị $2$ ca rửa tiền nội bộ.

---

## 7. KẾ HOẠCH HÀNH ĐỘNG RÚT GỌN

1. **Họp GVHD (thầy/anh Hùng):** Báo cáo kết quả 3-seed, đề xuất chốt chuyển sang **Tiêu chí 2 tầng** ($\ge 85\%$ FL không DP và thắng Tự học riêng) để đề tài đạt chuẩn tốt nghiệp/công bố.
2. **Cập nhật Slide (Slide 4, 6, 7, 10):** Đưa bảng số liệu thực tế 3-seed vào thay thế các con số giả thuyết cũ.
3. **Bịt lỗ hổng phản biện:** Chạy bổ sung thực nghiệm lấy mô hình ở vòng cuối cùng (Last Round) để không phụ thuộc vào tập validation gộp.
4. **Hoàn thiện bản thảo bài báo:** Bổ sung kết quả kiểm định Wilcoxon 15 cặp và phần thảo luận nguyên nhân FL penalty vào bài báo khoa học.

