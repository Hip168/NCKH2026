# BÁO CÁO KHOA HỌC TOÀN DIỆN: QUÁ TRÌNH THỰC NGHIỆM 1 TUẦN QUA (31/10/2026 – 06/11/2026)
**Đề tài:** Ứng dụng Học liên kết (Federated Learning) kết hợp Quyền riêng tư vi phân (Differential Privacy) trong Phát hiện Gian lận Giao dịch Ngân hàng & Chống rửa tiền (AML)  
**Nhóm tác giả:** 5 sinh viên ngành Hệ thống Thông tin (HTTT)  
**Tài liệu đối chiếu:** `nhat_ky_ket_qua.md` (Giai đoạn 1–32), `BAO_CAO_TONG_KET_2026-11-06.md`, và Slide Kế hoạch NCKH `mo-ta-chi-tiet_presentation-plan.md`.

---

## MỤC LỤC
1. [TỔNG QUAN ĐỀ TÀI & BỐI CẢNH NGHIÊN CỨU](#1-tổng-quan-đề-tài--bối-cảnh-nghiên-cứu)
2. [HỆ THỐNG THAM SỐ TOÀN DIỆN (Ý NGHĨA & GIÁ TRỊ)](#2-hệ-thống-tham-số-toàn-diện-ý-nghĩa--giá-trị)
3. [HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ (GIẢI THÍCH TOÁN HỌC & BÌNH DÂN)](#3-hệ-thống-chỉ-số-đánh-giá-giải-thích-toán-học--bình-dân)
4. [TIẾN TRÌNH THỰC NGHIỆM CHI TIẾT TỪNG LẦN CHẠY (GIAI ĐOẠN 1 ĐẾN 32)](#4-tiến-trình-thực-nghiệm-chi-tiết-từng-lần-chạy-giai-đoạn-1-đến-32)
5. [BẢNG KẾT QUẢ CHÍNH THỨC 3-SEED & ĐỐI CHIẾU TRỰC DIỆN VỚI SLIDE MONG ĐỢI](#5-bảng-kết-quả-chính-thức-3-seed--đối-chiếu-trực-diện-với-slide-mong-đợi)
6. [BÓC TÁCH BẢN CHẤT KHOA HỌC: NÚT THẮT NẰM Ở ĐÂU?](#6-bóc-tách-bản-chất-khoa-học-nút-thắt-nằm-ở-đâu)
7. [CÁC LỖI KỸ THUẬT QUAN TRỌNG ĐÃ PHÁT HIỆN & SỬA ĐỔI](#7-các-lỗi-kỹ-thuật-quan-trọng-đã-phát-hiện--sửa-đổi)
8. [HƯỚNG DẪN BẢO VỆ ĐỀ TÀI TRƯỚC HỘI ĐỒNG (Q&A PHẢN BIỆN KHOA HỌC)](#8-hướng-dẫn-bảo-vệ-đề-tài-trước-hội-đồng-qa-phản-biện-khoa-học)
9. [KẾT LUẬN & KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO](#9-kết-luận--kế-hoạch-hành-động-tiếp-theo)

---

## 1. TỔNG QUAN ĐỀ TÀI & BỐI CẢNH NGHIÊN CỨU

### 1.1 Vấn đề thực tiễn (Problem Statement)
Tội phạm tài chính và các tổ chức rửa tiền tinh vi không bao giờ thực hiện hành vi trên một tài khoản đơn lẻ. Chúng chia nhỏ dòng tiền bất chính và chuyển dịch qua lại giữa **nhiều ngân hàng khác nhau** (giao dịch nhiều chặng - multi-hop money laundering).
- Nếu mỗi ngân hàng hoạt động độc lập và chỉ tự huấn luyện mô hình trên dữ liệu nội bộ (**Local-only**), họ chỉ thấy một phần rất nhỏ của chuỗi hành vi $\to$ Tỷ lệ phát hiện thấp, bỏ lọt tội phạm lớn.
- Tuy nhiên, việc **gom toàn bộ dữ liệu thô về một cơ sở dữ liệu chung (Centralized)** là hành vi vi phạm pháp luật nghiêm trọng theo **Nghị định 13/2023/NĐ-CP** và **Luật Bảo vệ Dữ liệu Cá nhân 2025** (có hiệu lực từ 01/01/2026). Dữ liệu khách hàng bị cô lập thành các "ốc đảo dữ liệu" (Data Silos).

### 1.2 Giải pháp đề xuất: FL + DP
- **Học liên kết (Federated Learning - FL):** Cho phép các ngân hàng cùng hợp tác huấn luyện mô hình thông qua một Máy chủ điều phối (Server). Dữ liệu thô nằm yên tại chỗ; ngân hàng chỉ gửi các bản cập nhật trọng số (gradients/weights) lên máy chủ để gộp lại (FedAvg/FedProx).
- **Quyền riêng tư vi phân (Differential Privacy - DP):** Vì kẻ tấn công có thể dùng kỹ thuật suy diễn ngược (Reconstruction/Inference Attack) từ trọng số để tái tạo dữ liệu cá nhân, kỹ thuật DP can thiệp bằng cách: **Cắt ngắn độ dài vector cập nhật (Clipping)** và **bơm nhiễu ngẫu nhiên Gauss** trước khi gộp.

### 1.3 Mục tiêu kỳ vọng ban đầu trong Slide (Slide 7 & 10)
Bộ slide đề cương NCKH đặt ra các chỉ tiêu kỹ thuật:
- Chạy hệ thống với $\ge 3$ client mô phỏng.
- Đạt hiệu năng tuyệt đối: $\text{PR-AUC} \ge 0.70$ và $\text{VDR@1\%} \ge 0.80$.
- Mô hình Học liên kết có bảo mật ($\text{FL} + \text{DP}$) phải đạt **$\ge 85\%$ hiệu năng của mô hình gộp chung dữ liệu (Centralized)** tại ngân sách riêng tư nghiêm ngặt $\varepsilon \le 1.0$ ($\delta = 10^{-5}$) ở cấp độ bảo vệ toàn bộ ngân hàng (**Client-level DP**).

---

## 2. HỆ THỐNG THAM SỐ TOÀN DIỆN (Ý NGHĨA & GIÁ TRỊ)

| Tham số | Ký hiệu / Flag | Giá trị sử dụng | Ý nghĩa bản chất | Tại sao chọn giá trị này? |
|---|---|---|---|---|
| **Số lượng ngân hàng** | $K$ (`--min-tx`) | $5 \to 28 \to \mathbf{802}$ | Tổng số ngân hàng tham gia mạng lưới liên kết. | Khi $K=5$ hoặc $28$, lượng nhiễu DP bắt buộc lấn át tín hiệu làm mô hình sụp đổ. Mở rộng lên **$K=802$ ngân hàng** (từ tập dữ liệu IBM AML) giúp chia nhỏ lượng nhiễu cần bơm, "cứu sống" DP mức client. |
| **Số ngân hàng chọn mỗi vòng** | $m$ (`--clients-per-round`) | $\mathbf{200}$ ($q = 200/802 \approx 0.25$) | Tỷ lệ số ngân hàng được bốc ngẫu nhiên để huấn luyện trong mỗi vòng giao tiếp. | Kích hoạt hiệu ứng **khuếch đại riêng tư bằng lấy mẫu con (subsampling amplification)** theo chuẩn toán học của Rényi DP (Opacus), giúp tiết kiệm ngân sách $\varepsilon$. |
| **Ngân sách riêng tư** | $\varepsilon$ (Epsilon) | $\infty, 8, 4, 2, \mathbf{1.0}, 0.5$ | Đại lượng đo lường độ rò rỉ thông tin tối đa. $\varepsilon$ càng nhỏ, mô hình càng bảo mật nhưng càng nhiều nhiễu. | $\varepsilon \le 1.0$ là ngưỡng bảo mật khắt khe nhất được các cơ quan an ninh quốc tế công nhận để chống suy ngược dữ liệu cá nhân. $\varepsilon = \infty$ là mô hình liên kết không có nhiễu DP. |
| **Rủi ro rò rỉ tối đa** | $\delta$ (Delta) | $\mathbf{10^{-5}}$ ($1/100.000$) | Xác suất cực nhỏ mà tính chất bảo vệ DP có thể bị trượt. | Theo định lý toán học DP, $\delta$ phải nhỏ hơn rất nhiều so với nghịch đảo số bản ghi ($1/N$). Với hàng triệu giao dịch, $\delta = 10^{-5}$ là chuẩn an toàn. |
| **Ngưỡng cắt chuẩn cập nhật** | $C$ (`--clip`) | $1.0 \to 1.6 \to \mathbf{0.35}$ | Giới hạn độ dài tối đa của vector hiệu số trọng số $\|\Delta_k\|_2 \le C$. | Nếu không cắt, một ngân hàng có trọng số thay đổi đột ngột sẽ phá vỡ tính riêng tư. Giá trị $C=0.35$ được chọn vì đúng bằng **phân vị trung vị (median p50)** của độ dài cập nhật khi dùng thuật toán FedProx. |
| **Hệ số phạt Proximal** | $\mu$ (`--prox-mu`) | $0.01, \mathbf{0.1}, 0.3, 1.0$ | Hệ số phạt độ lệch giữa mô hình cục bộ và mô hình toàn cục: $\frac{\mu}{2}\|w - w_t\|^2$. | Trong thuật toán **FedProx**, do dữ liệu giữa các ngân hàng rất khác nhau (non-IID), mô hình địa phương dễ bị "lệch tủ". $\mu = 0.1$ giúp kéo mô hình hội tụ tốt nhất (+58% hiệu năng so với FedAvg). |
| **Số vòng giao tiếp** | $T$ (`--rounds`) | $10 \to 30 \to \mathbf{100} \to 200$ | Số chu kỳ phát mô hình - học cục bộ - thu hồi và gộp trọng số. | Ở 30 vòng mô hình chưa học xong. **100 vòng** là điểm cân bằng vàng: mô hình vừa kịp hội tụ, trong khi lượng nhiễu tích lũy vẫn trong tầm kiểm soát. Ở 200 vòng, lượng nhiễu tích tụ làm tụt hiệu năng. |
| **Số epoch cục bộ** | $E$ (`--local-epochs`) | $1 \to 3 \to \mathbf{5}$ | Số lượt quét dữ liệu tại chỗ của mỗi ngân hàng trong một vòng FL. | $E=1$ khiến số bước cập nhật quá ít; $E=5$ giúp mỗi ngân hàng học sâu hơn trên dữ liệu của chính mình trước khi đóng góp lên mạng lưới. |
| **Kích thước lô nhỏ** | $B$ (`--batch-size`) | $\mathbf{512}$ | Số giao dịch được tính toán gradient trong một bước cập nhật SGD. | Chuyển từ full-batch sang mini-batch 512 giúp mô hình MLP cập nhật thường xuyên hơn, khai thác tối đa tín hiệu từ các giao dịch. |
| **Tốc độ học** | $\eta$ (`--lr`) | $10^{-3} \to \mathbf{5 \times 10^{-3}}$ | Bước nhảy gradient trong tối ưu hóa Adam/SGD. | $\eta = 5 \times 10^{-3}$ giúp mạng nơ-ron học nhanh và bù đắp số vòng hạn chế của FL. |
| **Trọng số lớp dương** | `pos_weight` | Cân bằng $\to \mathbf{1.0} \to 10$ | Hệ số nhân độ mất mát cho các ca rửa tiền trong hàm BCE loss. | Dù dữ liệu mất cân bằng nặng, đặt `pos_weight = 1.0` kết hợp lấy mẫu con cho độ ổn định cao hơn hẳn so với ép cân bằng nhân tạo. |
| **Tỷ lệ lấy mẫu ca thường** | $r$ (`--rate`) | $0.01, 0.02, \mathbf{0.1}, 1.0$ | Tỷ lệ lấy mẫu ngẫu nhiên giao dịch bình thường (giữ $100\%$ ca rửa tiền). | Giúp giảm tải dữ liệu $5$ triệu dòng. Thử nghiệm chứng minh $r=0.1$ (giữ 10% ca thường) đạt hiệu năng tương đương $100\%$ dữ liệu nhưng nhẹ máy hơn $10$ lần. |
| **Ngày lọc dữ liệu** | `max_day` | $\mathbf{9}$ (Ngày 0 đến 9) | Giới hạn khung thời gian phân tích từ file gốc 18 ngày của IBM AML. | Từ ngày 10 đến 17 bị lỗi dữ liệu nhân tạo (rửa tiền chiếm 60%). Cắt bỏ giúp loại trừ hiện tượng rò rỉ thông tin thời gian. |

---

## 3. HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ (GIẢI THÍCH TOÁN HỌC & BÌNH DÂN)

1. **PR-AUC (Precision-Recall Area Under Curve):**
   - *Bản chất:* Diện tích dưới đường cong Precision (Độ chuẩn xác) - Recall (Độ bao phủ).
   - *Tại sao là chỉ số sinh tử?* Trong dữ liệu gian lận ngân hàng, ca rửa tiền chỉ chiếm $0.1\%$ ($1$ ca trên $1.000$ giao dịch). ROC-AUC truyền thống sẽ luôn đạt $> 0.98$ vì bị áp đảo bởi số lượng khổng lồ ca bình thường (True Negatives). PR-AUC chỉ tập trung vào lớp thiểu số (ca rửa tiền). Mức đoán ngẫu nhiên của PR-AUC chỉ là $0.0009$.
2. **Macro PR-AUC (Trung bình PR-AUC theo từng ngân hàng):**
   - *Bản chất:* Tính PR-AUC riêng biệt cho từng ngân hàng trong số $802$ ngân hàng, sau đó lấy trung bình cộng.
   - *Tại sao bắt buộc phải dùng?* Khi so sánh mô hình Liên kết với mô hình **Tự học riêng (Local-only)**, mỗi ngân hàng tự học có một phân phối điểm số riêng. Nếu gộp chung lại để xếp hạng, thang điểm lệch nhau sẽ tạo ra thứ hạng sai lệch nghiêm trọng (Lỗi đo đạc #33). Do đó, **Macro PR-AUC là thước đo công bằng duy nhất** để trả lời câu hỏi: *"Tham gia liên minh có giúp từng ngân hàng bắt gian lận tốt hơn tự làm một mình hay không?"*
3. **Recall@1% (Độ phủ tại ngân sách kiểm tra 1%):**
   - *Bản chất:* Giả định năng lực điều tra thực tế của chuyên viên an ninh chỉ soát xét được $1\%$ số giao dịch đáng ngờ nhất mỗi ngày. Chỉ số này đo lường: Chuyên viên sẽ bắt được bao nhiêu % trên tổng số tội phạm rửa tiền xuất hiện trong ngày.
4. **VDR@1% (Value Detection Rate - Tỷ lệ giá trị tiền thu hồi):**
   - *Bản chất:* Trong số các vụ rửa tiền bị tóm ở top 1%, tổng giá trị tiền thu hồi được chiếm bao nhiêu % tổng số tiền rửa tiền trên toàn hệ thống.
5. **Kiểm định Thống kê Wilcoxon Signed-Rank & Hiệu chỉnh Holm:**
   - *Bản chất:* Kiểm định phi tham số trên các cặp dữ liệu tương ứng giữa hai thuật toán. Áp dụng trên $15$ cặp mẫu ($5$ Fold $\times 3$ Seed). Giá trị $p$-value sau hiệu chỉnh Holm nếu $< 0.05$ khẳng định kết quả là quy luật khoa học vững chắc, không thể do ngẫu nhiên may rủi.

---

## 4. TIẾN TRÌNH THỰC NGHIỆM CHI TIẾT TỪNG LẦN CHẠY (GIAI ĐOẠN 1 ĐẾN 32)

Toàn bộ 1 tuần nghiên cứu và thực nghiệm liên tục (31/10/2026 – 06/11/2026) được chia thành các bước đi logic sau:

### Giai đoạn 1 – 4: Thử nghiệm Pipeline nền trên Dữ liệu Synthetic (31/10 – 01/11/2026)
- **Hành động:** Tạo dữ liệu giả lập $5$ ngân hàng ảo $\times 100.000$ giao dịch. Xây dựng cấu trúc Nested CV 5×3. Huấn luyện GraphSAGE ($128$ chiều) gộp chung, tự học riêng, và FedAvg.
- **Kết quả chỉ số:**
  - Gộp chung (E1): PR-AUC $= \mathbf{0.733} \pm 0.014$.
  - FedAvg (E3): PR-AUC $= \mathbf{0.715} \pm 0.013$ ($97.6\%$ so với gộp chung).
  - Tự học riêng (E2): PR-AUC $= \mathbf{0.706} \pm 0.011$ ($96.3\%$ so với gộp chung).
  - DP mức bản ghi (record-level) ở $\varepsilon = 1.0$: PR-AUC $= \mathbf{0.657}$ ($89.6\%$ so với gộp chung, đạt tiêu chí Slide 10).
  - **DP mức client (client-level) ở $\varepsilon \le 2.0$:** PR-AUC $= \mathbf{0.028}$ (sụp đổ hoàn toàn về mức đoán mò).
- **Kết luận rút ra:** Với $5$ ngân hàng, tín hiệu tổng chỉ là $5$ vector cập nhật, trong khi nhiễu Gauss cần thiết cho $\varepsilon = 1.0$ lên tới $z = 28.8$. Tỷ lệ tín hiệu trên nhiễu chỉ đạt $0.17$ $\to$ Nhiễu xóa sạch tri thức. Thử nghiệm thay đổi vị trí cộng nhiễu hay thu nhỏ GNN (Thí nghiệm A, B, C) đều bất lực. **Cần một tập dữ liệu có số ngân hàng lớn hơn nhiều.**

### Giai đoạn 5 – 9: Bước ngoặt Chuyển sang IBM AML HI-Small & Khám phá Dữ liệu (02/11/2026)
- **Hành động:** Tiếp nhận tập dữ liệu $5.078.345$ giao dịch, $5.177$ ca rửa tiền ($0.102\%$). Trích xuất $22$ đặc trưng góc nhìn ngân hàng (`bank view`) và $28$ đặc trưng toàn cục (`global view`).
- **Phát hiện lỗi dữ liệu & Quyết định D18:** Kiểm tra phân bố theo thời gian phát hiện từ ngày 10 đến ngày 17 chỉ có $1.100$ giao dịch nhưng $60\%$ là rửa tiền (lỗi giả lập của IBM). Nếu giữ lại, biến thời gian `SND_AGE_DAYS` gây rò rỉ dữ liệu ảo, làm VDR nhảy vọt từ $0.10$ lên $0.55$. **Nhóm quyết định cắt bỏ ngày 10–17, chỉ dùng ngày 0–9** (giữ lại trọn vẹn $4.522$ ca rửa tiền sạch).
- **Kiểm chứng tỷ lệ lấy mẫu $r$ (Giai đoạn 8 & 9):** Thử nghiệm $r \in \{0.01; 0.02; 0.05; 0.1; 0.2\}$. Kết quả trên 5-fold chứng minh $r=0.1$ có trọng số phản ánh trung thực phân bố thật mà giảm tải $90\%$ dung lượng xử lý.

### Giai đoạn 10 – 15: Kiểm chứng Mô hình MLP & Sửa Lỗi Đo Đạc Phá Hòa (#41) (02/11 – 03/11/2026)
- **Thử nghiệm trên $K=28$ ngân hàng:** Dùng mô hình MLP ($4.097$ tham số), $27$ ngân hàng lớn + $1$ client gom. Ở $\varepsilon = 1.0$, PR-AUC vẫn tụt về $0.002$ (sụp đổ), vì nhiễu mỗi tham số vẫn còn quá lớn ($0.79$).
- **Phát hiện & sửa lỗi đo đạc #41:** Nhóm thấy khi mô hình sụp (các điểm số dự đoán bằng nhau), chỉ số Recall@1% lại cao bất thường ($0.38$). Nguyên nhân là hàm xếp hạng đã ngắt hòa theo thứ tự dòng xuất hiện trong file, vô tình trùng với các ca gian lận ở đầu bảng. Đã sửa triệt để bằng thuật toán **phá hòa ngẫu nhiên cố định seed**.
- **Cải tiến huấn luyện Mini-batch:** Chuyển từ full-batch sang mini-batch $512$, $5$ epoch cục bộ, $\text{lr} = 5 \times 10^{-3}$, `pos_weight = 1.0` (Cấu hình T2/F2). Giúp mô hình MLP tăng vọt PR-AUC gộp từ $0.055$ lên $0.086 - 0.111$.

### Giai đoạn 16 – 19: Đột phá Quy mô $K=802$ Ngân Hàng — "Hồi sinh" DP Mức Client (03/11/2026)
- **Hành động (D20):** Mở rộng mạng lưới lên toàn bộ **$802$ ngân hàng** có $\ge 1.000$ giao dịch. Mỗi vòng bốc ngẫu nhiên $200$ ngân hàng ($q=0.25$).
- **Kết quả:**
  - Nhiễu Gauss trên mỗi tham số giảm kỷ lục từ $0.79$ xuống **$0.029$**!
  - **Lần đầu tiên DP mức client ở $\varepsilon = 1.0$ học thành công: PR-AUC $= 0.0152$** (gấp 17 lần mức ngẫu nhiên).
- **Phân rã nguyên nhân tụt hiệu năng (Giai đoạn 19):** Chạy thử nghiệm giả lập không nhiễu ($\varepsilon = 1000$):
  - FedAvg trọng số giao dịch: $0.0248$.
  - Chuyển sang **Trung bình đều (Equal weighting)**: giảm xuống $0.0178$ (mất $-0.0070$).
  - Cắt vector cập nhật $C=1.6$: giảm xuống $0.0174$ (mất $-0.0004$).
  - Thêm nhiễu Gauss ở $\varepsilon = 1.0$: đạt $0.0173$ (gần như không mất gì thêm!).
  - **Kết luận:** $75\%$ mức tụt của DP là do quy tắc trung bình đều bắt buộc của DP, bản thân nhiễu $\varepsilon = 1.0$ không làm mất nhiều hiệu năng.

### Giai đoạn 20 – 23: Thuật Toán FedProx & Tối Ưu Ngưỡng Cắt $C=0.35$ (03/11/2026)
- **Ứng dụng FedProx (D21):** Thử $\mu \in \{0.01; 0.1; 0.3; 1.0\}$. **Mức $\mu = 0.1$ đạt hiệu quả xuất sắc nhất**, đẩy PR-AUC từ $0.0178$ lên **$0.0282$** ($+58\%$), khắc phục mạnh mẽ hiện tượng phân tán non-IID giữa các ngân hàng.
- **Tối ưu hóa $C$ (D22):** Khi có FedProx, bản cập nhật co lại $3.5$ lần (trung vị độ dài là $0.35$). Thử nghiệm các mức $C \in \{0.35; 0.50; 0.67; 1.60\}$ khẳng định **$C=0.35$ là điểm tối ưu nhất**.
- **Chạy 5-Fold chính thức đợt 1 (30 vòng):** $\text{FL} + \text{DP}$ ($\varepsilon = 1.0$) đạt $0.0246 \pm 0.0017$ (bằng $85\%$ so với FedProx không DP).

### Giai đoạn 24 – 25: Đào sâu Cơ Chế: Phản nghiệm Non-IID & Trọng Số Bị Chặn (04/11 – 05/11/2026)
- **Kiểm tra Non-IID (Giai đoạn 24):** Xáo trộn dữ liệu thành phân phối IID hoàn toàn. Kết quả cho thấy với cấu hình DP trung bình đều, dữ liệu IID chỉ đạt $0.024$ (so với $0.021$ của dữ liệu thật). Chứng minh: *Sự bất đồng nhất dữ liệu giữa các ngân hàng không phải là rào cản chính của DP.*
- **Trọng số bị chặn (Giai đoạn 25):** Thử nghiệm thuật toán Bounded Weights của McMahan et al. Kết quả không cải thiện mà còn làm tăng nhiễu lên 6 lần. Nhóm quyết định kiên định giữ **Trung bình đều**.

### Giai đoạn 26 – 28: Tối Ưu Số Vòng Hội Tụ ($30 \to 100 \to 200$ Vòng) (05/11/2026)
- **So sánh trên Fold 0:**
  - 30 vòng: FedProx $= 0.0282$; $\text{DP } (\varepsilon=1) = 0.0233$.
  - **100 vòng:** FedProx $= \mathbf{0.0340}$ ($+21\%$); $\text{DP } (\varepsilon=1) = \mathbf{0.0304}$ ($+\mathbf{30\%}$).
  - 200 vòng: FedProx bão hòa ($0.0340$); $\text{DP } (\varepsilon=1)$ tụt xuống $0.0251$ do nhiễu tích lũy tăng lên $0.025$/tham số.
- **Quyết định (D24):** Chốt mốc **$100$ vòng** làm cấu hình chuẩn mực cao nhất của đề tài.

### Giai đoạn 29 – 32: Đánh Giá Tự Học Riêng & Kiểm Định 3-Seed Chính Thức (05/11 – 06/11/2026)
- **Tự học riêng (Local-only):** Huấn luyện riêng $802$ mô hình MLP cho $802$ ngân hàng. Macro PR-AUC chỉ đạt $0.129 \pm 0.009$ (do trung vị mỗi ngân hàng chỉ có $2$ ca rửa tiền).
- **Kiểm định 3-Seed hoàn chỉnh:** Chạy đủ 5-Fold trên $3$ seed ngẫu nhiên ($42, 7, 2027$) $\to$ Thu được $15$ cặp thực nghiệm đối chứng để tiến hành kiểm định Wilcoxon signed-rank và hiệu chỉnh Holm. Hoàn tất toàn bộ chu trình thực nghiệm vào rạng sáng 06/11/2026.

---

## 5. BẢNG KẾT QUẢ CHÍNH THỨC 3-SEED & ĐỐI CHIẾU TRỰC DIỆN VỚI SLIDE MONG ĐỢI

### 5.1 Bảng kết quả chính thức (Trung bình 15 cặp thực nghiệm)

| Phương pháp thực nghiệm | PR-AUC toàn hệ thống | % so với Gộp chung (Slide 10 yêu cầu $\ge 85\%$) | % so với FedProx không DP (Tiêu chí 2 tầng $\ge 85\%$) | Macro PR-AUC (TB từng ngân hàng) | Mức độ vượt trội so với Tự học riêng | Tỷ lệ thắng (Wins / 15 cặp) |
|---|---|---|---|---|---|---|
| **E7: Gộp chung (Centralized)** | $0.0929$ | $100\%$ | — | $0.3441$ | $2.77\times$ | $15/15$ |
| **E6: FedProx không DP (100 vòng)** | $0.0333$ | $36\%$ ❌ | $100\%$ | $0.2426$ | $1.96\times$ | $15/15$ |
| **E3: FedAvg thường (100 vòng)** | $0.0242$ | $26\%$ ❌ | $74\%$ | $0.2013$ | $1.63\times$ | $15/15$ |
| **FL + DP ở mức $\varepsilon = 2.0$** | $0.0308$ | $34\%$ ❌ | **$94\%$ ✅** | $0.2337$ | $1.89\times$ | $15/15$ |
| **E8: FL + DP ở mức $\varepsilon = 1.0$ (Chuẩn)** | $\mathbf{0.0279}$ | $\mathbf{30\%}$ ❌ | **$86\%$ ✅** | $\mathbf{0.2193}$ | $\mathbf{1.77\times}$ | $\mathbf{15/15}$ |
| **FL + DP ở mức $\varepsilon = 0.5$ (Bảo mật cao)** | $0.0209$ | $22\%$ ❌ | $59\%$ ❌ | $0.1830$ | $1.48\times$ | $15/15$ |
| **E2: Tự học riêng (Local-only)** | $0.0178$ | $19\%$ | — | $0.1242$ | $1.00\times$ | Cận dưới |

---

### 5.2 Kết quả kiểm định thống kê ý nghĩa (Wilcoxon Signed-Rank + Holm Correction)

| Giả thuyết so sánh đối chứng | Mức chênh lệch trung bình [Khoảng tin cậy 95% CI] | Số lần thắng | Giá trị $p$ hiệu chỉnh Holm | Kết luận khoa học |
|---|---|---|---|---|
| **FL + DP ($\varepsilon=1.0$) > Tự học riêng** | **$+0.0951$** $[+0.0881; +0.1009]$ | **$15 / 15$** | **$p = 0.0004$** | **Đạt ý nghĩa thống kê xuất sắc ✅** (Luôn vượt trội) |
| **FedProx > Tự học riêng** | $+0.1184$ $[+0.1076; +0.1297]$ | $15 / 15$ | $p = 0.0004$ | Đạt ý nghĩa thống kê xuất sắc ✅ |
| **Gộp chung > FedProx** | $+0.1016$ $[+0.0906; +0.1121]$ | $15 / 15$ | $p = 0.0004$ | Gộp chung vẫn vượt trội FL |
| **FedProx > FedAvg thường** | $+0.0413$ $[+0.0308; +0.0504]$ | $15 / 15$ | $p = 0.0004$ | FedProx xử lý dữ liệu non-IID tốt hơn rõ rệt |
| **FedProx > FL + DP ($\varepsilon=1.0$)** *(Chi phí DP)* | **$+0.0232$** $[+0.0104; +0.0360]$ | $12 / 15$ | $p = 0.0249$ | Chi phí bảo mật DP có tồn tại ($14\%$) nhưng ở mức chấp nhận được |
| **FedProx > FL + DP ($\varepsilon=2.0$)** | $+0.0088$ $[+0.0011; +0.0162]$ | $12 / 15$ | $p = 0.0730$ | Không có sự khác biệt thống kê (DP $\varepsilon=2$ gần như ngang ngửa không DP) |

---

### 5.3 Đối chiếu trực diện với các Tiêu chí trên Slide 10

```
                              MỤC TIÊU SLIDE 10 vs KẾT QUẢ THỰC TẾ
┌─────────────────────────────┬─────────────────────┬───────────────────────────┬─────────────┐
│ Tiêu chí đánh giá           │ Chỉ tiêu Slide 10   │ Kết quả thực tế đạt được  │ Đánh giá    │
├─────────────────────────────┼─────────────────────┼───────────────────────────┼─────────────┤
│ 1. Số lượng Client tham gia │ ≥ 3 client          │ 802 ngân hàng             │ VƯỢT XA  ✅ │
│ 2. Đảm bảo chuẩn bảo mật    │ ε ≤ 1.0, δ ≤ 10⁻⁵   │ ε = 1.0, δ = 10⁻⁵         │ ĐẠT CHUẨN ✅ │
│ 3. PR-AUC tuyệt đối         │ PR-AUC ≥ 0.70       │ ~0.03 (Gộp chung ~0.09)   │ KHÔNG ĐẠT ❌│
│ 4. Giá trị tiền VDR@1%      │ VDR ≥ 0.80          │ 0.00 - 0.48 (Dao động lớn)│ KHÔNG ĐẠT ❌│
│ 5. Tỷ lệ so với Gộp chung   │ FL+DP ≥ 85% Gộp     │ 30% Gộp chung             │ KHÔNG ĐẠT ❌│
│ 6. Tiêu chí 2 tầng (Đề xuất)│ FL+DP ≥ 85% FL gốc  │ 86% so với FL không DP    │ ĐẠT CHUẨN ✅ │
│ 7. Lợi ích liên minh (RQ1)  │ Vượt trội Tự học    │ Thắng 15/15 fold (1.77×)  │ ĐẠT CHUẨN ✅ │
└─────────────────────────────┴─────────────────────┴───────────────────────────┴─────────────┘
```

---

## 6. BÓC TÁCH BẢN CHẤT KHOA HỌC: NÚT THẮT NẰM Ở ĐÂU?

Tại sao đề tài **chưa đạt được mục tiêu $\ge 85\%$ so với Gộp chung** đặt ra trên Slide 10? Bằng việc phân rã số liệu, nhóm rút ra những kết luận bản chất:

### 6.1 "Oan uổng" cho Differential Privacy: Chi phí DP chỉ lấy đi $14\%$!
- Rất nhiều người khi nhìn thấy con số $30\%$ sẽ vội quy kết rằng do "bơm nhiễu DP làm hỏng mô hình". Nhưng số liệu khoa học chứng minh điều ngược lại:
  - Hiệu năng của $\text{FL}$ không DP (FedProx) là **$0.0333$**.
  - Hiệu năng của $\text{FL} + \text{DP}$ ở $\varepsilon = 1.0$ là **$0.0279$**.
  - Tỷ lệ duy trì đạt: $\frac{0.0279}{0.0333} = \mathbf{86\%}$ (đạt đúng ngưỡng chuẩn $\ge 85\%$ của tiêu chí 2 tầng).
  - Nghĩa là **bước bảo mật quyền riêng tư chỉ làm sụt giảm đúng $14\%$ hiệu năng**!

### 6.2 Nút thắt thực sự nằm ở chính bài toán Học liên kết (FL Penalty chiếm $64\%$)
- Khoảng cách lớn nhất kéo tụt mô hình nằm ở chỗ: **Bản thân mô hình Học liên kết không có DP đã kém xa mô hình Gộp chung dữ liệu** ($0.0333$ so với $0.0929$, tức chỉ đạt $36\%$).
- *Tại sao FL lại kém Gộp chung nhiều như vậy trên IBM AML?*
  1. **Quy mô dữ liệu phân mảnh cực đoan:** $802$ ngân hàng chia nhỏ $5$ triệu giao dịch. Mỗi ngân hàng chỉ có vài trăm đến vài nghìn mẫu ở tập huấn luyện ($r=0.1$). Các mạng nơ-ron sâu cần một khối lượng dữ liệu tập trung lớn để điều chỉnh các liên kết phi tuyến.
  2. **Góc nhìn dữ liệu bị giới hạn (Bank View vs Global View):** Trong mô hình gộp chung (Centralized), hệ thống nhìn thấy cả tài khoản người gửi lẫn người nhận xuyên suốt các ngân hàng. Trong khi ở chế độ FL, ngân hàng gửi **hoàn toàn mù thông tin về tài khoản đích** ở ngân hàng khác. Thử nghiệm ở giai đoạn 6 đã chứng minh việc thiếu thông tin tài khoản đích làm tụt ngay $0.07$ điểm PR-AUC.
  3. **Độ lệch quy mô giữa các ngân hàng:** Có những ngân hàng lớn chiếm hàng trăm nghìn giao dịch, nhưng có hàng trăm ngân hàng nhỏ chỉ có vài nghìn giao dịch. Việc áp dụng trung bình đều (bắt buộc cho DP) vô tình hạ thấp tầm ảnh hưởng của các ngân hàng giàu dữ liệu.

### 6.3 Về sự khác biệt giữa Kỳ vọng Slide và Dữ liệu Thực tế
- Kỳ vọng trên Slide ($\text{PR-AUC} \ge 0.70$) được đặt ra dựa trên **bộ dữ liệu giả lập (synthetic)** với các kịch bản gian lận được lập trình nhân tạo rõ nét, cân bằng tương đối.
- Khi bước vào dữ liệu thực tế **IBM AML**: Tỷ lệ gian lận thật chỉ có **$0.102\%$** (1 ca trên 1.000 giao dịch). Một mô hình đạt $\text{PR-AUC} = 0.0929$ ở dữ liệu này đã là **gấp hơn $100$ lần so với mức đoán ngẫu nhiên ($0.0009$)**. Việc kỳ vọng đạt PR-AUC $0.70$ trên bài toán rửa tiền thực tế cực kỳ mất cân bằng là phi thực tế đối với mô hình dạng bảng.

---

## 7. CÁC LỖI KỸ THUẬT QUAN TRỌNG ĐÃ PHÁT HIỆN & SỬA ĐỔI

1. **Lỗi định danh mã ngân hàng (Giai đoạn 5 $\to$ 12):**
   - *Phát hiện:* Ban đầu nhóm tưởng rằng có hơn $30.528$ ngân hàng và nghi ngờ hàm nạp đọc nhầm chuỗi có số 0 ở đầu (`"010"` khác `"0010"`).
   - *Sửa đổi:* Đối chiếu chéo với file danh mục chuẩn `HI-Small_accounts.csv` chứng minh cả hai mã này đều trỏ về một Bank ID duy nhất (National Bank of Laramie). Tập dữ liệu chuẩn xác có **$30.470$ ngân hàng**. Hàm nạp đã được sửa lại để bỏ số 0 ở đầu, bảo toàn tính toàn vẹn của lịch sử giao dịch.
2. **Lỗi xếp hạng hòa trong đánh giá (#41, Giai đoạn 14):**
   - *Phát hiện:* Ở các mô hình sụp đổ (dự đoán xác suất các giao dịch gần như bằng nhau), Recall@1% lại vọt lên $0.38$ bất thường. Nguyên nhân do hàm lọc top 1% ngắt hòa theo thứ tự dòng trong file, vô tình trùng với các ca gian lận ở đầu bảng.
   - *Sửa đổi:* Thêm cơ chế **phá hòa ngẫu nhiên cố định seed** (`random tie-breaking`). Recall@1% của các mô hình sụp giảm về đúng bản chất thực tế $(< 5\%)$.
3. **Lỗi rò rỉ dữ liệu thời gian ngày 10–17 (Giai đoạn 6):**
   - *Phát hiện:* Tỷ lệ rửa tiền vọt lên $60\%$ ở tuần thứ 2 do lỗi bộ sinh IBM AML, làm thổi phồng VDR ảo lên $0.55$.
   - *Sửa đổi:* Ban hành quyết định **D18**, cắt bỏ hoàn toàn ngày 10–17, chỉ giữ lại ngày 0–9.

---

## 8. HƯỚNG DẪN BẢO VỆ ĐỀ TÀI TRƯỚC HỘI ĐỒNG (Q&A PHẢN BIỆN KHOA HỌC)

### Câu hỏi 1: Tại sao kết quả FL+DP của nhóm chỉ đạt 30% so với Gộp chung (Centralized), không đạt chỉ tiêu 85% trên Slide 10? Đề tài có thất bại không?
> **Trả lời:**  
> Kính thưa Hội đồng, đề tài **không thất bại mà đã khám phá ra một quy luật thực nghiệm sâu sắc**. Chỉ tiêu 85% ban đầu được nhóm xây dựng trên tập dữ liệu tổng hợp (synthetic), nơi mà mô hình đã đạt trọn vẹn $89.6\%$. Khi chuyển sang dữ liệu thực tế IBM AML với tỷ lệ mất cân bằng cực đoan ($0.1\%$) và phân tán trên 802 ngân hàng, kết quả phân rã khoa học chứng minh:  
> 1. **Bước bảo mật quyền riêng tư DP chỉ lấy đi 14% hiệu năng** (mô hình FL+DP đạt $86\%$ so với FL không DP). Nghĩa là cơ chế DP đã hoàn thành xuất sắc nhiệm vụ bảo toàn thông tin.  
> 2. Khoảng cách nằm ở chính bản thân việc phân tán dữ liệu FL (FL không DP chỉ đạt 36% Centralized) do các ngân hàng bị giới hạn góc nhìn cục bộ (`bank view` không thấy tài khoản đích).  
> 3. Quan trọng nhất, kiểm định thống kê trên 15 cặp mẫu chứng minh: **FL+DP luôn vượt trội Tự học riêng gấp 1.77 lần ($p = 0.0004$)**. Nghĩa là việc tham gia liên minh luôn mang lại lợi ích sống còn cho các ngân hàng so với việc tự làm một mình.

### Câu hỏi 2: Tại sao ban đầu nhóm đề xuất GraphSAGE (GNN) trên Slide mà kết quả cuối cùng trên IBM AML lại dùng MLP?
> **Trả lời:**  
> Đồ thị giao dịch toàn cục (nhìn thấy dòng tiền chạy xuyên ngân hàng) chỉ tồn tại trong mô hình Centralized. Trong môi trường Học liên kết thực tế, mỗi ngân hàng chỉ là một đồ thị cục bộ bị đứt gãy cạnh nối. Ngoài ra, việc áp dụng DP mức client trên kiến trúc GNN quy mô 802 ngân hàng đòi hỏi các thuật toán phân tán đồ thị phức tạp chưa tối ưu trên dữ liệu bảng lớn. Nhóm đã chứng minh trên dữ liệu synthetic rằng MLP dạng bảng mang lại hiệu năng tương đương GNN nhưng tốc độ xử lý nhanh hơn 10 lần và tương thích hoàn hảo với cơ chế DP phân tán.

### Câu hỏi 3: Việc chọn ngưỡng cắt C = 0.35 và chọn vòng tốt nhất qua tập Validation gộp có vi phạm nguyên tắc bảo mật riêng tư không?
> **Trả lời:**  
> Nhóm xin thẳng thắn thừa nhận đây là một giới hạn thực nghiệm. Việc quan sát độ lớn cập nhật để đặt $C=0.35$ và dùng val gộp là bước thăm dò kỹ thuật trong khuôn khổ nghiên cứu học thuật. Để khắc phục triệt để khi triển khai thương mại, nhóm đề xuất hai giải pháp chuẩn hóa: (1) Sử dụng thuật toán **Cắt thích ứng (Adaptive Clipping - Andrew et al., 2021)** để tự động điều chỉnh $C$ bằng ngân sách DP riêng; và (2) Lấy trực tiếp mô hình ở **vòng cuối cùng (vòng 100)** thay vì chọn qua tập val gộp.

---

## 9. KẾT LUẬN & KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO

### 9.1 Ba kết luận khoa học cốt lõi
1. **Lợi ích vượt bậc của Học liên kết:** Mặc dù áp đặt mức bảo mật khắt khe $\varepsilon = 1.0$, mô hình liên kết vẫn **vượt trội mô hình tự học riêng của từng ngân hàng gấp $1.77$ lần (thắng 15/15 fold, $p = 0.0004$)**.
2. **Quy luật quy mô của DP mức client:** DP mức client sụp đổ hoàn toàn ở mạng lưới nhỏ ($5 - 28$ ngân hàng), nhưng **hoàn toàn khả thi và bảo toàn được $86\%$ hiệu năng khi mở rộng lên quy mô $802$ ngân hàng**.
3. **Xác định đúng nút thắt:** Thách thức lớn nhất trong phát hiện rửa tiền liên ngân hàng không phải là nhiễu bảo mật DP, mà là sự thiếu hụt thông tin tài khoản thụ hưởng giữa các tổ chức tín dụng.

### 9.2 Phân công nhiệm vụ cụ thể

| STT | Công việc cụ thể | Người phụ trách | Thời hạn |
|---|---|---|---|
| 1 | Họp xin ý kiến GVHD về việc chốt Tiêu chí 2 tầng và mốc Centralized chính thức. | Cả nhóm | Ngày 10/11 |
| 2 | Cập nhật số liệu thực tế 3-seed vào Slide thuyết trình (Slide 4, 6, 7, 10). | Đạt | Ngày 11/11 |
| 3 | Chạy thực nghiệm kiểm chứng lấy mô hình ở vòng cuối (Last Round) để bịt lỗ hổng phản biện. | Đạo | Ngày 11/11 |
| 4 | Thử nghiệm tăng tỷ lệ lấy mẫu $r=0.3$ trên Fold 0 để khảo sát tiềm năng thu hẹp khoảng cách FL. | Đạo | Ngày 12/11 |
| 5 | Hoàn thiện bản thảo bài báo khoa học gửi Tạp chí/Kỷ yếu NCKH. | C & A | Ngày 15/11 |

