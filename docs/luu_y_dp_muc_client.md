# Lưu ý: giữ DP mức client

Quyết định ngày 02/11/2026: **đề tài dùng DP mức client** (đúng thiết kế Slide 4), không chuyển sang DP mức bản ghi
như đề xuất D12 cũ. Mục đích là giữ bài toán ở dạng tổng quát. Người chốt: ............

Ai làm phần DP, sửa slide hoặc viết bài báo cần đọc tài liệu này trước.

---

## 1. Vì sao chọn mức client

- **Bảo vệ được cả một bên tham gia.** Không ai suy ra được ngân hàng nào đóng góp gì vào mô hình chung, và tất nhiên
  cũng không suy ra được từng giao dịch hay từng khách hàng bên trong ngân hàng đó.
- **Không phụ thuộc mô hình.** Chỉ cắt và thêm nhiễu vào bản cập nhật cuối của ngân hàng, nên áp dụng được cho
  **GraphSAGE**. DP mức bản ghi thì không áp dụng được cho GNN (mục 4.3 của BAO_CAO_TONG_HOP).
- **Hiện thực hiện tại không cần tin server.** Trong code hiện tại, mỗi ngân hàng tự cộng nhiễu trước khi gửi
  (`privatize_update` trong `src/fraudfl/privacy.py`), nên server tò mò cũng không đọc được gì thêm.
- **Bảo vệ cả khách hàng, không chỉ giao dịch.** Khách hàng nằm trọn trong một ngân hàng, nên đã che được ngân hàng
  thì cũng che được mọi khách hàng của ngân hàng đó. Vấn đề "một khách có nhiều giao dịch" của DP mức bản ghi không
  còn.

## 2. Vấn đề phải biết: với cấu hình hiện tại, mô hình sụp

Kết quả đã chạy (GraphSAGE, 5 ngân hàng, 50 vòng, fold 0):

| ε | Hệ số nhiễu z | PR-AUC |
|---|---|---|
| 2 | 15,2 | 0,028 |
| 1 | 28,8 | 0,028 |
| 0,5 | 55,0 | 0,028 |

PR-AUC 0,028 xấp xỉ mức đoán ngẫu nhiên (tỷ lệ gian lận khoảng 1,5%). Nghĩa là **với cấu hình hiện tại, tiêu chí
"học liên kết + DP ≥ 85% so với gộp chung, ε ≤ 1" của Slide 10 không đạt.**

Nguyên nhân: nhiễu phải đủ che một ngân hàng, nhưng tín hiệu chỉ là tổng của 5 ngân hàng.
- Mỗi bản cập nhật đã cắt có độ lớn ≤ C = 1, nên tín hiệu tối đa khoảng 5.
- Nhiễu cần cho ε = 1 sau 50 vòng là 28,8.
- Tín hiệu chỉ bằng khoảng 0,17 lần nhiễu.

Trong các nghiên cứu đã công bố, DP mức client chủ yếu dùng khi có hàng trăm đến hàng triệu bên tham gia.

**Thêm một điểm trong code.** Vì mỗi ngân hàng tự cộng nhiễu, tổng nhiễu sau khi gộp lớn gấp √K ≈ 2,2 lần so với
DP-FedAvg chuẩn (McMahan et al., 2018). Trong DP-FedAvg chuẩn, server cộng nhiễu một lần duy nhất vào tổng. Cách
hiện tại an toàn hơn (không cần tin server) nhưng nhiễu nhiều hơn.

## 3. Các cách làm cho DP mức client dùng được

Hệ số nhiễu z dưới đây tính bằng RDP accountant của Opacus, δ = 1e-5, ngày 02/11/2026.

**Bảng z theo ε và số vòng (5 ngân hàng, mọi ngân hàng tham gia mọi vòng):**

| ε | 50 vòng | 20 vòng | 10 vòng | 5 vòng |
|---|---|---|---|---|
| 1 | 28,8 | 18,1 | 12,8 | 9,1 |
| 2 | 15,2 | 9,6 | 6,8 | 4,8 |
| 4 | 8,2 | 5,2 | 3,7 | 2,6 |
| 8 | 4,5 | 2,9 | 2,0 | 1,4 |

Với 5 ngân hàng, tín hiệu tối đa là khoảng 5. Tín hiệu chỉ bắt đầu lớn hơn nhiễu khi z < 5. Ở ε ≤ 1, không cấu hình
nào trong bảng đạt được điều này.

| # | Cách | Ảnh hưởng | Đánh đổi |
|---|---|---|---|
| a | **Tăng số bên tham gia** (mô phỏng 20, 50, 100 ngân hàng) | Tín hiệu tăng theo K. Có thể chọn ngẫu nhiên một phần ngân hàng mỗi vòng để được khuếch đại nhờ lấy mẫu con: K = 100, mỗi vòng 10 ngân hàng, 50 vòng, ε = 1 chỉ cần z ≈ 3,2 | Phải thêm cấu hình nhiều ngân hàng vào `config.py`. Mỗi ngân hàng ít dữ liệu hơn |
| b | **Giảm số vòng, tăng epoch cục bộ** | 50 → 10 vòng: z giảm từ 28,8 xuống 12,8 ở ε = 1 | Có thể chưa hội tụ. Riêng cách này chưa đủ |
| c | **Server cộng nhiễu một lần (hoặc dùng SecAgg)** | Tổng nhiễu giảm √K lần | Phải tin server, hoặc phải cài SecAgg |
| d | **Báo cáo thêm ε lớn hơn (2, 4, 8)** | ε = 8, 10 vòng: z ≈ 2 | Không đạt "ε ≤ 1" của Slide 10 |
| e | **Cắt thích ứng** (adaptive clipping, Andrew et al., 2021) | C bám theo độ lớn thật của bản cập nhật, nên ít mất tín hiệu khi cắt | Tốn thêm một ít ngân sách riêng tư |

**Đề xuất:** làm thí nghiệm **quét số ngân hàng K ∈ {5, 20, 50, 100}** (cách a), kết hợp b và c. Kết quả sẽ trả lời
câu hỏi tổng quát: **"Cần bao nhiêu ngân hàng tham gia thì DP mức client mới dùng được?"** Câu hỏi này đúng tinh thần
"bài toán tổng quát", và có ý nghĩa thực tế, ví dụ một hiệp hội có nhiều ngân hàng nhỏ.

## 4. Việc phải làm ngay

1. **Chốt lại tiêu chí Slide 10 trước khi chạy thêm.** Chọn một trong ba hướng:
   - Giữ "≥ 85% ở ε ≤ 1" và chấp nhận với 5 ngân hàng là không đạt. Báo cáo đây là kết quả âm.
   - Đổi điều kiện thành "đạt khi K ≥ ...", hoặc nới ε.
   - Giữ cả hai mức DP: mức client là kết quả chính, mức bản ghi là phương án so sánh.

   **Không được đổi tiêu chí sau khi đã thấy kết quả mới.**
2. **Ghi vào `docs/decisions.md`:** D12 đã chốt là mức client. Thêm D16 cho cấu hình thí nghiệm quét K.
3. **Kế hoạch tuần 2–3:**
   - Thêm thí nghiệm quét K và thử cách b, c.
   - Thí nghiệm ngân hàng nhỏ (E11) vẫn giữ.
   - Phần DP-SGD mức bản ghi chuyển thành phương án so sánh, nếu nhóm đồng ý.
4. **Slide 4 giữ nguyên.** Slide 7 và Slide 10 cần sửa theo hướng chọn ở bước 1.
5. **Cách tính hiện tại (`sample_rate = 1, steps = số vòng`) là đúng** khi mọi ngân hàng tham gia mọi vòng. Nếu chọn
   ngẫu nhiên ngân hàng mỗi vòng (cách a), phải sửa `sample_rate = số ngân hàng mỗi vòng / K`.

## 5. Ghi trong bài báo

- Định nghĩa rõ đơn vị riêng tư: **thêm hoặc bớt một ngân hàng**. Nói rõ mô hình đe dọa: server không tin cậy (với
  cách mỗi ngân hàng tự cộng nhiễu như hiện tại), hoặc server tin cậy (nếu chuyển sang cách c).
- Báo cáo trung thực kết quả với 5 ngân hàng (PR-AUC 0,028). Đây là bằng chứng thực nghiệm cho giới hạn của DP mức
  client khi có ít bên tham gia, và là đóng góp có giá trị.
- Nếu làm thí nghiệm quét K: vẽ đồ thị PR-AUC theo K ở từng mức ε, và chỉ ra K tối thiểu để đạt 85% so với gộp chung.

## Tài liệu tham khảo

- McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning Differentially Private Recurrent Language
  Models. *ICLR*.
- Andrew, G., Thakkar, O., McMahan, H. B., & Ramaswamy, S. (2021). Differentially Private Learning with Adaptive
  Clipping. *NeurIPS*.
- Mironov, I. (2017). Rényi Differential Privacy. *IEEE CSF*.

C (Research lead) cần kiểm tra lại thông tin trích dẫn trước khi đưa vào bài báo.
