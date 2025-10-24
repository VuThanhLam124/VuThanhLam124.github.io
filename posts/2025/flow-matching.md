---
title: "Flow Matching: Từ Likelihood đến Regression"
date: "2025-01-20"
category: "flow-based-models"
tags: ["flow-matching", "optimal-transport", "generative-models", "regression", "pytorch"]
excerpt: "Từ đơn hàng 100 con rồng với deadline 7 ngày, người thợ gốm khám phá ra Flow Matching - phương pháp cách mạng biến bài toán likelihood phức tạp thành regression đơn giản. Hành trình từ CNF đắt đỏ đến CFM hiệu quả, với câu chuyện sinh động, toán học đầy đủ và code PyTorch thực chiến."
author: "ThanhLamDev"
readingTime: 45
featured: true
---

# Flow Matching: Từ Likelihood đến Regression

**Khi Người Thợ Gốm Tìm Ra Cách Học Đơn Giản Hơn**

Chào mừng trở lại với series Flow-based Models! Trong bài [Normalizing Flows & CNF](/posts/2025/normalizing-flows), chúng ta đã gặp người thợ gốm bậc thầy - người biến khối đất sét đơn giản thành tác phẩm nghệ thuật phức tạp qua **dòng chảy liên tục** $\frac{dz(t)}{dt} = v_t(z(t))$.

Nhưng cách học của người thợ trong CNF/FFJORD gặp phải vấn đề nghiêm trọng: **quá phức tạp và đắt đỏ**. Flow Matching ra đời để giải quyết vấn đề này bằng một insight thiên tài: **Thay vì học likelihood, hãy học vector field trực tiếp qua regression!**

## Mục lục

1. [Vấn đề của CNF: Tại sao Maximum Likelihood đắt đỏ?](#1-vấn-đề-của-cnf)
2. [Flow Matching: Từ Likelihood sang Regression](#2-flow-matching-từ-likelihood-sang-regression)
3. [Conditional Flow Matching (CFM)](#3-conditional-flow-matching-cfm)
4. [Optimal Transport Conditional Flow Matching](#4-optimal-transport-conditional-flow-matching)
5. [Chứng minh tính đúng đắn](#5-chứng-minh-tính-đúng-đắn)
6. [Implementation PyTorch đầy đủ](#6-implementation-pytorch-đầy-đủ)
7. [So sánh CNF, Diffusion và Flow Matching](#7-so-sánh-cnf-diffusion-và-flow-matching)
8. [Kết luận](#8-kết-luận)

---

## 1. Vấn đề của CNF

### Nhớ lại: Người thợ gốm với CNF

Trong bài trước, người thợ gốm học cách biến đổi bằng **Continuous Normalizing Flow**:

- **Phương trình ODE**: $\frac{dz(t)}{dt} = v_\theta(z(t), t)$
- **Mục tiêu**: Tối đa hóa likelihood $$\max_\theta \mathbb{E}_{x \sim p_{\text{data}}} [\log p_\theta(x)]$$
- **Công thức tính likelihood** (đã học trong bài CNF):

$$
\log p_\theta(x_1) = \log p_0(x_0) - \int_0^1 \text{Tr}\left(\frac{\partial v_\theta}{\partial x}(x(t), t)\right) dt
$$

### Câu chuyện: Buổi sáng bận rộn tại xưởng gốm

Một buổi sáng thứ Hai tháng Ba, ánh nắng len lỏi qua các ô cửa sổ nhỏ của xưởng gốm già nua bên bờ sông. Người thợ gốm bậc thầy vừa mở cửa xưởng, chưa kịp thưởng thức tách trà xanh buổi sáng, thì chuông cửa đã reo vang. Một vị khách mặc vest lịch sự, đại diện cho hội nghị nghệ thuật thành phố, bước vào với ánh mắt khẩn trương.

"Thưa ông," vị khách cúi chào, "chúng tôi cần **100 tác phẩm con rồng** giống hệt nhau làm quà tặng cho các khách mời trong hội nghị tuần sau. Tôi nghe nói ông là người duy nhất có thể tạo ra những con rồng đất nung hoàn hảo với kỹ thuật dòng chảy liên tục. Chúng tôi sẵn sàng trả mức giá tốt nhất."

Người thợ gốm gật đầu, nhìn ra kho chứa nơi có hơn 150 khối đất sét hình cầu đều đặn đã chuẩn bị sẵn - di sản từ bài học về **Continuous Normalizing Flows** mà anh đã thực hành trong nhiều tháng qua. Anh biết mình có thể làm được, nhưng với phương pháp CNF hiện tại, công việc sẽ cực kỳ vất vả.

"Tôi nhận," anh nói, nhưng trong lòng đã bắt đầu tính toán khối lượng công việc khổng lồ phía trước.

**Ngày 1 - Bắt đầu với phương pháp CNF truyền thống**

Người thợ gốm lấy ra 100 khối đất sét hình cầu giống nhau, mỗi khối nặng khoảng 2kg, đường kính 15cm. Anh biết với kỹ thuật **Maximum Likelihood** của CNF, anh phải thực hiện một quy trình phức tạp đến từng chi tiết:

**Bước 1 - Quan sát và ghi nhớ vi mô (Maximum Likelihood):**

Với mỗi khối đất sét, anh không chỉ đơn thuần nặn. Anh phải:

1. **Đánh dấu từng hạt đất sét**: Dùng kim nhỏ, anh đánh dấu 1000 điểm khác nhau trên bề mặt khối cầu - từ tâm đến vỏ ngoài. Mỗi điểm như một "tọa độ" mà anh cần theo dõi.

2. **Ghi lại quỹ đạo từng hạt**: Trong quá trình biến đổi từ khối cầu → con rồng, anh phải ghi chép vào sổ tay dày cộm: "Hạt A1 tại tâm khối cầu, sau 6 phút (t=0.1) di chuyển đến vị trí (0.5, 0.2, 0.1) - bắt đầu kéo dài để tạo thân rồng. Sau 12 phút (t=0.2) đến (1.2, 0.5, 0.3) - tiếp tục kéo..."

3. **Tính toán độ giãn nở**: Điều khủng khiếp nhất - tại TỪNG thời điểm, anh phải đo "độ giãn nở" của đất sét (trace của Jacobian matrix). Làm sao? Bằng cách đo thể tích một vùng nhỏ quanh điểm, so sánh với thể tích ban đầu.

**Ví dụ cụ thể với hạt đất A1 trên con rồng đầu tiên:**

```
Thời gian t=0.00 (0 phút):    Hạt A1 ở (0.0, 0.0, 0.0) - tâm khối cầu
         t=0.10 (6 phút):    Hạt A1 ở (0.5, 0.2, 0.1) - bắt đầu kéo ra phía thân
         t=0.20 (12 phút):   Hạt A1 ở (1.2, 0.5, 0.3) - thân rồng đang hình thành
         t=0.30 (18 phút):   Hạt A1 ở (2.0, 0.9, 0.5) - thân rồng dài ra
         ...
         t=1.00 (60 phút):   Hạt A1 ở (5.0, 2.0, 1.0) - đuôi rồng hoàn chỉnh

Đồng thời phải ghi độ giãn nở:
- Tại t=0.10: Vùng quanh A1 giãn ra 1.5 lần (det J = 1.5)
- Tại t=0.20: Giãn ra 2.3 lần (det J = 2.3)
- Tại t=0.50: Giãn ra 4.1 lần (det J = 4.1)
- ...
```

Cứ mỗi 6 phút (Δt = 0.1), anh phải dừng tay, lấy thước kẹp đo, ghi chép. Với 100 con rồng, mỗi con có 1000 điểm cần theo dõi, tại 10 thời điểm khác nhau trong quá trình 60 phút. **Tổng cộng: 100 × 1000 × 10 = 1,000,000 phép đo!**

**Bước 2 - Tính toán likelihood (ác mộng thực sự):**

Sau khi hoàn thành con rồng đầu tiên, người thợ gốm phải "đánh giá chất lượng" bằng cách tính **xác suất** (likelihood) của tác phẩm. Nhưng đây không phải việc nhìn và cảm nhận - đây là toán học phức tạp:

Để biết con rồng này có "xác suất cao" xuất hiện từ khối cầu Gaussian hay không, anh phải:

```python
# Pseudocode của CNF
def evaluate_dragon_quality(finished_dragon):
    # 1. TÍCH PHÂN NGƯỢC - Tìm lại khối cầu ban đầu
    current = finished_dragon
    log_prob_change = 0
    
    for t in reversed([1.0, 0.9, 0.8, ..., 0.1, 0.0]):  # 100 bước!
        # Tính trace tại MỖI bước
        jacobian = compute_jacobian(velocity_field, current, t)
        trace = hutchinson_trace_estimate(jacobian)  # Đắt!
        log_prob_change -= trace * dt
        
        # Tích phân ngược một bước nhỏ
        current = current - velocity_field(current, t) * dt
    
    # 2. Kiểm tra xem có về đúng khối cầu Gaussian không
    sphere_likelihood = gaussian_log_prob(current)
    
    # 3. Tổng hợp
    dragon_likelihood = sphere_likelihood + log_prob_change
    return dragon_likelihood
```

### Ba vấn đề chết người

**Vấn đề 1: Tính trace cực đắt**

Tại **mỗi thời điểm** $t$, người thợ phải đo "độ giãn nở" của đất sét:

```python
# Ví dụ cụ thể: Đo độ giãn tại t=0.5
hạt_A_ở = (2.5, 1.0, 0.5)
velocity = (1.0, 0.5, 0.2)  # Hướng di chuyển

# Tính Jacobian (ma trận đạo hàm)
#         ∂v₁/∂x₁  ∂v₁/∂x₂  ∂v₁/∂x₃
#   J =   ∂v₂/∂x₁  ∂v₂/∂x₂  ∂v₂/∂x₃
#         ∂v₃/∂x₁  ∂v₃/∂x₂  ∂v₃/∂x₃

# Trace = tổng đường chéo = ∂v₁/∂x₁ + ∂v₂/∂x₂ + ∂v₃/∂x₃
# Để tính cần: backpropagation nhiều lần!

for _ in range(10):  # Hutchinson estimator - 10 samples
    eps = random_vector()
    vjp = autograd.grad(velocity, position, eps)[0]  # Đắt!
    trace_estimate += dot(eps, vjp)
```

**Thống kê thực tế từ sổ tay người thợ gốm:**

Anh mở sổ ghi chép, dưới ánh đèn dầu yếu ớt, và ghi lại những con số đáng sợ:

- Mỗi lần đo và tính trace (độ giãn nở): **~3 phút thủ công** (tương đương ~50ms trong code)
- Cần đo tại 100 thời điểm khác nhau trong quá trình 60 phút: 100 điểm × 3 phút = **300 phút = 5 giờ** cho MỖI con rồng!
- Để hoàn thành đánh giá quality của 100 con rồng: 100 × 5h = **500 giờ = 20 ngày làm việc liên tục!**

"Thế này không ổn," anh lẩm bẩm, nhìn đống đất sét chưa động đến và deadline chỉ còn 7 ngày.

**Vấn đề 2: "Tua ngược" không ổn định - Bi kịch của con rồng thứ 5**

Vấn đề thứ hai xuất hiện khi người thợ cố gắng đánh giá con rồng thứ 5. Để tính likelihood, anh phải "tua ngược" quá trình nặn - như xem một video ngược từ con rồng hoàn chỉnh về lại khối cầu Gaussian ban đầu.

Tưởng tượng bạn có một đoạn video:
```
Video THUẬN (dễ hiểu, tự nhiên):
t=0.0 → Khối cầu tròn trịa
t=0.3 → Kéo dài thành hình ống
t=0.6 → Uốn cong tạo thân và cổ
t=1.0 → Con rồng hoàn chỉnh với đầu, thân, đuôi

Video NGƯỢC (khó, chống lại tự nhiên):
t=1.0 → Con rồng hoàn chỉnh
t=0.6 ← Thu cong, gỡ chi tiết
t=0.3 ← Thu ngắn thành ống
t=0.0 ← Khối cầu (hy vọng!)
        ↑ Nhưng có đúng khối cầu ban đầu không?
```

Người thợ gốm thử nghiệm với con rồng số 5. Anh đặt nó trên bàn xoay, bắt đầu "tưởng tượng ngược" từng bước. Nhưng vấn đề nảy sinh:

**Sai số tích lũy theo cấp số nhân:**

```
Bước ngược 1 (từ t=1.0 về t=0.9):
  Anh ước tính vị trí con rồng ở t=0.9
  Độ chính xác: 99% (1% sai số do ước lượng)

Bước ngược 2 (từ t=0.9 về t=0.8):
  Dựa trên vị trí ước lượng ở bước 1 (đã có 1% sai)
  Thêm 1% sai mới
  Độ chính xác: 99% × 99% = 98.01%

Bước ngược 3 (từ t=0.8 về t=0.7):
  Độ chính xác: 98.01% × 99% = 97.03%

Bước ngược 10:
  Độ chính xác: 0.99^10 ≈ 90.4%

Bước ngược 50:
  Độ chính xác: 0.99^50 ≈ 60.5%

Bước ngược 100 (về t=0.0 - khối cầu):
  Độ chính xác: 0.99^100 ≈ 36.6%
  → Khối cầu cuối cùng SAI LỆCH hơn 60% so với ban đầu!
```

"Con rồng số 5 của tôi cuối cùng 'ngược' về một cái gì đó... không phải khối cầu," người thợ gốm thở dài, nhìn sketch trong sổ tay - khối cầu ước tính bị méo mó, lệch tâm, không đều.

**Hậu quả thực tế:**

Anh thử kiểm chứng: Lấy khối cầu ban đầu đã đánh dấu, so sánh với khối cầu "ngược được" từ con rồng.
- Điểm A ban đầu ở (0, 0, 0), sau khi ngược về ở (0.3, -0.2, 0.1) ← Sai!
- Thể tích khối cầu ngược: 520 cm³, khối cầu gốc: 450 cm³ ← Chênh 15%!

"Nếu không tính được đúng likelihood, làm sao tôi biết mình đang học đúng kỹ thuật?" Đây là câu hỏi ám ảnh người thợ gốm suốt đêm hôm đó.

**Vấn đề 3: "Bước nhỏ vô tận" - Ác mộng của phép tính chính xác**

Vấn đề thứ ba khiến người thợ gốm gần như bỏ cuộc. Anh muốn tính toán chính xác quỹ đạo của từng hạt đất, nên quyết định dùng phương pháp **adaptive ODE solver** (giống thuật toán Runge-Kutta bậc 4-5, hay còn gọi là dopri5).

Ý tưởng ban đầu rất đẹp: "Tôi sẽ chia nhỏ mỗi bước thời gian. Khi nào quỹ đạo cong nhiều (biến đổi phức tạp), chia nhỏ hơn nữa để chính xác. Khi thẳng, bước to cũng được."

Nhưng thực tế? Con quỷ nằm trong chi tiết!

**Điều gì xảy ra với mỗi bước "logic":**

Khi người thợ muốn đi từ t=0.50 đến t=0.51 (một bước nhỏ dt=0.01):

```
Bước LOGIC (dt=0.01):
  Người thợ nghĩ: "Chỉ 1 phép tính velocity"

Thực tế (với RK45 adaptive):
  Micro-bước 1: Tính velocity tại t=0.500 (k1)
  Micro-bước 2: Tính velocity tại t=0.502 (k2) - dùng k1 ước lượng
  Micro-bước 3: Tính velocity tại t=0.505 (k3) - dùng k1,k2 ước lượng
  Micro-bước 4: Tính velocity tại t=0.507 (k4) - dùng k1,k2,k3
  Micro-bước 5: Tính velocity tại t=0.510 (k5) - dùng tất cả trước đó
  Micro-bước 6: Kiểm tra sai số - Nếu > threshold, CHIA NHỎ THÊM!
  
  → Trung bình: 4-8 micro-bước cho MỖI bước logic!
```

**Thống kê khủng khiếp:**

```
Mục tiêu: Từ t=0 → t=1 (quá trình 60 phút)
Số bước logic cần thiết (để đủ chính xác): 100 bước
Mỗi bước logic = 6 micro-steps trung bình

Tổng số lần gọi velocity_field():
  100 bước × 6 micro-steps = 600 lần!

Với 100 con rồng:
  100 rồng × 600 lần × 3 phút mỗi lần = 180,000 phút
  = 3,000 giờ = 125 ngày làm liên tục không ngừng!
```

Người thợ gốm ngồi thừ người ra nhìn con số "125 ngày". Deadline chỉ còn 6 ngày nữa.

### Đêm tư duy - Khi con số nói lên tất cả

Tối hôm đó, người thợ gốm ngồi bên cửa sổ, ánh trăng chiếu vào xưởng lạnh lẽo. Anh mở sổ tay, ghi lại bảng so sánh thực tế với những gì lý tưởng:

| Công việc | CNF/FFJORD (Thực tế) | Lý tưởng (Ước mơ) | Chênh lệch |
|-----------|----------------------|-------------------|------------|
| **Training:** Học kỹ thuật từ 1000 ví dụ | **30 giờ** | 30 phút | **60x chậm hơn** |
| **Generation:** Tạo 1 con rồng mới | **10 phút** | 1 phút | **10x chậm hơn** |
| **Evaluation:** Đánh giá quality 1 tác phẩm | **5 giờ** | Không cần | **Vô nghĩa** |
| **Tổng 100 con rồng** | **~125 ngày** | **7 ngày** | **Không kịp!** |

Anh gạch chân dòng cuối cùng ba lần, mực đỏ thẫm trên giấy vàng.

### Người thợ gốm suy nghĩ - Đêm trằn trọc mang đến insight

Đêm đó, người thợ gốm không ngủ được. Anh đốt ngọn đèn dầu nhỏ, ngồi giữa xưởng, xung quanh là 100 khối đất sét im lìm và những con rồng đã hoàn thành nằm rải rác. Ánh đèn vàng lập lòe tạo bóng đổ kỳ quái lên những tác phẩm gốm.

Anh nhặt lên một con rồng nhỏ vừa hoàn thành - **con rồng số 3**, với chiếc đuôi dài uốn lượn mềm mại, đôi cánh gấp gọn, đầu ngước lên như đang ngước nhìn trời. Anh vuốt nhẹ lên lớp men xanh bóng loáng.

**Câu hỏi lớn:**

> "Ta đã làm con rồng này. Ta BIẾT cách nặn nó.  
> Ta BIẾT tay cần đặt ở đâu, kéo theo hướng nào, bao lâu.  
> Vậy tại sao... ta phải quan tâm đến **XÁC SUẤT** từng hạt đất ở vị trí nào?  
> Tại sao phải TUA NGƯỢC lại từ con rồng về khối cầu để tính likelihood?  
> Tại sao phải ĐO ĐẠC độ giãn nở phức tạp đến vậy?"

Anh đặt con rồng xuống, cầm lên một khối đất sét mới. Lật đi lật lại trong tay, suy nghĩ. Một ý tưởng bắt đầu hình thành - đơn giản đến mức anh tự hỏi tại sao chưa nghĩ ra trước đó.

**Insight thiên tài:**

Mục tiêu cuối cùng của anh là gì? **TẠO RA con rồng**, không phải **TÍNH XÁC SUẤT** của con rồng!

Nếu anh biết được **HƯỚNG NẶN** (velocity) tại mỗi điểm và mỗi thời điểm, thì:

```python
# Quy trình đơn giản!
def create_dragon():
    position = sphere_center  # Khối cầu
    
    for time_step in range(100):
        # Chỉ cần biết: "Tay nên di chuyển theo hướng nào?"
        direction = get_molding_direction(position, time_step)
        
        # Và đơn giản...
        position = position + direction * dt
    
    return position  # Con rồng hoàn thành!
```

**Không cần:**
- Tính trace của ma trận Jacobian phức tạp
- Tích phân ngược ODE với 600 micro-steps
- Đánh giá likelihood với 5 giờ mỗi tác phẩm
- Đo đạc độ giãn nở tại 100 thời điểm

**Chỉ cần:**
- Biết "hướng nặn đúng" tại từng vị trí, từng thời điểm
- Một phép cộng đơn giản: `vị_trí_mới = vị_trí_cũ + hướng × bước`

**Ẩn dụ sâu sắc:**

Người thợ gốm nhớ lại cách mình học nghề thủ công thuở nhỏ:

- Không ai bảo anh: "Tính xác suất tay em ở đây là 73.5%"
- Thay vào đó, thầy nói: "Lúc này, đất sét đang ở giữa chừng, tay em cần đẩy về phía này, mạnh vừa phải"

**Học nghề = Học HƯỚNG DI CHUYỂN, không phải học XÁC SUẤT!**

Ánh sáng bừng lên trong đầu anh. Anh lật giở cuốn sổ cũ, tìm lại những ghi chép về **regression** - phương pháp toán học đơn giản nhất mà anh từng học.

"Nếu ta có thể dạy mô hình học HƯỚNG NẶN qua regression... Thì mọi chuyện sẽ đơn giản hơn CNF rất nhiều!"

**Câu hỏi then chốt mới:**

> **Làm sao biết "hướng nặn đúng" tại mỗi điểm $x$, mỗi thời điểm $t$, mà KHÔNG cần tính likelihood?**

Đây chính là lúc **Flow Matching** - phương pháp cách mạng - xuất hiện trong tâm trí người thợ gốm. Anh không biết thuật ngữ này. Nhưng anh biết đây là con đường đúng.

Anh đứng dậy, với quyết tâm mới. Mai sẽ là một ngày mới. Một cách tiếp cận mới.

---

## 2. Flow Matching: Từ Likelihood sang Regression

### Insight thiên tài: Học trực tiếp "cách nặn"

**Ngày 2 - Buổi sáng thay đổi**

Sáng hôm sau, người thợ gốm thức dậy với tâm trạng khác hẳn. Anh không còn căng thẳng như ngày hôm qua. Ánh nắng sớm len qua cửa sổ, chiếu sáng những con rồng đã hoàn thành - 5 con nằm gọn trên kệ gỗ, mỗi con đều hoàn hảo về hình dáng nhưng mất quá nhiều công sức để tạo ra.

Anh bước đến bàn làm việc, cầm lên khối đất sét số 6, và thay vì bắt đầu quy trình phức tạp CNF như mọi khi, anh dừng lại. Thay vì chuẩn bị sổ ghi chép dày cộm để đo đạc trace và tính likelihood, anh chỉ cầm một tờ giấy trắng nhỏ.

**Thay đổi cách đặt câu hỏi - Bước ngoặt then chốt:**

**Câu hỏi CŨ (theo CNF - Maximum Likelihood):**
> "Nếu tôi có con rồng này (đã hoàn thành), **XÁC SUẤT** để nó xuất hiện từ khối cầu Gaussian ban đầu là bao nhiêu?"
> 
> → Phải tua ngược, tính trace, đo độ giãn nở...
> → Mất 5 giờ mỗi con!

**Câu hỏi MỚI (theo Flow Matching - Direct Learning):**
> "Nếu tôi có một hạt đất đang ở vị trí $x$ tại thời điểm $t$, **NÓ NÊN DI CHUYỂN THEO HƯỚNG NÀO** để trở thành phần của con rồng?"
> 
> → Chỉ cần biết HƯỚNG (velocity vector)!
> → Không cần tính xác suất, không cần tua ngược!

**Thí nghiệm đầu tiên với tư duy mới:**

Người thợ gốm quyết định làm một thử nghiệm đơn giản. Anh đặt con rồng số 3 (đã hoàn thành) bên cạnh khối cầu ban đầu số 6 (chưa nặn). Anh chọn một điểm cụ thể trên khối cầu:

**Setup cho thí nghiệm:**
```
Khối cầu số 6: Tâm tại (0, 0, 0), bán kính 7.5cm
Điểm quan sát: Hạt đất A ở vị trí (1.0, 0.5, 0.2)
Thời điểm hiện tại: t=0.3 (18 phút trong quá trình 60 phút)
Câu hỏi: Hạt A nên di chuyển theo hướng nào?
```

**Phương pháp CŨ (CNF - Khó và phức tạp):**

```
Bước 1: Tìm TẤT CẢ các con rồng đã hoàn thành
Bước 2: Với MỖI con rồng:
  - Tua ngược về thời điểm t=0.3
  - Tìm xem ở t=0.3, có hạt nào ở gần (1.0, 0.5, 0.2) không?
  - Nếu có, nó đang đi về đâu?
Bước 3: Tính XÁC SUẤT mỗi hướng di chuyển
Bước 4: Trung bình có trọng số dựa trên xác suất

→ Phải làm việc với 100+ con rồng
→ Phải tính likelihood mỗi trajectoryMôi trajectory  
→ Mất hàng giờ!
```

**Phương pháp MỚI (Flow Matching - Đơn giản và trực quan):**

```
Bước 1: Quan sát con rồng số 1:
  Khi làm con rồng số 1, tại t=0.3:
  - Hạt ở (1.0, 0.5, 0.2) di chuyển về phía (1.8, 0.9, 0.4)
  - Hướng: (1.8-1.0, 0.9-0.5, 0.4-0.2) = (0.8, 0.4, 0.2)

Bước 2: Quan sát con rồng số 2:
  Tại t=0.3, hạt ở (1.0, 0.5, 0.2) đi về (1.7, 0.8, 0.5)
  - Hướng: (0.7, 0.3, 0.3)

Bước 3: Quan sát con rồng số 3:
  - Hướng: (0.9, 0.5, 0.1)

Bước 4: Học hướng TRUNG BÌNH (regression!):
  Hướng học được = (0.8 + 0.7 + 0.9)/3, (0.4 + 0.3 + 0.5)/3, (0.2 + 0.3 + 0.1)/3
                 = (0.8, 0.4, 0.2) ← Đơn giản là TRUNG BÌNH!

→ Không cần xác suất!
→ Không cần tua ngược phức tạp!
→ Chỉ cần ... CỘNG và CHIA!
```

**Khoảnh khắc Eureka:**

Người thợ gốm đứng bật dậy, tờ giấy nhỏ rơi xuống đất. Trên đó chỉ ghi mấy dòng đơn giản:

```
Con rồng 1: (0.8, 0.4, 0.2)
Con rồng 2: (0.7, 0.3, 0.3)  
Con rồng 3: (0.9, 0.5, 0.1)
Trung bình: (0.8, 0.4, 0.2)

→ Đây là REGRESSION!
→ Đây là BÀI TOÁN LỚP 8!
```

"Regression!" Anh thốt lên trong xưởng vắng. "Tôi chỉ cần dạy mô hình học **dự đoán hướng di chuyển** từ dữ liệu quan sát. Như dạy một đứa trẻ: 'Khi đất sét ở đây, lúc này, hãy đẩy về hướng kia.'"

**Khác biệt về bản chất:**

| Aspect | CNF (Likelihood) | Flow Matching (Regression) |
|--------|------------------|----------------------------|
| **Câu hỏi** | "Xác suất bao nhiêu?" | "Hướng nào?" |
| **Loại bài toán** | Probabilistic inference | Supervised learning |
| **Input → Output** | Data → Probability | (Position, Time) → Direction |
| **Phương pháp học** | Maximum likelihood | Mean Squared Error |
| **Độ phức tạp** | Tích phân phức tạp | Phép trừ và bình phương |
| **Liên hệ thực tế** | Tính xác suất sự kiện | Học từ thầy giáo |

### Toán học: Target Vector Field

Nếu biết **target vector field** $u_t(x)$ - hướng "đúng" tại mỗi điểm và thời điểm, chỉ cần huấn luyện mạng $v_\theta$ để xấp xỉ:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], \, x_t \sim p_t} \left[\left\| v_\theta(x_t, t) - u_t(x_t) \right\|^2\right]
$$

**Giải thích từng thành phần:**

- $\mathcal{L}_{\text{FM}}$: Loss function của Flow Matching
- $\mathbb{E}[\cdot]$: Kỳ vọng (trung bình) trên tất cả các mẫu
- $t \sim \mathcal{U}[0,1]$: Thời gian $t$ được chọn ngẫu nhiên từ 0 đến 1
- $x_t \sim p_t$: Vị trí tại thời điểm $t$ (lấy từ phân phối $p_t$)
- $v_\theta(x_t, t)$: Hướng mà mô hình **dự đoán** tại $(x_t, t)$
- $u_t(x_t)$: Hướng **đúng** (target) tại $(x_t, t)$
- $\|\cdot\|^2$: Bình phương khoảng cách Euclid (Mean Squared Error)

**Đây là bài toán regression đơn thuần!** Không cần:
- Tính trace của Jacobian
- Tích phân ODE ngược
- Đánh giá likelihood

### Ví dụ minh họa bằng số

**Setup:**
- Khối cầu ban đầu: $x_0 = (0, 0, 0)$
- Con rồng hoàn thành: $x_1 = (5, 3, 2)$
- Thời điểm: $t = 0.5$ (giữa chừng)

**Bước 1: Vị trí tại t=0.5**

Nội suy tuyến tính:
$$
x_t = (1-t) x_0 + t x_1 = 0.5 \times (0,0,0) + 0.5 \times (5,3,2) = (2.5, 1.5, 1.0)
$$

**Bước 2: Hướng di chuyển (target)**

$$
u_t(x_t) = x_1 - x_0 = (5,3,2) - (0,0,0) = (5, 3, 2)
$$

**Bước 3: Mô hình dự đoán**

Giả sử mạng neural dự đoán: $v_\theta(x_t, t) = (4.8, 3.1, 1.9)$

**Bước 4: Tính loss**

$$
\begin{aligned}
\text{Error} &= v_\theta - u_t = (4.8, 3.1, 1.9) - (5, 3, 2) = (-0.2, 0.1, -0.1) \\
\text{Loss} &= \|(-0.2, 0.1, -0.1)\|^2 \\
&= (-0.2)^2 + (0.1)^2 + (-0.1)^2 \\
&= 0.04 + 0.01 + 0.01 = 0.06
\end{aligned}
$$

**Bước 5: Cập nhật**

Gradient descent điều chỉnh mạng để $v_\theta$ gần $(5, 3, 2)$ hơn.

### So sánh code: CNF vs Flow Matching

**CNF (Maximum Likelihood) - Phức tạp:**

```python
def cnf_loss(model, x_data):
    # Bước 1: Tích phân NGƯỢC từ x_data về Gaussian
    x0, log_det_accumulate = odeint_backward(
        model, x_data, 
        t_span=[1.0, 0.0],  # Ngược!
        method='dopri5'
    )
    # Cần 100-200 function evaluations!
    
    # Bước 2: Tính trace tại MỖI thời điểm
    for t in linspace(0, 1, 20):
        x_t = get_trajectory_point(x0, t)
        
        # Hutchinson trace estimator
        trace = 0
        for _ in range(10):  # 10 random samples
            eps = torch.randn_like(x_t)
            vjp = autograd.grad(
                model(x_t, t), x_t, eps,
                create_graph=True  # Cần gradient của gradient!
            )[0]
            trace += (eps * vjp).sum()
        trace /= 10
        
        log_det_accumulate += trace * dt
    
    # Bước 3: Likelihood
    log_p0 = -0.5 * (x0 ** 2).sum()  # Gaussian
    log_p_data = log_p0 - log_det_accumulate
    
    return -log_p_data.mean()  # Negative log-likelihood
```

**Flow Matching (Regression) - Đơn giản:**

```python
def flow_matching_loss(model, x_data):
    batch_size = x_data.shape[0]
    
    # Bước 1: Sample thời gian
    t = torch.rand(batch_size, 1)  # [0, 1]
    
    # Bước 2: Sample điểm xuất phát
    x0 = torch.randn_like(x_data)  # Gaussian
    
    # Bước 3: Nội suy
    xt = (1 - t) * x0 + t * x_data
    
    # Bước 4: Target velocity (hằng số!)
    target = x_data - x0
    
    # Bước 5: Predicted velocity
    pred = model(xt, t)
    
    # Bước 6: MSE loss
    return ((pred - target) ** 2).mean()
```

**Độ phức tạp:**

| Aspect | CNF | Flow Matching |
|--------|-----|---------------|
| Số dòng code | ~50 dòng | ~10 dòng |
| Số lượt gọi model mỗi batch | 100-200 | 1 |
| Cần ODE solver? | Có | Không |
| Cần tính trace? | Có | Không |
| Cần create_graph? | Có | Không |
| Bộ nhớ GPU | ~8GB | ~2GB |

### Câu hỏi: Target vector field từ đâu?

Đây là câu hỏi quan trọng: **Làm sao biết hướng "đúng" $u_t(x)$ khi không có ground truth?**

**Định lý (Continuity Equation):**

Nếu phân phối $p_t(x)$ tiến hóa liên tục từ $p_0$ (Gaussian) đến $p_1$ (data), tồn tại duy nhất vector field $u_t$ thỏa mãn:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t u_t) = 0
$$

**Giải thích:**

- $\frac{\partial p_t}{\partial t}$: Tốc độ thay đổi mật độ xác suất theo thời gian
- $\nabla \cdot (p_t u_t)$: Divergence của "dòng chảy xác suất"
- Phương trình này nói: "Xác suất không tự sinh ra hay mất đi, chỉ di chuyển"

Vector field này có dạng:

$$
u_t(x) = \mathbb{E}_{X(t) | X(t) = x} \left[\frac{dX(t)}{dt}\right]
$$

**Giải thích bằng ví dụ:**

Tưởng tượng 1000 hạt đất, tất cả đang ở vị trí $x = (1, 1, 1)$ tại $t=0.5$:

```
Hạt 1 đang đi về phía (2, 1, 1) với vận tốc (2, 0, 0)
Hạt 2 đang đi về phía (1, 2, 1) với vận tốc (0, 2, 0)
Hạt 3 đang đi về phía (1, 1, 2) với vận tốc (0, 0, 2)
...
Hạt 1000 đang đi về phía (1.5, 1.5, 1.5) với vận tốc (1, 1, 1)

Hướng TRUNG BÌNH (target vector field):
u_t((1,1,1)) = (2 + 0 + 0 + ... + 1) / 1000
             = (~1.2, ~1.3, ~1.1)
```

### Vấn đề: Tính $u_t(x)$ trực tiếp vẫn khó!

Công thức $u_t(x) = \mathbb{E}[\cdot]$ yêu cầu biết **toàn bộ phân phối** $p_t(x)$ - điều ta không có!

Đây chính là lúc **Conditional Flow Matching** xuất hiện với một trick thiên tài: **Chia để trị**!

---

## 3. Conditional Flow Matching (CFM)

### Ý tưởng: Chia để trị

Người thợ gốm nhận ra một điều thú vị khi quan sát xưởng:

**Câu hỏi KHÓ (Global):**
> "Làm sao biến 1000 khối cầu ngẫu nhiên thành 1000 con rồng đa dạng?"

Quá phức tạp! Nhưng nếu chia nhỏ:

**Câu hỏi DỄ (Conditional):**
> "Nếu tôi muốn làm **chính xác con rồng này** (đã cho trước), thì khối cầu nào nên biến thành nó? Và đường đi như thế nào?"

**Ví dụ cụ thể:**

```
GLOBAL (Khó):
- Input: 1000 khối cầu (random)
- Output: 1000 con rồng (đa dạng: to nhỏ, màu sắc khác nhau)
- Hỏi: Làm sao matching tối ưu?
→ Quá nhiều khả năng!

CONDITIONAL (Dễ):
- Cho trước: Con rồng #42 (màu xanh, to, đuôi dài)
- Chọn: Khối cầu #73 (gần nhất)
- Hỏi: Đường đi từ #73 → #42?
→ Đơn giản: Đường thẳng!

     Khối cầu #73          Con rồng #42
          (0,0,0) --------→ (5,3,2)
                   t=0 → t=1
```

**Ý tưởng CFM:** Nếu biết cách làm với **TỪNG con rồng cụ thể**, tự động biết cách làm với **TẤT CẢ!**

### Toán học: Conditional Probability Path

**Conditional path:** Thay vì xét $p_t(x)$ (global), xét $p_t(x \mid x_1)$ - phân phối tại thời $t$ **biết điểm cuối** $x_1$.

Cho mỗi tác phẩm mục tiêu $x_1 \sim p_{\text{data}}$, định nghĩa:

$$
p_t(x \mid x_1) = \mathcal{N}(x \mid \mu_t(x_1), \sigma_t^2 I)
$$

**Giải thích từng thành phần:**

- $p_t(x \mid x_1)$: Phân phối xác suất của $x$ tại thời $t$, biết kết quả cuối cùng là $x_1$
- $\mathcal{N}(\cdot)$: Phân phối Gaussian (chuẩn)
- $\mu_t(x_1) = t x_1$: Trung bình - nội suy tuyến tính từ 0 đến $x_1$
- $\sigma_t = 1 - t$: Độ lệch chuẩn - giảm dần về 0

**Simple choice (Gaussian interpolation):**

$$
\begin{aligned}
\mu_t(x_1) &= t x_1 \quad \text{(từ 0 đến } x_1 \text{)} \\
\sigma_t &= 1 - t \quad \text{(từ 1 về 0)}
\end{aligned}
$$

**Ví dụ cụ thể với con rồng xanh:**

Giả sử con rồng xanh hoàn chỉnh ở $x_1 = (5, 3, 2)$.

```
Tại t=0:
  μ₀ = 0 × (5,3,2) = (0,0,0)
  σ₀ = 1 - 0 = 1
  → x₀ ~ N((0,0,0), 1²) = Gaussian chuẩn (khối cầu)

Tại t=0.5:
  μ₀.₅ = 0.5 × (5,3,2) = (2.5, 1.5, 1.0)
  σ₀.₅ = 1 - 0.5 = 0.5
  → x₀.₅ ~ N((2.5, 1.5, 1.0), 0.5²)
  
  Có thể ở: (2.3, 1.6, 0.9) hoặc (2.7, 1.4, 1.1)
           ↑ Dao động nhỏ quanh trung bình

Tại t=1:
  μ₁ = 1 × (5,3,2) = (5,3,2)
  σ₁ = 1 - 1 = 0
  → x₁ = (5,3,2) chính xác (con rồng xanh)
```

### Sample từ Conditional Path

Để lấy mẫu $x_t$ từ $p_t(x \mid x_1)$:

$$
x_t = t x_1 + (1-t) x_0, \quad x_0 \sim \mathcal{N}(0, I)
$$

**Chứng minh công thức này đúng:**

$$
\begin{aligned}
\mathbb{E}[x_t] &= \mathbb{E}[t x_1 + (1-t) x_0] \\
&= t x_1 + (1-t) \mathbb{E}[x_0] \\
&= t x_1 + (1-t) \times 0 = t x_1 \\
\\
\text{Var}(x_t) &= \text{Var}((1-t) x_0) \\
&= (1-t)^2 \text{Var}(x_0) \\
&= (1-t)^2 \times 1 = (1-t)^2
\end{aligned}
$$

**Ví dụ Python:**

```python
# Cho trước: Con rồng xanh
x1 = torch.tensor([5.0, 3.0, 2.0])

# Sample tại t=0.5
t = 0.5
x0 = torch.randn(3)  # Ví dụ: [-0.3, 0.7, -0.5]

xt = t * x1 + (1 - t) * x0
# = 0.5 * [5, 3, 2] + 0.5 * [-0.3, 0.7, -0.5]
# = [2.5, 1.5, 1.0] + [-0.15, 0.35, -0.25]
# = [2.35, 1.85, 0.75]
```

### Conditional Vector Field - Kết quả đẹp!

Với $x_t = tx_1 + (1-t)x_0$, lấy **đạo hàm theo thời gian** $t$:

$$
\frac{dx_t}{dt} = \frac{d}{dt}[tx_1 + (1-t)x_0] = x_1 - x_0
$$

**Giải thích từng bước:**

1. **Đạo hàm của** $tx_1$:
   $$\frac{d}{dt}(tx_1) = x_1 \quad \text{(vì } x_1 \text{ là hằng số theo } t \text{)}$$

2. **Đạo hàm của** $(1-t)x_0$:
   $$\frac{d}{dt}[(1-t)x_0] = -x_0$$

3. **Tổng hợp:**
   $$\frac{dx_t}{dt} = x_1 - x_0$$

Vậy **conditional velocity** là:

$$
u_t(x_t \mid x_1) = x_1 - x_0
$$

**Kết quả tuyệt vời:** Velocity **không phụ thuộc vào** $t$! Nó là hằng số = hiệu giữa đích và xuất phát.

**Ví dụ cụ thể:**

```
Con rồng xanh: x₁ = (5, 3, 2)
Khối cầu ban đầu: x₀ = (-0.3, 0.7, -0.5)

Velocity:
u_t = (5, 3, 2) - (-0.3, 0.7, -0.5)
    = (5.3, 2.3, 2.5)

Tại mọi thời điểm t ∈ [0, 1]:
  t=0.0: Velocity = (5.3, 2.3, 2.5)
  t=0.5: Velocity = (5.3, 2.3, 2.5)  ← Giống nhau!
  t=1.0: Velocity = (5.3, 2.3, 2.5)

Quỹ đạo:
x(t) = (-0.3, 0.7, -0.5) + t × (5.3, 2.3, 2.5)

t=0.0: x = (-0.3, 0.7, -0.5)      ← Khối cầu
t=0.5: x = (2.35, 1.85, 0.75)     ← Giữa chừng
t=1.0: x = (5.0, 3.0, 2.0)        ← Con rồng xanh

→ Đường THẲNG!
```

### Định lý CFM - Chứng minh đầy đủ

**Định lý:** Minimizing conditional FM objective:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_1, x_0} \left[\left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2\right]
$$

có **cùng gradient** với marginal FM objective:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_t \sim p_t} \left[\left\| v_\theta(x_t, t) - u_t(x_t) \right\|^2\right]
$$

**Chứng minh chi tiết:**

**Bước 1:** Viết lại CFM loss dưới dạng tích phân

$$
\begin{aligned}
\mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{t, x_1, x_0} \left[\left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2\right] \\
&= \int_0^1 dt \int p_{\text{data}}(x_1) dx_1 \int p_0(x_0) dx_0 \\
&\quad \times \left\| v_\theta(tx_1 + (1-t)x_0, t) - (x_1 - x_0) \right\|^2
\end{aligned}
$$

**Bước 2:** Đổi biến từ $(x_0, x_1)$ sang $(x_t, x_1)$

Định nghĩa: $x_t = tx_1 + (1-t)x_0$

Giải ra: $x_0 = \frac{x_t - tx_1}{1-t}$

**Jacobian của phép đổi biến:**

$$
\frac{\partial x_0}{\partial x_t} = \frac{1}{1-t} I
$$

Do đó:

$$
dx_0 = \frac{1}{(1-t)^d} dx_t
$$

với $d$ là số chiều.

**Bước 3:** Tính $p_t(x_t, x_1)$ (joint distribution)

$$
\begin{aligned}
p_t(x_t, x_1) &= p_{\text{data}}(x_1) \times p_0(x_0) \times \left|\frac{\partial x_0}{\partial x_t}\right| \\
&= p_{\text{data}}(x_1) \times p_0\left(\frac{x_t - tx_1}{1-t}\right) \times \frac{1}{(1-t)^d}
\end{aligned}
$$

Với $p_0(x_0) = \mathcal{N}(0, I)$:

$$
p_0\left(\frac{x_t - tx_1}{1-t}\right) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{1}{2}\left\|\frac{x_t - tx_1}{1-t}\right\|^2\right)
$$

Nhận ra: $x_t \mid x_1 \sim \mathcal{N}(tx_1, (1-t)^2 I)$

Do đó:

$$
p_t(x_t | x_1) = \frac{1}{(2\pi(1-t)^2)^{d/2}} \exp\left(-\frac{\|x_t - tx_1\|^2}{2(1-t)^2}\right)
$$

**Bước 4:** Tính marginal $p_t(x_t)$

$$
p_t(x_t) = \int p_t(x_t \mid x_1) p_{\text{data}}(x_1) dx_1
$$

**Bước 5:** Marginal vector field

$$
\begin{aligned}
u_t(x_t) &= \mathbb{E}_{x_1 \mid x_t}[x_1 - x_0] \\
&= \int (x_1 - x_0) \frac{p_t(x_t \mid x_1) p_{\text{data}}(x_1)}{p_t(x_t)} dx_1 \\
&= \int (x_1 - x_0) p(x_1 \mid x_t) dx_1
\end{aligned}
$$

với $x_0 = \frac{x_t - tx_1}{1-t}$.

**Bước 6:** Viết lại loss dưới dạng marginal

Thay đổi biến tích phân từ $(x_0, x_1)$ sang $(x_t, x_1)$ rồi tích phân theo $x_1$:

$$
\begin{aligned}
\mathcal{L}_{\text{CFM}}(\theta) &= \int_0^1 dt \int p_t(x_t) dx_t \int p(x_1 \mid x_t) dx_1 \\
&\quad \times \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \\
&= \int_0^1 dt \int p_t(x_t) dx_t \left\| v_\theta(x_t, t) - u_t(x_t) \right\|^2 \\
&= \mathcal{L}_{\text{FM}}(\theta)
\end{aligned}
$$

**Kết luận:** $\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$

→ **Gradient của CFM = Gradient của FM!** Minimizing CFM cũng chính là minimizing FM.

### Code siêu đơn giản

```python
def conditional_flow_matching_loss(model, x1_batch):
    """
    CFM loss - SIÊU ĐƠN GIẢN!
    
    Args:
        model: v_θ(x, t) - mạng neural
        x1_batch: (batch_size, dim) - dữ liệu thật (con rồng)
    
    Returns:
        loss: scalar - MSE loss
    """
    batch_size, dim = x1_batch.shape
    
    # Bước 1: Sample thời gian ngẫu nhiên
    t = torch.rand(batch_size, 1)  # [0, 1]
    
    # Bước 2: Sample điểm xuất phát (khối cầu)
    x0 = torch.randn_like(x1_batch)  # N(0, I)
    
    # Bước 3: Nội suy để được vị trí tại thời điểm t
    xt = t * x1_batch + (1 - t) * x0
    
    # Bước 4: Target velocity (conditional vector field)
    target = x1_batch - x0  # Hằng số!
    
    # Bước 5: Predicted velocity từ mô hình
    pred = model(xt, t)
    
    # Bước 6: MSE loss
    loss = ((pred - target) ** 2).mean()
    
    return loss
```

**Ví dụ cụ thể với 1 sample:**

```python
# Dữ liệu: Con rồng xanh
x1 = torch.tensor([[5.0, 3.0, 2.0]])  # shape: (1, 3)

# Random time
t = torch.tensor([[0.6]])  # shape: (1, 1)

# Random starting point
x0 = torch.randn(1, 3)  # Ví dụ: [[0.2, -0.5, 0.8]]

# Interpolate
xt = 0.6 * torch.tensor([[5.0, 3.0, 2.0]]) + 0.4 * torch.tensor([[0.2, -0.5, 0.8]])
   = torch.tensor([[3.08, 1.6, 1.52]])

# Target
target = torch.tensor([[5.0, 3.0, 2.0]]) - torch.tensor([[0.2, -0.5, 0.8]])
       = torch.tensor([[4.8, 3.5, 1.2]])

# Model prediction (giả sử)
pred = model(xt, t)  # Giả sử output: [[4.7, 3.6, 1.1]]

# Loss
loss = ((torch.tensor([[4.7, 3.6, 1.1]]) - torch.tensor([[4.8, 3.5, 1.2]])) ** 2).mean()
     = ((torch.tensor([[-0.1, 0.1, -0.1]])) ** 2).mean()
     = (0.01 + 0.01 + 0.01) / 3
     = 0.01
```

### Tại sao CFM hoạt động?

**Trực giác:**

1. **Mỗi con rồng** $x_1$ định nghĩa một "đường đi riêng" từ Gaussian
2. Tất cả các đường đi này **chồng lên nhau** tạo thành marginal flow
3. Học tốt trên từng đường đi riêng → Tự động học tốt trên marginal!

**Ví dụ hình ảnh:**

```
Marginal flow (phức tạp):
  1000 khối cầu → 1000 con rồng đa dạng
  ↑ KHÓ: Không biết hướng di chuyển trung bình

Conditional flows (đơn giản):
  Khối #1 → Con rồng xanh    (đường thẳng)
  Khối #2 → Con rồng đỏ      (đường thẳng)
  ...
  Khối #1000 → Con rồng vàng (đường thẳng)
  ↑ DỄ: Mỗi đường đều thẳng!

Khi học tốt TẤT CẢ các đường riêng
→ Tự động học tốt TRUNG BÌNH của chúng!
```

---

## 4. Optimal Transport Conditional Flow Matching

### Vấn đề: Đường đi không tối ưu

Người thợ gốm nhận ra CFM hoạt động tốt, nhưng có vấn đề: **Đường đi không ngắn nhất**.

**Ví dụ:** Hai hạt đất sét cần đến hai vị trí:
- Hạt A tại $(0, 0)$ → Đích A tại $(1, 0)$
- Hạt B tại $(0, 0)$ → Đích B tại $(0, 1)$

Với conditional path độc lập, cả hai xuất phát từ gốc. Nhưng điều này không tối ưu! Nếu:
- Hạt A tại $(0.1, 0)$ → Đích A
- Hạt B tại $(0, 0.1)$ → Đích B

Tổng quãng đường ngắn hơn.

### Optimal Transport (OT)

**Bài toán:** Tìm cách ghép cặp $(x_0, x_1)$ sao cho tổng "chi phí vận chuyển" nhỏ nhất.

**Monge-Kantorovich formulation:**

$$
\min_{\pi \in \Pi(p_0, p_1)} \int c(x_0, x_1) d\pi(x_0, x_1)
$$

với:
- $\pi(x_0, x_1)$: joint distribution (coupling)
- $c(x_0, x_1) = \|x_0 - x_1\|^2$: cost function
- $\Pi(p_0, p_1)$: tập hợp các coupling có marginals = $p_0, p_1$

### OT-CFM: Kết hợp OT và CFM

**Ý tưởng:** Thay vì sample $x_0$ ngẫu nhiên, tìm $x_0$ sao cho cặp $(x_0, x_1)$ minimize OT cost.

**Minibatch OT solution:**

Với batch $\{x_1^{(1)}, \ldots, x_1^{(B)}\}$ từ data và $\{x_0^{(1)}, \ldots, x_0^{(B)}\}$ từ Gaussian:

1. **Tính cost matrix:** $C_{ij} = \|x_0^{(i)} - x_1^{(j)}\|^2$
2. **Giải OT:** Sinkhorn algorithm
3. **Match:** Ghép $x_0^{(i)}$ với $x_1^{(\sigma(i))}$

**Code:**

```python
import ot  # Python Optimal Transport library

def optimal_transport_matching(x0_batch, x1_batch):
    """Find optimal coupling between x0 and x1"""
    # Cost matrix
    C = torch.cdist(x0_batch, x1_batch, p=2) ** 2
    C_np = C.detach().cpu().numpy()
    
    # Uniform marginals
    a = np.ones(len(x0_batch)) / len(x0_batch)
    b = np.ones(len(x1_batch)) / len(x1_batch)
    
    # Solve with Sinkhorn
    pi = ot.sinkhorn(a, b, C_np, reg=0.05)
    
    # Greedy matching
    indices = pi.argmax(axis=1)
    
    return x0_batch, x1_batch[indices]

# OT-CFM training
def ot_cfm_loss(model, x1_batch):
    x0_batch = torch.randn_like(x1_batch)
    
    # Optimal matching
    x0_matched, x1_matched = optimal_transport_matching(x0_batch, x1_batch)
    
    # Rest is same as CFM
    t = torch.rand(len(x1_batch), 1)
    xt = t * x1_matched + (1 - t) * x0_matched
    ut = x1_matched - x0_matched
    vt = model(xt, t)
    
    return ((vt - ut) ** 2).mean()
```

### Kết nối với Rectified Flows

**Rectified Flows** (bài tiếp theo) đưa ý tưởng này xa hơn:
1. Train với CFM
2. **Reflow**: Dùng model tạo cặp $(x_0, x_1)$ mới
3. Train lại → Quỹ đạo thẳng hơn

Kết quả: Sampling chỉ **1-2 bước** thay vì 10-20!

---

## 5. Chứng minh tính đúng đắn

### Định lý chính

**Định lý (Flow Matching Theorem):** Nếu $v_\theta$ minimize CFM objective và đủ expressive, thì trajectory $\frac{dx}{dt} = v_\theta(x, t)$ với $x(0) \sim p_0$ sẽ có:

$$
x(1) \sim q_1 \approx p_{\text{data}}
$$

### Chứng minh đầy đủ

**Bước 1: Marginal probability path**

Định nghĩa marginal path:

$$
p_t(x) = \int p_{\text{data}}(x_1) p_t(x | x_1) dx_1
$$

Với $p_t(x \mid x_1) = \mathcal{N}(x \mid tx_1, (1-t)^2 I)$, ta có:

$$
p_t(x) = \int p_{\text{data}}(x_1) \frac{1}{(2\pi(1-t)^2)^{d/2}} \exp\left(-\frac{\|x - tx_1\|^2}{2(1-t)^2}\right) dx_1
$$

**Boundary conditions:**
- $p_0(x) = \mathcal{N}(0, I)$ (Gaussian)
- $p_1(x) = p_{\text{data}}(x)$ (data distribution)

**Bước 2: Marginal vector field**

Marginal vector field thỏa mãn continuity equation:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t u_t) = 0
$$

Có thể chứng minh (qua tính toán):

$$
u_t(x) = \mathbb{E}_{x_1 \sim p_{\text{data}}, x_0 \sim \mathcal{N}(0,I) | x_t = x} [x_1 - x_0]
$$

**Bước 3: CFM objective = FM objective (gradient-wise)**

Đã chứng minh ở phần 3:

$$
\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{FM}}(\theta)
$$

**Bước 4: Convergence analysis**

Khi $\mathcal{L}_{\text{CFM}}(\theta) \to 0$:

$$
v_\theta(x, t) \to u_t(x) \quad \forall x, t
$$

Trajectory generated by $v_\theta$ thỏa mãn ODE:

$$
\frac{dx}{dt} = v_\theta(x, t) \approx u_t(x)
$$

Theo theory của ODEs, solution này sinh ra phân phối $q_t$ thỏa mãn:

$$
\frac{\partial q_t}{\partial t} + \nabla \cdot (q_t v_\theta) \approx 0
$$

Với điều kiện $q_0 = p_0$, ta có $q_1 \approx p_1 = p_{\text{data}}$.

### Regret Bound

**Định lý (Tong et al., 2023):** Nếu $\mathcal{L}_{\text{CFM}}(\theta) = \epsilon$, thì:

$$
\text{KL}(p_{\text{data}} \| q_1^\theta) \leq C \sqrt{\epsilon}
$$

với $C$ phụ thuộc Lipschitz constant của $u_t$.

**Ý nghĩa:** MSE nhỏ → KL divergence nhỏ → Phân phối generated gần data!

---

## 6. Implementation PyTorch đầy đủ

### 6.1. Vector Field Network

```python
import torch
import torch.nn as nn
import math

class VectorFieldNetwork(nn.Module):
    """Time-conditioned vector field v_θ(x, t)"""
    
    def __init__(self, dim, hidden_dim=256, num_layers=4, time_embed_dim=64):
        super().__init__()
        self.dim = dim
        
        # Sinusoidal time embedding
        self.time_embed_dim = time_embed_dim
        self.register_buffer('frequency', torch.exp(
            torch.linspace(0, math.log(1000), time_embed_dim // 2)
        ))
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main network
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)
    
    def time_embedding(self, t):
        """Sinusoidal embedding for t ∈ [0, 1]"""
        angles = t * self.frequency.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb
    
    def forward(self, x, t):
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1) or (batch_size,)
        Returns:
            v: (batch_size, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Embed time
        t_emb = self.time_embedding(t)
        t_feat = self.time_mlp(t_emb)
        
        # Process x with time conditioning
        h = self.net[0](x)
        h = self.net[1](h)
        h = h + t_feat  # Add time features
        
        for layer in self.net[2:]:
            h = layer(h)
        
        return h
```

### 6.2. Conditional Flow Matcher

```python
class ConditionalFlowMatcher:
    """CFM training and sampling"""
    
    def __init__(self, model):
        self.model = model
    
    def compute_conditional_flow(self, x0, x1, t):
        """
        Compute (x_t, u_t) for conditional path
        
        Args:
            x0: (B, D) - source
            x1: (B, D) - target
            t: (B, 1) - time
        
        Returns:
            xt, ut
        """
        xt = t * x1 + (1 - t) * x0
        ut = x1 - x0  # Constant velocity
        return xt, ut
    
    def training_loss(self, x1_batch):
        """CFM loss"""
        batch_size, dim = x1_batch.shape
        device = x1_batch.device
        
        # Sample time
        t = torch.rand(batch_size, 1, device=device)
        
        # Sample x0
        x0 = torch.randn_like(x1_batch)
        
        # Conditional flow
        xt, ut = self.compute_conditional_flow(x0, x1_batch, t)
        
        # Predict
        vt_pred = self.model(xt, t)
        
        # MSE loss
        loss = ((vt_pred - ut) ** 2).mean()
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, num_steps=50, device='cuda', method='euler'):
        """Generate samples by solving ODE"""
        self.model.eval()
        
        # Initial: x ~ N(0, I)
        x = torch.randn(num_samples, self.model.dim, device=device)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((num_samples, 1), i * dt, device=device)
            
            if method == 'euler':
                v = self.model(x, t)
                x = x + dt * v
            elif method == 'midpoint':
                v1 = self.model(x, t)
                x_mid = x + 0.5 * dt * v1
                t_mid = t + 0.5 * dt
                v2 = self.model(x_mid, t_mid)
                x = x + dt * v2
        
        return x
```

### 6.3. Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_flow_matching(dataset, dim, epochs=100, batch_size=256, lr=1e-3):
    """Full training pipeline"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model = VectorFieldNetwork(dim=dim, hidden_dim=256, num_layers=4).to(device)
    
    # CFM
    cfm = ConditionalFlowMatcher(model)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for x_batch in dataloader:
            x_batch = x_batch.to(device)
            
            # Loss
            loss = cfm.training_loss(x_batch)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        # Log
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        # Sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                samples = cfm.sample(1000, num_steps=50, device=device)
                # Visualize samples here
    
    return model, cfm
```

### 6.4. OT-CFM Extension

```python
try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

class OptimalTransportCFM(ConditionalFlowMatcher):
    """CFM with Optimal Transport"""
    
    def __init__(self, model, ot_reg=0.05):
        super().__init__(model)
        self.ot_reg = ot_reg
        if not HAS_OT:
            raise ImportError("Install POT: pip install POT")
    
    def optimal_coupling(self, x0, x1):
        """Find optimal coupling"""
        C = torch.cdist(x0, x1, p=2) ** 2
        C_np = C.detach().cpu().numpy()
        
        batch_size = len(x0)
        a = np.ones(batch_size) / batch_size
        b = np.ones(batch_size) / batch_size
        
        # Sinkhorn
        pi = ot.sinkhorn(a, b, C_np, reg=self.ot_reg)
        indices = pi.argmax(axis=1)
        
        return x0, x1[indices]
    
    def training_loss(self, x1_batch):
        """OT-CFM loss"""
        x0 = torch.randn_like(x1_batch)
        x0_matched, x1_matched = self.optimal_coupling(x0, x1_batch)
        
        t = torch.rand(len(x1_batch), 1, device=x1_batch.device)
        xt, ut = self.compute_conditional_flow(x0_matched, x1_matched, t)
        
        vt_pred = self.model(xt, t)
        return ((vt_pred - ut) ** 2).mean()
```

---

## 7. So sánh CNF, Diffusion và Flow Matching

### Bảng so sánh

| Khía cạnh | CNF/FFJORD | Diffusion | Flow Matching |
|-----------|------------|-----------|---------------|
| **Training** | Maximum likelihood | Denoising score matching | Regression (MSE) |
| **Cần Jacobian?** | Có (trace) | Không | Không |
| **ODE khi train?** | Có (backward) | Không | Không |
| **Training speed** | Chậm (~0.5 it/s) | Nhanh (~50 it/s) | Nhanh (~50 it/s) |
| **Sampling NFE** | 50-100 | 1000 | 10-20 |
| **Exact likelihood** | Có | Không | Không |
| **Flexibility** | Cao | Trung bình | Cao |

### Khi nào dùng gì?

**CNF/FFJORD:**
- Cần exact likelihood (anomaly detection, probabilistic modeling)
- Dataset nhỏ, có thời gian train

**Diffusion:**
- Sinh ảnh/video chất lượng cao
- Chấp nhận sampling chậm
- Ứng dụng: Stable Diffusion, DALL-E

**Flow Matching:**
- Cần training VÀ sampling nhanh
- Real-time generation
- Ứng dụng: Video synthesis, molecular design, Stable Diffusion 3

---

## 8. Kết luận

### Điểm chính

1. **Flow Matching = Regression, không phải Likelihood**
   - Thay vì $\max \log p_\theta(x)$ (đắt)
   - Minimize $\|v_\theta - u_t\|^2$ (đơn giản)

2. **Conditional Flow Matching**
   - Học từ conditional paths: $u_t(x_t \mid x_1) = x_1 - x_0$
   - Gradient equivalence với marginal FM

3. **Optimal Transport**
   - Tìm cặp $(x_0, x_1)$ tối ưu
   - Quỹ đạo ngắn hơn → Sampling nhanh hơn

4. **Implementation cực đơn giản**
   - ~10 dòng code cho CFM loss
   - Không cần ODE solver khi train
   - **Nhanh hơn CNF ~100x**

### So sánh với bài trước

Bài [Normalizing Flows & CNF](/posts/2025/normalizing-flows) dạy:
- Change of Variables
- Instantaneous CoV với trace
- FFJORD với Hutchinson

Flow Matching **vượt qua** tất cả:
- Không cần trace
- Không cần ODE ngược
- Chỉ cần regression!

### Hướng phát triển

1. **Rectified Flows** (bài tiếp theo)
   - Reflow để làm thẳng trajectory
   - Sampling 1-2 steps!

2. **Stochastic Interpolants**
   - Kết hợp FM và Diffusion

3. **Applications**
   - Stable Diffusion 3
   - Real-time video generation
   - Drug discovery

### Câu chuyện kết

Người thợ gốm cuối cùng tìm ra cách học hiệu quả:

- **Trước (CNF)**: Quan sát mọi hạt đất, tính xác suất → Phức tạp
- **Bây giờ (FM)**: Học hướng di chuyển qua regression → Đơn giản

**Bài học**: Đôi khi cách đơn giản hơn lại hiệu quả hơn!

---

## Tài liệu tham khảo

1. **Lipman et al. (2023)** - "Flow Matching for Generative Modeling" (ICLR)
2. **Tong et al. (2023)** - "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" (ICML)
3. **Liu et al. (2023)** - "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR)
4. **Albergo & Vanden-Eijnden (2023)** - "Building Normalizing Flows with Stochastic Interpolants" (ICLR)
5. **Pooladian et al. (2023)** - "Multisample Flow Matching: Straightening Flows with Minibatch Couplings"

---

**Bài tiếp theo:** [Rectified Flows: One-Step Generation](/posts/2025/rectified-flows)

<script src="/assets/js/katex-init.js"></script>
