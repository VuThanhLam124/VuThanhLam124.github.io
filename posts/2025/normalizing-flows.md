---
title: "Normalizing Flow & Continuous Normalizing Flow: Từ Lý thuyết đến Thực hành"
date: "2025-01-15"
category: "flow-based-models"
tags: ["normalizing-flows", "CNF", "neural-ode", "generative-models", "pytorch"]
excerpt: "Hành trình khám phá Normalizing Flows từ câu chuyện người thợ gốm đến toán học sâu và implementation chi tiết. Kết hợp storytelling, rigorous math với LaTeX đẹp, và complete PyTorch code."
author: "ThanhLamDev"
readingTime: 20
featured: true
---

# Normalizing Flow & Continuous Normalizing Flow

**Hành trình từ Đơn giản đến Phức tạp**

Chào mừng bạn đến với bài viết đầu tiên trong series về Flow-based Models. Thay vì đi thẳng vào công thức khô khan, chúng ta sẽ bắt đầu với một câu chuyện...

## Mục lục

1. [Câu chuyện về người thợ gốm](#1-câu-chuyện-về-người-thợ-gốm)
2. [Từ trực giác đến toán học](#2-từ-trực-giác-đến-toán-học)
3. [Change of Variables Formula](#3-change-of-variables-formula)
4. [Jacobian: "Phí co giãn" của không gian](#4-jacobian-phí-co-giãn-của-không-gian)
5. [Kiến trúc thông minh: Coupling Layers](#5-kiến-trúc-thông-minh-coupling-layers)
6. [Continuous Normalizing Flows](#6-continuous-normalizing-flows)
7. [Implementation đầy đủ với PyTorch](#7-implementation-đầy-đủ-với-pytorch)
8. [Advanced Topics & FFJORD](#8-advanced-topics--ffjord)
9. [Kết luận](#9-kết-luận)

---

## 1. Câu chuyện về người thợ gốm

Hãy tưởng tượng bạn là một nghệ nhân gốm. Trước mặt bạn là một khối đất sét hình cầu đơn giản, hoàn hảo, đối xứng - giống như một **phân phối Gaussian chuẩn** trong thống kê.

Nhiệm vụ của bạn? Biến khối đất sét đơn giản đó thành một tác phẩm nghệ thuật phức tạp - có thể là chiếc bình hình ngôi sao, hoặc hình con rồng. Đây chính là **data distribution** trong thế giới AI.

### Quá trình biến đổi

Bạn không thể một bước biến khối cầu thành con rồng. Thay vào đó, bạn thực hiện một **chuỗi các thao tác**:
1. Kéo dài một phần để tạo thân
2. Nặn nhỏ để tạo đầu
3. Uốn cong để tạo đuôi
4. Thêm chi tiết cho cánh, chân...

Mỗi bước là một **phép biến đổi** (transformation). Chuỗi này chính là **"flow"** - dòng chảy của các biến đổi.

### Điều kỳ diệu: Tính khả nghịch

Nếu bạn là một nghệ nhân bậc thầy, bạn có thể làm ngược lại: nhìn vào con rồng hoàn thành, bạn biết chính xác cách "tháo gỡ" từng bước để trở về khối cầu ban đầu. Đây chính là tính **invertible** (khả nghịch) - linh hồn của Normalizing Flow.

### Tại sao cần "người thợ gốm" giỏi?

Trong AI, chúng ta có nhiều "người thợ" với kỹ năng khác nhau:

**VAE (Variational Autoencoder)** - Người thợ tập sự:
- Có thể tạo ra những tác phẩm "tạm ổn"
- Nhưng luôn hơi mờ, thiếu sắc nét
- Chỉ ước tính được "độ khó" làm tác phẩm, không biết chính xác

**GAN (Generative Adversarial Network)** - Nghệ sĩ tài năng nhưng khó tính:
- Tạo ra tác phẩm cực kỳ đẹp, sắc nét
- Nhưng quá trình dạy rất khó khăn (training unstable)
- Không thể cho bạn biết xác suất để tạo ra một tác phẩm cụ thể

**Normalizing Flow** - Nghệ nhân bậc thầy:
- Tạo ra tác phẩm chất lượng cao
- **Tính được chính xác xác suất** (exact likelihood)
- Có thể đi cả hai chiều: tạo mới HOẶC phân tích ngược
- Quá trình dạy ổn định

Đây chính là lý do chúng ta cần Normalizing Flow!

---

## 2. Từ trực giác đến toán học

Bây giờ, hãy chuyển câu chuyện thành ngôn ngữ toán học.

### Định nghĩa Flow

**Flow** là một chuỗi các phép biến đổi khả nghịch:

$$
z_0 \xrightarrow{f_1} z_1 \xrightarrow{f_2} z_2 \xrightarrow{f_3} \cdots \xrightarrow{f_K} z_K = x
$$

Trong đó:
- $z_0 \sim p_0(z)$ là **base distribution** (khối đất sét ban đầu) - thường là $\mathcal{N}(0, I)$
- $x = f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0)$ là **data distribution** (tác phẩm hoàn thành)
- Mỗi $f_k$ phải có hàm ngược $f_k^{-1}$

**Ví dụ đơn giản nhất (1D):**

```python
import torch
import matplotlib.pyplot as plt

# Base distribution: Standard Gaussian
z = torch.randn(10000)  # z ~ N(0, 1)

# Flow transformation: affine
def f(z):
    return 2 * z + 3  # Scale by 2, shift by 3

x = f(z)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(z.numpy(), bins=50, density=True, alpha=0.7)
ax1.set_title("Base Distribution: z ~ N(0, 1)")
ax2.hist(x.numpy(), bins=50, density=True, alpha=0.7)
ax2.set_title("Transformed: x = 2z + 3 ~ N(3, 4)")
plt.show()
```

### So sánh với các mô hình khác

| Model | Tính toán Likelihood | Sinh mẫu hiệu quả | Huấn luyện ổn định | Biến đổi hai chiều |
|--------------------|-------------------------|--------------------|--------------------|---------------------|
| **VAE**            | Không (chặn dưới)       | Có                 | Có                 | Không               |
| **GAN**            | Không                   | Có                 | Không              | Không               |
| **Normalizing Flow** | Có (chính xác)          | Có                 | Có                 | Có                  |
| **Diffusion**      | Có (chính xác)          | Không (chậm)       | Có                 | Không               |

---

## 3. Change of Variables Formula

### Intuition: Bảo toàn "khối lượng"

Quay lại ví dụ người thợ gốm. Khi bạn kéo dài một phần của khối đất sét:
- Vùng bị kéo giãn → mật độ đất sét giảm (thưa hơn)
- Vùng bị nén lại → mật độ đất sét tăng (đặc hơn)
- Nhưng **tổng khối lượng đất sét không đổi**!

Trong xác suất, "khối lượng" là tích phân xác suất. Khi biến đổi, mật độ xác suất thay đổi nhưng phải đảm bảo:

$$
\int p_x(x) dx = \int p_z(z) dz = 1
$$

### Công thức toán học (1D)

Cho phép biến đổi $x = f(z)$ với $f$ khả nghịch:

$$
p_x(x) = p_z(f^{-1}(x)) \cdot \left\lvert\frac{df^{-1}}{dx}\right\rvert
$$

Hoặc dùng log (dễ tính toán hơn):

$$
\log p_x(x) = \log p_z(z) + \log\left\lvert\frac{df^{-1}}{dx}\right\rvert
$$

Ở đây, thành phần với giá trị tuyệt đối của đạo hàm chính là "phí co giãn" (stretching fee). Nó là một hệ số điều chỉnh mật độ để đảm bảo tổng xác suất luôn bằng 1, giống như tổng khối lượng đất sét được bảo toàn dù hình dạng thay đổi.

### Chứng minh hình học (2 chiều)

Hãy cùng hiểu tại sao công thức Change of Variables hoạt động thông qua một cách tiếp cận hình học trực quan. Chúng ta sẽ xem xét không gian 2 chiều để dễ hình dung, nhưng ý tưởng tổng quát hoá cho bất kỳ số chiều nào.

#### Bước 1: Xây dựng hình vuông nhỏ trong không gian ban đầu

Hãy tưởng tượng bạn đang làm việc với một lưới tọa độ 2D, không gian $Z = (z_1, z_2)$. Chọn một điểm bất kỳ $z = (z_1, z_2)$ trên không gian này.

Bây giờ, vẽ một **hình vuông nhỏ** xung quanh điểm $z$:
- Cạnh dọc: độ dài $\Delta z_1$ (dọc theo trục $z_1$)
- Cạnh ngang: độ dài $\Delta z_2$ (dọc theo trục $z_2$)

**Diện tích của hình vuông này:**
$$
\Delta S_z = \Delta z_1 \times \Delta z_2
$$

```
Không gian Z (ban đầu):

    z₂ ↑
       │
       │    ┌─────┐  ← Hình vuông nhỏ
       │    │  z  │     Diện tích = Δz₁ × Δz₂
       │    └─────┘
       │
       └──────────→ z₁
```

**Ý nghĩa xác suất:** Nếu $p_Z(z)$ là hàm mật độ xác suất tại điểm $z$, thì "khối lượng xác suất" trong hình vuông nhỏ này xấp xỉ bằng:
$$
\text{Khối lượng}_Z \approx p_Z(z) \times \Delta S_z = p_Z(z) \times \Delta z_1 \times \Delta z_2
$$

#### Bước 2: Biến đổi hình vuông qua hàm $g$

Giờ áp dụng phép biến đổi $x = g(z)$ để chuyển từ không gian $Z$ sang không gian $X$. Hàm $g$ có dạng:
$$
x = g(z) = \begin{bmatrix} g_1(z_1, z_2) \\ g_2(z_1, z_2) \end{bmatrix}
$$

**Câu hỏi quan trọng:** Hình vuông nhỏ ban đầu sẽ biến thành hình gì trong không gian $X$?

Để trả lời, chúng ta quan sát các **góc của hình vuông** bị biến đổi như thế nào. Hình vuông có 4 góc:
- $(z_1, z_2)$ - góc dưới trái
- $(z_1 + \Delta z_1, z_2)$ - góc dưới phải
- $(z_1, z_2 + \Delta z_2)$ - góc trên trái  
- $(z_1 + \Delta z_1, z_2 + \Delta z_2)$ - góc trên phải

Khi các góc này đi qua hàm $g$, chúng tạo thành một **hình bình hành** (không còn là hình vuông nữa!).

```
Không gian X (sau biến đổi):

    x₂ ↑
       │          ╱‾‾‾╲
       │         ╱  x  ╲  ← Hình bình hành
       │        ╱       ╲    (bị kéo dãn, xoay)
       │        ╲_______╱
       │
       └────────────────→ x₁
```

#### Bước 3: Tính các cạnh của hình bình hành bằng đạo hàm riêng

Để hiểu hình bình hành này, chúng ta cần biết độ dài và hướng của 2 cạnh xuất phát từ điểm $g(z)$.

**Cạnh thứ nhất** - khi dịch chuyển $\Delta z_1$ theo hướng $z_1$ (giữ nguyên $z_2$):

Điểm ban đầu: $(z_1, z_2) \to g(z_1, z_2)$

Điểm sau khi dịch: $(z_1 + \Delta z_1, z_2) \to g(z_1 + \Delta z_1, z_2)$

Vector thay đổi (xấp xỉ bằng đạo hàm riêng):
$$
\vec{v}_1 = g(z_1 + \Delta z_1, z_2) - g(z_1, z_2) \approx \frac{\partial g}{\partial z_1}(z) \times \Delta z_1
$$

Viết chi tiết từng thành phần:
$$
\vec{v}_1 = \begin{bmatrix} 
\dfrac{\partial g_1}{\partial z_1} \times \Delta z_1 \\[0.3em]
\dfrac{\partial g_2}{\partial z_1} \times \Delta z_1 
\end{bmatrix}
$$

**Cạnh thứ hai** - khi dịch chuyển $\Delta z_2$ theo hướng $z_2$ (giữ nguyên $z_1$):

Tương tự, vector thay đổi là:
$$
\vec{v}_2 = g(z_1, z_2 + \Delta z_2) - g(z_1, z_2) \approx \frac{\partial g}{\partial z_2}(z) \times \Delta z_2
$$

Viết chi tiết:
$$
\vec{v}_2 = \begin{bmatrix} 
\dfrac{\partial g_1}{\partial z_2} \times \Delta z_2 \\[0.3em]
\dfrac{\partial g_2}{\partial z_2} \times \Delta z_2 
\end{bmatrix}
$$

#### Bước 4: Diện tích hình bình hành = Định thức ma trận

Hình bình hành được tạo bởi 2 vector $\vec{v}_1$ và $\vec{v}_2$. Công thức tính diện tích hình bình hành trong không gian 2D là:

$$
\text{Diện tích} = \left\lvert \det \begin{bmatrix} v_1 & v_2 \end{bmatrix} \right\rvert
$$

Thay các thành phần của $\vec{v}_1$ và $\vec{v}_2$ vào:

$$
\Delta S_x = \left\lvert \det \begin{bmatrix} 
\dfrac{\partial g_1}{\partial z_1} \Delta z_1 & \dfrac{\partial g_1}{\partial z_2} \Delta z_2 \\[0.5em]
\dfrac{\partial g_2}{\partial z_1} \Delta z_1 & \dfrac{\partial g_2}{\partial z_2} \Delta z_2
\end{bmatrix} \right\rvert
$$

Đưa $\Delta z_1$ và $\Delta z_2$ ra ngoài định thức (tính chất của định thức):

$$
\Delta S_x = \left\lvert \det \begin{bmatrix} 
\dfrac{\partial g_1}{\partial z_1} & \dfrac{\partial g_1}{\partial z_2} \\[0.5em]
\dfrac{\partial g_2}{\partial z_1} & \dfrac{\partial g_2}{\partial z_2}
\end{bmatrix} \right\rvert \times \Delta z_1 \times \Delta z_2
$$

Ma trận này chính là **ma trận Jacobian** $J = \frac{\partial g}{\partial z}$. Vậy:

$$
\boxed{\Delta S_x = \left\lvert \det(J) \right\rvert \times \Delta S_z}
$$

**Ý nghĩa:** Định thức Jacobian cho biết **hệ số co dãn diện tích** khi biến đổi từ không gian $Z$ sang không gian $X$. 
- Nếu $\lvert\det(J)\rvert = 2$, diện tích tăng gấp đôi (giãn ra)
- Nếu $\lvert\det(J)\rvert = 0.5$, diện tích giảm một nửa (co lại)

#### Bước 5: Bảo toàn khối lượng xác suất

Đây là bước **quan trọng nhất**. Trong xác suất, "khối lượng" phải được bảo toàn qua phép biến đổi.

**Khối lượng xác suất trong không gian Z:**
$$
\text{Khối lượng}_Z = p_Z(z) \times \Delta S_z
$$

**Khối lượng xác suất trong không gian X:**
$$
\text{Khối lượng}_X = p_X(x) \times \Delta S_x
$$

**Điều kiện bảo toàn:** Hai khối lượng này phải bằng nhau!
$$
p_Z(z) \times \Delta S_z = p_X(x) \times \Delta S_x
$$

Thay $\Delta S_x = \lvert\det(J)\rvert \times \Delta S_z$ vào:
$$
p_Z(z) \times \Delta S_z = p_X(x) \times \lvert\det(J)\rvert \times \Delta S_z
$$

Chia cả hai vế cho $\Delta S_z$:
$$
p_Z(z) = p_X(x) \times \lvert\det(J)\rvert
$$

Sắp xếp lại:
$$
\boxed{p_X(x) = \frac{p_Z(z)}{\lvert\det(J)\rvert}}
$$

Với $J = \frac{\partial g}{\partial z}$ là Jacobian của hàm biến đổi $g: z \to x$.

#### Bước 6: Chuyển sang hàm nghịch đảo

Trong thực tế, chúng ta thường có $x$ (data) và muốn tìm $z = g^{-1}(x)$ (latent code). Vì thế, thuận tiện hơn nếu biểu diễn theo Jacobian của hàm nghịch đảo.

**Quan hệ giữa Jacobian thuận và nghịch:**
$$
\frac{\partial g}{\partial z} \times \frac{\partial g^{-1}}{\partial x} = I
$$

Lấy định thức hai vế:
$$
\det\left(\frac{\partial g}{\partial z}\right) \times \det\left(\frac{\partial g^{-1}}{\partial x}\right) = 1
$$

Suy ra:
$$
\left\lvert\det\left(\frac{\partial g}{\partial z}\right)\right\rvert = \frac{1}{\left\lvert\det\left(\frac{\partial g^{-1}}{\partial x}\right)\right\rvert}
$$

Thay vào công thức trên:
$$
\boxed{p_X(x) = p_Z(g^{-1}(x)) \times \left\lvert\det\left(\frac{\partial g^{-1}}{\partial x}\right)\right\rvert}
$$

#### Bước 7: Logarit hóa để tính toán ổn định

Trong thực tế, khi stack nhiều lớp biến đổi, các định thức nhân với nhau có thể rất lớn hoặc rất nhỏ, gây tràn số (overflow/underflow). Giải pháp: **lấy logarit**!

$$
\boxed{\log p_X(x) = \log p_Z(g^{-1}(x)) + \log \left\lvert\det\left(\frac{\partial g^{-1}}{\partial x}\right)\right\rvert}
$$

**Ưu điểm:**
- Phép nhân → Phép cộng (ổn định hơn)
- Tối ưu hóa dễ dàng hơn (gradient descent trên log-likelihood)
- Tránh tràn số

#### Ví dụ cụ thể: Biến đổi Affine 2D

Cho phép biến đổi:
$$
g(z) = Az + b \quad \text{với} \quad A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}, \quad b = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

**Bước 1:** Ma trận Jacobian
$$
J = \frac{\partial g}{\partial z} = A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}
$$

**Bước 2:** Tính định thức
$$
\det(J) = 2 \times 3 - 1 \times 0 = 6
$$

**Bước 3:** Diện tích co dãn
- Một hình vuông có diện tích 1 trong không gian $Z$
- Sẽ biến thành hình bình hành có diện tích 6 trong không gian $X$
- Diện tích tăng 6 lần!

**Bước 4:** Mật độ xác suất
$$
p_X(x) = \frac{p_Z(z)}{6}
$$

Vì diện tích tăng 6 lần, mật độ phải giảm 6 lần để bảo toàn tổng xác suất = 1.

**Visualization với code:**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Định nghĩa phép biến đổi
A = np.array([[2, 1], [0, 3]])
b = np.array([1, -1])

def transform(z):
    return A @ z + b

# Tạo hình vuông trong không gian Z
square_z = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

# Biến đổi sang không gian X
parallelogram_x = np.array([transform(pt) for pt in square_z])

# Vẽ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Không gian Z
ax1.add_patch(Polygon(square_z, fill=False, edgecolor='blue', linewidth=2))
ax1.scatter([0], [0], color='red', s=100, zorder=5)
ax1.set_xlim(-0.5, 2)
ax1.set_ylim(-0.5, 2)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Không gian Z (ban đầu)\nDiện tích = 1', fontsize=12)
ax1.set_xlabel('z₁')
ax1.set_ylabel('z₂')

# Không gian X
ax2.add_patch(Polygon(parallelogram_x, fill=False, edgecolor='red', linewidth=2))
ax2.scatter([1], [-1], color='red', s=100, zorder=5)
ax2.set_xlim(-0.5, 4)
ax2.set_ylim(-2, 4)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('Không gian X (sau biến đổi)\nDiện tích = 6', fontsize=12)
ax2.set_xlabel('x₁')
ax2.set_ylabel('x₂')

plt.tight_layout()
plt.show()

print(f"Định thức Jacobian: {np.linalg.det(A)}")
print(f"Diện tích hình vuông ban đầu: 1")
print(f"Diện tích hình bình hành sau biến đổi: {np.linalg.det(A)}")
```

#### Tổng kết trực quan

```
┌─────────────────────────────────────────────────────────────┐
│  CHANGE OF VARIABLES - TRỰC QUAN HÌNH HỌC                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Không gian Z          Biến đổi g          Không gian X     │
│                                                             │
│    ┌─────┐                              ╱────╲              │
│    │  z  │            ───────>         ╱  x   ╲             │
│    └─────┘                            ╱________╲            │
│                                                             │
│  Diện tích: ΔSz                     Diện tích: ΔSx          │
│                                                             │
│  Mật độ: p_Z(z)                     Mật độ: p_X(x)          │
│                                                             │
│  Khối lượng = p_Z(z) × ΔSz    =    p_X(x) × ΔSx             │
│                                                             │
│              ΔSx = |det(J)| × ΔSz                           │
│                                                             │
│              p_X(x) = p_Z(z) / |det(J)|                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Điểm mấu chốt cần nhớ:**

1. **Diện tích thay đổi** theo hệ số $\lvert\det(J)\rvert$
2. **Mật độ thay đổi ngược chiều** để bảo toàn khối lượng xác suất
3. **Định thức Jacobian** là "phí co giãn" của không gian
4. Công thức tổng quát hóa cho không gian bất kỳ chiều nào

### Ví dụ cụ thể

Cho $z \sim \mathcal{N}(0, 1)$ và $x = 2z + 1$:

**Bước 1: Hàm ngược**

$$
f^{-1}(x) = \frac{x - 1}{2}
$$

**Bước 2: Đạo hàm**

$$
\frac{df^{-1}}{dx} = \frac{1}{2}
$$

**Bước 3: Mật độ của x**

$$
p_x(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-1)^2}{8}\right) \cdot \frac{1}{2}
$$

**Kết quả:** $x \sim \mathcal{N}(1, 4)$ (trung bình 1, phương sai 4)

**Code minh họa:**

```python
import torch
import torch.distributions as D

# Base distribution
p_z = D.Normal(0, 1)
z = torch.linspace(-5, 5, 1000)

# Transformation: x = 2z + 1
x = 2 * z + 1

# Change of variables
log_p_z = p_z.log_prob(z)
# log |df/dz|. We need to subtract log |df/dz|
log_det_jacobian = torch.log(torch.tensor(2.0))
log_p_x = log_p_z - log_det_jacobian

# Verify: should match N(1, 2) distribution
# Note: D.Normal(mean, std_dev)
p_x_true = D.Normal(1, 2)
log_p_x_true = p_x_true.log_prob(x)

print(f"Max log-prob difference: {(log_p_x - log_p_x_true).abs().max():.6f}")
# Output: ~0.000000 (perfect match!)
```

---

## 4. Jacobian: "Phí co giãn" của không gian

### Từ 1D sang nhiều chiều

Khi người thợ gốm làm việc với một khối đất sét 3D, thao tác của họ phức tạp hơn nhiều. Họ không chỉ kéo dãn theo một hướng, mà còn bóp, xoắn, và nặn từ mọi phía. Mỗi hành động này làm thay đổi thể tích của một phần đất sét.

Trong không gian nhiều chiều (ví dụ: một bức ảnh có hàng ngàn pixel), "phí co giãn" không còn là một con số đơn giản. Nó trở thành **định thức (determinant)** của **ma trận Jacobian** - một công cụ toán học đo lường sự thay đổi "thể tích" trong không gian đa chiều.

### Ma trận Jacobian

Cho hàm biến đổi $f: \mathbb{R}^d \to \mathbb{R}^d$, ma trận Jacobian là một bảng tổng hợp tất cả các đạo hàm riêng, cho biết mỗi chiều của output thay đổi như thế nào khi một chiều của input thay đổi.

$$
J = \frac{\partial f}{\partial z} = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_d}{\partial z_1} & \cdots & \frac{\partial f_d}{\partial z_d}
\end{bmatrix}
$$

**Ý nghĩa hình học:** 

Hãy tưởng tượng một hình vuông nhỏ trong khối đất sét ban đầu. Sau khi người thợ nặn, nó có thể trở thành một hình bình hành bị kéo dài và xoay đi. Ma trận Jacobian mô tả chính xác phép biến đổi từ hình vuông sang hình bình hành đó.

**Định thức:** 

Giá trị tuyệt đối của định thức cho chúng ta biết **diện tích (hoặc thể tích)** của hình bình hành đó lớn gấp bao nhiêu lần hình vuông ban đầu. Đây chính là "phí co giãn" trong không gian đa chiều!

### Change of Variables (nhiều chiều)

Công thức tổng quát trở thành:

$$
\log p_x(x) = \log p_z(z) - \log\left\lvert\det\left(\frac{\partial f}{\partial z}\right)\right\rvert
$$

**Vấn đề lớn:** Việc tính toán định thức của ma trận Jacobian có độ phức tạp $O(d^3)$. Đây là một rào cản khổng lồ. Nếu "tác phẩm" của chúng ta là một bức ảnh 64x64 pixel ($d=4096$), phép tính này gần như bất khả thi.

$$
d = 4096 \implies O(d^3) \approx O(6.8 \times 10^{10}) \text{ phép tính}
$$

Làm thế nào người thợ gốm có thể thực hiện một thao tác phức tạp mà vẫn dễ dàng tính được sự thay đổi thể tích? Đây là lúc cần đến những "kỹ thuật" thông minh.

### Ví dụ: Affine transformation 2D

```python
import torch

def affine_flow_2d(z, A, b):
    """
    x = Az + b
    
    Args:
        z: (batch_size, 2)
        A: (2, 2) transformation matrix
        b: (2,) bias
    Returns:
        x: transformed data
        log_det: log |det(A)|
    """
    x = z @ A.T + b  # Matrix multiplication
    log_det = torch.logdet(A)  # Log-determinant
    return x, log_det

# Example
batch_size = 100
z = torch.randn(batch_size, 2)  # Base samples

A = torch.tensor([[2.0, 0.5], 
                  [0.0, 1.5]])
b = torch.tensor([1.0, -0.5])

x, log_det = affine_flow_2d(z, A, b)

print(f"Log-determinant: {log_det:.4f}")
# Expected: log(2.0 * 1.5) = log(3.0) ≈ 1.0986
```

---

## 5. Kiến trúc thông minh: Coupling Layers

### Ý tưởng thiên tài

Làm thế nào để người thợ gốm bậc thầy có thể tạo ra những tác phẩm phức tạp mà không cần phải tính toán những chi phí khổng lồ? Họ sử dụng một mẹo cực kỳ thông minh: **chia để trị**.

Thay vì biến đổi toàn bộ khối đất sét cùng một lúc, người thợ giữ một nửa khối đất sét bằng một tay, và dùng tay kia để nặn nửa còn lại. Quan trọng hơn, cách họ nặn nửa thứ hai lại **phụ thuộc vào hình dạng của nửa thứ nhất**.

Đây chính là ý tưởng cốt lõi của **Coupling Layers**.

### Coupling Layer (RealNVP)

**Ý tưởng chính:** Chia các chiều dữ liệu (dimensions) thành hai phần, $z_A$ và $z_B$.

1.  **Giữ nguyên phần A:** $x_A = z_A$. Giống như tay giữ yên một nửa khối đất.
2.  **Biến đổi phần B:** Nửa thứ hai, $x_B$, được biến đổi bằng một hàm đơn giản (scale và shift), nhưng các tham số của hàm này (độ co giãn `s` và độ dịch chuyển `t`) lại được tính toán từ nửa thứ nhất, $z_A$.

Công thức biến đổi:

$$
\begin{aligned}
x_A &= z_A \\
x_B &= z_B \odot \exp(s(z_A)) + t(z_A)
\end{aligned}
$$

Trong đó $s(\cdot)$ và $t(\cdot)$ là các mạng neural nhỏ.

**Jacobian có dạng tam giác:**

Phép biến đổi này tạo ra một ma trận Jacobian có dạng tam giác dưới (lower triangular):

$$
J = \begin{bmatrix}
I & 0 \\
\frac{\partial x_B}{\partial z_A} & \text{diag}(\exp(s(z_A)))
\end{bmatrix}
$$

**Điều kỳ diệu:** Định thức của một ma trận tam giác chỉ đơn giản là **tích các phần tử trên đường chéo chính**.

$$
\det(J) = \prod \exp(s_i(z_A))
$$

Và log-determinant trở thành một phép cộng:

$$
\log\lvert\det(J)\rvert = \sum s_i(z_A)
$$

Phép tính từ $O(d^3)$ đã trở thành $O(d)$! Người thợ gốm giờ đây có thể thực hiện một bước nặn phức tạp mà vẫn tính được "phí co giãn" một cách dễ dàng.

Để đảm bảo mọi phần của "khối đất" đều được nặn, chúng ta chỉ cần hoán vị các chiều dữ liệu và lặp lại quá trình này nhiều lần.

### Implementation PyTorch

```python
import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    """RealNVP Coupling Layer"""
    
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        
        # Scale network: z1 -> s(z1)
        self.scale_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.half_dim)
        )
        
        # Translation network: z1 -> t(z1)
        self.translate_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.half_dim)
        )
    
    def forward(self, z):
        """
        Forward: z -> x
        Returns: x, log_det_jacobian
        """
        z1, z2 = z[:, :self.half_dim], z[:, self.half_dim:]
        
        s = self.scale_net(z1)  # Scale
        t = self.translate_net(z1)  # Translation
        
        # Transform second half
        x1 = z1  # First half unchanged
        x2 = z2 * torch.exp(s) + t
        
        x = torch.cat([x1, x2], dim=1)
        log_det = s.sum(dim=1)  # Sum of scale parameters
        
        return x, log_det
    
    def inverse(self, x):
        """
        Inverse: x -> z
        Returns: z, log_det_jacobian
        """
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
        
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        
        # Invert transformation
        z1 = x1
        z2 = (x2 - t) * torch.exp(-s)
        
        z = torch.cat([z1, z2], dim=1)
        log_det = -s.sum(dim=1)
        
        return z, log_det
```

### Stacking nhiều Coupling Layers

Single coupling layer chưa đủ expressive. Stack nhiều layers:

```python
class NormalizingFlow(nn.Module):
    """Stack of Coupling Layers"""
    
    def __init__(self, dim, num_flows=8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        
        # Create flow layers
        self.flows = nn.ModuleList([
            CouplingLayer(dim, hidden_dim)
            for _ in range(num_flows)
        ])
        
        # Permutations between layers (for better mixing)
        self.permutations = [
            torch.randperm(dim) for _ in range(num_flows)
        ]
    
    def forward(self, z):
        """z -> x"""
        log_det_total = 0
        x = z
        
        for flow, perm in zip(self.flows, self.permutations):
            x = x[:, perm]  # Permute dimensions
            x, log_det = flow(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def inverse(self, x):
        """x -> z"""
        log_det_total = 0
        z = x
        
        for flow, perm in zip(reversed(self.flows), 
                             reversed(self.permutations)):
            z, log_det = flow.inverse(z)
            inv_perm = torch.argsort(perm)  # Inverse permutation
            z = z[:, inv_perm]
            log_det_total += log_det
        
        return z, log_det_total
    
    def log_prob(self, x):
        """Compute log p(x)"""
        z, log_det = self.inverse(x)
        
        # Log-prob of base Gaussian
        log_p_z = -0.5 * (z**2).sum(dim=1) - \
                  0.5 * self.dim * torch.log(torch.tensor(2 * 3.14159))
        
        # Change of variables
        log_p_x = log_p_z + log_det
        return log_p_x
    
    def sample(self, num_samples):
        """Generate samples"""
        z = torch.randn(num_samples, self.dim)
        x, _ = self.forward(z)
        return x
```

### Training example

```python
def sample_real_data(batch_size):
    """
    Placeholder function to generate dummy data.
    Replace this with your actual data loading logic.
    """
    # Assuming your real data has the same dimension as the flow model (dim=2 in this case)
    # If your data has a different dimension, change the '2' below accordingly.
    return torch.randn(batch_size, 2) * 2 + 1 # Example: Gaussian data with mean 1, std dev 2

# Initialize model
model = NormalizingFlow(dim=2, num_flows=6, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    # Get batch of real data
    x_batch = sample_real_data(batch_size=256)  # Your dataset
    
    # Compute negative log-likelihood
    log_p_x = model.log_prob(x_batch)
    loss = -log_p_x.mean()  # Maximize likelihood = minimize NLL
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | NLL: {loss.item():.4f}")

# Generate samples
with torch.no_grad():
    samples = model.sample(1000)
```

---

## 6. Continuous Normalizing Flows

### Từ rời rạc sang liên tục

Đến giờ, chúng ta hình dung người thợ gốm thực hiện một chuỗi các thao tác **rời rạc**: nặn, kéo, xoắn... giống như xem một cuốn sách lật (flipbook). Mỗi trang là một bước biến đổi.

**Continuous Normalizing Flows (CNF)** đề xuất một góc nhìn khác: điều gì sẽ xảy ra nếu quá trình nặn gốm là một **dòng chảy liên tục, mượt mà**? Thay vì các bước riêng lẻ, hãy tưởng tượng một video quay chậm, ghi lại chuyển động của từng phân tử đất sét từ khối cầu ban đầu đến con rồng hoàn thiện.

**Discrete NF (sách lật):**
$$
z_0 \xrightarrow{f_1} z_1 \xrightarrow{f_2} \cdots \to z_K = x
$$

**Continuous NF (video):**
Thay vì một chuỗi hàm, chúng ta học một **trường vector (vector field)** $f$ phụ thuộc vào thời gian. Trường vector này giống như một "dòng chảy" vô hình, chỉ đạo hướng di chuyển cho mỗi "phân tử" $z$ tại mỗi thời điểm $t$.

$$
\frac{dz(t)}{dt} = f(z(t), t; \theta) \quad \text{với } t \in [0, 1]
$$

- $z(0)$ là một điểm trong khối đất sét ban đầu.
- $z(1)$ là điểm tương ứng trên tác phẩm hoàn thiện.
- $f(z(t), t)$ là "vận tốc" của hạt đất sét tại vị trí $z$ và thời gian $t$.

### Instantaneous Change of Variables

Với cách tiếp cận này, "phí co giãn" cũng được tính một cách liên tục. Thay vì tính định thức của cả một bước biến đổi lớn, chúng ta chỉ cần tính "tốc độ thay đổi thể tích" tại mỗi khoảnh khắc. Tốc độ này được đo bằng **vết (trace)** của ma trận Jacobian, là tổng các phần tử trên đường chéo chính.

$$
\frac{d \log p_t(z(t))}{dt} = -\text{Tr}\left(\frac{\partial f}{\partial z(t)}\right)
$$

**Ưu điểm lớn:** Tính trace ($O(d^2)$) rẻ hơn nhiều so với tính determinant ($O(d^3)$). Hơn nữa, chúng ta có thể ước tính nó một cách hiệu quả với độ phức tạp chỉ $O(d)$ bằng **Hutchinson's Trace Estimator**. Người thợ gốm giờ đây có thể theo dõi sự thay đổi thể tích một cách liên tục và hiệu quả.

### Augmented ODE

Để vừa biến đổi hình dạng, vừa theo dõi "phí co giãn", chúng ta giải một phương trình vi phân kết hợp cả hai:

$$
\frac{d}{dt} \begin{bmatrix} z(t) \\ \log p_t(z(t)) \end{bmatrix} = \begin{bmatrix} f(z(t), t) \\ -\text{Tr}\left(\frac{\partial f}{\partial z(t)}\right) \end{bmatrix}
$$

Giải phương trình này từ $t=0$ đến $t=1$ sẽ cho chúng ta cả tác phẩm $x=z(1)$ và tổng "phí co giãn" tích lũy trên toàn bộ quá trình.

### Hutchinson's Trace Estimator

Tính trace chính xác vẫn cost $O(d^2)$. Dùng **stochastic estimator**:

$$
\text{Tr}(J) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [\epsilon^T J \epsilon]
$$

Chỉ cần:
- Sample 1 vector $\epsilon$
- Tính vector-Jacobian product $J\epsilon$ (via autograd)
- Dot product $\epsilon^T (J\epsilon)$

→ $O(d)$ complexity!

**Implementation:**

```python
def hutchinson_trace(func, z, num_samples=1):
    """
    Estimate tr(df/dz) using Hutchinson's estimator
    
    Args:
        func: function z -> f(z)
        z: input (batch_size, dim)
        num_samples: number of random vectors
    Returns:
        trace estimate (batch_size,)
    """
    batch_size, dim = z.shape
    trace = 0
    
    for _ in range(num_samples):
        # Random Rademacher vector (+1 or -1)
        eps = torch.randint(0, 2, (batch_size, dim), device=z.device) * 2 - 1
        eps = eps.float()
        
        # Compute ε^T (∂f/∂z) ε
        z.requires_grad_(True)
        f_z = func(z)
        
        vjp = torch.autograd.grad(
            outputs=f_z,
            inputs=z,
            grad_outputs=eps,
            create_graph=True,
            retain_graph=True
        )[0]
        
        trace += (eps * vjp).sum(dim=1)
    
    return trace / num_samples
```

---

## 7. Implementation đầy đủ với PyTorch

### Vector Field Network

```python
class TimeConditionedVectorField(nn.Module):
    """Vector field f(z, t) for CNF"""
    
    def __init__(self, dim, hidden_dim=128, time_embed_dim=32):
        super().__init__()
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Tanh(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Tanh()
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim + time_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, z):
        """
        Note: torchdiffeq expects signature (t, z)
        
        Args:
            t: scalar time
            z: (batch_size, dim)
        Returns:
            dz/dt: (batch_size, dim)
        """
        batch_size = z.shape[0]
        
        # Embed time
        t_embed = self.time_embed(t.view(1, 1).expand(batch_size, 1))
        
        # Concatenate z and time embedding
        tz = torch.cat([z, t_embed], dim=1)
        
        return self.net(tz)
```

### CNF với torchdiffeq

```python
from torchdiffeq import odeint

class ContinuousNormalizingFlow(nn.Module):
    """Basic CNF without trace computation"""
    
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.vf = TimeConditionedVectorField(dim, hidden_dim)
    
    def forward(self, z0, t_span=None):
        """
        Integrate from t=0 to t=1
        
        Args:
            z0: initial state (batch_size, dim)
            t_span: time points (default: [0, 1])
        Returns:
            trajectory of states
        """
        if t_span is None:
            t_span = torch.tensor([0., 1.]).to(z0.device)
        
        # Solve ODE
        z_traj = odeint(
            self.vf,
            z0,
            t_span,
            method='dopri5',  # Adaptive Runge-Kutta
            rtol=1e-5,
            atol=1e-7
        )
        
        return z_traj
    
    def sample(self, num_samples, device='cuda'):
        """Generate samples"""
        z0 = torch.randn(num_samples, self.dim).to(device)
        z_traj = self.forward(z0)
        return z_traj[-1]  # Return x = z(t=1)
```

---

## 8. Advanced Topics & FFJORD

### FFJORD: Nghệ thuật điêu khắc tự do

Nếu Coupling Layers là kỹ thuật "chia để trị" thông minh, thì **FFJORD** (Free-Form Jacobian of Reversible Dynamics) đưa người thợ gốm lên một tầm cao mới: **nghệ thuật điêu khắc tự do**.

Với các kiến trúc trước đây như RealNVP, người thợ bị giới hạn trong các thao tác có cấu trúc (giữ một nửa, nặn nửa kia). FFJORD giải phóng họ khỏi những ràng buộc này. Giờ đây, người thợ có thể dùng cả hai tay, nặn toàn bộ khối đất sét một cách tự do, tạo ra những hình dạng phức tạp và tự nhiên hơn nhiều.

FFJORD là một kiến trúc CNF hiện đại, cho phép sử dụng một mạng neural "tự do" (free-form) để định nghĩa trường vector $f(z, t)$. Nó kết hợp sức mạnh của CNF (kiến trúc không bị ràng buộc) và khả năng tính toán likelihood chính xác thông qua việc ước tính trace một cách hiệu quả.

```python
class FFJORD(nn.Module):
    """Complete CNF with exact log-likelihood computation"""
    
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.vf = TimeConditionedVectorField(dim, hidden_dim)
    
    def _augmented_dynamics(self, t, state):
        """
        Augmented ODE: [dz/dt, d(log p)/dt]
        
        Args:
            state: [z, log_p] concatenated (batch_size, dim+1)
        Returns:
            [dz/dt, dlogp/dt] (batch_size, dim+1)
        """
        z = state[:, :self.dim]
        
        # Compute dz/dt = f(z, t)
        dz_dt = self.vf(t, z)
        
        # Compute trace using Hutchinson estimator
        trace = hutchinson_trace(
            lambda z_: self.vf(t, z_),
            z,
            num_samples=1
        )
        
        # d(log p)/dt = -tr(df/dz)
        dlogp_dt = -trace
        
        # Concatenate
        return torch.cat([dz_dt, dlogp_dt.unsqueeze(1)], dim=1)
    
    def forward(self, z0, t_span=None):
        """Forward integration with log-prob tracking"""
        if t_span is None:
            t_span = torch.tensor([0., 1.]).to(z0.device)
        
        batch_size = z0.shape[0]
        
        # Initialize augmented state: [z, log_p=0]
        logp_0 = torch.zeros(batch_size, 1).to(z0)
        state_0 = torch.cat([z0, logp_0], dim=1)
        
        # Integrate
        state_traj = odeint(
            self._augmented_dynamics,
            state_0,
            t_span,
            method='dopri5',
            rtol=1e-5,
            atol=1e-7
        )
        
        return state_traj
    
    def log_prob(self, x):
        """
        Compute log p(x) exactly via backward integration
        
        Args:
            x: data points (batch_size, dim)
        Returns:
            log_p_x: (batch_size,)
        """
        # Integrate backward: x (t=1) -> z0 (t=0)
        t_span = torch.tensor([1., 0.]).to(x.device)
        batch_size = x.shape[0]
        
        # Initialize: [x, log_p=0]
        logp_1 = torch.zeros(batch_size, 1).to(x)
        state_1 = torch.cat([x, logp_1], dim=1)
        
        # Solve ODE
        state_traj = odeint(
            self._augmented_dynamics,
            state_1,
            t_span,
            method='dopri5'
        )
        
        state_0 = state_traj[-1]
        z0 = state_0[:, :self.dim]
        delta_logp = state_0[:, self.dim]
        
        # Base distribution log-prob
        log_p_z0 = -0.5 * (z0**2).sum(dim=1) - \
                   0.5 * self.dim * torch.log(torch.tensor(2 * 3.14159))
        
        # Add change-of-variables correction
        log_p_x = log_p_z0 - delta_logp
        
        return log_p_x
    
    def sample(self, num_samples, device='cuda'):
        """Generate samples"""
        z0 = torch.randn(num_samples, self.dim).to(device)
        state_traj = self.forward(z0)
        return state_traj[-1, :, :self.dim]  # Extract z(t=1)
```

### Training FFJORD

```python
import numpy as np

# Initialize model
model = FFJORD(dim=2, hidden_dim=128).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5000):
    # Sample batch from real data
    x_batch = sample_real_data(batch_size=128)
    x_batch = torch.tensor(x_batch).float().cuda()
    
    # Compute negative log-likelihood
    log_p_x = model.log_prob(x_batch)
    loss = -log_p_x.mean()
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (important for stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | NLL: {loss.item():.4f}")
        
        # Visualize samples
        with torch.no_grad():
            samples = model.sample(1000).cpu().numpy()
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
            plt.title(f"Epoch {epoch}")
            plt.show()
```

### Regularization Techniques

**Kinetic Energy Regularization:**

$$
\mathcal{L}_{\text{reg}} = \lambda \int_0^1 \|f(z(t), t)\|^2 dt
$$

Giúp vector field smooth hơn, giảm NFE (number of function evaluations).

```python
def kinetic_energy_loss(model, z0, lambda_reg=0.01):
    """Compute kinetic energy regularization"""
    t_samples = torch.linspace(0, 1, 10).to(z0.device)
    
    ke_loss = 0
    for t in t_samples:
        z_t = odeint(model.vf, z0, torch.tensor([0., t.item()]))[1]
        v_t = model.vf(t, z_t)
        ke_loss += (v_t ** 2).sum(dim=1).mean()
    
    return lambda_reg * ke_loss / len(t_samples)
```

---

## 9. Kết luận

### Key Takeaways

1. **Normalizing Flow = Chuỗi biến đổi khả nghịch**
   - Base distribution (simple) → Data distribution (complex)
   - Exact likelihood computation

2. **Change of variables formula**
   - Theo dõi sự thay đổi mật độ qua định thức Jacobian. Công thức log-likelihood:
     
     $$\log p_x(x) = \log p_z(z) - \log\lvert\det(J)\rvert$$

3. **Coupling Layers = Kiến trúc thông minh**
   - Jacobian có cấu trúc đặc biệt (triangular)
   - $O(d)$ thay vì $O(d^3)$ complexity

4. **CNF = Continuous-time dynamics**
   - ODE formulation: $dz/dt = f(z, t)$
   - Trace thay vì determinant

5. **FFJORD = State-of-the-art**
   - Free-form architectures
   - Hutchinson trace estimator
   - Exact likelihood với efficient computation

### So sánh Discrete NF vs CNF

| Aspect | Discrete NF | CNF |
|--------------|---------------------------------------|--------------------------|
| **Kiến trúc** | Bị ràng buộc (coupling/autoregressive) | Dạng tự do (Free-form)   |
| **Bộ nhớ**    | $O(K \cdot d)$ (với K lớp)            | $O(d)$ (hằng số)         |
| **Tính toán**  | Nhanh (một lượt)                       | Chậm (phụ thuộc ODE solver) |
| **Linh hoạt**  | Giới hạn bởi kiến trúc                | Về lý thuyết là không giới hạn |

### Applications

- **Density estimation**: Modeling complex distributions
- **Variational inference**: Flexible posteriors trong Bayesian models
- **Generative modeling**: Image/audio/molecular generation
- **Anomaly detection**: Out-of-distribution detection via likelihood

### Future Directions

- **Flow Matching**: Regression-based training (không cần ODE solver)
- **Rectified Flows**: Straighten trajectories cho faster sampling
- **Diffusion + Flows**: Kết hợp ưu điểm của cả hai

---

## Tài liệu tham khảo

1. **Rezende & Mohamed (2015)** - "Variational Inference with Normalizing Flows" (ICML)
2. **Dinh et al. (2017)** - "Density estimation using Real NVP" (ICLR)
3. **Kingma & Dhariwal (2018)** - "Glow: Generative Flow using Invertible 1x1 Convolutions" (NeurIPS)
4. **Chen et al. (2018)** - "Neural Ordinary Differential Equations" (NeurIPS Best Paper)
5. **Grathwohl et al. (2019)** - "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (ICLR)

<script src="/assets/js/katex-init.js"></script>
