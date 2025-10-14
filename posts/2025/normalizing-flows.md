# Normalizing Flows và Continuous Normalizing Flows: Nguồn gốc, Động cơ, Giải thích chi tiết & Ví dụ trực quan

## 1. Nguồn gốc và Lý do xây dựng thuật toán

Trong generative modeling, mục tiêu là học phân phối dữ liệu phức tạp để sinh ra mẫu dữ liệu mới tương tự thật. Các mô hình trước như VAEs (dễ sinh mẫu, nhưng likelihood chỉ xấp xỉ) hoặc GANs (sinh mẫu đẹp, nhưng không tính likelihood) đều có giới hạn. Ngược lại, **Normalizing Flows (NF)** giúp:
- Tính toán chính xác xác suất từng mẫu.
- Sinh mẫu hiệu quả qua biến đổi có thể đảo ngược giữa phân phối đơn giản (Gaussian) và dữ liệu thực tế.

---

## 2. Thành phần chi tiết & Ví dụ trực quan

### 2.1 Công thức biến đổi biến ngẫu nhiên

**Ví dụ 1: Biến đổi đơn giản**

Giả sử bạn có một biến ngẫu nhiên chuẩn $z \sim \mathcal{N}(0,1)$ và hàm $f(z) = 2z + 1$. Vậy $x = f(z)$ là phân phối nào? Dùng công thức biến đổi:
- $f^{-1}(x) = (x - 1)/2$
- Mật độ $p_x(x) = p_z(f^{-1}(x)) \cdot |df^{-1}/dx|$
- Với $df^{-1}/dx = 1/2$, nên

\[
p_x(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-1)^2}{8}\right) \cdot \frac{1}{2}
\]

Biến đổi kéo giãn và dịch chuyển Gaussian thành Gaussian mới.

**Ví dụ 2: Biến đổi phi tuyến**

Giả sử $f(z) = z^3$.
- $f^{-1}(x) = x^{1/3}$
- $df^{-1}/dx = \frac{1}{3}x^{-2/3}$

Mật độ mới sẽ bị bóp méo, không phải Gaussian, và vùng $x$ lớn/nhỏ sẽ scale khác nhau tùy thuộc độ dốc của $f$.

### 2.2 Thách thức Jacobian

**Ví dụ 3: Ảnh số & Ma trận lớn**

Với ảnh đầu vào $x \in \mathbb{R}^{4096}$ (64x64 pixels), ma trận Jacobian có $4096^2$ phần tử. Nếu dùng biến đổi tổng quát, việc tính determinant là không khả thi. Vì vậy các kiến trúc như RealNVP, Glow dùng coupling layers đảm bảo việc tính determinant chỉ còn là tích các số (cực kì nhanh).

### 2.3 Ghép nhiều flows

**Ví dụ 4: Chuỗi biến đổi đơn giản**

Thay vì chỉ dùng $f(z)$, ta xếp $f_1, f_2, ..., f_K$ để mô hình hóa dần sự phức tạp. Tưởng tượng như bạn xoay, kéo giãn, lật trong không gian đặc trưng, cứ mỗi biến đổi là một bước tiến sát đến phân phối dữ liệu thực tế hơn.

---

## 3. Continuous Normalizing Flows (CNF)

**Ví dụ 5: Dòng nước chảy**

Thay vì biến đổi từng bước, CNF là biến đổi liên tục: tưởng tượng một giọt nước di chuyển dọc theo dòng chảy vector field (hàm $f(z, t)$), chuyển từ vị trí gốc (phân phối base) tới vị trí đích (phân phối cuối) qua thời gian $t$.

Như vẽ xiên một vùng mực nước, mỗi điểm di chuyển theo vector hướng dẫn $f(z, t)$ - không phải nhảy từng bước mà biến đổi dần dần.

### Định lý Instantaneous Change of Variables

Thay vì đo thể tích của không gian tại điểm cuối bằng Jacobian, ta theo dõi liên tục **tốc độ “co giãn”** thể tích khi nước chảy, rồi tích hợp toàn bộ quá trình lại để tính mật độ cuối cùng.

---

## 4. Kết nối Neural ODE

**Ví dụ 6: Giải phương trình vi phân bằng máy tính**

Cũng như dự báo đường đi của vật thể dưới lực tác động, mẫu $z_0$ sẽ “trôi” theo vector field $f(z, t)$ để tới $x$. Máy tính dùng các solver số (thường Runge-Kutta) để tính toán các điểm “nước chảy” cách nhau $\Delta t$ nhỏ, càng nhỏ thì mô phỏng càng chính xác.

---

## 5. Mã nguồn minh hoạ

Một mạng vector field đơn giản (PyTorch):
```python
import torch.nn as nn
class VectorField(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, t, z):
        t_expand = t.expand(z.shape[0], 1)
        tz = torch.cat([t_expand, z], dim=1)
        return self.net(tz)
```

---

## 6. Tổng kết

- NF cho phép mô hình hoá phân phối phức tạp nhờ biến đổi đơn ánh với tính likelihood chính xác.
- CNF mở rộng biến đổi sang liên tục, giảm chi phí Jacobian, giúp mạng tự do hơn, cho phép sử dụng các kiến trúc tân tiến như Neural ODE.

Các ví dụ đơn lẻ giúp minh hoạ cách các khái niệm vận dụng vào bài toán thực - từ biến đổi số đơn giản, đến mô phỏng dòng chảy liên tục với neural network.