---
title: "RealNVP & Glow: Nghệ thuật biến đổi có thể đảo ngược"
date: "2025-02-12"
category: "flow-based-models"
tags: ["realnvp", "glow", "normalizing-flows", "invertible-networks", "pytorch"]
excerpt: "Học RealNVP và Glow qua câu chuyện, ví dụ đời thực, phân tích toán học cụ thể và đoạn code PyTorch hữu ích."
author: "ThanhLamDev"
readingTime: 18
featured: false
---

# RealNVP & Glow

**Tiếp nối bài Normalizing Flow, người thợ gốm của chúng ta giờ đã chuyển sang xưởng pha lê – nơi RealNVP và Glow trở thành những kỹ thuật nòng cốt để vừa giữ tính khả nghịch, vừa vận hành đủ nhanh cho khán giả đang chờ.**

## Mục lục

1. [Câu chuyện về xưởng pha lê thời gian thực](#1-câu-chuyện-về-xưởng-pha-lê-thời-gian-thực)
2. [Áp lực từ phòng trưng bày pha lê](#2-áp-lực-từ-phòng-trưng-bày-pha-lê)
3. [Từ trực giác đến RealNVP](#3-từ-trực-giác-đến-realnvp)
4. [Kiến trúc RealNVP từng lớp](#4-kiến-trúc-realnvp-từng-lớp)
5. [Ví dụ toán học: Coupling 2D tối giản](#5-ví-dụ-toán-học-coupling-2d-tối-giản)
6. [Glow: Khi RealNVP học được “điệu nhảy” 1x1](#6-glow-khi-realnvp-học-được-điệu-nhảy-1x1)
7. [Code thú vị: Mini RealNVP + Glow block với PyTorch](#7-code-thú-vị-mini-realnvp--glow-block-với-pytorch)
8. [Gợi ý thực nghiệm & các bẫy thường gặp](#8-gợi-ý-thực-nghiệm--các-bẫy-thường-gặp)
9. [Kết luận & tài liệu](#9-kết-luận--tài-liệu)

---

## 1. Câu chuyện về xưởng pha lê thời gian thực

Từ khối đất sét của bài trước, người thợ giờ làm việc với khối pha lê chuẩn – biểu tượng cho base Gaussian. Khách bước vào, chọn một mẫu bất kỳ và muốn thấy nó biến thành chiếc bình hay quả cầu theo ý thích. Mỗi thao tác phải đảo ngược được, vì chỉ cần khách đổi ý là anh phải trả khối pha lê về trạng thái ban đầu. Đó chính là trực giác đứng sau **Real-valued Non-Volume Preserving (RealNVP)**: biến đổi được thiết kế để dễ dàng tiến và lùi.

Glow là bước nâng cấp tự nhiên: trước khi nặn tiếp, người thợ xoay nhẹ khối pha lê để ánh sáng rọi đúng góc và chuẩn hóa nhiệt độ của lò nung. Hai động tác này tương ứng với invertible 1x1 convolution và ActNorm – giúp flow linh hoạt hơn mà vẫn kiểm soát được log-determinant một cách gọn gàng.

## 2. Áp lực từ phòng trưng bày pha lê

Phòng trưng bày mới đông khách khiến người thợ phải làm nhanh hơn nhưng không được phép đánh mất sự chính xác. Anh cần một quy trình biến đổi có “vũ đạo” rõ ràng: nửa khối pha lê làm điểm tựa, nửa còn lại biến hóa theo nhịp điệu do các mạng $s(\cdot)$ và $t(\cdot)$ quyết định; mọi thứ có thể đảo ngược tức khắc nếu khách yêu cầu chỉnh sửa. RealNVP cung cấp cấu trúc đó, còn Glow thay việc đổi mặt nạ thủ công bằng một phép xoay học được để ánh sáng được trộn đều hơn.

Bối cảnh đã đủ: chúng ta lùi khỏi câu chuyện, bước vào phần kỹ thuật để xem hai kiến trúc này vận hành ra sao.

## 3. Từ trực giác đến RealNVP

Ở bài Normalizing Flow & CNF, chúng ta dừng lại ở ý tưởng “chuỗi biến đổi có thể đảo”. Giờ nối tiếp câu chuyện, từ kinh nghiệm của người nghệ nhân, ta rút ra ba yêu cầu cho flow:

1. **Chuỗi phép biến đổi khả nghịch** $f_1, f_2, \dots, f_K$ để đi từ base Gaussian $z_0$ thành ảnh dữ liệu $x$.
2. **Tính toán log-likelihood chính xác**:
   
   $$
   \log p_X(x) = \log p_Z(z_0) - \sum_{k=1}^K \log\left\lvert\det\left(\frac{\partial f_k}{\partial z_{k-1}}\right)\right\rvert
   $$

3. **Độ phức tạp tuyến tính** theo số chiều (ảnh $64 \times 64 \times 3$ có 12,288 chiều).

RealNVP giải bài toán bằng cách thiết kế mỗi $f_k$ sao cho ma trận Jacobian là **tam giác** ⇒ định thức chỉ là tích đường chéo ⇒ phép tính $O(D)$.

## 4. Kiến trúc RealNVP từng lớp

### 4.1 Coupling layer kiểu "giữ - nặn"

Chia vector $z$ thành hai phần theo mask $m \in \{0,1\}^D$:

$$
\begin{aligned}
z_A &= m \odot z, \quad z_B = (1 - m) \odot z \\
t &= T_\theta(z_A), \quad s = S_\theta(z_A)
\end{aligned}
$$

Trong đó $T_\theta, S_\theta$ là các mạng nhỏ (MLP/CNN).

Forward (đi từ base → data):

$$
\begin{aligned}
x_A &= z_A \\
x_B &= z_B \odot \exp(s) + t
\end{aligned}
$$

Inverse (đi từ data → base) cực kỳ đơn giản:

$$
\begin{aligned}
z_A &= x_A, \\
z_B &= (x_B - t) \odot \exp(-s)
\end{aligned}
$$

**Log-det Jacobian**:

$$
\log\lvert\det J\rvert = \sum_{i: m_i = 0} s_i
$$

### 4.2 Hoán vị & multi-scale

Nếu ta giữ nguyên cùng một mặt nạ, chỉ đúng một nửa chiều được biến đổi. Vì vậy RealNVP:

- Xen kẽ các mask khác nhau (checkerboard ↔ channel-wise).
- Chèn bước **permutation** đơn giản giữa các layer (ví dụ đảo chiều channel).
- Ở cấp ảnh, sử dụng thủ thuật **squeeze** + **split**: sau vài bước, đưa một phần feature map ra latent và tiếp tục xử lý phần còn lại ⇒ multi-scale latent giống pyramids.

### 4.3 So sánh nhanh với MADE/IAF

| Tiêu chí | Coupling (RealNVP) | Autoregressive (MAF/IAF) |
|----------|--------------------|---------------------------|
| Forward tốc độ | Rất nhanh (song song) | Chậm (tuần tự) |
| Inverse tốc độ | Chậm (giải phương trình) | Rất nhanh |
| Ứng dụng điển hình | Generative sampling | Density estimation |

RealNVP được chọn vì forward sampling cần nhanh, huấn luyện có thể dùng mini-batch lớn.

## 5. Ví dụ toán học: Coupling 2D tối giản

Xét base distribution $z = [z_1, z_2]^\top \sim \mathcal{N}(\mathbf{0}, I)$ và mặt nạ `m = [1, 0]`. Ta đặt:

$$
s(z_1) = 0.8 z_1, \quad t(z_1) = 0.5 z_1
$$

Forward:

$$
\begin{aligned}
x_1 &= z_1 \\
x_2 &= z_2 \exp(0.8 z_1) + 0.5 z_1
\end{aligned}
$$

Inverse:

$$
\begin{aligned}
z_1 &= x_1 \\
z_2 &= \left(x_2 - 0.5 x_1\right) \exp(-0.8 x_1)
\end{aligned}
$$

Log-det Jacobian:

$$
\log|\det J| = 0.8 x_1
$$

Log-likelihood của điểm $x$:

$$
\log p_X(x) = \log p_Z(z) - 0.8 x_1
$$

Với $p_Z$ là Gaussian chuẩn, ta có:

$$
\log p_X(x) = -\frac{1}{2}\left(x_1^2 + \big(x_2 - 0.5 x_1\big)^2 \exp(-1.6 x_1)\right) - \log (2\pi) - 0.8 x_1
$$

Ví dụ số trong Python:

```python
import torch

x = torch.tensor([[1.2, -0.7]])
x1, x2 = x[:, 0], x[:, 1]
z1 = x1
z2 = (x2 - 0.5 * x1) * torch.exp(-0.8 * x1)

log_p_z = -0.5 * (z1**2 + z2**2) - torch.log(torch.tensor(2 * torch.pi))
log_det = -0.8 * x1  # Dấu trừ vì từ x -> z
log_p_x = log_p_z + log_det
print(float(log_p_x))  # ~ -1.967
```

Nhờ coupling layer, việc tính log-likelihood chỉ là vài phép cộng nhân thay vì xử lý ma trận lớn.

## 6. Glow: Khi RealNVP học được “điệu nhảy” 1x1

Glow (Kingma & Dhariwal, 2018) kế thừa RealNVP nhưng thêm ba ý tưởng giúp mẫu ảnh sắc nét:

1. **ActNorm**: mỗi channel có scale $s$ và bias $b$ được khởi tạo theo mini-batch đầu tiên để đảm bảo zero-mean, unit-var. Biến đổi:
   
   $$
   y = s \odot (x - b), \quad \log|\det J| = HW \sum_c \log |s_c|
   $$

2. **Invertible 1x1 Convolution**: thay permutation cố định bằng ma trận khả nghịch $W \in \mathbb{R}^{c \times c}$. Với ảnh hình $H \times W$:
   
   $$
   \log|\det J| = HW \cdot \log|\det W|
   $$
   
   Để tính nhanh, Glow lưu decomposition $W = PLU$ ⇒ $\log|\det W| = \sum_i \log|u_{ii}|$.

3. **Multi-scale architecture**: sau $K$ bước, “tách” một nửa channel thành latent, phần còn lại tiếp tục đi qua các scale tiếp theo. Điều này giúp mô hình tập trung vào chi tiết nhỏ ở những tầng sâu.

### 6.1 Quy trình một flow step của Glow

```
x ──► ActNorm ──► Invertible1x1Conv ──► Affine Coupling ──► x_next
             │                            │
             └── log|det J| contributions ┴──
```

Sau vài flow step, Glow thực hiện `split`, đưa một phần tensor vào danh sách latent $\{z^{(1)}, z^{(2)}, \dots\}$, phần còn lại tiếp tục qua scale kế tiếp.

### 6.2 Ưu & nhược điểm thực tế

- **Ưu**: mẫu ảnh 256×256 sắc nét, latent edit trực quan (nắn nụ cười, ánh sáng).
- **Nhược**: tiêu tốn bộ nhớ (nhất là ActNorm), slow sampling hơn GAN/Diffusion hiện đại, độ sâu lớn dễ gây underflow log-det ⇒ cần mixed-precision cẩn thận.

## 7. Code thú vị: Mini RealNVP + Glow block với PyTorch

Đoạn code dưới đây mô phỏng:

- `MiniRealNVP`: 8 coupling layer cho dữ liệu 2D (ví dụ 8-Gaussians).
- `GlowBlock`: block ActNorm + invertible 1x1 conv + affine coupling dành cho ảnh nhỏ.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def glorot_linear(in_dim, out_dim):
    w = torch.empty(in_dim, out_dim)
    nn.init.xavier_uniform_(w)
    return nn.Parameter(w)

class TwoLayerNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class RealNVPCoupling(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.scale = TwoLayerNN(dim // 2, hidden, dim // 2)
        self.shift = TwoLayerNN(dim // 2, hidden, dim // 2)

    def forward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        s = self.scale(z1)
        t = self.shift(z1)
        x1 = z1
        x2 = z2 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return torch.cat([x1, x2], dim=1), log_det

    def inverse(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        s = self.scale(x1)
        t = self.shift(x1)
        z1 = x1
        z2 = (x2 - t) * torch.exp(-s)
        log_det = -s.sum(dim=1)
        return torch.cat([z1, z2], dim=1), log_det

class MiniRealNVP(nn.Module):
    def __init__(self, dim=2, num_flows=8, hidden=128):
        super().__init__()
        self.flows = nn.ModuleList([RealNVPCoupling(dim, hidden) for _ in range(num_flows)])
        self.perms = nn.ParameterList([
            nn.Parameter(torch.randperm(dim), requires_grad=False)
            for _ in range(num_flows)
        ])

    def forward(self, z):
        log_det = 0
        x = z
        for flow, perm in zip(self.flows, self.perms):
            x = x[:, perm]
            x, det = flow(x)
            log_det += det
        return x, log_det

    def inverse(self, x):
        log_det = 0
        z = x
        for flow, perm in zip(reversed(self.flows), reversed(self.perms)):
            z, det = flow.inverse(z)
            inv_perm = torch.argsort(perm)
            z = z[:, inv_perm]
            log_det += det
        return z, log_det

class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.initialized = False

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            self.bias.data.copy_(-mean)
            self.log_s.data.copy_(torch.log(1.0 / (std + 1e-6)))
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        log_det = torch.sum(self.log_s) * x.size(2) * x.size(3)
        return torch.exp(self.log_s) * (x + self.bias), log_det

    def inverse(self, y):
        x = y * torch.exp(-self.log_s) - self.bias
        log_det = -torch.sum(self.log_s) * y.size(2) * y.size(3)
        return x, log_det

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        W = torch.qr(torch.randn(num_channels, num_channels)).Q
        self.weight = nn.Parameter(W)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.weight.view(c, c, 1, 1)
        z = F.conv2d(x, weight)
        log_det = h * w * torch.logdet(self.weight)
        return z, log_det

    def inverse(self, z):
        b, c, h, w = z.shape
        weight_inv = torch.inverse(self.weight).view(c, c, 1, 1)
        x = F.conv2d(z, weight_inv)
        log_det = -h * w * torch.logdet(self.weight)
        return x, log_det

class GlowBlock(nn.Module):
    def __init__(self, channels, hidden_channels=512):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = Invertible1x1Conv(channels)
        self.coupling = RealNVPCoupling(channels * 2, hidden_channels)

    def forward(self, x):
        x, det1 = self.actnorm(x)
        x, det2 = self.invconv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).reshape(b * h * w, c)
        x, det3 = self.coupling(x)
        x = x.view(b, h * w, c).permute(0, 2, 1).reshape(b, c, h, w)
        det3 = det3.view(b, h * w).sum(dim=1)
        return x, det1 + det2 + det3

    def inverse(self, x):
        b, c, h, w = x.shape
        z = x.view(b, c, h * w).permute(0, 2, 1).reshape(b * h * w, c)
        z, det3 = self.coupling.inverse(z)
        z = z.view(b, h * w, c).permute(0, 2, 1).reshape(b, c, h, w)
        z, det2 = self.invconv.inverse(z)
        z, det1 = self.actnorm.inverse(z)
        det3 = det3.view(b, h * w).sum(dim=1)
        return z, det1 + det2 + det3
```

**Gợi ý sử dụng nhanh:**

```python
def target_distribution(n):
    # tám gaussian xếp vòng tròn
    angles = torch.linspace(0, 2 * torch.pi, 9)[:-1]
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 3.0
    ids = torch.randint(0, 8, (n,))
    noise = 0.2 * torch.randn(n, 2)
    return centers[ids] + noise

flow = MiniRealNVP()
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

for step in range(5000):
    x = target_distribution(512)
    z, log_det = flow.inverse(x)
    log_pz = -0.5 * (z**2).sum(dim=1) - torch.log(torch.tensor(2 * torch.pi))
    loss = -(log_pz + log_det).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    samples, _ = flow.forward(torch.randn(2048, 2))
```

## 8. Gợi ý thực nghiệm & các bẫy thường gặp

- **Warm-up log-scale**: clamp đầu ra của `s(z_A)` trong khoảng `[-5, 5]` để tránh underflow.
- **Permutation học được**: Glow dùng invertible conv, nhưng RealNVP cổ điển chỉ cần shuffle channel/feature — hãy kết hợp random permutation cố định để tăng mixing.
- **Gradient clipping**: log-det lớn khiến gradient exploding; clip ở mức `1.0` hoặc `5.0`.
- **Mixed precision**: nếu huấn luyện Glow FP16, hãy lưu `logdet` ở FP32 để tránh mất chính xác khi cộng.
- **Regularize latent**: thêm term $\lambda \|z\|_2^2$ nhỏ giúp sampling ổn định khi multi-scale sâu.

## 9. Kết luận & tài liệu

### Key takeaways

- RealNVP là “người nghệ nhân” với thao tác giữ-nặn, mang lại Jacobian tuyến tính và log-likelihood chính xác.
- Glow thêm ActNorm + invertible 1x1 conv + multi-scale giúp mẫu ảnh sắc nét và latent editing trực quan.
- Combo RealNVP/Glow vẫn là baseline mạnh cho các bài toán cần invertibility (nén ảnh, anomaly detection, controllable generation).

Từ đây, chúng ta tiếp tục dõi theo xưởng pha lê khi họ kết hợp các kỹ thuật mới như Rectified Flows, Flow Matching và Schrödinger Bridge (đã hé lộ ở các bài tiếp nối trong repo) để tối ưu tốc độ và chất lượng cho pipeline generative.

### Tài liệu khuyến nghị

1. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). *Density Estimation using Real NVP*. ICLR.
2. Kingma, D. P., & Dhariwal, P. (2018). *Glow: Generative Flow with Invertible 1x1 Convolutions*. NeurIPS.
3. Papamakarios, G., et al. (2021). *Normalizing Flows for Probabilistic Modeling*. JMLR.
4. Rezende, D. J., & Mohamed, S. (2015). *Variational Inference with Normalizing Flows*. ICML.
5. Ho, J., et al. (2019). *Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design*. ICML.

---

<script src="/assets/js/katex-init.js"></script>
