---
title: "RealNVP & Glow: Nghệ Thuật Biến Đổi Có Thể Đảo Ngược"
date: "2025-01-15"
category: "flow-based-models"
tags: ["realnvp", "glow", "normalizing-flows", "invertible-networks", "pytorch"]
excerpt: "Trước khi học Flow Matching, người thợ gốm phải hiểu RealNVP và Glow - nghệ thuật tạo hình khả nghịch. Mỗi bước biến đổi đất sét đều có thể quay ngược lại, và tính toán log-likelihood chính xác."
author: "ThanhLamDev"
readingTime: 20
featured: false
---

# RealNVP & Glow: Nghệ Thuật Biến Đổi Có Thể Đảo Ngược

**Người Thợ Gốm Bắt Đầu Hành Trình**

Trong series về [Normalizing Flows](/posts/2025/normalizing-flows), chúng ta đã học lý thuyết tổng quát. Bây giờ, người thợ gốm của chúng ta bắt đầu học hai kỹ thuật cụ thể: **RealNVP** và **Glow** - những kiến trúc khả nghịch giúp anh biến khối đất sét thành tác phẩm nghệ thuật, đồng thời vẫn có thể quay ngược lại.

## Mục lục

1. [Câu chuyện: Xưởng gốm khả nghịch](#1-câu-chuyện-xưởng-gốm-khả-nghịch)
2. [Áp lực từ triển lãm](#2-áp-lực-từ-triển-lãm)
3. [Từ trực giác đến RealNVP](#3-từ-trực-giác-đến-realnvp)
4. [Kiến trúc RealNVP từng lớp](#4-kiến-trúc-realnvp-từng-lớp)
5. [Ví dụ toán học: Coupling 2D](#5-ví-dụ-toán-học-coupling-2d)
6. [Glow: Kỹ thuật nâng cao](#6-glow-kỹ-thuật-nâng-cao)
7. [Implementation PyTorch](#7-implementation-pytorch)
8. [Kinh nghiệm thực nghiệm](#8-kinh-nghiệm-thực-nghiệm)
9. [Kết luận](#9-kết-luận)

---

## 1. Câu chuyện: Xưởng gốm khả nghịch

### Ngày đầu tiên tại xưởng

Người thợ gốm mới vào nghề nhận được nhiệm vụ: **Biến khối đất sét nguyên bản thành tác phẩm nghệ thuật, nhưng mọi bước đều phải đảo ngược được**.

"Tại sao phải đảo ngược?" Anh hỏi sư phụ.

"Vì khách hàng thường đổi ý!" Sư phụ cười. "Họ nhìn tác phẩm gần hoàn thành rồi nói: 'À không, tôi muốn cao hơn một chút'. Nếu anh không thể quay lại bước trước, phải làm lại từ đầu - tốn thời gian và nguyên liệu!"

Anh hiểu ra: **Tính khả nghịch (invertibility)** không chỉ là đặc tính toán học, mà còn là yêu cầu thực tế.

### Khối đất sét = Gaussian noise

Mỗi sáng, anh bắt đầu với **khối đất sét chuẩn** - tương đương với phân phối Gaussian $z \sim \mathcal{N}(0, I)$. Nhiệm vụ là biến đổi nó thành:
- Bình hoa (data distribution 1)
- Tượng rồng (data distribution 2)
- Chén uống trà (data distribution 3)

Nhưng **mọi thao tác phải có công thức đảo ngược**:

```python
# Forward: Đất sét → Tác phẩm
x = f(z, theta)

# Inverse: Tác phẩm → Đất sết
z = f_inverse(x, theta)

# Requirement: f_inverse(f(z)) == z
```

### Vấn đề: Tính log-likelihood

Sư phụ hỏi: "Làm sao biết tác phẩm này 'giống' dữ liệu thật đến đâu?"

Anh học được công thức **change of variables**:

$$
\log p_X(x) = \log p_Z(z) - \log\left|\det\frac{\partial f}{\partial z}\right|
$$

**Vấn đề:** Tính $\det(\text{Jacobian})$ cho ảnh $256 \times 256 \times 3$ (196,608 chiều) là **KHÔNG THỂ** với ma trận đầy đủ!

→ Cần kiến trúc thông minh: **RealNVP**

## 2. Áp lực từ triển lãm

### Yêu cầu của triển lãm gốm

Triển lãm gốm nghệ thuật đặt ra 3 yêu cầu:

1. **Nhanh:** Tạo 100 tác phẩm trong 1 giờ
2. **Đa dạng:** Mỗi tác phẩm khác nhau (từ cùng khối đất)
3. **Đảo ngược:** Có thể điều chỉnh lại nếu khách không hài lòng

RealNVP được thiết kế để đáp ứng cả 3:

| Yêu cầu | RealNVP solution |
|---------|------------------|
| Nhanh | Jacobian tam giác → $O(D)$ thay vì $O(D^3)$ |
| Đa dạng | Sampling từ Gaussian → diverse outputs |
| Đảo ngược | Coupling layer có công thức inverse rõ ràng |

### So sánh với các approach khác

| Approach | Invertible? | Fast? | Exact likelihood? |
|----------|-------------|-------|-------------------|
| GAN | No | Yes (strong) | No |
| VAE | No | Yes | No (lower bound) |
| Diffusion | Yes | No | No |
| RealNVP/Glow | Yes (strong) | Yes | Yes (strong) |

## 3. Từ trực giác đến RealNVP

### Ý tưởng "Giữ nửa - Biến đổi nửa"

Người thợ gốm khám phá một kỹ thuật thông minh:

**"Mỗi lần, tôi GIỮ NGUYÊN một nửa khối đất, dùng nửa đó để quyết định cách nặn nửa còn lại!"**

Giả sử khối đất có 2 phần: $z = [z_A, z_B]$

**Forward transform:**

```
1. Giữ nguyên z_A:
   x_A = z_A

2. Biến đổi z_B dựa trên z_A:
   x_B = z_B * exp(scale(z_A)) + shift(z_A)
```

**Inverse transform (CỰC KỲ ĐƠN GIẢN):**

```
1. Lấy lại z_A:
   z_A = x_A

2. Đảo ngược phép biến đổi z_B:
   z_B = (x_B - shift(z_A)) * exp(-scale(z_A))
```

**Điểm mấu chốt:** Không cần giải phương trình phức tạp!

### Công thức toán học

Cho mask $m \in \{0,1\}^D$ (ví dụ `[1,1,0,0]` → giữ 2 chiều đầu):

$$
\begin{aligned}
z_A &= m \odot z, \quad z_B = (1 - m) \odot z \\
s &= S_\theta(z_A), \quad t = T_\theta(z_A)
\end{aligned}
$$

**Forward:**

$$
\begin{aligned}
x_A &= z_A \\
x_B &= z_B \odot \exp(s) + t
\end{aligned}
$$

**Inverse:**

$$
\begin{aligned}
z_A &= x_A \\
z_B &= (x_B - t) \odot \exp(-s)
\end{aligned}
$$

**Log-det Jacobian (CỰC KỲ ĐƠN GIẢN):**

$$
\log|\det J| = \sum_{i: m_i = 0} s_i
$$

Chỉ cần cộng các phần tử $s$ ở nửa được biến đổi!

## 4. Kiến trúc RealNVP từng lớp

### 4.1. Coupling Layer

```python
class CouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden_dim=128):
        super().__init__()
        self.mask = mask
        
        # Networks to compute scale and shift
        dim_A = mask.sum()
        dim_B = dim - dim_A
        
        self.scale_net = nn.Sequential(
            nn.Linear(dim_A, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_B)
        )
        
        self.shift_net = nn.Sequential(
            nn.Linear(dim_A, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_B)
        )
    
    def forward(self, z):
        z_A = z[:, self.mask == 1]
        z_B = z[:, self.mask == 0]
        
        s = self.scale_net(z_A)
        t = self.shift_net(z_A)
        
        x_B = z_B * torch.exp(s) + t
        
        x = z.clone()
        x[:, self.mask == 0] = x_B
        
        log_det = s.sum(dim=1)
        return x, log_det
    
    def inverse(self, x):
        x_A = x[:, self.mask == 1]
        x_B = x[:, self.mask == 0]
        
        s = self.scale_net(x_A)
        t = self.shift_net(x_A)
        
        z_B = (x_B - t) * torch.exp(-s)
        
        z = x.clone()
        z[:, self.mask == 0] = z_B
        
        return z
```

### 4.2. Xen kẽ masks

Để tất cả chiều đều được biến đổi, người thợ gốm **xen kẽ các coupling layers với masks khác nhau**:

```
Layer 1: mask = [1,1,0,0] → Giữ 2 đầu, biến 2 cuối
Layer 2: mask = [0,0,1,1] → Giữ 2 cuối, biến 2 đầu
Layer 3: mask = [1,0,1,0] → Checkerboard pattern
...
```

**Kết quả:** Sau 3-4 layers, tất cả chiều đều được biến đổi phụ thuộc vào nhau!

### 4.3. Multi-scale Architecture

Với ảnh lớn, RealNVP dùng **squeeze** và **split**:

```
Input: (batch, 3, 64, 64)
  ↓ Squeeze
(batch, 12, 32, 32)  # 4x channels, 0.5x resolution
  ↓ 4 coupling layers
  ↓ Split: Half to latent
Latent z1: (batch, 6, 32, 32)
Continue: (batch, 6, 32, 32)
  ↓ Squeeze
(batch, 24, 16, 16)
  ↓ 4 coupling layers
  ↓ Split: Half to latent
Latent z2: (batch, 12, 16, 16)
...
```

**Lợi ích:** Model tập trung vào chi tiết ở các scale khác nhau.

## 5. Ví dụ toán học: Coupling 2D

### Setup đơn giản

Cho $z = [z_1, z_2]^T \sim \mathcal{N}(0, I)$ và mask $m = [1, 0]$.

**Networks:**

$$
s(z_1) = 0.8 z_1, \quad t(z_1) = 0.5 z_1
$$

### Forward transform

$$
\begin{aligned}
x_1 &= z_1 \\
x_2 &= z_2 \cdot \exp(0.8 z_1) + 0.5 z_1
\end{aligned}
$$

**Ví dụ số:**

```python
z = [1.0, 0.5]  # Gaussian sample

# Forward
x1 = 1.0
x2 = 0.5 * exp(0.8 * 1.0) + 0.5 * 1.0
   = 0.5 * 2.226 + 0.5
   = 1.113 + 0.5
   = 1.613

x = [1.0, 1.613]
```

### Inverse transform

$$
\begin{aligned}
z_1 &= x_1 = 1.0 \\
z_2 &= (x_2 - 0.5 z_1) \cdot \exp(-0.8 z_1) \\
    &= (1.613 - 0.5) \cdot \exp(-0.8) \\
    &= 1.113 \cdot 0.449 \\
    &= 0.5 \quad \checkmark
\end{aligned}
$$

### Log-det Jacobian

$$
\log|\det J| = 0.8 z_1 = 0.8 \times 1.0 = 0.8
$$

### Log-likelihood

$$
\begin{aligned}
\log p_X(x) &= \log p_Z(z) - \log|\det J| \\
            &= -\frac{1}{2}(z_1^2 + z_2^2) - \log(2\pi) - 0.8 \\
            &= -\frac{1}{2}(1.0 + 0.25) - 1.838 - 0.8 \\
            &= -0.625 - 1.838 - 0.8 \\
            &= -3.263
\end{aligned}
$$

**Code PyTorch:**

```python
import torch

x = torch.tensor([[1.0, 1.613]])
z1 = x[:, 0]
z2 = (x[:, 1] - 0.5 * z1) * torch.exp(-0.8 * z1)

# Log p_Z(z)
log_pz = -0.5 * (z1**2 + z2**2) - torch.log(torch.tensor(2 * torch.pi))

# Log |det J|
log_det = 0.8 * z1

# Log p_X(x)
log_px = log_pz - log_det
print(float(log_px))  # ≈ -3.263
```

## 6. Glow: Kỹ thuật nâng cao

### Ngày thứ 5: Người thợ gốm học Glow

Sau khi thành thạo RealNVP, người thợ gốm học ba kỹ thuật mới từ **Glow** (Kingma & Dhariwal, 2018):

### 6.1. ActNorm - "Chuẩn hóa tự động"

**Vấn đề:** Batch Normalization không invertible!

**Giải pháp:** ActNorm = Affine transformation học được:

$$
y = s \odot (x - b)
$$

- $s, b$ khởi tạo từ mini-batch đầu tiên → zero mean, unit variance
- Sau đó học như tham số thông thường

**Log-det (với ảnh $H \times W \times C$):**

$$
\log|\det J| = HW \sum_{c=1}^C \log|s_c|
$$

**Code:**

```python
class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.initialized = False
    
    def forward(self, x):
        # x: (B, C, H, W)
        if not self.initialized:
            # Initialize from first batch
            with torch.no_grad():
                mean = x.mean(dim=[0, 2, 3])
                std = x.std(dim=[0, 2, 3])
                self.bias.copy_(-mean)
                self.scale.copy_(1.0 / (std + 1e-6))
            self.initialized = True
        
        B, C, H, W = x.shape
        y = self.scale.view(1, C, 1, 1) * (x + self.bias.view(1, C, 1, 1))
        
        log_det = H * W * torch.sum(torch.log(torch.abs(self.scale)))
        return y, log_det
    
    def inverse(self, y):
        x = y / self.scale.view(1, -1, 1, 1) - self.bias.view(1, -1, 1, 1)
        return x
```

### 6.2. Invertible 1x1 Convolution

**Vấn đề:** Permutation cố định giới hạn expressiveness.

**Giải pháp:** Học ma trận khả nghịch $W \in \mathbb{R}^{C \times C}$ cho 1x1 conv:

$$
y = W x
$$

**Log-det:**

$$
\log|\det J| = HW \cdot \log|\det W|
$$

**Tối ưu:** Dùng PLU decomposition:

$$
W = P L U
$$

- $P$: Permutation (cố định)
- $L$: Lower triangular
- $U$: Upper triangular

→ $\det W = \prod_{i} u_{ii}$

**Code:**

```python
class InvConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # Initialize as orthogonal matrix
        W = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.W = nn.Parameter(W)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Apply 1x1 conv
        y = F.conv2d(x, self.W.unsqueeze(-1).unsqueeze(-1))
        
        # Log-det
        log_det = H * W * torch.slogdet(self.W)[1]
        return y, log_det
    
    def inverse(self, y):
        W_inv = torch.inverse(self.W)
        x = F.conv2d(y, W_inv.unsqueeze(-1).unsqueeze(-1))
        return x
```

### 6.3. Glow Block - Kết hợp toàn bộ

```python
class GlowBlock(nn.Module):
    def __init__(self, num_channels, hidden_dim=512):
        super().__init__()
        self.actnorm = ActNorm(num_channels)
        self.invconv = InvConv1x1(num_channels)
        self.coupling = AffineCoupling(num_channels, hidden_dim)
    
    def forward(self, x):
        y, logdet1 = self.actnorm(x)
        y, logdet2 = self.invconv(y)
        y, logdet3 = self.coupling(y)
        
        total_logdet = logdet1 + logdet2 + logdet3
        return y, total_logdet
    
    def inverse(self, y):
        x = self.coupling.inverse(y)
        x = self.invconv.inverse(x)
        x = self.actnorm.inverse(x)
        return x
```

### Glow vs RealNVP

| Aspect | RealNVP | Glow |
|--------|---------|------|
| Normalization | Batch Norm (không invertible) | ActNorm (invertible) |
| Mixing | Fixed permutation | Learned 1x1 conv |
| Coupling | Affine | Affine (same) |
| Image quality | Good | Better |
| Training | Easier | Harder (more params) |

## 7. Implementation PyTorch

### 7.1. Complete RealNVP for 2D data

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RealNVP(nn.Module):
    def __init__(self, dim, num_layers=8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        
        # Create alternating masks
        masks = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            mask[i % dim::2] = 1
            masks.append(mask)
        
        self.masks = nn.ParameterList([
            nn.Parameter(m, requires_grad=False) for m in masks
        ])
        
        # Create scale and shift networks
        self.scale_nets = nn.ModuleList()
        self.shift_nets = nn.ModuleList()
        
        for mask in masks:
            dim_A = int(mask.sum())
            dim_B = dim - dim_A
            
            self.scale_nets.append(MLP(dim_A, hidden_dim, dim_B))
            self.shift_nets.append(MLP(dim_A, hidden_dim, dim_B))
    
    def forward(self, z):
        """z -> x, return x and log_det"""
        x = z
        sum_log_det = 0
        
        for mask, scale_net, shift_net in zip(
            self.masks, self.scale_nets, self.shift_nets
        ):
            x_A = x * mask
            x_B = x * (1 - mask)
            
            s = scale_net(x_A)
            t = shift_net(x_A)
            
            # Transform B part
            x_B_new = x_B * torch.exp(s) + t
            x = x_A + x_B_new
            
            sum_log_det = sum_log_det + s.sum(dim=1)
        
        return x, sum_log_det
    
    def inverse(self, x):
        """x -> z"""
        z = x
        
        for mask, scale_net, shift_net in reversed(list(zip(
            self.masks, self.scale_nets, self.shift_nets
        ))):
            z_A = z * mask
            z_B = z * (1 - mask)
            
            s = scale_net(z_A)
            t = shift_net(z_A)
            
            # Inverse transform
            z_B_new = (z_B - t) * torch.exp(-s)
            z = z_A + z_B_new
        
        return z
    
    def log_prob(self, x):
        """Compute log p(x)"""
        z, log_det = self.inverse(x), 0  # Need to track log_det in inverse too
        
        # For simplicity, compute log_det via forward
        z_temp = self.inverse(x)
        _, log_det = self.forward(z_temp)
        
        # Gaussian log prob
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim * torch.log(
            torch.tensor(2 * torch.pi)
        )
        
        log_px = log_pz - log_det
        return log_px
```

### 7.2. Training loop

```python
def train_realnvp(model, data_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        
        for x_batch in data_loader:
            # Compute negative log-likelihood
            log_px = model.log_prob(x_batch)
            loss = -log_px.mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch {epoch+1}/{epochs} | NLL: {avg_loss:.4f}")
    
    return model

# Usage
model = RealNVP(dim=2, num_layers=8)
# Assuming you have data_loader
# model = train_realnvp(model, data_loader, epochs=100)

# Sampling
z = torch.randn(100, 2)
with torch.no_grad():
    x_samples, _ = model(z)
```

## 8. Kinh nghiệm thực nghiệm

### 8.1. Hyperparameter tuning

Người thợ gốm học được:

1. **Số layers:** 
   - 2D data: 6-8 layers
   - Images: 32-48 layers (với multi-scale)

2. **Hidden dim:**
   - Nhỏ (64-128) cho 2D
   - Lớn (512-1024) cho images

3. **Coupling type:**
   - Additive: $x_B = z_B + t$ (đơn giản nhưng yếu)
   - Affine: $x_B = z_B \cdot \exp(s) + t$ (standard)

4. **Regularization:**
   - Weight decay: 1e-5
   - Gradient clipping: 5.0

### 8.2. Common issues

**Issue 1: Exploding log_det**

```python
# Bad: log_det becomes too large
s = scale_net(x_A)  # No constraint

# Good: Bound scale
s = torch.tanh(scale_net(x_A)) * 3  # s ∈ [-3, 3]
```

**Issue 2: Mode collapse**

- Add noise to data: `x_noisy = x + 0.01 * torch.randn_like(x)`
- Use dequantization for images

**Issue 3: Slow convergence**

- Use ActNorm instead of BatchNorm
- Warm-up learning rate
- Use multi-scale architecture

### 8.3. Evaluation metrics

```python
# 1. Negative log-likelihood (lower is better)
nll = -model.log_prob(test_data).mean()

# 2. Bits per dimension
bpd = nll / (np.log(2) * np.prod(data_shape))

# 3. Sample quality (visual inspection)
z = torch.randn(64, dim)
samples = model(z)[0]
plot_samples(samples)
```

## 9. Kết luận

### Bài học của người thợ gốm

Sau vài ngày học RealNVP và Glow, người thợ gốm ghi vào sổ tay:

> **RealNVP & Glow: Nghệ thuật khả nghịch**
>
> 1. **Coupling layers:** Giữ nửa - biến đổi nửa → Jacobian đơn giản
> 2. **Invertibility:** Mọi bước đều có công thức đảo ngược rõ ràng
> 3. **Exact likelihood:** Tính $\log p(x)$ chính xác qua change of variables
> 4. **Glow improvements:** ActNorm + 1x1 conv + multi-scale

### So sánh tổng hợp

| Method | Pros | Cons | Use case |
|--------|------|------|----------|
| **RealNVP** | Simple, exact likelihood | Many layers needed | Density estimation |
| **Glow** | Better quality, flexible | More complex | High-res generation |
| **vs GAN** | Exact likelihood, stable training | Slower, less sharp | When likelihood matters |
| **vs VAE** | No approximation | More constrained arch | Exact generation |
| **vs Diffusion** | Exact, invertible | Less flexible | Fast exact sampling |

### Hướng tiếp theo

"RealNVP và Glow đã dạy tôi về **khả nghịch**," anh nghĩ. "Nhưng vẫn còn một vấn đề: Training bằng likelihood rất **chậm và khó**. Liệu có cách nào học nhanh hơn không?"

→ Dẫn đến **FFJORD** và **Flow Matching** (các bài tiếp theo)

---

## Tài liệu tham khảo

1. **Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017)** - "Density estimation using Real NVP" _(ICLR 2017)_

2. **Kingma, D. P., & Dhariwal, P. (2018)** - "Glow: Generative Flow using Invertible 1x1 Convolutions" _(NeurIPS 2018)_

3. **Rezende, D. J., & Mohamed, S. (2015)** - "Variational Inference with Normalizing Flows" _(ICML 2015)_

4. **Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021)** - "Normalizing Flows for Probabilistic Modeling and Inference" _(JMLR)_

---

**Series:** [Generative AI Overview](/posts/2025/generative-ai-overview)

**Bài tiếp theo:** [FFJORD: Continuous Flows với Neural ODE](/posts/2025/ffjord-continuous-flows)

<script src="/assets/js/katex-init.js"></script>
