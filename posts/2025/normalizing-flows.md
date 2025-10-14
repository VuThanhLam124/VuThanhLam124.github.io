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
- ✅ Tạo ra tác phẩm chất lượng cao
- ✅ **Tính được chính xác xác suất** (exact likelihood)
- ✅ Có thể đi cả hai chiều: tạo mới HOẶC phân tích ngược
- ✅ Quá trình dạy ổn định

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

| Model | Exact Likelihood | Efficient Sampling | Training Stability | Bidirectional |
|-------|-----------------|-------------------|-------------------|---------------|
| **VAE** | ✗ (lower bound) | ✓ | ✓ | ✗ |
| **GAN** | ✗ | ✓ | ✗ | ✗ |
| **Normalizing Flow** | ✓ | ✓ | ✓ | ✓ |
| **Diffusion** | ✓ | ✗ (slow) | ✓ | ✗ |

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
p_x(x) = p_z(f^{-1}(x)) \left| \frac{df^{-1}}{dx} \right|
$$

Hoặc dùng log (dễ tính toán hơn):

$$
\log p_x(x) = \log p_z(z) + \log \left| \frac{df^{-1}}{dx} \right|
$$

**"Phí co giãn"** $\left| \frac{df^{-1}}{dx} \right|$ điều chỉnh mật độ để bảo toàn tổng xác suất.

### Ví dụ cụ thể

Cho $z \sim \mathcal{N}(0, 1)$ và $x = 2z + 1$:

**Bước 1:** Hàm ngược
$$
f^{-1}(x) = \frac{x - 1}{2}
$$

**Bước 2:** Đạo hàm
$$
\frac{df^{-1}}{dx} = \frac{1}{2}
$$

**Bước 3:** Mật độ của $x$
$$
p_x(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-1)^2}{8}\right) \cdot \frac{1}{2}
$$

Kết quả: $x \sim \mathcal{N}(1, 4)$ (trung bình 1, phương sai 4)

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
log_det_jacobian = -torch.log(torch.tensor(2.0))  # log |df/dz| = log(2)
log_p_x = log_p_z - log_det_jacobian

# Verify: should match N(1, 2) distribution
p_x_true = D.Normal(1, 2)
log_p_x_true = p_x_true.log_prob(x)

print(f"Max log-prob difference: {(log_p_x - log_p_x_true).abs().max():.6f}")
# Output: ~0.000000 (perfect match!)
```

---

## 4. Jacobian: "Phí co giãn" của không gian

### Từ 1D sang nhiều chiều

Khi dữ liệu có nhiều chiều (ví dụ: ảnh 64×64 = 4096 dimensions), "phí co giãn" không còn là một số đơn giản. Nó trở thành **định thức (determinant)** của **ma trận Jacobian**.

### Ma trận Jacobian

Cho $f: \mathbb{R}^d \to \mathbb{R}^d$, Jacobian là ma trận đạo hàm riêng:

$$
J = \frac{\partial f}{\partial z} = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_d}{\partial z_1} & \cdots & \frac{\partial f_d}{\partial z_d}
\end{bmatrix}
$$

**Ý nghĩa hình học:** Jacobian mô tả cách một vùng không gian nhỏ bị co giãn, xoay, biến dạng.

**Định thức:** $\det(J)$ cho biết **thể tích** của vùng đó thay đổi bao nhiêu lần.

### Change of Variables (nhiều chiều)

$$
\log p_x(x) = \log p_z(z) - \log \left| \det \frac{\partial f}{\partial z} \right|
$$

**Vấn đề lớn:** Tính $\det(J)$ cho ma trận $d \times d$ có độ phức tạp $O(d^3)$!

Với ảnh 64×64:
- $d = 4096$
- $O(d^3) = O(68{,}719{,}476{,}736)$ operations
- **Không khả thi!**

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

Để tránh tính toán $O(d^3)$, chúng ta thiết kế transformation sao cho Jacobian có **cấu trúc đặc biệt** → tính determinant chỉ mất $O(d)$!

### Coupling Layer (RealNVP)

**Ý tưởng:** Chia vector $z$ thành 2 nửa:

$$
\begin{aligned}
x_{1:d/2} &= z_{1:d/2} \quad \text{(giữ nguyên)} \\
x_{d/2+1:d} &= z_{d/2+1:d} \odot \exp(s(z_{1:d/2})) + t(z_{1:d/2})
\end{aligned}
$$

Trong đó:
- $s(\cdot)$: **scale network** (neural net)
- $t(\cdot)$: **translation network** (neural net)  
- $\odot$: element-wise multiplication

**Jacobian có dạng tam giác:**

$$
J = \begin{bmatrix}
I_{d/2} & 0 \\
\frac{\partial x_{d/2+1:d}}{\partial z_{1:d/2}} & \text{diag}(\exp(s(z_{1:d/2})))
\end{bmatrix}
$$

**Determinant cực kỳ đơn giản:**

$$
\log |\det(J)| = \sum_{i=d/2+1}^{d} s_i(z_{1:d/2})
$$

Chỉ cần **cộng các phần tử** → $O(d)$ thay vì $O(d^3)$!

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
# Initialize model
model = NormalizingFlow(dim=2, num_flows=6, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
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

Quay lại ví dụ người thợ gốm. Thay vì xem từng động tác rời rạc (như flipbook), **CNF** mô tả toàn bộ quá trình như một video mượt mà, liên tục.

**Discrete NF:**
$$
z_0 \to z_1 \to z_2 \to \cdots \to z_K = x
$$

**Continuous NF:**
$$
\frac{dz(t)}{dt} = f(z(t), t; \theta) \quad \text{với } t \in [0, 1]
$$

Trong đó:
- $z(0) \sim p_0$ (base distribution)
- $z(1) = x \sim p_{\text{data}}$
- $f(\cdot, \cdot; \theta)$ là **vector field** học được

### Ý nghĩa

- $f(z, t)$ cho biết "vận tốc" của điểm $z$ tại thời điểm $t$
- Tưởng tượng dòng nước chảy: mỗi giọt nước (data point) di chuyển theo hướng được chỉ định bởi vector field

### Instantaneous Change of Variables

Log-density evolution theo thời gian:

$$
\frac{d \log p_t(z(t))}{dt} = -\text{Tr}\left(\frac{\partial f}{\partial z(t)}\right)
$$

**Ưu điểm lớn:** Chỉ cần **trace** (sum of diagonal), không cần full determinant!

### Augmented ODE

Để tính log-likelihood, ta solve ODE augmented:

$$
\frac{d}{dt} \begin{bmatrix} z(t) \\ \log p_t(z(t)) \end{bmatrix} = \begin{bmatrix} f(z(t), t) \\ -\text{Tr}\left(\frac{\partial f}{\partial z(t)}\right) \end{bmatrix}
$$

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

### FFJORD Architecture

**Free-Form Jacobian of Reversible Dynamics** - state-of-the-art CNF với exact likelihood.

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
   - Tracking density changes qua Jacobian determinant
   - $\log p_x(x) = \log p_z(z) - \log |\det(J)|$

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
|--------|-------------|-----|
| Architecture | Constrained (coupling/autoregressive) | Free-form |
| Memory | $O(K \cdot d)$ (K layers) | $O(d)$ (constant) |
| Computation | Fast forward pass | Slow (ODE solver) |
| Expressivity | Limited by architecture | Theoretically unlimited |

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

---

**Bài viết tiếp theo:**
- [Flow Matching: Từ lý thuyết đến thực hành](/posts/2025/flow-matching-theory)
- [Real NVP & Glow: Deep Dive](/posts/2025/realnvp-glow)
- [Rectified Flows: Straight Paths to Generation](/posts/2025/rectified-flows)

<script src="/assets/js/katex-init.js"></script>
