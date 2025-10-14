---
title: "Normalizing Flow & Continuous Normalizing Flow: Từ Lý thuyết đến Thực hành"
date: "2025-01-15"
category: "flow-based-models"
tags: ["normalizing-flows", "CNF", "neural-ode", "generative-models", "pytorch"]
excerpt: "Deep dive vào Normalizing Flows và Continuous Normalizing Flows. Từ change of variables formula, Jacobian determinant, đến Neural ODEs và implementation chi tiết bằng PyTorch."
author: "ThanhLamDev"
readingTime: 18
featured: true
---

# Normalizing Flow & Continuous Normalizing Flow: Từ Lý thuyết đến Thực hành

Normalizing Flows là một trong những frameworks elegant nhất trong generative modeling, cho phép **exact likelihood computation** và **efficient sampling**. Bài viết này sẽ dẫn bạn từ intuition cơ bản đến mathematical foundations và cuối cùng là full implementation.

## Mục lục

1. [Giới thiệu: Flow là gì?](#1-giới-thiệu-flow-là-gì)
2. [Tại sao cần Normalizing Flow?](#2-tại-sao-cần-normalizing-flow)
3. [Change of Variables Formula](#3-change-of-variables-formula)
4. [Jacobian và Computational Challenges](#4-jacobian-và-computational-challenges)
5. [Normalizing Flows Architecture](#5-normalizing-flows-architecture)
6. [Continuous Normalizing Flows (CNF)](#6-continuous-normalizing-flows-cnf)
7. [Implementation với PyTorch](#7-implementation-với-pytorch)
8. [Training và Sampling](#8-training-và-sampling)
9. [Advanced Topics](#9-advanced-topics)
10. [Kết luận](#10-kết-luận)

---

## 1. Giới thiệu: Flow là gì?

Hãy tưởng tượng bạn có một khối đất sét hình cầu đơn giản (phân phối Gaussian) và muốn nặn nó thành một hình dạng phức tạp (phân phối dữ liệu thật). Quá trình biến đổi từng bước này chính là một **"flow"**.

**Định nghĩa toán học:**

Flow là một chuỗi các phép biến đổi khả nghịch (invertible transformations):

```
z_0 → z_1 → z_2 → ... → z_K = x
```

Trong đó:
- ```z_0``` ~ ```p_0(z)``` là base distribution (thường là Gaussian)
- ```x = f_K ∘ f_{K-1} ∘ ... ∘ f_1(z_0)``` là data distribution
- Mỗi ```f_k``` phải có inverse ```f_k^{-1}```

**Ví dụ đơn giản (1D):**

```python
import torch

# Base distribution: Standard Gaussian
z = torch.randn(1000)  # z ~ N(0, 1)

# Simple flow: affine transformation
x = 2 * z + 3  # x = f(z), where f(z) = 2z + 3

# Inverse: f^{-1}(x) = (x - 3) / 2
z_recovered = (x - 3) / 2
```

## 2. Tại sao cần Normalizing Flow?

### So sánh với các mô hình khác:

| Model | Exact Likelihood | Efficient Sampling | Training Stability |
|-------|-----------------|-------------------|-------------------|
| **VAE** | ✗ (approximate) | ✓ | ✓ |
| **GAN** | ✗ | ✓ | ✗ (unstable) |
| **Normalizing Flow** | ✓ | ✓ | ✓ |
| **Diffusion** | ✓ | ✗ (slow) | ✓ |

**Key advantages của Normalizing Flow:**

1. **Exact likelihood computation**: Tính được chính xác ```p(x)``` cho mọi data point
2. **Bidirectional mapping**: Có thể chuyển đổi cả 2 chiều ```x ↔ z```
3. **Stable training**: Không có adversarial loss như GAN
4. **Interpretable latent space**: ```z``` có ý nghĩa xác suất rõ ràng

## 3. Change of Variables Formula

### 3.1. Intuition

Khi ta biến đổi một biến ngẫu nhiên, mật độ xác suất của nó cũng thay đổi. Công thức change of variables giúp ta tracking sự thay đổi này.

**Ví dụ trực quan:**

Nếu bạn kéo giãn một khối đất sét ra gấp đôi, mật độ của nó sẽ giảm đi một nửa (để bảo toàn khối lượng/xác suất tổng).

### 3.2. Công thức toán học (1D)

Cho biến đổi ```x = f(z)``` với ```f``` khả nghịch:

```
p_x(x) = p_z(f^{-1}(x)) · |df^{-1}/dx|
```

Hoặc dùng log-likelihood (dễ optimize hơn):

```
log p_x(x) = log p_z(z) + log |df^{-1}/dx|
            = log p_z(z) - log |df/dz|
```

**Ví dụ cụ thể:**

```python
import torch
import torch.distributions as D
import matplotlib.pyplot as plt

# Base distribution: z ~ N(0, 1)
p_z = D.Normal(0, 1)
z = torch.linspace(-4, 4, 1000)

# Transformation: x = 2z + 1
def f(z):
    return 2 * z + 1

def f_inv(x):
    return (x - 1) / 2

# Change of variables
x = f(z)
log_p_z = p_z.log_prob(z)
log_det_jacobian = -torch.log(torch.tensor(2.0))  # log |df/dz| = log(2)
log_p_x = log_p_z - log_det_jacobian
p_x = torch.exp(log_p_x)

# Verify: p_x should be N(1, 4)
p_x_true = D.Normal(1, 2).log_prob(x).exp()

print(f"Max difference: {(p_x - p_x_true).abs().max():.6f}")  # Should be ~0
```

### 3.3. Công thức tổng quát (Multi-dimensional)

Cho ```x = f(z)``` với ```x, z ∈ ℝ^d```:

```
log p_x(x) = log p_z(z) - log |det(∂f/∂z)|
```

Trong đó ```∂f/∂z``` là **Jacobian matrix**:

```
J = ∂f/∂z = [∂f_i/∂z_j]_{i,j=1}^d
```

**Vấn đề:** Computing determinant có complexity ```O(d³)``` - prohibitive cho high-dimensional data!

### 3.4. Code example: 2D transformation

```python
def affine_flow_2d(z, A, b):
    """
    Affine transformation: x = Az + b
    Args:
        z: (batch_size, 2) input
        A: (2, 2) transformation matrix
        b: (2,) bias vector
    Returns:
        x: transformed data
        log_det_jac: log |det(A)|
    """
    x = z @ A.T + b
    log_det_jac = torch.logdet(A)
    return x, log_det_jac

# Example usage
z = torch.randn(100, 2)  # 100 samples from N(0, I)
A = torch.tensor([[2.0, 0.5], [0.0, 1.5]])
b = torch.tensor([1.0, -0.5])

x, log_det = affine_flow_2d(z, A, b)
print(f"Log-det Jacobian: {log_det:.4f}")  # log(2.0 * 1.5) = 1.0986
```

## 4. Jacobian và Computational Challenges

### 4.1. Vấn đề

Tính ```det(J)``` cho matrix ```d × d``` cost ```O(d³)```:
- Image 64×64: ```d = 4096``` → không khả thi!
- Cần thiết kế architectures thông minh

### 4.2. Giải pháp: Structured Jacobians

**Coupling Layers** (RealNVP, Glow):

Chia ```z``` thành 2 phần ```[z_1, z_2]```:

```
x_1 = z_1
x_2 = s(z_1) ⊙ z_2 + t(z_1)
```

Trong đó:
- ```s, t``` là neural networks
- ```⊙``` là element-wise multiplication

**Jacobian có dạng:**

```
J = [I      0   ]
    [∂t/∂z₁  diag(s(z₁))]
```

**Log-determinant:**

```
log |det(J)| = sum(log |s(z_1)|)  # O(d) complexity!
```

**Implementation:**

```python
class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        
        # Scale and translate networks
        self.s_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.half_dim)
        )
        self.t_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.half_dim)
        )
    
    def forward(self, z):
        z1, z2 = z[:, :self.half_dim], z[:, self.half_dim:]
        
        s = self.s_net(z1)
        t = self.t_net(z1)
        
        x1 = z1
        x2 = torch.exp(s) * z2 + t
        
        x = torch.cat([x1, x2], dim=1)
        log_det = s.sum(dim=1)  # Sum over dimensions
        
        return x, log_det
    
    def inverse(self, x):
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
        
        s = self.s_net(x1)
        t = self.t_net(x1)
        
        z1 = x1
        z2 = (x2 - t) * torch.exp(-s)
        
        z = torch.cat([z1, z2], dim=1)
        log_det = -s.sum(dim=1)
        
        return z, log_det
```

## 5. Normalizing Flows Architecture

### 5.1. Stacking multiple flows

Single flow thường không đủ expressive. Ta stack nhiều flows:

```
x = f_K ∘ f_{K-1} ∘ ... ∘ f_1(z)
```

Log-likelihood:

```
log p_x(x) = log p_z(z) - Σ_{k=1}^K log |det(J_k)|
```

**Complete NF model:**

```python
class NormalizingFlow(nn.Module):
    def __init__(self, dim, num_flows=8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        
        # Stack coupling layers
        self.flows = nn.ModuleList([
            CouplingLayer(dim, hidden_dim) 
            for _ in range(num_flows)
        ])
        
        # Base distribution
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std', torch.ones(dim))
    
    def forward(self, z):
        """Transform z -> x, return x and log_det"""
        log_det_total = 0
        x = z
        
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def inverse(self, x):
        """Transform x -> z, return z and log_det"""
        log_det_total = 0
        z = x
        
        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z)
            log_det_total += log_det
        
        return z, log_det_total
    
    def log_prob(self, x):
        """Compute log p(x)"""
        z, log_det = self.inverse(x)
        
        # Log prob of base distribution
        log_p_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * torch.log(torch.tensor(2 * 3.14159))
        
        # Change of variables
        log_p_x = log_p_z + log_det
        
        return log_p_x
    
    def sample(self, num_samples):
        """Generate samples"""
        z = torch.randn(num_samples, self.dim)
        x, _ = self.forward(z)
        return x
```

### 5.2. Training example

```python
# Initialize model
model = NormalizingFlow(dim=2, num_flows=8, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(1000):
    # Get batch of real data
    x_real = data_loader.get_batch()  # (batch_size, dim)
    
    # Compute negative log-likelihood
    log_p_x = model.log_prob(x_real)
    loss = -log_p_x.mean()
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, NLL: {loss.item():.4f}")

# Sample from trained model
samples = model.sample(1000)
```

## 6. Continuous Normalizing Flows (CNF)

### 6.1. Motivation

Discrete flows có hạn chế:
- Phải thiết kế architectures đặc biệt (coupling layers, autoregressive)
- Số lượng parameters tăng tuyến tính với số flows

**CNF idea:** Thay vì discrete steps, dùng **continuous-time dynamics**!

### 6.2. ODE formulation

CNF define trajectory của data point qua ODE:

```
dz(t)/dt = f(z(t), t; θ)
```

Trong đó:
- ```t ∈ [0, 1]``` là "time" parameter
- ```z(0) ~ p_0``` (base distribution)
- ```z(1) = x ~ p_data```
- ```f(·, ·; θ)``` là learned **vector field**

**Intuition:** Thay vì "nhảy" rời rạc, data point "trôi" liên tục theo vector field.

### 6.3. Instantaneous change of variables

Log-density evolution theo thời gian:

```
d log p(z(t))/dt = -Tr(∂f/∂z(t))
```

**Augmented ODE** (solve cả ```z``` và ```log p``` cùng lúc):

```
d/dt [z(t)       ] = [f(z(t), t)              ]
     [log p(z(t))]   [-Tr(∂f/∂z(t))]
```

**Key insight:** Chỉ cần trace (sum of diagonal), không cần full determinant!

### 6.4. Hutchinson's trace estimator

Tính ```Tr(∂f/∂z)``` chính xác vẫn cost ```O(d²)```. Dùng **stochastic estimator**:

```
Tr(∂f/∂z) ≈ E_ε[ε^T (∂f/∂z) ε]  với ε ~ N(0, I)
```

Chỉ cần 1 sample ```ε``` và 1 vector-Jacobian product → ```O(d)```!

**Implementation:**

```python
def hutchinson_trace_estimator(f, z, num_samples=1):
    """
    Estimate tr(df/dz) using Hutchinson's estimator
    Args:
        f: function z -> f(z)
        z: input tensor (batch_size, dim)
        num_samples: number of random vectors
    Returns:
        trace estimate (batch_size,)
    """
    batch_size, dim = z.shape
    trace = 0
    
    for _ in range(num_samples):
        # Random Rademacher vector
        eps = torch.randint(0, 2, (batch_size, dim), device=z.device) * 2 - 1
        eps = eps.float()
        
        # Compute ε^T (∂f/∂z) ε using vector-Jacobian product
        z.requires_grad_(True)
        f_z = f(z)
        
        eps_jac_eps = torch.autograd.grad(
            outputs=f_z,
            inputs=z,
            grad_outputs=eps,
            create_graph=True,
            retain_graph=True
        )[0]
        
        trace += (eps * eps_jac_eps).sum(dim=1)
    
    return trace / num_samples
```

## 7. Implementation với PyTorch

### 7.1. Vector field network

```python
class TimeConditionedVectorField(nn.Module):
    """Vector field f(z, t) for CNF"""
    def __init__(self, dim, hidden_dim=64, time_embed_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Tanh(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Tanh()
        )
        
        self.net = nn.Sequential(
            nn.Linear(dim + time_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, z):
        # Note: torchdiffeq expects (t, z) order
        # t: scalar, z: (batch_size, dim)
        batch_size = z.shape[0]
        t_embed = self.time_embed(t.view(1, 1).expand(batch_size, 1))
        tz = torch.cat([z, t_embed], dim=1)
        return self.net(tz)
```

### 7.2. CNF với torchdiffeq

```python
from torchdiffeq import odeint

class CNF(nn.Module):
    def __init__(self, dim, hidden_dim=64):
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
            z_trajectory: all states along path
        """
        if t_span is None:
            t_span = torch.tensor([0., 1.])
        
        z_traj = odeint(
            self.vf,
            z0,
            t_span,
            method='dopri5',  # Adaptive step-size Runge-Kutta
            rtol=1e-5,
            atol=1e-7
        )
        
        return z_traj
    
    def sample(self, num_samples, device='cuda'):
        """Generate samples from base distribution"""
        z0 = torch.randn(num_samples, self.dim).to(device)
        z_traj = self.forward(z0)
        return z_traj[-1]  # Return final state
    
    def log_prob(self, x):
        """
        Compute log p(x) via backward integration
        Returns log probability for each sample
        """
        # Integrate backward: x (t=1) -> z0 (t=0)
        t_span = torch.tensor([1., 0.])
        z_traj = odeint(self.vf, x, t_span, method='dopri5')
        z0 = z_traj[-1]
        
        # Compute trace term (simplified - full version needs augmented ODE)
        # This is placeholder - see FFJORD for complete implementation
        log_p_z0 = -0.5 * (z0**2).sum(dim=1) - 0.5 * self.dim * np.log(2 * np.pi)
        
        return log_p_z0  # Should include trace correction
```

### 7.3. FFJORD: Full CNF implementation

```python
class FFJORD(nn.Module):
    """
    Free-Form Jacobian of Reversible Dynamics
    Complete CNF with efficient trace estimation
    """
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.vf = TimeConditionedVectorField(dim, hidden_dim)
    
    def _augmented_dynamics(self, t, state):
        """
        Compute augmented dynamics: [dz/dt, d(log p)/dt]
        state: [z, log_p] concatenated
        """
        z = state[:, :self.dim]
        
        # Compute dz/dt
        dz_dt = self.vf(t, z)
        
        # Compute trace using Hutchinson estimator
        trace = hutchinson_trace_estimator(
            lambda z_: self.vf(t, z_), 
            z, 
            num_samples=1
        )
        
        # d(log p)/dt = -trace
        dlogp_dt = -trace
        
        # Concatenate
        return torch.cat([dz_dt, dlogp_dt.unsqueeze(1)], dim=1)
    
    def forward(self, z0, t_span=None):
        if t_span is None:
            t_span = torch.tensor([0., 1.])
        
        # Initialize augmented state: [z, log_p=0]
        batch_size = z0.shape[0]
        logp_0 = torch.zeros(batch_size, 1).to(z0)
        state_0 = torch.cat([z0, logp_0], dim=1)
        
        # Integrate
        state_traj = odeint(
            self._augmented_dynamics,
            state_0,
            t_span,
            method='dopri5'
        )
        
        return state_traj
    
    def log_prob(self, x):
        """Compute log p(x) exactly"""
        # Backward integration
        t_span = torch.tensor([1., 0.])
        batch_size = x.shape[0]
        
        logp_1 = torch.zeros(batch_size, 1).to(x)
        state_1 = torch.cat([x, logp_1], dim=1)
        
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
        log_p_z0 = -0.5 * (z0**2).sum(dim=1) - 0.5 * self.dim * np.log(2 * np.pi)
        
        # Add change-of-variables correction
        log_p_x = log_p_z0 - delta_logp
        
        return log_p_x
```

## 8. Training và Sampling

### 8.1. Training CNF

```python
import numpy as np

# Initialize
model = FFJORD(dim=2, hidden_dim=128).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5000):
    # Sample batch
    x_batch = sample_data(batch_size=256)  # Your data
    x_batch = torch.tensor(x_batch).float().cuda()
    
    # Compute negative log-likelihood
    log_p_x = model.log_prob(x_batch)
    loss = -log_p_x.mean()
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | NLL: {loss.item():.4f}")
```

### 8.2. Sampling

```python
# Generate samples
with torch.no_grad():
    z0 = torch.randn(1000, 2).cuda()
    state_traj = model.forward(z0)
    samples = state_traj[-1, :, :2]  # Extract z(t=1)
    
# Visualize
import matplotlib.pyplot as plt
samples_np = samples.cpu().numpy()
plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.3)
plt.title("Generated Samples")
plt.show()
```

## 9. Advanced Topics

### 9.1. Augmented flows

Thêm auxiliary dimensions để tăng expressivity:

```
z_aug = [z, u]  với u ~ N(0, I_k)
```

### 9.2. Regularization techniques

**Kinetic energy regularization:**

```
L_reg = λ · E_t[||f(z(t), t)||²]
```

Giúp vector field smooth hơn, giảm NFE (number of function evaluations).

### 9.3. Neural ODE solvers

- **Dopri5**: Adaptive 5th-order Runge-Kutta (default)
- **Euler**: Simple, fast but less accurate
- **Adaptive Heun**: Good trade-off

**Choosing solver:**

```python
odeint(func, z0, t_span, method='dopri5', rtol=1e-5, atol=1e-7)
```

Lower tolerance → more accurate but slower.

## 10. Kết luận

**Key takeaways:**

1. **Normalizing Flows** biến đổi simple distribution thành complex data distribution qua invertible transformations
2. **Change of variables** formula tracking density changes qua Jacobian determinant
3. **Coupling layers** giúp compute determinant hiệu quả (```O(d)``` thay vì ```O(d³)```)
4. **CNF** extend idea sang continuous-time với Neural ODEs
5. **Trace estimation** (Hutchinson) giúp CNF scalable
6. **FFJORD** là state-of-the-art CNF architecture

**So sánh CNF vs Discrete NF:**

| Aspect | Discrete NF | CNF |
|--------|-------------|-----|
| Architecture constraints | ✗ (cần coupling/autoregressive) | ✓ (free-form) |
| Memory cost | Tăng tuyến tính với K flows | Constant |
| Computation | Fast forward pass | Slow (ODE solver) |
| Expressivity | Limited | Theoretically unlimited |

**Applications:**

- Density estimation
- Variational inference
- Molecular generation
- Image synthesis
- Time series modeling

**Future directions:**

- Combine CNF with diffusion models → Rectified Flows, Flow Matching
- Efficient ODE solvers for faster sampling
- Better regularization for smoother flows

---

## Tài liệu tham khảo

1. Rezende & Mohamed (2015) - "Variational Inference with Normalizing Flows"
2. Dinh et al. (2017) - "Density estimation using Real NVP"
3. Kingma & Dhariwal (2018) - "Glow: Generative Flow using Invertible 1x1 Convolutions"
4. Chen et al. (2018) - "Neural Ordinary Differential Equations" (NeurIPS Best Paper)
5. Grathwohl et al. (2019) - "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"

**Bài viết tiếp theo:**
- [Flow Matching: Từ lý thuyết đến thực hành](/posts/2025/flow-matching-theory)
- [Real NVP & Glow: Invertible Architectures](/posts/2025/realnvp-glow)
- [Rectified Flows](/posts/2025/rectified-flows)

<script src="/assets/js/katex-init.js"></script>

---

## 1. Câu chuyện về người thợ gốm: "Flow" là gì?

Hãy tưởng tượng bạn là một người thợ gốm. Nhiệm vụ của bạn là tạo ra những chiếc bình gốm có hình dạng phức tạp (ví dụ: hình ngôi sao, hình con mèo). Tuy nhiên, bạn chỉ được bắt đầu với một khối đất sét hình cầu đơn giản.

Quá trình bạn nhào nặn, kéo, đẩy, và biến đổi khối đất sét hình cầu đó thành hình ngôi sao chính là một **"flow"** (dòng chảy). Đó là một chuỗi các phép biến đổi liên tiếp.

Quan trọng hơn, nếu bạn là một người thợ giỏi, bạn có thể làm ngược lại: biến chiếc bình hình ngôi sao trở lại thành khối cầu ban đầu. Quá trình biến đổi này có thể **đảo ngược (invertible)**.

Trong AI, "flow" cũng có ý nghĩa tương tự:
> **Flow** là một chuỗi các phép biến đổi toán học giúp chuyển một phân phối xác suất đơn giản (khối đất sét hình cầu) thành một phân phối phức tạp (chiếc bình hình ngôi sao).

## 2. Bài toán của AI: Tại sao cần một "người thợ gốm" giỏi?

Trong lĩnh vực mô hình sinh, mục tiêu của chúng ta là dạy cho máy tính cách tạo ra dữ liệu mới (ví dụ: ảnh khuôn mặt, giọng nói, văn bản) giống hệt dữ liệu thật.

Các "người thợ gốm" (mô hình) trước đây có một vài vấn đề:
- **VAE (Variational Autoencoder):** Giống như một người thợ mới vào nghề. Anh ta có thể tạo ra những chiếc bình trông khá giống hình ngôi sao, nhưng chúng thường hơi "mờ" và không sắc nét. Anh ta cũng chỉ có thể ước lượng "độ khó" để tạo ra một chiếc bình chứ không tính chính xác được.
- **GAN (Generative Adversarial Network):** Giống như một nghệ sĩ tài năng nhưng tính khí thất thường. Anh ta có thể tạo ra những chiếc bình hình ngôi sao cực kỳ đẹp và sắc nét. Tuy nhiên, quá trình dạy anh ta rất khó khăn (training không ổn định). Tệ hơn, anh ta không thể cho bạn biết xác suất để tạo ra một chiếc bình cụ thể là bao nhiêu. Anh ta chỉ biết "vẽ" thôi.

Đây là lúc chúng ta cần một phương pháp tốt hơn.

## 3. Normalizing Flow: Người thợ gốm vừa khéo tay, vừa minh bạch

**Normalizing Flow (NF)** là một loại mô hình sinh giống như một người thợ gốm bậc thầy, kết hợp ưu điểm của cả hai:

1.  **Sinh mẫu chất lượng cao:** Giống như GAN, NF có thể tạo ra dữ liệu sắc nét và chân thực.
2.  **Tính toán xác suất chính xác (Exact Likelihood):** Đây là điểm ăn tiền! Không giống GAN, NF có thể cho bạn biết chính xác xác suất để một mẫu dữ liệu (một chiếc bình cụ thể) tồn tại trong phân phối mà nó đã học. Điều này cực kỳ hữu ích trong các ứng dụng khoa học cần sự đo lường chính xác.

Cái tên "Normalizing" (chuẩn hóa) đến từ việc mô hình học cách biến đổi phân phối dữ liệu phức tạp *trở về* một phân phối "chuẩn" (thường là phân phối Gaussian). Vì phép biến đổi này đảo ngược được, chúng ta cũng có thể đi theo chiều ngược lại: từ phân phối chuẩn sinh ra dữ liệu phức tạp.

## 4. Phép màu toán học: Làm sao để theo dõi sự biến đổi?

Khi người thợ gốm biến đổi khối đất, mật độ của đất sét ở các vùng khác nhau sẽ thay đổi. Vùng bị kéo giãn ra sẽ có mật độ thấp hơn, vùng bị nén lại sẽ có mật độ cao hơn. Toán học cũng cần một cách để theo dõi sự "co giãn" này của không gian xác suất.

### 4.1. Công thức Biến đổi Biến số (Change of Variables)

Hãy bắt đầu với một ví dụ 1D siêu đơn giản.
Giả sử chúng ta có một biến ngẫu nhiên $z$ tuân theo phân phối Gaussian chuẩn (hình chuông đối xứng quanh 0). Ta định nghĩa một biến mới $x$ bằng một phép biến đổi đơn giản: $x = 2z + 1$.

- **Phép biến đổi:** $f(z) = 2z + 1$
- **Phép biến đổi ngược:** $f^{-1}(x) = (x-1)/2$

Phân phối của $x$ sẽ trông như thế nào? Nó vẫn là hình chuông, nhưng đã bị "kéo giãn" ra gấp 2 lần và "dịch chuyển" sang phải 1 đơn vị. Vì nó bị kéo giãn, chiều cao của đường cong mật độ xác suất phải giảm đi một nửa để đảm bảo tổng diện tích dưới đường cong vẫn bằng 1.

Công thức tổng quát cho sự thay đổi mật độ này là:
$$
p_x(x) = p_z(f^{-1}(x)) \left| \frac{d f^{-1}}{dx} \right|
$$
Trong đó $| \frac{d f^{-1}}{dx} |$ chính là "phí co giãn" mà chúng ta phải trả. Trong ví dụ trên, nó bằng $|1/2| = 1/2$.

> **Chú thích toán học:**
> - $p_z(z)$ là hàm mật độ xác suất của phân phối Gaussian chuẩn, $p_z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$.
> - Khi thay $z = f^{-1}(x) = (x-1)/2$, ta có $p_z(f^{-1}(x)) = \frac{1}{\sqrt{2\pi}} e^{-(x-1)^2/8}$.
> - Nhân với "phí co giãn" $1/2$, ta được mật độ của $x$: $p_x(x) = \frac{1}{2\sqrt{2\pi}} e^{-(x-1)^2/8}$. Đây chính là hàm mật độ của một phân phối Gaussian mới với trung bình là 1 và phương sai là 4 (độ lệch chuẩn là 2).

### 4.2. "Phí co giãn": Vai trò của Định thức Jacobian

Trong không gian nhiều chiều (ví dụ: một bức ảnh), "phí co giãn" không còn là một con số đơn giản nữa. Nó trở thành **định thức (determinant)** của **ma trận Jacobian**.

> **Ma trận Jacobian** là một ma trận chứa tất cả các đạo hàm riêng của phép biến đổi. Nó cho biết một vùng không gian nhỏ bị co giãn, xoay, và biến dạng như thế nào.
> **Định thức của Jacobian** là một con số duy nhất cho biết thể tích của vùng không gian đó thay đổi bao nhiêu lần.

Công thức log-likelihood trong không gian nhiều chiều trở thành:
$$
\log p_x(x) = \log p_z(z) - \log \left| \det \frac{\partial f}{\partial z} \right|
$$
Chúng ta lấy logarit vì nó biến phép nhân thành phép cộng, giúp việc tính toán và tối ưu dễ dàng hơn nhiều, đặc biệt khi chúng ta ghép nhiều phép biến đổi lại với nhau.

> **Góc lập trình:** Trong PyTorch, nếu `z` là đầu vào và `x` là đầu ra của một flow, ta có thể tính log-det-Jacobian một cách hiệu quả nếu phép biến đổi được thiết kế tốt.
> ```python
> # Giả sử 'flow' là một module biến đổi
> x, log_det_jacobian = flow(z) 
> 
> # Tính log-likelihood
> # log_prob_z là log-likelihood của z dưới phân phối base (Gaussian)
> log_prob_x = log_prob_z + log_det_jacobian
> ```
> Mục tiêu của việc training là tối đa hóa `log_prob_x` này.

## 5. Thách thức thực tế: Vấn đề của dữ liệu lớn

Việc tính định thức Jacobian cho một ma trận $d \times d$ (với $d$ là số chiều dữ liệu) có độ phức tạp tính toán là $O(d^3)$. Với một bức ảnh 64x64 pixel, $d = 4096$. Con số này là không tưởng!

Đây là lúc các kiến trúc thông minh ra đời. Các mô hình như **RealNVP** hay **Glow** thiết kế các phép biến đổi (gọi là *coupling layers*) cực kỳ khéo léo để ma trận Jacobian luôn có dạng tam giác. Nhờ đó, định thức của nó chỉ đơn giản là tích các phần tử trên đường chéo. Độ phức tạp giảm từ $O(d^3)$ xuống chỉ còn $O(d)$! Đây là một bước đột phá giúp Normalizing Flow trở nên thực tế.

> **Ví dụ về Coupling Layer:**
> Ý tưởng chính là chia các chiều của vector đầu vào $z$ thành 2 phần, $z_1$ và $z_2$.
> 1.  Phần đầu tiên được giữ nguyên: $x_1 = z_1$.
> 2.  Phần thứ hai được biến đổi bằng một hàm phụ thuộc vào phần đầu tiên: $x_2 = s(z_1) \odot z_2 + t(z_1)$, trong đó $s$ (scale) và $t$ (translate) là các mạng neural nhỏ.
> Phép biến đổi này rất dễ đảo ngược và ma trận Jacobian của nó là ma trận tam giác, giúp việc tính định thức trở nên cực nhanh.

## 6. Continuous Normalizing Flow: Từ rời rạc đến liên tục

Các mô hình như RealNVP thực hiện một chuỗi các phép biến đổi *rời rạc*. Hãy tưởng tượng nó như một cuốn sách lật (flipbook), mỗi trang là một bước biến đổi.

**Continuous Normalizing Flow (CNF)** đưa ý tưởng này lên một tầm cao mới:
> Thay vì các bước nhảy rời rạc, tại sao không mô tả sự biến đổi như một dòng chảy *liên tục* và mượt mà theo thời gian?

Hãy quay lại ví dụ người thợ gốm. Thay vì xem từng động tác riêng lẻ, CNF mô tả toàn bộ quá trình như một video mượt mà. Về mặt toán học, nó mô tả "vận tốc" thay đổi của mỗi điểm trong không gian tại mỗi thời điểm, thông qua một **Phương trình vi phân thông thường (Ordinary Differential Equation - ODE)**.

Mô hình học một trường vector (vector field) $f(z, t)$ để chỉ hướng cho các điểm di chuyển. Lợi ích lớn nhất của CNF là nó mang lại sự linh hoạt tối đa cho kiến trúc mạng, vì chúng ta không còn bị ràng buộc bởi các phép biến đổi phải dễ tính Jacobian nữa. Đây chính là nền tảng cho các mô hình hiện đại hơn như **Flow Matching**.

> **Góc toán học & lập trình:**
> Phương trình vi phân được giải bằng các "ODE solver". Trong PyTorch, thư viện `torchdiffeq` rất phổ biến cho việc này.
> $$
> z(t_1) = z(t_0) + \int_{t_0}^{t_1} f(z(t), t) dt
> $$
> Việc tính toán log-density cũng được chuyển thành một ODE khác, giúp tránh hoàn toàn việc tính Jacobian:
> $$
> \frac{d \log p(z(t))}{dt} = -\text{Tr}\left(\frac{\partial f}{\partial z(t)}\right)
> $$
> Dấu vết (Trace) của Jacobian dễ tính hơn nhiều so với định thức (determinant).

## 7. Tổng kết và bước tiếp theo

- **Flow** là một chuỗi các phép biến đổi có thể đảo ngược.
- **Normalizing Flow** là một mô hình sinh mạnh mẽ, vừa tạo ra mẫu chất lượng cao, vừa tính được xác suất chính xác.
- Chìa khóa toán học là **công thức biến đổi biến số** và **định thức Jacobian** để theo dõi sự thay đổi mật độ xác suất.
- Các kiến trúc thông minh như **coupling layers** giúp NF trở nên khả thi trên thực tế.
- **CNF** là một bước tiến hóa, mô tả dòng chảy như một quá trình liên tục bằng ODE.

Hy vọng qua bài viết này, bạn đã có một cái nhìn trực quan và dễ hiểu về Normalizing Flow. Trong các bài viết tiếp theo, chúng ta sẽ đi sâu hơn vào các kiến trúc cụ thể và cách chúng hoạt động.

---

**Bài viết tiếp theo:**
- [Flow Matching: Từ lý thuyết đến thực hành](/posts/2025/flow-matching-theory)
- [Real NVP & Glow: Các kiến trúc có thể đảo ngược](/posts/2025/realnvp-glow)
- [Rectified Flows: Con đường thẳng đến đích](/posts/2025/rectified-flows)
