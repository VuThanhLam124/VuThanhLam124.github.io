---
title: "FFJORD & CNF: Dòng Chảy Liên Tục với Neural ODE"
date: "2025-01-16"
category: "flow-based-models"
tags: ["ffjord", "continuous-normalizing-flows", "neural-ode", "cnf", "pytorch"]
excerpt: "Sau RealNVP, người thợ gốm khám phá FFJORD - kỹ thuật điều khiển dòng chảy đất sét một cách liên tục bằng Neural ODE, tính log-likelihood chính xác qua Hutchinson trace estimator."
author: "ThanhLamDev"
readingTime: 25
featured: false
---

# FFJORD & CNF: Dòng Chảy Liên Tục với Neural ODE

**Người Thợ Gốm Và Dòng Chảy Không Ngắt Quãng**

Sau khi học [RealNVP & Glow](/posts/2025/realnvp-glow), người thợ gốm nhận ra một giới hạn: **Mọi biến đổi đều rời rạc** - từng layer coupling một. Liệu có cách nào điều khiển dòng chảy đất sét một cách **liên tục**, như dòng sông chảy từ nguồn đến cửa biển?

FFJORD (Free-Form Continuous Dynamics) và CNF (Continuous Normalizing Flows) ra đời để giải quyết vấn đề này.

## Mục lục

1. [Câu chuyện: Dòng sông đất sét](#1-câu-chuyện-dòng-sông-đất-sét)
2. [FFJORD trong hệ sinh thái](#2-ffjord-trong-hệ-sinh-thái)
3. [Toán học CNF](#3-toán-học-cnf)
4. [FFJORD: Kỹ thuật tính toán](#4-ffjord-kỹ-thuật-tính-toán)
5. [Hàm mục tiêu](#5-hàm-mục-tiêu)
6. [Implementation PyTorch](#6-implementation-pytorch)
7. [So sánh các phương pháp](#7-so-sánh-các-phương-pháp)
8. [Kinh nghiệm thực nghiệm](#8-kinh-nghiệm-thực-nghiệm)
9. [Kết luận](#9-kết-luận)

---

## 1. Câu chuyện: Dòng sông đất sét

### Ngày thứ 6: Khách hàng yêu cầu đặc biệt

Sáng ngày thứ 6, khách hàng mang theo một video:

**"Anh xem này - tôi muốn quá trình tạo hình diễn ra MƯỢT MÀ như vậy, không giật cục!"**

Video cho thấy một khối đất sét chảy liên tục, từ từ biến đổi thành tác phẩm cuối cùng - như dòng sông.

Người thợ gốm nhìn lại quy trình RealNVP:

```
Step 1: Coupling layer 1 → Nhảy cục!
Step 2: Coupling layer 2 → Nhảy cục!
Step 3: Coupling layer 3 → Nhảy cục!
...
Step 8: Coupling layer 8 → Done
```

"Mỗi bước là một 'nhảy' rời rạc," anh nhận ra. "Nếu muốn mượt mà, cần **dòng chảy liên tục**!"

### Ý tưởng Neural ODE

Sư phụ giới thiệu anh với **Neural ODE** (Chen et al., 2018):

> "Thay vì biến đổi theo từng bước rời rạc, hãy định nghĩa một **phương trình vi phân** mô tả vận tốc biến đổi!"

$$
\frac{dz(t)}{dt} = f_\theta(z(t), t)
$$

**Giải thích:**
- $z(t)$: Trạng thái đất sét tại thời điểm $t$
- $f_\theta$: Mạng neural học vận tốc biến đổi
- Giải ODE từ $t=0$ đến $t=1$ → Quá trình liên tục!

"Giống như dòng sông!" Anh hào hứng. "Mỗi hạt nước (mỗi pixel đất sét) di chuyển theo phương trình, tạo thành dòng chảy mượt mà!"

### Vấn đề: Log-likelihood

"Nhưng làm sao tính log-likelihood?" Anh hỏi.

Sư phụ giải thích: **Instantaneous change of variables**:

$$
\frac{d}{dt} \log p(z(t)) = -\text{Tr}\left(\frac{\partial f_\theta}{\partial z}\right)
$$

"Tốc độ thay đổi log-density = Âm trace của Jacobian!"

**Vấn đề lớn:** Tính trace Jacobian cho ảnh $256 \times 256$ (65536 chiều) rất đắt: $O(D^2)$!

→ **Hutchinson trace estimator** xuất hiện.

## 2. FFJORD trong hệ sinh thái

### So sánh nhanh

| Method | Type | Likelihood | Flexibility | Speed |
|--------|------|------------|-------------|-------|
| **RealNVP** | Discrete | Exact | Low (coupling) | Fast |
| **Glow** | Discrete | Exact | Medium | Fast |
| **Diffusion** | Continuous SDE | Approximate | High | Slow |
| **Flow Matching** | Continuous ODE | No | High | Fast |
| **FFJORD/CNF** | Continuous ODE | **Exact** | **High** | Medium |

**Điểm mạnh FFJORD:**
- ✅ Exact likelihood (như RealNVP)
- ✅ Flexible architecture (như Diffusion)
- ✅ Continuous dynamics (mượt mà)

**Điểm yếu:**
- ❌ Slower than RealNVP (cần giải ODE)
- ❌ Harder to train (Hutchinson estimator có variance)

### Vị trí trong timeline

Người thợ gốm nhìn lại lịch sử:

```
Day 1-5: RealNVP & Glow
  → Discrete flows, exact likelihood
  → Vấn đề: Rời rạc, architecture constrained

Day 6-7: FFJORD & CNF
  → Continuous flows, exact likelihood
  → Giải quyết: Mượt mà, flexible
  → Nhưng: Chậm hơn, khó train hơn

Day 8+: Flow Matching, Rectified Flow
  → Continuous flows, NO exact likelihood
  → Trade-off: Nhanh nhưng mất likelihood
```

## 3. Toán học CNF

### 3.1. Động lực học Neural ODE

**Định nghĩa:** Cho $z(t) \in \mathbb{R}^D$ là trạng thái tại thời điểm $t$:

$$
\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad z(0) \sim p_0
$$

**Giải thích:**
- $f_\theta$: Neural network parametrize vận tốc
- $p_0$: Base distribution (thường $\mathcal{N}(0, I)$)
- Giải ODE → $z(t)$ tại mọi thời điểm

**Ví dụ 2D:**

```python
def f_theta(z, t):
    # z: (batch, 2)
    # t: scalar
    return nn.Sequential(
        nn.Linear(3, 64),  # [z1, z2, t] → 64
        nn.Tanh(),
        nn.Linear(64, 2)   # → [dz1/dt, dz2/dt]
    )(torch.cat([z, t * torch.ones(len(z), 1)], dim=1))
```

### 3.2. Instantaneous Change of Variables

**RealNVP (discrete):**

$$
\log p_{k+1}(z_{k+1}) = \log p_k(z_k) - \log|\det J_k|
$$

**CNF (continuous):**

$$
\frac{d}{dt} \log p(z(t)) = -\text{Tr}\left(\frac{\partial f_\theta}{\partial z}(z(t), t)\right)
$$

**Chứng minh (sketch):**

Từ change of variables: $p(z(t)) |\det J(t)| = p(z(0))$

Lấy logarithm và đạo hàm theo $t$:

$$
\frac{d}{dt}\log p(z(t)) + \frac{d}{dt}\log|\det J(t)| = 0
$$

Sử dụng Jacobi's formula:

$$
\frac{d}{dt}\log|\det J| = \text{Tr}\left(J^{-1} \frac{dJ}{dt}\right) = \text{Tr}\left(\frac{\partial f}{\partial z}\right)
$$

→ Kết quả.

### 3.3. Tích phân Log-Likelihood

**Log-density tại $t=1$:**

$$
\log p_1(z(1)) = \log p_0(z(0)) - \int_0^1 \text{Tr}\left(\frac{\partial f_\theta}{\partial z}(z(t), t)\right) dt
$$

**Cách sử dụng:**

```python
# Given data x (at t=1)
# 1. Solve ODE backward: x → z0
z0 = odeint(f_theta, x, t=[1, 0])[-1]

# 2. Compute integral of trace
integral_trace = odeint(
    lambda z, t: -trace(jacobian(f_theta, z, t)),
    x, t=[1, 0]
)

# 3. Log-likelihood
log_p1 = gaussian_log_prob(z0) - integral_trace
```

**Vấn đề:** Tính trace của Jacobian $D \times D$ rất đắt!

## 4. FFJORD: Kỹ thuật tính toán

### 4.1. Hutchinson Trace Estimator

**Vấn đề:** $\text{Tr}(\frac{\partial f}{\partial z})$ cần ma trận Jacobian đầy đủ → $O(D^2)$

**Giải pháp Hutchinson:**

Với vector ngẫu nhiên $\epsilon \sim \mathcal{N}(0, I)$:

$$
\text{Tr}\left(\frac{\partial f}{\partial z}\right) = \mathbb{E}_\epsilon\left[\epsilon^T \frac{\partial f}{\partial z} \epsilon\right]
$$

**Ước lượng (Monte Carlo):**

$$
\text{Tr}(J) \approx \epsilon^T J \epsilon
$$

**Tính toán hiệu quả:**

```python
def trace_estimator(f, z, t, num_samples=1):
    """
    Estimate trace using Hutchinson with vjp
    """
    traces = []
    
    for _ in range(num_samples):
        # Random Gaussian vector
        eps = torch.randn_like(z)
        
        # Compute f(z, t)
        f_z = f(z, t)
        
        # Vector-Jacobian product: eps^T @ J
        vjp = torch.autograd.grad(
            f_z, z,
            grad_outputs=eps,
            create_graph=True
        )[0]
        
        # Inner product: eps^T @ J @ eps
        trace = (vjp * eps).sum(dim=-1)
        traces.append(trace)
    
    return torch.stack(traces).mean(dim=0)
```

**Độ phức tạp:** $O(D)$ thay vì $O(D^2)$!

### 4.2. Adjoint Sensitivity Method

**Vấn đề:** Backprop qua ODE solver tốn memory (lưu toàn bộ trajectory)

**Giải pháp:** Adjoint method (Chen et al., 2018)

**Ý tưởng:** Giải ODE phụ để tính gradient:

$$
\frac{da(t)}{dt} = -a(t)^T \frac{\partial f_\theta}{\partial z}, \quad a(1) = \frac{\partial \mathcal{L}}{\partial z(1)}
$$

**Lợi ích:** Memory $O(1)$ thay vì $O(T)$!

**Trong `torchdiffeq`:**

```python
from torchdiffeq import odeint_adjoint

# Automatic adjoint method
z_final = odeint_adjoint(f_theta, z0, t=[0, 1])
```

### 4.3. Adaptive ODE Solver

FFJORD dùng adaptive solver (Dormand-Prince):

```python
z_final = odeint(
    f_theta, z0,
    t=torch.tensor([0.0, 1.0]),
    rtol=1e-5,  # Relative tolerance
    atol=1e-5,  # Absolute tolerance
    method='dopri5'
)
```

**Điều chỉnh step size tự động** để đảm bảo sai số < tolerance.

## 5. Hàm mục tiêu

### 5.1. Negative Log-Likelihood

```python
def nll_loss(model, x_batch):
    """
    Compute negative log-likelihood for FFJORD
    
    Args:
        model: FFJORD model
        x_batch: Data batch (B, D)
    
    Returns:
        loss: Scalar
    """
    # Solve ODE backward: x(1) → z(0)
    # Also track integral of trace
    
    z0, neg_log_det = model.inverse(x_batch)
    
    # Gaussian log-prob
    log_pz = -0.5 * (z0 ** 2).sum(dim=1) - \
              0.5 * D * np.log(2 * np.pi)
    
    # Log p(x) = log p(z) - integral(trace)
    log_px = log_pz - neg_log_det
    
    # Negative log-likelihood
    loss = -log_px.mean()
    return loss
```

### 5.2. Regularizers

**Kinetic energy regularization:**

$$
\mathcal{L}_{\text{kinetic}} = \int_0^1 \|f_\theta(z(t), t)\|^2 dt
$$

Giúp ODE mượt hơn, ít NFE (number of function evaluations) hơn.

**Directional penalty:**

$$
\mathcal{L}_{\text{dir}} = \int_0^1 \left\|\frac{\partial f_\theta}{\partial t}\right\|^2 dt
$$

**Total loss:**

$$
\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_1 \mathcal{L}_{\text{kinetic}} + \lambda_2 \mathcal{L}_{\text{dir}}
$$

## 6. Implementation PyTorch

### 6.1. ODEFunc with Trace

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class ODEFunc(nn.Module):
    """
    ODE function f_theta(z, t) with trace computation
    """
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, states):
        """
        Args:
            t: Time (scalar)
            states: (z, logp) concatenated
        
        Returns:
            (dz/dt, dlogp/dt)
        """
        z = states[0]
        
        # Ensure t is tensor
        t_vec = torch.ones(z.shape[0], 1, device=z.device) * t
        
        # Compute f(z, t)
        zt = torch.cat([z, t_vec], dim=1)
        dz_dt = self.net(zt)
        
        # Hutchinson trace estimator
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt_for_trace = self.net(torch.cat([z, t_vec], dim=1))
            
            # Random vector
            eps = torch.randn_like(z)
            
            # VJP: eps^T @ J
            vjp = torch.autograd.grad(
                dz_dt_for_trace, z,
                grad_outputs=eps,
                create_graph=True
            )[0]
            
            # Trace estimate
            trace = (vjp * eps).sum(dim=1, keepdim=True)
        
        # dlogp/dt = -trace
        dlogp_dt = -trace
        
        return (dz_dt, dlogp_dt)
```

### 6.2. CNF Model

```python
class CNF(nn.Module):
    """
    Continuous Normalizing Flow
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.odefunc = ODEFunc(dim)
    
    def forward(self, z0, integration_times=None):
        """
        Forward: z0 → z1
        
        Args:
            z0: Initial state (B, D)
            integration_times: [0, 1]
        
        Returns:
            z1: Final state
            delta_logp: Change in log-prob
        """
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0], device=z0.device)
        
        # Initial log-prob (zero change)
        logp_z0 = torch.zeros(z0.shape[0], 1, device=z0.device)
        
        # Solve ODE
        states_t = odeint_adjoint(
            self.odefunc,
            (z0, logp_z0),
            integration_times,
            rtol=1e-5,
            atol=1e-5,
            method='dopri5'
        )
        
        z1 = states_t[0][-1]
        delta_logp = states_t[1][-1]
        
        return z1, delta_logp
    
    def inverse(self, x):
        """
        Inverse: x (t=1) → z (t=0)
        
        Returns:
            z0: Base distribution
            delta_logp: Change in log-prob
        """
        integration_times = torch.tensor([1.0, 0.0], device=x.device)
        
        logp_x = torch.zeros(x.shape[0], 1, device=x.device)
        
        states_t = odeint_adjoint(
            self.odefunc,
            (x, logp_x),
            integration_times,
            rtol=1e-5,
            atol=1e-5,
            method='dopri5'
        )
        
        z0 = states_t[0][-1]
        delta_logp = states_t[1][-1]
        
        return z0, delta_logp
    
    def log_prob(self, x):
        """
        Compute log p(x)
        """
        z0, delta_logp = self.inverse(x)
        
        # Gaussian log-prob
        log_pz = -0.5 * (z0 ** 2).sum(dim=1, keepdim=True) - \
                  0.5 * self.dim * np.log(2 * np.pi)
        
        log_px = log_pz - delta_logp
        return log_px.squeeze()
```

### 6.3. Training Loop

```python
def train_cnf(model, data_loader, epochs=100, lr=1e-3):
    """
    Train CNF with maximum likelihood
    """
    import numpy as np
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        
        for x_batch in data_loader:
            # Negative log-likelihood
            log_px = model.log_prob(x_batch)
            loss = -log_px.mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch {epoch+1}/{epochs} | NLL: {avg_loss:.4f}")
    
    return model

# Usage
model = CNF(dim=2)
# model = train_cnf(model, data_loader, epochs=100)

# Sampling
z = torch.randn(100, 2)
with torch.no_grad():
    x_samples, _ = model(z)
```

### 6.4. Complete Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# Generate toy data (2 moons)
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=5000, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Train CNF
model = CNF(dim=2)
print("Training CNF...")
model = train_cnf(model, loader, epochs=50, lr=1e-3)

# Sample
print("\nSampling...")
z = torch.randn(1000, 2)
with torch.no_grad():
    x_samples, _ = model(z)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:, 0], X[:, 1], alpha=0.3, s=10)
axes[0].set_title("Data")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(z[:, 0], z[:, 1], alpha=0.3, s=10)
axes[1].set_title("Base (Gaussian)")
axes[1].grid(True, alpha=0.3)

axes[2].scatter(x_samples[:, 0], x_samples[:, 1], alpha=0.3, s=10)
axes[2].set_title("CNF Samples")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnf_result.png', dpi=150)
plt.show()
```

## 7. So sánh các phương pháp

### Bảng tổng hợp

| Method | Likelihood | Architecture | NFE | Training | Sampling |
|--------|------------|--------------|-----|----------|----------|
| **RealNVP** | Exact | Coupling (constrained) | 1 | Easy | Fast |
| **Glow** | Exact | Coupling + 1x1 | 1 | Medium | Fast |
| **FFJORD/CNF** | Exact | **Free-form** | 50-200 | Hard | Slow |
| **Flow Matching** | No | Free-form | 50-100 | Easy | Medium |
| **Rectified Flow** | No | Free-form | 1-5 | Easy | **Fast** |

**NFE** = Number of Function Evaluations (số lần gọi network)

### Timeline phát triển

```
2015: NICE (coupling layers)
  ↓
2017: RealNVP (affine coupling)
  ↓
2018: Glow (1x1 conv + ActNorm)
       + Neural ODE
  ↓
2019: FFJORD (CNF + Hutchinson)
  ↓
2021: Flow Matching (regression, no likelihood)
  ↓
2022: Rectified Flow (straight paths)
  ↓
2023: Stochastic Interpolants (unified framework)
```

### Khi nào dùng FFJORD?

**✅ Dùng khi:**

1. **Cần exact likelihood:** Bayesian inference, anomaly detection
2. **Flexible architecture:** Không bị ràng buộc coupling
3. **Continuous dynamics:** Mô phỏng quá trình vật lý
4. **Research:** Khám phá CNF theory

**❌ Không dùng khi:**

- Cần sampling cực nhanh → Dùng Rectified Flow
- Production với latency constraints → Dùng RealNVP/Glow
- Image generation quality ưu tiên → Dùng Diffusion
- Không quan tâm likelihood → Dùng Flow Matching

## 8. Kinh nghiệm thực nghiệm

### 8.1. Hyperparameters

Người thợ gốm học được:

**ODE Solver:**
```python
rtol = 1e-5  # Relative tolerance
atol = 1e-5  # Absolute tolerance
method = 'dopri5'  # Adaptive RK
```

**Architecture:**
```python
hidden_dim = 64-128  # Không quá sâu
activation = nn.Tanh()  # Smooth (Lipschitz)
num_layers = 3-4
```

**Training:**
```python
lr = 1e-3
batch_size = 64-256
grad_clip = 10.0
epochs = 100-500
```

### 8.2. Common Issues

**Issue 1: ODE không hội tụ**

```python
# Bad: Quá sâu, ReLU
net = nn.Sequential(
    nn.Linear(dim+1, 512),
    nn.ReLU(),  # Not smooth!
    *[nn.Linear(512, 512), nn.ReLU()] * 10,  # Too deep!
    nn.Linear(512, dim)
)

# Good: Nông, Tanh
net = nn.Sequential(
    nn.Linear(dim+1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, dim)
)
```

**Issue 2: Trace estimator variance cao**

```python
# Bad: 1 sample
trace = hutchinson_trace(f, z, num_samples=1)

# Good: 2-3 samples
trace = hutchinson_trace(f, z, num_samples=3)
```

**Issue 3: NFE quá cao (> 1000)**

- Giảm số layers
- Dùng activation mượt hơn (Tanh, Softplus)
- Thêm kinetic energy regularization
- Tăng tolerance (rtol=1e-3)

### 8.3. Monitoring

```python
# Log NFE per sample
print(f"Average NFE: {nfe.mean():.1f}")

# Acceptable: 50-200
# Too high: > 500 → Need architecture change
```

## 9. Kết luận

### Ngày thứ 7: Người thợ gốm tổng kết

Sau 2 ngày học CNF và FFJORD, người thợ gốm ghi vào sổ tay:

> **FFJORD & CNF: Dòng Chảy Liên Tục**
>
> 1. **Continuous dynamics:** ODE thay vì discrete layers
> 2. **Exact likelihood:** Instantaneous change of variables
> 3. **Hutchinson estimator:** Tính trace hiệu quả $O(D)$
> 4. **Adjoint method:** Backprop không tốn memory
> 5. **Trade-off:** Flexible + exact nhưng chậm hơn

### So với RealNVP

| Aspect | RealNVP/Glow | FFJORD/CNF |
|--------|--------------|------------|
| **Dynamics** | Discrete (K layers) | Continuous (ODE) |
| **Architecture** | Constrained (coupling) | Free-form |
| **Likelihood** | Exact (log-det) | Exact (trace integral) |
| **NFE** | K (fixed) | Adaptive (50-200) |
| **Speed** | Fast | Slower |
| **Use case** | Production | Research |

### Câu chuyện tiếp theo

"FFJORD cho tôi **flexibility** và **exact likelihood**," anh nghĩ. "Nhưng nó quá chậm cho production. Liệu có cách nào kết hợp được ưu điểm của CNF (continuous) nhưng nhanh hơn?"

→ Dẫn đến **Flow Matching** (bài tiếp theo) - trade exact likelihood để đổi lấy tốc độ!

---

## Tài liệu tham khảo

1. **Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019)** - "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" _(NeurIPS 2019)_

2. **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018)** - "Neural Ordinary Differential Equations" _(NeurIPS 2018 Best Paper)_

3. **Hutchinson, M. F. (1989)** - "A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines" _(Communications in Statistics)_

4. **Pontryagin, L. S. (1962)** - "The Mathematical Theory of Optimal Processes" _(Adjoint method origins)_

5. **Finlay, C., Jacobsen, J. H., Nurbekyan, L., & Oberman, A. M. (2020)** - "How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization" _(ICML 2020)_

6. **Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021)** - "Normalizing Flows for Probabilistic Modeling and Inference" _(JMLR)_

---

**Series:** [Generative AI Overview](/posts/2025/generative-ai-overview)

**Bài trước:** [RealNVP & Glow: Nghệ Thuật Biến Đổi Có Thể Đảo Ngược](/posts/2025/realnvp-glow)

**Bài tiếp theo:** [Flow Matching: Từ Likelihood Đến Regression](/posts/2025/conditional-flow-matching)

<script src="/assets/js/katex-init.js"></script>
