---
title: "Normalizing Flow & Continuous Normalizing Flow"
date: "2025-01-15"
category: "flow-based-models"
tags: ["normalizing-flows", "continuous-normalizing-flows", "neural-ode", "generative-models"]
description: "Tìm hiểu về Normalizing Flows và Continuous Normalizing Flows - nền tảng của các mô hình generative hiện đại với khả năng tính exact likelihood"
---

# Normalizing Flow & Continuous Normalizing Flow

## Tổng quan

Normalizing Flows (NF) là một họ các mô hình generative cho phép tính toán exact likelihood thông qua chuỗi các biến đổi invertible. Continuous Normalizing Flows (CNF) mở rộng ý tưởng này sang không gian liên tục với Neural ODEs.

## 1. Normalizing Flows - Foundations

### 1.1 Change of Variables Formula

Cho biến đổi invertible $f: \mathbb{R}^d \to \mathbb{R}^d$, với $z \sim p_z(z)$ và $x = f(z)$:

$$
p_x(x) = p_z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|
$$

Hoặc tương đương:

$$
\log p_x(x) = \log p_z(z) - \log \left| \det \frac{\partial f}{\partial z} \right|
$$

### 1.2 Stacking Multiple Flows

Với chuỗi transformations $f = f_K \circ f_{K-1} \circ ... \circ f_1$:

$$
\log p_x(x) = \log p_z(z_0) - \sum_{k=1}^K \log \left| \det \frac{\partial f_k}{\partial z_{k-1}} \right|
$$

### 1.3 Jacobian Computation Challenge

**Vấn đề:** Tính determinant của Jacobian matrix $d \times d$ có độ phức tạp $O(d^3)$ - không khả thi với high-dimensional data.

**Giải pháp:** Thiết kế architectures với Jacobian có cấu trúc đặc biệt:
- **Triangular Jacobian**: $O(d)$ computation
- **Coupling layers**: Split dimensions, transform subset

## 2. Continuous Normalizing Flows

### 2.1 Từ Discrete đến Continuous

Thay vì discrete transformations, CNF sử dụng continuous-time dynamics:

$$
\frac{dz(t)}{dt} = f(z(t), t, \theta)
$$

với $z(0) \sim p_0$ (base distribution) và $z(1) = x$ (data distribution).

### 2.2 Instantaneous Change of Variables

Thay vì tính tổng log-det Jacobians, CNF trace evolution của log-density:

$$
\frac{\partial \log p_t(z(t))}{\partial t} = -\text{Tr}\left(\frac{\partial f}{\partial z}\right)
$$

**Augmented ODE:**

$$
\frac{d}{dt}\begin{bmatrix} z(t) \\ \log p_t(z(t)) \end{bmatrix} = \begin{bmatrix} f(z(t), t, \theta) \\ -\text{Tr}\left(\frac{\partial f}{\partial z}\right) \end{bmatrix}
$$

### 2.3 Neural ODE Connection

CNF được implement thông qua Neural ODEs:
- **Forward pass**: Solve ODE từ $t=0$ đến $t=1$ để sinh samples
- **Backward pass**: Adjoint method để tính gradients
- **Flexible architectures**: Không yêu cầu invertibility tại mỗi timestep

## 3. Ưu điểm & Nhược điểm

### Ưu điểm của Normalizing Flows:
✅ **Exact likelihood computation** - không cần approximation
✅ **Exact sampling** - efficient generation process
✅ **Flexible architectures** (với CNF)
✅ **Stable training** - không có mode collapse như GANs

### Nhược điểm:
❌ **Jacobian constraints** - hạn chế architecture design (discrete NF)
❌ **Computational cost** - CNF yêu cầu ODE solvers
❌ **Quality trade-off** - thường tạo samples kém hơn GANs/Diffusion

## 4. Code Implementation

### 4.1 Simple Vector Field (PyTorch)

```python
import torch
import torch.nn as nn

class VectorField(nn.Module):
    """Neural network parameterizing the vector field f(z,t)"""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, z):
        # t: scalar time
        # z: (batch_size, dim)
        t_vec = t * torch.ones(z.shape[0], 1).to(z.device)
        tz = torch.cat([t_vec, z], dim=1)
        return self.net(tz)
```

### 4.2 CNF with torchdiffeq

```python
from torchdiffeq import odeint

class CNF(nn.Module):
    def __init__(self, vector_field):
        super().__init__()
        self.vf = vector_field
        
    def forward(self, z0, t_span):
        """
        z0: initial state (batch_size, dim)
        t_span: time points to evaluate, e.g., torch.tensor([0., 1.])
        """
        # Solve ODE
        z_t = odeint(self.vf, z0, t_span, method='dopri5')
        return z_t[-1]  # Return final state
    
    def sample(self, num_samples, device='cuda'):
        """Generate samples from base distribution"""
        z0 = torch.randn(num_samples, self.vf.net[0].in_features - 1).to(device)
        t_span = torch.tensor([0., 1.]).to(device)
        return self.forward(z0, t_span)
```

### 4.3 Training Loop

```python
def train_cnf(model, dataloader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch.to(device)  # Real data
            
            # Forward: x (t=1) -> z (t=0)
            t_span = torch.tensor([1., 0.]).to(device)
            z = model(x, t_span)
            
            # Compute negative log-likelihood
            # (Simplified - full implementation needs log-det computation)
            prior_logprob = -0.5 * (z**2).sum(dim=1).mean()
            loss = -prior_logprob
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 5. Applications

### 5.1 Density Estimation
- Modeling complex distributions
- Anomaly detection với likelihood thresholds

### 5.2 Generative Modeling
- Image synthesis
- Molecular generation

### 5.3 Variational Inference
- Flexible posterior approximation trong Bayesian inference

## 6. Connections to Flow Matching

Normalizing Flows, đặc biệt là CNF, là nền tảng cho **Flow Matching** - một framework hiện đại hơn:

- **CNF**: Học vector field để transport distributions
- **Flow Matching**: Regression-based training (không cần ODE solver trong training)
- **Rectified Flows**: Straighten trajectories để tăng efficiency

Flow Matching addresses computational bottlenecks của CNF trong khi giữ lại flexibility.

## 7. Key Takeaways

1. **Normalizing Flows** transform simple distributions thành complex ones qua invertible mappings
2. **CNF** mở rộng sang continuous-time với Neural ODEs
3. **Exact likelihood** là advantage lớn nhất so với GANs
4. **Computational cost** của ODE solvers là trade-off chính
5. **Foundation cho Flow Matching** - hiểu NF/CNF là critical để master modern flow-based models

## References

1. Rezende & Mohamed (2015) - "Variational Inference with Normalizing Flows"
2. Chen et al. (2018) - "Neural Ordinary Differential Equations" (NeurIPS Best Paper)
3. Grathwohl et al. (2018) - "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"
4. Papamakarios et al. (2021) - "Normalizing Flows for Probabilistic Modeling and Inference"

---

**Next Reading:**
- [Real NVP & Glow: Invertible Architectures](/posts/2025/realnvp-glow)
- [Flow Matching Theory](/posts/2025/flow-matching-theory)
- [Rectified Flows](/posts/2025/rectified-flows)
