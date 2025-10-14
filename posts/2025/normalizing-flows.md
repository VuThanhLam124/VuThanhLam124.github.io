# Normalizing Flow & Continuous Normalizing Flow

**Ngày đăng:** 14/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Flow-based Models, Generative AI

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Normalizing Flows: Foundations](#normalizing-flows-foundations)
3. [Continuous Normalizing Flows (CNF)](#continuous-normalizing-flows)
4. [Neural ODEs và CNF](#neural-odes-và-cnf)
5. [Implementation với PyTorch](#implementation-với-pytorch)
6. [Applications](#applications)
7. [Kết luận](#kết-luận)

---

## Giới thiệu

Normalizing Flows là một trong những approaches elegant nhất trong generative modeling. Khác với VAEs sử dụng approximate inference hoặc GANs với adversarial training, Normalizing Flows học direct bijective transformations giữa simple base distribution (thường là Gaussian) và complex data distribution.

Key advantages:
- **Exact likelihood computation** - không cần approximation
- **Efficient sampling** - single forward pass
- **Invertible transformations** - có thể map cả 2 chiều
- **Tractable density estimation**

## 1. Normalizing Flows: Foundations

### 1.1 Change of Variables Formula

Giả sử có bijective mapping $f: \mathbb{R}^d \rightarrow \mathbb{R}^d$, với inverse $f^{-1}$ và input distribution $p_z(z)$:

$$
p_x(x) = p_z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}(x)}{\partial x} \right|
$$

hoặc với log-likelihood:

$$
\log p_x(x) = \log p_z(z) + \log \left| \det \frac{\partial f}{\partial z} \right|^{-1}
$$

### 1.2 Jacobian và Computational Efficiency

Challenge chính: computing determinant của Jacobian matrix có complexity $O(d^3)$ - prohibitive cho high-dimensional data.

**Solution approaches:**
1. **Triangular Jacobians** - det computation trong $O(d)$
2. **Special structures** (e.g., autoregressive, coupling layers)
3. **Continuous-time limit** - CNF với ODE

### 1.3 Composing Flows

Single bijection thường không đủ expressive. Composition của nhiều flows:

$$
x = f_K \circ f_{K-1} \circ ... \circ f_1(z)
$$

với log-likelihood:

$$
\log p_x(x) = \log p_z(z) - \sum_{k=1}^K \log \left| \det J_{f_k}(z_k) \right|
$$

## 2. Continuous Normalizing Flows

### 2.1 From Discrete to Continuous

Thay vì sequence of discrete transformations, CNF parameterizes **continuous-time dynamics** qua ODE:

$$
\frac{dz(t)}{dt} = f(z(t), t, \theta)
$$

với $z(0) \sim p_0$ (base) và $z(1) \sim p_1$ (data).

### 2.2 Instantaneous Change of Variables

Log-density evolution theo thời gian:

$$
\frac{d \log p(z(t))}{dt} = -\text{tr}\left(\frac{\partial f}{\partial z(t)}\right)
$$

Integrating từ $t=0$ đến $t=1$:

$$
\log p_1(z(1)) = \log p_0(z(0)) - \int_0^1 \text{tr}\left(\frac{\partial f}{\partial z(t)}\right) dt
$$

**Advantage:** Không cần compute full Jacobian determinant - chỉ trace!

### 2.3 Free-form Jacobians

CNF cho phép **arbitrary architectures** cho vector field $f$ mà không constraint về Jacobian structure. Điều này contrasts với discrete flows cần specific architectures (coupling layers, autoregressive, etc.).

## 3. Neural ODEs và CNF

### 3.1 ODE Solvers

Solving ODE:
$$
z(t_1) = z(t_0) + \int_{t_0}^{t_1} f(z(t), t, \theta) dt
$$

**Adaptive solvers** (Dormand-Prince, Runge-Kutta):
- Automatic step size adjustment
- Error control
- Trade-off: accuracy vs. speed

### 3.2 Adjoint Sensitivity Method

Backpropagation through ODE solver expensive. **Adjoint method** computes gradients hiệu quả:

$$
\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial \theta} dt
$$

với adjoint state:
$$
a(t) = \frac{\partial L}{\partial z(t)}
$$

**Benefits:**
- Memory efficient (constant w.r.t. integration steps)
- Scalable to long integration times

## 4. Implementation với PyTorch

### 4.1 Vector Field Network

```python
import torch
import torch.nn as nn

class VectorField(nn.Module):
    """Neural network parameterizing the vector field f(z, t)"""
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
        # t: scalar time, z: [batch_size, dim]
        t_expand = t * torch.ones(z.shape[0], 1).to(z.device)
        tz = torch.cat([t_expand, z], dim=1)
        return self.net(tz)
```

### 4.2 CNF Model

```python
from torchdiffeq import odeint

class ContinuousNormalizingFlow(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.vf = VectorField(dim, hidden_dim)
        self.dim = dim
    
    def forward(self, z0, t=torch.linspace(0, 1, 2)):
        """
        Forward: z0 -> z1
        z0: [batch_size, dim]
        t: integration timepoints
        """
        # Solve ODE
        z_trajectory = odeint(
            self.vf, 
            z0, 
            t,
            method='dopri5',
            rtol=1e-5,
            atol=1e-7
        )
        return z_trajectory[-1]  # Return final state
    
    def log_prob(self, x):
        """Compute log p(x) via change of variables"""
        batch_size = x.shape[0]
        
        # Base distribution
        base_dist = torch.distributions.Normal(
            torch.zeros(self.dim),
            torch.ones(self.dim)
        )
        
        # Backward integration: x -> z0
        z0 = self.inverse(x)
        
        # Log prob from base
        log_p_z0 = base_dist.log_prob(z0).sum(dim=1)
        
        # Trace of Jacobian (compute via augmented ODE)
        log_det = self.compute_log_det(x, z0)
        
        return log_p_z0 - log_det
    
    def inverse(self, x):
        """Backward: x -> z0"""
        t_reverse = torch.linspace(1, 0, 2)
        z_trajectory = odeint(
            self.vf,
            x,
            t_reverse,
            method='dopri5'
        )
        return z_trajectory[-1]
    
    def sample(self, num_samples):
        """Generate samples"""
        z0 = torch.randn(num_samples, self.dim)
        x = self.forward(z0)
        return x
```

### 4.3 Training Loop

```python
def train_cnf(model, data_loader, num_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            x = batch[0]  # [batch_size, dim]
            
            # Compute negative log-likelihood
            log_px = model.log_prob(x)
            loss = -log_px.mean()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}")
```

### 4.4 Efficient Trace Computation

Thay vì compute full Jacobian, dùng **Hutchinson's trace estimator**:

```python
def hutchinson_trace(f, z, num_samples=1):
    """
    Estimate tr(df/dz) using Hutchinson's estimator
    E[v^T (df/dz) v] = tr(df/dz) for v ~ N(0, I)
    """
    trace_estimate = 0
    for _ in range(num_samples):
        v = torch.randn_like(z)
        df_dz_v = torch.autograd.grad(
            f, z, v, create_graph=True, retain_graph=True
        )[0]
        trace_estimate += (v * df_dz_v).sum(dim=1)
    
    return trace_estimate / num_samples
```

## 5. Architectures cho CNF

### 5.1 Time-Conditioned Networks

```python
class TimeConcatVectorField(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ConcatSquashLinear(dim, hidden_dim),
            nn.Tanh(),
            ConcatSquashLinear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, z):
        return self.net((t, z))

class ConcatSquashLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.hyper_gate = nn.Linear(1, out_dim)
        self.hyper_bias = nn.Linear(1, out_dim)
    
    def forward(self, t_z):
        t, z = t_z
        return self.linear(z) * torch.sigmoid(self.hyper_gate(t)) \
               + self.hyper_bias(t)
```

### 5.2 FFJORD Architecture

**Free-Form Jacobian of Reversible Dynamics** - state-of-the-art CNF:

```python
class FFJORD(nn.Module):
    def __init__(self, dim, hidden_dims=[64, 64, 64]):
        super().__init__()
        layers = []
        prev_dim = dim + 1  # +1 for time
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Softplus()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, dim))
        
        self.net = nn.Sequential(*layers)
        self.dim = dim
    
    def forward(self, t, state):
        # state: [batch, dim + 1] (z + log_det concatenated)
        z = state[:, :self.dim]
        
        # Compute dz/dt
        t_expand = t * torch.ones(z.shape[0], 1).to(z.device)
        tz = torch.cat([t_expand, z], dim=1)
        dz_dt = self.net(tz)
        
        # Compute trace(df/dz) using Hutchinson
        trace_df_dz = hutchinson_trace(dz_dt, z)
        
        # Augmented dynamics: [dz/dt, -trace]
        return torch.cat([dz_dt, -trace_df_dz.unsqueeze(1)], dim=1)
```

## 6. Applications

### 6.1 Density Estimation

CNF excellent cho modeling complex distributions:
- High-dimensional data
- Multimodal distributions
- Manifold learning

### 6.2 Variational Inference

Dùng CNF làm flexible posterior approximation trong VAE:

$$
q_\phi(z|x) = \text{CNF}_\phi(\text{encoder}(x))
$$

### 6.3 Generative Modeling

- Image generation
- Time series modeling
- Molecular design

### 6.4 Optimal Transport

CNF naturally learns optimal transport paths giữa distributions.

## 7. Challenges và Solutions

### 7.1 Computational Cost

**Problem:** ODE solving expensive, especially với many function evaluations.

**Solutions:**
- Adaptive solvers với error tolerance tuning
- Reduce NFE (number of function evaluations) qua better architectures
- Distillation sang simpler models

### 7.2 Numerical Stability

**Problem:** ODE integration có thể diverge.

**Solutions:**
- Gradient clipping
- Regularization terms (kinetic energy, Jacobian norm)
- Proper initialization

### 7.3 Expressivity vs. Efficiency

**Trade-off:** More complex vector fields → better expressivity nhưng slower inference.

**Balance:**
- Use simple architectures nhưng longer integration time
- Progressive training strategies
- Hybrid models (combine với other approaches)

## Kết luận

Normalizing Flows và CNF offer powerful framework cho generative modeling với exact likelihood và flexible architectures. CNF đặc biệt attractive vì:

✅ **Free-form Jacobians** - không constraint architecture  
✅ **Continuous dynamics** - smooth transformations  
✅ **Scalability** - efficient gradient computation  
✅ **Theoretical guarantees** - provable properties

Moving forward: combinations với diffusion models (Rectified Flows, Flow Matching) leverage best của cả hai worlds.

## Tài liệu tham khảo

1. Rezende, D. J., & Mohamed, S. (2015). "Variational Inference with Normalizing Flows" - ICML
2. Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). "Neural Ordinary Differential Equations" - NeurIPS
3. Grathwohl, W., Chen, R. T., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" - ICLR
4. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). "Normalizing Flows for Probabilistic Modeling and Inference" - JMLR

---

**Tags:** #NormalizingFlows #CNF #GenerativeModels #NeuralODE #FlowBasedModels #DeepLearning #PyTorch
