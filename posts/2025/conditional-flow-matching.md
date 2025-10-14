# Conditional Flow Matching: Controlled Generation

**Ngày đăng:** 17/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Flow-based Models, Deep Learning

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Conditional Flows Theory](#conditional-flows-theory)
3. [Optimal Transport Paths](#optimal-transport-paths)
4. [Training Objectives](#training-objectives)
5. [Implementation](#implementation)
6. [Applications](#applications)

---

## Giới thiệu

**Conditional Flow Matching (CFM)** extends Flow Matching to learn **conditional probability paths**, enabling controlled generation và efficient training.

**Key innovations:**
- Condition on source-target pairs
- Learn optimal transport paths
- Simulation-free training
- State-of-art performance

## 1. Conditional Flow Matching Framework

### 1.1 Problem Setup

Goal: Learn generative model $p_t(x|y)$ conditioned on $y$.

**Conditional flow:**
$$
\frac{dx_t}{dt} = v_t(x_t|y), \quad x_0 \sim p_0(x|y), \quad x_1 \sim p_1(x|y)
$$

### 1.2 Marginal vs Conditional

**Marginal path:** $p_t(x)$
**Conditional path:** $p_t(x|x_0, x_1)$

**Key insight:** Easy to construct conditional paths, then marginalize:
$$
p_t(x) = \int p_t(x|x_0, x_1) p(x_0, x_1) dx_0 dx_1
$$

## 2. Optimal Transport Conditional Paths

### 2.1 Affine Conditional Flow

Simple yet powerful: **linear interpolation**

$$
p_t(x|x_0, x_1) = \mathcal{N}(x; \mu_t(x_0, x_1), \sigma_t^2 I)
$$

với:
$$
\mu_t = (1-t)x_0 + tx_1, \quad \sigma_t = \sigma_{\min}
$$

**Conditional velocity:**
$$
u_t(x|x_0, x_1) = \frac{x_1 - x_0}{1}  = x_1 - x_0
$$

### 2.2 Training Objective

**Conditional Flow Matching loss:**
$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1, x_t}\left[\|v_\theta(x_t, t) - u_t(x_t|x_0, x_1)\|^2\right]
$$

với $x_t \sim p_t(x|x_0, x_1)$.

**Advantage:** No need to compute marginal $p_t(x)$!

## 3. Implementation

### 3.1 Conditional Velocity Network

```python
import torch
import torch.nn as nn

class ConditionalVelocityNet(nn.Module):
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        
        # Time embedding with sinusoidal encoding
        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main UNet-like architecture
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU()
            )
        ])
        
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU()
            )
        ])
        
        self.output = nn.Linear(hidden_dim, dim)
        
        # Initialize output to zero
        self.output.weight.data.zero_()
        self.output.bias.data.zero_()
    
    def time_embedding(self, t, dim=64):
        """Sinusoidal time embedding"""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, t):
        """
        x: [B, D] - current state
        t: [B] - time
        """
        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        skips = []
        h = x
        for layer in self.encoder:
            h = layer(h) + t_emb
            skips.append(h)
        
        # Middle
        h = self.middle(h) + t_emb
        
        # Decoder with skip connections
        for layer, skip in zip(self.decoder, reversed(skips)):
            h = torch.cat([h, skip], dim=-1)
            h = layer(h) + t_emb
        
        return self.output(h)
```

### 3.2 Training Loop

```python
def train_conditional_flow_matching(
    model, 
    x0_samples,  # Noise distribution
    x1_samples,  # Data distribution
    num_epochs=1000,
    batch_size=256,
    lr=1e-4
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(x0_samples, x1_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for x0, x1 in dataloader:
            # Sample random time
            t = torch.rand(x0.shape[0], device=x0.device)
            
            # Construct conditional path: linear interpolation
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            
            # Add small noise (optional, helps stability)
            sigma = 1e-4
            x_t = x_t + sigma * torch.randn_like(x_t)
            
            # Target: conditional velocity u_t = x1 - x0
            u_t = x1 - x0
            
            # Predict velocity
            v_pred = model(x_t, t)
            
            # CFM loss
            loss = torch.mean((v_pred - u_t) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    return model
```

### 3.3 Sampling with ODE Solver

```python
from torchdiffeq import odeint

def sample_cfm(model, x0, method='dopri5', num_steps=None):
    """
    Generate samples using ODE solver
    
    Args:
        model: trained velocity network
        x0: initial noise [B, D]
        method: ODE solver ('euler', 'rk4', 'dopri5')
        num_steps: discretization steps (for fixed-step methods)
    """
    def ode_func(t, x):
        t_batch = t * torch.ones(x.shape[0], device=x.device)
        return model(x, t_batch)
    
    # Time span
    if num_steps is None:
        t_span = torch.tensor([0.0, 1.0])
    else:
        t_span = torch.linspace(0, 1, num_steps)
    
    with torch.no_grad():
        trajectory = odeint(
            ode_func,
            x0,
            t_span,
            method=method,
            rtol=1e-5,
            atol=1e-5
        )
    
    return trajectory[-1]
```

### 3.4 Fast Euler Sampling

```python
def sample_cfm_euler(model, x0, num_steps=100):
    """
    Fast sampling with Euler method
    """
    dt = 1.0 / num_steps
    x = x0
    
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=x.device) * (i * dt)
            v = model(x, t)
            x = x + v * dt
    
    return x
```

## 4. Advanced Techniques

### 4.1 Stochastic Interpolants

Add noise to conditional path:

$$
x_t = (1-t)x_0 + tx_1 + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

```python
def train_with_stochastic_interpolants(model, x0_samples, x1_samples, sigma_min=1e-4):
    for epoch in range(num_epochs):
        for x0, x1 in dataloader:
            t = torch.rand(x0.shape[0])
            
            # Time-dependent noise
            sigma_t = sigma_min + (1 - sigma_min) * (1 - t)
            
            # Stochastic interpolation
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            x_t = x_t + sigma_t[:, None] * torch.randn_like(x_t)
            
            # Target velocity (adjusted for noise)
            u_t = x1 - x0 - sigma_t[:, None] * torch.randn_like(x_t)
            
            v_pred = model(x_t, t)
            loss = torch.mean((v_pred - u_t) ** 2)
            
            # ... optimize
```

### 4.2 Class-Conditional Generation

```python
class ClassConditionalCFM(nn.Module):
    def __init__(self, dim, num_classes, hidden_dim=512):
        super().__init__()
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        
        # Time + class conditioned network
        self.net = ConditionalVelocityNet(dim, hidden_dim)
    
    def forward(self, x, t, class_labels):
        """
        x: [B, D]
        t: [B]
        class_labels: [B] - integer class IDs
        """
        # Embed class
        c_emb = self.class_embed(class_labels)
        
        # Combine with input (simple addition to time embedding)
        v = self.net(x, t)  # Would need to modify net to accept c_emb
        
        return v

# Training
def train_class_conditional(model, data, labels):
    for x0, x1, y in dataloader:
        t = torch.rand(x0.shape[0])
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        u_t = x1 - x0
        
        v_pred = model(x_t, t, y)
        loss = torch.mean((v_pred - u_t) ** 2)
        
        # ... optimize

# Sampling
def sample_class_conditional(model, x0, class_label):
    def ode_func(t, x):
        t_batch = t * torch.ones(x.shape[0])
        y_batch = class_label * torch.ones(x.shape[0], dtype=torch.long)
        return model(x, t_batch, y_batch)
    
    t_span = torch.tensor([0.0, 1.0])
    trajectory = odeint(ode_func, x0, t_span, method='dopri5')
    return trajectory[-1]
```

## 5. Comparison với Standard Flow Matching

| Aspect | Flow Matching | Conditional Flow Matching |
|--------|--------------|---------------------------|
| **Training** | Needs marginal $p_t(x)$ | Uses conditional $p_t(x\|x_0,x_1)$ |
| **Efficiency** | May need simulation | Simulation-free |
| **Paths** | Any valid path | Optimal transport paths |
| **Quality** | Good | State-of-art |

## 6. Applications

### 6.1 Image Generation

```python
# Train on image dataset
images = load_images()  # [N, C, H, W]
noise = torch.randn_like(images)

model = ConditionalVelocityNet(dim=images[0].numel())
train_conditional_flow_matching(model, noise, images.flatten(1))

# Generate
new_noise = torch.randn(16, images[0].numel())
generated = sample_cfm(model, new_noise, num_steps=100)
generated_images = generated.view(16, C, H, W)
```

### 6.2 Molecular Generation

```python
# Learn distribution of molecules
molecules_encoded = encode_molecules(molecules)  # [N, D]
noise = torch.randn_like(molecules_encoded)

model = train_conditional_flow_matching(model, noise, molecules_encoded)

# Generate novel molecules
new_molecules = sample_cfm(model, torch.randn(100, D))
decoded_molecules = decode_molecules(new_molecules)
```

### 6.3 Super-Resolution

```python
# Learn mapping from low-res to high-res
low_res = load_low_res_images()
high_res = load_high_res_images()

model = train_conditional_flow_matching(model, low_res, high_res)

# Super-resolve
sr_images = sample_cfm(model, low_res_test)
```

## Kết luận

Conditional Flow Matching offers powerful framework cho generative modeling:

✅ **Simulation-free training** - efficient và scalable  
✅ **Optimal transport** - learns shortest paths  
✅ **State-of-art quality** - competitive với diffusion models  
✅ **Flexible** - easy to condition và control

Combines best of both worlds: simplicity của Flow Matching + efficiency của optimal transport.

## Tài liệu tham khảo

1. Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling" - ICML
2. Tong, A., et al. (2023). "Improving and Generalizing Flow-Based Generative Models" - TMLR
3. Albergo, M. S., & Vanden-Eijnden, E. (2023). "Building Normalizing Flows with Stochastic Interpolants"

---

**Tags:** #ConditionalFlowMatching #OptimalTransport #GenerativeModels #EfficientTraining

<script src="/assets/js/katex-init.js"></script>
