# Flow Map Matching: Efficient Path Learning

**Ngày đăng:** 16/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Flow-based Models, Deep Learning

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Flow Map Concept](#flow-map-concept)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Implementation](#implementation)
5. [Comparison với Flow Matching](#comparison)
6. [Applications](#applications)

---

## Giới thiệu

**Flow Map Matching** là alternative approach to Flow Matching, focusing on learning **flow maps** thay vì velocity fields.

**Core difference:**
- Flow Matching: Learn $v_t(x)$ velocity field
- Flow Map Matching: Learn $\phi_t(x)$ - direct mapping from $t=0$

**Advantages:**
- **Simpler training** - no ODE solving during training
- **Flexible discretization** - arbitrary time steps
- **Efficient inference** - direct evaluation

## 1. Flow Map Concept

### 1.1 Definitions

**Flow map** $\phi_t: \mathbb{R}^d \to \mathbb{R}^d$ maps initial point to position at time $t$:

$$
X_t = \phi_t(X_0)
$$

**Properties:**
- $\phi_0(x) = x$ (identity)
- $\phi_1(X_0) = X_1$ (maps noise to data)
- $\phi_s \circ \phi_t = \phi_{s+t}$ (composition)

### 1.2 Connection to Velocity Fields

Flow map và velocity field connected by:

$$
\frac{d\phi_t}{dt}(x) = v_t(\phi_t(x))
$$

**Integration:**
$$
\phi_t(x) = x + \int_0^t v_s(\phi_s(x)) ds
$$

## 2. Training Objective

### 2.1 Direct Matching Loss

Learn neural network $\phi_\theta(x, t)$ approximating flow map:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, X_0, X_1}\left[\|\phi_\theta(X_0, t) - X_t\|^2\right]
$$

với $X_t = (1-t)X_0 + tX_1$ (linear interpolation).

**No ODE solving needed during training!**

### 2.2 Consistency Constraints

Enforce composition property:

$$
\mathcal{L}_{\text{cons}}(\theta) = \mathbb{E}_{s,t,x}\left[\|\phi_\theta(\phi_\theta(x, s), t) - \phi_\theta(x, s+t)\|^2\right]
$$

## 3. Implementation

### 3.1 Flow Map Network

```python
import torch
import torch.nn as nn

class FlowMapNet(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Initialize last layer to near-zero
        self.net[-1].weight.data.mul_(0.01)
        self.net[-1].bias.data.zero_()
    
    def forward(self, x, t):
        """
        x: [B, D] - initial state
        t: [B, 1] - time
        Returns: [B, D] - position at time t
        """
        # Enforce identity at t=0
        t_embed = self.time_embed(t)
        inp = torch.cat([x, t_embed], dim=-1)
        
        # Residual: phi_t(x) = x + f_theta(x, t) * t
        delta = self.net(inp)
        return x + delta * t
```

### 3.2 Training

```python
def train_flow_map(model, x0_samples, x1_samples, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        # Sample random times
        t = torch.rand(len(x0_samples), 1)
        
        # Linear interpolation
        xt_target = (1 - t) * x0_samples + t * x1_samples
        
        # Predict flow map
        xt_pred = model(x0_samples, t)
        
        # MSE loss
        loss = ((xt_pred - xt_target) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return model
```

### 3.3 Sampling

```python
def sample_flow_map(model, x0, t=1.0):
    """
    Direct sampling - no ODE solving!
    
    Args:
        model: trained flow map
        x0: initial noise [B, D]
        t: target time (default 1.0 for generation)
    """
    with torch.no_grad():
        t_tensor = torch.ones(x0.shape[0], 1) * t
        x1 = model(x0, t_tensor)
    
    return x1
```

### 3.4 Multi-Step Refinement

```python
def sample_multistep(model, x0, num_steps=10):
    """
    Multi-step sampling for higher quality
    """
    x = x0
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.ones(x.shape[0], 1) * dt
            # Use composition: phi_dt(x) repeatedly
            x = model(x, t)
    
    return x
```

## 4. Advanced Techniques

### 4.1 Flow Map with Consistency Training

```python
def train_with_consistency(model, x0_samples, x1_samples, lambda_cons=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        # Standard matching loss
        t = torch.rand(len(x0_samples), 1)
        xt_target = (1 - t) * x0_samples + t * x1_samples
        xt_pred = model(x0_samples, t)
        loss_match = ((xt_pred - xt_target) ** 2).mean()
        
        # Consistency loss: phi_s(phi_t(x)) ≈ phi_{s+t}(x)
        s = torch.rand(len(x0_samples), 1)
        t2 = torch.rand(len(x0_samples), 1)
        
        xt = model(x0_samples, t2)
        xst = model(xt, s)  # Composition
        xst_direct = model(x0_samples, s + t2)  # Direct
        
        loss_cons = ((xst - xst_direct) ** 2).mean()
        
        # Total loss
        loss = loss_match + lambda_cons * loss_cons
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.2 Conditional Flow Maps

```python
class ConditionalFlowMap(nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim=256):
        super().__init__()
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim + 64 + 64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t, cond):
        """
        x: [B, D] - state
        t: [B, 1] - time
        cond: [B, C] - condition
        """
        t_embed = self.time_embed(t)
        c_embed = self.cond_embed(cond)
        
        inp = torch.cat([x, t_embed, c_embed], dim=-1)
        delta = self.net(inp)
        
        return x + delta * t
```

## 5. Comparison với Other Methods

### 5.1 Flow Map vs Flow Matching

| Aspect | Flow Matching | Flow Map Matching |
|--------|--------------|------------------|
| **Learn** | Velocity $v_t$ | Position $\phi_t$ |
| **Training** | ODE solve (optional) | Direct |
| **Inference** | ODE solve | Direct evaluation |
| **Memory** | Lower | Higher (stores positions) |
| **Flexibility** | High | Very high |

### 5.2 When to Use Flow Maps

**Use Flow Maps when:**
- Need fast, direct evaluation
- Want flexible time discretization
- Training efficiency is priority

**Use Flow Matching when:**
- Memory constrained
- Need continuous-time dynamics
- Physics-informed modeling

## 6. Applications

### 6.1 Image-to-Image Translation

```python
# Train on paired data
source_images = load_source()  # Domain A
target_images = load_target()  # Domain B

model = FlowMapNet(dim=image_dim)
train_flow_map(model, source_images, target_images)

# Translate
translated = sample_flow_map(model, source_test)
```

### 6.2 Interpolation

```python
def interpolate_flow_map(model, x_start, x_end, num_frames=20):
    """
    Smooth interpolation using flow map
    """
    frames = []
    
    # Learn flow from x_start to x_end
    model_local = FlowMapNet(dim=x_start.shape[-1])
    train_flow_map(model_local, x_start, x_end, num_epochs=50)
    
    # Sample intermediate frames
    for t in torch.linspace(0, 1, num_frames):
        t_tensor = torch.ones(x_start.shape[0], 1) * t
        frame = model_local(x_start, t_tensor)
        frames.append(frame)
    
    return torch.stack(frames)
```

### 6.3 Data Augmentation

```python
def augment_with_flow(model, data, num_augments=5):
    """
    Create augmented samples via flow map
    """
    augmented = []
    
    for _ in range(num_augments):
        t = torch.rand(len(data), 1) * 0.5  # Partial flow
        aug_data = model(data, t)
        augmented.append(aug_data)
    
    return torch.cat([data] + augmented)
```

## Kết luận

Flow Map Matching offers compelling advantages:

✅ **Direct training** - no ODE solving needed  
✅ **Fast inference** - single network evaluation  
✅ **Flexible** - arbitrary time discretization  
✅ **Simple** - straightforward regression

Trade-off: potentially higher memory footprint, but gains in speed và simplicity often worth it.

## Tài liệu tham khảo

1. Albergo, M. S., & Vanden-Eijnden, E. (2023). "Building Normalizing Flows with Stochastic Interpolants"
2. Liu, X., et al. (2022). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
3. Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling"

---

**Tags:** #FlowMaps #FlowMatching #GenerativeModels #EfficientTraining #DeepLearning
