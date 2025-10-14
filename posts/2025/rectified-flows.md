# Rectified Flows: Straight Paths in Generative Modeling

**Ngày đăng:** 15/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Flow-based Models, Deep Learning

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Rectification Problem](#rectification-problem)
3. [Reflow Algorithm](#reflow-algorithm)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Implementation](#implementation)
6. [Fast Sampling](#fast-sampling)
7. [Applications](#applications)

---

## Giới thiệu

**Rectified Flows** giải quyết fundamental problem của flow-based models: làm sao để ODE paths từ noise đến data **straight** và **simple** nhất có thể.

**Core insight:** Curved paths = waste computation. Straight paths = efficient sampling.

**Key contributions:**
- **Reflow procedure** - straighten arbitrary transport maps
- **One-step generation** - extremely fast sampling
- **Simple training** - no complex losses
- **Strong theoretical guarantees** - optimal transport connection

## 1. The Rectification Problem

### 1.1 Why Straight Paths Matter

Given two distributions $\pi_0$ (noise) và $\pi_1$ (data), flow models learn ODE:

$$
\frac{dx_t}{dt} = v_t(x_t), \quad x_0 \sim \pi_0, \quad x_1 \sim \pi_1
$$

**Problem:** $v_t$ thường highly curved → nhiều function evaluations cần cho ODE solve.

**Goal:** Find **rectified flow** với straight paths → fewer steps, faster sampling.

### 1.2 Measuring Straightness

**Transport cost:**
$$
\text{Cost} = \mathbb{E}\left[\int_0^1 \|v_t(X_t)\|^2 dt\right]
$$

Lower cost → straighter paths → faster sampling.

**Optimal transport:** Straight line paths minimize transport cost.

## 2. Mathematical Foundation

### 2.1 Marginal-Preserving Transport

**Goal:** Find $X_t$ path satisfying:
- $X_0 \sim \pi_0$ (noise distribution)
- $X_1 \sim \pi_1$ (data distribution)
- Minimize $\mathbb{E}[\int_0^1 \|dX_t/dt\|^2 dt]$

**Optimal solution:** Linear interpolation $X_t = (1-t)X_0 + tX_1$ khi $(X_0, X_1)$ optimally coupled.

### 2.2 Velocity Field

Given path $X_t$, velocity field:

$$
v_t(x) = \mathbb{E}[\dot{X}_t | X_t = x]
$$

**For linear interpolation:**
$$
X_t = (1-t)X_0 + tX_1 \implies \dot{X}_t = X_1 - X_0
$$

$$
v_t(x) = \mathbb{E}[X_1 - X_0 | X_t = x]
$$

### 2.3 Training Objective

Learn neural network $v_\theta$ approximating true velocity:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, X_t}\left[\|v_\theta(X_t, t) - (X_1 - X_0)\|^2\right]
$$

với $t \sim U[0,1]$, $X_0 \sim \pi_0$, $X_1 \sim \pi_1$, $X_t = (1-t)X_0 + tX_1$.

## 3. Reflow Algorithm

### 3.1 Core Idea

**Problem:** Initial coupling $(X_0, X_1)$ may not be optimal → paths not straight.

**Solution:** Iteratively "reflow" - straighten paths by:
1. Generate samples với current model
2. Use them to create better coupling
3. Train new model on straightened paths

### 3.2 Reflow Procedure

**Input:** Pretrained model $v_\theta^{(k)}$  
**Output:** Improved model $v_\theta^{(k+1)}$

```python
# Algorithm: Reflow
for k in iterations:
    # 1. Generate data with current model
    X0 ~ pi_0  # Sample noise
    X1 = ODE_solve(X0, v_theta_k)  # Generate via ODE
    
    # 2. Create linear coupling
    Xt = (1-t)*X0 + t*X1  # Linear interpolation
    
    # 3. Train new model
    v_theta_{k+1} = train(Xt, target=X1-X0)
```

**Key insight:** Mỗi reflow iteration, paths become straighter!

### 3.3 Convergence

**Theorem:** After $k$ reflow iterations, transport cost:

$$
\text{Cost}^{(k)} \leq \left(\frac{1}{2}\right)^k \text{Cost}^{(0)}
$$

Exponentially decreasing! Few iterations achieve near-straight paths.

## 4. Implementation với PyTorch

### 4.1 Dataset with Coupling

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class RectifiedFlowDataset(torch.utils.data.Dataset):
    def __init__(self, x0_samples, x1_samples):
        """
        x0_samples: noise samples [N, D]
        x1_samples: data samples [N, D]
        """
        self.x0 = x0_samples
        self.x1 = x1_samples
        assert len(x0_samples) == len(x1_samples)
    
    def __len__(self):
        return len(self.x0)
    
    def __getitem__(self, idx):
        t = torch.rand(1).item()  # Random time
        x0 = self.x0[idx]
        x1 = self.x1[idx]
        
        # Linear interpolation
        xt = (1 - t) * x0 + t * x1
        
        # Target: constant velocity
        target = x1 - x0
        
        return xt, torch.tensor([t]), target
```

### 4.2 Velocity Network

```python
class VelocityNet(nn.Module):
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        """
        x: [B, D] - state
        t: [B, 1] - time
        """
        t_embed = self.time_mlp(t)
        inp = torch.cat([x, t_embed], dim=-1)
        return self.net(inp)
```

### 4.3 Training Loop

```python
def train_rectified_flow(model, dataset, num_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for xt, t, target in dataloader:
            # Predict velocity
            v_pred = model(xt, t)
            
            # MSE loss
            loss = ((v_pred - target) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
    
    return model
```

### 4.4 Sampling (ODE Solve)

```python
def sample_rectified_flow(model, x0, num_steps=10):
    """
    Generate samples using ODE solver
    
    Args:
        model: trained velocity network
        x0: initial noise [B, D]
        num_steps: number of discretization steps
    """
    def ode_func(t, x):
        t_batch = torch.ones(x.shape[0], 1, device=x.device) * t
        return model(x, t_batch)
    
    # Solve ODE from t=0 to t=1
    t_span = torch.linspace(0, 1, num_steps + 1)
    
    with torch.no_grad():
        trajectory = odeint(
            ode_func, 
            x0, 
            t_span, 
            method='euler'  # Can use 'rk4', 'dopri5', etc.
        )
    
    return trajectory[-1]  # Return x1
```

### 4.5 Reflow Procedure

```python
def reflow(model_k, data_loader, num_samples=10000):
    """
    Perform one reflow iteration
    
    Args:
        model_k: current trained model
        data_loader: data distribution
        num_samples: number of samples to generate
    
    Returns:
        New dataset with straightened paths
    """
    x0_samples = []
    x1_samples = []
    
    with torch.no_grad():
        for _ in range(num_samples // 256):
            # Sample noise
            x0 = torch.randn(256, model_k.dim)
            
            # Generate using current model
            x1 = sample_rectified_flow(model_k, x0, num_steps=10)
            
            x0_samples.append(x0)
            x1_samples.append(x1)
    
    x0_samples = torch.cat(x0_samples, dim=0)
    x1_samples = torch.cat(x1_samples, dim=0)
    
    # Create new dataset with linear coupling
    new_dataset = RectifiedFlowDataset(x0_samples, x1_samples)
    
    return new_dataset

# Full reflow training
def train_with_reflow(data, num_reflows=3):
    # Initial training
    x0 = torch.randn(len(data), data.shape[1])
    x1 = data
    dataset = RectifiedFlowDataset(x0, x1)
    
    model = VelocityNet(dim=data.shape[1])
    model = train_rectified_flow(model, dataset)
    
    # Reflow iterations
    for k in range(num_reflows):
        print(f"\n=== Reflow iteration {k+1} ===")
        dataset = reflow(model, data)
        model = VelocityNet(dim=data.shape[1])
        model = train_rectified_flow(model, dataset)
    
    return model
```

## 5. Fast Sampling

### 5.1 One-Step Generation

After sufficient reflows, paths are nearly straight → **one Euler step** suffices!

```python
def one_step_sample(model, x0):
    """
    Ultra-fast sampling with single step
    """
    t = torch.ones(x0.shape[0], 1) * 0.5  # Mid-point
    v = model(x0, t)
    
    # Single Euler step from t=0 to t=1
    x1 = x0 + v
    
    return x1
```

**Speed:** ~1000x faster than diffusion models!

### 5.2 Few-Step Generation

For higher quality, use few steps:

```python
def few_step_sample(model, x0, num_steps=5):
    """
    Fast sampling with few steps
    """
    dt = 1.0 / num_steps
    x = x0
    
    for i in range(num_steps):
        t = torch.ones(x.shape[0], 1) * (i * dt)
        v = model(x, t)
        x = x + v * dt
    
    return x
```

## 6. Comparison với Diffusion Models

| Aspect | Diffusion Models | Rectified Flows |
|--------|-----------------|-----------------|
| **Sampling** | 50-1000 steps | 1-10 steps |
| **Training** | Complex (score matching) | Simple (velocity matching) |
| **Paths** | Curved, noisy | Straight, deterministic |
| **Speed** | Slow | Fast |
| **Quality** | State-of-art | Competitive |

## 7. Applications

### 7.1 Image Generation

```python
# Train on images
images = load_images()  # [N, C, H, W]
images_flat = images.view(len(images), -1)

model = train_with_reflow(images_flat, num_reflows=2)

# Generate new images
noise = torch.randn(16, images_flat.shape[1])
generated = one_step_sample(model, noise)
generated_images = generated.view(16, C, H, W)
```

### 7.2 Domain Translation

```python
# Learn mapping from domain A to domain B
dataset = RectifiedFlowDataset(x0=domain_A, x1=domain_B)
model = train_rectified_flow(model, dataset)

# Translate
translated = sample_rectified_flow(model, domain_A_test, num_steps=5)
```

### 7.3 Data Interpolation

```python
def interpolate(model, x0, x1, num_frames=10):
    """
    Smooth interpolation between two points
    """
    frames = []
    
    for t in torch.linspace(0, 1, num_frames):
        # Follow the rectified flow path
        xt = (1 - t) * x0 + t * x1
        frames.append(xt)
    
    return torch.stack(frames)
```

## 8. Theoretical Insights

### 8.1 Connection to Optimal Transport

Rectified flows approximate **optimal transport** maps:

$$
\min_{\gamma \in \Pi(\pi_0, \pi_1)} \int \|x_1 - x_0\|^2 d\gamma(x_0, x_1)
$$

Straight line paths = Wasserstein-2 geodesics.

### 8.2 Stochastic Interpolants

Rectified flows special case of **stochastic interpolants** framework:

$$
I_t(x_0, x_1) = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) \epsilon
$$

Rectified flows: $\alpha(t) = 1-t$, $\beta(t) = t$, $\gamma(t) = 0$ (deterministic).

## Kết luận

Rectified Flows achieve remarkable simplicity và efficiency:

✅ **Simple training** - straightforward regression loss  
✅ **Fast sampling** - 1-10 steps sufficient  
✅ **Strong theory** - optimal transport connection  
✅ **Flexible** - works for various modalities

Key insight: **Straightness matters!** Iterative refinement (reflow) produces extremely efficient generative models.

## Tài liệu tham khảo

1. Liu, X., et al. (2022). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" - ICLR 2023
2. Liu, X., et al. (2023). "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"
3. Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling"

---

**Tags:** #RectifiedFlows #OptimalTransport #FastSampling #GenerativeModels #DeepLearning

<script src="/assets/js/katex-init.js"></script>
