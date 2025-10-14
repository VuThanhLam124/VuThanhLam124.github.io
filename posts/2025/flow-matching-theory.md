---
title: "Flow Matching: Từ lý thuyết đến thực hành với PyTorch"
date: "2025-10-13"
category: "Flow Matching"
tags: ["Flow Matching", "PyTorch", "Generative AI", "Theory"]
excerpt: "Deep dive vào Flow Matching theory, so sánh với Score-based Diffusion Models, và complete implementation từ scratch."
author: "ThanhLamDev"
readingTime: 15
featured: true
image: "/assets/images/flow-matching.jpg"
---

# Flow Matching: Từ lý thuyết đến thực hành với PyTorch

Flow Matching đang trở thành một trong những phương pháp generative modeling mạnh mẽ nhất hiện nay, với khả năng tạo ra dữ liệu chất lượng cao và training stability vượt trội so với các methods truyền thống.

## Giới thiệu về Flow Matching

**Flow Matching** là một framework để training continuous normalizing flows bằng cách học một vector field mà transport một distribution đơn giản (như Gaussian) tới target data distribution. Khác với Score-based Diffusion Models, Flow Matching không cần thêm noise vào data mà trực tiếp học optimal transport path.

### Ưu điểm chính của Flow Matching:

- **Training stability cao hơn** so với GANs và VAEs
- **Sampling efficiency** tốt hơn Diffusion Models
- **Theoretical foundation** vững chắc dựa trên optimal transport
- **Flexible architectures** có thể adapt cho nhiều data types

## Lý thuyết toán học

### Continuous Normalizing Flows

Flow Matching dựa trên ý tưởng của continuous normalizing flows, được định nghĩa bởi ODE:

```
dx/dt = v_θ(x, t)
```

Trong đó:
- `x(t)` là trajectory của data point
- `v_θ(x, t)` là learned vector field
- `t ∈ [0, 1]` là time parameter

### Flow Matching Objective

Objective function của Flow Matching có dạng:

```
L_FM(θ) = E_{t,x(t)} [||v_θ(x(t), t) - u(x(t), t)||²]
```

Trong đó:
- `u(x(t), t)` là target vector field
- `x(t)` được sample từ marginal distribution tại thời điểm t

## Implementation với PyTorch

### Bước 1: Định nghĩa Vector Field Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorField(nn.Module):
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
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
        # Time embedding
        t_embed = self.time_embed(t.unsqueeze(-1))
        
        # Concatenate x and time embedding
        input_vec = torch.cat([x, t_embed], dim=-1)
        
        return self.net(input_vec)
```

### Bước 2: Flow Matching Loss

```python
class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, model, x0, x1, t):
        """
        x0: source distribution (e.g., Gaussian noise)
        x1: target distribution (real data)
        t: time steps
        """
        # Linear interpolation path
        x_t = (1 - t) * x0 + t * x1
        
        # Target vector field (derivative of interpolation)
        u_t = x1 - x0
        
        # Predicted vector field
        v_t = model(x_t, t)
        
        # Flow matching loss
        loss = F.mse_loss(v_t, u_t)
        return loss
```

### Bước 3: Training Loop

```python
def train_flow_matching(model, dataloader, num_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = FlowMatchingLoss()
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (real_data,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            batch_size = real_data.size(0)
            
            # Sample source noise
            x0 = torch.randn_like(real_data)
            x1 = real_data
            
            # Sample time steps
            t = torch.rand(batch_size, 1)
            
            # Compute loss
            loss = loss_fn(model, x0, x1, t.squeeze())
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
```

### Bước 4: Sampling Process

```python
def sample_flow_matching(model, num_samples, dim, num_steps=100):
    """
    Generate samples using trained Flow Matching model
    """
    model.eval()
    
    with torch.no_grad():
        # Start from Gaussian noise
        x = torch.randn(num_samples, dim)
        
        dt = 1.0 / num_steps
        
        # Solve ODE using Euler method
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt)
            
            # Get vector field
            v = model(x, t)
            
            # Update x
            x = x + v * dt
    
    return x

# Advanced sampling with adaptive step size
def sample_adaptive(model, num_samples, dim, tol=1e-5):
    from scipy.integrate import solve_ivp
    
    def ode_func(t, x):
        x_tensor = torch.from_numpy(x.reshape(num_samples, -1)).float()
        t_tensor = torch.full((num_samples,), t)
        
        with torch.no_grad():
            v = model(x_tensor, t_tensor)
        
        return v.numpy().flatten()
    
    # Initial condition
    x0 = torch.randn(num_samples, dim).numpy().flatten()
    
    # Solve ODE
    sol = solve_ivp(ode_func, [0, 1], x0, rtol=tol, atol=tol)
    
    return torch.from_numpy(sol.y[:, -1].reshape(num_samples, dim))
```

## So sánh với Diffusion Models

| Aspect | Flow Matching | DDPM |
|--------|---------------|------|
| Training Process | Direct vector field learning | Multi-step denoising |
| Sampling Speed | Fast (single ODE solve) | Slow (many denoising steps) |
| Theoretical Foundation | Optimal transport | Score matching |
| Training Stability | High | Moderate |
| Memory Usage | Lower | Higher |

## Ứng dụng thực tế

### 1. Image Generation

```python
# Example for 2D image generation
class ImageFlowMatching(nn.Module):
    def __init__(self, channels=3, resolution=32):
        super().__init__()
        self.vector_field = VectorField(
            dim=channels * resolution * resolution,
            hidden_dim=512
        )
    
    def forward(self, x, t):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        v_flat = self.vector_field(x_flat, t)
        return v_flat.view_as(x)
```

### 2. Text Generation

```python
# Continuous text representation
class TextFlowMatching(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.vector_field = VectorField(embed_dim)
    
    def forward(self, x_embed, t):
        return self.vector_field(x_embed, t)
```

## Kết luận

Flow Matching mang lại một perspective mới cho generative modeling với những ưu điểm vượt trội về tốc độ sampling và training stability. Việc implementation với PyTorch tương đối straightforward và có thể được extend cho nhiều applications khác nhau.

### Next Steps:

1. **Rectified Flow**: Cải thiện straight-line paths
2. **Conditional Flow Matching**: Thêm conditioning information
3. **Multi-scale architectures**: Cho high-resolution generation
4. **Evaluation metrics**: So sánh với state-of-the-art methods

---

**Tài liệu tham khảo:**

- Lipman et al. "Flow Matching for Generative Modeling" (2023)
- Liu et al. "Rectified Flow: A Marginal Preserving Approach to Optimal Transport" (2023)
- Tong et al. "Improving and Generalizing Flow-Based Generative Models" (2023)

**Code repository:** [GitHub - Flow Matching Implementation](https://github.com/VuThanhLam124/flow-matching-pytorch)
<script src="/assets/js/katex-init.js"></script>
