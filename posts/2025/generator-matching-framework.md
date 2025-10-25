---
title: "Generator Matching Framework: Thống Nhất Lý Thuyết"
date: "2025-01-24"
category: "flow-based-models"
tags: ["generator-matching", "flow-matching", "diffusion-models", "score-matching", "theory", "framework"]
excerpt: "Ngày thứ 13, người thợ gốm ngồi lại nhìn toàn bộ hành trình 12 ngày qua. Generator Matching Framework giúp anh thấy rằng tất cả các kỹ thuật đều là những cách khác nhau để 'khớp generator' - một bản đồ lý thuyết thống nhất."
author: "ThanhLamDev"
readingTime: 22
featured: true
---

# Generator Matching Framework: Thống Nhất Lý Thuyết

**Người Thợ Gốm Và Bản Đồ Tổng Hợp**

Sau 12 ngày học hỏi từ [CNF](/posts/2025/ffjord-continuous-flows) đến [Flow Map Matching](/posts/2025/flow-map-matching), người thợ gốm đã thành thạo nhiều kỹ thuật. Nhưng một câu hỏi lớn vẫn còn: **Liệu tất cả chúng có điểm chung nào không?**

## Mục lục

1. [Ngày thứ 13 - Buổi tổng kết](#1-ngày-thứ-13---buổi-tổng-kết)
2. [Generator là gì?](#2-generator-là-gì)
3. [Generator Matching Framework](#3-generator-matching-framework)
4. [So sánh các phương pháp trong GMF](#4-so-sánh-các-phương-pháp-trong-gmf)
5. [Implementation: Unified Matcher](#5-implementation-unified-matcher)
6. [Ứng dụng và nghiên cứu](#6-ứng-dụng-và-nghiên-cứu)
7. [Kết luận](#7-kết-luận)

---

## 1. Ngày thứ 13 - Buổi tổng kết

### Buổi sáng nhìn lại

Sáng ngày thứ 13, người thợ gốm ngồi trong xưởng, trải toàn bộ sổ tay ra bàn:

```
┌─────────────────────────────────────────────┐
│ 12 ngày qua tôi đã học:                     │
├─────────────────────────────────────────────┤
│ Ngày 1-2:   CNF - Likelihood training       │
│ Ngày 3:     Flow Matching - Regression      │
│ Ngày 4-7:   Rectified Flow - Straight paths │
│ Ngày 8-10:  Schrödinger Bridge - Noise      │
│ Ngày 11-12: Flow Map - Direct mapping       │
└─────────────────────────────────────────────┘
```

"Mỗi kỹ thuật giải quyết một vấn đề khác nhau," anh tự nhủ. "Nhưng tất cả đều làm cùng một việc: **Biến đổi từ noise thành data**."

Anh vẽ sơ đồ lớn:

```
Tất cả đều là: noise → [SOMETHING] → data

- CNF:     [ODE with likelihood]
- FM:      [Velocity field]
- RF:      [Straight velocity]
- SB:      [Optimal stochastic control]
- FlowMap: [Direct map]

Vậy [SOMETHING] là gì?
```

### Phát hiện quan trọng

Anh nhớ lại một bài báo mới đọc: **"Generator Matching for Generative Modeling"** (Tong et al., 2024).

**Ý tưởng cốt lõi:**

> Tất cả các phương pháp đều đang học một **"generator"** $G_t$ - quy tắc biến đổi noise thành data theo thời gian. Khác nhau chỉ là:
> 1. **Parametrization** của $G_t$ (velocity? score? map?)
> 2. **Loss function** để match với target generator

"Đây chính là bản đồ thống nhất tôi cần!" Anh hào hứng ghi chép.

## 2. Generator là gì?

### Định nghĩa trực quan

**Generator $G_t$:** Một quy tắc (có thể là hàm, toán tử, hoặc quá trình) biến đổi noise $Z$ thành sample tại thời điểm $t$.

$$
X_t = G_t(Z), \quad Z \sim p_Z
$$

**Mục tiêu cuối cùng:**

$$
X_1 = G_1(Z) \sim p_{\text{data}}
$$

### Các cách parametrize generator

Người thợ gốm vẽ bảng so sánh:

| Method | Generator $G_t$ | How it's defined |
|--------|-----------------|------------------|
| **CNF** | Solution of ODE | $\frac{dG_t}{dt} = v_\theta(G_t, t)$, $G_0(z) = z$ |
| **Flow Matching** | Same as CNF | Learn $v_\theta$ via regression |
| **Rectified Flow** | Same as FM | But $v_\theta$ aims for straight paths |
| **Diffusion** | Solution of SDE | $dG_t = f(G_t, t)dt + g(t)dW_t$ |
| **Score Matching** | SDE with score | $dG_t = \sigma^2 s_\theta(G_t, t)dt + \sigma dW_t$ |
| **Schrödinger Bridge** | Optimal SDE | Forward-backward SDE with IPF |
| **Flow Map** | Direct function | $G_t(z) = \phi_\theta(z, t)$ (no differential eq) |
| **GAN** | Static map | $G(z) = G_\theta(z)$ (no time) |

**Nhận xét:**

- **ODE-based:** CNF, FM, RF → Deterministic generator
- **SDE-based:** Diffusion, Score, SB → Stochastic generator  
- **Direct:** Flow Map → Explicit function
- **Static:** GAN → No time dimension

### Ví dụ cụ thể

**Ví dụ 1: Flow Matching**

```python
# Generator defined by ODE
def generator_FM(z, t, v_theta):
    # Solve: dx/dt = v_theta(x, t), x(0) = z
    x = ode_solve(v_theta, z, t_end=t)
    return x
```

**Ví dụ 2: Diffusion**

```python
# Generator defined by SDE
def generator_diffusion(z, t, score_theta, sigma):
    # Solve: dx = sigma^2 * score(x,t) dt + sigma dW
    x = sde_solve(score_theta, z, t_end=t, sigma=sigma)
    return x
```

**Ví dụ 3: Flow Map**

```python
# Generator is direct function
def generator_flowmap(z, t, phi_theta):
    # Direct mapping
    x = phi_theta(z, t)
    return x
```

## 3. Generator Matching Framework

### 3.1. Khung toán học tổng quát

Anh viết công thức chung nhất:

**Bài toán Generator Matching:**

$$
\min_{\theta} \mathbb{E}_{Z, t} \left[ D\big( G_\theta(Z, t), G^*(Z, t) \big) \right]
$$

**Giải thích:**
- $G_\theta$: Learned generator (với tham số $\theta$)
- $G^*$: Target generator (optimal hoặc từ data)
- $D(\cdot, \cdot)$: Distance metric
- Tối ưu $\theta$ để $G_\theta \approx G^*$

### 3.2. Các lựa chọn quantity để match

Thay vì match trực tiếp $G_t$, ta thường match một **quantity liên quan**:

$$
\min_{\theta} \mathbb{E} \left[ D\big( \mathcal{Q}_\theta, \mathcal{Q}^* \big) \right]
$$

Các lựa chọn $\mathcal{Q}$:

| Quantity $\mathcal{Q}$ | Phương pháp | Distance $D$ |
|------------------------|-------------|--------------|
| **Velocity** $v_t$ | Flow Matching | $L^2$ (MSE) |
| **Noise** $\epsilon$ | DDPM Diffusion | $L^2$ |
| **Score** $\nabla \log p_t$ | Score Matching | Fisher div |
| **Position** $\phi_t$ | Flow Map | $L^2$ |
| **Sample** $G(z)$ | GAN | JS / Wasserstein |

**Ưu điểm approach này:**

1. **Dễ tính toán:** Match quantity thay vì match toàn bộ generator
2. **Flexible:** Có thể chọn quantity phù hợp với bài toán
3. **Unified:** Tất cả methods đều fit vào framework

### 3.3. Flow Matching trong GMF

**Target generator (conditional):**

$$
G^*_t(z | x_1) = (1-t)z + t x_1
$$

**Conditional velocity:**

$$
v^*_t(x | x_1) = x_1 - x_0 = \frac{d}{dt}G^*_t
$$

**Marginal velocity (CFM):**

$$
v^*_t(x) = \mathbb{E}_{x_1 | x}[v^*_t(x | x_1)]
$$

**Loss:**

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{x_0, x_1, t} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

### 3.4. Schrödinger Bridge trong GMF

**Target generator:** Optimal stochastic bridge

$$
G^*: \arg\min_{G} \text{KL}(G \mid\mid G_{\text{ref}})
$$

với điều kiện $G_0 \sim p_0, G_1 \sim p_1$

**Quantity to match:** Forward + backward scores

$$
\mathcal{Q}^* = (s^f_t, s^b_t)
$$

**Loss:** IPF with score matching

### 3.5. Flow Map trong GMF

**Target generator:**

$$
G^*_t(z) = (1-t)z + t G^*_1(z)
$$

**Quantity:** Position map

$$
\mathcal{Q}_\theta = \phi_\theta(z, t)
$$

**Loss:**

$$
\mathcal{L}_{\text{map}}(\theta) = \mathbb{E}_{z, x_1, t} \left[ \| \phi_\theta(z, t) - [(1-t)z + tx_1] \|^2 \right]
$$

## 4. So sánh các phương pháp trong GMF

### Bảng tổng hợp toàn diện

| Method | Type | Quantity $\mathcal{Q}$ | Loss | Sampling | Speed | Diversity |
|--------|------|------------------------|------|----------|-------|-----------|
| **CNF** | ODE | Likelihood | NLL | ODE 50+ | Slow | Low |
| **Flow Matching** | ODE | Velocity | MSE | ODE 50+ | Slow | Low |
| **Rectified Flow** | ODE | Straight vel | MSE | ODE 1-5 | Fast | Low |
| **DDPM** | SDE | Noise | MSE | SDE 1000 | Very Slow | High |
| **Score Matching** | SDE | Score | DSM | SDE 1000 | Very Slow | High |
| **Schrödinger Bridge** | SDE | 2 scores | IPF | SDE 100 | Medium | Medium-High |
| **Flow Map** | Direct | Position | MSE | 1-step | Very Fast | Low |
| **GAN** | Static | Sample | Adversarial | 1-step | Very Fast | Medium |

### Trục tọa độ so sánh

Người thợ gốm vẽ biểu đồ 2D:

```
                    High Diversity
                          │
                          │ Diffusion, Score
                          │
                          │   SB
           GAN ───────────┼───────────────
                          │
                          │
    Slow ─────────────────┼──────────────── Fast
                          │
                          │    RF
                FM, CNF   │        FlowMap
                          │
                    Low Diversity
```

**Nhận xét:**

- **Top-right (Diffusion):** High quality + diversity, but slow
- **Bottom-right (FlowMap, RF):** Fast but low diversity
- **Center (SB):** Balance tốt
- **Left (FM, CNF):** Slow + low diversity (ít dùng)

### Timeline 14 ngày nhìn lại

```
┌──────────────────────────────────────────────────────┐
│ HÀNH TRÌNH 14 NGÀY CỦA NGƯỜI THỢ GỐM                │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Day 1-2:  CNF - Khám phá ODE, likelihood (slow)     │
│           Vấn đề: 41 ngày/100 rồng                  │
│                                                      │
│ Day 3:    Flow Matching - Regression breakthrough    │
│           Giải pháp: MSE thay vì NLL → 15 ngày      │
│                                                      │
│ Day 4-7:  Rectified Flow - Straight paths            │
│           Reflow algorithm → 7 ngày (đúng deadline!) │
│           One-step generation                        │
│                                                      │
│ Day 8-10: Schrödinger Bridge - Handling noise       │
│           IPF, optimal control trong môi trường gió │
│           50 con rồng với "natural randomness"      │
│                                                      │
│ Day 11-12: Flow Map - Direct mapping                │
│            Preset templates, ultra-fast (0.08s)     │
│            Không cần ODE solver                     │
│                                                      │
│ Day 13-14: Generator Matching Framework (GMF)       │
│            Thống nhất tất cả - bản đồ lý thuyết    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## 5. Implementation: Unified Matcher

### 5.1. Abstract Base Class

```python
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class GeneratorMatcher(nn.Module, ABC):
    """
    Abstract base class for all generator matching methods
    """
    
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def compute_quantity(self, x, t):
        """
        Compute the quantity to match (velocity, score, position, etc.)
        
        Args:
            x: Current state (batch, dim)
            t: Time (batch,)
        
        Returns:
            quantity: The quantity to match
        """
        pass
    
    @abstractmethod
    def compute_loss(self, x0, x1, t):
        """
        Compute matching loss
        
        Args:
            x0: Source samples (batch, dim)
            x1: Target samples (batch, dim)
            t: Time (batch,)
        
        Returns:
            loss: Scalar loss value
        """
        pass
    
    @abstractmethod
    def sample(self, z, num_steps=None):
        """
        Generate samples from noise
        
        Args:
            z: Noise (batch, dim)
            num_steps: Number of steps (if applicable)
        
        Returns:
            samples: Generated samples (batch, dim)
        """
        pass
```

### 5.2. Flow Matching Implementation

```python
class FlowMatcher(GeneratorMatcher):
    """Flow Matching: Match velocity field"""
    
    def __init__(self, dim, hidden_dim=256):
        super().__init__(dim, hidden_dim)
        
        # Velocity network
        self.velocity_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def compute_quantity(self, x, t):
        """Compute velocity"""
        t_expanded = t.unsqueeze(-1)
        inp = torch.cat([x, t_expanded], dim=-1)
        return self.velocity_net(inp)
    
    def compute_loss(self, x0, x1, t):
        """Flow matching loss"""
        # Interpolate
        t_expanded = t.unsqueeze(-1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Target velocity
        v_target = x1 - x0
        
        # Predicted velocity
        v_pred = self.compute_quantity(x_t, t)
        
        # MSE loss
        loss = ((v_pred - v_target) ** 2).mean()
        return loss
    
    def sample(self, z, num_steps=50):
        """Sample using Euler method"""
        x = z
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((x.shape[0],), i * dt, device=x.device)
                v = self.compute_quantity(x, t)
                x = x + v * dt
        
        return x
```

### 5.3. Score Matching Implementation

```python
class ScoreMatcher(GeneratorMatcher):
    """Score Matching: Match score function"""
    
    def __init__(self, dim, hidden_dim=256, sigma=0.1):
        super().__init__(dim, hidden_dim)
        self.sigma = sigma
        
        # Score network
        self.score_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def compute_quantity(self, x, t):
        """Compute score"""
        t_expanded = t.unsqueeze(-1)
        inp = torch.cat([x, t_expanded], dim=-1)
        return self.score_net(inp)
    
    def compute_loss(self, x0, x1, t):
        """Denoising score matching loss"""
        # Interpolate
        t_expanded = t.unsqueeze(-1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Add noise
        noise = torch.randn_like(x_t)
        x_noisy = x_t + self.sigma * noise
        
        # Predict score
        score_pred = self.compute_quantity(x_noisy, t)
        
        # True score (denoising)
        score_true = -noise / self.sigma
        
        # MSE loss
        loss = ((score_pred - score_true) ** 2).mean()
        return loss
    
    def sample(self, z, num_steps=100):
        """Sample using Euler-Maruyama"""
        import math
        x = z
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.full((x.shape[0],), i * dt, device=x.device)
                score = self.compute_quantity(x, t)
                drift = self.sigma**2 * score
                noise = torch.randn_like(x) * math.sqrt(dt)
                x = x + drift * dt + self.sigma * noise
        
        return x
```

### 5.4. Flow Map Implementation

```python
class FlowMapMatcher(GeneratorMatcher):
    """Flow Map Matching: Match position directly"""
    
    def __init__(self, dim, hidden_dim=256):
        super().__init__(dim, hidden_dim)
        
        # Flow map network
        self.map_net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def compute_quantity(self, x0, t):
        """Compute position at time t"""
        t_expanded = t.unsqueeze(-1)
        inp = torch.cat([x0, t_expanded], dim=-1)
        residual = self.map_net(inp)
        return x0 + residual  # Enforce φ(x0, 0) = x0
    
    def compute_loss(self, x0, x1, t):
        """Flow map matching loss"""
        # Target position
        t_expanded = t.unsqueeze(-1)
        x_t_target = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Predicted position
        x_t_pred = self.compute_quantity(x0, t)
        
        # MSE loss
        loss = ((x_t_pred - x_t_target) ** 2).mean()
        return loss
    
    def sample(self, z, num_steps=None):
        """Sample using direct map (1-step)"""
        with torch.no_grad():
            t = torch.ones(z.shape[0], device=z.device)
            x = self.compute_quantity(z, t)
        return x
```

### 5.5. Unified Training Loop

```python
def train_matcher(matcher, data_loader, epochs=100, device='cuda'):
    """
    Unified training loop for all matchers
    """
    matcher = matcher.to(device)
    optimizer = torch.optim.AdamW(matcher.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    matcher.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for x0, x1 in data_loader:
            x0, x1 = x0.to(device), x1.to(device)
            batch_size = x0.shape[0]
            
            # Sample time
            t = torch.rand(batch_size, device=device)
            
            # Compute loss (method-specific)
            loss = matcher.compute_loss(x0, x1, t)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(data_loader)
            print(f"[{matcher.__class__.__name__}] "
                  f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    return matcher
```

### 5.6. Usage Example

```python
from torch.utils.data import DataLoader, TensorDataset

# Generate data
p0 = torch.randn(5000, 2)
p1 = torch.randn(5000, 2) + 3  # Shifted Gaussian

dataset = TensorDataset(p0, p1)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Train different matchers
print("="*60)
print("Training Flow Matching...")
fm_matcher = FlowMatcher(dim=2)
fm_matcher = train_matcher(fm_matcher, loader, epochs=50)

print("\n" + "="*60)
print("Training Score Matching...")
sm_matcher = ScoreMatcher(dim=2, sigma=0.1)
sm_matcher = train_matcher(sm_matcher, loader, epochs=50)

print("\n" + "="*60)
print("Training Flow Map...")
map_matcher = FlowMapMatcher(dim=2)
map_matcher = train_matcher(map_matcher, loader, epochs=50)

# Sample and compare
z = torch.randn(1000, 2)

samples_fm = fm_matcher.sample(z, num_steps=50)
samples_sm = sm_matcher.sample(z, num_steps=100)
samples_map = map_matcher.sample(z)

print("\n" + "="*60)
print("Sampling complete!")
print(f"FM mean: {samples_fm.mean(0)}")
print(f"SM mean: {samples_sm.mean(0)}")
print(f"Map mean: {samples_map.mean(0)}")
print(f"Target mean: {p1.mean(0)}")
```

## 6. Ứng dụng và nghiên cứu

### 6.1. Lai ghép các phương pháp

**Ví dụ: Flow Matching + Score Matching**

```python
class HybridMatcher(GeneratorMatcher):
    """Combine Flow and Score matching"""
    
    def __init__(self, dim):
        super().__init__(dim)
        self.flow_matcher = FlowMatcher(dim)
        self.score_matcher = ScoreMatcher(dim)
        self.alpha = 0.5  # Balance weight
    
    def compute_loss(self, x0, x1, t):
        loss_flow = self.flow_matcher.compute_loss(x0, x1, t)
        loss_score = self.score_matcher.compute_loss(x0, x1, t)
        return self.alpha * loss_flow + (1 - self.alpha) * loss_score
    
    def sample(self, z, num_steps=50):
        # Use flow for fast initial transport
        x_mid = self.flow_matcher.sample(z, num_steps=10)
        # Use score for refinement
        x_final = self.score_matcher.sample(x_mid, num_steps=40)
        return x_final
```

### 6.2. Conditional Generation

**Ví dụ: Class-conditional FlowMatcher**

```python
class ConditionalFlowMatcher(FlowMatcher):
    """Flow Matching with class conditioning"""
    
    def __init__(self, dim, num_classes):
        super().__init__(dim)
        self.class_embed = nn.Embedding(num_classes, dim)
    
    def compute_quantity(self, x, t, class_label):
        # Inject class information
        class_feat = self.class_embed(class_label)
        x_cond = x + class_feat
        return super().compute_quantity(x_cond, t)
```

### 6.3. Hướng nghiên cứu mới

Người thợ gốm ghi chú các hướng có thể khám phá:

1. **Adaptive matching:** Học cách chọn phương pháp phù hợp cho từng sample
2. **Multi-scale GMF:** Matching ở nhiều resolution khác nhau
3. **Continuous-time GMF:** Mở rộng cho infinite time horizon
4. **Constrained GMF:** Thêm physics constraints vào framework

## 7. Kết luận

### Ngày thứ 14 - Hoàn thành hành trình

Chiều ngày thứ 14, người thợ gốm đóng sổ tay lại, mỉm cười.

"Hai tuần qua, tôi đã đi từ **không biết gì** về Generative Models đến **hiểu rõ** toàn bộ landscape!"

### Bản đồ cuối cùng

Anh vẽ bản đồ tổng kết:

```
┌────────────────────────────────────────────────────┐
│      GENERATOR MATCHING FRAMEWORK (GMF)            │
│                                                    │
│  Tất cả methods đều học: noise → data             │
│  Khác nhau ở:                                     │
│    1. Parametrization của generator               │
│    2. Quantity được match                         │
│    3. Loss function                               │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  ODE-based:                              │    │
│  │    CNF, Flow Matching, Rectified Flow    │    │
│  │    → Deterministic, Fast (with reflow)   │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  SDE-based:                              │    │
│  │    Diffusion, Score, Schrödinger Bridge  │    │
│  │    → Stochastic, High diversity          │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  Direct:                                 │    │
│  │    Flow Map, GAN                         │    │
│  │    → Ultra-fast, Low flexibility         │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Bài học quan trọng nhất

Anh viết lời kết:

> **Generator Matching Framework cho tôi thấy:**
>
> 1. **Unified view:** Tất cả methods đều match generators
> 2. **Trade-offs:** Không có "best" method, chỉ có "phù hợp nhất"
> 3. **Flexibility:** Có thể lai ghép các approaches
> 4. **Foundation:** Hiểu GMF là hiểu toàn bộ generative modeling

### Lời cảm ơn đến hành trình

"Từ một người thợ gốm không biết gì về AI, giờ tôi có thể:
- Hiểu sâu lý thuyết từ CNF đến SB
- Implement từ scratch bằng PyTorch
- So sánh và lựa chọn phương pháp phù hợp
- Thậm chí nghĩ ra các hybrid approaches mới!"

**Kết thúc hành trình 14 ngày, bắt đầu những khám phá mới!**

---

## Tài liệu tham khảo

1. **Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2024)** - "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" _(Generator Matching perspective)_

2. **Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023)** - "Flow Matching for Generative Modeling" _(Flow Matching)_

3. **Liu, X., Gong, C., & Liu, Q. (2022)** - "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" _(Rectified Flow)_

4. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021)** - "Score-Based Generative Modeling through Stochastic Differential Equations" _(Score SDE)_

5. **De Bortoli, V., Thornton, J., Heng, J., & Doucet, A. (2021)** - "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Models" _(Schrödinger Bridge)_

6. **Albergo, M. S., Boffi, N. M., & Vanden-Eijnden, E. (2023)** - "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions" _(Unified framework)_

7. **Pooladian, A. A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., & Chen, R. T. Q. (2023)** - "Multisample Flow Matching: Straightening Flows with Minibatch Couplings" _(Advanced techniques)_

---

**Bài trước:** [Flow Map Matching: Khi Bản Đồ Thay Cho Vận Tốc](/posts/2025/flow-map-matching)

**Series hoàn chỉnh:** [Generative AI Overview](/posts/2025/generative-ai-overview)

<script src="/assets/js/katex-init.js"></script>
