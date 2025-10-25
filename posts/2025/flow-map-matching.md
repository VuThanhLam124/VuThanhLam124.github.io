---
title: "Flow Map Matching: Khi Bản Đồ Thay Cho Vận Tốc"
date: "2025-01-23"
category: "flow-based-models"
tags: ["flow-map-matching", "flow-matching", "one-step-generation", "generative-models", "pytorch"]
excerpt: "Ngày thứ 11, người thợ gốm nhận ra: thay vì tính vận tốc mỗi lần, tại sao không lưu hẳn 'bản đồ' từ điểm đầu đến điểm cuối? Flow Map Matching giúp anh nhảy trực tiếp đến kết quả."
author: "ThanhLamDev"
readingTime: 25
featured: true
---

# Flow Map Matching: Khi Bản Đồ Thay Cho Vận Tốc

**Người Thợ Gốm Và Ý Tưởng "Preset"**

Sau khi làm chủ [Schrödinger Bridge](/posts/2025/schrodinger-bridge), người thợ gốm đã có trong tay ba kỹ thuật: Flow Matching (regression), Rectified Flow (straight paths), và Schrödinger Bridge (optimal noise). Nhưng anh nhận ra một vấn đề thực tế...

## Mục lục

1. [Ngày thứ 11 - Vấn đề preset](#1-ngày-thứ-11---vấn-đề-preset)
2. [Ý tưởng: Bản đồ thay vì vận tốc](#2-ý-tưởng-bản-đồ-thay-vì-vận-tốc)
3. [Toán học Flow Map](#3-toán-học-flow-map)
4. [Chiến lược training](#4-chiến-lược-training)
5. [Implementation PyTorch](#5-implementation-pytorch)
6. [So sánh với các phương pháp khác](#6-so-sánh-với-các-phương-pháp-khác)
7. [Kết luận](#7-kết-luận)

---

## 1. Ngày thứ 11 - Vấn đề preset

### Khách hàng quay lại

Sáng ngày thứ 11, khách hàng đơn 100 con rồng (Rectified Flow) gọi điện:

**"Anh ơi, tôi muốn order thêm 20 con rồng NỮA, giống y như lần trước!"**

"Được thôi," anh trả lời. "Để anh setup lại..."

Anh mở máy tính, load model Rectified Flow, chuẩn bị chạy lại quy trình:

```python
# Quy trình hiện tại (Rectified Flow)
1. Load velocity model v_theta
2. Khởi tạo z ~ N(0,1)
3. Giải ODE: x_t = ODESolve(v_theta, z, steps=1)
4. Trả về x_1 (con rồng)
```

**"Hmm..."** Anh dừng lại. "Để tính ra kết quả, anh vẫn phải **GIẢI ODE** mỗi lần. Dù chỉ 1 step, nhưng nếu khách muốn 1000 con rồng thì phải giải 1000 lần!"

### Thí nghiệm timing

Anh đo thời gian:

```python
import time

# Method 1: Rectified Flow (ODE solver)
start = time.time()
for i in range(100):
    z = randn()
    x = ode_solve(v_theta, z, steps=1)  # Mỗi lần gọi ODE
print(f"Rectified Flow: {time.time() - start:.2f}s")
# Kết quả: 0.85s

# Method 2: Direct lookup (giả sử có sẵn map)
start = time.time()
for i in range(100):
    z = randn()
    x = phi_theta(z, t=1)  # Tra bảng trực tiếp
print(f"Direct Map: {time.time() - start:.2f}s")
# Kết quả: 0.12s

# → Nhanh gấp 7 lần!
```

**Phát hiện:**

"ODE solver, dù chỉ 1 step, vẫn có **overhead**," anh ghi chú. "Nếu học trực tiếp **bản đồ** $\phi_t(x_0) = x_t$, có thể nhanh hơn!"

### Câu hỏi đặt ra

> **Có cách nào học trực tiếp "bản đồ" từ noise $z$ đến data $x$ không?**
> - Thay vì học vận tốc $v_t(x)$ rồi tích phân
> - Học luôn $\phi_t(x_0) = x_t$ (vị trí tại thời điểm $t$)
> - Giống như lưu "preset" trong xưởng!

## 2. Ý tưởng: Bản đồ thay vì vận tốc

### Sự khác biệt giữa hai approach

Anh vẽ sơ đồ so sánh:

```
┌─────────────────────────────────────────────────────┐
│ Flow Matching (velocity-based)                     │
├─────────────────────────────────────────────────────┤
│ Học: v_θ(x,t) - vận tốc tại mỗi điểm               │
│                                                     │
│ Sử dụng:                                           │
│   x₀ = z                                           │
│   for t in [0, 0.1, 0.2, ..., 1.0]:               │
│       x += v_θ(x, t) * dt  ← Tích phân từng bước  │
│   return x                                         │
│                                                     │
│ Ưu điểm: Linh hoạt, có thể dừng giữa chừng        │
│ Nhược điểm: Phải giải ODE (nhiều bước)            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Flow Map Matching (map-based)                      │
├─────────────────────────────────────────────────────┤
│ Học: φ_θ(x₀,t) - vị trí trực tiếp tại thời điểm t │
│                                                     │
│ Sử dụng:                                           │
│   x₀ = z                                           │
│   x_t = φ_θ(x₀, t)  ← Tra bảng 1 lần!             │
│   return x_t                                       │
│                                                     │
│ Ưu điểm: CỰC NHANH (1 forward pass)               │
│ Nhược điểm: Ít linh hoạt hơn                      │
└─────────────────────────────────────────────────────┘
```

### Trực giác về flow map

**Flow Map $\phi_t$:** Hàm "dịch chuyển" từ vị trí ban đầu $x_0$ đến vị trí $x_t$ tại thời điểm $t$.

$$
x_t = \phi_t(x_0)
$$

**Tính chất cần có:**

1. $\phi_0(x) = x$ (Identity - không đi đâu cả)
2. $\phi_1(x_0) \sim p_{\text{data}}$ (Đến đúng phân phối data)
3. $\phi_s \circ \phi_t = \phi_{s+t}$ (Composition - ghép được với nhau)

### Liên hệ với velocity

Nếu đã có velocity $v_t$, flow map là:

$$
\phi_t(x_0) = x_0 + \int_0^t v_s(\phi_s(x_0)) ds
$$

Ngược lại, nếu có flow map $\phi_t$, velocity là:

$$
v_t(x) = \frac{\partial}{\partial t} \phi_t(x_0) \Big|_{x_0 = \phi_t^{-1}(x)}
$$

**Ý nghĩa:** Hai cách nhìn bổ sung cho nhau!

## 3. Toán học Flow Map

### 3.1. Định nghĩa flow map

Cho cặp $(x_0, x_1)$ với $x_0 \sim p_0$ (noise), $x_1 \sim p_1$ (data).

**Linear interpolation:**

$$
x_t = (1-t) x_0 + t x_1
$$

**Mục tiêu:** Học $\phi_\theta(x_0, t)$ để:

$$
\phi_\theta(x_0, t) \approx x_t = (1-t) x_0 + t x_1
$$

### 3.2. Loss function

**Matching loss:**

$$
\mathcal{L}_{\text{match}}(\theta) = \mathbb{E}_{x_0, x_1, t} \left[ \| \phi_\theta(x_0, t) - x_t \|^2 \right]
$$

**Giải thích:**
- Sample $(x_0, x_1)$ từ data
- Sample $t \sim U(0,1)$
- Tính $x_t = (1-t)x_0 + tx_1$ (target)
- Predict $\hat{x}_t = \phi_\theta(x_0, t)$
- Minimize MSE

**Composition loss (optional):**

Để đảm bảo tính chất $\phi_s \circ \phi_t = \phi_{s+t}$:

$$
\mathcal{L}_{\text{comp}}(\theta) = \mathbb{E}_{x_0, s, t} \left[ \| \phi_\theta(\phi_\theta(x_0, s), t) - \phi_\theta(x_0, s+t) \|^2 \right]
$$

với $s + t \leq 1$.

**Tổng hợp:**

$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{match}} + \lambda \mathcal{L}_{\text{comp}}
$$

### 3.3. Boundary conditions

**Tại $t=0$:**

$$
\phi_\theta(x_0, 0) = x_0
$$

Có thể enforce bằng architecture (thêm residual connection).

**Tại $t=1$:**

$$
\phi_\theta(x_0, 1) \sim p_1
$$

Được đảm bảo qua matching loss với data.

## 4. Chiến lược training

### Ngày thứ 12 - Thực hiện Flow Map Matching

Sáng ngày thứ 12, người thợ gốm bắt đầu implement.

### 4.1. Dataset preparation

```python
# Chuẩn bị data
# x_0: Khối cầu (noise)
# x_1: Con rồng (data)

dataset = []
for dragon in dragons_data:
    sphere = randn()  # Noise
    dataset.append((sphere, dragon))
```

### 4.2. Training loop

```python
# Pseudocode
for epoch in range(epochs):
    for (x0, x1) in dataloader:
        # Sample time
        t = rand()
        
        # Target position
        x_t = (1 - t) * x0 + t * x1
        
        # Predict
        x_t_pred = phi_theta(x0, t)
        
        # Loss
        loss_match = MSE(x_t_pred, x_t)
        
        # Optional: composition loss
        if use_composition:
            s = rand() * (1 - t)
            x_s = phi_theta(x0, s)
            x_st_pred = phi_theta(x_s, t)
            x_st_true = phi_theta(x0, s + t)
            loss_comp = MSE(x_st_pred, x_st_true)
            loss = loss_match + lambda * loss_comp
        else:
            loss = loss_match
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4.3. Kết quả training

Sau 50 epochs:

```
Epoch 10: Loss = 0.542
Epoch 20: Loss = 0.287
Epoch 30: Loss = 0.142
Epoch 40: Loss = 0.078
Epoch 50: Loss = 0.032

Yes Converged!
```

### 4.4. Testing

Anh test với 100 con rồng:

```python
# Sampling
z = randn(100, dim)
dragons = phi_theta(z, t=1.0)  # 1 lần forward!

# So sánh
print("Rectified Flow (ODE): 0.85s")
print("Flow Map (Direct):     0.12s")
print("→ Nhanh gấp 7x!")
```

"Hoàn hảo!" Anh ghi chép. "Giờ tạo 'preset' chỉ cần 1 lần forward pass!"

## 5. Implementation PyTorch

### 5.1. Flow Map Network

```python
import torch
import torch.nn as nn
import math

class FlowMapNetwork(nn.Module):
    """
    Flow Map φ_θ(x₀, t) = x_t
    """
    
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.dim = dim
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        freqs = torch.exp(
            torch.linspace(0, math.log(1000), time_embed_dim // 2)
        )
        self.register_buffer('freqs', freqs)
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def time_embedding(self, t):
        """Sinusoidal time embedding"""
        angles = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb
    
    def forward(self, x0, t):
        """
        Args:
            x0: Initial state (batch, dim)
            t: Time (batch,) or (batch, 1)
        
        Returns:
            x_t: Position at time t (batch, dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        # Time embedding
        t_emb = self.time_embedding(t)
        t_feat = self.time_mlp(t_emb)
        
        # Process x0
        h = self.net[0](x0)
        h = self.net[1](h)
        h = self.net[2](h) + t_feat  # Add time
        
        for layer in self.net[3:]:
            h = layer(h)
        
        # Residual connection to enforce φ(x0, 0) ≈ x0
        return x0 + h
```

### 5.2. Training Function

```python
def train_flow_map(model, data_loader, epochs=100, 
                    use_composition=False, lambda_comp=0.1,
                    device='cuda'):
    """
    Train Flow Map network
    
    Args:
        model: FlowMapNetwork
        data_loader: DataLoader with (x0, x1) pairs
        epochs: Number of epochs
        use_composition: Whether to use composition loss
        lambda_comp: Weight for composition loss
        device: torch device
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    model.train()
    for epoch in range(epochs):
        epoch_loss_match = 0.0
        epoch_loss_comp = 0.0
        
        for x0, x1 in data_loader:
            x0, x1 = x0.to(device), x1.to(device)
            batch_size = x0.shape[0]
            
            # Sample time
            t = torch.rand(batch_size, device=device)
            
            # Target
            x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
            
            # Predict
            x_t_pred = model(x0, t)
            
            # Matching loss
            loss_match = ((x_t_pred - x_t) ** 2).mean()
            
            # Total loss
            loss = loss_match
            
            # Composition loss (optional)
            if use_composition:
                # Sample s < 1 - t
                s = torch.rand(batch_size, device=device) * (1 - t)
                
                # φ(x0, s)
                x_s = model(x0, s)
                
                # φ(φ(x0, s), t)
                x_st_composed = model(x_s, t)
                
                # φ(x0, s+t)
                x_st_direct = model(x0, s + t)
                
                # Composition loss
                loss_comp = ((x_st_composed - x_st_direct) ** 2).mean()
                loss = loss + lambda_comp * loss_comp
                epoch_loss_comp += loss_comp.item()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss_match += loss_match.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_match = epoch_loss_match / len(data_loader)
            if use_composition:
                avg_comp = epoch_loss_comp / len(data_loader)
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Match: {avg_match:.6f} | Comp: {avg_comp:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} | Match: {avg_match:.6f}")
    
    return model
```

### 5.3. Sampling

```python
@torch.no_grad()
def sample_flow_map(model, num_samples=100, dim=2, device='cuda'):
    """
    Sample using Flow Map
    
    Args:
        model: Trained FlowMapNetwork
        num_samples: Number of samples
        dim: Dimensionality
        device: torch device
    
    Returns:
        samples: (num_samples, dim)
    """
    model.eval()
    
    # Start from noise
    x0 = torch.randn(num_samples, dim, device=device)
    
    # Direct mapping to t=1
    t = torch.ones(num_samples, device=device)
    x1 = model(x0, t)
    
    return x1
```

### 5.4. Full Example

```python
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Generate data
# Source: Gaussian
p0_data = torch.randn(5000, 2)

# Target: Swiss Roll
t_np = np.linspace(0, 4*np.pi, 5000)
x_np = t_np * np.cos(t_np)
y_np = t_np * np.sin(t_np)
p1_data = torch.tensor(
    np.stack([x_np, y_np], axis=1), dtype=torch.float32
) / 10  # Scale

# Create dataset
dataset = TensorDataset(p0_data, p1_data)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Train Flow Map
print("Training Flow Map...")
model = FlowMapNetwork(dim=2, hidden_dim=128)
model = train_flow_map(
    model, dataloader, epochs=50,
    use_composition=True, lambda_comp=0.1
)

# Sample
print("\nSampling...")
samples = sample_flow_map(model, num_samples=1000, dim=2)

# Visualize
plt.figure(figsize=(15, 4))

plt.subplot(131)
plt.scatter(p0_data[:, 0], p0_data[:, 1], alpha=0.3, s=10)
plt.title("Source p₀ (Gaussian)")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.subplot(132)
plt.scatter(p1_data[:, 0], p1_data[:, 1], alpha=0.3, s=10)
plt.title("Target p₁ (Swiss Roll)")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.subplot(133)
plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.3, s=10)
plt.title("Flow Map Samples")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('flow_map_result.png', dpi=150)
plt.show()
```

### 5.5. Intermediate Visualization

```python
# Visualize trajectory
@torch.no_grad()
def visualize_trajectory(model, x0, num_steps=10, device='cuda'):
    """Visualize flow map at different times"""
    x0 = x0.to(device)
    
    plt.figure(figsize=(15, 3))
    
    for i, t_val in enumerate(np.linspace(0, 1, num_steps)):
        t = torch.full((x0.shape[0],), t_val, device=device)
        x_t = model(x0, t)
        
        plt.subplot(1, num_steps, i+1)
        plt.scatter(x_t[:, 0].cpu(), x_t[:, 1].cpu(), alpha=0.5, s=5)
        plt.title(f"t={t_val:.1f}")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
z = torch.randn(500, 2)
visualize_trajectory(model, z, num_steps=10)
```

## 6. So sánh với các phương pháp khác

### Bảng so sánh

| Aspect | Flow Matching | Rectified Flow | Flow Map Matching |
|--------|---------------|----------------|-------------------|
| **Học gì** | Velocity $v_t$ | Straight velocity | Map $\phi_t$ |
| **Sampling** | ODE solve (multi-step) | ODE solve (1-step) | Direct (1-step) |
| **Tốc độ** | Chậm (50-100 steps) | Nhanh (1-5 steps) | **Cực nhanh** (1 step) |
| **Linh hoạt** | Cao | Trung bình | Thấp |
| **Memory** | Thấp | Thấp | Trung bình (lưu map) |
| **Training** | Đơn giản | Đơn giản + Reflow | Đơn giản + Composition |
| **Use case** | General | Fast generation | Ultra-fast presets |

### Chi tiết so sánh

**Flow Matching:**
- Linh hoạt nhất - có thể dừng ở bất kỳ $t$ nào
- Dễ train
- Chậm - cần giải ODE nhiều bước

**Rectified Flow:**
- Nhanh hơn Flow Matching (đường thẳng)
- 1-step generation sau reflow
- Vẫn cần ODE solver

**Flow Map Matching:**
- YesYes Nhanh nhất - không cần ODE solver
- Direct lookup
- Ít linh hoạt (phải query đúng $t$)
- Khó enforce composition nếu không có loss

### Timing comparison

```python
import time

# Setup
model_fm = VelocityNet()  # Flow Matching
model_rf = VelocityNet()  # Rectified Flow
model_map = FlowMapNet()  # Flow Map

z = randn(1000, dim)

# Flow Matching (50 steps)
start = time.time()
x = ode_solve(model_fm, z, steps=50)
t_fm = time.time() - start

# Rectified Flow (1 step)
start = time.time()
x = ode_solve(model_rf, z, steps=1)
t_rf = time.time() - start

# Flow Map (direct)
start = time.time()
x = model_map(z, t=1)
t_map = time.time() - start

print(f"Flow Matching:   {t_fm:.3f}s (1.00x)")
print(f"Rectified Flow:  {t_rf:.3f}s ({t_fm/t_rf:.2f}x)")
print(f"Flow Map:        {t_map:.3f}s ({t_fm/t_map:.2f}x)")

# Typical output:
# Flow Matching:   2.450s (1.00x)
# Rectified Flow:  0.320s (7.66x)
# Flow Map:        0.045s (54.4x)
```

### Khi nào dùng Flow Map Matching?

Người thợ gốm ghi chép:

**Yes Dùng Flow Map khi:**

1. **Cần tốc độ tuyệt đối**
   - Real-time generation
   - Interactive applications
   - Serving hàng triệu requests/giây

2. **Có "preset" cố định**
   - Same start → same end (deterministic)
   - Batch generation với cùng điều kiện
   - Lưu trữ "templates"

3. **Không cần dừng giữa chừng**
   - Chỉ quan tâm $t=1$ (kết quả cuối)
   - Không cần visualize trajectory

**No Không dùng khi:**

- Cần flexibility (dừng ở $t$ bất kỳ)
- Cần stochastic sampling (diversity)
- Data thay đổi liên tục (không có preset)

## 7. Kết luận

### Ngày thứ 12 - Hoàn thành Flow Map

Chiều ngày thứ 12, người thợ gốm test hệ thống mới:

```python
# Khách hàng order
print("Tạo 1000 con rồng preset...")

start = time.time()
dragons = phi_theta(randn(1000, dim), t=1.0)
print(f"Hoàn thành trong {time.time() - start:.2f}s")

# Output: Hoàn thành trong 0.08s
# → So với Rectified Flow (0.85s): Nhanh gấp 10x!
```

**Khách hàng cực kỳ hài lòng:**

"Nhanh thật! Anh có thể làm hàng ngàn con mỗi ngày mà không tốn thời gian!"

### Bài học về Flow Map

Anh ghi vào sổ tay:

> **Flow Map Matching: Khi Bản Đồ Thay Cho Vận Tốc**
>
> 1. **Velocity → Map:** Thay vì tích phân vận tốc, học trực tiếp vị trí
> 2. **1-step direct:** Không cần ODE solver, forward 1 lần
> 3. **Composition loss:** Đảm bảo tính chất ghép nối
> 4. **Trade-off:** Tốc độ vs Linh hoạt

### Timeline cập nhật

```
Ngày 1-2:   CNF (likelihood, slow)
Ngày 3:     Flow Matching (regression, faster)
Ngày 4-7:   Rectified Flow (straight, 1-step ODE)
Ngày 8-10:  Schrödinger Bridge (optimal noise)
Ngày 11-12: Flow Map Matching (direct map, ultra-fast)
```

### Điểm chính cần nhớ

1. **Flow Map $\phi_t$:** Học trực tiếp $x_0 \to x_t$
2. **Matching loss:** MSE giữa prediction và linear interpolation
3. **Composition loss:** Enforce $\phi_s \circ \phi_t = \phi_{s+t}$
4. **Sampling:** 1 forward pass, không cần ODE
5. **Trade-off:** Tốc độ cực cao nhưng ít linh hoạt

### Hướng tiếp theo

"Giờ tôi đã có nhiều kỹ thuật," anh nghĩ. "Có cách nào **thống nhất** chúng lại không? Một framework chung để hiểu tất cả?"

→ Dẫn đến **Generator Matching Framework** (bài tiếp theo)

---

## Tài liệu tham khảo

1. **Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023)** - "Flow Matching for Generative Modeling" _(Original Flow Matching paper)_

2. **Liu, X., Gong, C., & Liu, Q. (2022)** - "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" _(Rectified Flow)_

3. **Tong, A., Malkin, N., Fatras, K., Atanackovic, L., Zhang, Y., Huguet, G., ... & Bengio, Y. (2024)** - "Simulation-Free Schrödinger Bridges via Score and Flow Matching" _(Unified framework)_

4. **Pooladian, A. A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., & Chen, R. T. Q. (2023)** - "Multisample Flow Matching: Straightening Flows with Minibatch Couplings" _(Advanced Flow Map techniques)_

5. **Albergo, M. S., & Vanden-Eijnden, E. (2023)** - "Building Normalizing Flows with Stochastic Interpolants" _(Stochastic Interpolants - generalization)_

---

**Bài trước:** [Schrödinger Bridge: Khi Đường Đi Gặp Nhiễu](/posts/2025/schrodinger-bridge)

**Bài tiếp theo:** [Generator Matching Framework: Thống Nhất Lý Thuyết](/posts/2025/generator-matching-framework)

<script src="/assets/js/katex-init.js"></script>
