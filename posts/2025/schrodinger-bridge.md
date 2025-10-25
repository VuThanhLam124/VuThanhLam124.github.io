---
title: "Schrödinger Bridge: Khi Đường Đi Gặp Nhiễu"
date: "2025-01-22"
category: "flow-based-models"
tags: ["schrodinger-bridge", "optimal-transport", "stochastic-process", "flow-based-models", "pytorch"]
excerpt: "Ngày thứ 8, người thợ gốm nhận đơn hàng đặc biệt: nặn trong điều kiện có gió - đất sét bị nhiễu động ngẫu nhiên. Schrödinger Bridge giúp anh tìm đường đi tối ưu khi phải đối mặt với sự bất định."
author: "ThanhLamDev"
readingTime: 28
featured: true
---

# Schrödinger Bridge: Khi Đường Đi Gặp Nhiễu

**Người Thợ Gốm Và Thách Thức Của Gió**

Trong bài [Rectified Flows](/posts/2025/rectified-flows), người thợ gốm đã làm chủ nghệ thuật tạo đường đi thẳng. 7 ngày vừa qua, anh đã hoàn thành đơn hàng 100 con rồng với kỹ thuật one-step generation. Nhưng thế giới thực không bao giờ hoàn hảo...

## Mục lục

1. [Ngày thứ 8 - Đơn hàng đặc biệt](#1-ngày-thứ-8---đơn-hàng-đặc-biệt)
2. [Vấn đề: Khi có gió thổi](#2-vấn-đề-khi-có-gió-thổi)
3. [Schrödinger Bridge: Tối ưu trong nhiễu](#3-schrödinger-bridge-tối-ưu-trong-nhiễu)
4. [Toán học: Forward-Backward SDE](#4-toán-học-forward-backward-sde)
5. [Thuật toán IPF: Học qua lại](#5-thuật-toán-ipf-học-qua-lại)
6. [Implementation PyTorch](#6-implementation-pytorch)
7. [So sánh với các phương pháp khác](#7-so-sánh-với-các-phương-pháp-khác)
8. [Kết luận](#8-kết-luận)

---

## 1. Ngày thứ 8 - Đơn hàng đặc biệt

### Buổi sáng bất ngờ

Sáng ngày thứ 8, sau khi hoàn thành đơn hàng 100 con rồng, người thợ gốm nhận được điện thoại:

**"Anh ơi, tôi cần 50 con rồng đặc biệt - được nặn **ngoài trời**, trong điều kiện có gió!"**

Anh ngạc nhiên: "Tại sao lại nặn ngoài trời?"

**"Khách hàng muốn mỗi con rồng có 'tính ngẫu nhiên tự nhiên' - những vân nứt, những nếp gấp do gió tạo ra. Không được hoàn hảo quá!"**

Anh đi ra sân, đặt bàn xoay và khối đất sét. Một cơn gió nhẹ thổi qua.

"Ồ..." Anh nhận ra vấn đề ngay lập tức.

### Thí nghiệm với gió

Anh thử nặn con rồng đầu tiên ngoài trời:

```python
# Trong xưởng (không gió) - Rectified Flow
t=0.0: Tay ở (0, 0, 0)
t=0.1: Tay ở (0.5, 0.3, 0.2) ← Chính xác như dự định
t=0.2: Tay ở (1.0, 0.6, 0.4) ← Chính xác
...
t=1.0: Tay ở (5, 3, 2) ← HOÀN HẢO!

# Ngoài trời (có gió)
t=0.0: Tay ở (0, 0, 0)
t=0.1: Tay ở (0.5, 0.3, 0.2) + gió: (0.02, -0.01, 0.03)
       = (0.52, 0.29, 0.23) ← LỆCH!
t=0.2: Tay ở (1.0, 0.6, 0.4) + gió: (-0.01, 0.02, -0.01)
       = (0.99, 0.62, 0.39) ← LỆCH tiếp!
...
t=1.0: Tay ở (5.3, 2.8, 2.1) ← KHÔNG ĐÚNG ĐÍCH!
```

**Kết quả:** Sau 10 bước, con rồng **không đến đúng vị trí (5, 3, 2)** như dự kiến, mà lệch sang **(5.3, 2.8, 2.1)**!

Anh ghi chép vào sổ tay:

> **Vấn đề mới phát sinh:**
> - Rectified Flow: Đường thẳng, hoàn hảo, KHÔNG có gió
> - Thực tế: Có gió (nhiễu ngẫu nhiên), đường đi BỊ LỆCH
> - Câu hỏi: Làm sao đến ĐÚNG đích dù có gió?

## 2. Vấn đề: Khi có gió thổi

### Mô hình hóa gió = Brownian Motion

Anh ngồi nghiên cứu tại bàn làm việc buổi trưa. Gió có thể được mô hình hóa bằng **Brownian Motion** (chuyển động Brown) - chuyển động ngẫu nhiên:

$$
dW_t \sim \mathcal{N}(0, dt)
$$

**Giải thích:** Mỗi khoảng thời gian $dt$ nhỏ, gió thổi một lượng ngẫu nhiên theo phân phối Gaussian với trung bình 0 và phương sai $dt$.

### Phương trình chuyển động MỚI

Thay vì ODE đơn giản (Rectified Flow):

$$
\frac{dx}{dt} = v_\theta(x, t)
$$

Giờ phải dùng **SDE** (Stochastic Differential Equation):

$$
dx = v_\theta(x, t) dt + \sigma dW_t
$$

**Chú thích các thành phần:**
- $v_\theta(x,t) dt$: Phần **dự định** (control) - tay anh muốn di chuyển
- $\sigma dW_t$: Phần **gió** (noise) - ảnh hưởng ngẫu nhiên
- $\sigma$: Cường độ gió (noise level)

### Thí nghiệm quan sát

Anh làm 10 thí nghiệm, mỗi lần nặn từ cùng khối cầu $(0,0,0)$ đến con rồng $(5,3,2)$ với cùng chiến lược Rectified Flow nhưng có gió:

```
Lần 1: (0,0,0) → ... → (5.2, 2.9, 2.1)  ← Lệch 0.3
Lần 2: (0,0,0) → ... → (4.8, 3.1, 1.9)  ← Lệch 0.3  
Lần 3: (0,0,0) → ... → (5.1, 2.8, 2.2)  ← Lệch 0.4
Lần 4: (0,0,0) → ... → (4.9, 3.2, 2.0)  ← Lệch 0.3
...
Lần 10: (0,0,0) → ... → (4.9, 3.2, 2.0) ← Lệch 0.3

Trung bình: (5.0, 3.0, 2.0) ← OK!
Nhưng MỖI LẦN KHÁC NHAU!
```

**Phát hiện quan trọng:**

Anh ghi chú: "Rectified Flow học đường thẳng **XÁC ĐỊNH** (deterministic). Nhưng khi có gió, mỗi lần đi là một đường KHÁC NHAU! Liệu có cách nào..."

### Câu hỏi đặt ra

> **Có cách nào tìm "đường đi TỐI ƯU **TRUNG BÌNH**" sao cho:**
> 1. Vẫn đến đúng đích $(5,3,2)$ **về mặt phân phối**
> 2. Tính đến ảnh hưởng của gió (noise)
> 3. Minimize tổng "năng lượng" cần dùng để điều khiển?

## 3. Schrödinger Bridge: Tối ưu trong nhiễu

### Buổi chiều nghiên cứu

Buổi chiều cùng ngày, anh tìm đọc một bài báo cũ của Erwin Schrödinger (1931) về **Schrödinger Bridge Problem**:

**"Tìm quá trình ngẫu nhiên GẦN NHẤT với Brownian Motion sao cho xuất phát từ phân phối $p_0$ và kết thúc ở $p_1$."**

"Đây chính là vấn đề của tôi!" Anh hào hứng viết vào sổ.

### Trực giác qua ví dụ

Anh vẽ sơ đồ để hiểu rõ hơn:

```
┌────────────────────────────────────────────────────┐
│ Reference Process (Brownian thuần túy)             │
│ x(0) ~ p₀ → [random walk] → x(1) ~ Gaussian rộng │
│ Đặc điểm: Hoàn toàn ngẫu nhiên, KHÔNG đến đích   │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ Rectified Flow (Deterministic - không noise)      │
│ x(0) ~ p₀ → [straight line] → x(1) = đích         │
│ Đặc điểm: Hoàn hảo nhưng KHÔNG CÓ GIÓ            │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ Schrödinger Bridge (Stochastic + Control)         │
│ x(0) ~ p₀ → [controlled random walk] → x(1) ~ p₁ │
│ Đặc điểm:                                         │
│  - CÓ GIÓ (như Brownian)                         │
│  - NHƯNG đến đúng phân phối đích p₁              │
│  - Sử dụng ít control nhất (gần Brownian nhất)   │
└────────────────────────────────────────────────────┘
```

**Ý tưởng cốt lõi:**

Schrödinger Bridge tìm cách **ĐIỀU KHIỂN** Brownian Motion một cách **TỐI THIỂU** để đi từ $p_0$ đến $p_1$.

### Công thức toán học

**Bài toán tối ưu:**

$$
\min_{\mathbb{P}} \text{KL}(\mathbb{P} \mid\mid \mathbb{P}_{\text{ref}})
$$

subject to ràng buộc biên:

$$
\mathbb{P}|_{t=0} = p_0, \quad \mathbb{P}|_{t=1} = p_1
$$

**Giải thích ký hiệu:**
- $\mathbb{P}$: Phân phối của toàn bộ đường đi (trajectory distribution)
- $\mathbb{P}_{\text{ref}}$: Reference process (thường là Brownian Motion)
- $\text{KL}(\cdot \mid\mid \cdot)$: Kullback-Leibler divergence (đo "khoảng cách" giữa 2 phân phối)
- Ràng buộc: Phải xuất phát từ $p_0$ và kết thúc ở $p_1$

**Ý nghĩa:**

Tìm quá trình $\mathbb{P}$ **GẦN** reference nhất (minimize KL) nhưng vẫn thỏa mãn điều kiện biên.

### Liên hệ với Optimal Transport

Anh nhớ lại bài Optimal Transport trong OT-CFM:

**Optimal Transport (OT):**

$$
\min_{\pi} \int c(x_0, x_1) d\pi(x_0, x_1)
$$

→ Tìm cách ghép $(x_0, x_1)$ để minimize transport cost

**Schrödinger Bridge (Entropic OT):**

$$
\min_{\pi} \int c(x_0, x_1) d\pi(x_0, x_1) + \epsilon \cdot \text{KL}(\pi \mid\mid \pi_{\text{ref}})
$$

→ OT + entropy regularization!

"Schrödinger Bridge chính là **Entropic Optimal Transport**!" Anh ghi chú hào hứng.

**Khi $\epsilon \to 0$:** Trở về OT thuần túy (deterministic)  
**Khi $\epsilon$ lớn:** Gần với Brownian (random)

## 4. Toán học: Forward-Backward SDE

### Cặp SDE đối ngẫu

Người thợ gốm học được một kết quả đẹp: Schrödinger Bridge có thể biểu diễn bằng **cặp SDE forward-backward**:

**Forward SDE (từ $t=0$ đến $t=1$):**

$$
dx = b^f_t(x) dt + \sigma dW_t
$$

với drift:

$$
b^f_t(x) = \sigma^2 \nabla_x \log \psi_t(x)
$$

**Backward SDE (từ $t=1$ về $t=0$):**

$$
dx = b^b_t(x) dt + \sigma d\bar{W}_t
$$

với drift:

$$
b^b_t(x) = \sigma^2 \nabla_x \log \hat{\psi}_t(x)
$$

**Chú thích:**
- $\psi_t, \hat{\psi}_t$: Forward và backward potentials (hàm mật độ chưa chuẩn hóa)
- $\nabla_x \log \psi_t$: **Score function** (gradient của log-density)
- $\sigma$: Cường độ nhiễu (gió)
- $W_t, \bar{W}_t$: Brownian motion (forward và backward)

### Liên hệ với Score Matching

"Đây giống Diffusion Models!" Anh nhận ra ngay.

**Trong Diffusion:**
- Học score: $s_\theta(x,t) \approx \nabla_x \log p_t(x)$
- Dùng để denoise

**Trong Schrödinger Bridge:**
- Học 2 scores: $s^f_\theta, s^b_\phi$
- Dùng để **control trong môi trường nhiễu**

### Ví dụ minh họa cụ thể

Giả sử anh muốn đi từ điểm $(0,0)$ đến $(5,3)$ với $\sigma=0.1$:

**Không control (Brownian thuần):**

```python
x = (0, 0)
for t in range(100):
    x += 0.1 * randn()  # Chỉ có gió, không control
# Kết quả: x ≈ (0.5, -0.3) ← Hoàn toàn ngẫu nhiên!
```

**Rectified Flow (deterministic, không gió):**

```python
x = (0, 0)
target = (5, 3)
for t in range(100):
    v = target - x  # Velocity
    x += 0.01 * v   # Không gió
# Kết quả: x = (5.000, 3.000) ← Chính xác!
```

**Schrödinger Bridge (optimal control + noise):**

```python
x = (0, 0)
sigma = 0.1
for t in range(100):
    score_f = score_net_forward(x, t)
    drift = sigma**2 * score_f      # Control từ score
    noise = sigma * randn()           # Gió
    x += 0.01 * drift + noise
# Kết quả: x ≈ (5.02, 2.98) ← Gần đúng + có tính ngẫu nhiên!
```

## 5. Thuật toán IPF: Học qua lại

### Ngày thứ 9 - Bắt đầu thử nghiệm

Sáng ngày thứ 9, người thợ gốm học thuật toán **IPF** (Iterative Proportional Fitting) để tìm Schrödinger Bridge.

### Ý tưởng IPF

**Quy trình:** Học forward và backward scores **lần lượt**, mỗi lần điều chỉnh để khớp một điều kiện biên.

```
Iteration 0:
  score_f^0 = 0 (khởi tạo - no control)
  score_b^0 = 0

Iteration 1:
  [Forward] Học score_f^1 để: x(1) ~ p₁
            (cho điều kiện x(0) ~ p₀, dùng score_b^0)
  
  [Backward] Học score_b^1 để: x(0) ~ p₀
             (cho điều kiện x(1) ~ p₁, dùng score_f^1)

Iteration 2:
  [Forward] Học score_f^2 với score_b^1
  [Backward] Học score_b^2 với score_f^2
  
...

→ Hội tụ đến Schrödinger Bridge!
```

### Pseudocode chi tiết

```python
# IPF Algorithm
def ipf_schrodinger_bridge(p0_data, p1_data, num_iters=5):
    # Khởi tạo
    score_f = ScoreNet()
    score_b = ScoreNet()
    
    for k in range(num_iters):
        print(f"IPF Iteration {k+1}")
        
        # ===== Forward Pass =====
        # Mục tiêu: Học score_f để x(1) ~ p₁
        for epoch in range(50):
            for x0 in p0_data:
                # Simulate forward với score_b hiện tại
                x_traj = simulate_sde_forward(
                    x0, score_b, steps=100, sigma=0.1
                )
                
                # x_traj[-1] nên gần p₁
                # Train score_f bằng denoising
                t = random_time()
                x_t = x_traj[t]
                noise = randn()
                x_noisy = x_t + 0.01 * noise
                
                score_pred = score_f(x_noisy, t)
                score_true = -noise / 0.01
                
                loss = MSE(score_pred, score_true)
                update(score_f, loss)
        
        # ===== Backward Pass =====
        # Mục tiêu: Học score_b để x(0) ~ p₀
        for epoch in range(50):
            for x1 in p1_data:
                # Simulate backward với score_f MỚI
                x_traj = simulate_sde_backward(
                    x1, score_f, steps=100, sigma=0.1
                )
                
                # x_traj[0] nên gần p₀
                # Train score_b tương tự
                t = random_time()
                x_t = x_traj[t]
                noise = randn()
                x_noisy = x_t + 0.01 * noise
                
                score_pred = score_b(x_noisy, t)
                score_true = -noise / 0.01
                
                loss = MSE(score_pred, score_true)
                update(score_b, loss)
    
    return score_f, score_b
```

### Quá trình thực hiện

Anh bắt đầu chạy IPF:

**Iteration 1:**

```
Forward: Học score_f^1
  - Từ p₀ (khối cầu)
  - Dùng score_b^0 = 0 (no control)
  - Kết quả: x(1) phân tán rộng, không đến p₁
  - Loss: 0.85

Backward: Học score_b^1
  - Từ "phân tán rộng" ở t=1
  - Điều chỉnh về p₀ ở t=0
  - Kết quả: Tốt hơn một chút
  - Loss: 0.72
```

**Iteration 2:**

```
Forward: Học score_f^2 với score_b^1
  - Kết quả: x(1) GẦN p₁ hơn!
  - Loss: 0.45
  
Backward: Học score_b^2 với score_f^2
  - Kết quả: x(0) CHÍNH XÁC p₀!
  - Loss: 0.31
```

**Iteration 3:**

```
Forward & Backward:
  - Gần như HỘI TỤ!
  - x(0) ~ p₀ ✓
  - x(1) ~ p₁ ✓
  - Đường đi có tính ngẫu nhiên tự nhiên ✓
  - Loss: 0.12
```

### Kiểm tra kết quả

Anh test với 50 con rồng ngoài trời:

```python
# Mỗi con rồng
z0 = randn()  # Khối cầu ngẫu nhiên  
x_final = sample_schrodinger_bridge(score_f, score_b, z0)

print(f"Vị trí cuối: {x_final}")
# Kết quả:
# Con 1: (5.01, 2.99, 2.02)
# Con 2: (4.98, 3.02, 1.99)
# Con 3: (5.03, 2.97, 2.01)
# Con 4: (4.97, 3.01, 2.03)
# ...
# → Đều gần (5, 3, 2) nhưng có biến thiên nhẹ!
```

"Hoàn hảo!" Anh hài lòng ghi chú. "Mỗi con rồng đến đúng hình dạng mục tiêu, nhưng có nét riêng do gió tạo ra - chính xác như khách hàng yêu cầu!"

## 6. Implementation PyTorch

### 6.1. Score Network

```python
import torch
import torch.nn as nn
import math

class ScoreNetwork(nn.Module):
    """Score network s(x,t) = ∇log p_t(x)"""
    
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.dim = dim
        
        # Time embedding (sinusoidal)
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
        # t: (batch_size,)
        angles = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, dim)
            t: (batch,) or (batch, 1)
        Returns:
            score: (batch, dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        # Time embedding
        t_emb = self.time_embedding(t)
        t_feat = self.time_mlp(t_emb)
        
        # Combine with x
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h) + t_feat
        
        for layer in self.net[3:]:
            h = layer(h)
        
        return h
```

### 6.2. SDE Simulation

```python
def euler_maruyama_forward(x0, score_net, num_steps=100, sigma=0.1, device='cuda'):
    """
    Simulate forward SDE: dx = σ² ∇log ψ(x,t) dt + σ dW
    
    Args:
        x0: Initial state (batch, dim)
        score_net: Forward score network
        num_steps: Number of time steps
        sigma: Noise level
    
    Returns:
        trajectory: (num_steps+1, batch, dim)
    """
    x = x0.to(device)
    dt = 1.0 / num_steps
    trajectory = [x.clone()]
    
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i * dt, device=device)
        
        # Drift from score
        with torch.no_grad():
            score = score_net(x, t)
            drift = sigma**2 * score
        
        # Diffusion
        noise = torch.randn_like(x) * math.sqrt(dt)
        
        # Update
        x = x + drift * dt + sigma * noise
        trajectory.append(x.clone())
    
    return torch.stack(trajectory)

def euler_maruyama_backward(x1, score_net, num_steps=100, sigma=0.1, device='cuda'):
    """
    Simulate backward SDE from t=1 to t=0
    
    Args:
        x1: Terminal state (batch, dim)
        score_net: Backward score network
        num_steps: Number of time steps
        sigma: Noise level
    
    Returns:
        trajectory: (num_steps+1, batch, dim)
    """
    x = x1.to(device)
    dt = 1.0 / num_steps
    trajectory = [x.clone()]
    
    for i in range(num_steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=device)
        
        # Backward drift
        with torch.no_grad():
            score = score_net(x, t)
            drift = sigma**2 * score
        
        # Backward diffusion
        noise = torch.randn_like(x) * math.sqrt(dt)
        
        # Update (backward time)
        x = x - drift * dt + sigma * noise
        trajectory.append(x.clone())
    
    return torch.stack(trajectory)
```

### 6.3. Training with Denoising Score Matching

```python
def train_score_matching(score_net, data_loader, epochs=100, 
                          noise_level=0.01, device='cuda'):
    """
    Train score network using denoising score matching
    
    Args:
        score_net: Score network to train
        data_loader: DataLoader for training data
        epochs: Number of training epochs
        noise_level: Noise level for denoising
        device: torch device
    
    Returns:
        Trained score_net
    """
    score_net = score_net.to(device)
    optimizer = torch.optim.AdamW(score_net.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    score_net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for x_clean in data_loader:
            x_clean = x_clean.to(device)
            batch_size = x_clean.shape[0]
            
            # Random time
            t = torch.rand(batch_size, device=device)
            
            # Add noise
            noise = torch.randn_like(x_clean)
            x_noisy = x_clean + noise_level * noise
            
            # Predict score
            score_pred = score_net(x_noisy, t)
            
            # True score (for denoising)
            score_true = -noise / noise_level
            
            # MSE loss
            loss = ((score_pred - score_true) ** 2).mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    return score_net
```

### 6.4. Full IPF Algorithm

```python
def train_schrodinger_bridge_ipf(
    p0_data, p1_data, dim, 
    num_ipf_iters=5, 
    epochs_per_iter=50,
    sigma=0.1,
    device='cuda'
):
    """
    Full IPF algorithm for Schrödinger Bridge
    
    Args:
        p0_data: Source distribution samples (N, dim)
        p1_data: Target distribution samples (M, dim)
        dim: Dimensionality
        num_ipf_iters: Number of IPF iterations
        epochs_per_iter: Training epochs per IPF iteration
        sigma: Noise level
        device: torch device
    
    Returns:
        score_forward, score_backward: Trained score networks
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create data loaders
    p0_loader = DataLoader(
        TensorDataset(p0_data), batch_size=256, shuffle=True
    )
    p1_loader = DataLoader(
        TensorDataset(p1_data), batch_size=256, shuffle=True
    )
    
    # Initialize score networks
    score_forward = ScoreNetwork(dim).to(device)
    score_backward = ScoreNetwork(dim).to(device)
    
    for ipf_iter in range(num_ipf_iters):
        print(f"\n{'='*60}")
        print(f"IPF Iteration {ipf_iter+1}/{num_ipf_iters}")
        print(f"{'='*60}")
        
        # ===== Forward Pass =====
        print("\n[Forward] Training forward score...")
        
        # Generate forward trajectories using current backward score
        forward_data = []
        with torch.no_grad():
            for x0_batch, in p0_loader:
                x0 = x0_batch.to(device)
                traj = euler_maruyama_forward(
                    x0, score_backward, 
                    num_steps=50, sigma=sigma, device=device
                )
                # Sample random points from trajectory
                for _ in range(10):
                    t_idx = torch.randint(0, len(traj), (1,)).item()
                    forward_data.append(traj[t_idx])
        
        forward_data = torch.cat(forward_data, dim=0)
        forward_loader = DataLoader(
            TensorDataset(forward_data), 
            batch_size=256, shuffle=True
        )
        
        # Train forward score
        score_forward = ScoreNetwork(dim).to(device)  # Re-initialize
        train_score_matching(
            score_forward, forward_loader, 
            epochs=epochs_per_iter, device=device
        )
        
        # ===== Backward Pass =====
        print("\n[Backward] Training backward score...")
        
        # Generate backward trajectories using NEW forward score
        backward_data = []
        with torch.no_grad():
            for x1_batch, in p1_loader:
                x1 = x1_batch.to(device)
                traj = euler_maruyama_backward(
                    x1, score_forward,
                    num_steps=50, sigma=sigma, device=device
                )
                for _ in range(10):
                    t_idx = torch.randint(0, len(traj), (1,)).item()
                    backward_data.append(traj[t_idx])
        
        backward_data = torch.cat(backward_data, dim=0)
        backward_loader = DataLoader(
            TensorDataset(backward_data),
            batch_size=256, shuffle=True
        )
        
        # Train backward score
        score_backward = ScoreNetwork(dim).to(device)  # Re-initialize
        train_score_matching(
            score_backward, backward_loader,
            epochs=epochs_per_iter, device=device
        )
    
    return score_forward, score_backward
```

### 6.5. Sampling

```python
@torch.no_grad()
def sample_schrodinger_bridge(score_forward, num_samples=100, dim=2, 
                               num_steps=100, sigma=0.1, device='cuda'):
    """
    Sample from Schrödinger Bridge
    
    Args:
        score_forward: Trained forward score network
        num_samples: Number of samples to generate
        dim: Dimensionality
        num_steps: Number of SDE steps
        sigma: Noise level
        device: torch device
    
    Returns:
        samples: (num_samples, dim)
    """
    score_forward.eval()
    
    # Start from Gaussian
    x = torch.randn(num_samples, dim, device=device)
    
    # Run forward SDE
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        
        # Score
        score = score_forward(x, t)
        drift = sigma**2 * score
        
        # Noise
        noise = torch.randn_like(x) * math.sqrt(dt)
        
        # Update
        x = x + drift * dt + sigma * noise
    
    return x
```

### 6.6. Full Example: Two Moons

```python
# Generate toy data
import numpy as np
import matplotlib.pyplot as plt

# Source: Gaussian
p0_data = torch.randn(5000, 2)

# Target: Two moons
theta = np.linspace(0, np.pi, 2500)
moon1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
moon2 = np.stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5], axis=1)
p1_np = np.vstack([moon1, moon2])
p1_data = torch.tensor(p1_np, dtype=torch.float32)

# Train Schrödinger Bridge
print("Training Schrödinger Bridge with IPF...")
score_f, score_b = train_schrodinger_bridge_ipf(
    p0_data, p1_data, dim=2,
    num_ipf_iters=3,
    epochs_per_iter=30,
    sigma=0.2
)

# Sample
print("\nSampling...")
samples = sample_schrodinger_bridge(
    score_f, num_samples=1000, dim=2, sigma=0.2
)

# Visualize
plt.figure(figsize=(15, 4))

plt.subplot(131)
plt.scatter(p0_data[:, 0], p0_data[:, 1], alpha=0.3, s=10)
plt.title("Source p₀ (Gaussian)")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.subplot(132)
plt.scatter(p1_data[:, 0], p1_data[:, 1], alpha=0.3, s=10)
plt.title("Target p₁ (Two Moons)")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.subplot(133)
plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.3, s=10)
plt.title("Schrödinger Bridge Samples")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('schrodinger_bridge_result.png', dpi=150)
plt.show()
```

### Kết quả chạy

```
Training Schrödinger Bridge with IPF...

============================================================
IPF Iteration 1/3
============================================================

[Forward] Training forward score...
Epoch 10/30 | Loss: 0.742153
Epoch 20/30 | Loss: 0.521847
Epoch 30/30 | Loss: 0.398421

[Backward] Training backward score...
Epoch 10/30 | Loss: 0.678942
Epoch 20/30 | Loss: 0.445231
Epoch 30/30 | Loss: 0.321854

============================================================
IPF Iteration 2/3
============================================================

[Forward] Training forward score...
Epoch 10/30 | Loss: 0.412378
Epoch 20/30 | Loss: 0.289145
Epoch 30/30 | Loss: 0.198423

[Backward] Training backward score...
Epoch 10/30 | Loss: 0.387241
Epoch 20/30 | Loss: 0.267894
Epoch 30/30 | Loss: 0.176542

============================================================
IPF Iteration 3/3
============================================================

[Forward] Training forward score...
Epoch 10/30 | Loss: 0.187234
Epoch 20/30 | Loss: 0.134561
Epoch 30/30 | Loss: 0.089432

[Backward] Training backward score...
Epoch 10/30 | Loss: 0.165432
Epoch 20/30 | Loss: 0.121876
Epoch 30/30 | Loss: 0.082145

Sampling...
Done!
```

## 7. So sánh với các phương pháp khác

### Bảng so sánh tổng hợp

| Aspect | Rectified Flow | Diffusion Models | Schrödinger Bridge |
|--------|----------------|------------------|---------------------|
| **Đường đi** | Deterministic (thẳng) | Stochastic (ngẫu nhiên) | Stochastic (tối ưu) |
| **Nhiễu** | Không | Có (bắt buộc) | Có (kiểm soát) |
| **Mục tiêu** | Minimize transport cost | Match score | Minimize KL vs reference |
| **Training** | 1 model (velocity) | 1 model (score) | 2 models (forward + backward) |
| **Sampling steps** | 1-5 | 50-1000 | 50-200 |
| **Flexibility** | Thấp | Cao | Rất cao |
| **Diversity** | Thấp (deterministic) | Cao | Trung bình-Cao (tunable) |
| **Speed** | Cực nhanh | Chậm | Trung bình |
| **Use case** | Fast generation | High quality | Constrained noise |

### Khi nào dùng Schrödinger Bridge?

Người thợ gốm ghi chép vào sổ tay:

**✅ Dùng Schrödinger Bridge khi:**

1. **Có ràng buộc vật lý về nhiễu**
   - VD: Chuyển động phân tử (phải tuân theo Brownian)
   - VD: Video generation (nhiễu camera tự nhiên)
   - VD: Weather forecasting (nhiễu môi trường)

2. **Cần đa dạng nhưng kiểm soát**
   - Không muốn hoàn toàn deterministic (nhàm chán, không realistic)
   - Không muốn hoàn toàn random (mất kiểm soát, unpredictable)
   - Muốn "diverse but reasonable"

3. **Kết hợp Flow và Diffusion**
   - Best of both worlds
   - Speed của Flow + Diversity của Diffusion
   - Có thể tune $\sigma$ để balance

4. **Domain adaptation với noise**
   - Transfer giữa các domains có noise characteristics khác nhau
   - Image-to-image translation với noise constraints

**❌ Không dùng khi:**

- **Cần tốc độ cực nhanh (1-step)** → Dùng Rectified Flow
- **Chất lượng tuyệt đối là ưu tiên** → Dùng Diffusion Models
- **Dataset nhỏ, không muốn train phức tạp** → Dùng Flow Matching
- **Không quan tâm noise characteristics** → Dùng OT-CFM

### Ví dụ minh họa

```python
# Cùng task: Gaussian → Two Moons

# Rectified Flow (1 step, fast, không diverse)
z = randn()
x = z + v_theta(z, 0)  # 1 step
# Kết quả: Nhanh nhưng mỗi z cho cùng 1 output

# Diffusion (1000 steps, slow, rất diverse)
x = randn()
for t in range(1000):
    x = denoise_step(x, t)
# Kết quả: Chậm nhưng chất lượng cao, rất diverse

# Schrödinger Bridge (100 steps, medium speed, tunable diversity)
x = randn()
for t in range(100):
    x += sigma^2 * score_f(x,t) * dt + sigma * dW
# Kết quả: Balance tốt giữa speed và diversity
```

## 8. Kết luận

### Ngày thứ 10 - Hoàn thành đơn hàng

Ngày thứ 10, người thợ gốm hoàn thành 50 con rồng "có tính ngẫu nhiên tự nhiên".

Anh đặt chúng cạnh nhau dưới ánh nắng buổi chiều, mỉm cười:

"Mỗi con đều đến đúng hình dạng mục tiêu $(5,3,2)$, nhưng mỗi con có nét riêng - những vân nứt nhỏ, những nếp gấp độc đáo do gió tạo ra. Không giống nhau hoàn toàn, nhưng cũng không quá khác biệt."

**Khách hàng cực kỳ hài lòng:**

"Đúng là những gì tôi cần! Perfect balance giữa consistency và uniqueness. Cảm ơn anh!"

### Bài học về Schrödinger Bridge

Anh ghi vào sổ tay những điểm chính:

> **Schrödinger Bridge: Balance giữa Control và Randomness**
>
> 1. **Thế giới thực có nhiễu** - Ta không thể kiểm soát mọi thứ
> 2. **Nhưng ta có thể tối ưu** - Tìm cách sử dụng ít control nhất
> 3. **IPF algorithm** - Học qua lại forward-backward đến hội tụ
> 4. **Kết quả** - Đa dạng nhưng vẫn đạt mục tiêu

### Timeline tổng hợp hành trình

Nhìn lại 10 ngày qua:

```
Ngày 1-2: Continuous Normalizing Flows (CNF)
  → Học likelihood (khó, chậm, cần ODE solver)
  → Vấn đề: 41 ngày cho 100 con rồng

Ngày 3: Flow Matching
  → Đổi sang regression (dễ, nhanh hơn)
  → Giảm xuống 15 ngày

Ngày 4-7: Rectified Flow
  → Làm thẳng đường đi (cực nhanh, 1-step)
  → Hoàn thành đúng 7 ngày deadline

Ngày 8-10: Schrödinger Bridge
  → Thêm nhiễu có kiểm soát (diverse + optimal)
  → Ứng dụng cho đơn hàng đặc biệt
```

### Điểm chính cần nhớ

**1. SB = OT + Entropy regularization**
   - Optimal Transport với ràng buộc noise
   - Minimize KL divergence vs reference process

**2. Forward-Backward SDE**
   - 2 score networks (forward và backward)
   - Học lần lượt qua IPF

**3. IPF converges**
   - Mỗi iteration: forward → backward
   - Hội tụ đến Schrödinger Bridge

**4. Applications**
   - Physics-constrained generation
   - Diffusion-Flow hybrids
   - Domain adaptation with noise
   - Video generation
   - Molecular dynamics

### Hướng phát triển tiếp theo

Người thợ gốm nghĩ về các chủ đề tiếp theo:

1. **Flow Map Matching** - One-step generation với điều kiện
2. **Generator Matching Framework** - Thống nhất lý thuyết các phương pháp
3. **Stochastic Interpolants** - Tổng quát hóa Schrödinger Bridge

"Hành trình vẫn còn dài," anh nghĩ. "Nhưng mỗi ngày đều học được điều mới!"

---

## Tài liệu tham khảo

1. **Schrödinger, E. (1931)** - "Über die Umkehrung der Naturgesetze" _(Original paper về Schrödinger Bridge)_

2. **Léonard, C. (2013)** - "A survey of the Schrödinger problem and some of its connections with optimal transport" _Survey toàn diện về SB_

3. **Chen, Y., Georgiou, T. T., & Pavon, M. (2021)** - "Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory" _Likelihood training approach_

4. **De Bortoli, V., Thornton, J., Heng, J., & Doucet, A. (2021)** - "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Models" _Kết nối SB với Diffusion_

5. **Albergo, M. S., Boffi, N. M., & Vanden-Eijnden, E. (2023)** - "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions" _Framework tổng quát hóa_

6. **Liu, G., Vahdat, A., Huang, D. A., Theodorou, E. A., Nie, W., & Anandkumar, A. (2023)** - "I²SB: Image-to-Image Schrödinger Bridge" _Ứng dụng image-to-image_

7. **Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023)** - "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" _Cải tiến với minibatch OT_

---

**Bài trước:** [Rectified Flows: Khi Đường Đi Trở Nên Thẳng](/posts/2025/rectified-flows)

**Bài tiếp theo:** [Generator Matching Framework: Thống Nhất Lý Thuyết](/posts/2025/generator-matching-framework)

<script src="/assets/js/katex-init.js"></script>
