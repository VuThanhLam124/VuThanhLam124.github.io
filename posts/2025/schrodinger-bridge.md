---
title: "Schrödinger Bridge & Flow-based Models: Khi dòng chảy phải tôn trọng nhiễu"
date: "2025-03-12"
category: "flow-based-models"
tags: ["schrodinger-bridge", "optimal-transport", "stochastic-process", "flow-based-models", "pytorch"]
excerpt: "Tiếp nối câu chuyện xưởng pha lê: Schrödinger Bridge giúp người thợ dựng dòng chảy ngắn nhất nhưng phải đi qua môi trường nhiễu. Bài viết giải thích trực giác, toán học forward–backward SDE, IPF, cùng code PyTorch minh họa."
author: "ThanhLamDev"
readingTime: 19
featured: false
---

# Schrödinger Bridge & Flow-based Models

**Sau Conditional Flow Matching, người thợ pha lê được mời tham gia một triển lãm ngoài trời, nơi gió lạnh và hạt bụi khiến dòng ánh sáng luôn nhiễu. Schrödinger Bridge (SB) là cách anh bảo đảm sản phẩm vẫn tới đích đúng cách – dù phải đương đầu với chuyển động ngẫu nhiên.**

## Mục lục

1. [Câu chuyện: Dòng pha lê giữa sương mù](#1-câu-chuyện-dòng-pha-lê-giữa-sương-mù)
2. [Trực giác: Entropic Optimal Transport & Bridge](#2-trực-giác-entropic-optimal-transport--bridge)
3. [Toán học nền tảng](#3-toán-học-nền-tảng)
4. [Thuật toán IPF (Iterative Proportional Fitting)](#4-thuật-toán-ipf-iterative-proportional-fitting)
5. [Code PyTorch mẫu](#5-code-pytorch-mẫu)
6. [Ứng dụng trong Flow-based Models](#6-ứng-dụng-trong-flow-based-models)
7. [Mẹo thực nghiệm](#7-mẹo-thực-nghiệm)
8. [Kết nối series & tài liệu](#8-kết-nối-series--tài-liệu)

---

## 1. Câu chuyện: Dòng pha lê giữa sương mù

Tại triển lãm, người thợ phải dẫn ánh sáng qua một gian phòng đầy sương – tượng trưng cho nhiễu Gaussian. Anh vẫn muốn tuyến đường tối ưu như Rectified Flow, nhưng giờ phải **tuân thủ động lực nhiễu**: dòng chảy không thể hoàn toàn xác định, mà phải phù hợp với chuyển động ngẫu nhiên nền (Brownian motion).

Schrödinger Bridge chính là bài toán tìm “dòng chảy khả dĩ nhất” giữa hai phân phối khi ta biết rằng môi trường có nhiễu xác định trước. Nó mở ra cách kết hợp Flow Matching với các ràng buộc vật lý/phân tán.

## 2. Trực giác: Entropic Optimal Transport & Bridge

- **Optimal Transport (OT)**: tìm đường đi ngắn nhất (geodesic) giữa $p_0$ và $p_1$.
- **Schrödinger Bridge**: thêm điều kiện “đường đi phải giống với một quá trình ngẫu nhiên tham chiếu”, ví dụ Brownian motion. Ta cực tiểu **KL divergence** giữa quá trình thực và quá trình tham chiếu.
- **Entropic OT**: OT với regularization entropy; khi entropy lớn, đường đi “mềm” hơn – chính là cầu Schrödinger.

## 3. Toán học nền tảng

### 3.1 Bài toán tối ưu

Cho quá trình tham chiếu $\pi_{[0,1]}$ (thường là Brownian motion). Ta tìm phân phối đường đi $p_{[0,1]}$ thỏa:

$$
\min_{p_{[0,1]}} \mathrm{KL}\big(p_{[0,1]} \,\|\, \pi_{[0,1]}\big)
$$

với ràng buộc biên:

$$
p_{[0,1]}|_{t=0} = p_0, \qquad p_{[0,1]}|_{t=1} = p_1
$$

**Chú thích:** $\mathrm{KL}$ là Kullback–Leibler divergence giữa hai quá trình; $p_{[0,1]}$ ký hiệu cho phân phối toàn bộ quỹ đạo từ $t=0$ tới $t=1$.

### 3.2 Dạng SDE hai chiều (forward–backward)

SB có thể mô tả bằng cặp SDE:

$$
dx_t = b_t^f(x_t)\,dt + \sigma\,dW_t, \qquad dx_t = b_t^b(x_t)\,dt + \sigma\,d\bar{W}_t
$$

**Chú thích:** $b_t^f, b_t^b$ là drift forward/backward; $W_t, \bar{W}_t$ là Brownian motion độc lập; $\sigma$ là độ mạnh nhiễu.

Drift liên hệ với score (gradient của log-density):

$$
b_t^f(x) = b_t^{\text{ref}}(x) + \sigma^2 \nabla_x \log \varphi_t(x), \qquad
b_t^b(x) = b_t^{\text{ref}}(x) + \sigma^2 \nabla_x \log \hat{\varphi}_t(x)
$$

với $\varphi_t, \hat{\varphi}_t$ là “potentials” giải phương trình Fokker–Planck và đóng vai trò giống forward/backward score.

### 3.3 Liên hệ entropic OT

SB tương đương bài toán entropic OT với chi phí $c(x,y) = \frac{\|x - y\|^2}{2\sigma^2}$:

$$
\min_{\gamma \in \Pi(p_0,p_1)} \int c(x,y) \, d\gamma(x,y) + \epsilon\,\mathrm{KL}\big(\gamma \,\|\, \pi_0 \otimes \pi_1\big)
$$

**Chú thích:** $\gamma$ là plan ghép hai phân phối; $\epsilon$ đóng vai trò entropy weight. Khi $\epsilon \to 0$, ta quay về OT chuẩn.

## 4. Thuật toán IPF (Iterative Proportional Fitting)

### 4.1 Quy trình

1. **Khởi tạo** bằng quá trình tham chiếu.
2. **Forward update:** điều chỉnh để khớp phân phối đích $p_1$.
3. **Backward update:** điều chỉnh để khớp phân phối nguồn $p_0$.
4. Lặp cho tới khi hội tụ.

Pseudo-code:

```text
Initialize process Q^0 = reference diffusion
for k = 0, 1, ...:
    Forward step:  Q^{k+1/2} = Condition(Q^k, end = p1)
    Backward step: Q^{k+1}   = Condition(Q^{k+1/2}, start = p0)
```

### 4.2 Học bằng Score Matching

Ta học hai mạng score:

$$
s_\theta^f(x,t) \approx \nabla_x \log p_t^f(x), \qquad
s_\phi^b(x,t) \approx \nabla_x \log p_t^b(x)
$$

Loss dạng denoising score matching (DSM) giống diffusion models. Các mạng này được cập nhật lần lượt trong mỗi vòng IPF.

## 5. Code PyTorch mẫu

### 5.1 Mạng score chia sẻ cấu trúc

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def time_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device) * torch.log(torch.tensor(10000.0)) / (half - 1)
    )
    angles = t[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

class ScoreNet(nn.Module):
    def __init__(self, dim, hidden=512, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(time_embedding(t.squeeze(-1)))
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        h = h + t_emb
        h = self.net[3](h)
        h = self.net[4](h)
        h = self.net[5](h)
        h = self.net[6](h) + t_emb
        return self.net[7](h)
```

### 5.2 Vòng IPF rút gọn

```python
@torch.no_grad()
def euler_maruyama_step(x, t, dt, score_net, sigma):
    noise = torch.randn_like(x)
    score = score_net(x, t)
    drift = sigma**2 * score
    return x + drift * dt + sigma * torch.sqrt(dt) * noise

def train_schrodinger_bridge(p0_loader, p1_loader, dim, device="cuda",
                             ipf_iters=5, epochs_per_iter=40, sigma=1.0):
    score_f = ScoreNet(dim).to(device)
    score_b = ScoreNet(dim).to(device)

    for it in range(ipf_iters):
        print(f"\n=== IPF iteration {it+1}/{ipf_iters} ===")

        # Forward score
        opt_f = torch.optim.AdamW(score_f.parameters(), lr=2e-4)
        for epoch in range(epochs_per_iter):
            for x0 in p0_loader:
                x0 = x0[0].to(device)
                t = torch.rand(x0.size(0), 1, device=device)
                x = euler_maruyama_step(x0, t, dt=0.01, score_net=score_b, sigma=sigma)
                noise = torch.randn_like(x)
                x_noisy = x + 0.02 * noise
                score_pred = score_f(x_noisy, t)
                score_target = -noise / 0.02
                loss = F.mse_loss(score_pred, score_target)
                opt_f.zero_grad()
                loss.backward()
                opt_f.step()

        # Backward score
        opt_b = torch.optim.AdamW(score_b.parameters(), lr=2e-4)
        for epoch in range(epochs_per_iter):
            for x1 in p1_loader:
                x1 = x1[0].to(device)
                t = torch.rand(x1.size(0), 1, device=device)
                x = euler_maruyama_step(x1, 1 - t, dt=0.01, score_net=score_f, sigma=sigma)
                noise = torch.randn_like(x)
                x_noisy = x + 0.02 * noise
                score_pred = score_b(x_noisy, t)
                score_target = -noise / 0.02
                loss = F.mse_loss(score_pred, score_target)
                opt_b.zero_grad()
                loss.backward()
                opt_b.step()

    return score_f, score_b
```

### 5.3 Lấy mẫu (Bridge sampling)

```python
@torch.no_grad()
def sample_bridge(score_f, score_b, num_steps=50, sigma=1.0, device="cuda"):
    x = torch.randn(1, score_f.net[-1].out_features, device=device)
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((1, 1), k * dt, device=device)
        score = score_f(x, t)
        x = x + sigma**2 * score * dt + sigma * torch.sqrt(dt) * torch.randn_like(x)
    return x
```

## 6. Ứng dụng trong Flow-based Models

- **Flow Matching with noise:** dùng SB để sinh dữ liệu có nhiễu bắt buộc (ví dụ chuyển động chất lỏng, video).
- **Diffusion + Flow hybrid:** SB đóng vai trò layer entropic, kết hợp với Rectified Flow để cân bằng tốc độ (flow) và độ linh hoạt (diffusion).
- **Domain adaptation:** coi SB như đường vận chuyển xác suất từ domain nguồn sang domain đích dưới nhiễu chung.

## 7. Mẹo thực nghiệm

- Chuẩn hóa dữ liệu trước khi giải SB để Brownian có variance phù hợp.
- Dùng số bước Euler nhỏ (dt 0.01) trong IPF để tránh sai số tích lũy.
- Chia sẻ trọng số giữa score forward/backward giúp tiết kiệm tham số nhưng cần locking cẩn thận.
- Theo dõi KL hoặc năng lượng để đánh giá hội tụ IPF.
- Khi sử dụng trong mô hình lớn, kết hợp với EMA (exponential moving average) cho score nets để ổn định.

## 8. Kết nối series & tài liệu

Schrödinger Bridge mang chúng ta từ flow tuyến tính sang những dòng chảy chịu ràng buộc nhiễu, mở đường cho các mô hình lai diffusion-flow hiện đại. Bước tiếp theo trong series là **Generator Matching Framework**, nơi người thợ tổng hợp lại toàn bộ lý thuyết để hiểu mối quan hệ giữa flow, diffusion và GAN.

### Tài liệu tham khảo

1. Léonard, C. (2013). *A survey of the Schrödinger problem and some of its connections with optimal transport*.
2. Chen, Y., et al. (2022). *Schrödinger Bridge Meets Flow Matching*.
3. De Bortoli, V., et al. (2021). *Diffusion Schrödinger Bridge with Applications to Score-Based Models*.
4. Pavon, M., & Tabak, E. (2013). *Hamilton-Jacobi-Bellman approach to optimal transport*.

---

<script src="/assets/js/katex-init.js"></script>
