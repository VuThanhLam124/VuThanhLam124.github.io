---
title: "Conditional Flow Matching & Optimal Transport: Tùy biến dòng chảy theo ý khách"
date: "2025-03-08"
category: "flow-based-models"
tags: ["conditional-flow-matching", "flow-matching", "optimal-transport", "generative-models", "pytorch"]
excerpt: "Người thợ pha lê nay phải cá nhân hóa sản phẩm theo khách hàng. Conditional Flow Matching kết hợp Optimal Transport giúp mô hình hóa đường đi phụ thuộc điều kiện, với giải thích toán học chi tiết và code PyTorch."
author: "ThanhLamDev"
readingTime: 18
featured: false
---

# Conditional Flow Matching & Optimal Transport

**Câu chuyện tiếp theo: khách VIP bước vào xưởng pha lê với bảng yêu cầu chi tiết. Người thợ cần điều khiển dòng chảy dựa trên “điều kiện” mà mỗi khách đưa ra. Conditional Flow Matching (CFM) chính là cuốn sổ tay giúp anh ghép từng điều kiện với một quỹ đạo ánh sáng riêng, vẫn bám sát tối ưu vận chuyển.**

## Mục lục

1. [Câu chuyện: Đơn hàng theo điều kiện](#1-câu-chuyện-đơn-hàng-theo-điều-kiện)
2. [Trực giác về Conditional Flow Matching](#2-trực-giác-về-conditional-flow-matching)
3. [Toán học & Optimal Transport có điều kiện](#3-toán-học--optimal-transport-có-điều-kiện)
4. [Mục tiêu huấn luyện CFM](#4-mục-tiêu-huấn-luyện-cfm)
5. [Code PyTorch mẫu](#5-code-pytorch-mẫu)
6. [Kết hợp Optimal Transport để định tuyến tốt hơn](#6-kết-hợp-optimal-transport-để-định-tuyến-tốt-hơn)
7. [Mẹo thực nghiệm](#7-mẹo-thực-nghiệm)
8. [Kết nối series & tài liệu](#8-kết-nối-series--tài-liệu)

---

## 1. Câu chuyện: Đơn hàng theo điều kiện

Sau khi lưu được bản đồ Flow Map, xưởng pha lê nhận thêm dịch vụ mới: “Thiết kế theo thị hiếu từng vùng”. Khách từ Kyoto muốn hoa văn hoa anh đào; khách từ Dubai thích ánh vàng; mỗi điều kiện ($y$) kéo theo đường đi khác nhau từ khối pha lê chuẩn $z$ đến sản phẩm $x$.

Thay vì viết lại toàn bộ flow cho mỗi khách, người thợ cần một mô hình biết **chèn điều kiện** vào dòng chảy – vừa nhanh, vừa linh hoạt. Conditional Flow Matching cho phép anh học trực tiếp vận tốc phụ thuộc điều kiện, đồng thời tối ưu sao cho đường đi tiết kiệm năng lượng giống bài toán Optimal Transport.

## 2. Trực giác về Conditional Flow Matching

- **Flow Matching** (cơ bản): học trường vận tốc $v_t(x)$ để nối Gaussian với dữ liệu.
- **Conditional Flow Matching (CFM):** mở rộng sang $v_t(x \mid y)$, tức vận tốc phụ thuộc điều kiện $y$ (nhãn, văn bản, embedding...).
- **Ý tưởng:** xây dựng đường đi có điều kiện dễ định nghĩa, rồi huấn luyện mạng học lại vận tốc của đường đi đó mà không cần marginal hóa phức tạp.

Chúng ta vẫn coi thời gian $t \in [0,1]$. Khi $t=0$, mẫu nằm ở base distribution tùy vào $y$; khi $t=1$, mẫu khớp phân phối dữ liệu tương ứng điều kiện đó.

## 3. Toán học & Optimal Transport có điều kiện

### 3.1 Định nghĩa đường có điều kiện

Với điều kiện $y$, ta mô tả quỹ đạo bằng ODE:

$$
\frac{d x_t}{dt} = v_t(x_t \mid y), \qquad x_0 \sim p_0(x \mid y), \quad x_1 \sim p_1(x \mid y)
$$

**Chú thích:** $x_t$ là trạng thái tại thời gian $t$; $v_t$ là trường vận tốc phụ thuộc $y$; $p_0, p_1$ lần lượt là phân phối đầu/cuối tương ứng điều kiện.

### 3.2 Đường có điều kiện dễ định nghĩa

Thay vì trực tiếp mô hình hóa $p_t(x \mid y)$, ta xây dựng đường dẫn dựa trên cặp $(x_0, x_1)$ đã ghép với điều kiện:

$$
x_t = (1 - t) \, x_0 + t \, x_1
$$

**Chú thích:** Nội suy tuyến tính vẫn hoạt động khi cả $x_0$ và $x_1$ cùng ràng buộc bởi $y$. Với dữ liệu hình ảnh có điều kiện, $x_1$ là ảnh thực thuộc điều kiện, còn $x_0$ lấy từ Gaussian độc lập với $y$ (hoặc Gaussian đã shift theo embed $y$).

Từ quỹ đạo này, vận tốc “thật” là:

$$
u_t(x_t \mid x_0, x_1, y) = x_1 - x_0
$$

### 3.3 Kết nối Optimal Transport

Trong OT, ta tìm ánh xạ tối ưu $T_y$ đưa $x_0$ tới $x_1$ với chi phí tối thiểu. Nội suy tuyến tính chính là đường địa cực (geodesic) trong không gian Wasserstein khi cặp $(x_0, x_1)$ được ghép tối ưu. Nhờ thế, CFM mặc nhiên thừa hưởng tính “thẳng” tương tự Rectified Flow nhưng ở cấp độ điều kiện.

## 4. Mục tiêu huấn luyện CFM

### 4.1 Loss chính

Đặt mạng $v_\theta(x, t, y)$ xấp xỉ vận tốc. Ta huấn luyện bằng loss bình phương:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{y, x_0, x_1, t}\Big[\big\|v_\theta(x_t, t, y) - (x_1 - x_0)\big\|^2\Big]
$$

**Chú thích:** $x_t$ được dựng từ nội suy ở trên; kỳ vọng lấy trung bình trên điều kiện $y$, cặp mẫu $(x_0, x_1)$ và thời gian $t$.

### 4.2 Regularizer tính nhất quán

Để đảm bảo vận tốc không “lệch pha” khi ghép điều kiện, ta thêm regularizer identity hoặc consistency:

$$
\mathcal{L}_{\text{id}} = \mathbb{E}_{y, x_0}\big[\|v_\theta(x_0, 0, y) - (x_1 - x_0)\big\|^2\big]
$$

Hoặc đơn giản hơn, buộc $v_\theta(x, 0, y)$ gần 0 để tránh drift đầu kỳ.

### 4.3 Loss tổng

$$
\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}} + \lambda_{\text{reg}} \|v_\theta\|^2
$$

**Chú thích:** $\lambda_{\text{id}}, \lambda_{\text{reg}}$ là hệ số điều chỉnh; $\|v_\theta\|^2$ giúp regularize gradient.

## 5. Code PyTorch mẫu

### 5.1 Mạng vận tốc có điều kiện

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    freqs = torch.exp(
        -torch.arange(half_dim, device=t.device) * torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    )
    angles = t[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

class ConditionalVelocityNet(nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim=512, time_embed_dim=128):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ) for _ in range(4)
        ])
        self.output = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, cond):
        # x: [B, D], t: [B, 1], cond: [B, C]
        t_emb = sinusoidal_embedding(t.squeeze(-1), dim=128)
        t_ctx = self.time_proj(t_emb)
        c_ctx = self.cond_proj(cond)
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = h + t_ctx + c_ctx
            h = block(h)
        return self.output(h)
```

### 5.2 Huấn luyện

```python
def train_cfm(model, loader, epochs=200, lr=2e-4, device="cuda", lambda_id=0.05):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            x0, x1, cond = [b.to(device) for b in batch]
            bsz = x0.size(0)

            t = torch.rand(bsz, 1, device=device)
            xt = (1 - t) * x0 + t * x1

            target = x1 - x0
            pred = model(xt, t, cond)
            loss_match = F.mse_loss(pred, target)

            pred_id = model(x0, torch.zeros_like(t), cond)
            loss_id = F.mse_loss(pred_id, target.detach())

            loss = loss_match + lambda_id * loss_id
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            total += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | loss = {total / len(loader):.6f}")
```

### 5.3 Lấy mẫu có điều kiện

```python
@torch.no_grad()
def sample_cfm(model, cond, steps=8, device="cuda"):
    model.eval()
    cond = cond.to(device)
    bsz = cond.size(0)
    dim = model.output.out_features

    x = torch.randn(bsz, dim, device=device)  # z0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((bsz, 1), i * dt, device=device)
        velocity = model(x, t, cond)
        x = x + velocity * dt
    return x
```

## 6. Kết hợp Optimal Transport để định tuyến tốt hơn

- **OT coupling:** trước khi huấn luyện, ghép $(x_0, x_1)$ theo OT (ví dụ EMD trên latent) giúp đường đi tuyến tính phù hợp hơn.
- **Barycentric projection:** dùng OT để tìm map $T_y$ rồi đặt $x_t = (1 - t) x_0 + t T_y(x_0)$, giảm variance của mục tiêu.
- **Regularization:** thêm chi phí $\|x_t - \mu_t(y)\|^2$ với $\mu_t(y)$ là barycenter để giữ đường “thẳng” quanh quỹ đạo trung bình.

## 7. Mẹo thực nghiệm

- Chuẩn hóa điều kiện $y$ (hoặc dùng embedding từ mô hình ngôn ngữ) trước khi đưa vào mạng.
- Khi điều kiện dạng văn bản dài, nên dùng cross-attention thay vì concat đơn giản.
- Huấn luyện song song với Flow Map Matching để reuse cùng dataset cặp $(x_0, x_1)$.
- Nếu sampling nhiều bước bị drift, giảm learning rate hoặc tăng số residual block.
- Kết hợp classifier-free guidance bằng cách thêm điều kiện rỗng và trộn kết quả ở inference.

## 8. Kết nối series & tài liệu

Conditional Flow Matching giúp người thợ pha lê biến mỗi điều kiện thành một tuyến đường riêng, nhưng vẫn giữ tinh thần “đường ngắn nhất” của Optimal Transport. Bài tiếp theo về **Schrödinger Bridge** sẽ kể câu chuyện khi dòng chảy phải tuân theo ràng buộc động lực (có nhiễu, có nguyên tắc vật lý).

### Tài liệu nên đọc

1. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*.
2. Tong, Z., et al. (2024). *Conditional Flow Matching for Controlled Generation*.
3. Albergo, M. S., & Vanden-Eijnden, E. (2023). *Stochastic Interpolants and Conditional Paths*.
4. Courty, N., et al. (2017). *Optimal Transport for Domain Adaptation*.

---

<script src="/assets/js/katex-init.js"></script>
