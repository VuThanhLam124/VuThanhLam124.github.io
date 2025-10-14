---
title: "Generator Matching Framework: Bản đồ lý thuyết của mọi dòng chảy"
date: "2025-03-15"
category: "flow-based-models"
tags: ["generator-matching", "flow-matching", "diffusion-models", "score-matching", "theory"]
excerpt: "Sau hành trình qua RealNVP, Rectified Flow, CFM và Schrödinger Bridge, người thợ pha lê tổng kết một khung lý thuyết chung: Generator Matching. Bài viết giải thích trực giác, ngôn ngữ toán học, và cách ánh xạ giữa các mô hình generative."
author: "ThanhLamDev"
readingTime: 17
featured: false
---

# Generator Matching Framework

**Đây là chương “tổng kết” của xưởng pha lê: người thợ mở sổ tay, gom lại mọi kỹ thuật đã học. Generator Matching Framework (GMF) cho anh một bản đồ chung – nơi Flow Matching, Diffusion, Score Matching và GAN được nhìn như những cách khác nhau để khớp “generator” của dòng chảy.**

## Mục lục

1. [Câu chuyện: Tổng duyệt tại xưởng](#1-câu-chuyện-tổng-duyệt-tại-xưởng)
2. [Trực giác: Generator là gì?](#2-trực-giác-generator-là-gì)
3. [Khung toán học](#3-khung-toán-học)
4. [So sánh các mô hình trong GMF](#4-so-sánh-các-mô-hình-trong-gmf)
5. [Code mẫu cấu trúc “matcher”](#5-code-mẫu-cấu-trúc-matcher)
6. [Gợi ý ứng dụng & nghiên cứu](#6-gợi-ý-ứng-dụng--nghiên-cứu)
7. [Kết nối series & tài liệu](#7-kết-nối-series--tài-liệu)

---

## 1. Câu chuyện: Tổng duyệt tại xưởng

Trước buổi trình diễn lớn, người thợ pha lê cần hệ thống hóa toàn bộ phương pháp. Anh nhận ra: dù là RealNVP, Rectified Flow hay Schrödinger Bridge, ta luôn mô tả một **dòng chảy từ noise đến data**. Điểm khác biệt là ở cách mô hình mô tả **generator** – hàm (hoặc toán tử) xác định sự biến đổi theo thời gian.

GMF giúp anh so sánh, kết hợp và lựa chọn kỹ thuật phù hợp chỉ bằng việc xem xét generator và loss tương ứng.

## 2. Trực giác: Generator là gì?

- **Generator** $G_t$ là quy tắc biến đổi noise $Z$ thành mẫu tại thời gian $t$.
- Với ODE: generator gắn với trường vận tốc.
- Với SDE/diffusion: generator là toán tử Kolmogorov (Liouville operator).
- Với GAN: generator là mạng tạo ảnh duy nhất (không thời gian).

Ta muốn học $G_\theta$ sao cho quỹ đạo sinh ra tương đồng với quỹ đạo “chuẩn” $G^*$ (optimal). Mỗi phương pháp xây dựng loss khác nhau để ép $G_\theta$ khớp $G^*$.

## 3. Khung toán học

### 3.1 Generator như ánh xạ theo thời gian

$$
X_t = G_t(Z), \qquad Z \sim p_Z
$$

**Chú thích:** $p_Z$ thường là Gaussian; $G_t$ có thể là hàm hữu hạn (ODE) hoặc toán tử (SDE). Khi $t=1$, $G_1(Z)$ cần phân phối như dữ liệu.

### 3.2 Phương trình điều khiển generator

- **ODE generator:** $G_t$ thỏa $\frac{d}{dt} G_t(z) = v_t(G_t(z))$.
- **SDE generator:** $dG_t(z) = b_t(G_t(z)) dt + \sigma_t(G_t(z)) dW_t$.
- **Static generator (GAN):** chỉ định $G(Z)$ và tối ưu so với phân phối dữ liệu.

Ta định nghĩa loss tổng quát:

$$
\mathcal{L}(\theta) = \mathbb{E}_{Z, t}\Big[D\big(\mathcal{G}_\theta(Z,t), \mathcal{G}^*(Z,t)\big)\Big]
$$

**Chú thích:** $\mathcal{G}$ có thể là vận tốc, score, noise residual… tùy phương pháp; $D$ là khoảng cách (L2, KL, JS...).

### 3.3 Các lựa chọn $\mathcal{G}$

- Flow Matching: $\mathcal{G}$ là vận tốc $v_t$.
- Diffusion/Score: $\mathcal{G}$ là score $\nabla_x \log p_t$ hoặc noise $\epsilon$.
- Schrödinger Bridge: $\mathcal{G}$ là forward/backward score theo IPF.
- GAN: $\mathcal{G}$ là mẫu sinh trực tiếp, loss là phân biệt (JS/hinge).

## 4. So sánh các mô hình trong GMF

| Phương pháp | Generator $G_t$ | Quantity matching | Loss đặc trưng | Ghi chú |
|-------------|-----------------|-------------------|----------------|--------|
| **Flow Matching** | ODE tích phân vận tốc | $v_\theta(x,t)$ vs $v^*(x,t)$ | L2 (MSE) | CFM, Rectified Flow là biến thể |
| **Diffusion (DDPM)** | Giải ngược SDE | $\epsilon_\theta(x,t)$ vs noise | L2 | Training giống DSM |
| **Score Matching** | SDE forward/backward | $s_\theta(x,t)$ vs score | Fisher divergence | Bao gồm NCSN, SB |
| **GAN** | Hàm tĩnh $G(z)$ | $G_\theta(z)$ vs data | Adversarial | Không có thời gian, nhưng vẫn là GMF với $t=1$ |
| **Flow Map Matching** | Map $\phi_t$ | $\phi_\theta(x,t)$ vs interpolant | L2 | Dạng discrete-time của GMF |

GMF cho phép ta “nhìn” xem mỗi phương pháp đang matching quantity nào, từ đó dễ dàng lai ghép (ví dụ Flow Matching + noise prediction).

## 5. Code mẫu cấu trúc “matcher”

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorMatcher(nn.Module):
    def __init__(self, dim, mode="flow"):
        super().__init__()
        self.mode = mode
        self.net = UNet(dim=dim)  # giả sử đã định nghĩa UNet

    def forward(self, x, t, extra=None):
        if self.mode == "flow":
            return self.net(x, t)  # dự đoán vận tốc
        if self.mode == "diffusion":
            return self.net(x, t)  # dự đoán noise
        if self.mode == "score":
            return self.net(x, t)  # dự đoán score
        if self.mode == "gan":
            return self.net(x, torch.zeros_like(t))  # t không dùng
        if self.mode == "flow_map":
            return x + t * self.net(x, t)  # map trực tiếp
        raise ValueError("Unsupported mode")

    def loss(self, batch):
        mode = self.mode
        x0, x1, t, aux = batch["x0"], batch["x1"], batch["t"], batch.get("aux")
        if mode == "flow":
            target = x1 - x0
            pred = self.forward((1 - t) * x0 + t * x1, t, aux)
            return F.mse_loss(pred, target)
        if mode == "diffusion":
            noise = batch["noise"]
            pred = self.forward(batch["noisy"], t, aux)
            return F.mse_loss(pred, noise)
        if mode == "score":
            target = batch["score_target"]
            pred = self.forward(batch["noisy"], t, aux)
            return F.mse_loss(pred, target)
        if mode == "gan":
            # Phần loss adversarial được định nghĩa bên ngoài
            return None
        if mode == "flow_map":
            xt = (1 - t) * x0 + t * x1
            pred = self.forward(x0, t, aux)
            return F.mse_loss(pred, xt)
```

## 6. Gợi ý ứng dụng & nghiên cứu

- **Lai ghép:** kết hợp loss Flow Matching và Diffusion để tận dụng điểm mạnh đôi bên.
- **Curriculum:** bắt đầu bằng Flow Map Matching (đường thẳng), sau đó thêm noise (Schrödinger Bridge) – tất cả nằm trong GMF.
- **Phân tích lý thuyết:** GMF giúp chứng minh hội tụ bằng cách xem xét generator như toán tử tạo semigroup.
- **Tooling:** xây dựng pipeline tái sử dụng mạng chính, chỉ thay loss và scheduler.

## 7. Kết nối series & tài liệu

Generator Matching Framework chính là tấm bản đồ tổng quát cho xưởng pha lê: mọi kỹ thuật trước đó đều là trường hợp riêng của việc “điều khiển generator”. Từ đây, người thợ có thể sáng tạo mô hình mới bằng cách chọn quantity muốn matching. Bạn có thể tiếp tục khám phá các biến thể mới như **flow matching in latent space**, **consistency models**, hay **generator matching với mô phỏng vật lý**.

### Tài liệu nên đọc

1. Bauer, M., et al. (2023). *Generator Matching: A Unified Framework for Generative Modeling*.
2. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*.
3. Song, Y., et al. (2021). *Score-Based Generative Modeling through SDEs*.
4. Ho, J., et al. (2020). *Denoising Diffusion Probabilistic Models*.

---

<script src="/assets/js/katex-init.js"></script>
