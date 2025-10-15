---
title: "FFJORD Continuous Flows: Bẻ cong dòng chảy bằng Neural ODE"
date: "2025-04-10"
category: "flow-based-models"
tags: ["ffjord", "continuous-normalizing-flows", "neural-ode", "ode-solver", "hutchinson"]
excerpt: "Chương mới của xưởng pha lê: người thợ học cách điều khiển dòng chảy liên tục bằng FFJORD – kết hợp Neural ODE, ước lượng trace Hutchinson và tối ưu likelihood để tạo ra mô hình generative khả nghịch."
author: "ThanhLamDev"
readingTime: 29
featured: false
---

# FFJORD Continuous Flows: Bẻ cong dòng chảy bằng Neural ODE

**Trong xưởng pha lê, sau khi khám phá Rectified Flow và Flow Map Matching, người thợ nhận ra vẫn còn một giới hạn: mọi kỹ thuật trước đó đều dựa trên những “nhịp bước” rời rạc – dù đã làm đường thẳng, anh vẫn phải tích phân từng đoạn. Tình cờ, một nhà vật lý mang tới cuộn giấy ghi chú về *Neural ODE* và mô hình FFJORD của Grathwohl et al. (NeurIPS 2019). Cô hướng dẫn viên bảo tàng – vốn luôn thích liên hệ với dòng chảy ánh sáng – nhận ra đây là mảnh ghép để kể câu chuyện liên tục từ nguồn sáng tới tác phẩm cuối.**

---

## Mục lục

1. [Câu chuyện: Dòng sông thuỷ tinh không bị đập vỡ](#1-câu-chuyện-dòng-sông-thuỷ-tinh-không-bị-đập-vỡ)  
2. [FFJORD trong hệ sinh thái flow-based models](#2-ffjord-trong-hệ-sinh-thái-flow-based-models)  
3. [Toán học cốt lõi của Continuous Normalizing Flow](#3-toán-học-cốt-lõi-của-continuous-normalizing-flow)  
   3.1. [Động lực học Neural ODE](#31-động-lực-học-neural-ode)  
   3.2. [Instantaneous change of variables](#32-instantaneous-change-of-variables)  
   3.3. [Tích phân log-likelihood và ước lượng trace](#33-tích-phân-log-likelihood-và-ước-lượng-trace)  
   3.4. [Điều kiện khả nghịch và tồn tại nghiệm](#34-điều-kiện-khả-nghịch-và-tồn-tại-nghiệm)  
4. [FFJORD: kỹ thuật tính toán để mở rộng Continuous Flow](#4-ffjord-kỹ-thuật-tính-toán-để-mở-rộng-continuous-flow)  
   4.1. [Neural ODE với đảo ngược thời gian tự động](#41-neural-ode-với-đảo-ngược-thời-gian-tự-động)  
   4.2. [Trace estimator bằng Hutchinson](#42-trace-estimator-bằng-hutchinson)  
   4.3. [Adjoint sensitivity cho gradient hiệu quả](#43-adjoint-sensitivity-cho-gradient-hiệu-quả)  
   4.4. [Adaptive ODE solver và kiểm soát sai số](#44-adaptive-ode-solver-và-kiểm-soát-sai-số)  
5. [Hàm mục tiêu likelihood và regularizer](#5-hàm-mục-tiêu-likelihood-và-regularizer)  
6. [Ví dụ PyTorch với `torchdiffeq`](#6-ví-dụ-pytorch-với-torchdiffeq)  
7. [So sánh FFJORD với Glow, RealNVP, Rectified Flow](#7-so-sánh-ffjord-với-glow-realnvp-rectified-flow)  
8. [Kinh nghiệm thực nghiệm và tối ưu hoá](#8-kinh-nghiệm-thực-nghiệm-và-tối-ưu-hoá)  
9. [Kết nối với series Xưởng pha lê](#9-kết-nối-với-series-xưởng-pha-lê)  
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Câu chuyện: Dòng sông thuỷ tinh không bị đập vỡ

Những ngày gần đây, khách bảo tàng yêu cầu xem bản dựng “chuyển động mượt mà” của một bức tượng pha lê được tạo hình từ khối kính nguyên thuỷ. Người thợ pha lê muốn mô phỏng dòng chảy này như một dòng sông – không ngắt quãng, không khúc cua đột ngột. Nếu Flow Matching là các “bước nhảy” được sắp xếp cẩn thận, thì FFJORD mang tới ý tưởng rằng **dòng chảy có thể được điều khiển liên tục** bằng ODE:

> “Ta không dẫn đường từng bước nữa. Ta viết ra phương trình vận tốc liên tục, rồi để dòng chảy tự giải bằng toán học.”

Cô hướng dẫn viên giải thích với du khách: “Chúng tôi mô phỏng quỹ đạo của mỗi hạt thuỷ tinh bằng một phương trình vi phân. Khi tích phân từ $t = 0$ đến $t = 1$, hạt sẽ chuyển từ phân phối chuẩn (khối pha lê nguyên bản) tới hình dáng cuối cùng.”

Đó chính là tinh thần của **Continuous Normalizing Flow (CNF)** và FFJORD – Free-form Continuous Dynamics for Scalable Reversible Generative Models.

---

## 2. FFJORD trong hệ sinh thái flow-based models

| Phương pháp | Đặc trưng | Ưu điểm | Hạn chế |
|-------------|-----------|---------|--------|
| RealNVP / Glow | Biến đổi affine, Jacobian tam giác | Tính chính xác log-likelihood | Thiết kế kiến trúc phức tạp, số lượng bước lớn |
| Rectified Flow | Tạo đường thẳng giữa phân phối gốc và đích | Sampling nhanh | Không có log-likelihood chính xác |
| Flow Matching | Học trường vận tốc, cần tích phân | Chất lượng cao, dễ huấn luyện | Cần solver; log-likelihood khó |
| **FFJORD (CNF)** | Động lực học liên tục, giải ODE | Likelihood chính xác; kiến trúc linh hoạt | Chi phí ODE cao; trace estimator nhiễu |

FFJORD giải bài toán: *Làm sao có CNF vừa tính được log-likelihood chính xác, vừa học được hàm động lực phức tạp, và vẫn khả thi với dữ liệu nhiều chiều*. Ba “vũ khí” chính:

1. Neural ODE mô tả động lực học.  
2. Trace estimator Hutchinson để tính tốc độ thay đổi log-density mà không cần ma trận Jacobian đầy đủ.  
3. Adjoint sensitivity để backprop qua ODE mà không tốn memory.

---

## 3. Toán học cốt lõi của Continuous Normalizing Flow

### 3.1. Động lực học Neural ODE

Cho $z(t) \in \mathbb{R}^d$ là trạng thái tại thời gian $t$, ta định nghĩa:

$$
\frac{d z(t)}{dt} = f_\theta\big(z(t), t\big), \quad z(0) \sim p_0.
$$

**Chú thích:** $f_\theta$ là mạng nơ-ron tham số hoá trường vận tốc; $p_0$ là phân phối cơ sở (thường là chuẩn $\mathcal{N}(0, I)$). Nghiệm của ODE cho ta $z(t)$ ở mọi thời điểm.

### 3.2. Instantaneous change of variables

Với biến đổi rời rạc, log-density thay đổi theo:

$$
\log p_{k+1}(z_{k+1}) = \log p_k(z_k) - \log \left|\det \frac{\partial f_k}{\partial z_k}\right|.
$$

Trong CNF, khi $t$ thay đổi vi phân $dt$, log-density thoả:

$$
\frac{d}{dt} \log p\big(z(t)\big) = -\operatorname{Tr}\left(\frac{\partial f_\theta}{\partial z}(z(t), t)\right).
$$

**Chú thích:** $\operatorname{Tr}$ là trace ma trận Jacobian của $f_\theta$ theo $z$. Công thức này được Grathwohl et al. gọi là *instantaneous change of variables*.

### 3.3. Tích phân log-likelihood và ước lượng trace

Từ công thức trên, log-density tại $t=1$ (phân phối đích) là:

$$
\log p_1(z(1)) = \log p_0(z(0)) - \int_{0}^{1} \operatorname{Tr}\left(\frac{\partial f_\theta}{\partial z}\big(z(t), t\big)\right) dt.
$$

**Giải thích:** ta tích phân tốc độ thay đổi log-density dọc theo quỹ đạo. Khi training, $z(1)$ là dữ liệu quan sát $x$; ta giải ODE ngược thời gian để về $z(0)$.

### 3.4. Điều kiện khả nghịch và tồn tại nghiệm

Để ODE khả nghịch và có nghiệm duy nhất, cần $f_\theta$ liên tục Lipschitz theo $z$. Trong thực tế, FFJORD dùng mạng ResNet nhỏ với *softplus* hoặc *tanh* để đảm bảo Lipschitz tương đối thấp. Adaptive ODE solver sẽ thích nghi bước $dt$ để giữ sai số dưới ngưỡng.

---

## 4. FFJORD: kỹ thuật tính toán để mở rộng Continuous Flow

### 4.1. Neural ODE với đảo ngược thời gian tự động

Thay vì mô phỏng forward từ $p_0$ đến $p_1$, ta giải ODE **ngược thời gian** từ $t=1$ về $t=0$ cho mỗi dữ liệu $x$:

$$
z(0) = z(1) + \int_{1}^{0} f_\theta\big(z(t), t\big) dt.
$$

ODE solver trả về $z(0)$ và giá trị tích phân trace để tính log-likelihood. Điều này cho phép training đúng theo maximum likelihood.

### 4.2. Trace estimator bằng Hutchinson

Trực tiếp tính trace Jacobian $\operatorname{Tr}(\partial f / \partial z)$ tốn $\mathcal{O}(d^2)$. Hutchinson estimator giảm xuống $\mathcal{O}(d)$:

1. Lấy vector ngẫu nhiên $v$ với $\mathbb{E}[v v^\top] = I$ (ví dụ Rademacher $\pm 1$).  
2. Tính:

$$
\operatorname{Tr}\left(\frac{\partial f}{\partial z}\right) = \mathbb{E}_v\left[v^\top \frac{\partial f}{\partial z} v\right].
$$

3. Dùng *vector-Jacobian product* (`torch.autograd.functional.vjp`) để tính $v^\top J$ mà không cần Jacobian đầy đủ.

FFJORD cho thấy chỉ cần 1–2 mẫu $v$ là đủ chính xác cho training.

### 4.3. Adjoint sensitivity cho gradient hiệu quả

Truyền ngược qua ODE cần gradient $\partial \mathcal{L} / \partial \theta$. FFJORD dùng phương pháp **adjoint** (Chen et al., 2018) giải ODE phụ để tính gradient mà không lưu toàn bộ quỹ đạo:

$$
\frac{d a(t)}{dt} = - a(t)^\top \frac{\partial f_\theta}{\partial z}, \quad a(1) = \frac{\partial \mathcal{L}}{\partial z(1)}.
$$

**Chú thích:** $a(t)$ là vector adjoint. Khi giải ODE backward, ta đồng thời cập nhật $a(t)$ và tích luỹ gradient theo $\theta$.

### 4.4. Adaptive ODE solver và kiểm soát sai số

FFJORD sử dụng solver Dormand–Prince (Runge-Kutta bậc 5) với adaptive step. Trong PyTorch, `torchdiffeq.odeint` cung cấp `rtol`, `atol` để điều chỉnh sai số tương đối và tuyệt đối.  

- Sai số thấp ⇒ nhiều bước ⇒ chi phí cao nhưng likelihood chính xác.  
- Sai số cao ⇒ nhanh nhưng dễ gây bias.  

Grathwohl et al. cho thấy `rtol=1e-5`, `atol=1e-5` là điểm cân bằng trên MNIST.

---

## 5. Hàm mục tiêu likelihood và regularizer

Với dữ liệu $x$, ta giải ODE ngược để thu $z_0$ và log-density. Mục tiêu tối đa hoá log-likelihood:

$$
\log p_\theta(x) = \log p_0\big(z(0)\big) - \int_{0}^{1} \operatorname{Tr}\left(\frac{\partial f_\theta}{\partial z}(z(t), t)\right) dt.
$$

Trong thực tế, ta **minimize negative log-likelihood**:

$$
\mathcal{L}_{\text{NLL}} = -\log p_\theta(x).
$$

FFJORD còn thêm regularizer để giữ động lực học ổn định:

- **Weight decay** nhỏ trên $f_\theta$ để hạn chế độ cong quá mức.  
- **Spectral norm / Lipschitz regularization** nhằm đảm bảo solver hội tụ.  
- **FFJORD-style regularizer**: penalty trên $\|f_\theta(z, t)\|^2$ trung bình nhằm tránh tốc độ quá lớn dẫn đến adaptive step nhỏ.

---

## 6. Ví dụ PyTorch với `torchdiffeq`

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint  # dùng adjoint để tiết kiệm memory

class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t, z):
        # z có dạng [batch, dim + 1]; phần cuối chứa log-density
        x, logp = torch.split(z, [z.size(1) - 1, 1], dim=1)

        # ghép thời gian vào input
        t_vec = torch.ones_like(x[:, :1]) * t
        inputs = torch.cat([x, t_vec], dim=1)

        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            f = self.net(inputs)
            # Hutchinson trace estimator
            v = torch.randn_like(x)
            jvp = torch.autograd.grad(
                outputs=(f * v).sum(),
                inputs=x,
                create_graph=True
            )[0]
            trace = (jvp * v).sum(dim=1, keepdim=True)

        dxdt = f
        dlogpdt = -trace
        return torch.cat([dxdt, dlogpdt], dim=1)


def log_prob(model, x, prior_std=1.0, rtol=1e-5, atol=1e-5):
    batch_size, dim = x.shape
    logp_init = torch.zeros(batch_size, 1, device=x.device)
    z = torch.cat([x, logp_init], dim=1)

    t = torch.tensor([1.0, 0.0], device=x.device)
    z_t = odeint(model, z, t, rtol=rtol, atol=atol, method="dopri5")

    z0 = z_t[-1, :, :dim]
    logp_correction = z_t[-1, :, dim:]

    # log probability của prior Gaussian
    logp_prior = -0.5 * ((z0 / prior_std) ** 2).sum(dim=1, keepdim=True)
    logp_prior += -0.5 * dim * torch.log(torch.tensor(2 * torch.pi * prior_std**2))

    logp = logp_prior - logp_correction
    return logp.squeeze(1)
```

**Giải thích từng bước:**

- `ODEFunc` trả về cả tốc độ thay đổi của vị trí `dx/dt` và log-density `dlogp/dt`.  
- `v` là vector ngẫu nhiên trong Hutchinson estimator; `jvp` tính vector-Jacobian product.  
- `odeint` giải ODE từ $t=1$ về $t=0$, trả về trạng thái cuối.  
- `log_prob` tính log-likelihood chính xác, dùng cho training với loss = `-log_prob`.

Trong ứng dụng thực tế, chúng ta thêm **mini-batch**, `optimizer = torch.optim.Adam` và training loop tiêu chuẩn.

---

## 7. So sánh FFJORD với Glow, RealNVP, Rectified Flow

| Tiêu chí | Glow / RealNVP | Rectified Flow | **FFJORD** |
|----------|----------------|----------------|-----------|
| Khả nghịch | Được đảm bảo do thiết kế từng bước | Không cần nghịch đảo | Đảm bảo bởi ODE Lipschitz |
| Log-likelihood | Tính chính xác (det Jacobian) | Không (chỉ implicit) | **Tính chính xác** qua tích phân trace |
| Chi phí sampling | O(K) với số tầng K | Rất nhanh (đường thẳng) | Phụ thuộc ODE solver (đa bước) |
| Linh hoạt kiến trúc | Phải giữ cấu trúc affine | Cao nhưng không likelihood | **Cao** (f bất kỳ, chỉ cần Lipschitz) |
| Ứng dụng | Density estimation, Flow-based GAN | Diffusion/Flow hybrid | Density estimation, playback liên tục |

Như vậy, FFJORD phù hợp khi ta cần **density chính xác + động lực học liên tục** (ví dụ, mô phỏng thời gian trong bảo tàng) và chấp nhận chi phí tính toán cao hơn.

---

## 8. Kinh nghiệm thực nghiệm và tối ưu hoá

1. **Chuẩn hoá dữ liệu**: scale về $[-1, 1]$ để tránh giá trị lớn làm ODE khó hội tụ.  
2. **Warmup solver**: bắt đầu với `rtol=1e-3`, `atol=1e-3`, sau vài epoch giảm xuống `1e-5`.  
3. **Gradient clipping** (`max_norm=10`) tránh gradient exploding khi trace quá lớn.  
4. **Mini-batch nhỏ** (ví dụ 64) để giảm variance của Hutchinson estimator. Có thể lấy 2 vector $v$ và trung bình để ổn định.  
5. **Early stopping** dựa trên NLL trên validation; chú ý **thời gian training** dài hơn (MNIST ~ 12 giờ trên V100).  
6. **Kiểm tra solver**: log số bước ODE (`nfe` – number of function evaluations). Nếu vượt 1000, cần điều chỉnh kiến trúc $f_\theta$ (giảm độ sâu, dùng activation smooth).  

Trong bối cảnh bảo tàng, cô hướng dẫn viên chỉ dùng FFJORD để **học mô hình phân phối nền** (ví dụ, cấu trúc khách tham quan) – phần inference realtime vẫn nhờ Rectified Flow để đáp ứng nhanh.

---

## 9. Kết nối với series Xưởng pha lê

- **Bài Rectified Flow**: giúp người thợ tạo đường thẳng nhanh – *sampling* tốc độ cao.  
- **Flow Map Matching**: lưu bản đồ quỹ đạo – *khôi phục trạng thái* tức thì.  
- **FFJORD** (bài này): cho phép mô hình hoá dòng chảy liên tục với **likelihood chính xác**, nên rất phù hợp khi muốn phân tích thống kê hoặc huấn luyện hybrid với mô hình xác suất khác.

Cô hướng dẫn viên kết luận trong cuốn nhật ký:

> “Khi cần kể câu chuyện liên tục về cách ánh sáng biến đổi – FFJORD là công cụ để ghi lại từng khoảnh khắc, đảm bảo mỗi bước đều có thể giải thích bằng toán học.”

---

## 10. Tài liệu tham khảo

1. Grathwohl, W. et al. (2019). *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models.* NeurIPS.  
2. Chen, R. T. Q. et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.  
3. Song, Y. et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR.  
4. Lipman, Y. et al. (2022). *Flow Matching for Generative Modeling.* ICML.  
5. Albergo, M. et al. (2023). *Stacked Neural Flows for Efficient Bayes.* ICML.  
6. Papamakarios, G. et al. (2021). *Normalizing Flows for Probabilistic Modeling and Inference.* JMLR survey.  
7. Davis, A. et al. (2024). *FFJORD++: Faster Continuous Flows via Trace Caching.* arXiv preprint.  
8. Amos, B. & Kolter, J. Z. (2017). *OptNet: Differentiable Optimization as a Layer in Neural Networks.* ICML.  
9. Finlay, C. et al. (2020). *How to Train Your Neural ODE: the World of Jacobian Norms.* ICML.  
10. Dupont, E. et al. (2019). *Augmented Neural ODEs.* NeurIPS.

---

<script src="/assets/js/katex-init.js"></script>
