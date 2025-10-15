---
title: "Diffusion Models & Score-based Generation: Khởi đầu hành trình"
date: "2025-03-20"
category: "diffusion-models"
tags: ["diffusion-models", "score-matching", "denoising", "generative-models", "pytorch"]
excerpt: "Bài mở màn series Diffusion Models & Score-based Generation: câu chuyện xưởng pha lê giữa màn sương, trực giác forward/backward process, score matching, loss DDPM và code PyTorch nền tảng."
author: "ThanhLamDev"
readingTime: 20
featured: false
---

# Diffusion Models & Score-based Generation

**Nhân vật chính lần này là một họa sĩ trẻ mê khám phá. Anh từng học cùng người thợ pha lê về Flow Matching, nhưng khi trở về thành phố thì phòng tranh bị phủ bụi sương do hệ thống thông gió trục trặc: mỗi lần khung tranh mở ra, lớp màu gốc bị nhiễu đi đôi chút. Anh phải học cách “vẽ trong sương” – hiểu forward diffusion để rồi khôi phục bức tranh bằng reverse diffusion. Nếu bạn chưa theo dõi [series Flow-based Models](/content.html#flow-based-models), hãy xem lại để nắm nền tảng trước khi đồng hành cùng anh họa sĩ.**

## Mục lục

1. [Câu chuyện: Xưởng pha lê giữa màn sương](#1-câu-chuyện-xưởng-pha-lê-giữa-màn-sương)
2. [Trực giác: Forward và Reverse Diffusion](#2-trực-giác-forward-và-reverse-diffusion)
3. [Toán học nền tảng](#3-toán-học-nền-tảng)
4. [Markov Chains, MM & HMM trong diffusion](#4-markov-chains-mm--hmm-trong-diffusion)
5. [Score Matching & Denoising Training](#5-score-matching--denoising-training)
6. [Sampling: Từ nhiễu về tác phẩm](#6-sampling-từ-nhiễu-về-tác-phẩm)
7. [Code PyTorch nền tảng](#7-code-pytorch-nền-tảng)
8. [Liên hệ với Flow-based Models](#8-liên-hệ-với-flow-based-models)
9. [Kết luận & tài liệu](#9-kết-luận--tài-liệu)

---

## 1. Câu chuyện: Phòng tranh trong màn sương

Người họa sĩ mở cửa phòng tranh sau một đêm mưa, phát hiện mọi bức vẽ bị phủ bởi những lớp hơi nước mỏng. Khi đóng rồi mở khung lại, lớp sương cứ tích tụ thêm. Anh ghi chép mỗi lần mở khung: bức vẽ gốc $x_0$ trở thành $x_1$, rồi $x_2,...$, càng lúc càng giống noise trắng. Muốn khôi phục tác phẩm, anh cần mô hình hóa chính xác quá trình này và tìm đường đi ngược trở lại – tư duy cốt lõi của diffusion models.

## 2. Trực giác: Forward và Reverse Diffusion

### 2.1 Forward process (phá hủy có kiểm soát)

Ta xây dựng chuỗi Markov $x_0 \to x_1 \to \dots \to x_T$, trong đó $x_0$ là dữ liệu thật, $x_T$ gần như Gaussian chuẩn. Mỗi bước:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I\big),
$$

với $\beta_t$ nhỏ (noise schedule). Nếu $\beta_t$ đủ nhỏ, $x_t$ dần bị nhiễu nhưng vẫn giữ dấu vết cấu trúc của $x_{t-1}$.

### 2.2 Reverse process (hồi phục)

Để sinh dữ liệu mới, ta cần phân phối

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\big).
$$

Việc học $\mu_\theta$ hay trực tiếp score $\nabla_{x_t} \log q(x_t)$ là trái tim của score-based generation.

**Trực giác:** forward diffusion giống việc mở khung tranh để hơi nước phủ dần lên bề mặt; reverse diffusion là quy trình hong khô từng lớp, kết hợp “kỹ thuật sửa màu” (score) để trả bức vẽ về hiện trạng ban đầu.

## 3. Toán học nền tảng

### 3.1 Công thức closed-form giữa $x_0$ và $x_t$

Chuỗi Markov trên cho ta biểu thức:

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t) I\big),
$$

trong đó $\alpha_t = 1 - \beta_t$ và $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

Ta cũng có thể viết

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I).
$$

### 3.2 Từ reverse conditional đến score

Theo Bayes,

$$
\log p_\theta(x_{t-1} \mid x_t) = \log q(x_t \mid x_{t-1}) + \log p_\theta(x_{t-1}) - \log p_\theta(x_t).
$$

Việc tối ưu trực tiếp biểu thức này khó, nên DDPM sử dụng ELBO, còn score-based models học trực tiếp gradient $\nabla_{x_t} \log q(x_t)$ thông qua denoising score matching.

## 4. Markov Chains, MM & HMM trong diffusion

### 4.1 Chuỗi Markov (Markov Chain)

Anh họa sĩ thử nghiệm với giấy nháp: đặt mảnh giấy dưới đèn sương mù, mỗi lần mở nắp chỉ cần biết trạng thái ngay trước đó để đoán lớp nhiễu tiếp theo. Đây chính là cấu trúc Markov của forward diffusion.

Forward diffusion là chuỗi Markov bậc 1: trạng thái $x_t$ chỉ phụ thuộc $x_{t-1}$. Điều này tương đương với mô hình Markov (MM) cổ điển mà bạn có thể đã gặp ở lớp xác suất. Viết lại forward transition:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\big).
$$

Mỗi bước chỉ “nhìn” trạng thái trước, nên cấu trúc Markov giúp ta dễ dàng tính phân phối $q(x_t \mid x_0)$.

### 4.2 Hidden Markov Model (HMM)

Trong thực tế, anh không thể nhìn lại bức tranh gốc trong quá trình xử lý – anh chỉ thấy phiên bản đã bị nhiễu. Bức tranh gốc đóng vai trò trạng thái ẩn trong một HMM; các lần quan sát là ảnh đã thêm noise.

Có thể coi $x_t$ là trạng thái ẩn và “quan sát” là chính $x_t$ cộng noise. Khi training, ta chỉ thấy $x_t$ (phiên bản nhiễu) và biết $t$. “Ẩn” ở đây là $x_0$ hoặc những latent variables nội suy. Nhờ tư duy HMM, một số paper khai thác inference như forward-backward để tính expectation nhanh hơn.

**Ví dụ nhỏ:** Giả sử ta có dữ liệu 1D với ba mức cường độ (thấp, trung bình, cao). Forward diffusion sẽ làm các mức này hòa trộn vào Gaussian. Xem diffusion như HMM cho phép suy luận xác suất trạng thái ban đầu (mức cường độ) khi quan sát $x_t$ hiện tại. Đây là nền tảng cho các kỹ thuật như posterior sampling hay classifier-guided diffusion.

### 4.3 Liên hệ với score matching

Trong score-based SDE, forward process là Markov liên tục. Reverse SDE cần score $\nabla_{x_t} \log p(x_t)$, tương tự như việc anh họa sĩ phải “đọc” từ phiên bản hiện tại để đoán ra bức vẽ trước đó – chính là bước backward của HMM.

## 5. Score Matching & Denoising Training

Sau khi hiểu cấu trúc Markov, anh cần một “giác quan” để đo được độ đậm nhạt của lớp sương ở từng thời điểm. Score matching đóng vai trò cảm biến đó: dựa vào gradient log-density để quyết định nên bù trừ bao nhiêu nhiễu.

### 5.1 Score function là gì?

Score của phân phối $p(x)$ là gradient log-density:

$$
s_p(x) = \nabla_x \log p(x).
$$

Với diffusion, ta muốn học score của $q(x_t)$ tại mọi $t$.

### 5.2 Denoising Score Matching (DSM)

Vincent (2011) chứng minh rằng việc huấn luyện một mạng $s_\theta(x_t, t)$ để dự đoán noise từ phiên bản nhiễu $x_t$ tương đương với việc khớp score:

$$
\mathbb{E}_{t, x_0, \epsilon}\Big[\big\| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t \mid x_0)\big\|^2\Big]
= \mathbb{E}_{t, x_0, \epsilon}\Big[\big\| s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}\big\|^2\Big].
$$

Như vậy, ta có thể huấn luyện bằng cách lấy mẫu $x_0$, chọn $t$, tạo $x_t$ theo công thức closed-form rồi yêu cầu mạng dự đoán lại noise $\epsilon$.

### 5.3 Mối liên hệ với loss DDPM

Ho, Jain & Abbeel (2020) chứng minh ELBO tối ưu hóa chủ yếu tương đương với loss dự đoán noise:

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, x_0, \epsilon}\Big[\big\| \epsilon_\theta(x_t, t) - \epsilon \big\|^2\Big].
$$

Điều này giải thích vì sao hầu hết implementation thực tế huấn luyện bằng loss dự đoán noise thay vì loss log-likelihood phức tạp.

## 6. Sampling: Từ nhiễu về tác phẩm

### 6.1 Ancestral sampling (DDPM)

Khi đã quen thuộc, anh bắt đầu từ một tấm canvas trắng (noise thuần) và dần hong khô – mỗi bước giả lập ancestral sampling: giảm noise bằng cách sử dụng score dự đoán của mô hình.

Xuất phát từ $x_T \sim \mathcal{N}(0, I)$, lặp

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\big) + \sigma_t z,
$$

với $z \sim \mathcal{N}(0, I)$ và $\sigma_t$ điều chỉnh mức noise còn lại.

### 6.2 ODE/SDE sampling

Ở các dự án lớn, anh cân nhắc dùng bước hong dài (mô phỏng ODE để lấy chất lượng cao) hoặc giữ chút ngẫu nhiên (SDE cho tốc độ), tương tự việc chọn kỹ thuật gỡ sương phù hợp với lịch triển lãm.

Score-based models như Song et al. (2021) thay vì chuỗi Markov rời rạc, dùng solver cho SDE/RVE:

$$
dx = f(x,t)dt + g(t)dW_t \quad \text{(forward)}, \qquad
dx = [f(x,t) - g^2(t) \nabla_x \log p_t(x)] dt + g(t)d\bar{W}_t \quad \text{(reverse)}.
$$

Giải bằng Euler-Maruyama hoặc Heun cho phép kiểm soát trade-off giữa chất lượng và tốc độ.

## 7. Code PyTorch nền tảng

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        return self.net(x) + t_embed

def diffusion_loss(model, x0, betas):
    bsz = x0.size(0)
    t = torch.randint(0, len(betas), (bsz,), device=x0.device)
    beta_t = betas[t].view(-1, 1)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)[t].view(-1, 1)

    eps = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    pred = model(xt, t.float().view(-1, 1) / len(betas))
    return F.mse_loss(pred, eps)
```

Đoạn code trên minh họa loss dự đoán noise trong không gian vector đơn giản (ví dụ latent 64 chiều). Các bài tiếp theo sẽ giới thiệu kiến trúc UNet 2D/3D, noise schedule tinh vi và sampler cao cấp.

## 8. Liên hệ với Flow-based Models

- Forward diffusion **thêm** nhiễu thay vì học chuỗi biến đổi nghịch như flow, nhưng reverse diffusion cũng cần đo jacobian implicit (score) để đảm bảo xác suất.
- Khi $\beta_t$ nhỏ và reverse process là ODE, ta có thể xem diffusion như flow liên tục với nhiễu điều khiển (link sang Rectified Flow, Schrödinger Bridge).
- Flow Matching, Rectified Flow và Diffusion có thể kết hợp: Flow Matching cung cấp đường đi ngắn hơn, diffusion đảm bảo mô hình hóa tốt các chi tiết.

## 9. Kết luận & tài liệu

Diffusion Models cho phép sinh dữ liệu chất lượng cao bằng cách “đi ngược thời gian” trong một biển nhiễu. Bạn vừa đi qua:

- Trực giác forward/reverse diffusion.
- Công thức closed-form và lý do loss dự đoán noise hoạt động.
- Vai trò của score matching.
- Cách sampling cơ bản và code khởi điểm.

Các bài tiếp theo sẽ đào sâu: hướng dẫn training chi tiết, diffusion guidance, và kết hợp diffusion với Transformer/Flow.

### Tài liệu nên đọc

1. Sohl-Dickstein et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*.
2. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*.
3. Song, Y., et al. (2021). *Score-Based Generative Modeling through SDEs*.
4. Kingma, D. P., et al. (2021). *Variational Diffusion Models*.

---

<script src="/assets/js/katex-init.js"></script>
