---
title: "Diffusion Models & Score-based Generation (Phần II): Score Matching, SDE và kỹ thuật nâng cao"
date: "2025-04-10"
category: "diffusion-models"
tags: ["diffusion-models", "score-matching", "sde", "ddim", "guidance", "pytorch"]
excerpt: "Phần II tiếp tục series Diffusion Models & Score-based Generation: theo chân một thám tử vô danh giải mã score matching, SDE song song, các biến thể sampling (DDIM, PLMS), classifier-free guidance, đánh giá và checklist triển khai."
author: "ThanhLamDev"
readingTime: 32
featured: false
---

# Diffusion Models & Score-based Generation (Phần II)

**Từ nhiễu đến điều khiển có chủ đích**

Sau khi tái dựng được ảnh gốc trong Phần I, thám tử vô danh nhận ra vụ án chưa kết thúc: kẻ giả mạo không chỉ tạo nhiễu mà còn điều khiển phong cách, thay đổi ánh sáng, thậm chí biến chữ viết tay thành nội dung khác. Để đối phó, anh cần những công cụ mạnh hơn – từ score matching, stochastic differential equation (SDE) cho đến các kỹ thuật sampling và guidance tinh vi. Phần II ghi lại toàn bộ ghi chép nâng cao ấy.

---

## Mục lục

1. [Ôn lại cơ sở từ Phần I](#1-ôn-lại-cơ-sở-từ-phần-i)  
2. [Score matching và mối liên hệ với diffusion](#2-score-matching-và-mối-liên-hệ-với-diffusion)  
3. [Formulation bằng SDE liên tục](#3-formulation-bằng-sde-liên-tục)  
4. [Sampling nâng cao: DDIM, PLMS, Heun](#4-sampling-nâng-cao-ddim-plms-heun)  
5. [Guidance: từ classifier đến classifier-free](#5-guidance-từ-classifier-đến-classifier-free)  
6. [Kết hợp điều kiện văn bản, layout và control](#6-kết-hợp-điều-kiện-văn-bản-layout-và-control)  
7. [Đánh giá và chẩn đoán mô hình diffusion](#7-đánh-giá-và-chẩn-đoán-mô-hình-diffusion)  
8. [Checklist triển khai sản phẩm](#8-checklist-triển-khai-sản-phẩm)  
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Ôn lại cơ sở từ Phần I

- Forward diffusion thêm nhiễu Gaussian có lịch $\{\beta_t\}_{t=1}^T$.
- Reverse diffusion dự đoán nhiễu $\epsilon_\theta(x_t, t)$ để tính mean $\mu_\theta$.
- Loss đơn giản: $\mathcal{L} = \mathbb{E}\left[\left\|\epsilon - \epsilon_\theta(x_t, t)\right\|^2\right]$.
- Sampling chuẩn cần tối đa 1000 bước → tốc độ còn chậm.

Phần II làm rõ lý thuyết score matching và các mẹo giảm thời gian sampling, đồng thời mở đường cho việc điều khiển đầu ra.

---

## 2. Score matching và mối liên hệ với diffusion

Khi đã quen thuộc với việc dự đoán nhiễu, thám tử tiếp tục đào sâu vào cuốn sổ tay. Trang tiếp theo ghi chú về một đại lượng gọi là **score** – thứ giúp anh định hình “đường dốc” mà dữ liệu thật nằm trên.

**Score** của phân phối $p(x)$ là gradient của log-density:

$$
\nabla_x \log p(x).
$$

Song & Ermon (2019) đề xuất học score thay vì density trực tiếp bằng cách tối thiểu hóa:

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{p_{\text{data}}} \left[ \left\| s_\theta(x) - \nabla_x \log p_{\text{data}}(x) \right\|_2^2 \right],
$$

Do không tính được $\nabla_x \log p_{\text{data}}(x)$, họ thêm nhiễu Gaussian $x + \sigma z$ và dùng **denoising score matching**:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{\sigma}\, \mathbb{E}_{x, z}\left[ \left\| s_\theta(x + \sigma z, \sigma) + \frac{z}{\sigma} \right\|_2^2 \right],
$$

Điểm nối với DDPM: nếu đặt $\sigma_t^2 = 1 - \bar{\alpha}_t$ và $s_\theta(x_t, t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)$, loss DSM trùng với loss DDPM. Vì vậy, diffusion có thể xem là một dạng score matching rời rạc theo thời gian.

---

## 3. Formulation bằng SDE liên tục

Trong sổ tay, thám tử ghi chú rằng quá trình nhiễu có thể xem như một **phương trình vi phân ngẫu nhiên** (SDE):

$$
dx = f(x, t)\, dt + g(t)\, d w_t,
$$

trong đó:

- $f(x, t)$ là **drift** – hướng trung bình mà điểm ảnh dịch chuyển,
- $g(t)$ điều khiển độ mạnh của nhiễu,
- $w_t$ là chuyển động Brown (Wiener process).

Ví dụ quen thuộc là **Variance Preserving SDE (VP-SDE)**:

$$
dx = -\frac{1}{2} \beta(t) x\, dt + \sqrt{\beta(t)}\, d w_t,
$$

với $\beta(t)$ là lịch nhiễu liên tục (thường dạng tuyến tính hoặc cosine). Khi giải forward SDE từ $t=0$ đến $1$, mọi ảnh đều trở thành Gaussian chuẩn.

Reverse SDE cho biết cách đi ngược:

$$
dx = \Big[ -f(x, t) + g(t)^2 \nabla_x \log p_t(x) \Big] dt + g(t) \, d \bar{w}_t,
$$

trong đó $\bar{w}_t$ là Wiener process chạy ngược thời gian. Một khi biết score $\nabla_x \log p_t(x)$ (từ mạng đã học), ta có thể dùng các solver ODE/SDE (Heun, RK45, DPM-Solver++) để tái dựng ảnh nhanh hơn so với 1000 bước discrete truyền thống.

### Tại sao dùng SDE?

- Cho phép **continuous-time training** (Noise Conditional Score Network).
- Dễ kết hợp với adaptive step-size solver → giảm số bước sampling.
- Linh hoạt chọn dạng SDE: VP-SDE, VE-SDE (Variance Exploding), sub-VP.

---

## 4. Sampling nâng cao: DDIM, PLMS, Heun

### 4.1. DDIM (Denoising Diffusion Implicit Models)

- Giữ nguyên forward diffusion nhưng thiết kế reverse deterministic.
- Công thức:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \, \epsilon_\theta(x_t, t).
$$

- Cho phép dùng **subsequence** các bước (ví dụ 50 steps) mà không cần sampling Gaussian ngẫu nhiên.

### 4.2. PLMS (Pseudo Linear Multistep)

- Áp dụng công thức Adams-Bashforth (ODE solver) vào reverse process:

$$
x_{t-1} = x_t + \frac{\Delta t}{2} \left(3 f(x_t, t) - f(x_{t+1}, t+1)\right),
$$

trong đó $f$ đại diện drift của reverse SDE. Ứng dụng trong Stable Diffusion để giảm thời gian inference.

### 4.3. Heun / Predictor-Corrector

- Bước predictor (Euler), bước corrector dựa trên score. Lặp lại cho đến khi hội tụ.
- Phù hợp khi dùng score từ SDE continuous-time.
- Trong thực tế điều tra, thám tử thường dùng Heun cho các bước cuối để ảnh ít rung nhiễu hơn.

---

## 5. Guidance: từ classifier đến classifier-free

Sau khi phục dựng ảnh, thám tử muốn thử nghiệm: nếu thay đổi câu lệnh mô tả, liệu có tái tạo được các biến thể giả mạo khác? Đó là lúc các kỹ thuật **guidance** phát huy tác dụng.

### 5.1. Classifier guidance

Dhariwal & Nichol (2021) huấn luyện classifier $p_\phi(y \mid x_t)$ trên các sample noised. Trong sampling:

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t}\, w \nabla_{x_t} \log p_\phi(y \mid x_t),
$$

với $w$ là hệ số hướng dẫn. Cải thiện chất lượng nhưng cần train classifier riêng.

### 5.2. Classifier-free guidance (CFG)

- Train diffusion với và **không** có điều kiện (ví dụ text prompt).
- During sampling:

$$
\epsilon_{\text{cfg}} = (1 + w) \epsilon_\theta(x_t, t, y) - w \, \epsilon_\theta(x_t, t, \varnothing),
$$

trong đó $\varnothing$ đại diện prompt rỗng. CFG trở thành tiêu chuẩn trong Stable Diffusion.

### 5.3. Guidance tuyến tính khác

- **Style guidance:** trộn hai embedding text.
- **Layout guidance:** dùng ControlNet, IP-Adapter để bổ sung thông tin điều khiển.

---

## 6. Kết hợp điều kiện văn bản, layout và control

### 6.1. Text-to-image (Stable Diffusion)

- Latent diffusion: encode ảnh vào latent $z = E(x)$ (Autoencoder VAE).
- Diffusion diễn ra trên latent $z$ thay vì pixel.
- CLIP text encoder tạo embedding phục vụ cross-attention.

### 6.2. ControlNet

- Sao chép weight UNet, thêm input branch điều kiện (canny, pose, depth).
- Freeze backbone, train branch mới để kiểm soát cấu trúc.

### 6.3. IP-Adapter / T2I Adapter

- Thêm image embedding (ví dụ từ CLIP) vào cross-attention.
- Cho phép remix phong cách hoặc reference hình ảnh.

---

## 7. Đánh giá và chẩn đoán mô hình diffusion

| Metric | Ý nghĩa | Ghi chú |
|--------|---------|---------|
| FID | So sánh distribution với dataset thật | Compute trên 50k mẫu |
| IS | Đo diversity và quality (Inception Score) | Không tin cậy bằng FID |
| Precision/Recall | Phạm vi và độ chi tiết | Sử dụng k-NN trong feature space |
| CLIPScore | Độ phù hợp text-image | Dùng cho text-to-image |
| Diversity metrics | LPIPS giữa các sample | Kiểm tra mode collapse |

### Chẩn đoán phổ biến

1. **Sampling grainy:** giảm step size, thêm corrector.
2. **Oversmoothing:** tăng guidance, đổi lịch $\beta$.
3. **Prompt drift:** log intermediate $x_t$ để kiểm tra alignment.
4. **NaN:** kiểm tra gradient explosion, clip norm, sử dụng $\epsilon$-prediction.

---

## 8. Checklist triển khai sản phẩm

1. **Chọn kiến trúc UNet:** số channel, attention layer (mid/high resolution).
2. **Ở inference:** precompute $\alpha_t$, $\sigma_t$ cho scheduler.
3. **Quản lý VRAM:** dùng xformers attention, 16-bit, hoặc offload VAE encoder.
4. **Caching text embedding:** cho prompt lặp lại nhiều lần.
5. **Safety filter:** dùng CLIP hoặc model riêng để chặn nội dung nhạy cảm.
6. **Monitoring:** log thời gian per step, FID sample, memory.
7. **Hạ tầng:** pipeline microservice tách encode text, diffusion core, decode ảnh.
8. **Phiên bản:** gắn version cho weight UNet, VAE, text encoder, scheduler.

---

## 9. Tài liệu tham khảo

1. Song, Y., Sohl-Dickstein, J., Kingma, D., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR.  
2. Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS.  
3. Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS.  
4. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.  
5. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.  
6. Zhang, J. et al. (2023). *ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models.* arXiv.  
7. Zhang, Z. et al. (2023). *Classifier-Free Guidance is All You Need for Test-Time Adaptation.* arXiv.  

---

<script src="/assets/js/katex-init.js"></script>
