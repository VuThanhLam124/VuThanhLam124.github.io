---
title: "DDPM Explained: Toán học đằng sau Diffusion Models"
date: "2025-10-12"
category: "DDPM"
tags: ["DDPM", "Diffusion", "Mathematics", "Deep Learning"]
excerpt: "Phân tích chi tiết forward process, reverse process, training objective và sampling algorithms. So sánh DDPM vs DDIM vs DPM-Solver với practical benchmarks."
author: "ThanhLamDev"
readingTime: 12
featured: false
---

# DDPM Explained: Toán học đằng sau Diffusion Models

Diffusion Models trở thành một trong những kiến trúc generative quan trọng nhất vài năm gần đây. Bài viết này tóm tắt lại phần toán học đứng phía sau Denoising Diffusion Probabilistic Models (DDPM) và chỉ ra cách triển khai thực tế.

## 1. Forward process

Forward process là Markov chain thêm nhiễu Gaussian vào dữ liệu gốc. Ở mỗi bước ```t``` ta có:

```
q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
```

Khi triển khai, ta thường dùng schedule ```\{\beta_t\}``` tuyến tính hoặc cosine. Điều thú vị là ta có thể sample trực tiếp ```x_t``` từ ```x_0``` thông qua công thức closed-form, giúp training dễ dàng hơn.

## 2. Reverse process

Reverse process học phân phối ```p_\theta(x_{t-1} | x_t)``` bằng cách huấn luyện mạng dự đoán nhiễu hoặc x0. Loss thường gặp là MSE giữa nhiễu thật và nhiễu dự đoán.

## 3. Training objective

Loss tổng quát của DDPM có dạng:

```
\mathbb{E}_{t, x_0, \epsilon}[ || \epsilon - \epsilon_\theta(x_t, t) ||_2^2 ]
```

Ta có thể thêm weighting hoặc loss phụ trợ để cải thiện chất lượng sinh mẫu.

## 4. Sampling strategy

Sampling cơ bản đòi hỏi chạy đủ ```T``` bước reverse. Để tăng tốc có thể dùng:

- **DDIM**: deterministic sampling và cho phép skip bước
- **DPM-Solver**: giải trực tiếp ODE tương ứng bằng high-order solver
- **PLMS**: pseudo linear multistep method

## 5. Tài nguyên tham khảo

- Paper gốc DDPM (Ho et al.)
- Blog của Hugging Face về Diffusion Models
- Repo open-source: ```openai/guided-diffusion```

Trong các bài viết tiếp theo, chúng ta sẽ xây dựng training script hoàn chỉnh và benchmark các chiến lược sampling trên dataset thực tế.

<script src="/assets/js/katex-init.js"></script>
