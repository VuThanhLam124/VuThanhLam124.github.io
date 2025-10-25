---
title: "Diffusion Models & Score-based Generation (Phần I): Từ nhiễu tới tái tạo"
date: "2025-04-10"
category: "diffusion-models"
tags: ["diffusion-models", "ddpm", "generative-moTiếp tục ví dụ $2 \times 2$ ở trên. Biết $x_1$ và $\epsilon$ thật, ta có thể tính mean "chuẩn":

$$
\tilde{\mu}(x_1, x_0, 1) = \frac{1}{\sqrt{\alpha_1}}\left(x_1 - \frac{1 - \alpha_1}{\sqrt{1 - \bar{\alpha}_1}} \epsilon\right).
$$

Với $\alpha_1 = 0.9$, $\bar{\alpha}_1 = 0.9$, kết quả 

$$
\tilde{\mu} \approx \begin{bmatrix}0.80 & 0.60 \\ 0.40 & 0.20\end{bmatrix}
$$ 

đúng bằng $x_0$. Khi huấn luyện, mạng $\epsilon_\theta$ học cách dự đoán $\epsilon$ sao cho mean tính ra gần $x_0$ nhất có thể."score-based", "pytorch"]
excerpt: "Phần I mở đầu series Diffusion Models & Score-based Generation. Bài viết theo chân một thám tử điều tra vụ ảnh giả mạo, giải thích forward diffusion, reverse denoising, huấn luyện DDPM, kèm công thức chi tiết và code PyTorch đầy đủ."
author: "ThanhLamDev"
readingTime: 28
featured: false
---

# Diffusion Models & Score-based Generation (Phần I)

**Hành trình đưa nhiễu quay ngược thời gian**

Sau loạt vụ án ảnh giả mạo lan truyền trên mạng, một thám tử vô danh được triệu tập. Anh không mang súng hay phù hiệu; vũ khí duy nhất là những cuộn sổ ghi chép chằng chịt công thức. Nhiệm vụ: từ những tấm ảnh bị bôi trắng bởi nhiễu Gaussian, phải truy ngược lại hình gốc – như tái dựng hiện trường trong tâm trí. Những ghi chép có tên “Denoising Diffusion Probabilistic Models” chính là manh mối, và bài viết này kể lại toàn bộ quá trình vị thám tử học cách biến nhiễu thành bằng chứng.

---

## Mục lục

1. [Câu chuyện điều tra nhiễu](#1-câu-chuyện-điều-tra-nhiễu)  
2. [Trực giác: Forward diffusion](#2-trực-giác-forward-diffusion)  
   - [Cấu trúc Markov của chuỗi nhiễu](#21-cấu-trúc-markov-của-chuỗi-nhiễu)  
   - [Ví dụ điều tra 2x2 pixel](#22-ví-dụ-điều-tra-2x2-pixel)  
3. [Reverse diffusion và cấu trúc trung bình](#3-reverse-diffusion-và-cấu-trúc-trung-bình)  
4. [DDPM objective: dẫn xuất từng bước](#4-ddpm-objective-dẫn-xuất-từng-bước)  
5. [Parameterization và lịch nhiễu](#5-parameterization-và-lịch-nhiễu)  
6. [Sampling algorithm chi tiết](#6-sampling-algorithm-chi-tiết)  
7. [Implementation PyTorch: bộ khung đầy đủ](#7-implementation-pytorch-bộ-khung-đầy-đủ)  
8. [Quan sát thực nghiệm và mẹo tối ưu](#8-quan-sát-thực-nghiệm-và-mẹo-tối-ưu)  
9. [Kết luận và hướng tới Phần II](#9-kết-luận-và-hướng-tới-phần-ii)  
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Câu chuyện điều tra nhiễu

Anh thám tử vô danh nhận được một ổ đĩa USB – bằng chứng duy nhất của vụ án ảnh giả mạo đang gây hoang mang dư luận. Khi mở ra, tất cả hình ảnh đều đã bị phủ kín bởi nhiễu Gaussian, chỉ còn lấp ló vài đường nét. Để truy tìm thủ phạm, anh phải tái dựng hiện trường từ đống nhiễu trắng.

Trong cuốn sổ tay cũ kỹ, thám tử ghi lại kế hoạch điều tra:

1. **Tiếp cận hiện trường (forward diffusion):** mô hình hóa cách mà nghi phạm có thể đã huỷ hoại bức ảnh, mỗi bước đổ thêm một lớp nhiễu có kiểm soát cho tới khi hình gốc biến mất hoàn toàn.
2. **Lần ngược dấu vết (reverse diffusion):** thiết kế một cơ chế đảo ngược từng bước nhiễu, đưa ảnh về trạng thái ban đầu. Nếu làm đúng, thám tử sẽ khôi phục được bằng chứng chân thực.

Để thực hiện chuyên án, mọi bước phải được viết thành phương trình rõ ràng – vì chỉ có GPU và toán học mới đủ kiên nhẫn để bóc tách hàng nghìn lớp nhiễu.

---

## 2. Trực giác: Forward diffusion

Forward diffusion (hay **noising process**) tạo chuỗi biến ngẫu nhiên $\{x_t\}_{t=0}^T$ bắt đầu từ ảnh gốc $x_0$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I\right),
$$

trong đó $\beta_t \in (0, 1)$ là lượng nhiễu thêm ở bước $t$. Sau nhiều bước (tới $T$), $x_T$ tiệm cận phân phối Gaussian chuẩn – chính là đống ảnh nhiễu mà thám tử nhận được.

### Công thức rút gọn theo $x_0$

DDPM có ưu điểm quan trọng: phân phối $x_t$ theo $x_0$ có dạng khép kín. Bằng cách thay thế đệ quy liên tiếp, ta thu được:

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I),
$$

hay tương đương:

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I \right),
$$

trong đó $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, và $\epsilon$ là nhiễu Gaussian độc lập. Công thức này cho phép thám tử tạo ra bất kỳ bước nhiễu nào trực tiếp từ ảnh gốc mà không cần mô phỏng toàn bộ chuỗi – rất hữu ích khi xây dựng loss hay sinh dữ liệu huấn luyện.

### 2.1. Cấu trúc Markov của chuỗi nhiễu

Forward diffusion là một **chuỗi Markov hữu hạn**: mỗi bước chỉ phụ thuộc vào bước trước đó. Thám tử ghi chú trong sổ:

$$
q(x_{0:T}) = q(x_0) \prod_{t=1}^T q(x_t \mid x_{t-1}).
$$

- $x_0$ là trạng thái sạch.
- $\{x_t\}$ là chuỗi ảnh bị nhiễu dần.
- Khi chỉ quan sát $x_t$, ta đang đối mặt với một **Hidden Markov Model (HMM)**: trạng thái ẩn $x_{t-1}$ sinh ra quan sát $x_t$ theo Gaussian. Việc huấn luyện DDPM chính là học một mô hình đi ngược chuỗi Markov này.

Reverse process mong muốn là:

$$
q(x_{t-1} \mid x_t) = \int q(x_{t-1} \mid x_t, x_0)\, q(x_0 \mid x_t) \, dx_0,
$$

nhưng vì $q(x_0 \mid x_t)$ không tính được, ta thay bằng $p_\theta(x_{t-1} \mid x_t)$ và tối ưu sao cho hai phân phối gần nhau (phần 4).

**Liên hệ câu chuyện:**  
Hãy coi mỗi bức ảnh bị nhiễu là một “lời khai” méo mó về hiện trường. Chuỗi Markov chính là chuỗi lời khai mà thủ phạm để lại: mỗi câu nói mới chỉ dựa trên câu ngay trước đó. HMM diễn giải rằng đằng sau mỗi lời khai (quan sát $x_t$) luôn có một trạng thái thật ($x_{t-1}$) mà ta không thấy, và nhiệm vụ của thám tử là lần ngược lại các trạng thái ấy.

**Ví dụ Markov đơn giản:**  
Giả sử nghi phạm thêm nhiễu vào ảnh theo ba mức: “nhẹ”, “vừa”, “nặng”. Chuỗi trạng thái $S_t \in \{\text{nhẹ}, \text{vừa}, \text{nặng}\}$ tiến hoá theo ma trận chuyển

$$
P = \begin{bmatrix}
0.8 & 0.2 & 0 \\
0 & 0.7 & 0.3 \\
0 & 0 & 1
\end{bmatrix},
$$

tức là khi đã sang mức “nặng” (tương đương $x_T$) thì không quay lại. Quan sát $X_t$ chính là ảnh nhiễu mà camera ghi được. Đây chính là một HMM cổ điển: trạng thái ẩn (mức nhiễu) sinh ra quan sát (ảnh). Khi huấn luyện DDPM, ta không cố gắng đoán trạng thái bằng thuật toán Viterbi; thay vào đó, ta học trực tiếp hàm $p_\theta(x_{t-1} \mid x_t)$ để đi ngược chuỗi.

**Ví dụ Markov ẩn với câu chuyện:**  
Áp dụng vào vụ án, thám tử giả lập ba sự kiện:

1. **Sáng sớm ($t=0$):** camera chụp bức ảnh thật.  
2. **Trưa ($t=1$):** nghi phạm phủ một lớp nhiễu “nhẹ” để che mặt.  
3. **Chiều tối ($t=2$):** nghi phạm phủ thêm nhiễu “nặng” khiến toàn ảnh trắng xóa.

Thám tử chỉ nhìn thấy ảnh chiều tối ($x_2$). HMM nói rằng ảnh trưa ($x_1$) là trạng thái ẩn sinh ra ảnh tối, và ảnh sáng sớm ($x_0$) lại sinh ra ảnh trưa. Để truy ra $x_0$, anh phải học được quy tắc sinh nhiễu ở từng bước – chính là nhiệm vụ của reverse diffusion.

### 2.2. Ví dụ điều tra 2x2 pixel

Để trực quan, thám tử thử nghiệm trên ảnh xám $2 \times 2$:

$$
x_0 = \begin{bmatrix}0.8 & 0.6 \\ 0.4 & 0.2\end{bmatrix}, \qquad \beta_1 = 0.1.
$$

Forward bước 1:

$$
x_1 = \sqrt{1 - \beta_1}\, x_0 + \sqrt{\beta_1}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

Giả sử $\epsilon = \begin{bmatrix}0.5 & -0.3 \\ 0.1 & -0.2\end{bmatrix}$, ta có

$$
x_1 \approx \begin{bmatrix}0.76 & 0.51 \\ 0.39 & 0.17\end{bmatrix}.
$$

Lặp lại vài lần, ma trận tiến dần về nhiễu trắng. Nhiệm vụ của reverse diffusion là dự đoán đúng $\epsilon$ ở mỗi bước để quay lại $x_0$ – giống như tái hiện nét bút gốc trong ảnh.

### Lập lịch $\beta_t$

- **Linear schedule:** $\beta_t$ tăng đều từ $10^{-4}$ đến $2 \times 10^{-2}$ (DDPM gốc).
- **Cosine schedule:** cải thiện chất lượng vì duy trì nhiễu nhỏ ở đầu và lớn ở cuối (Nichol & Dhariwal, 2021).
- Điều kiện: $\bar{\alpha}_t$ không được giảm quá nhanh để tránh mất thông tin sớm.

> **Ký hiệu nhanh:** trong các công thức tiếp theo, $x_0$ là ảnh sạch, $x_t$ là ảnh đã nhiễu, $I$ là ma trận đơn vị, còn $\beta_t$, $\alpha_t$, $\bar{\alpha}_t$ là các tham số điều khiển cường độ nhiễu.

---

## 3. Reverse diffusion và cấu trúc trung bình

Để lần ngược lại quá trình phá hoại, thám tử cần mô hình hóa phân phối:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\right).
$$

Mục tiêu là tìm tham số $\theta$ sao cho $p_\theta$ gần đúng phân phối thật $q(x_{t-1} \mid x_t)$.

### Công thức mean đúng

Từ Bayes rule, ta có:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}(x_t, x_0, t), \tilde{\beta}_t I\right),
$$

trong đó

$$
\tilde{\mu}(x_t, x_0, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \, \epsilon_t\right), \quad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t,
$$

và nhiễu trung gian

$$
\epsilon_t = \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}}
$$

chính là lượng nhiễu mà forward process đã thêm vào ở bước $t$.

Như vậy, chỉ cần mô hình hóa chính xác nhiễu $\epsilon_t$ ta sẽ hồi phục được mean của reverse process.

### Ví dụ: phục hồi bước đầu

Tiếp tục ví dụ $2 \times 2$ ở trên. Biết $x_1$ và $\epsilon$ thật, ta có thể tính mean “chuẩn”:

$$
\tilde{\mu}(x_1, x_0, 1) = \frac{1}{\sqrt{\alpha_1}}\left(x_1 - \frac{1 - \alpha_1}{\sqrt{1 - \bar{\alpha}_1}} \epsilon\right).
$$

Với $\alpha_1 = 0.9$, $\bar{\alpha}_1 = 0.9$, kết quả $\tilde{\mu} \approx \begin{bmatrix}0.80 & 0.60 \\ 0.40 & 0.20\end{bmatrix}$ – đúng bằng $x_0$. Khi huấn luyện, mạng $\epsilon_\theta$ học cách dự đoán $\epsilon$ sao cho mean tính ra gần $x_0$ nhất có thể.

### Thuật toán

1. Chọn $t$ từ 1 đến $T$.
2. Dự đoán nhiễu $\epsilon_\theta(x_t, t)$ bằng mạng neural.
3. Tính mean $\mu_\theta$ từ công thức.
4. Lấy mẫu $x_{t-1} \sim \mathcal{N}(\mu_\theta, \tilde{\beta}_t I)$.

---

## 4. DDPM Objective: dẫn xuất từng bước

Để mô hình hóa thao tác “làm sạch” của thám tử một cách tối ưu, ta tối thiểu hóa **Evidence Lower Bound (ELBO)** trên phân phối dữ liệu:

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q}\left[-\log p_\theta(x_0 \mid x_1)\right] + \sum_{t=2}^T \mathbb{E}_{q}\left[ D_{\text{KL}}\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big) \right] + D_{\text{KL}}\big(q(x_T \mid x_0) \,\|\, p(x_T)\big).
$$

- **Term đầu:** đảm bảo bước cuối cùng tái tạo ảnh sạch hợp lý.
- **Term giữa:** đưa reverse model $p_\theta$ tiến gần phân phối thật $q$ ở mọi bước – giống như việc kiểm tra từng lời khai với sự thật.
- **Term cuối:** buộc phân phối ở thời điểm $T$ trùng Gaussian chuẩn (ta chọn $p(x_T) = \mathcal{N}(0, I)$).

Với lịch nhiễu chuẩn, term đầu và cuối có thể tính chính xác và xem như hằng số. Ta rút gọn được **loss dự đoán nhiễu**:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|_2^2\right],
$$

trong đó $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$ và $\epsilon \sim \mathcal{N}(0, I)$. Nghĩa là ta chỉ cần học chính xác nhiễu đã thêm vào ở bước $t$.

**Giải thích:**

- Loss đơn giản chỉ yêu cầu dự đoán nhiễu $\epsilon$ chính xác.
- Đây là lý do diffusion model đôi khi được gọi là **noise predictor**.
- Khi inference, ta sử dụng $\epsilon_\theta$ để tính mean reverse và lấy mẫu.

### Phân loại biến thể của loss

| Biến thể | Công thức | Lưu ý |
|----------|-----------|-------|
| $\epsilon$-prediction | $\hat{\epsilon}_\theta$ | Dễ train, nhưng liên quan $\sigma$ cố định |
| $x_0$-prediction | $\hat{x}_{0,\theta}$ | Phù hợp khi cần guidance trực tiếp trên ảnh |
| $v$-prediction | $\hat{v}_\theta = \alpha_t \epsilon + \sigma_t x_0$ | Dùng trong Stable Diffusion v2, cân bằng giữa hai dạng trên |

---

## 5. Parameterization và lịch nhiễu

### 5.1. Chọn thời điểm huấn luyện

Khi tối ưu $\mathcal{L}_{\text{simple}}$, chỉ số thời gian $t$ thường được chọn ngẫu nhiên. Một trick phổ biến là lấy $t$ từ phân phối trọng số $w_t$ thay vì uniform:

- **Uniform:** đơn giản, nhưng các bước đầu dễ bị overfit vì nhiễu thấp.
- **Trọng số theo $\sqrt{\bar{\alpha}_t}$:** ưu tiên bước khó (nhiễu cao) nhiều hơn.
- **Trọng số cosine:** phù hợp với beta cosine schedule.

Thám tử thực nghiệm thấy rằng sampling $t$ với xác suất tỷ lệ $\beta_t$ giúp mô hình chú ý hơn đến những bước nhiễu mạnh – vốn là nơi ảnh dễ bị mất chi tiết.

### 5.2. Các dạng parameterization

- **$\epsilon$-prediction:** mặc định của DDPM, ổn định và tương thích với nhiều thư viện.
- **$x_0$-prediction:** thay vì dự đoán nhiễu, model dự đoán trực tiếp ảnh sạch. Mean reverse khi đó tính lại bằng cách thế $\hat{x}_{0,\theta}$ vào công thức.
- **$v$-prediction:** định nghĩa $v = \alpha_t \epsilon - \sqrt{1-\alpha_t^2} x_0$, giúp loss phân phối đều giữa các bước (đặc biệt khi kết hợp latent diffusion).

Trong code, ta có thể chuyển đổi giữa các parameterization bằng các hàm tiện ích, đảm bảo model inference nhất quán.

### 5.3. Lịch nhiễu và các biến thể

- **Cosine schedule:** $\bar{\alpha}_t = \frac{\cos^2\left(\frac{t/T + s}{1 + s} \frac{\pi}{2}\right)}{\cos^2\left(\frac{s}{1 + s} \frac{\pi}{2}\right)}$ với $s=0.008$; giúp tránh việc $\bar{\alpha}_t$ tụt nhanh.
- **Learned schedule:** một số công trình (VDM, EDM) tối ưu lịch nhiễu bằng gradient. Thực nghiệm cho thấy có thể giảm 0.3–0.5 FID.
- **Discrete→Continuous:** với continuous SDE, ta định nghĩa $\beta(t)$ liên tục; schedule discrete chỉ là lấy mẫu của hàm này.

---

## 6. Sampling algorithm chi tiết

Giả sử đã train được $\epsilon_\theta$, quá trình sinh ảnh:

```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    eps_pred = epsilon_theta(x_t, t)
    mu = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred)
    if t > 1:
        z = N(0, I)
    else:
        z = 0
    x_{t-1} = mu + sqrt(beta_t_tilde) * z
return x_0
```

Trong thực tế, $T$ thường nằm trong khoảng 1000–4000. Các cải tiến (DDIM, PLMS) sẽ được bàn trong Phần II.
Thám tử thường bắt đầu từ $x_T$ là ảnh trắng, chạy vòng lặp trên để dựng lại bằng chứng; mỗi iteration tương đương một lần anh loại bỏ một lớp bụi giả mạo.

### 6.1. Chi phí và lựa chọn bước

- **Độ phức tạp:** $O(T)$ lần forward UNet. Với UNet 860M tham số, 1000 bước tương đương vài giây trên GPU A100.
- **Thời gian thực tế:** sampling 50 bước (như Stable Diffusion) mất ~8 giây trên GPU tầm trung; 20 bước với sampler tốt có thể xuống 3–4 giây.
- **Heuristic giảm bước:** chọn tập con $\mathcal{T} = \{t_1, \dots, t_K\}$ (ví dụ 50 bước) rồi nội suy $\alpha_t$ tương ứng. Đặt chặt hơn ở đầu (nhiễu cao) để giữ chi tiết.

### 6.2. Lưu ý số học

- Khi tính $\sqrt{1 - \bar{\alpha}_t}$, sử dụng `torch.sqrt(torch.clamp(..., min=1e-12))` để tránh mất ổn định.
- Đối với $\beta_t$ nhỏ, log-variance có thể âm => clamp ở mức `log(1e-20)`.
- Nếu dùng FP16, nên chuyển các hệ số lịch nhiễu sang FP32 để giảm sai số phép trừ.

---

## 7. Implementation PyTorch: bộ khung đầy đủ

Trong phần này, ta xây dựng "phòng thí nghiệm số" của thám tử: một UNet đơn giản mô phỏng cách anh dự đoán nhiễu và tái dựng ảnh.

Chúng ta triển khai một UNet đơn giản cho $32 \times 32$ (ví dụ CIFAR-10). Bạn có thể coi đây là phòng thí nghiệm mô phỏng nơi thám tử thử nghiệm các thuật toán trước khi áp dụng cho ảnh thật. Code tập trung vào pipeline training & sampling.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.activation(self.norm1(x)))
        time_emb = self.time_emb(t)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    def __init__(self, channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)

        self.down1 = ResidualBlock(channels, channels * 2, time_emb_dim)
        self.down2 = ResidualBlock(channels * 2, channels * 4, time_emb_dim)

        self.mid = ResidualBlock(channels * 4, channels * 4, time_emb_dim)

        self.up2 = ResidualBlock(channels * 8, channels * 2, time_emb_dim)
        self.up1 = ResidualBlock(channels * 4, channels, time_emb_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)

        x1 = self.conv_in(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(F.avg_pool2d(x2, 2), t)

        m = self.mid(x3, t)

        u3 = torch.cat([m, x3], dim=1)
        u2 = self.up2(u3, t)
        u2 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u2 = torch.cat([u2, x2], dim=1)
        u1 = self.up1(u2, t)
        out = self.conv_out(u1)
        return out


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.device = device
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def noise_images(self, x0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

    def step(self, model, x_t, t):
        beta_t = self.beta[t][:, None, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        alpha_t = self.alpha[t][:, None, None, None]
        sqrt_inv_alpha = torch.rsqrt(alpha_t)

        eps_theta = model(x_t, t)
        mean = sqrt_inv_alpha * (x_t - beta_t / sqrt_one_minus_alpha * eps_theta)

        if (t == 0).all():
            noise = torch.zeros_like(x_t)
        else:
            beta_tilde = beta_t * (1 - self.alpha_bar[t - 1][:, None, None, None]) / (1 - self.alpha_bar[t])
            noise = torch.sqrt(beta_tilde) * torch.randn_like(x_t)
        return mean + noise


def train(model, diffusion, dataloader, optimizer, epochs=50, device="cuda"):
    model.train()
    mse = nn.MSELoss()

    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.to(device)
            t = diffusion.sample_timesteps(images.size(0))
            x_t, noise = diffusion.noise_images(images, t)

            pred_noise = model(x_t, t)
            loss = mse(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")


def sample_images(model, diffusion, num_samples=16, device="cuda"):
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(num_samples, 3, 32, 32, device=device)
        for t in reversed(range(diffusion.timesteps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x_t = diffusion.step(model, x_t, t_batch)
        return x_t.clamp(-1, 1)
```

Code trên cung cấp khung huấn luyện cơ bản: UNet đơn giản, scheduler tuyến tính, loss MSE. Thực tế, cần thêm data augmentation, EMA weights, gradient clipping để ổn định.

### 7.1. Chuẩn bị dữ liệu và tiền xử lý

- **Chuẩn hóa:** đưa ảnh về $[-1, 1]$ bằng `transforms.Normalize((0.5,)*3, (0.5,)*3)`.
- **Augmentation:** random crop, horizontal flip giúp mô hình robust hơn, đặc biệt khi dataset nhỏ.
- **Tải dữ liệu:** sử dụng `DataLoader` với `pin_memory=True`, `num_workers>=8` để tránh nghẽn CPU.

### 7.2. Huấn luyện thực tế

- **Optimizer:** AdamW ($\text{lr}=1\mathrm{e}{-4}$, $\beta=(0.9,0.999)$) kết hợp gradient clipping 1.0.
- **EMA:** duy trì bản sao $\theta_{\text{EMA}} = 0.999 \theta_{\text{EMA}} + (1-0.999)\theta$ mỗi step; sampling dùng EMA cho ảnh sạch hơn.
- **Mixed precision:** dùng `torch.cuda.amp` giảm VRAM ~40% mà không giảm chất lượng.
- **Logging:** log loss, learning rate, và hình mẫu `sample_images` sau mỗi vài epoch để kiểm soát chất lượng.

### 7.3. Kiểm thử

- Sinh 8–16 ảnh trên validation set và so sánh với ảnh gốc bằng LPIPS.
- Tính FID bằng bộ 10k ảnh để đánh giá sau mỗi 10 epoch.
- Nếu loss dừng giảm nhưng ảnh vẫn nhiễu hạt ⇒ tăng số bước sampling hoặc đổi lịch $\beta$.

---

## 8. Quan sát thực nghiệm và mẹo tối ưu

Trong nhật ký điều tra, thám tử ghi lại các lưu ý sau để mọi lần phục dựng đều ổn định:

1. **Batch size nhỏ vẫn hoạt động:** Diffusion không nhạy batch size như GAN; batch 32–64 là ổn.
2. **EMA model:** duy trì bản sao EMA của $\theta$, dùng để sampling (giảm nhiễu).
3. **FP16 training:** sử dụng `torch.cuda.amp.autocast` và `GradScaler` để giảm VRAM.
4. **Cosine scheduler:** thay linear schedule giúp giảm artifacts (theo Nichol & Dhariwal).
5. **Loss weighting:** giai đoạn cuối (t lớn) khó dự đoán → có thể thêm weight $w_t$ để cân bằng.
6. **Validation bằng FID:** kiểm tra FID trên set nhỏ sau mỗi vài epoch để chọn checkpoint.
7. **Tối ưu thời gian inference:** thử nghiệm với 50 bước DDIM, guidance 7.5 (CFG) cho ảnh 512², sau đó điều chỉnh tùy nhu cầu chất lượng.
8. **Kiểm soát màu sắc:** thêm regularizer CLIP hoặc histogram matching khi mẫu có xu hướng lệch màu.
9. **Phân tích lỗi:** log trung gian $x_t$ (ví dụ t=100, 200, ...) để xem mạng làm sạch ra sao; nếu ảnh bị mất chi tiết quá sớm, cần điều chỉnh lịch nhiễu.

---

## 9. Kết luận và hướng tới Phần II

Câu chuyện của thám tử cho thấy: muốn khôi phục sự thật từ nhiễu, chúng ta cần hiểu rõ forward/reverse diffusion, loss và cách triển khai. Phần I đã:

- Xây dựng trực giác forward/reverse diffusion.
- Dẫn xuất loss DDPM và ý nghĩa việc dự đoán nhiễu.
- Cung cấp thuật toán sampling và code PyTorch cơ bản.

Phần II sẽ tiếp tục cuộc điều tra với **score-based SDE**, **DDIM/PLMS**, **classifier-free guidance**, và cách thám tử phối hợp các nguồn manh mối (văn bản, layout, reference image) để điều khiển kết quả. Mời bạn tiếp tục đọc [Phần II](/posts/2025/diffusion-models-part2).

---

## 10. Tài liệu tham khảo

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.  
2. Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML.  
3. Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS.  
4. Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS.  
5. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.  

---

<script src="/assets/js/katex-init.js"></script>
