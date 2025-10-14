---
title: "Rectified Flows: Khi xưởng pha lê muốn đường thẳng hoàn hảo"
date: "2025-02-19"
category: "flow-based-models"
tags: ["rectified-flows", "reflow", "optimal-transport", "generative-models", "pytorch"]
excerpt: "Chương tiếp theo của series flow-based: lý do Rectified Flows ra đời, toán học phía sau những đường thẳng, thuật toán Reflow, code minh họa và mẹo thực nghiệm."
author: "ThanhLamDev"
readingTime: 17
featured: false
---

# Rectified Flows

**Sau khi làm chủ RealNVP và Glow, xưởng pha lê phải giải quyết một vấn đề mới: khách muốn xem mẫu gần như ngay lập tức. Chỉ có Rectified Flows mới giúp người thợ dẫn ánh sáng đi theo đường thẳng ngắn nhất.**

## Mục lục

1. [Câu chuyện: Đường vận chuyển ánh sáng](#1-câu-chuyện-đường-vận-chuyển-ánh-sáng)
2. [Cú hích từ quầy thử ánh sáng](#2-cú-hích-từ-quầy-thử-ánh-sáng)
3. [Từ đường cong đến đường thẳng: trực giác Rectified Flow](#3-từ-đường-cong-đến-đường-thẳng-trực-giác-rectified-flow)
4. [Toán học nền tảng](#4-toán-học-nền-tảng)
5. [Thuật toán Reflow: làm thẳng từng bước](#5-thuật-toán-reflow-làm-thẳng-từng-bước)
6. [Code minh họa với PyTorch](#6-code-minh-họa-với-pytorch)
7. [Sampling nhanh: 1 bước hay vài bước?](#7-sampling-nhanh-1-bước-hay-vài-bước)
8. [Thực nghiệm & các bẫy thường gặp](#8-thực-nghiệm--các-bẫy-thường-gặp)
9. [Kết nối series & tài liệu](#9-kết-nối-series--tài-liệu)

---

## 1. Câu chuyện: Đường vận chuyển ánh sáng

Sau buổi workshop về Glow, người thợ pha lê nhận về một yêu cầu khó: **khách hàng muốn thử nhiều biến thể ánh sáng khi đứng trước quầy**, không phải chờ 30–60 giây để mô hình sinh hình ảnh. Các đường “dòng chảy” mà anh đã dùng vẫn quá cong; phải giải tích ODE nhiều bước mới ra sản phẩm.

Anh quan sát những tia sáng phản xạ qua pha lê và nhận ra: **tia nào đi gần đường thẳng nhất thì sáng rõ nhất**. Nếu các flow của anh cũng thẳng như vậy, anh có thể từ latent Gaussian đến tác phẩm chỉ trong một bước. Từ trực giác đó, anh phát minh ra **Rectified Flows** – nghệ thuật “chỉnh thẳng” dòng chảy.

## 2. Cú hích từ quầy thử ánh sáng

Phòng trưng bày của xưởng ngày càng đông. Khách không chỉ muốn xem thành phẩm; họ muốn đứng ngay quầy thử, chọn một tông màu trên tablet và thấy tia sáng chạy qua khối pha lê tức thì. Những đường flow cong queo khiến người thợ phải vặn núm điều khiển nhiều lần, khiến khách mất kiên nhẫn.

Anh ghi chép vào sổ tay: “Muốn phục vụ tại quầy, mình cần **đường dẫn ánh sáng thật thẳng**. Càng ít vòng vo, càng ít thời gian lấy mẫu.” Đây chính là động lực để anh nghiên cứu Rectified Flows: vẫn là câu chuyện từ Gaussian tới tác phẩm, nhưng mục tiêu mới là khiến hành trình giữa hai điểm trở thành một đoạn thẳng gọn gàng.

## 3. Từ đường cong đến đường thẳng: trực giác Rectified Flow

Ở các bài trước, flow được mô tả bởi ODE:

$$
\frac{d x_t}{dt} = v_t(x_t), \quad x_0 \sim \pi_0 \ (\text{Gaussian}), \quad x_1 \sim \pi_1 \ (\text{data})
$$

> **Chú thích ký hiệu:** $x_t$ là trạng thái tại thời điểm $t$; $\frac{d x_t}{dt}$ (hay $\dot{x}_t$) là đạo hàm theo thời gian; $v_t$ là trường vận tốc; ký hiệu $\sim$ nghĩa là “được lấy mẫu từ”.

Nếu vector field $v_t$ uốn cong, ta phải dùng nhiều bước tích phân ⇒ chậm.

**Rectified Flow** đặt mục tiêu:

- Duy trì đúng phân phối đầu-cuối.
- Tạo ra đường đi $x_t$ càng thẳng càng tốt.
- Giảm số lần đánh giá trường vector khi sampling.

Độ “thẳng” được đo bằng chi phí vận chuyển (transport cost):

$$
\mathcal{C} = \mathbb{E}\left[\int_0^1 \|v_t(X_t)\|^2 dt\right]
$$

> **Chú thích:** $\mathbb{E}[\cdot]$ là kỳ vọng (trung bình); $\|\cdot\|$ là chuẩn Euclid; tích phân từ $0$ đến $1$ đo tổng năng lượng vận tốc trong toàn bộ hành trình.

Đường thẳng tối ưu (geodesic) tương ứng với chi phí thấp nhất. Rectified Flows tìm cách xấp xỉ geodesic đó – giống như người thợ tìm cách mài đường truyền ánh sáng ngắn nhất.

## 4. Toán học nền tảng

### 4.1 Linear interpolant và vận tốc

Cho cặp mẫu $(X_0, X_1)$ đã được **coupling tối ưu**. Khi ta nội suy tuyến tính:

$$
X_t = (1 - t) X_0 + t X_1
$$

> **Chú thích:** Đây là phép nội suy tuyến tính giữa hai điểm $X_0$ và $X_1$; hệ số $(1-t)$, $t$ đảm bảo khi $t=0$ ta ở $X_0$, khi $t=1$ ta ở $X_1$.

Ta có vận tốc không đổi:

$$
\dot{X}_t = X_1 - X_0
$$

> **Chú thích:** Dấu chấm biểu diễn đạo hàm theo thời gian. Vì nội suy tuyến tính nên vận tốc luôn bằng hiệu $X_1 - X_0$.

Điều kiện cần: tồn tại coupling đủ tốt giữa hai phân phối để đoạn thẳng này “hợp lý”.

### 4.2 Velocity field mục tiêu

Từ quan sát trên:

$$
v_t(x) = \mathbb{E}\big[X_1 - X_0 \mid X_t = x\big]
$$

> **Chú thích:** Đây là kỳ vọng có điều kiện – trung bình của hiệu $X_1 - X_0$ khi biết trạng thái hiện tại bằng $x$.

Trong thực tế ta dùng mạng neural $v_\theta(x, t)$ để xấp xỉ và tối thiểu hóa:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, X_t}\left[\|v_\theta(X_t, t) - (X_1 - X_0)\|^2\right]
$$

> **Chú thích:** $\mathcal{L}$ là hàm loss; chỉ số $\theta$ biểu diễn tham số mạng; bình phương chuẩn $\|\cdot\|^2$ là mean squared error giữa vận tốc dự đoán và vận tốc mục tiêu.
Với $t \sim \mathcal{U}[0, 1]$, $X_t = (1 - t) X_0 + t X_1$.

### 4.3 Khi coupling chưa tối ưu

Nếu $(X_0, X_1)$ lấy từ dữ liệu gốc, đường có thể cong. **Reflow** xuất hiện để cải thiện coupling bằng cách sinh thêm dữ liệu trung gian từ chính mô hình.

## 5. Thuật toán Reflow: làm thẳng từng bước

### 5.1 Ý tưởng

1. Huấn luyện model ban đầu với cặp $(X_0, X_1)$ (Gaussian → data).
2. Dùng model này sinh ra mẫu mới rồi **nội suy tuyến tính** để tạo cặp “đã được làm thẳng hơn”.
3. Huấn luyện model mới trên dataset mới.
4. Lặp lại đến khi chi phí vận chuyển giảm đủ thấp.

### 5.2 Pseudocode

```python
# Reflow iteration k -> k+1
for k in range(num_reflows):
    # 1. Sinh dữ liệu bằng model hiện tại
    z0 = sample_gaussian(batch_size)
    x1 = integrate_ode(z0, v_theta[k])  # forward flow

    # 2. Xây dựng coupling tuyến tính
    t = torch.rand_like_scalar()
    xt = (1 - t) * z0 + t * x1
    target = x1 - z0

    # 3. Huấn luyện model mới
    v_theta[k+1] = optimize_MSE(xt, t, target)
```

### 5.3 Bằng chứng hội tụ

Liu et al. (2022) chứng minh: sau mỗi lần reflow, transport cost giảm ít nhất một nửa. Sau vài lần lặp, quỹ đạo gần như thẳng → sampling bằng Euler một bước là khả thi.

## 6. Code minh họa với PyTorch

### 6.1 Dataset tạo cặp $(X_0, X_1)$

```python
import torch
import torch.nn as nn

class RectifiedFlowDataset(torch.utils.data.Dataset):
    def __init__(self, x0_samples, x1_samples):
        assert len(x0_samples) == len(x1_samples)
        self.x0 = x0_samples
        self.x1 = x1_samples

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        t = torch.rand(1)
        x0 = self.x0[idx]
        x1 = self.x1[idx]
        xt = (1 - t) * x0 + t * x1
        target = x1 - x0
        return xt, t.unsqueeze(0), target
```

### 6.2 Velocity network

```python
class VelocityNet(nn.Module):
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t):
        t_embed = self.time_mlp(t)
        inp = torch.cat([x, t_embed], dim=-1)
        return self.net(inp)
```

### 6.3 Huấn luyện cơ bản

```python
def train_rectified_flow(model, dataset, epochs=100, lr=1e-3):
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running = 0.0
        for xt, t, target in loader:
            pred = model(xt, t)
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss = {running / len(loader):.6f}")
    return model
```

### 6.4 Sampling với ODE

```python
from torchdiffeq import odeint

def sample_rectified_flow(model, x0, num_steps=10):
    def ode_func(t, x):
        t_batch = torch.full((x.shape[0], 1), t, device=x.device)
        return model(x, t_batch)

    t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=x0.device)
    with torch.no_grad():
        traj = odeint(ode_func, x0, t_span, method='dopri5')
    return traj[-1]
```

### 6.5 Qui trình reflow đầy đủ

```python
def reflow_iteration(model, dim, batches=40, batch_size=256):
    x0_all, x1_all = [], []
    with torch.no_grad():
        for _ in range(batches):
            x0 = torch.randn(batch_size, dim)
            x1 = sample_rectified_flow(model, x0, num_steps=8)
            x0_all.append(x0)
            x1_all.append(x1)
    x0_all = torch.cat(x0_all)
    x1_all = torch.cat(x1_all)
    return RectifiedFlowDataset(x0_all, x1_all)

def train_with_reflow(data, dim, num_reflows=2):
    base_dataset = RectifiedFlowDataset(
        torch.randn(len(data), dim), data
    )
    model = VelocityNet(dim)
    model = train_rectified_flow(model, base_dataset)
    for k in range(num_reflows):
        print(f"\n=== Reflow iteration {k+1} ===")
        new_dataset = reflow_iteration(model, dim)
        model = VelocityNet(dim)
        model = train_rectified_flow(model, new_dataset)
    return model
```

## 7. Sampling nhanh: 1 bước hay vài bước?

Sau vài vòng reflow, đường đi gần như thẳng. Khi đó có thể dùng Euler một bước:

```python
def one_step_sample(model, z0):
    t_mid = torch.full((z0.shape[0], 1), 0.5, device=z0.device)
    v = model(z0, t_mid)
    return z0 + v  # dt = 1
```

Nếu muốn chắc chắn hơn, dùng vài bước:

```python
def few_step_sample(model, z0, steps=5):
    dt = 1.0 / steps
    x = z0
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device)
        v = model(x, t)
        x = x + v * dt
    return x
```

Trong thực nghiệm, 3–5 bước đã đủ đạt chất lượng ngang Glow-32-step với latency nhỏ hơn nhiều lần.

## 8. Thực nghiệm & các bẫy thường gặp

- **Coupling chưa tốt:** nếu dữ liệu quá khác Gaussian, chạy thêm reflow hoặc dùng kỹ thuật Flow Matching để khởi tạo coupling tốt hơn.
- **Sụp đổ số học khi dt lớn:** dùng chuẩn hóa đầu ra velocity (LayerNorm) hoặc clip $\|v\|$.
- **ODE solver thiếu ổn định:** mặc dù đường thẳng, đôi khi dùng RK4 thay vì Euler ở giai đoạn đầu để tránh sai lệch.
- **Mất đa dạng:** thêm nhiễu nhẹ vào $X_1 - X_0$ khi tính target giúp regularize.
- **Triển khai FP16:** giữ phép cộng log-likelihood ở FP32 vì dễ tràn số khi số bước ít.

## 9. Kết nối series & tài liệu

Rectified Flows là bước đệm giữa Glow và các kỹ thuật hiện đại như **Flow Matching** hay **Consistency Models** (xem các bài tiếp theo trong repo). Người thợ pha lê giờ đã có công cụ để vừa kiểm soát xác suất, vừa phục vụ khách gần như tức thì.

### Tài liệu nên đọc

1. Liu, X., et al. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR.
2. Liu, X., et al. (2023). *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*.
3. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*.
4. Albergo, M., et al. (2023). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*.
5. Tong, Z., et al. (2024). *Rectified Flow Meets Diffusion: Hybrid Approaches for Fast Generation*.

---

<script src="/assets/js/katex-init.js"></script>
