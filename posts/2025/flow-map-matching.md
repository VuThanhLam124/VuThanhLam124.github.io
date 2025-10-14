---
title: "Flow Map Matching: Khi bản đồ dòng chảy thay cho trường vận tốc"
date: "2025-03-05"
category: "flow-based-models"
tags: ["flow-map-matching", "flow-matching", "generative-models", "pytorch"]
excerpt: "Chương tiếp theo của series xưởng pha lê: Flow Map Matching giúp người thợ lưu cả quỹ đạo thay vì chỉ vận tốc. Bài viết kết hợp câu chuyện, trực giác toán học, loss function, code PyTorch và so sánh thực tiễn."
author: "ThanhLamDev"
readingTime: 16
featured: false
---

# Flow Map Matching

**Sau Rectified Flows, người thợ pha lê nhận ra: đôi khi lưu trữ cả đường đi còn tiện hơn việc nhớ từng vận tốc. Flow Map Matching là “bản đồ” giúp anh nhảy thẳng đến kết quả mà không cần giải ODE mỗi lần.**

## Mục lục

1. [Câu chuyện: Bản đồ trong sổ tay của người thợ](#1-câu-chuyện-bản-đồ-trong-sổ-tay-của-người-thợ)
2. [Trực giác: Từ vận tốc sang bản đồ vị trí](#2-trực-giác-từ-vận-tốc-sang-bản-đồ-vị-trí)
3. [Toán học Flow Map Matching](#3-toán-học-flow-map-matching)
4. [Chiến lược huấn luyện](#4-chiến-lược-huấn-luyện)
5. [Sampling & composition](#5-sampling--composition)
6. [Code PyTorch mẫu](#6-code-pytorch-mẫu)
7. [So sánh với Flow Matching & Rectified Flow](#7-so-sánh-với-flow-matching--rectified-flow)
8. [Mẹo thực nghiệm & lưu ý](#8-mẹo-thực-nghiệm--lưu-ý)
9. [Kết nối series & tài liệu](#9-kết-nối-series--tài-liệu)

---

## 1. Câu chuyện: Bản đồ trong sổ tay của người thợ

Ở phòng thử ánh sáng, Rectified Flows đã giúp người thợ tạo đường thẳng nhanh hơn. Nhưng khi khách muốn lưu lại “preset” yêu thích, anh lại phải chạy lại toàn bộ quy trình để tái tạo quỹ đạo. Vậy tại sao không **ghi hẳn bản đồ** từ khối pha lê chuẩn đến tác phẩm cuối cùng?

Anh mở sổ tay, vẽ các điểm mốc theo thời gian $t = 0 \to 1$ rồi đánh dấu trạng thái khối pha lê ở mỗi mốc. Mỗi khi khách quay lại, anh chỉ việc tra bản đồ: tại thời điểm $t$, tác phẩm ở đâu? Flow Map Matching chính là cách học trực tiếp $\phi_t(x_0)$ – vị trí của mẫu sau thời gian $t$ – thay vì chỉ học vận tốc $v_t$.

## 2. Trực giác: Từ vận tốc sang bản đồ vị trí

- **Flow Matching:** học $v_t(x)$, rồi tích phân (ODE) để tìm vị trí.
- **Rectified Flow:** cố gắng làm đường đi thẳng để tích phân nhanh hơn.
- **Flow Map Matching:** bỏ qua bước tích phân, học trực tiếp $\phi_t(x_0)$ sao cho
$$
X_t = \phi_t(X_0), \quad \phi_0(x) = x, \quad \phi_1(X_0) \sim \text{data}.
$$

> **Chú thích ký hiệu:** $\phi_t$ là map đưa trạng thái gốc $X_0$ tới vị trí ở thời điểm $t$; $\phi_0$ là đồng nhất; ký hiệu $\sim$ nghĩa là “có phân phối”.

Ta có thể coi $\phi_t$ là bộ “bản đồ thời gian thực” – tra cứu vị trí ngay lập tức. Đổi lại, ta cần học một hàm có đầu ra có ý nghĩa vật lý mạnh hơn, và phải đảm bảo các bản đồ ghép lại vẫn hợp lý (tính chất composition).

## 3. Toán học Flow Map Matching

### 3.1 Liên hệ với trường vận tốc

Nếu đã có $v_t$, ta có thể thu được flow map qua tích phân:

$$
\phi_t(x) = x + \int_0^t v_s\big(\phi_s(x)\big) ds.
$$

> **Chú thích:** Tích phân cộng dồn vận tốc theo thời gian; biến $s$ dùng để phân biệt với $t$ trong tích phân.

Ngược lại, nếu biết $\phi_t$, vận tốc suy ra bằng đạo hàm theo thời gian:

$$
v_t(x) = \left.\frac{d}{dt}\phi_t(z)\right|_{z = \phi_t^{-1}(x)}.
$$

> **Chú thích:** Ta đạo hàm $\phi_t$ rồi thế $z = \phi_t^{-1}(x)$ để chuyển từ không gian gốc sang vị trí hiện tại.

### 3.2 Mục tiêu học $\phi_\theta$

Ta giả định đã có cặp $(X_0, X_1)$ và nội suy tuyến tính:

$$
X_t = (1 - t) X_0 + t X_1.
$$

Loss cơ bản:

$$
\mathcal{L}_{\text{match}}(\theta) = \mathbb{E}_{X_0, X_1, t}\big[\big\|\phi_\theta(X_0, t) - X_t\big\|^2\big].
$$

> **Chú thích:** $\mathcal{L}_{\text{match}}$ là loss cốt lõi; ký hiệu $\|\cdot\|$ là chuẩn Euclid; kỳ vọng lấy trung bình trên $X_0, X_1$ và thời gian $t$.

Để các bản đồ ghép được với nhau, thêm loss **composition/consistency**:

$$
\mathcal{L}_{\text{cons}}(\theta) = \mathbb{E}_{x, s, t}\big[\|\phi_\theta(\phi_\theta(x, s), t) - \phi_\theta(x, s+t)\|^2\big],
$$

> **Chú thích:** Term này đảm bảo tính chất ghép bản đồ; $s, t$ là hai thời gian nhỏ; $\phi_\theta(\phi_\theta(x, s), t)$ nghĩa là áp dụng map $t$ sau khi đã đi $s$.

trong đó $s, t$ được lấy sao cho $s+t \le 1$.

### 3.3 Ràng buộc biên

- $\phi_\theta(x, 0) \approx x$ để đảm bảo identity.
- Với $t=1$, $\phi_\theta$ phải khớp phân phối data ⇒ có thể thêm regularizer likelihood (nếu cần) hoặc training adversarial khi áp dụng thực tế.

## 4. Chiến lược huấn luyện

1. **Chuẩn bị cặp $(X_0, X_1)$**: $X_0$ lấy từ Gaussian chuẩn, $X_1$ từ dữ liệu. Có thể dùng kỹ thuật dequantize đối với ảnh.
2. **Sampling thời gian**: $t \sim \mathcal{U}[0, 1]$.
3. **Loss tổng**:
$$
\mathcal{L} = \mathcal{L}_{\text{match}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}} + \lambda_{\text{id}} \| \phi_\theta(x, 0) - x \|^2.
$$

> **Chú thích:** $\lambda_{\text{cons}}, \lambda_{\text{id}}$ là hệ số điều chỉnh; hạng cuối buộc mô hình giữ identity tại $t=0$.
4. **Optimization**: Adam / AdamW, warmup learning rate và gradient clipping giúp ổn định vì output có giá trị tuyệt đối lớn.

## 5. Sampling & composition

Sau khi huấn luyện, ta có thể:

- **Sampling một bước**: $\hat{x}_1 = \phi_\theta(z, 1)$ với $z \sim \mathcal{N}(0, I)$.
- **Sampling nhiều bước**: lặp $\phi_\theta(\cdot, \Delta t)$ nhiều lần để giảm lỗi tích lũy.
- **Composition**: thay vì gọi lại model từ $t=0$, ta có thể nối các bước cục bộ: $\phi_{\Delta t} \circ \phi_{\Delta t} \circ \dots$.

Đây chính là điểm mạnh của flow map: linh hoạt về cách rải thời gian, tận dụng caching để tái sử dụng kết quả.

## 6. Code PyTorch mẫu

### 6.1 Kiến trúc Flow Map

```python
import torch
import torch.nn as nn

class FlowMapNet(nn.Module):
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, x, t):
        t_embed = self.time_mlp(t)
        inp = torch.cat([x, t_embed], dim=-1)
        delta = self.net(inp)
        return x + t * delta  # đảm bảo gần identity khi t nhỏ
```

### 6.2 Huấn luyện với consistency

```python
def train_flow_map(model, x0_loader, x1_loader, epochs=100,
                   lambda_cons=0.1, device="cuda"):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(epochs):
        running = 0.0
        for x0, x1 in zip(x0_loader, x1_loader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            bsz = x0.size(0)

            t = torch.rand(bsz, 1, device=device)
            target = (1 - t) * x0 + t * x1
            pred = model(x0, t)
            loss_match = ((pred - target) ** 2).mean()

            s = torch.rand(bsz, 1, device=device)
            u = torch.rand(bsz, 1, device=device)
            mask = (s + u).clamp(max=1.0)  # tránh vượt quá 1

            phi_u = model(x0, u)
            comp = model(phi_u, s)
            direct = model(x0, mask)
            loss_cons = ((comp - direct) ** 2).mean()

            identity_loss = ((model(x0, torch.zeros_like(t)) - x0) ** 2).mean()

            loss = loss_match + lambda_cons * loss_cons + 0.1 * identity_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            running += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss = {running / len(x0_loader):.6f}")
```

### 6.3 Sampling

```python
@torch.no_grad()
def sample_flow_map(model, num_samples, dim, device="cuda", steps=1):
    z = torch.randn(num_samples, dim, device=device)
    if steps == 1:
        t = torch.ones(num_samples, 1, device=device)
        return model(z, t)

    dt = 1.0 / steps
    x = z
    for _ in range(steps):
        t_step = torch.ones(num_samples, 1, device=device) * dt
        x = model(x, t_step)
    return x
```

### 6.4 Conditional Flow Map (tùy chọn)

```python
class ConditionalFlowMap(nn.Module):
    def __init__(self, dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + 64 + 64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t, cond):
        inp = torch.cat([x, self.time_mlp(t), self.cond_mlp(cond)], dim=-1)
        delta = self.net(inp)
        return x + t * delta
```

## 7. So sánh với Flow Matching & Rectified Flow

| Tiêu chí | Flow Matching | Rectified Flow | Flow Map Matching |
|----------|---------------|----------------|-------------------|
| Đối tượng học | $v_t(x)$ | $v_t(x)$ (được “thẳng hóa”) | $\phi_t(x)$ |
| Huấn luyện | Có thể cần ODE / adjoint | Có reflow iterations | Chỉ cần regression |
| Suy luận | Giải ODE (nhiều bước) | Vài bước Euler | Đánh giá trực tiếp |
| Bộ nhớ | Nhỏ | Trung bình | Cao hơn (lưu vị trí) |
| Kiểm soát thời gian | Liên tục | Gần tuyến tính | Rất linh hoạt (chọn t tùy ý) |

Khi bạn ưu tiên tốc độ suy luận và muốn cache kết quả từng bước, Flow Map Matching đặc biệt hữu ích. Nếu cần mô hình hóa liên tục chính xác hoặc hạn chế bộ nhớ, Flow Matching/Rectified Flow vẫn phù hợp hơn.

## 8. Mẹo thực nghiệm & lưu ý

- **Clamp thời gian**: đảm bảo $s + t \le 1$ khi tính consistency; nếu không, hãy dùng modulo hoặc tái chuẩn hóa.
- **Regularize identity**: thêm loss nhỏ buộc $\phi(x, 0) \approx x$ giúp model ổn định.
- **Gradient clipping**: tránh bùng nổ khi output có biên độ lớn.
- **Cache step nhỏ**: trong inference, có thể precompute $\phi_{\Delta t}(x)$ cho một số $\Delta t$ cố định rồi tái sử dụng như lookup table.
- **Kết hợp likelihood**: nếu cần điểm số định lượng, huấn luyện thêm một head ước lượng log-density hoặc sử dụng change-of-variable với Jacobian của $\phi_t$ (có thể tính qua autograd).

## 9. Kết nối series & tài liệu

Flow Map Matching mở ra khả năng biến xưởng pha lê thành “trung tâm lưu trữ preset”: mỗi bản đồ $t$ là một trạng thái trung gian, có thể chia sẻ giữa các nghệ nhân. Từ đây, series sẽ dẫn sang **Conditional Flow Matching** và **Schrödinger Bridge** – nơi câu chuyện mở rộng sang điều kiện và ràng buộc vật lý.

### Tài liệu nên đọc

1. Albergo, M. S., & Vanden-Eijnden, E. (2023). *Building Normalizing Flows with Stochastic Interpolants*.
2. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*.
3. Tong, Z., et al. (2024). *Flow Map Matching: Learning Transport Maps without Solving ODEs*.
4. Liu, X., et al. (2022). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*.

---

<script src="/assets/js/katex-init.js"></script>
