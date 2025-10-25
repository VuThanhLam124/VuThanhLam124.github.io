---
title: "Rectified Flows: Khi Đường Đi Trở Nên Thẳng"
date: "2025-01-21"
categor## 1. Câu chuyện: Đường vận chuyển ánh sáng

Sau buổi workshop về Glow, người thợ gốm nhận về một yêu cầu khó: **khách hàng muốn thử nhiều biến thể khi đứng trước quầy**, không phải chờ 30–60 giây để mô hình sinh hình ảnh. Các đường "dòng chảy" mà anh đã dùng vẫn quá cong; phải giải tích ODE nhiều bước mới ra sản phẩm.

Anh quan sát những đường nứt tự nhiên trên đất sét và nhận ra: **đường nào ngắn nhất thì hiệu quả nhất**. Nếu các flow của anh cũng thẳng như vậy, anh có thể từ latent Gaussian đến tác phẩm chỉ trong một bước. Từ trực giác đó, anh phát minh ra **Rectified Flows** – nghệ thuật "chỉnh thẳng" dòng chảy.

## 2. Cú hích từ quầy thử ánh sáng

Phòng trưng bày của xưởng ngày càng đông. Khách không chỉ muốn xem thành phẩm; họ muốn đứng ngay quầy thử, chọn một mẫu trên tablet và thấy kết quả tức thì. Những đường flow cong queo khiến người thợ phải điều chỉnh nhiều lần, khiến khách mất kiên nhẫn.based-models"
tags: ["rectified-flows", "reflow", "optimal-transport", "generative-models", "pytorch"]
excerpt: "Sau khi khám phá Flow Matching, người thợ gốm nhận ra đường đi của mình vẫn còn uốn cong. Liệu có cách nào làm thẳng chúng để nặn con rồng chỉ trong 1-2 bước thay vì 10-20 bước?"
author: "ThanhLamDev"
readingTime: 25
featured: true
---

# Rectified Flows: Khi Đường Đi Trở Nên Thẳng

**Người Thợ Gốm Khám Phá Nghệ Thuật "Chỉnh Thẳng" Đường Đi**

Chào mừng trở lại! Trong bài [Flow Matching](/posts/2025/flow-matching), người thợ gốm đã tìm ra cách học đơn giản hơn - thay vì tính xác suất (likelihood), chỉ cần học hướng di chuyển (regression). Nhưng câu chuyện chưa kết thúc...

## Mục lục

1. [Ngày thứ 4 - Phát hiện bất ngờ](#1-ngày-thứ-4---phát-hiện-bất-ngờ)
2. [Vấn đề: Đường đi vẫn uốn cong](#2-vấn-đề-đường-đi-vẫn-uốn-cong)
3. [Ý tưởng Rectified Flow: Làm thẳng đường đi](#3-ý-tưởng-rectified-flow-làm-thẳng-đường-đi)
4. [Toán học: Từ đường cong đến đường thẳng](#4-toán-học-từ-đường-cong-đến-đường-thẳng)
5. [Thuật toán Reflow: Học lại để thẳng hơn](#5-thuật-toán-reflow-học-lại-để-thẳng-hơn)
6. [Implementation PyTorch đầy đủ](#6-implementation-pytorch-đầy-đủ)
7. [One-Step Generation: Giấc mơ thành hiện thực](#7-one-step-generation-giấc-mơ-thành-hiện-thực)
8. [So sánh Flow Matching vs Rectified Flow](#8-so-sánh-flow-matching-vs-rectified-flow)
9. [Kết luận](#9-kết-luận)

---

## 1. Ngày thứ 4 - Phát hiện bất ngờ

### Buổi sáng đánh giá thành quả

Ngày thứ 4, người thợ gốm thức dậy sớm với tâm trạng phấn khởi. Anh đã hoàn thành 30 con rồng bằng phương pháp Flow Matching mới - nhanh hơn CNF rất nhiều!

Nhưng khi ngồi uống cà phê sáng, anh để ý một điều:

**"Tại sao vẫn cần 10-20 bước để nặn một con rồng?"**

Anh mở sổ tay, vẽ lại quỹ đạo của một hạt đất từ khối cầu (t=0) đến con rồng (t=1):

```
Lý thuyết (mong muốn):
  Khối cầu (0,0,0) --------→ Con rồng (5,3,2)
                    Đường thẳng!

Thực tế (quan sát):
  Khối cầu (0,0,0) --→ ↗ → ↘ --→ Con rồng (5,3,2)
                    Đường... hơi cong!
```

"Ồ..." Anh nhíu mày. "Đường đi vẫn còn uốn cong. Mỗi khúc cong nghĩa là tôi phải 'điều chỉnh tay' nhiều lần!"

### Thí nghiệm đo đường đi

Anh quyết định đo chính xác. Lấy 1 con rồng mẫu, anh ghi lại vị trí tay mình ở 10 thời điểm:

```python
# Dữ liệu thực tế từ Flow Matching
t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x = [
    (0.0, 0.0, 0.0),   # t=0.0: Khối cầu
    (0.3, 0.2, 0.1),   # t=0.1
    (0.8, 0.5, 0.3),   # t=0.2
    (1.5, 0.9, 0.6),   # t=0.3
    (2.1, 1.4, 1.0),   # t=0.4
    (2.8, 1.8, 1.3),   # t=0.5
    (3.4, 2.2, 1.6),   # t=0.6
    (4.0, 2.5, 1.8),   # t=0.7
    (4.5, 2.8, 1.9),   # t=0.8
    (4.8, 2.9, 2.0),   # t=0.9
    (5.0, 3.0, 2.0),   # t=1.0: Con rồng hoàn chỉnh
]
```

Anh vẽ đồ thị và thấy rõ: **đường đi KHÔNG PHẢI đường thẳng từ (0,0,0) đến (5,3,2)!**

Nếu là đường thẳng hoàn hảo, tại t=0.5 phải ở (2.5, 1.5, 1.0), nhưng thực tế lại ở (2.8, 1.8, 1.3) - **lệch khỏi đường thẳng!**

---

## 1. Câu chuyện: Đường vận chuyển ánh sáng

Sau buổi workshop về Glow, người thợ gốm nhận về một yêu cầu khó: **khách hàng muốn thử nhiều biến thể khi đứng trước quầy**, không phải chờ 30–60 giây để mô hình sinh hình ảnh. Các đường “dòng chảy” mà anh đã dùng vẫn quá cong; phải giải tích ODE nhiều bước mới ra sản phẩm.

Anh quan sát những đường nứt tự nhiên trên đất sét và nhận ra: **tia nào đi gần đường thẳng nhất thì sáng rõ nhất**. Nếu các flow của anh cũng thẳng như vậy, anh có thể từ latent Gaussian đến tác phẩm chỉ trong một bước. Từ trực giác đó, anh phát minh ra **Rectified Flows** – nghệ thuật “chỉnh thẳng” dòng chảy.

## 2. Cú hích từ quầy thử ánh sáng

Phòng trưng bày của xưởng ngày càng đông. Khách không chỉ muốn xem thành phẩm; họ muốn đứng ngay quầy thử, chọn một tông màu trên tablet và thấy kết quả biến đổi đất sét tức thì. Những đường flow cong queo khiến người thợ phải vặn núm điều khiển nhiều lần, khiến khách mất kiên nhẫn.

Anh ghi chép vào sổ tay: “Muốn phục vụ tại quầy, mình cần **đường biến đổi thật thẳng**. Càng ít vòng vo, càng ít thời gian lấy mẫu.” Đây chính là động lực để anh nghiên cứu Rectified Flows: vẫn là câu chuyện từ Gaussian tới tác phẩm, nhưng mục tiêu mới là khiến hành trình giữa hai điểm trở thành một đoạn thẳng gọn gàng.

## 3. Từ đường cong đến đường thẳng: trực giác Rectified Flow

Ở các bài trước, flow được mô tả bởi ODE:

$$
\frac{d x_t}{dt} = v_t(x_t), \quad x_0 \sim \pi_0 \ (\text{Gaussian}), \quad x_1 \sim \pi_1 \ (\text{data})
$$

**Chú thích:** $x_t$ là trạng thái tại thời điểm $t$; $\frac{d x_t}{dt}$ (hay $\dot{x}_t$) là đạo hàm theo thời gian; $v_t$ là trường vận tốc; ký hiệu $\sim$ nghĩa là “được lấy mẫu từ”.

Nếu vector field $v_t$ uốn cong, ta phải dùng nhiều bước tích phân ⇒ chậm.

**Rectified Flow** đặt mục tiêu:

- Duy trì đúng phân phối đầu-cuối.
- Tạo ra đường đi $x_t$ càng thẳng càng tốt.
- Giảm số lần đánh giá trường vector khi sampling.

Độ “thẳng” được đo bằng chi phí vận chuyển (transport cost):

$$
\mathcal{C} = \mathbb{E}\left[\int_0^1 \|v_t(X_t)\|^2 dt\right]
$$

**Chú thích:** $\mathbb{E}[\cdot]$ là kỳ vọng (trung bình); $\|\cdot\|$ là chuẩn Euclid; tích phân từ $0$ đến $1$ đo tổng năng lượng vận tốc trong toàn bộ hành trình.

Đường thẳng tối ưu (geodesic) tương ứng với chi phí thấp nhất. Rectified Flows tìm cách xấp xỉ geodesic đó – giống như người thợ tìm cách mài đường truyền ánh sáng ngắn nhất.

## 4. Toán học nền tảng

### 4.1 Linear interpolant và vận tốc

Cho cặp mẫu $(X_0, X_1)$ đã được **coupling tối ưu**. Khi ta nội suy tuyến tính:

$$
X_t = (1 - t) X_0 + t X_1
$$

**Chú thích:** Đây là phép nội suy tuyến tính giữa hai điểm $X_0$ và $X_1$; hệ số $(1-t)$, $t$ đảm bảo khi $t=0$ ta ở $X_0$, khi $t=1$ ta ở $X_1$.

Ta có vận tốc không đổi:

$$
\dot{X}_t = X_1 - X_0
$$

**Chú thích:** Dấu chấm biểu diễn đạo hàm theo thời gian. Vì nội suy tuyến tính nên vận tốc luôn bằng hiệu $X_1 - X_0$.

Điều kiện cần: tồn tại coupling đủ tốt giữa hai phân phối để đoạn thẳng này “hợp lý”.

### 4.2 Velocity field mục tiêu

Từ quan sát trên:

$$
v_t(x) = \mathbb{E}\big[X_1 - X_0 \mid X_t = x\big]
$$

**Chú thích:** Đây là kỳ vọng có điều kiện – trung bình của hiệu $X_1 - X_0$ khi biết trạng thái hiện tại bằng $x$.

Trong thực tế ta dùng mạng neural $v_\theta(x, t)$ để xấp xỉ và tối thiểu hóa:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, X_t}\left[\|v_\theta(X_t, t) - (X_1 - X_0)\|^2\right]
$$

**Chú thích:** $\mathcal{L}$ là hàm loss; chỉ số $\theta$ biểu diễn tham số mạng; bình phương chuẩn $\|\cdot\|^2$ là mean squared error giữa vận tốc dự đoán và vận tốc mục tiêu.
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

## 6. Implementation PyTorch đầy đủ

### 6.1. Dataset cho Rectified Flow

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RectifiedFlowDataset(Dataset):
    """Dataset với cặp cố định (z0, x1)"""
    
    def __init__(self, z0_samples, x1_samples):
        """
        Args:
            z0_samples: (N, dim) - Gaussian starting points
            x1_samples: (N, dim) - Data endpoints
        """
        assert len(z0_samples) == len(x1_samples)
        self.z0 = z0_samples
        self.x1 = x1_samples
    
    def __len__(self):
        return len(self.z0)
    
    def __getitem__(self, idx):
        # Sample random time
        t = torch.rand(1)
        
        # Get fixed pair
        z0 = self.z0[idx]
        x1 = self.x1[idx]
        
        # Linear interpolation (straight line!)
        xt = (1 - t) * z0 + t * x1
        
        # Constant velocity target
        velocity = x1 - z0
        
        return xt, t, velocity
```

### 6.2. Velocity Network (giống Flow Matching)

```python
class VelocityNet(nn.Module):
    """Time-conditioned vector field v_θ(x, t)"""
    
    def __init__(self, dim, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.dim = dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            v: (batch_size, dim) - predicted velocity
        """
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)
```

### 6.3. Training Rectified Flow

```python
def train_rectified_flow(model, dataset, epochs=100, lr=1e-3, device='cuda'):
    """Train một model Rectified Flow"""
    
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for xt, t, target_v in loader:
            xt = xt.to(device)
            t = t.to(device)
            target_v = target_v.to(device)
            
            # Predict velocity
            pred_v = model(xt, t)
            
            # MSE loss (regression!)
            loss = ((pred_v - target_v) ** 2).mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
    
    return model
```

### 6.4. Sampling (ODE Solver)

```python
def sample_rectified_flow(model, z0, num_steps=10, device='cuda'):
    """
    Sinh mẫu bằng cách giải ODE từ z0
    
    Args:
        model: Trained velocity network
        z0: (batch_size, dim) - Gaussian starting points
        num_steps: Số bước Euler
    """
    model.eval()
    x = z0.to(device)
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((x.shape[0], 1), i * dt, device=device)
            v = model(x, t)
            x = x + v * dt  # Euler step
    
    return x
```

### 6.5. Reflow Algorithm - Hoàn chỉnh

```python
def reflow_iteration(model_old, dim, num_samples=10000, num_steps=20, device='cuda'):
    """
    Một iteration của Reflow:
    1. Dùng model cũ sinh cặp (z0, x1) mới
    2. Return dataset mới
    """
    print(f"Generating {num_samples} new pairs...")
    
    z0_list = []
    x1_list = []
    
    model_old.eval()
    batch_size = 256
    num_batches = num_samples // batch_size
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Sample Gaussian
            z0 = torch.randn(batch_size, dim)
            
            # Run flow with old model
            x1 = sample_rectified_flow(model_old, z0, num_steps, device)
            
            z0_list.append(z0.cpu())
            x1_list.append(x1.cpu())
    
    # Concatenate all
    z0_all = torch.cat(z0_list, dim=0)
    x1_all = torch.cat(x1_list, dim=0)
    
    print(f"Generated {len(z0_all)} pairs!")
    
    return RectifiedFlowDataset(z0_all, x1_all)

def train_with_reflow(data, dim, num_reflows=2, epochs_per_reflow=100):
    """
    Pipeline hoàn chỉnh: Flow Matching → Reflow iterations
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Train initial Flow Matching model
    print("="*60)
    print("Step 1: Training initial Flow Matching model")
    print("="*60)
    
    # Create initial dataset (random coupling)
    z0_init = torch.randn(len(data), dim)
    dataset_init = RectifiedFlowDataset(z0_init, data)
    
    model = VelocityNet(dim).to(device)
    model = train_rectified_flow(model, dataset_init, epochs=epochs_per_reflow, device=device)
    
    # Step 2: Reflow iterations
    for k in range(num_reflows):
        print("\n" + "="*60)
        print(f"Step {k+2}: Reflow iteration {k+1}/{num_reflows}")
        print("="*60)
        
        # Generate new dataset using current model
        dataset_new = reflow_iteration(model, dim, num_samples=len(data), device=device)
        
        # Train new model on new dataset
        model = VelocityNet(dim).to(device)
        model = train_rectified_flow(model, dataset_new, epochs=epochs_per_reflow, device=device)
    
    return model
```

### 6.6. Full Example

```python
# Giả sử có dữ liệu 2D
import numpy as np
import matplotlib.pyplot as plt

# Generate toy data (spiral)
theta = np.linspace(0, 4*np.pi, 1000)
r = theta
data_np = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
data = torch.tensor(data_np, dtype=torch.float32)

# Normalize
data = (data - data.mean(0)) / data.std(0)

# Train with reflow
model_final = train_with_reflow(data, dim=2, num_reflows=2, epochs_per_reflow=50)

# Sample from final model
z0 = torch.randn(1000, 2)
samples = sample_rectified_flow(model_final, z0, num_steps=5)  # Chỉ cần 5 bước!

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.title("Original Data")

plt.subplot(132)
plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5)
plt.title("Generated (5 steps)")

plt.subplot(133)
# Compare with many steps
samples_many = sample_rectified_flow(model_final, z0, num_steps=50)
plt.scatter(samples_many[:, 0].cpu(), samples_many[:, 1].cpu(), alpha=0.5)
plt.title("Generated (50 steps)")

plt.tight_layout()
plt.show()
```

## 7. One-Step Generation: Giấc mơ thành hiện thực

### Ngày thứ 6 - Thử nghiệm cực đoan

Sáng ngày thứ 6, sau 2 lần reflow, người thợ gốm nghĩ:

**"Nếu đường đi ĐÃ THẲNG, tôi có thể nhảy TRỰC TIẾP từ đầu đến cuối không?"**

Anh thử nghiệm táo bạo:

```python
def one_step_generation(model, z0):
    """
    Sampling CHỈ 1 BƯỚC!
    
    Ý tưởng: Nếu đường thẳng, vận tốc không đổi
    → Chỉ cần hỏi model tại t=0.5 (giữa đường)
    → Nhảy thẳng đến cuối!
    """
    model.eval()
    with torch.no_grad():
        # Hỏi vận tốc tại giữa đường
        t_mid = torch.full((z0.shape[0], 1), 0.5)
        v = model(z0, t_mid)
        
        # Nhảy toàn bộ quãng đường (dt = 1.0)
        x1 = z0 + v * 1.0
    
    return x1
```

Anh test:

```python
# Model sau 2 lần reflow
z0 = torch.randn(100, dim)

# 1 bước
samples_1step = one_step_generation(model_reflow2, z0)

# 5 bước
samples_5step = sample_rectified_flow(model_reflow2, z0, num_steps=5)

# So sánh chất lượng
print(f"1-step quality: {evaluate(samples_1step)}")  # 0.82
print(f"5-step quality: {evaluate(samples_5step)}")  # 0.85
```

"Chỉ chênh 3%!" Anh kinh ngạc. "Với ứng dụng thực tế, **1 bước là đủ**!"

### So sánh tốc độ

Anh đo thời gian:

| Method | Steps | Time/sample | Quality |
|--------|-------|-------------|---------|
| Flow Matching | 20 | 200ms | 0.90 |
| Rectified Flow (1 reflow) | 5 | 50ms | 0.88 |
| Rectified Flow (2 reflows) | 2 | 20ms | 0.87 |
| **Rectified Flow (2 reflows) + 1-step** | **1** | **10ms** | **0.82** |

**Kết luận:** Nhanh hơn **20 lần** với chất lượng chấp nhận được!

### Khi nào dùng 1-step vs few-steps?

Người thợ gốm ghi chú:

**1-step (dt=1.0):**
- Ưu điểm: Cực nhanh (real-time)
- Nhược điểm: Chất lượng giảm 5-10%
- Dùng khi: Demo, interactive apps, real-time generation

**Few-steps (3-5 steps):**
- Ưu điểm: Balance tốt giữa tốc độ và chất lượng
- Nhược điểm: Chậm hơn 1-step
- Dùng khi: Production, cần chất lượng cao

**Code adaptive:**

```python
def adaptive_sampling(model, z0, quality_target=0.85):
    """Tự động chọn số bước dựa trên quality mong muốn"""
    
    if quality_target < 0.83:
        # OK với 1-step
        return one_step_generation(model, z0), 1
    elif quality_target < 0.88:
        # Cần 3-5 steps
        return sample_rectified_flow(model, z0, num_steps=3), 3
    else:
        # Cần nhiều steps hơn
        return sample_rectified_flow(model, z0, num_steps=10), 10
```

## 8. So sánh Flow Matching vs Rectified Flow

### Bảng so sánh tổng quan

| Khía cạnh | Flow Matching | Rectified Flow (1 reflow) | Rectified Flow (2 reflows) |
|-----------|---------------|---------------------------|----------------------------|
| **Coupling** | Random | Fixed (from model) | Fixed (straighter) |
| **Đường đi** | Hơi cong | Thẳng hơn | Gần như thẳng hoàn toàn |
| **Transport cost** | ~52 | ~26 (giảm 50%) | ~13 (giảm 75%) |
| **Training time** | 1× | 2× (train 2 lần) | 3× (train 3 lần) |
| **Sampling steps** | 10-20 | 3-5 | 1-2 |
| **Sampling speed** | 1× | 4× nhanh hơn | 10-20× nhanh hơn |
| **Quality** | 0.90 | 0.88 | 0.85 (1-step) / 0.87 (2-steps) |
| **Use case** | General purpose | Fast sampling needed | Real-time, interactive |

### Khi nào dùng gì?

**Flow Matching:**
- Khi cần chất lượng cao nhất
- Dataset nhỏ, không muốn train nhiều lần
- Không quan tâm tốc độ sampling

**Rectified Flow (1 reflow):**
- Balance tốt giữa quality và speed
- Production systems cần sampling nhanh
- Có thể train 2 lần

**Rectified Flow (2+ reflows):**
- Real-time applications
- Interactive demos
- Mobile/edge devices
- Cần 1-step generation

### Ví dụ ứng dụng thực tế

Người thợ gốm ghi chép các trường hợp sử dụng:

**1. Xưởng nghệ thuật (Flow Matching):**
- Tạo tác phẩm cao cấp
- Có thể đợi 2-3 giây
- Chất lượng là ưu tiên số 1

**2. Phòng trưng bày (Rectified 1 reflow):**
- Khách xem mẫu
- Đợi 0.5-1 giây OK
- Chất lượng vẫn tốt

**3. Quầy thử nghiệm (Rectified 2 reflows + 1-step):**
- Khách tương tác trực tiếp
- Cần phản hồi tức thì (<100ms)
- Chất lượng chấp nhận được là đủ

## 9. Kết luận

### Câu chuyện kết thúc - Ngày thứ 7

Ngày thứ 7, người thợ gốm hoàn thành đơn hàng 100 con rồng đúng deadline!

Anh ngồi uống trà, nhìn lại hành trình:

**Ngày 1-2 (CNF):**
- "Tôi phải tính xác suất mỗi con rồng - mất 41 ngày!"
- "Không kịp deadline..."

**Ngày 3 (Flow Matching - Eureka!):**
- "Khoan! Tôi không cần tính xác suất!"
- "Chỉ cần học HƯỚNG di chuyển!"
- "Regression thay vì likelihood - đơn giản hơn 100 lần!"

**Ngày 4-5 (Rectified Flow - Tiếp tục cải tiến!):**
- "Đường đi vẫn cong - vẫn cần 10-20 bước"
- "Nếu tạo cặp CỐ ĐỊNH từ model cũ..."
- "Đường đi trở nên THẲNG - chỉ cần 1-2 bước!"

**Ngày 6-7 (Hoàn thành):**
- 100 con rồng xong
- Mỗi con chỉ mất vài giây
- Chất lượng tuyệt vời!

### Bài học quan trọng

Anh ghi vào trang cuối sổ tay:

> **Ba bài học về Flow-based Models:**
>
> 1. **CNF → Flow Matching:** Đổi từ likelihood sang regression
>    - Đơn giản hóa bài toán
>    - Không cần trace, không cần ODE ngược
>    - Nhanh hơn 100× khi training
>
> 2. **Flow Matching → Rectified Flow:** Làm thẳng đường đi
>    - Dùng model cũ sinh cặp mới
>    - Đường đi từ cong → thẳng
>    - Nhanh hơn 10-20× khi sampling
>
> 3. **Rectified Flow → One-Step:** Giấc mơ thành hiện thực
>    - 1 bước thay vì 1000 bước (Diffusion)
>    - Real-time generation
>    - Tương lai của generative models

### Điểm chính cần nhớ

1. **Reflow = "Học lại với cặp tốt hơn"**
   - Iteration 1: Random coupling → Đường cong
   - Iteration 2: Fixed coupling → Đường thẳng hơn
   - Iteration 3: Straighter coupling → Gần như thẳng hoàn toàn

2. **Transport cost giảm theo cấp số nhân**
   - Mỗi lần reflow: Cost giảm ≥ 50%
   - 2-3 iterations là đủ

3. **Trade-off: Training time vs Sampling speed**
   - Train 1 lần: Sampling chậm (10-20 steps)
   - Train 3 lần: Sampling cực nhanh (1-2 steps)

4. **One-step generation là khả thi**
   - Sau 2 reflows, đường đi đủ thẳng
   - Quality drop ~5-10% so với multi-step
   - Tốc độ tăng 10-20×

### So sánh với series

| Bài | Method | Insight chính | Speed (sampling) |
|-----|--------|---------------|------------------|
| 1 | CNF/FFJORD | Continuous flows | Chậm (600 NFE) |
| 2 | **Flow Matching** | Likelihood → Regression | Trung bình (10-20 steps) |
| 3 | **Rectified Flow** | Đường cong → Đường thẳng | Nhanh (1-5 steps) |

### Hướng phát triển tiếp theo

Người thợ gốm nghĩ về tương lai:

1. **Consistency Models** (bài tiếp theo)
   - Học trực tiếp ánh xạ x₀ → x₁
   - Không cần ODE solver
   - One-step generation từ đầu

2. **Stochastic Interpolants**
   - Kết hợp Rectified Flow và Diffusion
   - Best of both worlds

3. **Applications**
   - Stable Diffusion 3 dùng Rectified Flow
   - InstaFlow: Text-to-Image in 1 step
   - Real-time video generation

### Lời kết

"Từ 41 ngày xuống còn 7 ngày," anh mỉm cười. "Và từ 20 bước xuống còn 1 bước."

"Đôi khi, giải pháp không phải là làm việc chăm chỉ hơn, mà là **suy nghĩ thông minh hơn**."

**Bài học cuối cùng:** Con đường thẳng nhất không phải lúc nào cũng rõ ràng ngay từ đầu. Đôi khi, ta phải đi qua con đường cong (Flow Matching) để nhận ra cách làm thẳng nó (Rectified Flow).

---

## Tài liệu tham khảo

1. **Liu, X., et al. (2022)** - "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR)
2. **Liu, X., et al. (2023)** - "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"
3. **Lipman, Y., et al. (2023)** - "Flow Matching for Generative Modeling" (ICLR)
4. **Albergo, M., et al. (2023)** - "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"
5. **Esser, P., et al. (2024)** - "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Stable Diffusion 3)

---

**Bài trước:** [Flow Matching: Từ Likelihood đến Regression](/posts/2025/flow-matching)

**Bài tiếp theo:** [Consistency Models: One-Step Generation](/posts/2025/consistency-models)

<script src="/assets/js/katex-init.js"></script>
