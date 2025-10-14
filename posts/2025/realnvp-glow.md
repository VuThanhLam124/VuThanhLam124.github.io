# Real NVP & Glow: Invertible Architectures

**Ngày đăng:** 14/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Flow-based Models, Deep Learning

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Real NVP Architecture](#real-nvp-architecture)
3. [Coupling Layers](#coupling-layers)
4. [Glow: Generative Flow](#glow-generative-flow)
5. [Invertible 1x1 Convolutions](#invertible-1x1-convolutions)
6. [ActNorm](#actnorm)
7. [Implementation](#implementation)
8. [Applications](#applications)

---

## Giới thiệu

Real NVP (Real-valued Non-Volume Preserving transformations) và Glow là hai architectures quan trọng trong history của flow-based models. Chúng giải quyết challenge chính của Normalizing Flows: làm sao design bijective transformations có **efficient Jacobian computation** mà vẫn đủ **expressive**.

**Key innovations:**
- **Coupling layers** - partition và transform strategy
- **Invertible 1x1 convolutions** (Glow) - learnable permutations
- **ActNorm** - activation normalization thay batch norm
- **Multi-scale architecture** - hierarchical latent variables

## 1. Real NVP Architecture

### 1.1 Coupling Layer Mechanism

Core idea: Chia input thành 2 parts, transform một part dựa trên part kia.

Given input $x \in \mathbb{R}^D$, partition thành $x = [x_{1:d}, x_{d+1:D}]$:

**Forward:**
$$
\begin{aligned}
y_{1:d} &= x_{1:d} \\
y_{d+1:D} &= x_{d+1:D} \odot \exp(s(x_{1:d})) + t(x_{1:d})
\end{aligned}
$$

**Inverse:**
$$
\begin{aligned}
x_{1:d} &= y_{1:d} \\
x_{d+1:D} &= (y_{d+1:D} - t(y_{1:d})) \odot \exp(-s(y_{1:d}))
\end{aligned}
$$

Với $s$ (scale) và $t$ (translation) là neural networks, $\odot$ là element-wise multiplication.

### 1.2 Jacobian Determinant

**Triangular Jacobian:**
$$
J = \begin{bmatrix}
I_{d \times d} & 0 \\
\frac{\partial y_{d+1:D}}{\partial x_{1:d}} & \text{diag}(\exp(s(x_{1:d})))
\end{bmatrix}
$$

**Determinant:**
$$
\det(J) = \prod_{i=d+1}^D \exp(s(x_{1:d})_i) = \exp\left(\sum_{i=d+1}^D s(x_{1:d})_i\right)
$$

**Complexity:** $O(D)$ - linear trong dimension!

### 1.3 Masking Strategies

Different ways partition input:

**Spatial checkerboard:**
- Dùng cho images
- Alternate pixels như checkerboard pattern

**Channel-wise:**
- Split theo channel dimension
- Half channels identity, half transformed

**Alternating patterns:**
- Change partition scheme mỗi layer
- Ensure all dimensions được transform

## 2. Glow Architecture

### 2.1 Core Components

Glow improves Real NVP với 3 key operations mỗi "flow step":

1. **ActNorm** - Activation normalization
2. **Invertible 1x1 conv** - Learnable permutation
3. **Coupling layer** - Affine transformation

**Full flow:**
```
for scale in scales:
    for step in K_steps:
        x = actnorm(x)
        x = inv_1x1_conv(x)
        x = coupling_layer(x)
    x, z = split(x)  # Multi-scale architecture
```

### 2.2 Invertible 1x1 Convolutions

Thay fixed permutations, Glow learns them:

**Forward:**
$$
y = Wx
$$

với $W \in \mathbb{R}^{c \times c}$ là learnable invertible matrix.

**Jacobian:**
$$
\log|\det J| = h \cdot w \cdot \log|\det W|
$$

với $h, w$ là spatial dimensions.

**Efficient computation:**
- Parameterize via LU decomposition: $W = PLU$
- $P$ fixed, $L$ lower triangular, $U$ upper triangular
- $\det(W) = \det(L) \det(U) = \prod L_{ii} \prod U_{ii}$

### 2.3 ActNorm (Activation Normalization)

Alternative to batch normalization, works cho batch size = 1:

$$
y = s \odot (x - b)
$$

**Initialization:** Data-dependent để ensure activations have zero mean, unit variance initially.

**Benefits:**
- Independent of batch size
- Reversible với simple inverse
- Stable training

## 3. Implementation với PyTorch

### 3.1 Coupling Layer

```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        )
        self.nn[-1].weight.data.zero_()  # Initialize to identity
        self.nn[-1].bias.data.zero_()
    
    def forward(self, x, reverse=False):
        x_a, x_b = x.chunk(2, dim=1)  # Split along channel
        
        if not reverse:
            # Forward
            log_s, t = self.nn(x_a).chunk(2, dim=1)
            s = torch.sigmoid(log_s + 2)  # Ensure s > 0
            y_b = (x_b + t) * s
            log_det = torch.sum(torch.log(s).view(x.size(0), -1), dim=1)
            return torch.cat([x_a, y_b], dim=1), log_det
        else:
            # Reverse
            log_s, t = self.nn(x_a).chunk(2, dim=1)
            s = torch.sigmoid(log_s + 2)
            y_b = x_b / s - t
            return torch.cat([x_a, y_b], dim=1)
```

### 3.2 Invertible 1x1 Conv

```python
class InvConv2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # Initialize with random rotation matrix
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(w_shape))[0]
        
        # LU decomposition
        self.register_buffer("w_p", torch.eye(num_channels))
        self.w_l = nn.Parameter(torch.tril(w_init, -1))
        self.w_s = nn.Parameter(torch.diag(w_init).log())
        self.w_u = nn.Parameter(torch.triu(w_init, 1))
    
    def get_weight(self, reverse=False):
        w = (
            self.w_p
            @ (self.w_l + torch.eye(self.w_l.size(0), device=self.w_l.device))
            @ (torch.diag(self.w_s.exp()) + self.w_u)
        )
        
        if reverse:
            w = torch.inverse(w)
        
        return w
    
    def forward(self, x, reverse=False):
        weight = self.get_weight(reverse).view(
            *self.w_s.shape, 1, 1
        )
        
        if not reverse:
            z = F.conv2d(x, weight)
            log_det = self.w_s.sum() * x.size(2) * x.size(3)
            return z, log_det
        else:
            z = F.conv2d(x, weight)
            return z
```

### 3.3 ActNorm

```python
class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.initialized = False
    
    def initialize(self, x):
        with torch.no_grad():
            # Compute mean and std
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            
            self.log_scale.data.copy_(-torch.log(std + 1e-6))
            self.bias.data.copy_(-mean)
            
            self.initialized = True
    
    def forward(self, x, reverse=False):
        if not self.initialized:
            self.initialize(x)
        
        if not reverse:
            # Forward
            z = (x + self.bias) * torch.exp(self.log_scale)
            log_det = self.log_scale.sum() * x.size(2) * x.size(3)
            return z, log_det
        else:
            # Reverse
            z = x * torch.exp(-self.log_scale) - self.bias
            return z
```

### 3.4 Full Glow Block

```python
class GlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.inv_conv = InvConv2d(in_channels)
        self.coupling = AffineCoupling(in_channels, hidden_channels)
    
    def forward(self, x, reverse=False):
        if not reverse:
            # Forward pass
            x, log_det_actnorm = self.actnorm(x)
            x, log_det_conv = self.inv_conv(x)
            x, log_det_coupling = self.coupling(x)
            
            log_det = log_det_actnorm + log_det_conv + log_det_coupling
            return x, log_det
        else:
            # Reverse pass
            x = self.coupling(x, reverse=True)
            x = self.inv_conv(x, reverse=True)
            x = self.actnorm(x, reverse=True)
            return x
```

## 4. Multi-Scale Architecture

### 4.1 Concept

Thay vì transform toàn bộ latent ở cuối, "squeeze out" một phần latents ở multiple scales:

```python
class MultiScaleGlow(nn.Module):
    def __init__(self, in_channels=3, K=32, L=3):
        super().__init__()
        self.flows = nn.ModuleList()
        
        for level in range(L):
            flows_at_level = nn.ModuleList([
                GlowBlock(in_channels * 4) for _ in range(K)
            ])
            self.flows.append(flows_at_level)
            
            if level < L - 1:
                in_channels = in_channels * 2  # After split
    
    def squeeze(self, x):
        """Squeeze operation: H x W x C -> H/2 x W/2 x 4C"""
        b, c, h, w = x.size()
        x = x.view(b, c, h//2, 2, w//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c*4, h//2, w//2)
        return x
    
    def forward(self, x, reverse=False):
        if not reverse:
            log_det_total = 0
            z_list = []
            
            for level, flows in enumerate(self.flows):
                x = self.squeeze(x)
                
                for flow in flows:
                    x, log_det = flow(x)
                    log_det_total += log_det
                
                # Split
                if level < len(self.flows) - 1:
                    x, z = x.chunk(2, dim=1)
                    z_list.append(z)
            
            z_list.append(x)
            return z_list, log_det_total
        else:
            # Reverse: generate from z_list
            # Implementation here
            pass
```

## 5. Training

```python
def train_glow(model, dataloader, num_epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch[0]  # [B, C, H, W]
            
            # Forward pass
            z, log_det = model(x)
            
            # Compute negative log-likelihood
            log_pz = -0.5 * sum([torch.sum(zi**2) for zi in z])
            nll = -(log_pz + log_det) / (x.size(0) * x.size(1) * x.size(2) * x.size(3))
            
            # Bits per dimension
            bpd = nll / math.log(2)
            
            # Optimize
            optimizer.zero_grad()
            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

## 6. Applications

### 6.1 High-Quality Image Generation

Glow generates high-resolution images (256x256+) với:
- Sharp details
- Diverse samples
- Controllable generation

### 6.2 Latent Space Manipulation

Invertible nature allows:
- Semantic attribute editing
- Interpolation in latent space
- Attribute arithmetic

### 6.3 Exact Likelihood

Useful cho:
- Density estimation
- Anomaly detection
- Data compression

## Kết luận

Real NVP và Glow demonstrated power của carefully designed flow architectures:

✅ **Efficiency** - $O(D)$ Jacobian computation  
✅ **Expressivity** - Deep, multi-scale architectures  
✅ **Exact likelihoods** - No approximations  
✅ **Invertibility** - Perfect reconstruction

Though newer methods (Flow Matching, Rectified Flows) have emerged, principles từ Real NVP/Glow remain foundational cho flow-based modeling.

## Tài liệu tham khảo

1. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). "Density estimation using Real NVP" - ICLR
2. Kingma, D. P., & Dhariwal, P. (2018). "Glow: Generative Flow with Invertible 1x1 Convolutions" - NeurIPS
3. Papamakarios, G., et al. (2021). "Normalizing Flows for Probabilistic Modeling" - JMLR

---

**Tags:** #RealNVP #Glow #FlowModels #InvertibleNetworks #GenerativeAI #DeepLearning
