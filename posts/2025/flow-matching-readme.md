# Flow Matching Implementation

Complete PyTorch implementation của Flow Matching cho generative modeling.

## Tổng quan

Flow Matching là một phương pháp generative modeling tiên tiến, học vector field để transport từ noise distribution tới data distribution thông qua optimal transport paths.

## Cấu trúc Files

```
flow-matching/
├── flow_matching_pytorch.py    # Main implementation
├── requirements.txt            # Dependencies
├── README.md                  # This file
├── examples/                  # Example scripts
│   ├── toy_2d_example.py     # 2D toy datasets
│   ├── image_generation.py   # Image generation example  
│   └── text_generation.py    # Text generation example
└── utils/
    ├── visualization.py       # Plotting utilities
    ├── datasets.py           # Dataset utilities
    └── metrics.py            # Evaluation metrics
```

## Installation

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn
```

## Quick Start

### Bước 1: Import và Setup

```python
import torch
from flow_matching_pytorch import FlowMatching, create_toy_dataset
from torch.utils.data import TensorDataset, DataLoader

# Create toy 2D dataset
data = create_toy_dataset('moons', n_samples=2000)

# Setup data loader
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
```

### Bước 2: Initialize và Train Model

```python
# Initialize Flow Matching
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fm = FlowMatching(dim=2, device=device)

# Train model
losses = fm.train(dataloader, num_epochs=500, log_interval=100)
```

### Bước 3: Generate Samples

```python
# Generate new samples
samples = fm.sample(num_samples=1000, num_steps=100, method='euler')

# Or use adaptive integration for higher quality
samples_adaptive = fm.sample(num_samples=1000, method='adaptive')
```

## Key Features

### 1. Flexible Vector Field Architecture
- Multi-layer perceptron với time embedding
- LayerNorm và Dropout cho training stability  
- Customizable hidden dimensions và số layers

### 2. Multiple Sampling Methods
- **Euler Method**: Fast, fixed step size
- **Adaptive Method**: Higher quality, variable step size using scipy's ODE solver

### 3. Training Features
- AdamW optimizer với learning rate scheduling
- Gradient clipping cho stability
- Checkpoint saving/loading
- Comprehensive logging

### 4. Evaluation Tools
- Likelihood computation using change of variables
- 2D flow field visualization
- Training loss monitoring

## Advanced Usage

### Custom Vector Field Architecture

```python
class CustomVectorField(nn.Module):
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        # Your custom architecture here
        
    def forward(self, x, t):
        # Your forward pass
        return vector_field

# Use custom architecture
fm = FlowMatching(dim=2)
fm.model = CustomVectorField(dim=2).to(device)
```

### Conditional Generation

```python
class ConditionalVectorField(nn.Module):
    def __init__(self, dim, condition_dim, hidden_dim=256):
        super().__init__()
        self.time_embed = nn.Linear(1, 64)
        self.condition_embed = nn.Linear(condition_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(dim + 64 + 64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t, condition):
        t_embed = self.time_embed(t.unsqueeze(-1))
        c_embed = self.condition_embed(condition)
        input_vec = torch.cat([x, t_embed, c_embed], dim=-1)
        return self.net(input_vec)
```

### High-Dimensional Data

```python
# For image data
class ImageFlowMatching(FlowMatching):
    def __init__(self, channels=3, height=32, width=32):
        dim = channels * height * width
        super().__init__(dim)
        self.shape = (channels, height, width)
    
    def sample_images(self, num_samples):
        samples_flat = self.sample(num_samples)
        return samples_flat.view(num_samples, *self.shape)
```

## Ưu điểm so với Other Methods

| Method | Training Speed | Sampling Speed | Quality | Stability |
|--------|----------------|----------------|---------|-----------|
| GANs | Fast | Very Fast | High | Low |
| VAEs | Fast | Very Fast | Medium | High |
| DDPM | Slow | Very Slow | Very High | High |
| **Flow Matching** | **Fast** | **Fast** | **High** | **Very High** |

## Applications

### 1. 2D Toy Data
```python
# Generate different 2D distributions
datasets = ['moons', 'circles', 'blobs']
for dataset_type in datasets:
    data = create_toy_dataset(dataset_type, 1000)
    # Train and sample...
```

### 2. Image Generation
```python
# CIFAR-10 example
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

fm_images = FlowMatching(dim=3*32*32)
losses = fm_images.train(dataloader, num_epochs=1000)
```

### 3. Molecule Generation
```python
# For molecular graphs (simplified)
class MoleculeFlowMatching(FlowMatching):
    def __init__(self, max_atoms=20, atom_features=10):
        dim = max_atoms * atom_features
        super().__init__(dim)
    
    def generate_molecules(self, num_samples):
        samples = self.sample(num_samples)
        return self.decode_molecules(samples)
```

## Hyperparameter Tuning

### Training Parameters
```python
# Learning rate scheduling
fm.setup_optimizer(lr=1e-3, weight_decay=1e-5)

# Training with different batch sizes
for batch_size in [64, 128, 256, 512]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = fm.train(dataloader, num_epochs=100)
```

### Architecture Parameters
```python
# Test different architectures
architectures = [
    {'hidden_dim': 128, 'num_layers': 3},
    {'hidden_dim': 256, 'num_layers': 4}, 
    {'hidden_dim': 512, 'num_layers': 5}
]

for config in architectures:
    fm = FlowMatching(dim=2)
    fm.model = VectorField(dim=2, **config)
    # Train and evaluate...
```

## Evaluation Metrics

### Quantitative Metrics
```python
# FID score for images
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(feature=2048)
real_images = ... # Your real images
fake_images = fm.sample_images(1000)

fid.update(real_images, real=True)
fid.update(fake_images, real=False)
fid_score = fid.compute()
```

### Qualitative Assessment
```python
# Visualize 2D flow field
visualize_2d_flow(fm.model, device, grid_size=50)

# Plot samples vs real data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.6, label='Real')
plt.subplot(1, 2, 2) 
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, label='Generated')
plt.show()
```

## Troubleshooting

### Common Issues

1. **Training Instability**
   - Giảm learning rate
   - Tăng gradient clipping
   - Thêm regularization

2. **Poor Sample Quality**
   - Tăng số training epochs
   - Sử dụng adaptive sampling
   - Tăng model capacity

3. **Slow Sampling**
   - Giảm num_steps trong Euler method
   - Sử dụng pre-trained model
   - Optimize với TorchScript

### Performance Tips

```python
# Compile model for faster training (PyTorch 2.0+)
fm.model = torch.compile(fm.model)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = fm.flow_matching_loss(x0, x1, t)
    
scaler.scale(loss).backward()
scaler.step(fm.optimizer)
scaler.update()
```

## Research Extensions

### 1. Rectified Flow
- Implement straight-line paths
- Reduce NFE (Number of Function Evaluations)

### 2. Conditional Flow Matching  
- Add class conditioning
- Text-to-image generation

### 3. Consistency Models
- Distill Flow Matching into faster sampling
- Single-step generation

---

## Citation

Nếu bạn sử dụng code này trong nghiên cứu, please cite:

```bibtex
@misc{flowmatching2025,
  title={Flow Matching Implementation in PyTorch},
  author={ThanhLamDev},
  year={2025},
  url={https://github.com/VuThanhLam124/flow-matching-pytorch}
}
```

## License

MIT License - feel free to use cho cả academic và commercial purposes.

## Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Add tests cho new features  
4. Submit pull request

---

**Contact:** vuthanhlam848@gmail.com  
**GitHub:** https://github.com/VuThanhLam124
<script src="/assets/js/katex-init.js"></script>
