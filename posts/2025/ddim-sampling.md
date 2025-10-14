# DDIM Sampling: Fast Diffusion Generation

**Ngày đăng:** 20/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models

## Giới thiệu

DDIM enables **10-50x faster sampling** from diffusion models through deterministic sampling và skip-step strategies.

## Key Concepts

- **Non-Markovian forward process**
- **Deterministic reverse**
- **Accelerated sampling**

## Implementation

```python
def ddim_sample(model, steps=50, eta=0.0):
    x = torch.randn(shape)
    for t in reversed(range(0, 1000, 1000//steps)):
        eps = model(x, t)
        x = ddim_step(x, eps, t, eta)
    return x
```

## Tài liệu

- Song et al. "Denoising Diffusion Implicit Models" ICLR 2021

**Tags:** #DDIM #FastSampling #DiffusionModels
