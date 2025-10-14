# Elucidating Diffusion Models (EDM)

**Ngày đăng:** 25/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models

## Giới thiệu

EDM provides **unified analysis** và improved design choices for diffusion models.

## Key Insights

1. **Preconditioning** - proper input/output scaling
2. **Noise schedule** - optimal σ(t) selection
3. **Loss weighting** - importance sampling
4. **Sampling** - higher-order ODE solvers

## Preconditioning

```python
def preconditioned_denoiser(model, x, sigma):
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / sqrt(sigma**2 + sigma_data**2)
    c_in = 1 / sqrt(sigma_data**2 + sigma**2)
    
    F_x = model(c_in * x, sigma)
    D_x = c_skip * x + c_out * F_x
    return D_x
```

## Tài liệu

- Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models" NeurIPS 2022

**Tags:** #EDM #DiffusionDesign #OptimalDiffusion
