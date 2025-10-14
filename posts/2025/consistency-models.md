# Consistency Models: One-Step Generation

**Ngày đăng:** 23/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models

## Giới thiệu

Consistency models enable **one-step generation** from noise to data.

## Core Idea

Learn consistency function $f$ where:
$$f(x_t, t) = f(x_{t'}, t') = x_0 \quad \forall t, t'$$

## Training

```python
def consistency_training_loss(model, x):
    t, t_next = sample_timesteps()
    x_t = add_noise(x, t)
    x_t_next = add_noise(x, t_next)
    
    # Self-consistency
    pred_t = model(x_t, t)
    pred_t_next = model(x_t_next, t_next)
    
    return ((pred_t - pred_t_next.detach()) ** 2).mean()
```

## Tài liệu

- Song et al. "Consistency Models" ICML 2023

**Tags:** #ConsistencyModels #OneStepGeneration
