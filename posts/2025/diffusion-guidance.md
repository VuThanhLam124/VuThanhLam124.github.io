# Classifier-Free Diffusion Guidance

**Ngày đăng:** 22/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models

## Giới thiệu

Classifier-free guidance enables **controlled generation** without separate classifier.

## Method

Train model conditionally và unconditionally:

$$\tilde{\epsilon}_\theta = \epsilon_\theta(x,\emptyset) + w \cdot (\epsilon_\theta(x,c) - \epsilon_\theta(x,\emptyset))$$

## Implementation

```python
def guided_diffusion_step(model, x, t, condition, w=7.5):
    # Unconditional prediction
    eps_uncond = model(x, t, condition=None)
    # Conditional prediction
    eps_cond = model(x, t, condition=condition)
    # Guided prediction
    eps = eps_uncond + w * (eps_cond - eps_uncond)
    return denoise_step(x, eps, t)
```

## Tài liệu

- Ho & Salimans "Classifier-Free Diffusion Guidance" NeurIPS 2021 Workshop

**Tags:** #GuidedDiffusion #ConditionalGeneration

<script src="/assets/js/katex-init.js"></script>
