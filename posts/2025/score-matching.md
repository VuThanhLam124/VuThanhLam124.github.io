# Score Matching: Learning Data Distributions

**Ngày đăng:** 21/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models

## Giới thiệu

Score matching learns **gradient of log-density** without computing partition function.

## Theory

$$\nabla_x \log p(x) = s_\theta(x)$$

**Denoising score matching:**
$$\mathcal{L} = \mathbb{E}[\|s_\theta(x + \sigma \epsilon) - (-\epsilon/\sigma)\|^2]$$

## Implementation

```python
def score_matching_loss(model, x, sigma=0.1):
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score_pred = model(x_noisy)
    score_target = -noise / sigma
    return ((score_pred - score_target) ** 2).mean()
```

## Tài liệu

- Song & Ermon "Generative Modeling by Estimating Gradients" NeurIPS 2019

**Tags:** #ScoreMatching #ScoreBasedModels
