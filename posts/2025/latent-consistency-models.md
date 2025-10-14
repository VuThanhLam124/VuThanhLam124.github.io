# Latent Consistency Models (LCM)

**Ngày đăng:** 26/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models

## Giới thiệu

LCM extends consistency models to **latent space**, enabling fast high-resolution generation.

## Method

Apply consistency distillation trong latent space của Stable Diffusion:

1. Start với pretrained latent diffusion
2. Distill to consistency model
3. Generate với 2-4 steps

## Implementation

```python
class LatentConsistencyModel(nn.Module):
    def __init__(self, vae, unet):
        super().__init__()
        self.vae = vae
        self.unet = unet  # Consistency-trained
    
    def generate(self, z_T, num_steps=4):
        z = z_T
        for t in linspace(T, 0, num_steps):
            z = self.unet(z, t)
        img = self.vae.decode(z)
        return img
```

## Tài liệu

- Luo et al. "Latent Consistency Models" 2023

**Tags:** #LCM #FastGeneration #LatentDiffusion
