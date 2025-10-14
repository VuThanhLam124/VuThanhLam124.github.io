# Diffusion Transformers (DiT)

**Ngày đăng:** 24/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Diffusion Models, Transformers

## Giới thiệu

DiT replaces U-Net backbone với **Vision Transformer** architecture trong diffusion models.

## Architecture

- Patchify image
- Transformer blocks với adaptive normalization
- Unpatchify to image

## Implementation

```python
class DiTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = MultiHeadAttention(dim)
        self.mlp = MLP(dim)
        self.adaLN = AdaptiveLayerNorm(dim)
    
    def forward(self, x, t_emb):
        scale, shift = self.adaLN(t_emb)
        x = x + self.attn(self.norm(x, scale, shift))
        x = x + self.mlp(self.norm(x, scale, shift))
        return x
```

## Tài liệu

- Peebles & Xie "Scalable Diffusion Models with Transformers" ICCV 2023

**Tags:** #DiffusionTransformers #DiT #ScalableModels

<script src="/assets/js/katex-init.js"></script>
