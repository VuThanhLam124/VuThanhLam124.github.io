# Vision Encoder & Tokenization trong VLMs

**Ngày đăng:** 28/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** VLM, Computer Vision

## Giới thiệu

Vision encoders transform images thành representations mà language models có thể process.

## Architectures

### CLIP Vision Encoder
- ViT-based architecture
- Contrastive learning
- Resolution: 224x224 → 336x336

### Other Approaches
- CNN backbones (ResNet)
- Hybrid CNN-Transformer
- Adaptive pooling

## Tokenization Strategies

```python
class VisionTokenizer(nn.Module):
    def __init__(self, patch_size=14):
        super().__init__()
        self.encoder = ViT(patch_size=patch_size)
        self.projector = nn.Linear(vision_dim, llm_dim)
    
    def forward(self, images):
        # Extract patches
        patches = patchify(images, self.patch_size)
        # Encode
        features = self.encoder(patches)
        # Project to LLM space
        tokens = self.projector(features)
        return tokens
```

## Best Practices

1. **Resolution scaling** - higher resolution = better performance
2. **Aspect ratio preservation**
3. **Patch size selection**
4. **Feature pooling strategies**

## Tài liệu

- Radford et al. "Learning Transferable Visual Models" (CLIP)
- Dosovitskiy et al. "An Image is Worth 16x16 Words" (ViT)

**Tags:** #VisionEncoder #Tokenization #VLM

<script src="/assets/js/katex-init.js"></script>
