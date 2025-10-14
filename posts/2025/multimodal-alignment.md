# Multimodal Alignment: Connecting Vision & Language

**Ngày đăng:** 29/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** VLM, Multimodal Learning

## Giới thiệu

Multimodal alignment ensures vision và language representations **semantically aligned** trong shared embedding space.

## Alignment Methods

### 1. Contrastive Learning
$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_j \exp(\text{sim}(v_i, t_j)/\tau)}$$

### 2. Cross-Modal Attention
- Vision tokens attend to language
- Language attends to vision features
- Bidirectional alignment

### 3. Projection Layers
```python
class AlignmentModule(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.cross_attn = CrossAttention(shared_dim)
    
    def forward(self, vision_feats, text_feats):
        v = self.vision_proj(vision_feats)
        t = self.text_proj(text_feats)
        aligned_v, aligned_t = self.cross_attn(v, t)
        return aligned_v, aligned_t
```

## Training Objectives

1. **Image-Text Matching** (ITM)
2. **Masked Language Modeling** (MLM)
3. **Image-Text Contrastive** (ITC)

## Tài liệu

- Li et al. "BLIP" ICML 2022
- Alayrac et al. "Flamingo" NeurIPS 2022

**Tags:** #MultimodalAlignment #VisionLanguage #VLM

<script src="/assets/js/katex-init.js"></script>
