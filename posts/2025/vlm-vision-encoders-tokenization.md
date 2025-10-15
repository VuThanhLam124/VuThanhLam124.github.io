---
title: "Vision Encoders & Tokenization cho VLM: ViT, CNN và các thủ thuật patch"
date: "2025-03-23"
category: "vision-language-models"
tags: ["vision-encoder", "tokenization", "clip", "vit", "vlm"]
excerpt: "So sánh các kiến trúc vision encoder, chiến lược tokenization và dự án mã nguồn giúp đưa hình ảnh vào không gian ngôn ngữ."
author: "ThanhLamDev"
readingTime: 20
featured: false
---

# Vision Encoders & Tokenization

Anh họa sĩ của chúng ta bắt đầu bằng việc biến mỗi bức tranh thành chuỗi token mà ngôn ngữ hiểu được. Bài này phân tích kỹ ViT, CNN và các kỹ thuật chia patch.

## 1. ViT-based encoders

- **CLIP ViT-L/14**: patch size 14, positional embedding học.
- **EVA-CLIP**: ViT tinh chỉnh với masked image modeling, cải thiện zero-shot.
- **SigLIP**: thay InfoNCE bằng sigmoid loss, hỗ trợ scaling resolution.

## 2. CNN & hybrid

- ResNet pretrain trên ImageNet + projection.
- CoAtNet, ConvNeXt × Transformer – trade-off tốc độ và độ chính xác.

## 3. Tokenization workflows

```python
class PatchTokenizer(nn.Module):
    def __init__(self, encoder_ckpt, target_dim):
        super().__init__()
        self.encoder = timm.create_model(encoder_ckpt, pretrained=True)
        self.projector = nn.Linear(self.encoder.num_features, target_dim)

    def forward(self, images):
        feats = self.encoder.forward_features(images)
        tokens = self.projector(feats)
        return tokens  # [B, N_patches, target_dim]
```

## 4. Tips lựa chọn patch

| Patch size | Ưu điểm | Nhược điểm | Ứng dụng |
|------------|---------|-----------|----------|
| 32 | nhanh | mất chi tiết | quick prototyping |
| 16 | cân bằng | tốn FLOPs | VQA, captioning |
| 14/12 | fine detail | cần GPU mạnh | medical, OCR |

## 5. Feature pooling & resampler

- Mean pooling, CLS token, attention pooling.
- Perceiver Resampler (Flamingo), Q-Former (BLIP-2) – giảm số token gửi LLM.

## 6. Tài nguyên mã nguồn

- `code/vlm/tokenizer_demo.ipynb`: patchify + visualize attention map.
- `configs/vlm/tokenizer/clip_vit-L14.yaml`: config sẵn.

## 7. Tài liệu

1. Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words*.
2. Tu et al. (2023). *EVA: Exploring the Limits of Masked Pretraining*.
3. Alayrac et al. (2022). *Flamingo: Visual Language Models with Few-Shot Learning*.

---

<script src="/assets/js/katex-init.js"></script>
