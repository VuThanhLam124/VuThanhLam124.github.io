---
title: "Token Compression & Efficiency trong VLM"
date: "2025-03-30"
category: "vision-language-models"
tags: ["token-compression", "perceiver-resampler", "token-merging", "efficiency"]
excerpt: "Tối ưu số lượng token hình ảnh: Perceiver Resampler, Q-Former, Token Merging, EVA patch pruning và benchmark hiệu năng."
author: "ThanhLamDev"
readingTime: 19
featured: false
---

# Token Compression trong VLM

## 1. Động lực

- ViT-L/14 tạo 576 token → tốn 1.3× FLOPs so với text.
- Multi-image conversation càng nặng. Cần nén token mà giữ chất lượng.

## 2. Perceiver Resampler (Flamingo)

- Learnable latent array (64 token) cross-attn với image features.
- Linear complexity theo latent, không phụ thuộc patch count.

## 3. Q-Former (BLIP-2)

- Query token học được, 32–64 length.
- Dễ fine-tune, gắn kèm instruction.

## 4. Token Merging (ToMe) & EVA patch pruning

```python
def token_merge(features, ratio=0.5):
    metric = cosine_similarity(features)
    clusters = merge_pair(metric, ratio)
    return average_cluster(features, clusters)
```

- Merge patch giống nhau, giảm 30–50% token.
- EVA patch pruning: dùng mask learnable để bỏ patch ít quan trọng.

## 5. Đánh đổi chất lượng

| Phương pháp | Giảm token | Δ FID (caption) | Δ VQA acc | Ghi chú |
|-------------|------------|-----------------|-----------|---------|
| Perceiver | -80% | -0.5 | -0.8 | tốt cho few-shot |
| ToMe 0.5 | -50% | -1.2 | -1.5 | không cần retrain |
| Patch prune | -40% | -0.7 | -0.9 | cần fine-tuning lại |

## 6. Kết hợp thực tế

- LLaVA: apply ToMe ở inference để tăng throughput 1.6×.
- InternVL2: dùng Dynamic Resolution + Query Transformer.
- RAG multimodal: compress ảnh trước khi vector store.

## 7. Tài liệu

1. Alayrac et al. (2022). *Flamingo: Visual Language Models with Few-Shot Learning*.
2. Li et al. (2023). *BLIP-2*.
3. Bolya et al. (2023). *Token Merging: Your ViT but Faster*.
4. Sun et al. (2024). *EVA-CLIP*.

---

<script src="/assets/js/katex-init.js"></script>
