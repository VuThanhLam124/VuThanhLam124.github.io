---
title: "Multimodal Alignment: Đồng bộ hóa không gian thị giác và ngôn ngữ"
date: "2025-03-27"
category: "vision-language-models"
tags: ["vlm", "contrastive-learning", "clip", "align", "alignment"]
excerpt: "Phân tích sâu cơ chế alignment giữa ảnh và text: contrastive learning của CLIP/ALIGN, sampling chiến lược và mẹo prompt."
author: "ThanhLamDev"
readingTime: 20
featured: false
---

# Multimodal Alignment

## 1. Bài toán alignment

- Mục tiêu: map feature ảnh $v$ và text $t$ vào cùng không gian, tăng sim với cặp thật, giảm sim với cặp giả.
- Định nghĩa distance bằng cosine, normalize vector để ổn định training.

## 2. Contrastive learning (CLIP, ALIGN)

$$
\mathcal{L}_{\text{contrastive}} = -\frac{1}{N} \sum_i \left[\log \frac{\exp(v_i^\top t_i / \tau)}{\sum_j \exp(v_i^\top t_j / \tau)} + \log \frac{\exp(v_i^\top t_i / \tau)}{\sum_j \exp(v_j^\top t_i / \tau)} \right].
$$

- Temperature $\tau$ trainable, giúp phân tách distribution.
- Batch large (>= 4096) tăng negative coverage.
- ALIGN sử dụng noisy web data + logistic loss (sigmoid).

## 3. Kỹ thuật sampling

- Mix web data + curated dataset để giảm bias.
- Caption template augmentation để đa dạng context.
- Hard negative mining: lấy caption gần nhau theo embedding.

## 4. Prompt engineering cho zero-shot

```python
def build_prompts(label, templates):
    return [template.format(label) for template in templates]
```

- Sử dụng 50+ template, ví dụ: “a photo of a {}.”, “{} in a museum.”.
- Ensemble text embeddings bằng mean để tăng accuracy.

## 5. Evaluation alignment

- Zero-shot accuracy trên ImageNet, CIFAR, Food101.
- Retrieval recall@K, image-text, text-image.
- CLIP score để đánh giá consistency image-caption.

## 6. Hạn chế & hướng cải thiện

- Noise trong web captions → cần filtering (BLIP caption, language detection).
- Language coverage: ALIGN sử dụng multilingual text encoder (mBERT).
- Region-level alignment: sau CLIP, cần layer cross-attn (ALBEF) để chi tiết hơn.

## 7. Tài liệu

1. Radford et al. (2021). *CLIP*.
2. Jia et al. (2021). *ALIGN*.
3. Cherti et al. (2023). *OpenCLIP*.

---

<script src="/assets/js/katex-init.js"></script>
