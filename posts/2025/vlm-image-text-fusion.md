---
title: "Image–Text Fusion Techniques: Kỹ thuật cross-attention và co-embedding"
date: "2025-04-04"
category: "vision-language-models"
tags: ["vlm", "fusion", "uniter", "albef", "coca", "cross-attention"]
excerpt: "Khảo sát các layer fusion nâng cao: UNITER, ALBEF, CoCa, Flamingo – cách kết hợp cross-attention và co-embedding để tăng khả năng reasoning."
author: "ThanhLamDev"
readingTime: 21
featured: false
---

# Image–Text Fusion Techniques

## 1. Fusion là gì?

- Sau khi có embedding ảnh & text, cần module kết hợp để reasoning.
- Hai hướng chính: **co-embedding** (joint transformer) và **cross-attention**.

## 2. UNITER & LXMERT

- Joint transformer stack: concat region features + word embeddings.
- Objective: MLM, MRM, ITM.
- Region features từ Faster R-CNN → fine-grained nhưng nặng.

## 3. ALBEF

- Image encoder ViT, text encoder BERT, cross-modal transformer nhỏ.
- Pretraining contrastive + ITM + MLM.
- Use momentum distillation -> stabilize training.

## 4. CoCa & Flamingo

- **CoCa**: shared transformer, first stage contrastive, second stage captioning.
- **Flamingo**: perceiver resampler + gated cross-attention, few-shot strong.

## 5. GFlowVLM overview

- Introduces flow matching for multimodal tokens, enabling multi-step reasoning.
- Use CLIP backbone + flow-based fusion module.

## 6. Chọn kỹ thuật fusion

| Yêu cầu | Gợi ý |
|---------|-------|
| Tốc độ, realtime | cross-attn nhẹ (ALBEF-style) |
| Fine-grained grounding | UNITER/LXMERT |
| Few-shot reasoning | Flamingo |
| Instruction chat | LLaVA-style (vision tokens → LLM) |

## 7. Tài liệu

1. Chen et al. (2020). *UNITER*.
2. Li et al. (2021). *ALBEF*.
3. Yu et al. (2022). *CoCa*.
4. Alayrac et al. (2022). *Flamingo*.
5. Geng et al. (2024). *GFlowVLM*.

---

<script src="/assets/js/katex-init.js"></script>
