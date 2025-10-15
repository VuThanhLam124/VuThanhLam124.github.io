---
title: "Pretraining Objectives: Mô hình học đa nhiệm và hybrid objectives"
date: "2025-03-29"
category: "vision-language-models"
tags: ["vlm", "pretraining", "contrastive", "generative", "masked-modeling"]
excerpt: "So sánh contrastive, generative và masked modeling trong pretraining VLM; cách kết hợp hybrid để cân bằng retrieval và reasoning."
author: "ThanhLamDev"
readingTime: 19
featured: false
---

# Pretraining Objectives

## 1. Contrastive-only

- CLIP, ALIGN, SigLIP: mạnh về retrieval, zero-shot classification.
- Loss InfoNCE hoặc sigmoid binary loss.

## 2. Generative captioning

- BLIP, VinVL: predict caption given image.
- Loss cross-entropy, teacher forcing, sequence length 30–40 tokens.
- Beam search + CIDEr optimization (SCST) cải thiện caption metrics.

## 3. Masked image & language modeling

- UNITER, VinVL: mask region features, mask tokens.
- Loss: $L = L_{\text{MLM}} + L_{\text{MRM}} + L_{\text{ITM}}$.
- Yêu cầu region features (Faster R-CNN) → nặng.

## 4. Hybrid objectives

| Model | Loss | Ưu điểm | Ghi chú |
|-------|------|---------|---------|
| BLIP-2 | caption + ITM + contrastive | linh hoạt | Q-Former bridging |
| CoCa | contrastive + captioning chung backbone | gọn nhẹ | 1 stage training |
| PaLI-X | mixture of caption, translation, OCR | multi-language | require TPU cluster |

## 5. Công thức hybrid ví dụ (CoCa)

$$
L_{\text{CoCa}} = \lambda_{\text{cap}} L_{\text{cap}} + \lambda_{\text{contrast}} L_{\text{contrast}}.
$$

- Balance $\lambda$ để giữ cả retrieval và generative quality.

## 6. Thực tiễn chọn objective

- Nếu mục tiêu zero-shot → ưu tiên contrastive.
- Nếu caption/chat → kết hợp captioning + alignment.
- Nếu OCR/multi-lingual → multi-task translation + bridging tokens.

## 7. Tài liệu

1. Li et al. (2022). *BLIP*.
2. Yu et al. (2022). *CoCa*.
3. Chen et al. (2023). *PaLI-X*.

---

<script src="/assets/js/katex-init.js"></script>
