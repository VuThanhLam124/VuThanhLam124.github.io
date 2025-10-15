---
title: "Pretraining Objectives cho VLM: Contrastive, Generative và Hybrid"
date: "2025-03-25"
category: "vision-language-models"
tags: ["pretraining", "contrastive", "captioning", "matching", "vlm"]
excerpt: "Phân loại các mục tiêu huấn luyện đa phương thức và so sánh ưu/nhược điểm của từng nhóm với minh họa toán học."
author: "ThanhLamDev"
readingTime: 17
featured: false
---

# Pretraining Objectives

Để anh họa sĩ “hiểu” cả tranh lẫn lời mô tả, chúng ta cần mục tiêu huấn luyện phù hợp. Bài này hệ thống các loss tiêu biểu.

## 1. Contrastive learning

$$
\mathcal{L}_{\text{clip}} = \frac{1}{2}(\mathcal{L}_{\text{img→txt}} + \mathcal{L}_{\text{txt→img}})
$$

- Ưu: dễ scale, zero-shot tốt.
- Nhược: thiếu thông tin cấu trúc câu dài.

## 2. Generative captioning

- Loss cross-entropy giữa caption dự đoán và caption thật.
- Dùng teacher forcing, kèm scheduled sampling để giảm exposure bias.

## 3. Matching & ranking

- Multi-choice VQA: dùng softmax trên tập candidate answer.
- Retrieval: optimize pairwise margin `max(0, m - sim_pos + sim_neg)`.

## 4. Hybrid frameworks

- **BLIP-2**: pretrain với captioning + contrastive, sau đó instruction tuning.
- **MiniGPT-4**: align projection layer bằng set question-answer trước khi fine-tune.
- **Qwen-VL**: mixture of denoising (prefix LM) và alignment loss trên region tags.

## 5. Chọn loss trong thực tiễn

| Kịch bản | Loss đề xuất | Lý do |
|---------|--------------|-------|
| Retrieval | Contrastive + ITC | tối ưu similarity |
| Captioning | XE + CIDEr optimization | tôn trọng ngữ pháp |
| VQA | Cross-entropy multi-class | phù hợp label |
| Multimodal chat | SFT + RLHF | cần style consistent |

## 6. Tài liệu

1. Li et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training*.
2. You et al. (2021). *Align and Prompt: VQA with Large Pretrained Models*.
3. Li et al. (2024). *Qwen-VL: A Versatile Vision-Language Model*.

---

<script src="/assets/js/katex-init.js"></script>
