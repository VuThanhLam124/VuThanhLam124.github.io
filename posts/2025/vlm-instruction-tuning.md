---
title: "Instruction Tuning & Alignment cho VLM"
date: "2025-03-29"
category: "vision-language-models"
tags: ["instruction-tuning", "alignment", "rlhf", "multimodal"]
excerpt: "Chiến thuật align VLM với yêu cầu người dùng: dữ liệu đối thoại, SFT, RLHF, evaluator và safety filter."
author: "ThanhLamDev"
readingTime: 21
featured: false
---

# Instruction Tuning cho VLM

## 1. Dữ liệu đối thoại đa phương thức

- Dạng conversation: `<image>` + multi-turn QA.
- Phải cân bằng domain: lifestyle, science, charts.
- Gắn thẻ difficulty để curriculum learning.

## 2. Supervised Fine-Tuning (SFT)

- Loss cross-entropy với teacher forcing.
- Chèn token `<IMG_CONTEXT>` để đánh dấu range vision features.
- Giữ dropout cao để tránh overfit (0.3).

## 3. Preference & RLHF

- Thu thập pairwise feedback (A better than B).
- Train reward model $R_\phi$ dự đoán mức độ hữu ích.
- Optimize bằng PPO hoặc DPO (Direct Preference Optimization).

## 4. Safety alignment

- Classifier detect NSFW, violence; combine CLIP score + textual filter.
- Sử dụng policy guideline (OpenAI style) -> apply rejection sampling.

## 5. Evaluation alignment

- Automatic: GPT-4V judge, CLEVER-FEWS shot.
- Human-in-the-loop: 5-scale rating, annotation UI.

## 6. Best practices

- Bắt đầu SFT với 80% data, giữ 20% cho RLHF.
- Track hallucination rate (percentage answers “invented”).
- Triển khai guardrail pipeline (vision moderation + text moderation).

## 7. Tài liệu

1. Liu et al. (2024). *LLaVA 1.5/NeXT*.
2. Cui et al. (2023). *MiniGPT-4: Enhancing Vision-Language Understanding with Advanced LMs*.
3. Rafailov et al. (2023). *Direct Preference Optimization*.

---

<script src="/assets/js/katex-init.js"></script>
