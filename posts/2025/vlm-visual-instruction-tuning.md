---
title: "Visual Instruction Tuning: Huấn luyện theo hướng dẫn trực quan"
date: "2025-04-02"
category: "vision-language-models"
tags: ["vlm", "instruction-tuning", "llava", "instructblip", "alignment"]
excerpt: "Từ LLaVA đến InstructBLIP: pipeline xử lý dữ liệu, SFT, RLHF và mẹo giảm hallucination cho VLM follow-instruction hiệu quả."
author: "ThanhLamDev"
readingTime: 22
featured: false
---

# Visual Instruction Tuning

## 1. Dữ liệu hướng dẫn

- Format: `<image>\nUSER: ...\nASSISTANT: ...`.
- Nguồn: LLaVA-Instruct, MiniGPT-4 conversations, tự gắn nhãn.
- Cleaning: remove NSFW, duplicate; label difficulty (easy/medium/hard).

## 2. Huấn luyện SFT

```python
loss = cross_entropy(logits[:, :-1], target[:, 1:], ignore_index=pad_id)
```

- Freeze vision encoder, fine-tune projection + LLM (LoRA).
- Add system prompt: “You are a helpful visual assistant...”.

## 3. InstructBLIP vs LLaVA

- InstructBLIP: Q-Former + Flan-T5, strong caption grounding.
- LLaVA: CLIP + Vicuna, conversation style.
- Mix data: caption, QA, reasoning, OCR.

## 4. RLHF / DPO

- Collect preference pairs (good vs bad answer).
- Train reward model (visual + text).
- Optimize bằng DPO để tránh instabilities.

## 5. Giảm hallucination

- Vision consistency check (CLIP score).
- Force cite bounding box info khi possible.
- Penalize answers “I think” / “maybe” qua regex filter.

## 6. Evaluation

- LLaVA-Bench, SEED-Bench, MMBench (instruct subset).
- Human eval 5-point scale, categorize failure cases.

## 7. Tài liệu

1. Liu et al. (2023). *Visual Instruction Tuning*.
2. Ye et al. (2023). *InstructBLIP*.
3. Rafailov et al. (2023). *Direct Preference Optimization*.

---

<script src="/assets/js/katex-init.js"></script>
