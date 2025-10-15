---
title: "VLM SOTA 2025: Gemini, GPT-4V, InternVL2 và những hướng nghiên cứu mới"
date: "2025-04-03"
category: "vision-language-models"
tags: ["sota", "gemini", "gpt-4v", "internvl", "roadmap"]
excerpt: "Tổng hợp các mô hình VLM mạnh nhất 2024–2025, phân tích kỹ thuật nổi bật và dự báo xu hướng nghiên cứu."
author: "ThanhLamDev"
readingTime: 23
featured: true
---

# VLM SOTA 2025

## 1. Gemini 1.5 / Gemini Ultra

- Multimodal natively: text, image, video, audio.
- Dynamic token routing, mixture-of-experts.
- Performance: top MMBench, MME, VideoMM.

## 2. GPT-4V

- Vision encoder proprietary, align qua multi-round RLHF.
- Strong reasoning, OCR, chart understanding.
- Limitations: closed-source, latency cao.

## 3. InternVL2

- Multi-resolution vision encoder + Query Transformer.
- Token dynamic pooling, achieve SOTA open-source.
- Released weights 8B/20B, support multi-image conversation.

## 4. Qwen-VL Max & Kosmos-2

- Qwen-VL: support 20+ languages, high performance on OCR.
- Kosmos-2: grounding với bounding box output.

## 5. Xu hướng nghiên cứu

1. **World grounding**: embodied AI, robotics (VIMA, RT-2).
2. **Video & 3D**: LLaVA-NeXT-Video, Dreamer-V.
3. **Long-context multimodal**: streaming tokens, memory module.
4. **Data governance**: chất lượng dataset, privacy.

## 6. Chiến lược cho kỹ sư

- Kết hợp open-source (InternVL2) với API (Gemini/GPT-4V) → hybrid.
- Đầu tư pipeline alignment & evaluation (bài 9 + 12).
- Theo dõi arXiv: `cs.CV`, `cs.CL` tag “multimodal”, “vision-language”.

## 7. Tài liệu

1. Team Gemini (2024). *Gemini 1.5 Technical Report*.
2. OpenAI (2023). *GPT-4 Technical Report*.
3. Chen et al. (2024). *InternVL2*.
4. Bai et al. (2024). *Qwen-VL: A Versatile Vision-Language Model*.

---

<script src="/assets/js/katex-init.js"></script>
