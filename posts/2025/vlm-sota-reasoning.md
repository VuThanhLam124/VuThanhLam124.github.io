---
title: "SOTA Multimodal Reasoning: VLM thế hệ mới"
date: "2025-04-06"
category: "vision-language-models"
tags: ["vlm", "flamingo", "gpt-4v", "gflowvlm", "reasoning"]
excerpt: "Tổng hợp Flamingo, GPT-4V, GFlowVLM và các hướng reasoning đa bước: memory, tool-use, multi-image context."
author: "ThanhLamDev"
readingTime: 22
featured: true
---

# SOTA Multimodal Reasoning

## 1. Flamingo

- Few-shot strong: perceiver resampler + gated cross-attention.
- Supports multi-image context; memory tokens.
- Limitation: closed weights, heavy compute.

## 2. GPT-4V / Gemini

- Native multimodal: share token space cho text + image.
- RLHF đa vòng, safety guardrail.
- Capabilities: chart reasoning, instruction follow, limited transparency.

## 3. GFlowVLM

- Sử dụng flow matching cho fusion, iterative refinement.
- Reasoning multi-step: output intermediate thought chain.
- Open-source 7B/13B checkpoints.

## 4. Xu hướng reasoning

| Hướng | Ví dụ | Thách thức |
|-------|-------|------------|
| Tool use | ReAct + VLM | cần planner tốt |
| World grounding | RT-2, VIMA | data chế tạo tốn kém |
| Memory module | Flamingo, LongVLM | kiểm soát drift |
| Video reasoning | Video-LLaVA | compute lớn |

## 5. Đánh giá reasoning

- Benchmarks: MMMU, ScienceQA, MathVista, ChartQA.
- Use GPT-4V judge + human verification.

## 6. Tài liệu

1. Alayrac et al. (2022). *Flamingo*.
2. OpenAI (2023). *GPT-4 Technical Report*.
3. Geng et al. (2024). *GFlowVLM*.
4. Liu et al. (2024). *MMMU*.

---

<script src="/assets/js/katex-init.js"></script>
