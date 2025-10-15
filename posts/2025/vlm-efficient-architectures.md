---
title: "Efficient Architectures: Giảm kích thước, tăng tốc và distillation"
date: "2025-03-31"
category: "vision-language-models"
tags: ["vlm", "efficiency", "distillation", "tinyvlm", "pruning"]
excerpt: "Tổng hợp các kỹ thuật xây VLM nhẹ: TinyVLM, DistilCLIP, knowledge distillation, pruning và quantization cho triển khai thực tế."
author: "ThanhLamDev"
readingTime: 20
featured: false
---

# Efficient Architectures

## 1. Động lực

- VLM gốc nặng 7B–80B → khó deploy edge.
- Nhu cầu: chatbot trên mobile, AR glasses, search real-time.

## 2. Distillation

- **DistilCLIP**: teacher CLIP ViT-L/14 → student ResNet50.
- Hard & soft targets: combine InfoNCE + MSE embedding loss.
- **MiniCPM-V**: distill LLaVA vào backbone 1.8B.

## 3. TinyVLM/TinyLLaVA

- Reduce hidden dim, số layer; apply LoRA for quick adaptation.
- Use low-rank adapter + shared projection weights.

## 4. Pruning & quantization

- Structured pruning vision encoder (remove channels).
- Token pruning (EVA patch mask) + quantization INT8/bf16.
- Quantization-aware training cho LLM head.

## 5. Benchmark chi phí

| Model | Params | VRAM (8bit) | Speed-up | ΔAcc |
|-------|--------|-------------|----------|------|
| TinyVLM 1.3B | 1.3B | 6GB | 2.1× | -4.2 |
| DistilCLIP | 150M | 2GB | 2.8× | -3.5 |
| LLaVA-1.5 7B Int8 | 7B | 11GB | 1.4× | -1.3 |

## 6. Tips triển khai

- Cache vision features; share across multi-turn.
- Sử dụng vLLM/FastChat để stream response.
- Benchmark latency 99p, throughput req/s trước khi production.

## 7. Tài liệu

1. Zhang et al. (2023). *TinyViT*.
2. Wu et al. (2023). *MiniGPT-4/TinyLLaVA*.
3. Goh et al. (2022). *DistilCLIP*.

---

<script src="/assets/js/katex-init.js"></script>
