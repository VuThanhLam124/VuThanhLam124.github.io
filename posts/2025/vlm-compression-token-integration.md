---
title: "Compression & Token Integration: Nén ảnh trực tiếp vào token text"
date: "2025-04-08"
category: "vision-language-models"
tags: ["vlm", "compression", "vq-gan", "neural-codec", "seaformer"]
excerpt: "Khám phá xu hướng hợp nhất ảnh vào chuỗi token: Sea-Former, VQ-GAN, neural codec và pipeline embedding liền mạch."
author: "ThanhLamDev"
readingTime: 21
featured: false
---

# Compression & Token Integration

## 1. Tại sao cần token integration?

- LLM xử lý chuỗi token → nén ảnh thành codebook/token tiết kiệm chi phí.
- Cho phép reuse hạ tầng LLM (attention, KV cache).

## 2. VQ-GAN & discrete tokens

- Encode ảnh thành mã $z$ (32x32) với codebook kích thước 1024.
- Loss: reconstruction + perceptual + adversarial.
- Token hóa: flatten $z$ → sequence dài 1024 tokens → LLM xử lý.

## 3. Sea-Former & neural codec

- Sea-Former: self-encoding attention, token compress ratio 16×.
- Neural codec (LiT-Codec) convert ảnh thành 64 tokens 16-bit.
- Combine với LLM qua prefix/prompt.

## 4. Integration pipeline

```python
codes = vqgan.encode(image)          # [B, L]
tokens = tokenizer.map_code(codes)   # map to textual vocab
prompt = "<IMG_TOK> " + " ".join(tokens) + " <IMG_END>"
answer = llm.generate(prompt, question)
```

## 5. Thách thức

- Mất chi tiết → cần decoder cao cấp.
- Token dài → tăng chi phí context.
- Chưa giải quyết alignment fine-grained (cần cross-attn bổ sung).

## 6. Ứng dụng

- Streaming video token (DALL-E 3 pipeline).
- Embedding unify image/text cho RAG store.
- Privacy: compress + encrypt token.

## 7. Tài liệu

1. Esser et al. (2021). *VQ-GAN*.
2. Yu et al. (2023). *Sea-Former*.
3. Yang et al. (2024). *LiT-Codec*.

---

<script src="/assets/js/katex-init.js"></script>
