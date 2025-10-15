---
title: "Text Encoders & Language Grounding trong VLM"
date: "2025-03-24"
category: "vision-language-models"
tags: ["text-encoder", "bert", "llm", "grounding", "vlm"]
excerpt: "Bóc tách vai trò encoder ngôn ngữ, cách kết nối với vision features và các chiến lược grounding câu hỏi-hình ảnh."
author: "ThanhLamDev"
readingTime: 18
featured: false
---

# Text Encoders & Grounding

Nếu phần nhìn đã sẵn sàng, thì “tai nghe – miệng nói” của VLM chính là text encoder. Bài này phân tích từ BERT đến LLM và cú pháp grounding.

## 1. Lựa chọn text encoder

- **BERT/RoBERTa**: nhẹ, tối ưu cho contrastive pretraining (CLIP text tower).
- **T5/FLAN**: sequence-to-sequence phù hợp captioning.
- **Vicuna/Qwen**: LLM mạnh mẽ cho instruction tuning (LLaVA, Qwen-VL).

## 2. Prompt normalization

```python
def format_prompt(text, template="A photo of {}."):
    if "{}" in template:
        return template.format(text)
    return template + " " + text
```

- Template quan trọng trong CLIP zero-shot.
- Với LLM, dùng system prompt để cố định phong cách trả lời.

## 3. Grounding cơ bản

| Phương pháp | Ý tưởng | Ví dụ |
|-------------|--------|-------|
| Contrastive | maximize sim(image,text) | CLIP |
| Captioning | predict text given image | BLIP |
| QA | condition on question + image tokens | LLaVA |

## 4. Alignment kỹ thuật

- Projection linear: `W_proj` map từ dimension vision sang language.
- Adapter cross-attn: Query từ text, K/V từ vision tokens.
- Prefix/prompt tuning: thêm token ảo biểu diễn thông tin hình ảnh.

## 5. Tài liệu

1. Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision*.
2. Li et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*.
3. Xu et al. (2023). *LLaVA: Large Language and Vision Assistant*.

---

<script src="/assets/js/katex-init.js"></script>
