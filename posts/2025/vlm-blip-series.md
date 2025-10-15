---
title: "BLIP & BLIP-2: Cầu nối encoder hình ảnh và LLM"
date: "2025-03-27"
category: "vision-language-models"
tags: ["blip", "blip2", "q-former", "multimodal"]
excerpt: "Giải thích cách BLIP và BLIP-2 tách rời vision encoder, text encoder và LLM; phân tích Q-Former, captioning, VQA và các trick fine-tuning."
author: "ThanhLamDev"
readingTime: 21
featured: false
---

# BLIP & BLIP-2

## 1. BLIP recap

- Dựa trên ViT + BERT.
- Sử dụng tri-training: captioning, filtered captioning, image-text matching.
- Support dual-mode: zero-shot captioning & VQA.

## 2. BLIP-2 kiến trúc

- Vision encoder frozen (ViT-G/14 hoặc EVA).
- **Q-Former**: transformer 32 query token học được, cross-attn với image tokens.
- Projection sang LLM (Vicuna, OPT, Flan-T5).

## 3. Loss & training stages

1. Pretrain Q-Former với image-text pairs (ITM + captioning).
2. Align Q-Former output với LLM qua supervised fine-tuning.
3. Instruction tuning cho VQA/chat.

## 4. Q-Former chi tiết

```python
class QFormer(nn.Module):
    def __init__(self, num_query=32, dim=768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_query, dim))
        self.cross_attn = nn.TransformerEncoderLayer(dim, nhead=12)

    def forward(self, image_tokens):
        q = self.query.unsqueeze(0).expand(image_tokens.size(0), -1, -1)
        out = self.cross_attn(q, src_key_padding_mask=None, memory=image_tokens)
        return out
```

## 5. Khi nào dùng BLIP/BLIP-2?

| Nhu cầu | Gợi ý |
|---------|-------|
| Captioning chất lượng cao | BLIP fine-tune |
| Multimodal chat nhẹ | BLIP-2 + Vicuna 7B |
| Need open-source checkpoint | BLIP-2 Flan-T5 |

## 6. Tài liệu

1. Li et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training*.
2. Li et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*.

---

<script src="/assets/js/katex-init.js"></script>
