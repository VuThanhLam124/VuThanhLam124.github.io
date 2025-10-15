---
title: "CLIP Deep Dive: Kiến trúc, huấn luyện và bài học thực tế"
date: "2025-03-26"
category: "vision-language-models"
tags: ["clip", "contrastive-learning", "openai", "multimodal"]
excerpt: "Giải phẫu CLIP: kiến trúc đôi tower, loss InfoNCE, tricks huấn luyện và cách fine-tune CLIP làm nền cho dự án VLM."
author: "ThanhLamDev"
readingTime: 20
featured: false
---

# CLIP Deep Dive

## 1. Kiến trúc đôi tower

- Vision encoder: ViT-B/L/16/14 hoặc ResNet-50/101.
- Text encoder: Transformer 12 lớp, vocab BPE 49k.
- Projection linear để đưa cả hai vào không gian 512 chiều chung.

## 2. Training pipeline

- Dataset: 400M (image, text) web-scale.
- Optimizer: AdamW, lr warmup 200 steps, cosine decay.
- Augment: random resize crop, color jitter, random caption template.

## 3. Loss InfoNCE

$$
\mathcal{L} = \mathcal{L}_{\text{img→txt}} + \mathcal{L}_{\text{txt→img}}
$$

Với

$$
\mathcal{L}_{\text{img→txt}} = -\log \frac{\exp(v_i^\top t_i / \tau)}{\sum_j \exp(v_i^\top t_j/\tau)}
$$

## 4. Fine-tuning chiến thuật

| Quy mô dữ liệu | Chiến lược | Ghi chú |
|---------------|-----------|---------|
| <10k | Freeze vision, tune text + projector | nhanh |
| 10k-100k | LoRA trên cả hai tower | cân bằng |
| >100k | full fine-tune | cần GPU nhiều |

```python
def clip_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora = LoRALinear(module, rank=rank)
            setattr(model, name, lora)
    return model
```

## 5. Ứng dụng thực tế

- Zero-shot classification: chỉ cần prompt template.
- Visual search: index embedding, cosine similarity.
- Base cho LLaVA/BLIP-2: sử dụng image features từ CLIP ViT-L/14.

## 6. Lưu ý khi triển khai

- Chuẩn hóa input size, mean/std = (0.48145466, 0.4578275, 0.40821073).
- Sử dụng `torch.cuda.amp` cho inference nhanh.
- Cache text embeddings của prompt để giảm latency.

## 7. Tài liệu

1. Radford et al. (2021). *Learning Transferable Visual Models with Natural Language Supervision*.
2. Ilharco et al. (2021). *OpenCLIP*.

---

<script src="/assets/js/katex-init.js"></script>
