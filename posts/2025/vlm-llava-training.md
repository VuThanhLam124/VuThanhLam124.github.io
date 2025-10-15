---
title: "Huấn luyện LLaVA: Từ data chuẩn hóa đến inference chat đa hình"
date: "2025-03-28"
category: "vision-language-models"
tags: ["llava", "instruction-tuning", "conversation", "multimodal"]
excerpt: "Walkthrough chi tiết pipeline LLaVA: chuẩn hoá dữ liệu, projection layer, LoRA finetuning và deploy inference server."
author: "ThanhLamDev"
readingTime: 22
featured: false
---

# LLaVA Training Pipeline

## 1. Kiến trúc tổng quan

- Vision encoder: CLIP ViT-L/14 frozen.
- Projection MLP 2 lớp đưa embedding 1024 → 4096.
- Language backbone: Vicuna-7B/13B.

## 2. Chuẩn bị dữ liệu

- Format: `<image>\nUSER: câu hỏi\nASSISTANT: câu trả lời`.
- Dùng dataset LLaVA-Instruct-150K, LAION-Chat, hoặc tự annotate.
- Tiền xử lý: resize 336x336, convert caption special tokens `<IMAGE>`/`</IMAGE>`.

## 3. Fine-tuning với LoRA

```python
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05
)
model = get_peft_model(vicuna_model, lora_config)
```

- Freeze vision encoder + projection.
- Sử dụng mixed precision bf16, gradient checkpointing.

## 4. Loss & hyperparameters

- Cross-entropy trên token output.
- lr = 2e-4, warmup 500 step, batch size hiệu dụng 128, 3 epoch.
- Gradient clipping 1.0, apply label smoothing 0.1.

## 5. Đánh giá

- VQAv2 accuracy, LLaVA-bench, MMBench.
- Manual evaluation: coherence, hallucination.

## 6. Deploy inference

- Serve CLIP encoder + Vicuna với FastAPI, stream token.
- Dùng `torch.compile` (triton) hoặc FlashAttention-2 để tăng tốc.
- Cache vision features để trả lời nhiều câu hỏi cho cùng ảnh.

## 7. Tài nguyên

- `code/vlm/llava_train.py`: script huấn luyện.
- `docker/llava/Dockerfile`: môi trường inference GPU A100.

## 8. Paper tham khảo

1. Liu et al. (2023). *Visual Instruction Tuning*.
2. Xu et al. (2023). *LLaVA: Large Language and Vision Assistant*.

---

<script src="/assets/js/katex-init.js"></script>
