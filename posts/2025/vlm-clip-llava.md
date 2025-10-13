---
title: "Vision-Language Models: CLIP, LLaVA và beyond"
date: "2025-10-11"
category: "VLM"
tags: ["VLM", "CLIP", "LLaVA", "Multimodal", "Computer Vision"]
excerpt: "Khám phá kiến trúc và training strategies của VLMs hiện đại. Hands-on với multimodal fine-tuning và evaluation metrics cho practical applications."
author: "ThanhLamDev"
readingTime: 18
featured: true
---

# Vision-Language Models: CLIP, LLaVA và beyond

Thế hệ mô hình đa phương thức (VLM) mở ra khả năng hiểu ngôn ngữ và hình ảnh cùng lúc. Bài viết lấy CLIP và LLaVA làm ví dụ để phân tích kiến trúc, cơ chế học và phương pháp fine-tuning.

## 1. Kiến trúc tổng quan

- **CLIP**: huấn luyện contrastive giữa image encoder và text encoder, tối đa hóa cosine similarity cho cặp (image, caption) thật.
- **LLaVA**: kết hợp vision encoder (ViT) với language model (Vicuna) thông qua projection module, hỗ trợ conversational grounding.

## 2. Pipeline fine-tuning

1. Chuẩn hóa dữ liệu: caption hoặc conversation dạng ```<image> question -> answer```.
2. Freeze vision encoder, fine-tune language head bằng LoRA.
3. Sử dụng loss ```CrossEntropy``` trên token output, đôi khi kèm supervised contrastive.

## 3. Đánh giá và metrics

Cần đo trên cả nhiệm vụ zero-shot và instruction following:

- **Zero-shot classification**: ImageNet, CIFAR, Food101.
- **Captioning**: BLEU, CIDEr.
- **VQA**: Accuracy trên VQAv2, VizWiz.

## 4. Thực thi trong dự án

Trong thư mục ```code/vlm/``` có sẵn script:

- Fine-tuning CLIP bằng LoRA.
- Benchmark LLaVA trên bộ câu hỏi nội bộ.
- Utilities cho prompt engineering và evaluation.

## 5. Hướng mở rộng

- Kết hợp embedding của CLIP với retrieval augmented generation.
- Multimodal agents: nối thêm speech encoder hoặc audio input.
- Distillation sang mô hình nhẹ để deploy trên thiết bị biên.

Bài viết tiếp theo sẽ đào sâu pipeline training LLaVA với dataset custom và thiết lập inference phục vụ production API.
