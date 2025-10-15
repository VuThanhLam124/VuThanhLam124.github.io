---
title: "Đánh giá VLM: Benchmark, Metrics và quy trình phân tích lỗi"
date: "2025-04-01"
category: "vision-language-models"
tags: ["evaluation", "benchmark", "metrics", "vlm"]
excerpt: "Danh mục benchmark quan trọng cho VLM, hướng dẫn tính metrics và pipeline phân tích lỗi/hallucination."
author: "ThanhLamDev"
readingTime: 19
featured: false
---

# VLM Evaluation & Benchmarks

## 1. Nhóm benchmark chính

| Nhóm | Bộ dữ liệu | Năng lực đo lường |
|------|------------|------------------|
| VQA | VQAv2, VizWiz, TextVQA | hiểu ảnh + trả lời |
| Captioning | COCO, NoCaps | mô tả ngữ cảnh |
| Reasoning | NLVR2, Winoground | hiểu quan hệ phức tạp |
| Đa nhiệm | MMBench, SEED-Bench | tổng hợp |

## 2. Metrics phổ biến

- Accuracy cho VQA, Hit@k cho retrieval.
- BLEU, CIDEr, METEOR cho captioning.
- GPT-4V judge cho evaluation chủ quan.

```python
def evaluate_vqa(model, dataset):
    preds = model.generate(dataset.images, dataset.questions)
    return (preds == dataset.answers).mean()
```

## 3. Phân tích lỗi

- Phân loại theo motif: object, attribute, OCR, commonsense.
- Heatmap attention để phát hiện vùng bị bỏ qua.
- Manual review top 50 lỗi – ghi chú pattern.

## 4. Hallucination & safety

- Dùng CLIP score kiểm tra consistency image-caption.
- Safety filter: BLIP-based content moderation.
- Evaluate fairness: xem accuracy giữa nhóm demographic.

## 5. KPI cho production

- Latency 99th percentile < 2s.
- Memory footprint < 12GB (LoRA).
- Monitoring drift: log CLIP similarity theo thời gian.

## 6. Tài liệu

1. Liu et al. (2023). *SEED-Bench*.
2. Fu et al. (2023). *MMBench*.
3. Li et al. (2024). *HallusionBench*.

---

<script src="/assets/js/katex-init.js"></script>
