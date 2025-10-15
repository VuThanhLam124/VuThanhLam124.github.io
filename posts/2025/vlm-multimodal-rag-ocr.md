---
title: "Ứng dụng VLM: Multimodal Retrieval, OCR & RAG"
date: "2025-04-02"
category: "vision-language-models"
tags: ["vlm", "retrieval", "ocr", "rag", "applications"]
excerpt: "Hướng dẫn xây dựng hệ thống RAG đa phương thức, xử lý OCR với VLM và tối ưu pipeline tìm kiếm hình ảnh."
author: "ThanhLamDev"
readingTime: 20
featured: false
---

# Ứng dụng VLM: Retrieval & OCR

## 1. Retrieval đa phương thức

- Sử dụng CLIP/BLIP embedding → vector store (FAISS, Milvus).
- Hỗ trợ search ngược: text → image, image → text.
- Ranking: combine CLIP score + caption score.

## 2. OCR & document understanding

- Dùng Donut, DocVQA dataset để fine-tune BLIP-2.
- Kết hợp layout embedding (LayoutLMv3) với vision encoder.
- Pipeline: detect → OCR (Tesseract/TrOCR) → fuse context → VLM answer.

## 3. Multimodal RAG (Retrieval-Augmented Generation)

```python
def multimodal_rag(query, images):
    image_keys = clip_index.search(images, topk=5)
    text_keys = text_index.search(query, topk=5)
    context = build_prompt(image_keys + text_keys)
    return llava.generate(query, context)
```

- Cache vision tokens cho ảnh được truy hồi.
- Sử dụng caption tóm tắt + bounding box info.

## 4. Kiến trúc hệ thống

- Client upload ảnh → feature extractor.
- Retrieval service trả top-K hình + mô tả.
- LLM trả lời/giải thích, log output cho feedback loop.

## 5. Case study

- E-commerce: tìm sản phẩm tương tự từ ảnh sản phẩm.
- Hỗ trợ nội bộ: tra cứu sơ đồ, tài liệu scan.
- Trợ lý báo cáo: chèn hình trong báo cáo → VLM sinh mô tả + con số.

## 6. Tài liệu

1. Kim et al. (2022). *Donut: Document Understanding Transformer*.
2. Huang et al. (2023). *BLIP-2 for Document VQA*.
3. Shen et al. (2024). *MM-RAG: Retrieval-Augmented Generation for Multi-Modal Tasks*.

---

<script src="/assets/js/katex-init.js"></script>
