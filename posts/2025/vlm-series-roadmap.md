---
title: "VLM Roadmap: Từ nền tảng đến triển khai thực tế"
date: "2025-03-21"
category: "vision-language-models"
tags: ["vlm", "multimodal", "roadmap", "overview"]
excerpt: "Bức tranh toàn cảnh về series Vision-Language Models: cấu trúc 4 tầng, liên kết tới từng bài chi tiết, mục tiêu học tập và cách khai thác tài nguyên mã nguồn."
author: "ThanhLamDev"
readingTime: 12
featured: true
---

# VLM Roadmap

Series Vision-Language Models (VLM) được thiết kế thành **bốn tầng**: nền tảng, huấn luyện, tối ưu nâng cao và xu hướng mới. Mỗi tầng gồm các bài đào sâu ở từng góc độ từ lý thuyết đến mã nguồn thực chiến. Bài viết này cung cấp sơ đồ tổng thể giúp bạn định vị kiến thức và lựa chọn lộ trình phù hợp.

## 1. Mục tiêu học tập

- Hiểu kiến trúc chung của VLM: encoder hình ảnh, encoder/ngôn ngữ, cầu nối đa phương thức.
- Nắm vững mục tiêu pretraining (contrastive, captioning, matching) và các biến thể mới (instruction tuning).
- Thực chiến fine-tuning với LoRA, quản lý dữ liệu hội thoại đa hình.
- Theo dõi công trình SOTA 2024–2025, đánh giá chất lượng, tối ưu triển khai.

## 2. Dòng thời gian 14 bài

| Giai đoạn | Bài viết | Điểm nhấn |
|-----------|----------|-----------|
| **2015–2019: Khởi nguyên** | 1. *Khởi nguyên VLM (Show-and-Tell → ViLBERT)*<br>2. *Multimodal Fundamentals* | Captioning, VQA đầu tiên, transformer đa phương thức |
| **2020–2021: Contrastive Revolution** | 3. *Vision Encoders & Tokenization*<br>4. *Text Encoders & Grounding*<br>5. *Pretraining Objectives*<br>6. *CLIP Deep Dive* | ViT, BERT, InfoNCE, bài học từ CLIP/ALIGN |
| **2022–2023: Cầu nối tới LLM** | 7. *BLIP & BLIP-2*<br>8. *Token Compression & Efficiency*<br>9. *LLaVA Training Pipeline*<br>10. *Instruction Tuning & Alignment* | Q-Former, Perceiver Resampler, instruction tuning đa hình |
| **2023–2024: Fine-tuning & Đánh giá** | 11. *Fine-tuning với LoRA*<br>12. *Evaluation & Benchmarks* | Toolkit PEFT, benchmark toàn diện |
| **2024–2025: Ứng dụng & SOTA** | 13. *Multimodal Retrieval, OCR & RAG*<br>14. *VLM SOTA 2025 & Trends* | Ứng dụng đặc thù, Gemini/GPT-4V/InternVL2 |

## 3. Hành trình đề xuất

1. **Bước 1** – Đọc 3 bài nền tảng để hiểu cách hình ảnh được đưa vào mô hình ngôn ngữ.
2. **Bước 2** – Lựa chọn kiến trúc phù hợp (CLIP, BLIP-2, LLaVA) dựa trên bài toán.
3. **Bước 3** – Áp dụng các chiến thuật fine-tuning, đánh giá bằng tiêu chuẩn ngành.
4. **Bước 4** – Thử nghiệm ứng dụng đặc thù (OCR, RAG) và triển khai tối ưu.

## 4. Liên kết mã nguồn

- `code/vlm/` chứa script fine-tuning, LoRA adapters, evaluation pipeline.
- Notebook demo đặt tại `notebooks/vlm/` (sẽ cập nhật kèm từng bài).
- Dataset chuẩn hóa trong `data/vlm/metadata.json` (đang xây dựng).

## 5. Câu hỏi thường gặp

- **Tôi nên bắt đầu từ đâu nếu chỉ quen NLP?** – Đọc bài 2 & 3 để hiểu vision encoders và kết nối với LLM.
- **Tôi muốn triển khai nhanh?** – Bắt đầu với bài 8 (LLaVA) + bài 11 (LoRA Toolkit).
- **Tôi quan tâm SOTA?** – Xem bài 14 về Gemini/GPT-4V, InternVL2, và hướng nghiên cứu mới.

## 6. Lộ trình cập nhật

Các bài viết sẽ được cập nhật khi xuất hiện paper mới hoặc thư viện mở nguồn quan trọng. Bạn có thể theo dõi changelog tại `docs/changelog.md` (khởi tạo trong tương lai gần).

---

<script src="/assets/js/katex-init.js"></script>
