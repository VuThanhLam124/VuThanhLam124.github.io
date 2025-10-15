---
title: "Multimodal Fundamentals: Nền tảng lý thuyết cho Vision-Language Models"
date: "2025-03-22"
category: "vision-language-models"
tags: ["multimodal-learning", "representation", "theory", "vlm"]
excerpt: "Phân tích khung toán học của học đa phương thức, từ biểu diễn liên kết đến cơ chế chú ý chéo – bước đệm cho các kiến trúc VLM hiện đại."
author: "ThanhLamDev"
readingTime: 18
featured: false
---

# Multimodal Fundamentals

Trước khi đi vào các kiến trúc cụ thể, cần hiểu lý thuyết học đa phương thức: cách biểu diễn, đồng bộ hóa và kết hợp tín hiệu vision + language.

## 1. Biểu diễn đa phương thức

- **Early fusion**: nối trực tiếp vector hình ảnh – văn bản, thiệt hại về linh hoạt.
- **Late fusion**: xử lý riêng lẻ rồi kết hợp điểm tương đồng (CLIP sử dụng).
- **Joint embedding**: học space chung qua contrastive hoặc matching loss.

## 2. Toán học align representation

Cho embedding $v_i$ từ ảnh, $t_i$ từ caption. Mục tiêu contrastive:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_j \exp(\text{sim}(v_i, t_j)/\tau)}.
$$

**Chú thích:** $\text{sim}$ thường là cosine; $\tau$ là temperature.

## 3. Attention chéo (Cross-Attention)

- Query từ text, key/value từ image tokens (BLIP-2, LLaVA).
- Điều chỉnh bằng gating hoặc adapter để tránh catastrophic forgetting.

## 4. Challenges thường gặp

1. **Modality gap**: chênh lệch phân phối feature – giải pháp: projection layer, adapter.
2. **Scaling law khác nhau**: image encoder cần nhiều FLOPs; text encoder mở rộng bằng prefix.
3. **Data bias**: caption noisy, imbalance region; cần filtering & re-weighting.

## 5. Lời khuyên thực tế

- Dùng checkpoint encoder đã pretrain thay vì huấn luyện từ đầu.
- Sử dụng mixed precision (bf16) để giảm tiêu tốn khi joint training.
- Tận dụng dataset web-scale (LAION, CC3M) nhưng phải apply caption filter.

## 6. Tài liệu

1. Baltrusaitis et al. (2019). *Multimodal Machine Learning: A Survey and Taxonomy*.
2. Tsai et al. (2019). *Learning Factorized Multimodal Representations*.

---

<script src="/assets/js/katex-init.js"></script>
