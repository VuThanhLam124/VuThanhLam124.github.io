---
title: "Từ Show-and-Tell đến ViLBERT: Khởi nguyên Vision-Language Models"
date: "2025-03-21"
category: "vision-language-models"
tags: ["history", "vilbert", "captioning", "multimodal"]
excerpt: "Ôn lại các mốc quan trọng giai đoạn 2015–2019: image captioning, attention, co-attention transformers – nền móng cho VLM hiện đại."
author: "ThanhLamDev"
readingTime: 17
featured: false
---

# Khởi nguyên VLM (2015–2019)

Trước khi CLIP xuất hiện, cộng đồng đã xây dựng những viên gạch đầu tiên cho mô hình nhìn-hiểu nói. Bài này tóm lược các công trình tiêu biểu giúp hình thành VLM.

## 1. Image Captioning thế hệ đầu

- **Show and Tell (2015)**: CNN encoder + LSTM decoder.
- **Show, Attend and Tell (2015)**: thêm attention vùng.
- **NIC, m-RNN**: tăng cường beam search, dataset MSCOCO.

## 2. Visual Question Answering ban đầu

- **VQA v1 (2015)**: kết hợp CNN + LSTM với simple fusion.
- **Stacked Attention Networks (2016)**: multi-hop attention.

## 3. Transformers đa phương thức

- **ViLBERT (2019)**: hai stream BERT, co-attention.
- **VisualBERT, UNITER**: joint embedding + masked LM.
- **LXMERT**: pretrain trên multi-task (VQA, caption).

## 4. Kết luận giai đoạn

- Từ captioning sang pretrain multi-task → chuẩn bị cho datasets web-scale.
- Hạn chế: data nhỏ (COCO, VG), architecture nặng khó scale.

## 5. Tài liệu

1. Vinyals et al. (2015). *Show and Tell*.
2. Xu et al. (2015). *Show, Attend and Tell*.
3. Lu et al. (2019). *ViLBERT*.
4. Chen et al. (2020). *UNITER*.

---

<script src="/assets/js/katex-init.js"></script>
