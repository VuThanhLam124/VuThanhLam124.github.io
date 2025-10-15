---
title: "Multimodal Alignment: Đồng bộ hóa không gian thị giác và ngôn ngữ"
date: "2025-03-27"
category: "vision-language-models"
tags: ["vlm", "alignment", "clip", "align", "contrastive-learning"]
excerpt: "Câu chuyện tiếp nối của hướng dẫn viên bảo tàng: sau khi đã token hóa bức tranh, cô cần đồng bộ hóa chuỗi hình ảnh với câu chữ bằng contrastive learning (CLIP, ALIGN) và kỹ thuật calibration thực tế."
author: "ThanhLamDev"
readingTime: 24
featured: false
---

# Multimodal Alignment: Đồng bộ hóa không gian thị giác và ngôn ngữ

**Hành trình tại Bảo tàng Giao Thoa tiếp tục. Sau khi chuyển mỗi bức tranh thành chuỗi token (bài 1), cô hướng dẫn viên phải đảm bảo rằng “ghi chú hình ảnh” và “câu chữ” luôn khớp nhau. Nếu không, khách khiếm thính và khách suy giảm thị lực sẽ hiểu sai lệch. Bài viết này kể cách cô dùng kỹ thuật Alignment – tương đương với contrastive learning trong VLM – để đồng bộ hóa hai miền thị giác và ngôn ngữ.**

## Mục lục

1. [Tình huống thực tế tại bảo tàng](#1-tình-huống-thực-tế-tại-bảo-tàng)
2. [Khái niệm alignment và InfoNCE loss](#2-khái-niệm-alignment-và-infonce-loss)
3. [Toán học chi tiết](#3-toán-học-chi-tiết)
4. [Chiến lược sampling và xử lý dữ liệu](#4-chiến-lược-sampling-và-xử-lý-dữ-liệu)
5. [Pipeline huấn luyện CLIP/ALIGN thực tế](#5-pipeline-huấn-luyện-clipalign-thực-tế)
6. [Ví dụ PyTorch: contrastive training loop](#6-ví-dụ-pytorch-contrastive-training-loop)
7. [Đo lường, calibration và xử lý sai lệch](#7-đo-lường-calibration-và-xử-lý-sai-lệch)
8. [Liên kết với các bài tiếp theo](#8-liên-kết-với-các-bài-tiếp-theo)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Tình huống thực tế tại bảo tàng

Sau buổi hướng dẫn đầu tiên, cô nhận ra một vấn đề: khi mô tả “bình gốm xanh bên cửa sổ”, hệ thống đôi lúc ghép nhầm ảnh với câu chữ vì dữ liệu huấn luyện chưa đủ đồng bộ. Để cả hai nhóm khách hiểu đúng, cô phải dạy hệ thống nhận ra rằng hình ảnh của bình gốm phải đi kèm câu mô tả tương ứng và ngược lại. Đây chính là **alignment** – học một không gian chung nơi vector ảnh và vector văn bản của cùng một nội dung nằm gần nhau.

## 2. Khái niệm alignment và InfoNCE loss

Điểm cốt lõi của CLIP/ALIGN là **contrastive learning**: kéo cặp (ảnh, mô tả) thật lại gần, đẩy các cặp sai ra xa. Cho batch $\{(x_i, y_i)\}_{i=1}^N$ với $x_i$ là ảnh, $y_i$ là văn bản. Ta encode thành $v_i = f_{	ext{img}}(x_i)$ và $t_i = g_{	ext{text}}(y_i)$ rồi chuẩn hóa để $\|v_i\|=\|t_i\|=1$. Loss InfoNCE dạng đối xứng:

$$
\mathcal{L}_{\text{clip}} = \frac{1}{2}\left(\mathcal{L}_{\text{img}\rightarrow\text{text}} + \mathcal{L}_{\text{text}\rightarrow\text{img}}\right),
$$
trong đó
$$
\mathcal{L}_{\text{img}\rightarrow\text{text}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(v_i^\top t_i / \tau)}{\sum_{j=1}^N \exp(v_i^\top t_j / \tau)}.
$$

$\tau$ là hệ số nhiệt độ (temperature) được học, giúp điều chỉnh độ sắc nét của phân phối.

## 3. Toán học chi tiết

### 3.1 Chuẩn hóa embedding

$$
\hat{v}_i = \frac{v_i}{\|v_i\|_2}, \quad \hat{t}_i = \frac{t_i}{\|t_i\|_2}.
$$
Điều này biến cosine similarity thành tích vô hướng $\hat{v}_i^\top \hat{t}_j$ – dễ tính và ổn định.

### 3.2 Gradient insight

Gradient của loss theo $v_i$ khi xét hướng ảnh → text:
$$
\frac{\partial \mathcal{L}}{\partial v_i} = \frac{1}{\tau}\left( \sum_{j=1}^N p_{i,j} t_j - t_i \right),
$$
trong đó $p_{i,j} = \frac{\exp(v_i^\top t_j / \tau)}{\sum_{k} \exp(v_i^\top t_k / \tau)}$. Nghĩa là vector ảnh được kéo về mô tả đúng và đẩy xa các mô tả sai theo xác suất softmax.

### 3.3 Liên hệ xác suất

Loss InfoNCE tương đương tối đa hóa lower bound của Mutual Information $I(X;Y)$. Đây là lý do alignment giúp hệ thống “hiểu” mối quan hệ giữa ảnh và câu chữ tốt hơn.

## 4. Chiến lược sampling và xử lý dữ liệu

### 4.1 Từ trải nghiệm hướng dẫn

Cô ghi lại mỗi câu thuyết minh cùng ảnh chụp góc tương ứng. Để tránh lệch, cô chuẩn hóa caption bằng template thống nhất (ví dụ “A photo of {object} in {context}”).

### 4.2 Negative sampling

- **In-batch negatives:** dùng caption của các ảnh khác trong batch.
- **Hard negatives:** chọn câu có embedding gần nhưng sai (ví dụ caption “bình gốm đỏ” cho ảnh bình gốm xanh).
- **Multilingual augmentation:** dịch caption sang nhiều ngôn ngữ để tăng tính đa dạng.

### 4.3 Data cleaning

- Loại bỏ caption quá ngắn (<3 từ) hoặc chứa ký tự lỗi.
- Dùng model OCR để chắc chắn caption thực sự mô tả ảnh (quan trọng với tranh có chữ).

## 5. Pipeline huấn luyện CLIP/ALIGN thực tế

```
Ảnh & Caption -> DataLoader -> ViT Encoder + Text Encoder -> Normalize -> Similarity Matrix -> InfoNCE Loss -> Optimizer -> Logging (ImageNet zero-shot, VQA recall)
```

Các bước chính:

1. **Encoder**: ViT-L/14 (ảnh), Transformer 12-layer (text).
2. **Optimizer**: AdamW, learning rate 1e-5, weight decay 0.2.
3. **Batch size**: 4096 (sử dụng gradient accumulation nếu GPU nhỏ).
4. **Temperature**: khởi tạo $\tau=0.07$, để trainable.
5. **Evaluation định kỳ**: zero-shot trên ImageNet, recall@K trên Flickr30k.

## 6. Ví dụ PyTorch: contrastive training loop

```python
import torch
import torch.nn.functional as F

def contrastive_step(model, batch, optimizer, tau):
    images, texts = batch
    v = model.encode_image(images)        # [B, d]
    t = model.encode_text(texts)          # [B, d]

    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)

    logits = v @ t.T / tau
    labels = torch.arange(len(images), device=images.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_i + loss_t) / 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

**Chú thích:**
- `logits = v @ t.T / tau` tạo ma trận similarity $N\times N$.
- Tính cross-entropy hai chiều để đối xử công bằng ảnh → text và text → ảnh.

## 7. Đo lường, calibration và xử lý sai lệch

- **Zero-shot accuracy**: kiểm tra nhanh bằng cách dùng caption template cho ImageNet.
- **Recall@K cho retrieval**: đảm bảo ảnh và câu đúng nằm trong top K.
- **CLIP score consistency**: dùng CLIP score để phát hiện caption lệch.
- **Calibration**: nếu phát hiện bias (ví dụ nhầm bình gốm xanh thành đỏ), tăng sampling hard negatives hoặc thêm prompt mô tả màu.
- **Human-in-the-loop**: giống cô hướng dẫn viên, định kỳ nghe phản hồi từ khách để điều chỉnh dataset (ví dụ thêm góc chụp mới, sửa caption lỗi).

## 8. Liên kết với các bài tiếp theo

Sau khi bảo đảm “ảnh nào đi với lời nào”, cô hướng dẫn viên quan tâm tới việc mô hình có thể xử lý nhiều nhiệm vụ cùng lúc (captioning, QA, matching). Bài 3 sẽ giới thiệu **pretraining objectives hybrid** – giúp mô hình vừa giỏi retrieval, vừa giỏi sinh mô tả. Bài 4 sẽ bàn về **efficient architectures** để hệ thống chạy nhanh trong tour đông khách.

## 9. Tài liệu tham khảo

1. Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision (CLIP).* 
2. Jia et al. (2021). *Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN).* 
3. Cherti et al. (2023). *Revisiting Contrastive Methods for CLIP Training at Scale (OpenCLIP).* 
4. Van den Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding.*

---

<script src="/assets/js/katex-init.js"></script>
