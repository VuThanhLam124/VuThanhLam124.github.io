---
title: "Multimodal Alignment: Đồng bộ hóa không gian thị giác và ngôn ngữ"
date: "2025-03-27"
### 3.3 Liên hệ xác suất

Loss InfoNCE tương đương tối đa hóa lower bound của Mutual Information $I(X;Y)$. Đây là lý do alignment giúp hệ thống "hiểu" mối quan hệ giữa ảnh và câu chữ tốt hơn.

**Chứng minh trực quan:**

InfoNCE loss có dạng:

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{f(x, y^+)}{\sum_{i=1}^K f(x, y_i)}\right]
$$

trong đó $y^+$ là positive pair, $\{y_i\}$ là tập K negative samples. Khi $K \to \infty$:

$$
\mathcal{L}_{\text{InfoNCE}} \geq -I(X;Y) + \log K
$$

Do đó minimize loss $\Rightarrow$ maximize mutual information giữa ảnh và text.

### 3.4 Temperature scaling: vai trò của $\tau$

Temperature $\tau$ điều chỉnh "độ tự tin" của model:

- **$\tau$ nhỏ** (0.01-0.05): softmax sắc nét, model phải rất chắc chắn về cặp đúng
- **$\tau$ lớn** (0.1-0.5): softmax mượt hơn, model học "từ từ" hơn

Trong CLIP, $\tau$ được khởi tạo ở 0.07 và được học cùng với các tham số khác:

$$
\tau = \text{clamp}(\exp(\log \tau_0), \tau_{\min}, \tau_{\max})
$$

**Ví dụ số:**

Giả sử có 3 caption với similarity scores: $[0.9, 0.3, 0.2]$

- Với $\tau = 0.1$: softmax $\approx [0.996, 0.002, 0.002]$ (rất tự tin)
- Với $\tau = 0.5$: softmax $\approx [0.65, 0.20, 0.15]$ (ít tự tin hơn)egory: "vision-language-models"
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

Điểm cốt lõi của CLIP/ALIGN là **contrastive learning**: kéo cặp (ảnh, mô tả) thật lại gần, đẩy các cặp sai ra xa. Cho batch với $N$ cặp:

$$
\{(x_i, y_i)\}_{i=1}^N
$$

trong đó $x_i$ là ảnh, $y_i$ là văn bản. Ta encode thành:

$$
v_i = f_{\text{img}}(x_i), \quad t_i = g_{\text{text}}(y_i)
$$

rồi chuẩn hóa để đảm bảo:

$$
\lVert v_i \rVert = \lVert t_i \rVert = 1
$$

**Loss InfoNCE dạng đối xứng:**

$$
\mathcal{L}_{\text{clip}} = \frac{1}{2}\left(\mathcal{L}_{\text{img}\rightarrow\text{text}} + \mathcal{L}_{\text{text}\rightarrow\text{img}}\right)
$$

trong đó:

$$
\mathcal{L}_{\text{img}\rightarrow\text{text}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(v_i^\top t_i / \tau)}{\sum_{j=1}^N \exp(v_i^\top t_j / \tau)}
$$

**Giải thích các thành phần:**

- $\tau$ là hệ số nhiệt độ (temperature) được học, giúp điều chỉnh độ sắc nét của phân phối
- Tử số: độ tương đồng giữa ảnh $i$ với caption đúng $t_i$
- Mẫu số: tổng độ tương đồng với tất cả caption trong batch (bao gồm cả đúng và sai)
- Loss này khuyến khích embedding ảnh $v_i$ gần với caption đúng $t_i$ hơn so với các caption sai $t_j$ ($j \neq i$)

## 3. Toán học chi tiết

### 3.1 Chuẩn hóa embedding

$$
\hat{v}_i = \frac{v_i}{\lVert v_i \rVert_2}, \quad \hat{t}_i = \frac{t_i}{\lVert t_i \rVert_2}
$$

**Tại sao chuẩn hóa quan trọng?**

Điều này biến cosine similarity thành tích vô hướng $\hat{v}_i^\top \hat{t}_j$ – dễ tính và ổn định. Cụ thể:

$$
\text{cosine\_sim}(v_i, t_j) = \frac{v_i^\top t_j}{\lVert v_i \rVert \cdot \lVert t_j \rVert} = \hat{v}_i^\top \hat{t}_j
$$

Sau khi chuẩn hóa, similarity score nằm trong khoảng $[-1, 1]$, giúp training ổn định hơn.

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

### 6.1 Implementation cơ bản

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, images):
        features = self.vision_encoder(images)
        return F.normalize(features, dim=-1)
    
    def encode_text(self, texts):
        features = self.text_encoder(texts)
        return F.normalize(features, dim=-1)
    
    def forward(self, images, texts):
        image_features = self.encode_image(images)   # [B, d]
        text_features = self.encode_text(texts)       # [B, d]
        
        # Scaled similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T  # [B, B]
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text

def contrastive_loss(logits_per_image, logits_per_text):
    """
    Compute symmetric contrastive loss
    
    Args:
        logits_per_image: [B, B] similarity matrix from image perspective
        logits_per_text: [B, B] similarity matrix from text perspective
    
    Returns:
        loss: scalar contrastive loss
    """
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # Image-to-text loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    
    # Text-to-image loss
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    
    # Symmetric loss
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss

def training_step(model, batch, optimizer):
    images, texts = batch
    
    # Forward pass
    logits_per_image, logits_per_text = model(images, texts)
    
    # Compute loss
    loss = contrastive_loss(logits_per_image, logits_per_text)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Log metrics
    with torch.no_grad():
        acc_i2t = (logits_per_image.argmax(dim=1) == torch.arange(len(images), device=images.device)).float().mean()
        acc_t2i = (logits_per_text.argmax(dim=1) == torch.arange(len(texts), device=texts.device)).float().mean()
    
    return {
        'loss': loss.item(),
        'acc_i2t': acc_i2t.item(),
        'acc_t2i': acc_t2i.item(),
        'temperature': model.logit_scale.exp().item()
    }
```

**Chú thích chi tiết:**

- `logits = v @ t.T / tau` tạo ma trận similarity $N\times N$ - mỗi ảnh so sánh với mọi caption
- `labels = torch.arange(B)` - diagonal chính là cặp đúng (ảnh 0 với caption 0, ảnh 1 với caption 1,...)
- Tính cross-entropy hai chiều để đối xử công bằng ảnh → text và text → ảnh
- Gradient clipping tránh exploding gradients khi batch size lớn

### 6.2 Training loop hoàn chỉnh với hard negative mining

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, texts) in enumerate(pbar):
        images = images.to(device)
        texts = texts.to(device)
        
        metrics = training_step(model, (images, texts), optimizer)
        total_loss += metrics['loss']
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc_i2t': f"{metrics['acc_i2t']:.2%}",
            'acc_t2i': f"{metrics['acc_t2i']:.2%}",
            'temp': f"{metrics['temperature']:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    return total_loss / len(dataloader)

# Full training setup
def train_clip_model(model, train_loader, val_loader, epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer with different learning rates for different components
    vision_params = list(model.vision_encoder.parameters())
    text_params = list(model.text_encoder.parameters())
    other_params = [model.logit_scale]
    
    optimizer = torch.optim.AdamW([
        {'params': vision_params, 'lr': 1e-5, 'weight_decay': 0.2},
        {'params': text_params, 'lr': 1e-5, 'weight_decay': 0.2},
        {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.0}
    ])
    
    # Cosine annealing scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Validation
        val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_clip_model.pt')

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    for images, texts in dataloader:
        images, texts = images.to(device), texts.to(device)
        logits_i, logits_t = model(images, texts)
        loss = contrastive_loss(logits_i, logits_t)
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 6.3 Zero-shot inference

```python
@torch.no_grad()
def zero_shot_classification(model, image, class_names, device, template="a photo of a {}"):
    model.eval()
    
    # Encode image
    image = image.unsqueeze(0).to(device)
    image_features = model.encode_image(image)  # [1, d]
    
    # Encode all class names
    text_tokens = [template.format(c) for c in class_names]
    text_features = model.encode_text(text_tokens)  # [num_classes, d]
    
    # Compute similarities
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.T  # [1, num_classes]
    probs = F.softmax(logits, dim=-1)
    
    # Get top predictions
    top5_probs, top5_indices = probs[0].topk(5)
    
    results = []
    for prob, idx in zip(top5_probs, top5_indices):
        results.append({
            'class': class_names[idx],
            'confidence': prob.item()
        })
    
    return results

# Usage example
class_names = ['cat', 'dog', 'bird', 'fish']
predictions = zero_shot_classification(model, test_image, class_names, device)
for pred in predictions:
    print(f"{pred['class']}: {pred['confidence']:.2%}")
```

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
