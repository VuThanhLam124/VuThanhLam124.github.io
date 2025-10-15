---
title: "Image–Text Fusion Techniques: Kỹ thuật cross-attention và co-embedding"
date: "2025-04-04"
category: "vision-language-models"
tags: ["vlm", "fusion", "cross-attention", "co-embedding", "transformer"]
excerpt: "Khảo sát các layer fusion nâng cao: UNITER, ALBEF, CoCa, Flamingo – cách kết hợp cross-attention và co-embedding để tăng khả năng reasoning."
author: "ThanhLamDev"
readingTime: 21
featured: false
---

# Image-Text Fusion Techniques: Kỹ thuật cross-attention và co-embedding

**Câu chuyện tại Bảo tàng Giao Thoa tiếp diễn. Sau khi alignment đảm bảo "ảnh nào với lời nào" (bài 2), cô hướng dẫn viên nhận ra cần một bước quan trọng hơn: không chỉ ghép cặp mà còn phải **kết hợp sâu** hai luồng thông tin. Ví dụ, khi khách hỏi "Chiếc bình gốm màu gì?", hệ thống phải biết liên kết từ "bình gốm" trong câu với vùng pixel cụ thể trong ảnh. Bài viết này giới thiệu các kỹ thuật fusion tiên tiến: cross-attention, co-embedding, và kiến trúc multimodal Transformer.**

## Mục lục

1. [Bối cảnh: Tại sao cần fusion sâu?](#1-bối-cảnh-tại-sao-cần-fusion-sâu)
2. [Cross-Attention: Cơ chế tương tác hai chiều](#2-cross-attention-cơ-chế-tương-tác-hai-chiều)
3. [Co-embedding: Học biểu diễn chung](#3-co-embedding-học-biểu-diễn-chung)
4. [Kiến trúc Multimodal Transformer](#4-kiến-trúc-multimodal-transformer)
5. [So sánh các phương pháp fusion](#5-so-sánh-các-phương-pháp-fusion)
6. [Implementation PyTorch chi tiết](#6-implementation-pytorch-chi-tiết)
7. [Training strategies và best practices](#7-training-strategies-và-best-practices)
8. [Ứng dụng thực tế: VQA, Image Captioning, Grounding](#8-ứng-dụng-thực-tế-vqa-image-captioning-grounding)
9. [Liên kết với các bài tiếp theo](#9-liên-kết-với-các-bài-tiếp-theo)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Bối cảnh: Tại sao cần fusion sâu?

Trong tour hướng dẫn, cô gặp những câu hỏi phức tạp:

- **"Bình gốm này có hoa văn gì?"** → cần liên kết từ "hoa văn" với vùng cụ thể trên bình
- **"Tại sao tranh này được vẽ trong mùa thu?"** → cần reasoning kết hợp màu sắc (visual) với ngữ cảnh (text)
- **"Có bao nhiêu người trong tranh?"** → counting yêu cầu grounding chính xác

**Alignment đơn thuần (CLIP)** chỉ đảm bảo embedding tổng thể của ảnh và text gần nhau, nhưng không tạo tương tác **chi tiết token-by-token**. 

**Fusion techniques** giải quyết vấn đề này bằng cách:

1. **Cross-attention**: Cho phép mỗi token văn bản "nhìn" vào các patch ảnh liên quan và ngược lại
2. **Co-embedding**: Học một không gian chung mà cả token ảnh và text đều được biểu diễn
3. **Multimodal Transformer**: Kết hợp cả self-attention và cross-attention trong một kiến trúc thống nhất

**Ví dụ so sánh:**

| Phương pháp | Tương tác | Ví dụ |
|------------|-----------|--------|
| **Late fusion** (CLIP) | Chỉ ở mức embedding cuối | Cosine similarity giữa `[CLS]` token |
| **Early fusion** | Concat input rồi xử lý chung | `[img_tokens, text_tokens]` → Transformer |
| **Cross-attention** | Token-level interaction | Text token "gốm" attend to các patch chứa bình gốm |

Cô hướng dẫn viên cần **cross-attention** để giải thích tương tác chi tiết giữa lời và ảnh.

---

## 2. Cross-Attention: Cơ chế tương tác hai chiều

### 2.1 Định nghĩa toán học

Trong self-attention thông thường, Query, Key, Value đều từ cùng một nguồn. Trong **cross-attention**, chúng đến từ hai modality khác nhau.

**Vision → Text attention** (text attend to image):

$$
Q_t = W_Q^t \cdot T, \quad K_v = W_K^v \cdot V, \quad V_v = W_V^v \cdot V
$$

trong đó:
- $T \in \mathbb{R}^{N_t \times d}$: text embeddings ($N_t$ tokens)
- $V \in \mathbb{R}^{N_v \times d}$: vision embeddings ($N_v$ patches)
- $W_Q^t, W_K^v, W_V^v$: projection matrices

**Attention scores:**

$$
A_{t \to v} = \text{softmax}\left(\frac{Q_t K_v^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{N_t \times N_v}
$$

**Output:**

$$
T' = A_{t \to v} V_v
$$

Mỗi text token giờ được "bổ sung" thông tin từ các image patches liên quan.

**Text → Vision attention** (image attend to text):

Tương tự, đổi vai trò Query và Key:

$$
Q_v = W_Q^v \cdot V, \quad K_t = W_K^t \cdot T, \quad V_t = W_V^t \cdot T
$$

$$
A_{v \to t} = \text{softmax}\left(\frac{Q_v K_t^\top}{\sqrt{d_k}}\right)
$$

$$
V' = A_{v \to t} V_t
$$

### 2.2 Giải thích trực quan

Khi khách hỏi **"Bình gốm màu gì?"**:

1. Token "bình gốm" trong câu hỏi tạo Query
2. Các image patches tạo Key/Value
3. Attention weight cao ở patches chứa bình gốm
4. Token "bình gốm" được update với visual features của vùng đó
5. Từ đó model có thể trả lời "xanh" dựa trên màu sắc trong visual features

### 2.3 Multi-head Cross-Attention

Giống self-attention, ta dùng nhiều head để học các kiểu tương tác khác nhau:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

trong đó:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

**Ý nghĩa các head khác nhau:**

- Head 1: học tương tác màu sắc (color grounding)
- Head 2: học spatial relationship (vị trí)
- Head 3: học object-attribute binding (đối tượng-thuộc tính)

---

## 3. Co-embedding: Học biểu diễn chung

### 3.1 Khái niệm

Thay vì giữ vision và text embeddings riêng biệt, **co-embedding** học một không gian chung $\mathbb{R}^d$ mà cả hai modality đều được project vào.

**Quy trình:**

1. Encode riêng: $V_{\text{raw}} = \text{ViT}(I)$, $T_{\text{raw}} = \text{BERT}(S)$
2. Project về không gian chung:
   
   $$
   V_{\text{shared}} = W_v V_{\text{raw}} + b_v
   $$
   
   $$
   T_{\text{shared}} = W_t T_{\text{raw}} + b_t
   $$

3. Concat và xử lý chung:
   
   $$
   M = [V_{\text{shared}}; T_{\text{shared}}] \in \mathbb{R}^{(N_v + N_t) \times d}
   $$

4. Transformer blocks xử lý $M$ với self-attention

### 3.2 Ưu nhược điểm

**Ưu điểm:**

- Tương tác tự nhiên thông qua self-attention
- Dễ mở rộng sang 3+ modalities (audio, video, etc.)
- Training đơn giản hơn cross-attention (ít hyperparameters)

**Nhược điểm:**

- Mất một phần đặc thù của từng modality (visual inductive bias, language structure)
- Chi phí tính toán cao khi $N_v + N_t$ lớn (quadratic attention)

### 3.3 Toán học: Self-attention trên co-embedded space

Sau khi concat $M = [V_{\text{shared}}; T_{\text{shared}}]$:

$$
Q = M W_Q, \quad K = M W_K, \quad V = M W_V
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

Điểm quan trọng: mỗi token (dù ảnh hay text) có thể attend to tất cả tokens khác, tạo **full cross-modal interaction**.

---

## 4. Kiến trúc Multimodal Transformer

### 4.1 Thiết kế hai luồng (Dual-stream)

Kiến trúc này kết hợp tốt nhất của hai thế giới:

1. **Vision stream**: xử lý riêng image features với self-attention
2. **Text stream**: xử lý riêng text features với self-attention
3. **Cross-attention layers**: kết nối hai stream

**Công thức toán học:**

Tại layer $l$:

$$
V^{(l)} = \text{SelfAttn}_v(V^{(l-1)}) + V^{(l-1)}
$$

$$
T^{(l)} = \text{SelfAttn}_t(T^{(l-1)}) + T^{(l-1)}
$$

$$
V^{(l)} = \text{CrossAttn}(V^{(l)}, T^{(l)}) + V^{(l)}
$$

$$
T^{(l)} = \text{CrossAttn}(T^{(l)}, V^{(l)}) + T^{(l)}
$$

$$
V^{(l)} = \text{FFN}(V^{(l)}) + V^{(l)}
$$

$$
T^{(l)} = \text{FFN}(T^{(l)}) + T^{(l)}
$$

### 4.2 Single-stream architecture (BERT-style)

Ví dụ: **VisualBERT, UNITER, OSCAR**

```
Input: [CLS] text_tokens [SEP] vision_tokens [SEP]
       ↓
    Transformer layers (self-attention)
       ↓
    [CLS] output → classification/VQA
```

**Segment embeddings:**

$$
E_{\text{total}} = E_{\text{token}} + E_{\text{pos}} + E_{\text{segment}}
$$

trong đó:
- $E_{\text{segment}} = 0$ cho text tokens
- $E_{\text{segment}} = 1$ cho vision tokens

### 4.3 So sánh Dual-stream vs Single-stream

| Tiêu chí | Dual-stream | Single-stream |
|----------|-------------|---------------|
| **Tương tác** | Explicit cross-attention | Implicit qua self-attention |
| **Modality-specific processing** | Có (hai stream riêng) | Không (xử lý chung) |
| **Tính toán** | Nhẹ hơn (attention riêng) | Nặng hơn (full quadratic) |
| **Flexibility** | Dễ thêm modality mới | Khó modify |
| **Ví dụ** | CLIP, FLAVA | VisualBERT, UNITER |

---

## 5. So sánh các phương pháp fusion

### 5.1 Phân loại theo thời điểm fusion

**Early fusion:**

$$
f([V; T]) \rightarrow \text{output}
$$

Concat ngay từ đầu, xử lý chung qua Transformer.

**Late fusion:**

$$
g(f_v(V), f_t(T)) \rightarrow \text{output}
$$

Xử lý riêng rồi kết hợp ở output (ví dụ: CLIP).

**Intermediate fusion (Cross-attention):**

$$
V' = f_v(V, T), \quad T' = f_t(T, V)
$$

Tương tác ở các layer trung gian.

### 5.2 Bảng so sánh tổng hợp

| Phương pháp | Độ phức tạp | Tương tác | Use case |
|------------|-------------|-----------|----------|
| **Concatenation** | $O((N_v + N_t)^2)$ | Full | VQA, Captioning |
| **Cross-attention** | $O(N_v N_t)$ | Bidirectional | Grounding, Reasoning |
| **Co-attention** | $O(N_v N_t)$ | Symmetric | Visual Reasoning |
| **Gated fusion** | $O(d)$ | Feature-level | Multimodal classification |

### 5.3 Công thức Gated Fusion

Một biến thể thú vị là **gated fusion**, học trọng số động cho mỗi modality:

$$
\alpha = \sigma(W_g [V; T] + b_g)
$$

$$
M_{\text{fused}} = \alpha \odot V + (1 - \alpha) \odot T
$$

trong đó $\sigma$ là sigmoid, $\odot$ là element-wise product.

---

## 6. Implementation PyTorch chi tiết

### 6.1 Cross-Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for vision-language fusion
    
    Args:
        dim: embedding dimension
        num_heads: number of attention heads
        dropout: dropout probability
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query: [B, N_q, D] - query modality (e.g., text)
            key: [B, N_k, D] - key modality (e.g., vision)
            value: [B, N_v, D] - value modality (same as key)
            attention_mask: [B, N_q, N_k] - optional mask
            
        Returns:
            output: [B, N_q, D] - attended query features
            attention_weights: [B, num_heads, N_q, N_k]
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]
        
        # Linear projections and reshape for multi-head attention
        # [B, N, D] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        Q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [B, num_heads, N_q, head_dim] @ [B, num_heads, head_dim, N_k]
        # -> [B, num_heads, N_q, N_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1) == 0, 
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, N_q, N_k] @ [B, num_heads, N_k, head_dim]
        # -> [B, num_heads, N_q, head_dim]
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        # [B, num_heads, N_q, head_dim] -> [B, N_q, num_heads, head_dim]
        # -> [B, N_q, D]
        output = output.transpose(1, 2).reshape(B, N_q, D)
        output = self.o_proj(output)
        
        return output, attn_weights
```

### 6.2 Multimodal Fusion Layer

```python
class MultimodalFusionLayer(nn.Module):
    """
    Complete fusion layer with self-attention and cross-attention
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # Vision self-attention
        self.vision_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.vision_norm1 = nn.LayerNorm(dim)
        
        # Text self-attention
        self.text_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_norm1 = nn.LayerNorm(dim)
        
        # Cross-attention: vision attends to text
        self.v2t_cross_attn = CrossAttention(dim, num_heads, dropout)
        self.vision_norm2 = nn.LayerNorm(dim)
        
        # Cross-attention: text attends to vision
        self.t2v_cross_attn = CrossAttention(dim, num_heads, dropout)
        self.text_norm2 = nn.LayerNorm(dim)
        
        # Feed-forward networks
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.vision_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.vision_norm3 = nn.LayerNorm(dim)
        
        self.text_mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.text_norm3 = nn.LayerNorm(dim)
        
    def forward(self, vision_feats, text_feats, vision_mask=None, text_mask=None):
        """
        Args:
            vision_feats: [B, N_v, D]
            text_feats: [B, N_t, D]
            vision_mask: [B, N_v] optional
            text_mask: [B, N_t] optional
            
        Returns:
            vision_out: [B, N_v, D]
            text_out: [B, N_t, D]
        """
        # Self-attention for vision
        v_self, _ = self.vision_self_attn(
            vision_feats, vision_feats, vision_feats, 
            key_padding_mask=vision_mask
        )
        vision_feats = self.vision_norm1(vision_feats + v_self)
        
        # Self-attention for text
        t_self, _ = self.text_self_attn(
            text_feats, text_feats, text_feats,
            key_padding_mask=text_mask
        )
        text_feats = self.text_norm1(text_feats + t_self)
        
        # Cross-attention: vision → text
        v_cross, v2t_weights = self.v2t_cross_attn(
            vision_feats, text_feats, text_feats
        )
        vision_feats = self.vision_norm2(vision_feats + v_cross)
        
        # Cross-attention: text → vision
        t_cross, t2v_weights = self.t2v_cross_attn(
            text_feats, vision_feats, vision_feats
        )
        text_feats = self.text_norm2(text_feats + t_cross)
        
        # Feed-forward
        vision_feats = self.vision_norm3(vision_feats + self.vision_mlp(vision_feats))
        text_feats = self.text_norm3(text_feats + self.text_mlp(text_feats))
        
        return vision_feats, text_feats, (v2t_weights, t2v_weights)
```

### 6.3 Complete Multimodal Transformer

```python
class MultimodalTransformer(nn.Module):
    """
    Full multimodal transformer with vision and text encoders
    """
    def __init__(
        self, 
        vision_encoder,
        text_encoder,
        fusion_dim=768,
        num_fusion_layers=6,
        num_heads=12,
        dropout=0.1
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Projection to common dimension
        vision_dim = vision_encoder.embed_dim
        text_dim = text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            MultimodalFusionLayer(fusion_dim, num_heads, dropout=dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # Task-specific heads (examples)
        self.vqa_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 3129)  # VQA answer vocab size
        )
        
        self.matching_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2)  # binary: match or not
        )
        
    def forward(self, images, input_ids, attention_mask, task='vqa'):
        """
        Args:
            images: [B, 3, H, W]
            input_ids: [B, L] text token ids
            attention_mask: [B, L] text attention mask
            task: 'vqa', 'matching', 'captioning'
            
        Returns:
            task-specific output
        """
        # Encode separately
        vision_feats = self.vision_encoder(images)  # [B, N_v, D_v]
        text_feats = self.text_encoder(
            input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state  # [B, L, D_t]
        
        # Project to common space
        vision_feats = self.vision_proj(vision_feats)  # [B, N_v, D]
        text_feats = self.text_proj(text_feats)  # [B, L, D]
        
        # Fusion layers
        all_attention_weights = []
        for layer in self.fusion_layers:
            vision_feats, text_feats, attn_weights = layer(
                vision_feats, text_feats
            )
            all_attention_weights.append(attn_weights)
        
        # Task-specific processing
        if task == 'vqa':
            # Use [CLS] token (first text token)
            cls_feat = text_feats[:, 0]  # [B, D]
            logits = self.vqa_head(cls_feat)
            return logits
            
        elif task == 'matching':
            # Pool both modalities
            vision_pooled = vision_feats.mean(dim=1)  # [B, D]
            text_pooled = text_feats[:, 0]  # [B, D]
            combined = torch.cat([vision_pooled, text_pooled], dim=-1)
            logits = self.matching_head(combined)
            return logits
            
        else:
            return vision_feats, text_feats, all_attention_weights
```

### 6.4 Co-embedding Approach

```python
class CoEmbeddingTransformer(nn.Module):
    """
    Single-stream approach: concat vision and text in shared space
    """
    def __init__(self, dim=768, num_layers=12, num_heads=12, dropout=0.1):
        super().__init__()
        
        # Modality-specific embeddings
        self.vision_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Unified transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, vision_embeds, text_embeds):
        """
        Args:
            vision_embeds: [B, N_v, D]
            text_embeds: [B, N_t, D]
            
        Returns:
            output: [B, 1 + N_v + N_t, D]
        """
        B = vision_embeds.size(0)
        
        # Add modality type embeddings
        vision_embeds = vision_embeds + self.vision_type_embed
        text_embeds = text_embeds + self.text_type_embed
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenate: [CLS] + vision + text
        combined = torch.cat([cls_tokens, vision_embeds, text_embeds], dim=1)
        
        # Process with transformer
        output = self.transformer(combined)
        
        return output
```

---

## 7. Training strategies và best practices

### 7.1 Multi-task Learning

Thay vì train riêng cho từng task, train đồng thời nhiều objectives:

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{ITM}} + \lambda_2 \mathcal{L}_{\text{MLM}} + \lambda_3 \mathcal{L}_{\text{VQA}}
$$

**Image-Text Matching (ITM):**

$$
\mathcal{L}_{\text{ITM}} = -\sum_{i=1}^B \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

trong đó $y_i = 1$ nếu ảnh và text khớp, $0$ nếu không.

**Masked Language Modeling (MLM):**

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(w_i \mid w_{\setminus \mathcal{M}}, V)
$$

$\mathcal{M}$: tập các masked positions.

### 7.2 Hard Negative Mining cho Cross-modal Matching

```python
def hard_negative_mining(vision_feats, text_feats, labels, k=5):
    """
    Select hard negatives for contrastive learning
    
    Args:
        vision_feats: [B, D]
        text_feats: [B, D]
        labels: [B] - matching labels
        k: number of hard negatives
        
    Returns:
        hard_negatives: indices of hard negative samples
    """
    B = vision_feats.size(0)
    
    # Compute all pairwise similarities
    similarities = torch.matmul(vision_feats, text_feats.T)  # [B, B]
    
    # Mask out positive pairs
    mask = torch.eye(B, device=vision_feats.device).bool()
    similarities = similarities.masked_fill(mask, float('-inf'))
    
    # Get top-k most similar (but incorrect) pairs
    hard_neg_indices = similarities.topk(k, dim=1).indices
    
    return hard_neg_indices
```

### 7.3 Gradient Accumulation cho Batch Size lớn

```python
def train_step_with_gradient_accumulation(
    model, batch, optimizer, accumulation_steps=4
):
    """
    Training with gradient accumulation for large effective batch size
    """
    model.train()
    total_loss = 0
    
    for i, (images, texts, labels) in enumerate(batch):
        # Forward pass
        logits = model(images, texts)
        loss = F.cross_entropy(logits, labels)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(batch)
```

### 7.4 Learning Rate Scheduling

```python
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7
):
    """
    Cosine learning rate schedule with warmup
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine annealing
        progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## 8. Ứng dụng thực tế: VQA, Image Captioning, Grounding

### 8.1 Visual Question Answering (VQA)

```python
class VQAModel(nn.Module):
    """
    VQA model using multimodal fusion
    """
    def __init__(self, multimodal_encoder, answer_vocab_size):
        super().__init__()
        self.encoder = multimodal_encoder
        self.classifier = nn.Linear(768, answer_vocab_size)
        
    def forward(self, images, questions, question_mask):
        # Fuse vision and language
        output = self.encoder(
            images, questions, question_mask, task='fusion'
        )
        
        # Use [CLS] token for classification
        cls_output = output[0][:, 0]  # [B, D]
        logits = self.classifier(cls_output)
        
        return logits

# Training loop
def train_vqa(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, questions, masks, answers in dataloader:
        images = images.to(device)
        questions = questions.to(device)
        masks = masks.to(device)
        answers = answers.to(device)
        
        # Forward
        logits = model(images, questions, masks)
        loss = F.cross_entropy(logits, answers)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 8.2 Image Captioning

```python
class CaptioningModel(nn.Module):
    """
    Image captioning with cross-attention decoder
    """
    def __init__(self, vision_encoder, decoder, vocab_size):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.decoder = decoder  # Transformer decoder
        self.word_embed = nn.Embedding(vocab_size, 768)
        self.lm_head = nn.Linear(768, vocab_size)
        
    def forward(self, images, captions, caption_mask):
        # Encode image
        vision_feats = self.vision_encoder(images)  # [B, N, D]
        
        # Embed captions
        caption_embeds = self.word_embed(captions)  # [B, L, D]
        
        # Decoder with cross-attention to vision
        decoder_output = self.decoder(
            caption_embeds,
            vision_feats,
            tgt_mask=caption_mask
        )
        
        # Predict next words
        logits = self.lm_head(decoder_output)
        
        return logits
    
    @torch.no_grad()
    def generate(self, images, max_length=20, temperature=1.0):
        """
        Autoregressive generation
        """
        B = images.size(0)
        device = images.device
        
        # Encode image once
        vision_feats = self.vision_encoder(images)
        
        # Start with [BOS] token
        generated = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Embed current sequence
            embeds = self.word_embed(generated)
            
            # Decode
            output = self.decoder(embeds, vision_feats)
            
            # Get logits for last position
            logits = self.lm_head(output[:, -1, :]) / temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generated [EOS]
            if (next_token == self.eos_token_id).all():
                break
        
        return generated
```

### 8.3 Visual Grounding

```python
class GroundingModel(nn.Module):
    """
    Predict bounding box for phrase in image
    """
    def __init__(self, multimodal_encoder):
        super().__init__()
        self.encoder = multimodal_encoder
        
        # Bbox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4)  # (x, y, w, h)
        )
        
    def forward(self, images, phrases, phrase_mask):
        # Fuse vision and phrase
        vision_out, text_out, attn_weights = self.encoder(
            images, phrases, phrase_mask, task='fusion'
        )
        
        # Use attention weights to locate phrase in image
        # Average over all heads and layers
        v2t_attn = attn_weights[-1][0]  # Last layer, vision→text
        phrase_attn = v2t_attn.mean(dim=1)  # [B, N_v, N_t]
        
        # Aggregate phrase attention
        phrase_importance = phrase_attn.mean(dim=-1)  # [B, N_v]
        
        # Weighted pooling of vision features
        weighted_vision = (vision_out * phrase_importance.unsqueeze(-1)).sum(dim=1)
        
        # Predict bbox
        bbox = self.bbox_head(weighted_vision)
        bbox = torch.sigmoid(bbox)  # Normalize to [0, 1]
        
        return bbox, phrase_attn
```

---

## 9. Liên kết với các bài tiếp theo

Sau khi thành thạo fusion techniques, cô hướng dẫn viên quan tâm đến:

1. **Bài 4 - Instruction Tuning**: Làm sao dạy mô hình "nghe lời" các câu lệnh phức tạp?
2. **Bài 5 - Efficient VLM**: Làm sao giảm chi phí tính toán khi xử lý hàng trăm ảnh mỗi tour?
3. **Bài 6 - VLM Evaluation**: Đo lường chất lượng trả lời của hệ thống như thế nào?

---

## 10. Tài liệu tham khảo

1. Lu et al. (2019). *ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks.* NeurIPS.
2. Li et al. (2019). *VisualBERT: A Simple and Performant Baseline for Vision and Language.* arXiv.
3. Chen et al. (2020). *UNITER: UNiversal Image-TExt Representation Learning.* ECCV.
4. Li et al. (2021). *ALBEF: Align before Fuse - Vision and Language Representation Learning with Momentum Distillation.* NeurIPS.
5. Wang et al. (2021). *SimVLM: Simple Visual Language Model Pretraining with Weak Supervision.* ICLR.
6. Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS.
7. Bain et al. (2021). *Frozen in Time: A Joint Video and Language Encoder for End-to-End Retrieval.* ICCV.

---

<script src="/assets/js/katex-init.js"></script>
