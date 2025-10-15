---
title: "Foundations: Vision Encoder và Image Tokenization"
date: "2025-03-25"
category: "vision-language-models"
tags: ["vlm", "vision-encoder", "tokenization", "vit", "cnn"]
excerpt: "Khởi đầu series VLM với câu chuyện của một hướng dẫn viên bảo tàng: làm sao chuyển từng bức tranh thành chuỗi thông tin để khách khiếm thị và khiếm thính cùng hiểu."
author: "ThanhLamDev"
readingTime: 24
featured: false
---

# Foundations: Vision Encoder và Image Tokenization

**Series VLM khởi đầu với một nhân vật quen thuộc – *hướng dẫn viên bảo tàng*. Mỗi ngày, cô dẫn những đoàn khách đa dạng: có học sinh khiếm thính chỉ đọc bằng mắt, có cụ già suy giảm thị lực chỉ nghe lời mô tả. Nhiệm vụ của cô là biến bức tranh thành chuỗi thông tin rõ ràng để cả hai nhóm đều hiểu cùng một câu chuyện. Đó chính là lý do Vision Encoder và Image Tokenization tồn tại trong các Vision-Language Model.**

## Mục lục

1. [Câu chuyện tại Bảo tàng Giao Thoa](#1-câu-chuyện-tại-bảo-tàng-giao-thoa)
2. [Tại sao phải token hóa hình ảnh?](#2-tại-sao-phải-token-hóa-hình-ảnh)
3. [Các kiến trúc encoder nền tảng](#3-các-kiến-trúc-encoder-nền-tảng)
4. [Quy trình tokenization từng bước](#4-quy-trình-tokenization-từng-bước)
5. [Chi tiết toán học: từ patch tới embedding](#5-chi-tiết-toán-học-từ-patch-tới-embedding)
6. [Ví dụ PyTorch: chuẩn bị token cho LLM](#6-ví-dụ-pytorch-chuẩn-bị-token-cho-llm)
7. [Best practices & bảng so sánh](#7-best-practices--bảng-so-sánh)
8. [Nhìn về các bài tiếp theo](#8-nhìn-về-các-bài-tiếp-theo)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Câu chuyện tại Bảo tàng Giao Thoa

Một buổi sáng, cô hướng dẫn viên đón một đoàn khách đặc biệt. Nửa đoàn là học sinh khiếm thính – các em chỉ cảm nhận được nội dung khi nhìn thấy chi tiết đầy đủ. Nửa còn lại là những vị khách lớn tuổi suy giảm thị lực, họ cần lời thuyết minh rõ ràng để hình dung. Cô dùng máy tính bảng kết nối với hệ thống VLM của bảo tàng để "dịch" bức tranh sang ngôn từ. Muốn làm được điều đó, cô phải:

1. **Dùng Vision Encoder** trích xuất đặc trưng từng vùng ảnh.
2. **Token hóa** các đặc trưng đó thành chuỗi vector – giống như ghi chú chi tiết để mô hình ngôn ngữ tạo lời mô tả chính xác.

> *Ghi chú xuyên suốt series:* Các bài viết tiếp theo sẽ mô tả cách cô phối hợp với những kỹ thuật khác (alignment, instruction tuning, compression...). Bài hôm nay tập trung vào bước nền tảng: chuyển ảnh thành "ngôn ngữ trung gian".

## 2. Tại sao phải token hóa hình ảnh?

- Ngôn ngữ xử lý chuỗi; ảnh lại là tensor ba chiều. Tokenization là phép biến đổi $f_{\text{vision}}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{N \times d}$ để ánh xạ ảnh thành chuỗi $N$ vector chiều $d$ mà mô hình ngôn ngữ có thể tiếp nhận giống như token văn bản.
- Quyết định tokenization tác động tới ba yếu tố: **độ chính xác** (alignment ở bài 2), **chi phí tính toán** (bài 4) và **khả năng reasoning đa bước** (bài 7).
- Về mặt xác suất, tokenization tương ứng với việc xây dựng phân phối ẩn $q(Z \mid X)$ cho chuỗi latent $Z$; các bài sau sẽ khai thác phân phối này khi tối ưu objective hybrid.

### Bảng ký hiệu

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $X$ | ảnh đầu vào kích thước $H \times W \times 3$ |
| $P$ | kích thước patch (ví dụ 16) |
| $N = (H/P)\cdot(W/P)$ | số lượng patch/token ảnh |
| $d$ | chiều embedding sau encoder |
| $d_{\text{LLM}}$ | chiều không gian mô hình ngôn ngữ |
| $W_E, b_E$ | trọng số embedding patch |
| $E_{\text{pos}}$ | positional embedding |
| $W_{\text{proj}}$ | projector từ $d$ sang $d_{\text{LLM}}$ |

## 3. Các kiến trúc encoder nền tảng

### 3.1 Vision Transformer (ViT)

Tại phòng thí nghiệm của bảo tàng, cô dùng **máy chia patch** (ViT) để tách bức tranh thành những ô nhỏ và phân tích chúng.

1. **Chia patch:** mỗi patch kích thước $P \times P$.
2. **Khắc chữ lên patch** (linear embedding):
   $$
   z_{i,j} = W_E \cdot \text{vec}\big(X[iP:(i+1)P, jP:(j+1)P,:]\big) + b_E,\quad W_E \in \mathbb{R}^{d \times 3P^2}.
   $$
3. **Thêm vị trí & modality:** $\tilde{z}_{i,j} = z_{i,j} + E_{\text{pos}}(i,j) + E_{\text{mod}}$.
4. **Hội nghị tự chú ý:**
   $$
   \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V,\quad Q=ZW_Q,\; K=ZW_K,\; V=ZW_V.
   $$

Sơ đồ kiến trúc:
```
Image -> Patchify -> Linear Embed -> +Positional -> Transformer Blocks -> Patch Tokens
```

Các biến thể: ViT-B/16, ViT-L/14 khác độ sâu và chiều embedding; EVA-CLIP, SigLIP bổ sung regularization/sigmoid loss để tăng độ ổn định và hỗ trợ đa ngôn ngữ.

### 3.2 CNN & hybrid backbone

Khi dẫn tour lưu động, cô chuyển sang **bộ quét CNN/Hybrid** vì gọn nhẹ và xử lý nhanh hơn.

- **Quét ảnh:** convolution sinh feature map $F \in \mathbb{R}^{h \times w \times c}$.
- **Gộp vùng thành token:**
  $$
  t_k = \frac{1}{|R_k|} \sum_{(u,v)\in R_k} F[u,v,:],
  $$
  với $R_k$ là vùng receptive.
- **Flatten thành chuỗi**: sắp xếp $\{t_k\}$ thành $T \in \mathbb{R}^{N \times c}$ và đưa vào projector.

Sơ đồ:
```
Image -> CNN Stages -> Feature Map -> Pool / Flatten -> Tokens
```

ResNet, ConvNeXt phù hợp edge device; Swin Transformer, CoAtNet kết hợp convolution & attention cục bộ để vừa giữ locality vừa mở rộng receptive field.

### 3.3 Multi-scale & feature pyramid

Khi thuyết minh trước khán phòng lớn, cô chuẩn bị **nhiều độ phân giải** của cùng bức tranh để hệ thống hiểu được cả chi tiết và tổng thể. Encoder sinh $Z^{(s)}$; sau đó:
$$
Z = \text{Concat}(Z^{(1)}, Z^{(2)}, \ldots) \quad \text{hoặc} \quad Z = \text{Attention}(Z^{(1)}, Z^{(2)}, \ldots).
$$
Nhờ vậy, câu chuyện mô tả được cả chi tiết nhỏ (mặt nước) lẫn bối cảnh lớn (bầu trời).

Sơ đồ multi-scale:
```
Image
 ├─ resize 224 -> ViT-B/16 -> Tokens224
 └─ resize 336 -> ViT-L/14 -> Tokens336
Concat -> Linear projector -> Unified tokens
```

Ví dụ PyTorch:
```python
class MultiScaleTokenizer(nn.Module):
    def __init__(self, backbones, scales, target_dim):
        super().__init__()
        self.encoders = nn.ModuleList([
            timm.create_model(name, pretrained=True) for name in backbones
        ])
        for enc in self.encoders:
            enc.reset_classifier(0)
        self.scales = scales
        dim_sum = sum(enc.num_features for enc in self.encoders)
        self.projector = nn.Linear(dim_sum, target_dim)

    def forward(self, images):
        feats = []
        for scale, enc in zip(self.scales, self.encoders):
            x = F.interpolate(images, size=(scale, scale), mode="bicubic", align_corners=False)
            feats.append(enc.forward_features(x))  # [B, N_s, C_s]
        merged = torch.cat(feats, dim=-1)  # concat theo chiều feature
        return self.projector(merged)
```

## 4. Quy trình tokenization từng bước

1. **Chuẩn hóa ảnh**: $X \leftarrow \text{Resize}(X, H \times W)$, sau đó chuẩn hóa bằng mean/std của encoder (ví dụ CLIP mean = (0.481, 0.457, 0.408)).
2. **Patchify / Feature extraction**: thu tensor $Z \in \mathbb{R}^{B \times N \times d}$.
3. **Projection tới không gian LLM**:
   $$
   Y = \text{LayerNorm}(Z) W_{\text{proj}} + b_{\text{proj}}, \quad W_{\text{proj}} \in \mathbb{R}^{d \times d_{\text{LLM}}}.
   $$
4. **Thêm embedding modality & vị trí**: $Y = Y + E_{\text{mod}} + E_{\text{pos}}$.
5. **Ghép prompt**: chèn token đặc biệt `<IMG_TOKEN>` và `<IMG_END>` bao quanh chuỗi $Y$ khi đưa vào LLM.

> **Chú thích:** LayerNorm trước projection giúp ổn định gradient; $E_{\text{mod}}$ đánh dấu đây là token ảnh, giúp LLM phân biệt với token văn bản.

## 5. Chi tiết toán học: từ patch tới embedding

Cho ảnh $X \in \mathbb{R}^{H \times W \times 3}$. Chọn patch size $P$, số patch mỗi chiều $N_H = H/P$, $N_W = W/P$.

### 5.1 Trích patch

$$
\text{Patch}_{i,j} = X[iP:(i+1)P, \; jP:(j+1)P, :],\quad 0 \le i < N_H, 0 \le j < N_W.
$$

Flatten patch thành vector $p_{i,j} \in \mathbb{R}^{3P^2}$. Đây là bước “cắt tranh” ra $N$ mảnh nhỏ.

### 5.2 Linear embedding

$$
z_{i,j} = W_E p_{i,j} + b_E, \quad W_E \in \mathbb{R}^{d \times 3P^2}.
$$

**Giải thích:** $z_{i,j}$ là “từ vựng” mô tả patch $(i,j)$ trong không gian $d$ chiều. Trong câu chuyện, đây tương tự mảnh thông tin mà hướng dẫn viên ghi lại cho cư dân Lời Sáng.

### 5.3 Thêm positional & modality encoding

$$
\tilde{z}_{i,j} = z_{i,j} + E_{\text{pos}}(i,j) + E_{\text{mod}}.
$$

$E_{\text{mod}}$ giúp LLM biết rằng token này xuất phát từ ảnh; $E_{\text{pos}}$ giữ vị trí patch. Chuỗi cuối $Z = [\tilde{z}_{0,0}, \dots, \tilde{z}_{N_H-1,N_W-1}]$ sẽ được projector đưa sang không gian $d_{\text{LLM}}$.

## 6. Ví dụ PyTorch: chuẩn bị token cho LLM

```python
import torch
import timm
import torch.nn as nn

class VisionTokenizer(nn.Module):
    def __init__(self, backbone="openai/clip-vit-large-patch14", target_dim=4096):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True)
        self.encoder.reset_classifier(0)
        dim = self.encoder.num_features
        self.projector = nn.Linear(dim, target_dim)
        self.modality_embed = nn.Parameter(torch.randn(1, 1, target_dim))

    def forward(self, images):
        feats = self.encoder.forward_features(images)  # [B, N, dim]
        tokens = self.projector(feats)
        tokens = tokens + self.modality_embed  # đánh dấu "ảnh"
        return tokens

# Usage
tokenizer = VisionTokenizer()
images = torch.randn(4, 3, 336, 336)
vision_tokens = tokenizer(images)  # [4, N_tokens, 4096]

# Gắn vào prompt ngôn ngữ
def build_prompt(vision_tokens, text_tokens):
    img_tokens = tokenizer.modality_embed.repeat(1, vision_tokens.size(1), 1)
    return torch.cat([img_tokens + vision_tokens, text_tokens], dim=1)
```

**Chú thích:** `forward_features` của ViT trả về patch embedding trước CLS token; nếu muốn giữ CLS, cần concatenate thủ công. Hàm `build_prompt` minh họa cách nối token ảnh với token văn bản khi inference.

## 7. Best practices & bảng so sánh

| Tiêu chí | Lựa chọn khuyến nghị | Ghi chú |
|----------|----------------------|---------|
| Độ chi tiết cao | ViT-L/14, resolution 336 | OCR, ảnh y tế |
| Tốc độ realtime | ResNet50 + projector | latency thấp |
| Đa ngôn ngữ | SigLIP, EVA-CLIP | pretrain đa ngôn ngữ |
| Tiết kiệm bộ nhớ | Token merging 0.5 hoặc Perceiver Resampler | khác biệt sẽ phân tích ở bài “Compression” |
| Edge deployment | ConvNeXt-Tiny + pooling | phù hợp 8GB VRAM |

Tips thực tế:
- Chuẩn hóa ảnh trong `DataLoader` để đảm bảo batch consistency.
- Cache vision features trong multi-turn chat để không encode lại ảnh.
- Dùng mixed precision (bf16) và `channels_last` để tối ưu throughput.
- Theo dõi số token $N$: tăng patch size hoặc dùng merging khi $N>1024$ để tránh tắc nghẽn attention.

## 8. Nhìn về các bài tiếp theo

Sau khi thiết lập được quy trình “chuyển tranh thành lời”, cô hướng dẫn viên cần học cách **đồng bộ** giữa hình ảnh và câu mô tả – đó là chủ đề của bài 2 (contrastive learning với CLIP/ALIGN). Bài 3 bàn về các mục tiêu pretraining hybrid để lời thuyết minh phong phú hơn. Những bài tiếp theo tiếp tục hành trình: tối ưu encoder cho chạy nhanh, huấn luyện theo hướng dẫn và cuối cùng là hợp nhất ảnh vào chuỗi token ngôn ngữ.

## 9. Tài liệu tham khảo

1. Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
2. Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision (CLIP)*.
3. Tu et al. (2023). *EVA-CLIP: Advanced ViT for vision-language*.

---

<script src="/assets/js/katex-init.js"></script>
