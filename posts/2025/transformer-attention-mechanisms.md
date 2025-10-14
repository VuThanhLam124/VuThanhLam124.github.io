---
title: "Attention Mechanisms trong Transformers: Self, Cross, và Multi-Head"
date: "2025-10-08"
category: "Transformers"
tags: ["Transformers", "Attention", "NLP", "Architecture"]
excerpt: "Deep dive vào attention mechanisms - core component của Transformers. Phân tích mathematical foundations và practical implementations cho NLP và Computer Vision."
author: "ThanhLamDev"
readingTime: 16
featured: false
---

# Attention Mechanisms trong Transformers

Attention là trái tim của mọi kiến trúc Transformer. Bài viết phá vỡ từng thành phần và chỉ ra cách triển khai tối ưu trong PyTorch.

## 1. Self-attention

Self-attention cho phép mỗi token nhìn thấy toàn bộ sequence. Công thức chuẩn:

```
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
```

Trong practice cần:

- Additive bias để encode relative position.
- Masking cho causal hoặc padded tokens.

## 2. Cross-attention

Dùng trong encoder-decoder hoặc multimodal. Query đến từ decoder, Keys/Values từ encoder. Nên chú ý:

- Chuẩn hoá kích thước embeddings.
- Layer scaling để tránh gradient vanishing khi xếp chồng nhiều layer.

## 3. Multi-head attention

Chia embedding thành nhiều head giúp mô hình học nhiều pattern song song. Implementation gợi ý:

- Sử dụng ```einops``` để reshape gọn.
- Áp dụng FlashAttention hoặc xformers để tăng tốc.

## 4. Ứng dụng thực tế

- **NLP**: Machine translation, summarization.
- **Vision**: Vision Transformer, DETR.
- **Audio**: Speech recognition với conformer.

## 5. Code snippet

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out(out)
```

## 6. Tips cho production

- Tận dụng quantization hoặc low-rank adaptation để giảm memory.
- Cache Keys/Values cho inference streaming.
- Monitor gradient norm vì attention dễ bị nan nếu init sai.

Trong series kế tiếp, chúng ta sẽ xây dựng Transformer encoder tối ưu hóa bằng FlashAttention và benchmark trên sequence dài.

<script src="/assets/js/katex-init.js"></script>
