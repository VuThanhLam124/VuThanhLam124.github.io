---
title: "Compression & Token Integration (Phần II): Neural codec, tích hợp vào LLM và huấn luyện đa giai đoạn"
date: "2025-04-10"
category: "vision-language-models"
tags: ["vlm", "neural-codec", "palme", "perceiver-resampler", "token-integration", "llm-training", "rlhf"]
excerpt: "Phần II tiếp nối nền tảng của phần I, tập trung vào neural codec, hyperprior, Perceiver Resampler, PaLM-E, Kosmos-2, IDEFICS-2, chiến lược gắn token ảnh vào LLM, pipeline SFT/RLHF và đánh giá. Mọi khái niệm đều được giải thích cặn kẽ với công thức, sơ đồ, pseudo-code và checklist triển khai."
author: "ThanhLamDev"
readingTime: 62
featured: false
---

# Compression & Token Integration (Phần II): Neural codec, tích hợp vào LLM và huấn luyện đa giai đoạn

**Phần II tiếp tục hành trình của cô hướng dẫn viên cùng đội kỹ thuật đa phương thức tại Bảo tàng Giao Thoa. Sau khi đã nắm vững nền tảng vector quantization ở phần I, họ chuyển sang bài toán lớn hơn: làm sao để đưa token ảnh vào LLM với chi phí context tối ưu, đồng thời duy trì độ trung thực khi reasoning đa bước. Tất cả các kỹ thuật cutting-edge – từ hyperprior codec, Perceiver Resampler, PaLM-E, Kosmos-2 đến pipeline fine-tuning với RLHF – sẽ được giải thích chi tiết trong bài này.**

---

## Lộ trình đọc phần II

1. [Nhắc lại kết quả phần I và mục tiêu phần II](#1-nhắc-lại-kết-quả-phần-i-và-mục-tiêu-phần-ii)  
2. [Neural codec với hyperprior và entropy model nâng cao](#2-neural-codec-với-hyperprior-và-entropy-model-nâng-cao)  
   2.1. [Kiến trúc Ballé 2018: scale hyperprior](#21-kiến-trúc-ballé-2018-scale-hyperprior)  
   2.2. [Entropy model học được và arithmetic coding](#22-entropy-model-học-được-và-arithmetic-coding)  
   2.3. [Gaussian, Laplace, logistic: chọn distribution nào?](#23-gaussian-laplace-logistic-chọn-distribution-nào)  
   2.4. [Analytic gradient cho loss hyperprior](#24-analytic-gradient-cho-loss-hyperprior)  
   2.5. [Hierarchical + autoregressive kết hợp](#25-hierarchical--autoregressive-kết-hợp)  
3. [Sea-Former, LiT-Codec và các neural codec mới](#3-sea-former-lit-codec-và-các-neural-codec-mới)  
   3.1. [Sea-Former: Self-Encoding Attention](#31-sea-former-self-encoding-attention)  
   3.2. [LiT-Codec: Token hóa 64 mã 16-bit](#32-lit-codec-token-hóa-64-mã-16-bit)  
   3.3. [NeRV-Codec và streaming video tokens](#33-nerv-codec-và-streaming-video-tokens)  
   3.4. [So sánh hiệu năng: bảng bpp vs PSNR](#34-so-sánh-hiệu-năng-bảng-bpp-vs-psnr)  
4. [Perceiver Resampler và các module thu gọn token](#4-perceiver-resampler-và-các-module-thu-gọn-token)  
   4.1. [Cấu trúc Perceiver: cross-attention dạng latent](#41-cấu-trúc-perceiver-cross-attention-dạng-latent)  
   4.2. [Phân tích độ phức tạp O($NM$)](#42-phân-tích-độ-phức-tạp-onm)  
   4.3. [TokenLearner, TokenFusion, TokenMerging](#43-tokenlearner-tokenfusion-tokenmerging)  
   4.4. [Adaptive token budget và bài toán tối ưu](#44-adaptive-token-budget-và-bài-toán-tối-ưu)  
5. [PaLM-E, Kosmos-2, IDEFICS-2, Gemini: bóc tách kiến trúc](#5-palme-kosmos-2-idefics-2-gemini-bóc-tách-kiến-trúc)  
   5.1. [PaLM-E: ViT + Perceiver Resampler + PaLM](#51-palme-vit--perceiver-resampler--palm)  
   5.2. [Kosmos-2: Modality adapter và grounding tag](#52-kosmos-2-modality-adapter-và-grounding-tag)  
   5.3. [IDEFICS-2: Multi-scale tokenizer + LLaMA-3.1](#53-idefics-2-multi-scale-tokenizer--llama-31)  
   5.4. [Gemini: RLHF đa phương thức và toolformer](#54-gemini-rlhf-đa-phương-thức-và-toolformer)  
   5.5. [So sánh kiến trúc: bảng tổng hợp chi tiết](#55-so-sánh-kiến-trúc-bảng-tổng-hợp-chi-tiết)  
6. [Chiến lược gắn token ảnh vào LLM – giải thích sâu](#6-chiến-lược-gắn-token-ảnh-vào-llm-–-giải-thích-sâu)  
   6.1. [Prefix linear: ưu nhược điểm và tối ưu hóa](#61-prefix-linear-ưu-nhược-điểm-và-tối-ưu-hóa)  
   6.2. [Interleaving pointer: thiết kế attention mask](#62-interleaving-pointer-thiết-kế-attention-mask)  
   6.3. [Gated fusion (Flamingo-style) và KV adapter](#63-gated-fusion-flamingo-style-và-kv-adapter)  
   6.4. [Hybrid strategies và routing network](#64-hybrid-strategies-và-routing-network)  
   6.5. [Bài toán độ dài context: phân tích chi phí](#65-bài-toán-độ-dài-context-phân-tích-chi-phí)  
7. [Prompt template, tokenizer mapping và logging chi tiết](#7-prompt-template-tokenizer-mapping-và-logging-chi-tiết)  
8. [Pipeline huấn luyện đa giai đoạn](#8-pipeline-huấn-luyện-đa-giai-đoạn)  
   8.1. [Stage 0: Warm-up và alignment latent](#81-stage-0-warm-up-và-alignment-latent)  
   8.2. [Stage 1: Instruction tuning với CoT](#82-stage-1-instruction-tuning-với-cot)  
   8.3. [Stage 2: RLHF đa phương thức](#83-stage-2-rlhf-đa-phương-thức)  
   8.4. [Stage 3: Self-consistency và distillation](#84-stage-3-self-consistency-và-distillation)  
   8.5. [Stage 4: Post-training optimization](#85-stage-4-post-training-optimization)  
9. [Ví dụ code hoàn chỉnh: từ token tới prompt builder](#9-ví-dụ-code-hoàn-chỉnh-từ-token-tới-prompt-builder)  
   9.1. [Module MappingTokenToText](#91-module-mappingtokentotext)  
   9.2. [Prompt template nhiều ảnh và nhiều câu hỏi](#92-prompt-template-nhiều-ảnh-và-nhiều-câu-hỏi)  
   9.3. [Huấn luyện LoRA cho LLaMA-3 với token ảnh](#93-huấn-luyện-lora-cho-llama-3-với-token-ảnh)  
10. [Đánh giá và benchmark đa chiều](#10-đánh-giá-và-benchmark-đa-chiều)  
   10.1. [Dataset chuẩn: ScienceQA, ChartQA, MMMU, VizWiz](#101-dataset-chuẩn-scienceqa-chartqa-mmmu-vizwiz)  
   10.2. [Thiết kế bảng benchmark: format và lưu trữ](#102-thiết-kế-bảng-benchmark-format-và-lưu-trữ)  
   10.3. [Phân tích lỗi: taxonomy và quy trình](#103-phân-tích-lỗi-taxonomy-và-quy-trình)  
   10.4. [Phương pháp đánh giá subjective](#104-phương-pháp-đánh-giá-subjective)  
11. [Hướng nghiên cứu mở](#11-hướng-nghiên-cứu-mở)  
12. [Checklist thực thi trong môi trường sản phẩm](#12-checklist-thực-thi-trong-môi-trường-sản-phẩm)  
13. [Kết luận chung của bộ đôi bài viết](#13-kết-luận-chung-của-bộ-đôi-bài-viết)  
14. [Phụ lục A: Bảng ký hiệu mới](#14-phụ-lục-a-bảng-ký-hiệu-mới)  
15. [Phụ lục B: Công thức và chứng minh bổ sung](#15-phụ-lục-b-công-thức-và-chứng-minh-bổ-sung)  
16. [Tài liệu tham khảo](#16-tài-liệu-tham-khảo)

---

## 1. Nhắc lại kết quả phần I và mục tiêu phần II

- Phần I đã phát triển toàn bộ nền tảng:
  - Information theory (entropy, rate–distortion).
  - Vector quantization, VQ-VAE, VQ-GAN, MaskGIT.
  - Pipeline encoder–decoder, EMA, metric đánh giá.
  - Code VQ-VAE từ đầu, checklist thực nghiệm.
- Phần II tiếp tục với các chủ đề nâng cao mà đội kỹ thuật và cô hướng dẫn viên phải xử lý khi triển khai thực tế:
  1. Neural codec sử dụng hyperprior và entropy model để đạt bitrate thấp hơn.
  2. Các kiến trúc state-of-the-art cho token hóa ảnh (Sea-Former, LiT-Codec).
  3. Cách tích hợp token với LLM lớn (PaLM-E, Kosmos-2, IDEFICS-2, Gemini).
  4. Chiến lược prompt, attention mask, KV cache, memory layout.
  5. Training pipeline đa giai đoạn (warm-up, instruction tuning, RLHF, self-consistency).
  6. Đánh giá, benchmark, hướng nghiên cứu tương lai.

---

## 2. Neural codec với hyperprior và entropy model nâng cao

### 2.1. Kiến trúc Ballé 2018: scale hyperprior

- Ballé et al. (2018) đề xuất mô hình có hai nhánh:
  - **Main encoder** $g_a$: $y = g_a(x)$.
  - **Hyper encoder** $h_a$: $z = h_a\!\left(\left| y \right|\right)$.
  - **Hyper decoder** $h_s$: dự đoán tham số scale $\sigma$ cho phân phối của $y$.
  - **Main decoder** $g_s$: tái tạo ảnh $\hat{x} = g_s(\hat{y})$.
- Loss:
  $$
  \mathcal{L} = \mathbb{E}\left[-\log p_{\hat{y} \mid \hat{z}}(\hat{y} \mid \hat{z})\right] + \mathbb{E}\left[-\log p_{\hat{z}}(\hat{z})\right] + \lambda \mathbb{E}[d(x, \hat{x})].
  $$
- Ý nghĩa:
  - $p_{\hat{y} \mid \hat{z}}$ mô hình hoá entropy mã chính.
  - $p_{\hat{z}}$ mô hình hoá entropy của hyper latent.

### 2.2. Entropy model học được và arithmetic coding

- Để chuyển xác suất thành bitstream, ta dùng arithmetic coding (hoặc range coding).
- Pipeline:
  1. Từ $z$, dự đoán $\mu_y$, $\sigma_y$.
  2. Xem $y$ theo phân phối $p\left(y \mid \mu_y, \sigma_y\right)$ (thường Gaussian).
  3. Lượng tử hoá $y$ → $\hat{y}$, encode chênh lệch bằng probability mass function.
  4. Sử dụng arithmetic coder để viết bitstream.
- Arithmetic coding đảm bảo số bit gần nhất với $-\log_2 p(\hat{y})$.
- Trong training, ta thay rounding bằng \( \hat{y} = y + u, u \sim \mathcal{U}(-0.5, 0.5) \) để gradient trơn.

### 2.3. Gaussian, Laplace, logistic: chọn distribution nào?

- **Gaussian**: $p(x \mid \mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$.
  - Thích hợp khi residual gần zero mean.
- **Laplace**: tail dày hơn, phù hợp khi residual có phân phối heavy-tail.
- **Logistic**: convenient vì CDF có dạng sigmoid; logistic mixture (HiFiC) cho flexibility.
- Thực nghiệm: logistic mixture (K=5) cho PSNR cao nhất nhưng training chậm.
- Lựa chọn distribution ảnh hưởng directly đến rate (qua $-\log p$).

### 2.4. Analytic gradient cho loss hyperprior

- Loss term $-\log p_{\hat{y} \mid \hat{z}}$:
  $$
  -\log p_{\hat{y} \mid \hat{z}}(\hat{y} \mid \hat{z}) = \frac{(\hat{y}-\mu(\hat{z}))^2}{2\sigma^2(\hat{z})} + \log \sigma(\hat{z}) + C.
  $$
- Gradient wrt $\mu$: $\frac{\hat{y}-\mu}{\sigma^2}$.
- Gradient wrt $\sigma$: $-\frac{(\hat{y}-\mu)^2}{\sigma^3} + \frac{1}{\sigma}$.
- Trong implementation, output log-scale $s = \log \sigma$ để ổn định: $\sigma = \exp(s)$.

### 2.5. Hierarchical + autoregressive kết hợp

- Entropy model có thể kết hợp hyperprior + autoregressive (Mentzer 2020):
  - Hyperprior dự đoán $\mu,\sigma$ global.
  - Autoregressive (masked conv) dự đoán phần residual local.
  - Probability:
    $$
    p(\hat{y}|\hat{z}) = \prod_i p(\hat{y}_i | \mu_i(\hat{z}, \hat{y}_{<i}), \sigma_i(\hat{z}, \hat{y}_{<i})).
    $$
- Trade-off: autoregressive accurate nhưng inference chậm (sequential). Thường dùng cho offline compression (không realtime).

### 2.6. Triển khai arithmetic coding trong thực tế

- **Tính PMF từ entropy model**: từ $\mu_i, \sigma_i$ tính CDF logistic hoặc Gaussian để xử lý rounding.
- **Range coder**:
  - Duy trì `low`, `high`, `value`.
  - Cập nhật theo cumulative probability `c(i)`:
    ```
    range = high - low
    high = low + range * c(i+1)
    low = low + range * c(i)
    ```
  - Khi range nhỏ, shift bit ra output để tránh underflow.
- **Decoder**: dùng `value` để xác định subinterval chứa, giải mã ngược.
- **Precision**: dùng số nguyên 32-bit hoặc 64-bit; floating point dễ lỗi.
- **Thư viện**: `compressai`, `torchac` hỗ trợ logistic mixture; cần convert sang CUDA khi inference thời gian thực.

### 2.7. Rounding vs noise trong training

- Training dùng surrogate `\hat{y} = y + u, u \sim \mathcal{U}(-0.5, 0.5)` để gradient trơn.
- Inference dùng `\hat{y} = \text{round}(y)`.
- Chênh lệch distribution (mismatch) có thể gây artefact; giải pháp:
  1. Anneal noise (giảm biên độ dần về 0).
  2. Sử dụng straight-through rounding (như STE).
  3. Thêm regularizer encourage $y$ nằm gần center của bin.

### 2.8. Multi-rate codec và condition bằng bitrate

- Để phục vụ nhiều mức bitrate, condition encoder bằng scalar $b$:
  $$
  y = g_a(x, b), \quad \hat{x} = g_s(\hat{y}, b).
  $$
- Loss:
  $$
  \mathcal{L} = \mathbb{E}[-\log p(\hat{y}|b)] + \lambda(b) \mathbb{E}[d(x,\hat{x})].
  $$
- Trong inference, chọn $b$ dựa trên context limit LLM → dynamic bitrate.

---

## 3. Sea-Former, LiT-Codec và các neural codec mới

### 3.1. Sea-Former: Self-Encoding Attention

- Sea-Former (Yu et al., 2023) đề xuất kiến trúc Transformer thuần cho codec.
- Encoder gồm:
  1. Patch embedding.
  2. Self-attention block (multi-head).
  3. Downsample attention (pooling).
- Resampler layer giảm token (tương tự Perceiver) → 16× compression.
- Decoder dùng cross-attention để tái tạo pixel.
- Loss: rate–distortion + perceptual.
- Sea-Former cho thấy Transformer có thể thay thế CNN trong codec.

### 3.2. LiT-Codec: Token hóa 64 mã 16-bit

- LiT-Codec (Yang et al., 2024) hướng tới VLM:
  - Input 512².
  - Output sequence 64 token, mỗi token 16-bit (chỉ số 0–65535).
  - Dùng hierarchical VQ + entropy coding để đảm bảo fidelity.
  - Kết hợp text alignment loss (CLIP) ngay trong training.
- Kết quả: PSNR 33 dB ở 0.08 bpp, CLIPScore cao hơn VQ-GAN.
- LiT-Codec cung cấp API mapping token → `<lit_i>` cho LLM.

### 3.3. NeRV-Codec và streaming video tokens

- NeRV-Codec (2024) encode video sequence:
  - Neural ODE + recurrent quantization.
  - Tạo token stream 25fps; mỗi frame ~80 token.
  - Sử dụng attention along time để giữ consistency.
- Ứng dụng: livestream tích hợp vào agent (ví dụ: hướng dẫn viên ảo).

### 3.4. So sánh hiệu năng: bảng bpp vs PSNR

| Codec | Bpp | PSNR | LPIPS | Token/frame | Ghi chú |
|-------|-----|------|-------|--------------|---------|
| VQ-GAN | 0.48 | 30.5 | 0.27 | 256 | Nhanh, open-source |
| Sea-Former | 0.25 | 31.8 | 0.24 | 128 | Transformer |
| LiT-Codec | 0.08 | 33.0 | 0.21 | 64 | Hướng tới VLM |
| HiFiC | 0.14 | 33.7 | 0.20 | continuous | Không token rời rạc |
| NeRV-Codec (video) | 0.12 | 32.1 | 0.25 | 80 | Streaming |

- Khi chọn codec cho VLM, cần cân bằng: PSNR cao, token ít, integration dễ.

### 3.5. Kết quả ablation từ LiT-Codec

- Bỏ CLIP alignment loss → CLIPScore giảm ~3 điểm, BLEU giảm 4%.
- Tăng token từ 64 lên 128 cải thiện PSNR +0.7 dB nhưng chi phí context gấp 2.
- Logistic mixture 5 thành phần tốt hơn 3 thành phần (LPIPS giảm 0.02).
- Học codebook multi-stage (pretrain VQ-VAE → fine-tune LiT) giúp convergence nhanh.

### 3.6. Tiền xử lý ảnh cho codec

- Normalize về [-1, 1].
- Áp dụng color space YUV, nén mạnh kênh U,V (ít nhạy cảm).
- HDR → áp dụng tone mapping.

### 3.7. Định dạng lưu token

- JSONL: dễ debug, mỗi dòng {id, tokens}.
- Binary: dùng `struct.pack('H', idx)` để lưu uint16; prefix header (version, length).
- Parquet: lưu cột tokens (list<int16>), question, answer.
- Đính kèm CRC32 để detect corruption.

---

## 4. Perceiver Resampler và các module thu gọn token

### 4.1. Cấu trúc Perceiver: cross-attention dạng latent

- Perceiver (Jaegle et al.) đưa ra ý tưởng latent array $Z \in \mathbb{R}^{M \times d}$ (M nhỏ).
- Cross-attention:
  $$
  Z' = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V,
  $$
  trong đó:
  - $Q = Z W_Q$.
  - $K = X W_K$, $V = X W_V$ với $X$ là input tokens (viền patch).
- Perceiver Resampler (PaLM-E) lặp cross-attention nhiều lần, interleave feed-forward.
- Ưu điểm: complexity $O(NM)$ thay vì $O(N^2)$.

### 4.2. Phân tích độ phức tạp O($NM$)

- Giả sử input $N = 196$ patch, latent $M = 16$.
- Matrix multiply:
  - $QK^\top$: $M \times d$ nhân $d \times N$ → $O(MNd)$.
  - Softmax + V multiply: $O(MNd)$.
- Khi $M \ll N$, chi phí thấp.
- PaLM-E dùng $M=16$ → 16 token output, context friendly.

### 4.3. TokenLearner, TokenFusion, TokenMerging

- **TokenLearner** (Ryoo et al., 2021):
  - Tạo mask attention cho patch quan trọng.
  - Output 8–16 token adaptive.
- **TokenFusion** (Meta 2023):
  - Hợp nhất token dựa vào similarity.
- **TokenMerging** (Bolya et al., 2023):
  - Merge token liền kề trong ViT inference để tăng tốc.
- Các kỹ thuật này có thể kết hợp vector quantization: reduce token trước khi map sang codebook.

### 4.4. Adaptive token budget và bài toán tối ưu

- Bài toán: cho context limit $C$, text token $L$, cần chọn $M$ token ảnh sao cho $M + L \le C$ và maximize fidelity.
- Có thể mô hình thành optimization:
  $$
  \max_M \; \text{Quality}(M) \quad \text{s.t.} \quad M + L \le C, \; M \in \mathbb{Z}_+.
  $$
- Quality(M) có thể fit bằng logistic: $Q(M) = a - \frac{b}{M + c}$.
- Giải: $M^\* = \min(C - L, \text{argmax}_{M} Q(M))$.
- Thực tế: Precompute mapping context→token budget và embed vào prompt builder.

### 4.5. Liên hệ với attention sparsity và FlashAttention

- Khi token ảnh giảm, attention matrix sparse; tận dụng FlashAttention 2 để compute block-sparse.
- Thiết lập mask block 64×64 cho token ảnh, text full.
- Benchmark: FlashAttention giảm latency 30% so với attention chuẩn.

### 4.6. End-to-end joint training encoder + Perceiver

- Loss kết hợp:
  $$
  \mathcal{L} = \mathcal{L}_{\text{RD}} + \alpha \mathcal{L}_{\text{VLM}} + \beta \mathcal{L}_{\text{coverage}}.
  $$
- $\mathcal{L}_{\text{coverage}}$ đảm bảo token gia tăng khi task yêu cầu (ví dụ counting).
- Gradient backprop qua Perceiver; cần gradient checkpoint để tiết kiệm VRAM.

---

## 5. PaLM-E, Kosmos-2, IDEFICS-2, Gemini: bóc tách kiến trúc

### 5.1. PaLM-E: ViT + Perceiver Resampler + PaLM

- Input: ảnh + text + sensor.
- ViT-L/16 → 196 patch token.
- Perceiver Resampler → 16 latent token.
- Linear projection → map sang embedding PaLM (d=3072).
- Token ghép vào prompt (decoder-only).
- Training:
  - Pretrain PaLM 540B.
  - Fine-tune end-to-end với dataset robotics + web image-text.
  - Loss = cross-entropy token sequence.
- Lợi ích: few-shot reasoning, embodied tasks.

### 5.2. Kosmos-2: Modality adapter và grounding tag

- Input: patch embedding + bounding box.
- Modality adapter = MLP + LayerNorm → unify dims.
- Vision tokens + text tokens fed vào decoder Transformer.
- Grounding tags `<loc=...>` gán cho token chứa tọa độ bounding box.
- Training: mixture dataset (COCO Caption, VQA, OCR).
- Output: text + bounding boxes (multi-head).

### 5.3. IDEFICS-2: Multi-scale tokenizer + LLaMA-3.1

- Tokenizer 2 tầng (8×, 16×) → 64 hoặc 256 token.
- Text encoder = LLaMA-3.1 8B/80B.
- Fusion via cross-attention adapter (Q-former-like).
- Instruction tuning: dataset curated (ShareGPT4V, MMMU, DocVQA).
- Open-source weights, kèm script mapping token `<image_patch_i>`.

### 5.4. Gemini: RLHF đa phương thức và toolformer

- Gemini Ultra: backbone mixture-of-experts (MoE) + vision tower.
- Vision tower = ViT-G patch 256 token.
- Toolformer module: khi cần, model output `<API:python>` + code -> exec -> observation.
- RLHF pipeline:
  - Collect multimodal SFT dataset.
  - Train reward model (image + text).
  - PPO with KL penalty to SFT policy.
- Gemini log tokens ẩn; tuy nhiên paper mô tả token budget dynamic (32–512).

### 5.5. So sánh kiến trúc: bảng tổng hợp chi tiết

| Model | Tokenizer | #Token/ảnh | Integration | Training scale | Open? |
|-------|-----------|------------|-------------|----------------|-------|
| PaLM-E | Perceiver Resampler | 16 | Prefix (decoder) | 562B + robotics | Closed |
| Kosmos-2 | Modality adapter + bounding tags | 256 | Interleaved | 15B | Closed weights (2023) |
| IDEFICS-2 | Multi-scale VQ | 64/256 | Q-former adapter | 8B/80B | Open |
| Gemini | ViT-G + MoE | up to 512 | Hybrid + toolformer | >1T | Closed |
| LLaVA-1.5 | CLIP ViT | 576 | Linear projection | 13B | Open |

- Nhận xét:
  - Ít token (PaLM-E) -> context nhẹ nhưng cần module Perceiver.
  - 64 token (IDEFICS-2) là điểm cân bằng open-source.
  - 576 token (LLaVA) dễ training nhưng nặng context.

---

## 6. Chiến lược gắn token ảnh vào LLM – giải thích sâu

### 6.1. Prefix linear: ưu nhược điểm và tối ưu hóa

- Prompt dạng:
  ```
  <image> <v1> <v2> ... <vM> </image>
  Câu hỏi: ...
  ```
- Ưu điểm: implement dễ, không cần sửa kiến trúc.
- Nhược điểm: attention matrix $(M+L)^2$.
- Tối ưu:
  - Chunk token ảnh thành block 32, sinh summary token (mean pooling).
  - Dùng RoPE (rotary positional embedding) tách domain: assign position negative index cho ảnh để giữ text alignment.
  - Giảm precision: store token embedding FP16.

### 6.2. Interleaving pointer: thiết kế attention mask

- Prompt:
  ```
  Câu 1 về <IMG:1> ...
  <IMG_START id=1> ...tokens... <IMG_END>
  ```
- Cần mask cho text trước khi tokens:
  - Token text chỉ attend backward.
  - Token ảnh có thể attend lẫn nhau và text trước.
- Implementation:
  - Build `attention_mask`  (N+M)x(N+M).
  - For `i` in image tokens, allow attend to tokens in same block.
- Lợi ích: context matching; LLM biết token nào gắn với câu hỏi.

### 6.3. Gated fusion (Flamingo-style) và KV adapter

- Flamingo chèn Gated Cross-Attention (GCA) layer:
  $$
  T_t = U_t + \sigma(W_g[U_t; C_t]) \odot C_t,
  $$
  - $U_t$: hidden từ LM layer.
  - $C_t$: cross-attention output với vision token.
- Triển khai:
  - Thêm module adapter giữa layer LM.
  - Vision token encode sẵn (CLIP).
- Ưu điểm: vision injection selective, token ảnh không chiếm context.

### 6.4. Hybrid strategies và routing network

- Kết hợp prefix + gating:
  - Prefix 64 token (global context).
  - Mining top-k region -> gating injection.
- Routing network:
  - Policy network chọn strategy (prefix/adapter) dựa input.
  - RL-based: reward = accuracy - latency cost.

### 6.5. Bài toán độ dài context: phân tích chi phí

- Với LLM 8k token, text 2k, ta còn 6k cho ảnh.
- Mỗi ảnh 256 token ⇒ 23 ảnh.
- Chi phí attention (per layer):
  $T_{\text{attn}} \approx 4d(M+L)^2/\text{FLOPS}$.
- Ví dụ: LLaMA-3 8B (d=4096), $M+L=4096$ ⇒ 0.27ms/layer.
- 32 layer → 8.6ms; inference 1.2s (kể cả MLP, kv cache).
- Kết luận: prefix large tokens chậm; gating + latent (M=64) tối ưu hơn.

### 6.6. Positional encoding cho token ảnh

- Có ba lựa chọn phổ biến:
  1. **RoPE chung với text**: dùng offset âm để tránh overlap.
  2. **2D sinusoidal**: encode hàng, cột, sau đó cộng vector.
  3. **Learned absolute embedding**: trainable, cho phép LLM tự học mapping.
- PaLM-E dùng embedding random trainable; IDEFICS-2 dùng 2D sine-cosine.

### 6.7. Chuẩn hóa embedding

- Trước khi ghép vào LLM, apply LayerNorm: `tokens = LayerNorm(tokens)`.
- Hoặc align mean/variance:
  $$
  \tilde{e} = \frac{e - \mu_{\text{text}}}{\sigma_{\text{text}}}.
  $$
- Tránh distribution shift khiến LLM output không ổn định.

### 6.8. Chiến lược cache ảnh dài hạn

- Sau khi trả lời xong, thay token ảnh bằng summary text "[Ảnh 1: bình gốm xanh]".
- Lưu embedding summary -> tiết kiệm context cho conversation dài.

---

## 7. Prompt template, tokenizer mapping và logging chi tiết

- Mapping token ID → string:
  - Tạo file `tokens_vocab.json`: `{ "0": "<v0>", "1": "<v1>", ... }`.
  - Giữ consistent trong tokenizer LLaMA.
- Prompt builder:
  1. Load template YAML:
     ```yaml
     system: |
       Bạn là hướng dẫn viên bảo tàng. Sử dụng thông tin hình ảnh để trả lời.
     user: |
       <image id={img_id}>
       {image_tokens}
       </image>
      {question}
    ```
  2. Render template cho mỗi ảnh.
- Logging:
  - Lưu prompt final (đã chứa token) + response.
  - Lưu mapping token → patch (coordinate).
- Security:
  - Sanitize: token `<v123>` không chứa ký tự lạ.

### 7.1. Chuẩn hóa ID ảnh và quản lý phiên

- Đặt quy ước `image_id = {collection}_{index}` ví dụ `ancient_vase_042`.
- Khi phiên đối thoại kéo dài, gắn `session_id` để truy trace.
- Lưu metadata:
  ```json
  {
    "session": "2025-04-10T12:04Z",
    "image_id": "ancient_vase_042",
    "token_count": 64,
    "prompt_hash": "a3f0..."
  }
  ```

### 7.2. Kiểm tra sức khỏe mapping trước inference

- Script `validate_tokens.py`:
  - Đọc vocab.
  - Kiểm thử `n` token random -> ensure mapping exist.
  - Báo cáo codeword thiếu (dead id).

### 7.3. Biểu diễn token trong UI

- Frontend highlight patch khi hover token `<v123>`.
- Tạo overlay color-coded theo ID (dựa JSON mapping).
- Giúp curator phê duyệt câu trả lời nhanh chóng.

---

## 8. Pipeline huấn luyện đa giai đoạn

### 8.1. Stage 0: Warm-up và alignment latent

- Mục tiêu: align encoder với LLM embedding.
- Bước:
  1. Freeze LLM.
  2. Train adapter linear mapping `E_{img} -> EmbeddingSpace`.
  3. Loss = cosine distance + small cross-entropy (matching simple caption).
- Kết quả: LLM không bị shock khi thấy token ảnh.

### 8.2. Stage 1: Instruction tuning với CoT

- Dataset: curated QA (ScienceQA, ChartQA, nội bộ).
- Prompt:
  ```
  [SYSTEM] ...
  [USER] <image> ... </image> câu hỏi
  [ASSISTANT] [THOUGHT] ... [ANSWER] ...
  ```
- Huấn luyện LLM (LoRA hoặc full fine-tune):
  - Learning rate 1e-4 cho LoRA, 1e-5 cho full.
  - Gradient checkpointing để tiết kiệm VRAM.
- Ensure CoT token (Thought, Action) consistent.

### 8.2.1. Cấu trúc dữ liệu huấn luyện

- JSONL entry:
  ```json
  {
    "image_tokens": ["<v45>", "<v102>", "..."],
    "question": "Màu nền của bức tranh là gì?",
    "thought": [
      "Kiểm tra token liên quan đến nền",
      "So sánh histogram màu"
    ],
    "answer": "Nền màu xanh dương."
  }
  ```
- Preprocess: nối `thought` thành chuỗi `[THOUGHT] ...`.

### 8.2.2. Loss masking

- Nếu muốn giấu reasoning, mask loss trên token `[THOUGHT]` (gán -100).
- Cũng có thể training hai chế độ: open-COT và hidden-COT.

### 8.2.3. Curriculum learning

- Bắt đầu training với câu hỏi đơn giản (caption).
- Dần thêm câu hỏi multi-hop (so sánh 2 ảnh).
- Theo dõi accuracy theo level để điều chỉnh dataset.

### 8.3. Stage 2: RLHF đa phương thức

- Components:
  1. **Reward model** $R_\phi$ nhận (image tokens, question, answer).
  2. **Preference data**: pair (answer_good, answer_bad).
  3. Train $R_\phi$ bằng logistic loss:
     $$
     \mathcal{L}_R = -\log \sigma(R_\phi(x, y_{\text{good}}) - R_\phi(x, y_{\text{bad}})).
     $$
- RL (PPO):
  - Policy = LLM.
  - Reward = $R_\phi$ + coverage penalty (cite patch).
  - KL penalty giữ policy gần SFT.
- Consider multi-sample RL: roll-out 4 answer, pick best.

### 8.3.1. Xây dựng tập preference

- Quy trình:
  1. Sinh 3 đáp án bằng policy hiện tại.
  2. Annotator chọn best/worst dựa guideline:
     - Đúng nội dung.
     - Lý luận rõ ràng.
     - Trích dẫn token chính xác.
  3. Lưu pair `(good, bad)`.
- Có thể nhờ GPT-4V làm annotator sơ bộ rồi người thật kiểm tra 20%.

### 8.3.2. Reward shaping

- Reward tổng:
  $$
  r = R_\phi + \alpha \cdot \text{CLIPScore} + \beta \cdot \text{CitationPrecision}.
  $$
- `CitationPrecision` = số cite đúng / tổng cite.
- Điều chỉnh $\alpha, \beta$ qua grid search (0.2, 0.5).

### 8.3.3. Lịch KL penalty

- Bắt đầu $\beta_{\text{KL}} = 0.1$.
- Nếu KL tăng > 0.2, nâng 0.15 để policy không drift.
- Log KL per batch để theo dõi.

### 8.4. Stage 3: Self-consistency và distillation

- Generate K reasoning chain per input (K=5).
- Majority vote answer.
- Distill: train student model minimize KL giữa distribution teacher vs student.
- Giữ token ảnh fixed; iterate until variance low.

#### 8.4.1. Voting heuristic

- Tính điểm từng đáp án bằng reward model.
- Nếu top-1 và top-2 chênh < 0.05, giữ cả hai làm target mềm.
- Khi variance quá cao, quay lại Stage 2 để tinh chỉnh reward model.

#### 8.4.2. Distillation loss chi tiết

- Student minimize:
  $$
  \mathcal{L}_{\text{distill}} = \sum_i q_i \log \frac{q_i}{p_i},
  $$
  với $q_i$ distribution teacher (temperature $T=2$), $p_i$ student.
- Có thể thêm MSE cho hidden state nếu muốn student học representation.

### 8.5. Stage 4: Post-training optimization

- Techniques:
  - Quantization aware training (QAT) cho adapter.
  - KV cache compression (FP8, INT4).
  - Prompt caching: precompute embedding cho token ảnh static.
  - Evaluate throughput, latency.

#### 8.5.1. KV cache compression chi tiết

- Dùng `autoawq` hoặc `bitsandbytes` để quantize KV xuống INT4/FP8.
- Giảm memory ~40% nhưng giữ accuracy (testing ScienceQA ±0.3%).
- Cần quantize riêng adapter LoRA để tránh suy giảm.

#### 8.5.2. Prompt caching mechanism

- Với ảnh static (catalog), precompute embedding:
  - `cache[id] = model.encode_image(tokens)`.
  - Khi build prompt, chỉ ghép embedding (không re-encode).
- Đo lường: latency giảm từ 1.3s → 0.8s cho 10 ảnh.

---

## 9. Ví dụ code hoàn chỉnh: từ token tới prompt builder

### 9.1. Module MappingTokenToText

```python
class TokenMapper:
    def __init__(self, vocab_path: str, max_image_tokens: int = 512):
        import json
        with open(vocab_path) as f:
            mapping = json.load(f)
        self.id_to_token = {int(k): v for k, v in mapping.items()}
        self.max_image_tokens = max_image_tokens

    def map_ids(self, token_ids):
        tokens = []
        for idx in token_ids:
            if idx not in self.id_to_token:
                raise ValueError(f"Unknown token id {idx}")
            tokens.append(self.id_to_token[idx])
        if len(tokens) > self.max_image_tokens:
            raise ValueError("Token length exceeds limit")
        return " ".join(tokens)
```

### 9.2. Prompt template nhiều ảnh và nhiều câu hỏi

```python
from jinja2 import Template

PROMPT_TEMPLATE = Template("""
<system>
Bạn là hướng dẫn viên AI của Bảo tàng Giao Thoa.
Khi trả lời, hãy lập luận rõ ràng và trích dẫn token ảnh tương ứng dưới dạng [CITE:<vID>].
</system>
{% for item in items %}
<user>
<image id="{{ item.img_id }}">
{{ item.image_tokens }}
</image>
{{ item.question }}
</user>
<assistant>
</assistant>
{% endfor %}
""")

def build_prompt(items):
    return PROMPT_TEMPLATE.render(items=items)
```

### 9.3. Huấn luyện LoRA cho LLaMA-3 với token ảnh

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)

def collate_fn(batch):
    prompts, responses = zip(*batch)
    batch_encoding = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer(responses, return_tensors="pt", padding=True, truncation=True).input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return {**batch_encoding, "labels": labels}
```

Tiếp tục với optimizer AdamW, learning rate 2e-4, gradient clipping, training 3 epoch.

### 9.4. Inference với beam search và cite patch

```python
def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=3,
        temperature=0.7,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

Sau khi nhận answer, parse `[CITE:<vID>]` để mapping ngược patch.

### 9.5. Evaluation pipeline tích hợp

```python
def evaluate_model(model, tokenizer, dataset, mapper, metric_fn):
    results = []
    for sample in dataset:
        token_str = mapper.map_ids(sample["image_tokens"])
        prompt = build_prompt([{
            "img_id": sample["image_id"],
            "image_tokens": token_str,
            "question": sample["question"]
        }])
        answer = generate_answer(model, tokenizer, prompt)
        metric = metric_fn(answer, sample["answer_gt"])
        results.append({
            "id": sample["id"],
            "answer": answer,
            "metric": metric
        })
    return results
```

### 9.6. Bộ công cụ logging và giám sát

- Dùng `structlog` hoặc `logging` Python:
  ```python
  logger.info(
      "inference",
      session=session_id,
      image=sample["image_id"],
      token_count=len(sample["image_tokens"]),
      latency_ms=latency * 1000,
      metric=metric
  )
  ```
- Push log lên Elasticsearch/Grafana để theo dõi thời gian thực.

### 9.7. Tích hợp với vector store

- Lưu embedding token (mean pooling block) vào FAISS.
- Cho phép truy xuất ảnh tương tự khi cần so sánh.
- Kết hợp metadata (nghệ sĩ, năm sáng tác) để filter.

### 9.8. So khớp token với metadata ngoài

- Gắn metadata trong JSON:
  ```json
  {
    "tokens": [...],
    "metadata": {
      "artist": "Nguyễn Gia Trí",
      "year": 1950,
      "collection": "Lacquer"
    }
  }
  ```
- Prompt builder có thể chèn metadata và token song song, tăng tính giải thích.

### 9.9. Batch inference và phân bổ tài nguyên

- Sử dụng `generate` với `return_dict_in_generate=True` để lấy attention weights.
- Batch 8 prompt, accumulate gradient (nếu fine-tune online).
- Cân bằng GPU/CPU: encode token trên GPU, LLM inference multi-GPU.

---

## 10. Đánh giá và benchmark đa chiều

### 10.1. Dataset chuẩn: ScienceQA, ChartQA, MMMU, VizWiz

- **ScienceQA**: 21k câu hỏi, gồm ảnh thí nghiệm, biểu đồ.
- **ChartQA**: 20k chart (bar, line), yêu cầu đọc số liệu.
- **MMMU**: multi-discipline, 30 domain (toán, y khoa, vật lý).
- **VizWiz**: ảnh chụp bởi người khiếm thị, chất lượng thấp.
- **DocVQA**: tài liệu scan, OCR heavy.
- Mỗi dataset có script convert sang prompt format (bao gồm token ảnh).

### 10.2. Thiết kế bảng benchmark: format và lưu trữ

- CSV schema:
  - `dataset`, `split`, `question`, `answer_gt`, `answer_model`, `tokens`, `metrics`.
- Metrics:
  - Accuracy, BLEU, Exact Match, CLIPScore.
  - Latency (ms), token count, memory usage.
- Lưu trữ: Parquet (efficient), versioning bằng DVC.

### 10.3. Phân tích lỗi: taxonomy và quy trình

- Taxonomy:
  1. **Perception error**: nhận biết đối tượng sai (token missing).
  2. **Localization error**: cite patch sai.
  3. **Reasoning error**: chain-of-thought sai logic.
  4. **Language error**: ngữ pháp, format.
- Quy trình:
  - Replay thought chain (log `[THOUGHT]`).
  - Đối chiếu patch map (token -> coordinate).
  - Ghi chú trong error report (YAML).

### 10.4. Phương pháp đánh giá subjective

- Mời giám khảo (curator bảo tàng) chấm 3 tiêu chí (0–5):
  - Độ chính xác nội dung.
  - Độ rõ ràng lời giải thích.
  - Mức độ trích dẫn chính xác.
- Lấy trung bình; so sánh variant model.

### 10.5. Bảng tóm tắt kết quả mẫu

| Model | Token strategy | ScienceQA Acc | ChartQA EM | MMMU score | Avg latency (s) |
|-------|----------------|---------------|------------|------------|-----------------|
| Baseline prefix | 256 token | 78.2 | 64.5 | 38.1 | 1.95 |
| Prefix + adapter | 128 token + gating | 81.4 | 67.2 | 41.5 | 1.32 |
| Hybrid routing | adaptive | 82.7 | 69.0 | 42.6 | 1.18 |

- Bảng minh hoạ cho lợi ích adaptive token.

### 10.6. Tự động hóa benchmark

- Viết script `run_benchmark.py` chạy sequential dataset → save Parquet.
- Scheduling qua Airflow mỗi tuần.

### 10.7. Phân tích thống kê kết quả

- Tính trung bình và độ lệch chuẩn trên nhiều seed.
- Thực hiện kiểm định t-test giữa model A,B để xác nhận khác biệt đáng kể.
- Báo cáo khoảng tin cậy 95%.

### 10.8. Visualization dashboard

- Vẽ scatter `CLIPScore vs latency`.
- Heatmap confusion matrix cho MMMU (domain x accuracy).
- Bar chart token usage theo dataset.

---

## 11. Hướng nghiên cứu mở

1. **Token co-design với LLM**: joint training encoder và LLM end-to-end thay vì pipeline rời.
2. **Adaptive bitrate**: tùy câu hỏi mà encode nhiều hay ít token.
3. **Cross-modal co-compression**: nén ảnh+âm thanh+text chung codebook.
4. **Continual learning**: cập nhật codebook khi thêm tác phẩm mới mà không quên kiến thức cũ.
5. **Privacy**: mã hoá token bằng homomorphic encryption hoặc secure enclave.
6. **Explainable token**: token kèm metadata semantics (ví dụ: “bình gốm”, “cửa sổ”) để LLM reasoning rõ ràng.
7. **Federated compression**: huấn luyện encoder trên nhiều bảo tàng mà không chia sẻ dữ liệu raw.
8. **Differential privacy**: thêm noise có kiểm soát để bảo vệ chi tiết nhạy cảm.
9. **Hardware-aware neural codec**: thiết kế encoder tối ưu cho chip Edge TPU / mobile GPU.
10. **Evaluation framework chuẩn hoá**: cộng đồng cần benchmark chung cho token→LLM.

---

## 12. Checklist thực thi trong môi trường sản phẩm

1. **Thẩm định pháp lý**: kiểm tra bản quyền tác phẩm trước khi lưu token.
2. **Monitoring**:
   - TPS encode, latency, error rate.
   - CLIPScore trending.
3. **Fallback plan**: khi token decode lỗi, fallback CLIP raw image + caption.
4. **Security**: token file có thể leak info; mã hoá at-rest, role-based access.
5. **Versioning**: gắn version cho codebook, tokenizer, LLM weights.
6. **A/B testing**: so sánh variant (prefix vs gating) dựa trên satisfaction score.
7. **Documentation**: update knowledge base cho nhân viên bảo tàng.
8. **Disaster recovery**: backup token store (đa vùng) để tránh mất dữ liệu.
9. **Latency SLA**: đặt mục tiêu (<2s) và alert khi vượt.
10. **User feedback loop**: ghi lại khi khách sửa thông tin -> feed vào training.

---

## 13. Kết luận chung của bộ đôi bài viết

- Bộ đôi bài viết (Phần I & II) cung cấp toàn bộ hành trình:
  - Nền tảng toán học (rate–distortion, vector quantization).
  - Kiến trúc VQ-VAE, VQ-GAN, MaskGIT.
  - Neural codec hyperprior, Perceiver Resampler, PaLM-E, Kosmos-2, IDEFICS-2, Gemini.
  - Chiến lược tích hợp token ảnh vào LLM, pipeline huấn luyện và benchmark.
- Với kiến thức này, cô hướng dẫn viên và đội kỹ thuật của bảo tàng có thể xây dựng hệ thống VLM nén ảnh thành token văn bản, phục vụ khách tham quan một cách hiệu quả, minh bạch và có thể kiểm chứng.
- Độc giả có thể quay lại [Phần I](/posts/2025/vlm-compression-token-integration-part1) để xem lại nền tảng, hoặc tiếp tục nghiên cứu các bài khác trong series VLM.

---

## 14. Phụ lục A: Bảng ký hiệu mới

- $y$: latent chính của codec.
- $z$: hyper latent.
- $\mu(\hat{z})$, $\sigma(\hat{z})$: tham số phân phối của $y$ điều kiện theo $\hat{z}$.
- $C$: context length tối đa LLM.
- $L$: số token text.
- $M$: số token ảnh.
- $R_\phi$: reward model.
- $P_\theta$: policy (LLM).
- $v_k$: vector ngẫu nhiên Hutchinson.
- $\beta$: hệ số penalty cite patch.
- $S$: số vòng self-consistency.

---

## 15. Phụ lục B: Công thức và chứng minh bổ sung

### 15.1. Phân phối logistic mixture cho entropy model

- PDF logistic:
  $$
  f(x|\mu, s) = \frac{\exp\left(-\frac{x-\mu}{s}\right)}{s \left(1 + \exp\left(-\frac{x-\mu}{s}\right)\right)^2}.
  $$
- Mixture với K thành phần:
  $$
  p(x) = \sum_{k=1}^K \pi_k f(x|\mu_k, s_k).
  $$
- CDF analytic, dễ dùng trong range coding.

### 15.2. KL divergence giữa distribution thực và model

- Với distribution thực $p$ và model $q$, bitrate thực tế:
  $$
  R = \mathbb{E}_p[-\log_2 q(x)] = H(p) + D_{\text{KL}}(p \| q).
  $$
- Mục tiêu entropy model: minimize KL ⇒ $q$ gần $p$.

### 15.3. Công thức gradient PPO với KL penalty

$$
L^{\text{PPO}} = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right) - \beta \text{KL}(P_\theta || P_{\text{ref}})\right].
$$

- $r_t(\theta) = \frac{P_\theta(a_t|s_t)}{P_{\theta_{\text{old}}}(a_t|s_t)}$.
- $\beta$ điều chỉnh khoảng cách tới policy tham chiếu (SFT).

### 15.4. Cơ chế toolformer trong Gemini

- Khi output `<API:python>`, LLM cung cấp code.
- Executor chạy code, trả `Observation`.
- Prompt update:
  ```
  [THOUGHT] cần tính diện tích
  [ACTION] <API:python>
  print(compute_area(...))
  </API>
  [OBSERVATION] 24.5
  ```
- Quy trình ReAct (Reason + Action) giúp reasoning phức tạp.

---

## 16. Tài liệu tham khảo

1. Ballé, J., Minnen, D., Singh, S., Hwang, S. J., & Johnston, N. (2018). *Variational Image Compression with a Scale Hyperprior.* ICLR.  
2. Mentzer, F., Agustsson, E., Tschannen, M., Timofte, R., & Gool, L. V. (2018). *Conditional Probability Models for Deep Image Compression.* CVPR.  
3. Yang, Y. et al. (2024). *LiT-Codec: Tokenizing Images for Large Multimodal Models.* arXiv.  
4. Yu, J. et al. (2023). *SeaFormer: Transformer-based Neural Image Codec.* arXiv.  
5. Jaegle, A. et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.  
6. Driess, D. et al. (2023). *PaLM-E: An Embodied Multimodal Language Model.* ICLR.  
7. Zhai, X. et al. (2023). *Kosmos-2: Grounding Multimodal Large Language Models to the World.* arXiv.  
8. Laurençon, H. et al. (2024). *IDEFICS-2: An Open Multimodal Generative and Instruction-Tuned Model Family.* arXiv.  
9. Reid, M. et al. (2024). *Gemini 1.0: Multimodal Foundation Models Built From the Ground Up.* arXiv.  
10. Bolya, D. et al. (2023). *Token Merging for Fast Stable Diffusion.* arXiv.  
11. Ryoo, M. et al. (2021). *TokenLearner: Adaptive Space-Time Tokenization for Videos.* NeurIPS.  
12. Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv.  
13. OpenAI (2023). *GPT-4 Technical Report.* arXiv.  
14. Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR.  
15. Liu, H. et al. (2023). *Visual Instruction Tuning.* arXiv.  
16. Masry, A. et al. (2022). *ChartQA: A Benchmark for Question Answering about Charts.* arXiv.  
17. Aroyo, L. et al. (2024). *MMMU: A Massive Multi-discipline Multimodal Understanding Benchmark.* NeurIPS Datasets.  
18. Dehaene, P. et al. (2024). *NeRV-Codec: Neural Video Representation with Multi-scale Tokens.* arXiv.  
19. Cheng, Z. et al. (2020). *Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules.* CVPR.  
20. Xu, Y. et al. (2024). *Self-Consistency Improves Chain of Thought Reasoning in Multimodal LLMs.* arXiv.

---

<script src="/assets/js/katex-init.js"></script>
