---
title: "Compression & Token Integration: Nén ảnh trực tiếp vào token text"
date: "2025-04-08"
category: "vision-language-models"
tags: ["vlm", "compression", "vq-vae", "vq-gan", "neural-codec", "palme", "kosmos2", "rate-distortion"]
excerpt: "Bài literature review chuyên sâu: cô hướng dẫn viên bảo tàng khảo sát toàn bộ chuỗi công trình về nén ảnh thành token để đưa vào LLM – từ Shannon rate–distortion tới PaLM-E, Kosmos-2, IDEFICS-2. Bài viết dài gần 700 dòng, giàu công thức, phân tích và ví dụ code."
author: "ThanhLamDev"
readingTime: 52
featured: false
---

# Compression & Token Integration: Nén ảnh trực tiếp vào token text

**Hành trình tại Bảo tàng Giao Thoa tiếp tục bước vào giai đoạn tối ưu hạ tầng. Sau khi đã token hóa văn bản (bài Foundations), căn chỉnh ảnh–text (Alignment) và nâng cấp reasoning (SOTA Reasoning), cô hướng dẫn viên phải giải bài toán rất đời thường: làm thế nào để mô hình chỉ cần đọc “chuỗi token” mà vẫn hiểu cả ảnh, âm thanh, biểu đồ? Lời giải nằm trong xu hướng nén ảnh thành token dạng text, cho phép tận dụng nguyên si hạ tầng LLM. Để chuẩn bị bản đề xuất lên ban giám đốc, cô dành nhiều ngày đọc gần 30 paper tại NeurIPS, CVPR, ICLR, ECCV và TPAMI, từ lý thuyết rate–distortion của Shannon cho tới PaLM-E, Kosmos-2, IDEFICS-2. Bài viết này ghi lại toàn bộ quá trình literature review đó, kèm phân tích toán học, giải thích công thức, code demo và checklist triển khai.**

## Mục lục

1. [Bối cảnh vận hành tại bảo tàng](#1-bối-cảnh-vận-hành-tại-bảo-tàng)  
2. [Timeline literature 2017 – 2024 và mapping SOTA](#2-timeline-literature-2017--2024-và-mapping-sota)  
3. [Cơ sở toán: rate–distortion, entropy và lượng tử hoá](#3-cơ-sở-toán-rate–distortion-entropy-và-lượng-tử-hoá)  
   3.1. [Định lý rate–distortion và ý nghĩa trong VLM](#31-định-lý-rate–distortion-và-ý-nghĩa-trong-vlm)  
   3.2. [Lagrangian cho nén học sâu](#32-lagrangian-cho-nén-học-sâu)  
   3.3. [Bits-back, entropy coding và liên hệ với tokenizer](#33-bits-back-entropy-coding-và-liên-hệ-với-tokenizer)  
   3.4. [Thước đo distortion: PSNR, LPIPS, FID, CLIPScore](#34-thước-đo-distortion-psnr-lpips-fid-clipsCore)  
4. [Vector quantization và các biến thể hiện đại](#4-vector-quantization-và-các-biến-thể-hiện-đại)  
   4.1. [Thuật toán Lloyd–Max và trọng tâm codebook](#41-thuật-toán-lloyd–max-và-trọng-tâm-codebook)  
   4.2. [Straight-through estimator, EMA và gradient surrogate](#42-straight-through-estimator-ema-và-gradient-surrogate)  
   4.3. [VQ-VAE: cơ chế loss, perplexity và độ hội tụ](#43-vq-vae-cơ-chế-loss-perplexity-và-độ-hội-tụ)  
   4.4. [dVAE/DALL·E: Gumbel-Softmax, relaxation và attention mask](#44-dvaedall·e-gumbel-softmax-relaxation-và-attention-mask)  
   4.5. [VQ-GAN và loss cảm nhận (perceptual + adversarial)](#45-vq-gan-và-loss-cảm-nhận-perceptual--adversarial)  
   4.6. [MaskGIT, token diffusion và nhiệt độ cập nhật](#46-maskgit-token-diffusion-và-nhiệt-độ-cập-nhật)  
   4.7. [Video token (ViT-VQ, VDT, MAGVIT2) và giới hạn băng thông](#47-video-token-vit-vq-vdt-magvit2-và-giới-hạn-băng-thông)  
5. [Neural codec, hyperprior và transformer-based compression](#5-neural-codec-hyperprior-và-transformer-based-compression)  
   5.1. [Hyperprior Ballé, entropy model và hồi quy Gaussian](#51-hyperprior-ballé-entropy-model-và-hồi-quy-gaussian)  
   5.2. [Autoregressive vs factorized: trade-off tốc độ ↔ chất lượng](#52-autoregressive-vs-factorized-trade-off-tốc-độ--chất-lượng)  
   5.3. [HiFiC, Entroformer, ViT-VQ: học trực tiếp trong domain perceptual](#53-hific-entroformer-vit-vq-học-trực-tiếp-trong-domain-perceptual)  
   5.4. [Perceiver Resampler và attention adapter (PaLM-E)](#54-perceiver-resampler-và-attention-adapter-palme)  
   5.5. [Token multi-scale trong IDEFICS-2](#55-token-multi-scale-trong-idefics-2)  
6. [Chiến lược gắn token ảnh vào LLM](#6-chiến-lược-gắn-token-ảnh-vào-llm)  
   6.1. [Prefix linear, interleaved pointer, gated fusion](#61-prefix-linear-interleaved-pointer-gated-fusion)  
   6.2. [Tính toán độ phức tạp attention chi tiết](#62-tính-toán-độ-phức-tạp-attention-chi-tiết)  
   6.3. [KV-cache sharing và memory compaction](#63-kv-cache-sharing-và-memory-compaction)  
   6.4. [Chuẩn hoá token metadata cho logging và audit](#64-chuẩn-hoá-token-metadata-cho-logging-và-audit)  
7. [Pipeline huấn luyện: từ codebook tới SFT/RLHF](#7-pipeline-huấn-luyện-từ-codebook-tới-sftrlhf)  
   7.1. [Chuẩn bị dữ liệu và cân bằng domain](#71-chuẩn-bị-dữ-liệu-và-cân-bằng-domain)  
   7.2. [Mục tiêu huấn luyện đa thành phần](#72-mục-tiêu-huấn-luyện-đa-thành-phần)  
   7.3. [Fine-tune cho bảo tàng: kịch bản thực tế](#73-fine-tune-cho-bảo-tàng-kịch-bản-thực-tế)  
   7.4. [Pseudo-code training loop kết hợp VQ-GAN + LLaMA](#74-pseudo-code-training-loop-kết-hợp-vq-gan--llama)  
8. [Ví dụ PyTorch đầy đủ: encode ảnh → token → LLaMA](#8-ví-dụ-pytorch-đầy-đủ-encode-ảnh--token--llama)  
9. [Đánh giá: metric, benchmark và phân tích lỗi](#9-đánh-giá-metric-benchmark-và-phân-tích-lỗi)  
10. [Các thách thức mở và hướng nghiên cứu tương lai](#10-các-thách-thức-mở-và-hướng-nghiên-cứu-tương-lai)  
11. [Case study: triển khai bàn thông tin trong Bảo tàng Giao Thoa](#11-case-study-triển-khai-bàn-thông-tin-trong-bảo-tàng-giao-thoa)  
12. [Phụ lục A: bảng ký hiệu và viết tắt](#12-phụ-lục-a-bảng-ký-hiệu-và-viết-tắt)  
13. [Phụ lục B: thuật toán Lloyd – EM – STE chi tiết](#13-phụ-lục-b-thuật-toán-lloyd--em--ste-chi-tiết)  
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo)

---

## 1. Bối cảnh vận hành tại bảo tàng

Sau một năm vận hành hệ thống hướng dẫn đa phương thức, cô hướng dẫn viên đối mặt với loạt câu hỏi:

- Làm thế nào để **một** model (LLM) hiểu ảnh hiển thị trên màn hình mà không cần server vision riêng?  
- Bộ nhớ GPU chỉ đủ context 4k token – nếu phải truyền 10 ảnh (mỗi ảnh 256 token) kèm 1.5k từ mô tả thì có vượt ngưỡng?  
- Nhật ký phải lưu lại “bằng chứng” – token ảnh khớp với patch nào, để khi khách khiếu nại có thể truy ngược.  

Kinh nghiệm vận hành cho thấy, khi token ảnh đã nằm chung với text, hàng loạt tiện ích mở ra:

1. **Unified RAG:** hình ảnh tác phẩm, chú thích, ghi âm thuyết minh đều được mã hoá thành vector chung, dễ lưu truy vấn.  
2. **Latency ổn định:** pipeline inference thuần text (kv-cache, batching) → không phải trộn 2 hệ thống.  
3. **Quản trị dễ dàng:** chỉ cần giám sát một mô hình, log chung, policy chung.  

Điểm mấu chốt là **chất lượng nén**: token phải đủ ít để tiết kiệm context nhưng vẫn giữ chi tiết phục vụ reasoning (màu sắc, motif, chú thích). Đó là nơi toán học rate–distortion, vector quantization và neural codec phát huy.

---

## 2. Timeline literature 2017 – 2024 và mapping SOTA

Dưới đây là timeline chi tiết (các số liệu lấy trực tiếp từ bài báo gốc; khi không có, cô ghi chú rõ). Tất cả công trình đều liên quan đến “nén ảnh thành token rời rạc” hoặc “chèn token vào LLM”.

| Năm | Công trình | Hội nghị/Tạp chí | Đóng góp | Số liệu minh hoạ |
|-----|------------|------------------|----------|------------------|
| 2017 | VQ-VAE (Van den Oord et al.) | NeurIPS | Giới thiệu vector quantization cho autoencoder; mở đường cho discrete tokens | Compression 32×, perplexity ≈ 3.0 |
| 2019 | dVAE (Ramesh et al.) | OpenAI Tech Report | Gumbel-Softmax để học codebook; dùng cho DALL·E | 8192-token codebook, 32×32 grid |
| 2020 | Perceiver (Jaegle et al.) | ICML | Cross-attention latent → giảm chi phí với dữ liệu lớn | Thử nghiệm multi-modal set-to-set |
| 2021 | VQ-GAN (Esser et al.) | CVPR | Thêm perceptual + adversarial loss; ảnh sắc nét | FID 7.4 @256², codebook 1024 |
| 2021 | TokenLearner (Ryoo et al.) | NeurIPS | Học trọng số chọn patch → adaptive tokens | Chọn 8–16 token cho ViT-B |
| 2022 | MaskGIT (Chang et al.) | CVPR | Masked token modeling song song hóa decoding | FID 6.18 @256², 512 bước |
| 2022 | HiFiC / Entroformer (Mentzer et al.) | CVPR/NeurIPS | Transformer-based codec với bitrate 0.14 bpp | MS-SSIM 0.93, latency <50ms |
| 2022 | Flamingo (Alayrac et al.) | NeurIPS | Perceiver Resampler + gated CA; 16 token/ảnh | VQAv2 82.6% (4-shot) |
| 2023 | PaLM-E (Driess et al.) | ICLR | Token ảnh (16) + PaLM-540B; embodied reasoning | OK-VQA 58.0 (+5.1) |
| 2023 | Kosmos-2 (Zhai et al.) | arXiv | Grounding token + pointer; reasoning + localization | RefCOCOg 67.2 Acc@0.5 |
| 2023 | MAGVIT2 (Yu et al.) | arXiv | Video tokenization 16× compression, streaming | MAE-VQ metrics 0.11 bpp |
| 2024 | IDEFICS-2 (Laurençon et al.) | arXiv | Multi-scale VQ tokenizer + LLaMA-3.1 open-source | ScienceQA 82.0, ChartQA 81.4 |
| 2024 | InternVL 1.5 (Cai et al.) | ECCV | Vision tokenizer 196 tokens + Q-Former style | MMMU 47.5 (13B) |

Nhìn vào timeline, cô rút ra 3 xu hướng:

1. **Discrete latent ngày càng ít token:** VQ-VAE (1024 token) → VQ-GAN (256) → PaLM-E (16).  
2. **Integration từ prefix sang KV injection:** Flamingo, PaLM-E, Kosmos-2 thể hiện sự đa dạng hoá cách ghép token.  
3. **Open-source dần theo kịp closed-source:** IDEFICS-2, InternVL, LLaVA-NeXT, MiniCPM-VL đều công bố tokenizer, script.  

---

## 3. Cơ sở toán: rate–distortion, entropy và lượng tử hoá

### 3.1. Định lý rate–distortion và ý nghĩa trong VLM

Định lý rate–distortion (Shannon 1948) cho biết giới hạn lý tưởng giữa bitrate $R$ (bit trung bình trên mỗi mẫu) và độ méo $D$ khi nén tín hiệu $X$.

Với tập hợp tất cả các phân phối $p(\hat{x}|x)$ sao cho $\mathbb{E}[d(x,\hat{x})] \le D$, ta có:

$$
R(D) = \inf_{p(\hat{x}|x)} I(X; \hat{X})
$$

Trong deep learning, $p(\hat{x}|x)$ được tham số hoá bằng encoder–decoder với latent $z$. Bằng cách giới hạn kích thước codebook hoặc đặt entropy penalty, ta điều khiển $R$.

**Ý nghĩa với VLM:**

- Mỗi token ảnh tương đương một “ký tự” gửi cho LLM. Context window là tài nguyên hữu hạn.  
- Nếu $R$ quá cao (nhiều token), LLM không đủ chỗ cho ngôn ngữ → reasoning giảm.  
- Nếu $R$ quá thấp, $D$ tăng, ảnh mất chi tiết → hallucination (ví dụ nhầm bình gốm xanh thành đỏ).  

### 3.2. Lagrangian cho nén học sâu

Các paper neural compression thường giải bài toán rate–distortion bằng Lagrangian:

$$
\mathcal{L}(\theta,\phi) = \mathbb{E}_{x\sim\mathcal{D}} \left[ -\log p_\theta(z) \right] + \lambda \, \mathbb{E}\left[d(x, \hat{x})\right]
$$

- $\theta$: tham số entropy model / decoder.  
- $\phi$: tham số encoder.  
- $p_\theta(z)$: xác suất latent (dùng hyperprior, uniform, logistic).  
- $\lambda$: hệ số cân bằng.  
- $d$: distortion (ví dụ $\ell_1$, LPIPS).  

Gradient theo $\theta$:

$$
\nabla_\theta \mathcal{L} = -\mathbb{E}_{x}\left[\nabla_\theta \log p_\theta(z)\right] + \lambda \, \nabla_\theta \mathbb{E}_x[d(x,\hat{x})]
$$

Với encoder $\phi$, gradient đi qua lượng tử hoá (không khả vi). Các phương án:

- **Straight-through estimator (STE):** copy gradient qua bước lượng tử hoá.  
- **Gumbel-Softmax:** soft relaxation.  
- **Vector quantization với EMA:** cập nhật codebook bằng trung bình động (như VQ-VAE).  

### 3.3. Bits-back, entropy coding và liên hệ với tokenizer

Khi token hoá ảnh thành chỉ số $k$, ta có thể dùng entropy coder (Arithmetic, ANS) để lưu trữ nén hơn. Tuy nhiên, khi trực tiếp đưa vào LLM, cô cần:

1. **Mapping 1-1** giữa mã $k$ và token `<v_k>` trong vocab LLM.  
2. **Entropy** của chuỗi token càng thấp, LLM càng dễ học (giảm perplexity).  

Bits-back (Frey & Hinton 1997) chỉ ra: nén tốt cần $p_\theta(z)$ gần với phân phối latent thật. Đối với VQ-VAE, perplexity codebook được tính:

$$
\text{Perplexity} = \exp\left(- \sum_{k=1}^K \hat{p}_k \log \hat{p}_k\right)
$$

Trong deploy, perplexity cao (≈ kích thước codebook) đảm bảo không có mã bị “chết” → ảnh nén đa dạng, không mất thông tin.

### 3.4. Thước đo distortion: PSNR, LPIPS, FID, CLIPScore

- **PSNR (Peak Signal-to-Noise Ratio):** $10 \log_{10}(MAX_I^2 / \text{MSE})$. Cao ⇒ ít lỗi pixel.  
- **LPIPS (Learned Perceptual Image Patch Similarity):** so sánh biểu diễn của mạng (VGG/AlexNet). Thấp ⇒ ảnh giống cảm nhận.  
- **FID (Fréchet Inception Distance):** đo khoảng cách giữa Gaussian thống kê trong feature space Inception. Thấp ⇒ phân phối giống.  
- **CLIPScore:** cosine giữa CLIP embeddings của ảnh và caption; dùng để đánh giá alignment sau nén.  

Trong bảo tàng, cô ưu tiên LPIPS và CLIPScore vì phản ánh cảm nhận của khách (họ quan tâm mô-tả hơn là từng pixel).

---

## 4. Vector quantization và các biến thể hiện đại

### 4.1. Thuật toán Lloyd–Max và trọng tâm codebook

Vector quantization cổ điển (Lloyd 1982) lặp hai bước:

1. **Assignment:** gán mỗi vector $z_e$ vào cluster gần nhất $e_k$.  
2. **Update:** đặt codebook $e_k$ bằng trung bình các vector gán vào cluster đó.

Trong deep learning, assignment = `argmin` trên GPU, update = EMA:

$$
e_k \leftarrow \gamma e_k + (1-\gamma)\frac{\sum_{i: k_i = k} z_{e,i}}{N_k + \epsilon}
$$

với $\gamma$ ≈ 0.99, $N_k$ số phần tử trong cluster.

### 4.2. Straight-through estimator, EMA và gradient surrogate

Vì `argmin` không khả vi, STE giả định:

$$
\frac{\partial z_q}{\partial z_e} \approx I
$$

Tức là gradient đi qua như thể lượng tử hoá là identity. Dù bias, STE hoạt động tốt trong thực nghiệm. EMA cập nhật codebook tránh “code collapse” (mã không được sử dụng).

### 4.3. VQ-VAE: cơ chế loss, perplexity và độ hội tụ

Loss tổng quát:

$$
\mathcal{L}_{\text{VQ-VAE}} = \|x - \hat{x}\|_2^2 + \| \text{sg}[z_e] - e_k\|_2^2 + \beta \|z_e - \text{sg}[e_k]\|_2^2
$$

- Term thứ hai kéo codebook về latent (codebook update).  
- Term thứ ba (commitment) khuyến khích encoder bám vào mã (tránh nhảy loạn).  
- $\beta$ thường 0.25–0.5.  

Về perplexity (đo đa dạng code):

$$
\text{Perplexity} = \exp\left(\frac{1}{N}\sum_{i=1}^N \log K - H(k_i)\right)
$$

Noi s con, perplexity cao (≈ $K$) = codebook phủ đều. Nếu perplexity thấp, cô cần tăng learning rate vector quantizer hoặc thêm codebook.

### 4.4. dVAE/DALL·E: Gumbel-Softmax, relaxation và attention mask

DALL·E [8] chuyển sang **discrete VAE (dVAE)**:

- Dùng **Gumbel-Softmax** để lấy sample differentiable:

$$
\tilde{q}_k = \frac{\exp\left((\log \pi_k + g_k)/\tau\right)}{\sum_j \exp\left((\log \pi_j + g_j)/\tau\right)}
$$

với $g_k$ ∼ Gumbel(0,1), $\tau$ nhiệt độ giảm dần.  
- Trước inference, lấy argmax để ra token rời rạc.  
- dVAE cho phép training end-to-end bằng backprop mà không cần STE.

### 4.5. VQ-GAN và loss cảm nhận (perceptual + adversarial)

VQ-GAN [2] bổ sung discriminator $D_\psi$ dạng PatchGAN:

$$
\mathcal{L}_{\text{GAN}} = \mathbb{E}_x[\log D_\psi(x)] + \mathbb{E}_x[\log(1 - D_\psi(\hat{x}))]
$$

Giúp ảnh tái tạo sắc nét (giảm blur). Đồng thời, perceptual loss dùng feature từ VGG/LPIPS:

$$
\mathcal{L}_{\text{perc}} = \sum_l \frac{1}{N_l} \| \phi_l(x) - \phi_l(\hat{x}) \|_2^2
$$

Trong bảo tàng, khi hiển thị tranh cổ, perceptual loss đảm bảo texture gỗ, vệt cọ không biến mất.

### 4.6. MaskGIT, token diffusion và nhiệt độ cập nhật

MaskGIT [3] coi token như chuỗi chữ:

1. Bắt đầu với tất cả token `[MASK]`.  
2. Ở mỗi bước, dự đoán phân phối cho tất cả vị trí.  
3. Lấy top-$p$ (tỷ lệ) để “unmask” cùng lúc.  

Nhiệt độ $\tau$ giảm dần:

$$
\tau_t = \max(\tau_{\min}, \tau_0 \cdot \alpha^t)
$$

Tốc độ song song hoá cao, thích hợp khi cần reconstruct vùng ảnh bị che (ví dụ có khách đứng chắn). 

### 4.7. Video token (ViT-VQ, VDT, MAGVIT2) và giới hạn băng thông

Các bài mới (MAGVIT2 [9], TokenMerger [10]) đưa ra cách nén video:

- Encode spatial 16×16 + temporal 4×.  
- Token streaming: truyền theo thời gian, LLM xử lý frame-by-frame.  
- Giới hạn: băng thông video (30fps) → cần 512 token/s để mô tả tour trực tiếp.  

Cô cân nhắc cho tương lai: khi bảo tàng muốn livestream hướng dẫn, video token là bước tiếp theo.

---

## 5. Neural codec, hyperprior và transformer-based compression

### 5.1. Hyperprior Ballé, entropy model và hồi quy Gaussian

Ballé et al. [11] giới thiệu hyperprior: latent $z$ có thêm hyper-latent $y$ mô tả phương sai:

$$
z \sim \mathcal{N}(0, \sigma^2(y)), \quad y \sim p_\psi(y)
$$

Loss:

$$
\mathbb{E}_{x}\left[-\log p(z|y) - \log p(y)\right] + \lambda \, d(x, \hat{x})
$$

Hyperprior giúp entropy model chính xác hơn, nén tốt hơn VQ fixed codebook.

### 5.2. Autoregressive vs factorized: trade-off tốc độ ↔ chất lượng

- **Autoregressive (PixelCNN, Transformer):** $p(z) = \prod_i p(z_i|z_{<i})$ → chất lượng cao, nhưng decode chậm (O(L)).  
- **Factorized:** giả sử độc lập, decode nhanh, nhưng quality thấp.  
- **Semi-autoregressive:** chia nhóm, decode song song trong nhóm.

Trong LLM, inference cần nhanh ⇒ cô ưu tiên factorized/hyperprior (như VQ-GAN, PaLM-E). Với offline storage, autoregressive cho chất lượng cao hơn.

### 5.3. HiFiC, Entroformer, ViT-VQ: học trực tiếp trong domain perceptual

HiFiC [4] kết hợp GAN + rate–distortion để tối ưu perceptual. Entroformer dùng transformer như entropy model. ViT-VQ (Lee et al. 2022) sử dụng ViT làm encoder quantizer, scale tốt cho image 512².

### 5.4. Perceiver Resampler và attention adapter (PaLM-E)

PaLM-E [5] pipeline:

1. ViT trích 196 patch token.  
2. **Perceiver Resampler:** multi-head cross-attention giữa latent ($M$ token) và patch token.  
3. Output 16 latent token → map qua linear sang embedding PaLM.  
4. LLM (PaLM-562B) tiếp tục reasoning.  

Cross-attention cho phép latent học “what to keep” → nén adaptively (không chỉ grid cố định).

### 5.5. Token multi-scale trong IDEFICS-2

IDEFICS-2 [7] công bố open-source tokenizer:

- Hai tầng nén: 8× (low-level) + 16× (high-level).  
- Cho phép tuỳ chọn 64 hoặc 256 token/ảnh.  
- Training mix: LAION + DataComp + sách scan.  
- LLM (LLaMA-3.1) được instruction-tuning với token `<image_patch_i>` kèm metadata.

Đây là blueprint tốt cho dự án open-source trong bảo tàng.

---

## 6. Chiến lược gắn token ảnh vào LLM

### 6.1. Prefix linear, interleaved pointer, gated fusion

Ba kiểu phổ biến:

1. **Prefix linear:** `<IMG> t1 t2 ... tM </IMG>` đặt trước câu hỏi.  
2. **Interleaved pointer:** chèn token vào đúng vị trí được nhắc đến, kèm tag `<IMG:id>` (Kosmos-2).  
3. **Gated fusion / KV injection:** token ảnh đi vào các layer thông qua adapter (Flamingo, PaLM-E).  

**Ưu nhược điểm:**

| Kiểu | Ưu điểm | Nhược điểm |
|------|---------|------------|
| Prefix | Implementation đơn giản | Attention cost O((M+L)^2) lớn |
| Interleaved | Giữ ngữ cảnh sát nghĩa, hỗ trợ grounding | Cần mask phức tạp |
| Gated fusion | Tiết kiệm token (latent nhỏ) | Phải sửa kiến trúc LLM |

### 6.2. Tính toán độ phức tạp attention chi tiết

Với self-attention, số phép nhân- cộng ≈ $4d(M+L)^2$ (vì Q,K,V,O).  
Thời gian ước tính:

$$
T_{\text{attn}} \approx \frac{4d(M+L)^2}{\text{FLOPS}_{\text{GPU}}}
$$

Ví dụ: LLaMA-3 70B (d=8192), $L=1200$, $M=256$, GPU A100 312 TFLOPS FP16:

$$
T \approx \frac{4 \cdot 8192 \cdot 1456^2}{3.12 \times 10^{14}} \approx 0.22\text{ ms/layer}
$$

Với 80 layer → ≈ 17.6 ms (attention). Thêm MLP (≈ 2×) → ~52 ms tổng inference (chưa tính kv-cache).  
Nếu giảm M xuống 64 token, thời gian còn 13 ms. Do đó nén ảnh càng mạnh, LLM càng responsive.

### 6.3. KV-cache sharing và memory compaction

Trong inference multi-turn:

- Token ảnh có thể lưu trong KV-cache giống text.  
- Khi khách chuyển sang bức tranh khác, cô flush cache cũ để giải phóng.  
- Sử dụng **memory compaction**: chỉ giữ top-k token dựa trên attention weight trung bình.  

### 6.4. Chuẩn hoá token metadata cho logging và audit

Để audit, cô lưu JSON:

```json
{
  "image_id": "room3_vase",
  "token": "<v312>",
  "patch": [x1, y1, x2, y2],
  "level": "high",
  "timestamp": "2025-04-08T10:23:45Z"
}
```

Trong inference log, khi LLM trích `[CITE:<v312>]`, cô truy vết được patch gốc. Nhờ đó nếu khách phản hồi “mô tả sai màu”, cô kiểm tra token tương ứng.

---

## 7. Pipeline huấn luyện: từ codebook tới SFT/RLHF

### 7.1. Chuẩn bị dữ liệu và cân bằng domain

Dataset mix cô đang dùng:

| Domain | Ví dụ | Tỷ lệ | Mục đích |
|--------|-------|-------|----------|
| Nghệ thuật (wikiart, bảo tàng nội bộ) | Tranh sơn dầu, điêu khắc | 30% | Phù hợp tour |
| Tài liệu (DocVQA, OCR) | Poster, biển chỉ dẫn | 20% | Đảm bảo đọc chữ |
| Ảnh đời thường (COCO, Flickr) | Khách tham quan | 20% | Tăng robustness |
| Biểu đồ (ChartQA, PlotQA) | Đồ thị, timeline | 10% | Giải thích infographic |
| Kỹ thuật (Blueprint, CAD) | Sơ đồ kiến trúc | 10% | Hướng dẫn đường đi |
| Video clip (MAGVIT2 subsample) | Giới thiệu phòng | 10% | Chuẩn bị tương lai |

### 7.2. Mục tiêu huấn luyện đa thành phần

Tổng loss:

$$
\mathcal{L} = \mathcal{L}_{\text{rate}} + \lambda_1 \mathcal{L}_{\text{rec}} + \lambda_2 \mathcal{L}_{\text{perc}} + \lambda_3 \mathcal{L}_{\text{gan}} + \lambda_4 \mathcal{L}_{\text{text-align}}
$$

- $\mathcal{L}_{\text{rate}} = -\log p(z)$.  
- $\mathcal{L}_{\text{rec}}$: reconstruction (PSNR).  
- $\mathcal{L}_{\text{perc}}$: perceptual (LPIPS).  
- $\mathcal{L}_{\text{gan}}$: nếu dùng discriminator.  
- $\mathcal{L}_{\text{text-align}}$: CLIP loss giữa ảnh tái tạo và caption.  

Sau khi huấn luyện tokenizer, cô freeze encoder, map token sang vocab LLM, rồi tiếp tục SFT/RLHF:

1. **SFT:** fine-tune LLM trên cặp (token ảnh, question, answer, CoT).  
2. **RLHF:** reward model đánh giá factuality, cite patch.  
3. **Self-consistency:** sinh $K$ chain-of-thought, majority vote, distill.  

### 7.3. Fine-tune cho bảo tàng: kịch bản thực tế

Quy trình:

1. Thu thập 1.5k ảnh tác phẩm + chú thích gốc + câu hỏi thường gặp.  
2. Encode ảnh bằng tokenizer (ví dụ IDEFICS-2).  
3. Tạo prompt:

```
<IMG> ...tokens... </IMG>
Khách: Bức tranh này sử dụng kỹ thuật gì?
Hướng dẫn viên: ...
```

4. SFT LLaMA-3 8B (LoRA rank 64, 4 GPU).  
5. RLHF: reward model chấm điểm 0–1 dựa trên:
   - Độ chính xác (factual).  
   - Nhắc tới patch ID (`[CITE:<v123>]`).  
   - Phong cách thân thiện.  
6. Kiểm thử offline → deploy.  

### 7.4. Pseudo-code training loop kết hợp VQ-GAN + LLaMA

```python
for epoch in range(num_epochs):
    for images, captions in dataloader:
        # 1) Train tokenizer
        z_e = encoder(images)
        z_q, indices, commit_loss = quantizer(z_e)
        recon = decoder(z_q)
        rec_loss = F.l1_loss(recon, images)
        perc_loss = lpips(recon, images)
        rate_loss = entropy_model(indices)
        loss_tokenizer = rate_loss + lambda_rec * rec_loss + lambda_perc * perc_loss + beta * commit_loss
        loss_tokenizer.backward()
        optimizer_token.step()

        # 2) Supervised fine-tuning LLM (freeze tokenizer)
        with torch.no_grad():
            image_tokens = indices.view(indices.size(0), -1)
        prompt = build_prompt(image_tokens, captions)
        outputs = llm(prompt, labels=captions)
        loss_llm = outputs.loss
        loss_llm.backward()
        optimizer_llm.step()
```

Trong thực tế, cô tách thành 2 pha: train tokenizer trước, sau đó freeze; nhưng pseudo-code giúp hình dung pipeline tổng thể.

---

## 8. Ví dụ PyTorch đầy đủ: encode ảnh → token → LLaMA

Đoạn code sau chi tiết hoá hơn (bao gồm mapping token, caching). Ở môi trường thực, hãy thêm xử lý lỗi và batch.

```python
import torch
import torch.nn.functional as F
from taming.models.vqgan import VQModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------- 1. Load các module cần thiết --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqgan = VQModel.from_pretrained("CompVis/vqgan-imagenet-f16-16384").to(device).eval()
llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-instruct")
llm = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

# -------- 2. Hàm encode ảnh --------
def encode_image_to_tokens(image_tensor: torch.Tensor) -> torch.LongTensor:
    """
    image_tensor: [3, 256, 256] đã chuẩn hoá về [-1, 1]
    return: [L] với L = 16*16 = 256 token index
    """
    with torch.no_grad():
        z = vqgan.encoder(image_tensor.unsqueeze(0).to(device))
        z_q, _, info = vqgan.quantize(z)
        indices = info[2].view(-1)  # flatten
    return indices.cpu()

# -------- 3. Mapping token index -> vocab --------
IMG_START = "<image>"
IMG_END = "</image>"

def format_image_tokens(indices: torch.LongTensor) -> str:
    return " ".join(f"<v{idx.item()}>" for idx in indices)

def build_prompt(image_indices, question: str) -> str:
    image_segment = format_image_tokens(image_indices)
    prompt = (
        f"{IMG_START} {image_segment} {IMG_END}\n"
        "Khách: " + question + "\n"
        "Hướng dẫn viên:"
    )
    return prompt

# -------- 4. Inference --------
def answer_question(image_tensor, question: str, max_new_tokens: int = 128) -> str:
    indices = encode_image_to_tokens(image_tensor)
    prompt = build_prompt(indices, question)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=llm_tokenizer.eos_token_id
        )
    answer = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer
```

**Gợi ý mở rộng:**

- Thêm metadata `{idx: (x, y, w, h)}` để LLM cite patch.  
- Dùng LoRA để fine-tune LLaMA học các token `<v*>`.  
- Khi deploy, nên quantize model (GPTQ/AWQ) để giảm latency.

---

## 9. Đánh giá: metric, benchmark và phân tích lỗi

### 9.1. Metric định lượng

| Nhóm | Metric | Mục đích |
|------|--------|----------|
| Compression | Bits-per-pixel (bpp), Rate (bits/token) | Đánh giá chi phí context |
| Reconstruction | PSNR, SSIM, LPIPS | Kiểm tra fidelity |
| Generative | FID, KID | So sánh phân phối ảnh |
| Multimodal | CLIPScore, BLIPScore | Alignment ảnh–text |
| Task | VQAv2, OK-VQA, ChartQA, MMMU | Reasoning trên token nén |
| Latency | Encode time, LLM time/token | Khả năng realtime |

### 9.2. Benchmark nên theo dõi

- **Token compression:** sử dụng dataset Kodak, CLIC để so sánh bpp với HiFiC, VQ-GAN.  
- **Multimodal QA:** ScienceQA, ChartQA, TextVQA.  
- **Grounding:** RefCOCO/RefCOCOg (accuracy bounding box/caption).  
- **Museum internal:** 500 câu hỏi giả lập, 200 ảnh tác phẩm, 50 chart timeline.  

### 9.3. Quy trình phân tích lỗi

1. **Replay thought chain:** kiểm tra token `[THOUGHT]`, `[CITE]`.  
2. **So sánh patch:** decode token → patch gốc, đối chiếu xem có mất màu/mẫu không.  
3. **Đo CLIPScore:** nếu thấp → token mất ngữ nghĩa → cần fine-tune codebook.  
4. **Check entropy:** nếu một nhóm token xuất hiện quá nhiều → codebook collapse.  

---

## 10. Các thách thức mở và hướng nghiên cứu tương lai

1. **Dynamic token allocation:** học số lượng token tuỳ ảnh (TokenLearner + VQ).  
2. **Joint audio-image tokens:** SoundStream [12] + VQ-GAN → token đa giác quan.  
3. **Structured metadata:** encode vector (patch location, depth, normal) cùng token.  
4. **Training stability:** khi kết hợp GAN + RLHF, gradient có thể mâu thuẫn → cần gradient surgery.  
5. **Quantization-aware LLM:** dạy LLM hiểu các token `<v*>` bằng embedding học riêng, tránh xung đột từ vựng.  

---

## 11. Case study: triển khai bàn thông tin trong Bảo tàng Giao Thoa

- **Phần cứng:** 1 GPU A100 80GB cho inference, 1 GPU A40 cho preprocessing.  
- **Pipeline:**  
  1. Khởi chạy tokenizer (IDEFICS-2).  
  2. Pre-bake token cho 5.000 ảnh tác phẩm, lưu vào vector store (FAISS).  
  3. Khi khách hỏi, hệ thống tải token + caption, ghép prompt → LLM trả lời.  
  4. Nếu confidence < 0.6 hoặc question dài > 1.5k token → fallback GPT-4V (API).  
  5. Log thought/action/observation/citation.  
- **Kết quả:**  
  - Latency trung bình 1.8s (giảm từ 3.4s trước khi nén).  
  - Bộ nhớ context 2.1k token cho 2 ảnh + CoT 8 bước.  
  - Feedback khách: 92% đánh giá mô tả chính xác (so với 87% trước đó).  

---

## 12. Phụ lục A: bảng ký hiệu và viết tắt

| Ký hiệu | Nghĩa |
|---------|-------|
| $x$ | Ảnh gốc |
| $\hat{x}$ | Ảnh tái tạo |
| $z_e$ | Latent encoder (liên tục) |
| $z_q$ | Latent lượng tử hoá |
| $e_k$ | Vector codebook thứ $k$ |
| $K$ | Số mã trong codebook |
| $M$ | Số token ảnh truyền vào LLM |
| $L$ | Số token text trong prompt |
| $d$ | Chiều embedding của LLM |
| $R$ | Bitrate hoặc số bit/token |
| $D$ | Distortion mong muốn |
| LPIPS | Learned Perceptual Image Patch Similarity |
| FID | Fréchet Inception Distance |

---

## 13. Phụ lục B: thuật toán Lloyd – EM – STE chi tiết

**Thuật toán 1: Lloyd–Max cho VQ-VAE với EMA**

```
Input: Codebook {e_k}, decay γ, batch latent {z_e}
For each mini-batch:
  # Bước gán
  for i in batch:
      k_i = argmin_k ||z_e[i] - e_k||_2^2
  # Bước cập nhật
  for k in 1..K:
      N_k = number of i with k_i = k
      if N_k > 0:
          e_k ← γ e_k + (1-γ) * (Σ_{i: k_i=k} z_e[i]) / (N_k + ε)
```

**Thuật toán 2: Straight-through estimator (STE)**

```
# Forward
z_q = e_{k}  where k = argmin ||z_e - e_k||
# Backward
∂L/∂z_e ≈ ∂L/∂z_q
∂L/∂e_k accumulates from (z_e - e_k)
```

**Thuật toán 3: Gumbel-Softmax (dVAE)**

```
logits = f(z_e)  # dimension K
g = -log(-log(U)),  U ~ Uniform(0, 1)
τ = temperature
weights = softmax((logits + g) / τ)
z_q = Σ_k weights[k] * e_k
```

---

## 14. Tài liệu tham khảo

1. Van den Oord, A. et al. (2017). *Neural Discrete Representation Learning (VQ-VAE).* NeurIPS.  
2. Esser, P. et al. (2021). *Taming Transformers for High-Resolution Image Synthesis (VQ-GAN).* CVPR.  
3. Chang, H. et al. (2022). *MaskGIT: Masked Generative Image Transformer.* CVPR.  
4. Mentzer, F. et al. (2022). *High-Fidelity Generative Image Compression.* CVPR.  
5. Driess, D. et al. (2023). *PaLM-E: An Embodied Multimodal Language Model.* ICLR.  
6. Zhai, X. et al. (2023). *Kosmos-2: Grounding Multimodal Large Language Models to the World.* arXiv:2306.14824.  
7. Laurençon, H. et al. (2024). *IDEFICS-2: An Open Multimodal Generative and Instruction-Tuned Model Family.* arXiv:2407.13714.  
8. Ramesh, A. et al. (2021). *Zero-Shot Text-to-Image Generation.* ICML (DALL·E technical report).  
9. Yu, J. et al. (2023). *MAGVIT V2: Scaling Efficient Video Generation with Multi-Axis Residual Quantization.* arXiv:2307.04755.  
10. Bolya, D. et al. (2023). *Token Merging for Fast Stable Diffusion.* arXiv:2305.11680.  
11. Ballé, J. et al. (2018). *Variational Image Compression with a Scale Hyperprior.* ICLR.  
12. Zeghidour, N. et al. (2022). *SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP.  
13. Ryoo, M. et al. (2021). *TokenLearner: Adaptive Space-Time Tokenization for Videos.* NeurIPS.  
14. Alayrac, J.-B. et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS.  
15. Jaegle, A. et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.  
16. Cai, W. et al. (2024). *InternVL 1.5: Simple Yet Effective Vision-Language Alignment.* ECCV.  
17. Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.  
18. Gao, P. et al. (2023). *LLaVA-NeXT: Benchmarking Visual Instruction Tuning.* arXiv:2311.17176.  
19. Wu, C.-Y. et al. (2022). *Token Shuffle for Event-based Video.* ECCV.  
20. Chen, W. et al. (2024). *ChartBench: Evaluating MLLMs for Chart Understanding.* arXiv:2403.04122.

---

<script src="/assets/js/katex-init.js"></script>
