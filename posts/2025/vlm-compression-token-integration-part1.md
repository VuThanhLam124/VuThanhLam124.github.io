---
title: "Compression & Token Integration (Phần I): Nền tảng toán học và kiến trúc lượng tử hóa"
date: "2025-04-10"
category: "vision-language-models"
tags: ["vlm", "compression", "rate-distortion", "vq-vae", "vq-gan", "image-tokenization", "information-theory"]
excerpt: "Phần I đặt nền tảng cho việc nén ảnh thành token văn bản: từ lý thuyết rate–distortion, entropy, vector quantization, VQ-VAE/VQ-GAN, MaskGIT cho tới các kiến trúc mã hóa rời rạc và pipeline huấn luyện. Mỗi khái niệm đều được diễn giải chi tiết với công thức, trực giác và ví dụ triển khai."
author: "ThanhLamDev"
readingTime: 54
featured: false
---

# Compression & Token Integration (Phần I): Nền tảng toán học và kiến trúc lượng tử hóa

**Phần I mở đầu bộ đôi bài viết chuyên sâu về cách nén ảnh thành chuỗi token để đưa trực tiếp vào LLM. Từ góc nhìn của cô hướng dẫn viên tại Bảo tàng Giao Thoa, chúng ta quay lại phòng thí nghiệm cùng người thợ pha lê – nơi mọi phép tính, mọi biến đổi đều phải được ghi chép cẩn thận trước khi một bức tranh hay bức tượng được chuyển thành những ký hiệu mà LLM hiểu được.**

---

## Mục lục chi tiết

1. [Bối cảnh và mục tiêu của bộ đôi bài viết](#1-bối-cảnh-và-mục-tiêu-của-bộ-đôi-bài-viết)  
2. [Lược đồ tổng thể: từ photon đến token](#2-lược-đồ-tổng-thể-từ-photon-đến-token)  
3. [Cơ sở toán học](#3-cơ-sở-toán-học)  
   3.1. [Không gian xác suất và biến ngẫu nhiên ảnh](#31-không-gian-xác-suất-và-biến-ngẫu-nhiên-ảnh)  
   3.2. [Entropy, cross-entropy và KL divergence](#32-entropy-cross-entropy-và-kl-divergence)  
   3.3. [Mutual information và vai trò trong VLM](#33-mutual-information-và-vai-trò-trong-vlm)  
   3.4. [Rate–distortion theory: định nghĩa và tiên đề](#34-rate–distortion-theory-định-nghĩa-và-tiên-đề)  
   3.5. [Chứng minh đường cong $R(D)$ cho nguồn Gaussian](#35-chứng-minh-đường-cong-rd-cho-nguồn-gaussian)  
   3.6. [Lagrangian trong tối ưu hoá nén học sâu](#36-lagrangian-trong-tối-ưu-hoá-nén-học-sâu)  
   3.7. [Liên hệ với AutoEncoder và ELBO](#37-liên-hệ-với-autoencoder-và-elbo)  
4. [Vector Quantization (VQ) toàn cảnh](#4-vector-quantization-vq-toàn-cảnh)  
   4.1. [Định nghĩa, ký hiệu và các dạng lượng tử hoá](#41-định-nghĩa-ký-hiệu-và-các-dạng-lượng-tử-hoá)  
   4.2. [Thuật toán Lloyd–Max và diễn giải trực giác](#42-thuật-toán-lloyd–max-và-diễn-giải-trực-giác)  
   4.3. [Chứng minh hội tụ và điều kiện tối ưu cấp hai](#43-chứng-minh-hội-tụ-và-điều-kiện-tối-ưu-cấp-hai)  
   4.4. [K-means, GLA và các biến thể](#44-k-means-gla-và-các-biến-thể)  
   4.5. [Product Quantization, Residual Quantization, Ternary Quantization](#45-product-quantization-residual-quantization-ternary-quantization)  
   4.6. [Geometric view: Voronoi diagram trên manifold ảnh](#46-geometric-view-voronoi-diagram-trên-manifold-ảnh)  
5. [Từ lý thuyết đến kiến trúc cụ thể](#5-từ-lý-thuyết-đến-kiến-trúc-cụ-thể)  
   5.1. [VQ-VAE: thiết kế encoder–decoder và codebook](#51-vq-vae-thiết-kế-encoder–decoder-và-codebook)  
   5.2. [Cơ chế loss và gradient của VQ-VAE](#52-cơ-chế-loss-và-gradient-của-vq-vae)  
   5.3. [EMA codebook update và phân tích ổn định](#53-ema-codebook-update-và-phân-tích-ổn-định)  
   5.4. [VQ-VAE-2 và hierarchical latent](#54-vq-vae-2-và-hierarchical-latent)  
   5.5. [dVAE và Gumbel-Softmax relaxation](#55-dvae-và-gumbel-softmax-relaxation)  
   5.6. [VQ-GAN: perceptual loss và discriminator](#56-vq-gan-perceptual-loss-và-discriminator)  
   5.7. [MaskGIT và masked token modeling](#57-maskgit-và-masked-token-modeling)  
   5.8. [MAE-VQ và token learning adaptive](#58-mae-vq-và-token-learning-adaptive)  
6. [Đo lường chất lượng: distortion, perceptual, task-level](#6-đo-lường-chất-lượng-distortion-perceptual-task-level)  
   6.1. [PSNR, SSIM, MS-SSIM: công thức và limitations](#61-psnr-ssim-ms-ssim-công-thức-và-limitations)  
   6.2. [LPIPS, DISTS, FID, KID: ý nghĩa với VLM](#62-lpips-dists-fid-kid-ý-nghĩa-với-vlm)  
   6.3. [CLIPScore, BLIPScore và Alignment metric](#63-clipsCore-blipScore-và-alignment-metric)  
   6.4. [Token diversity metrics: perplexity, utilization](#64-token-diversity-metrics-perplexity-utilization)  
7. [Timeline nghiên cứu 2017–2021 chi tiết](#7-timeline-nghiên-cứu-2017–2021-chi-tiết)  
8. [Hạ tầng triển khai encoder lượng tử hoá](#8-hạ-tầng-triển-khai-encoder-lượng-tử-hoá)  
   8.1. [Design hardware pipeline và throughput](#81-design-hardware-pipeline-và-throughput)  
   8.2. [Data loading, augmentation, mixed precision](#82-data-loading-augmentation-mixed-precision)  
   8.3. [Quản lý checkpoint, logging và reproducibility](#83-quản-lý-checkpoint-logging-và-reproducibility)  
9. [Ví dụ PyTorch chi tiết: xây dựng VQ-VAE từ đầu](#9-ví-dụ-pytorch-chi-tiết-xây-dựng-vq-vae-từ-đầu)  
10. [Checklist kỹ thuật và bài học thực nghiệm](#10-checklist-kỹ-thuật-và-bài-học-thực-nghiệm)  
11. [Kết luận phần I và giới thiệu phần II](#11-kết-luận-phần-i-và-giới-thiệu-phần-ii)  
12. [Phụ lục A: Ký hiệu và quy ước](#12-phụ-lục-a-ký-hiệu-và-quy-ước)  
13. [Phụ lục B: Chứng minh bổ sung](#13-phụ-lục-b-chứng-minh-bổ-sung)  
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo)

---

## 1. Bối cảnh và mục tiêu của bộ đôi bài viết

Phần I có nhiệm vụ đặt nền móng lý thuyết và kiến trúc cho việc nén ảnh thành token.
Tất cả những khái niệm xuất hiện trong bài đều được định nghĩa và giải thích kỹ lưỡng trước khi sử dụng trong các phần tiếp theo.

- Chúng ta sẽ bắt đầu bằng bối cảnh câu chuyện tại Bảo tàng Giao Thoa, nơi người thợ pha lê và cô hướng dẫn viên cùng hợp tác để số hóa các tác phẩm nghệ thuật.
- Từ nguồn ảnh thực tế, chúng ta xây dựng một pipeline toán học chặt chẽ: mô hình hóa ảnh như biến ngẫu nhiên, áp dụng information theory để định lượng chi phí truyền thông tin.
- Tiếp theo, bài viết lần lượt trình bày vector quantization, VQ-VAE, VQ-GAN, MaskGIT – những khối xây dựng cốt lõi giúp biến ảnh thành chuỗi mã rời rạc.
- Mỗi kiến trúc sẽ được mô tả chi tiết: từ công thức loss, gradient, cách cập nhật codebook cho tới khó khăn thực nghiệm.
- Cuối cùng, chúng ta cung cấp ví dụ PyTorch tự viết toàn bộ VQ-VAE – nền tảng thực hành trước khi chuyển sang Part II (nói về neural codec, integration vào LLM, training đa giai đoạn).

Phần II (ở file khác) sẽ tiếp tục các nội dung nâng cao: hyperprior, neural codec, Perceiver Resampler, PaLM-E, token integration với LLaMA/GPT, pipeline SFT+RLHF, đánh giá và hệ thống hóa.

---

## 2. Lược đồ tổng thể: từ photon đến token

Để hình dung rõ pipeline, hãy mô tả từng bước mà phòng thí nghiệm của người thợ phải thực hiện.

1. **Thu nhận ánh sáng**  
   - Camera hoặc cảm biến nắm bắt photon, chuyển đổi thành tín hiệu analog điện.
   - Bộ ADC (Analog-to-Digital Converter) chuyển tín hiệu analog sang giá trị số (thường 12–14 bit mỗi kênh).
   - Kết quả là tensor ảnh $X \in [0, 255]^{H \times W \times C}$.

2. **Chuẩn hóa và mô hình hóa**  
   - Ảnh được chuyển về miền $[0, 1]$ hoặc $[-1, 1]$.
   - Xem ảnh như biến ngẫu nhiên $X$, có phân phối thực nghiệm $\hat{p}_{\text{data}}$.

3. **Encoder**  
   - Một mạng CNN / Vision Transformer nhận $X$ và tạo latent liên tục $Z_e = E_\phi(X)$.
   - Latent thường có dạng $Z_e \in \mathbb{R}^{h \times w \times d}$ với $h = H/s$, $w = W/s$, $d$ số kênh.

4. **Vector Quantization**  
   - Lượng tử hoá $Z_e$ thành mã rời rạc $Z_q$ bằng codebook $\mathcal{E} = \{e_k\}_{k=1}^K$.
   - Mỗi vector $Z_e(i,j)$ chọn mã gần nhất $e_{k^\*}$ → token chỉ số $k^\*$.
   - Chuỗi token thu được: $[k_1, k_2, \ldots, k_{hw}]$.

5. **Decoder (tái tạo)**  
   - Mạng $D_\theta$ nhận $Z_q$ và tái tạo $\hat{X} = D_\theta(Z_q)$.
   - Loss tái tạo + regularizer đảm bảo $\hat{X}$ gần $X$.

6. **Mapping sang vocab text**  
   - Người thợ tạo ánh xạ $k \mapsto$ token văn bản, ví dụ `<v123>`.
   - Chuỗi token ảnh được ghép vào prompt LLM.

7. **Integration**  
   - LLM đọc prompt chứa token ảnh, reasoning để trả lời các câu hỏi của khách.
   - Trong Part II, chúng ta sẽ nghiên cứu cách gắn token hiệu quả (prefix, interleaving, KV injection).

---

## 3. Cơ sở toán học

### 3.1. Không gian xác suất và biến ngẫu nhiên ảnh

- Một ảnh màu $X$ được xem là biến ngẫu nhiên trên không gian đo $(\Omega, \mathcal{F}, \mathbb{P})$.
- Mỗi phần tử $\omega \in \Omega$ đại diện cho một lần chụp ảnh.
- Hàm $X: \Omega \to \mathbb{R}^{H \times W \times C}$ đo được theo sigma-algebra $\mathcal{F}$.
- Phân phối thật $p_{\text{data}}$ không thể truy cập trực tiếp; ta làm việc với phân phối thực nghiệm $\hat{p}_{\text{data}}$ dựa trên tập mẫu $\{x^{(i)}\}_{i=1}^N$.
- Mục đích nén: tìm ánh xạ $Q: \mathbb{R}^{H \times W \times C} \to \mathcal{Z}^L$ (chuỗi mã) sao cho:
  1. **Tốc độ** $R = \frac{1}{N} \sum_{i=1}^N \log_2 |\mathcal{Z}|$ càng nhỏ càng tốt.
  2. **Độ méo** $D = \frac{1}{N} \sum_{i=1}^N d(x^{(i)}, \hat{x}^{(i)})$ không vượt quá ngưỡng đặt trước.

### 3.2. Entropy, cross-entropy và KL divergence

- **Entropy** $H(X) = -\sum_x p(x) \log p(x)$ đo lượng thông tin kỳ vọng của biến $X$.
- **Cross-entropy** giữa $p$ và $q$: $H(p, q) = -\sum_x p(x) \log q(x)$.
- **KL divergence**: $D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$.
- Liên hệ: $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$.
- Trong nén ảnh, cross-entropy tương ứng số bit cần thiết khi mã hóa $X$ bằng mô hình $q$. Mục tiêu: thiết kế $q$ gần đúng $p$ để giảm số bit.

### 3.3. Mutual information và vai trò trong VLM

- Mutual information $I(X; Y) = H(X) - H(X|Y)$ đo lượng thông tin chung giữa hai biến.
- Khi chuỗi token ảnh được ghép với văn bản, mutual information giữa token và caption phải đủ lớn để LLM khai thác.
- Trong VQ-VAE, ta tối ưu sao cho embedding $Z_q$ giữ information về $X$ → tương đương maximize $I(X; Z_q)$ dưới ràng buộc bitrate.

### 3.4. Rate–distortion theory: định nghĩa và tiên đề

- Với biến nguồn $X$ và biến tái tạo $\hat{X}$, distortion function $d: \mathcal{X} \times \hat{\mathcal{X}} \to \mathbb{R}_{\ge 0}$.
- Rate–distortion function:
  $$
  R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(X, \hat{X})] \le D} I(X; \hat{X}).
  $$
- $R(D)$ là cận dưới thông tin cho mọi scheme nén: không thể đạt bitrate trung bình thấp hơn $R(D)$ với distortion không vượt quá $D$.
- **Tiên đề**:
  1. $R(D)$ không tăng khi $D$ tăng.
  2. $R(D)$ lồi theo $D$.
  3. $R(D) = 0$ khi $D \ge D_{\max}$ (distortion đủ lớn chấp nhận tái tạo cố định).

### 3.5. Chứng minh đường cong $R(D)$ cho nguồn Gaussian

- Với $X \sim \mathcal{N}(0, \sigma^2)$ và distortion $d(x, \hat{x}) = (x - \hat{x})^2$, ta có:
  $$
  R(D) = \max\left\{0, \frac{1}{2} \log \frac{\sigma^2}{D}\right\}.
  $$
- **Phác thảo chứng minh**:
  1. Xét biến tái tạo $\hat{X} = X + N$ với $N \sim \mathcal{N}(0, D)$ độc lập. Khi đó $I(X; \hat{X}) = \frac{1}{2} \log \frac{\sigma^2 + D}{D}$.
  2. Tối ưu $I$ theo $D$ dẫn tới công thức trên.
  3. Sử dụng tính chất lồi để chứng minh đó là cận dưới chặt.
- **Ý nghĩa**: với ảnh natural image, giả định Gaussian không chính xác hoàn toàn nhưng cung cấp trực giác: distortion giảm một nửa đồng nghĩa bitrate tăng ~0.5 bit/pixel.

### 3.6. Lagrangian trong tối ưu hoá nén học sâu

- Ta giải bài toán constrained tối ưu bằng Lagrangian:
  $$
  \mathcal{L}_{\text{RD}} = \mathbb{E}[-\log p_\theta(z)] + \lambda \mathbb{E}[d(x, \hat{x})].
  $$
- $\lambda$ đóng vai trò hệ số trade-off. Khi training, thay đổi $\lambda$ tương đương chọn điểm khác trên đường cong $R(D)$.
- Ở VQ-VAE/VQ-GAN, $p_\theta(z)$ thường giả sử uniform nên term đầu = $\log K$. Với hyperprior, $p_\theta$ được học.
- Backpropagation yêu cầu gradient qua lượng tử hoá – đây là lý do cần STE hoặc relaxation (phần 5).

### 3.7. Liên hệ với AutoEncoder và ELBO

- AutoEncoder tiêu chuẩn tối ưu $\|x - \hat{x}\|^2$ nhưng không kiểm soát bitrate. VQ-VAE bổ sung lượng tử hoá để ràng buộc entropy latent.
- VAE tối ưu Evidence Lower Bound:
  $$
  \text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z)).
  $$
- Nếu $q_\phi(z|x)$ bị ràng buộc trở thành phân phối rời rạc với entropy hữu hạn, ta thu được formulation tương tự rate–distortion.
- Kết luận: VQ-VAE là cầu nối giữa AutoEncoder và lý thuyết nén.

---

## 4. Vector Quantization (VQ) toàn cảnh

### 4.1. Định nghĩa, ký hiệu và các dạng lượng tử hoá

- Vector quantization là phép ánh xạ $\mathbb{R}^d \to \{e_1, \ldots, e_K\}$.
- Mỗi vector $z \in \mathbb{R}^d$ được gán vào codeword $e_k$ sao cho khoảng cách $d(z, e_k)$ nhỏ nhất.
- Codebook $\mathcal{E}$ có thể cố định hoặc học được.
- Phân biệt:
  1. **Scalar quantization**: $d=1$.
  2. **Vector quantization**: $d>1$, xem xét khối vector.
  3. **Structured quantization**: product, residual, tree-based.
- Mục tiêu: giảm số bit mô tả vector (từ liên tục vô hạn sang rời rạc $K$ lựa chọn).

### 4.2. Thuật toán Lloyd–Max và diễn giải trực giác

- Lloyd–Max (Generalized Lloyd Algorithm – GLA) tối ưu codebook theo hai bước lặp:
  1. **Assignment step (E-step)**: với codebook hiện tại, gán mỗi điểm $z$ vào cluster $S_k = \{z: k = \arg\min_j \|z - e_j\|^2\}$.
  2. **Update step (M-step)**: cập nhật $e_k = \frac{1}{|S_k|} \sum_{z \in S_k} z$.
- Algorithm hội tụ tới điểm dừng local optimum.
- Trực giác: ta xây dựng Voronoi partition trong không gian latent; mỗi centroid là barycenter của cluster.
- Đối với VQ-VAE, assignment step diễn ra trong forward pass (argmin), update step được thực hiện qua EMA.

### 4.3. Chứng minh hội tụ và điều kiện tối ưu cấp hai

- **Hội tụ**: distortion $D = \sum_k \sum_{z \in S_k} \|z - e_k\|^2$ giảm không tăng qua mỗi vòng.
- Lý do: assignment step giảm $D$ vì chọn cluster tối ưu; update step tính trung bình → gradient zero.
- **Điều kiện cần**:  
  - $S_k$ là Voronoi region: $\forall z \in S_k, \|z - e_k\|^2 \le \|z - e_j\|^2$ với mọi $j$.
  - $e_k$ là centroid: $\nabla_{e_k} D = 0$ ⇒ $e_k = \frac{1}{|S_k|} \sum_{z \in S_k} z$.
- **Điều kiện đủ cục bộ**: Hessian là ma trận dương xác định trong không gian tangent.

### 4.4. K-means, GLA và các biến thể

- **K-means**: giống Lloyd nhưng với dữ liệu i.i.d. offline.
- **Online k-means**: cập nhật codeword từng điểm (stochastic).
- **Mini-batch k-means**: dùng batch nhỏ, update incremental (Google 2010).
- **Elkan k-means**: dùng bất đẳng thức tam giác để tăng tốc.
- **Product quantization (PQ)**: chia vector $z$ thành $m$ subvectors, lượng tử hoá độc lập để giảm chi phí.
- **Residual quantization (RQ)**: lượng tử hoá phần dư sau mỗi bước, kết hợp codebook theo dạng additive.

### 4.5. Product Quantization, Residual Quantization, Ternary Quantization

1. **Product Quantization (PQ)**  
   - $z = [z^{(1)}, \ldots, z^{(m)}]$ với $z^{(i)} \in \mathbb{R}^{d/m}$.
   - Mỗi subvector có codebook riêng $\mathcal{E}^{(i)}$.
   - Token cuối cùng là tuple $(k_1, \ldots, k_m)$ ⇒ tiết kiệm bộ nhớ.
   - Dùng nhiều trong FAISS, CLIP vector store.

2. **Residual Quantization (RQ)**  
   - Lần 1 lượng tử hoá $z$ bằng $e_{k_1}$, lấy residual $r_1 = z - e_{k_1}$.
   - Lần 2 lượng tử hoá $r_1$ bằng $e_{k_2}$, ...
   - Reconstruction $\hat{z} = e_{k_1} + e_{k_2} + \cdots$.
   - RQ cho phép chất lượng cao với codebook nhỏ.

3. **Ternary / Binary Quantization**  
   - Hạn chế giá trị codeword vào $\{-1, 0, 1\}$ hoặc $\{-1, 1\}$.
   - Giảm chi phí suy luận (bitwise operations) nhưng distortion cao.
   - Ít dùng cho ảnh VLM, nhưng hữu ích trong inference edge.

### 4.6. Geometric view: Voronoi diagram trên manifold ảnh

- Không gian latent của ảnh thường nằm trên manifold thấp chiều.
- Vector quantization tương ứng phủ manifold bằng các cell Voronoi.
- Độ chính xác phụ thuộc vào:
  1. Số codeword $K$.
  2. Phân bố codeword – cần thích ứng với mật độ dữ liệu.
  3. Metric dùng để đo khoảng cách (thường Euclid; có thể thay bằng Mahalanobis).
- Trong VQ-GAN, latent dimension $d=256$ → codeword 256 chiều; Voronoi cell khó trực quan nhưng khái niệm vẫn đúng.

---

## 5. Từ lý thuyết đến kiến trúc cụ thể

### 5.1. VQ-VAE: thiết kế encoder–decoder và codebook

- Kiến trúc cơ bản:
  - **Encoder $E_\phi$**: convolutional stack → downsample → output $Z_e \in \mathbb{R}^{h \times w \times d}$.
  - **Quantizer $Q$**: với mỗi vị trí $(i, j)$ chọn codeword $e_{k}$ gần nhất.
  - **Decoder $D_\theta$**: transposed conv → upsample → reconstruct $\hat{X}$.
- Codebook $\mathcal{E} \in \mathbb{R}^{K \times d}$ là tham số học được.
- Số lượng token $L = h \cdot w$. Ví dụ input $256 \times 256$ với stride 16 → $h = w = 16$ ⇒ $L=256$.
- Thông số thường dùng:
  - $K = 1024$.
  - $d = 256$.
  - Encoder depth 4 block ResNet.

### 5.2. Cơ chế loss và gradient của VQ-VAE

Loss gốc từ [Van den Oord et al., 2017]:

$$
\mathcal{L} = \|x - \hat{x}\|_2^2 + \| \text{sg}[z_e] - e_k\|_2^2 + \beta \| z_e - \text{sg}[e_k] \|_2^2.
$$

- **Term 1**: reconstruction loss.
- **Term 2**: mã hóa codebook (stop-gradient trên $z_e$).
- **Term 3**: commitment loss ($\beta$ thường 0.25).

**Gradient phân tích**:

- W.r.t. decoder $\theta$: $\nabla_\theta \mathcal{L} = 2 (x - \hat{x}) \frac{\partial \hat{x}}{\partial \theta}$.
- W.r.t. codeword $e_k$:  
  $\nabla_{e_k} \mathcal{L} = 2 (e_k - \text{sg}[z_e]) + 2 \beta (e_k - z_e)$.
- W.r.t. encoder:  
  $\nabla_{z_e} \mathcal{L} = 2 \beta (z_e - \text{sg}[e_k]) + \text{grad from reconstruction via decoder}$.
- Sử dụng Straight-Through Estimator: gradient từ decoder được copy trực tiếp từ $\hat{x}$ về $z_e$.

### 5.3. EMA codebook update và phân tích ổn định

- Để tránh phải update codebook qua gradient, VQ-VAE áp dụng EMA:
  - $N_k \leftarrow \gamma N_k + (1 - \gamma) \sum_{i,j} \mathbb{1}[k_{ij} = k]$.
  - $m_k \leftarrow \gamma m_k + (1 - \gamma) \sum_{i,j} \mathbb{1}[k_{ij} = k] z_{e, ij}$.
  - $e_k \leftarrow \frac{m_k}{N_k}$.
- Với $\gamma \approx 0.99$, update mượt mà.
- **Ổn định**:
  - Nếu codeword không được sử dụng (dead code), $N_k$ nhỏ → $e_k$ chậm cập nhật.
  - Khắc phục: reset codeword random khi $N_k$ < threshold.
  - Dùng temperature annealing cho assignment (soft) giai đoạn đầu.

### 5.4. VQ-VAE-2 và hierarchical latent

- VQ-VAE-2 (Razavi et al., 2019) dùng hai tầng latent:
  1. Latent coarse cấp cao $z^{(1)}$ (resolution nhỏ) – capture global structure.
  2. Latent fine cấp thấp $z^{(2)}$ – capture chi tiết.
- Decoder generate theo hai cấp:  
  - $D_1$ tái tạo feature coarse.  
  - $D_2$ kết hợp coarse + token fine.
- Lợi ích:
  - Chất lượng ảnh cao (FID thấp).
  - Sampling hierarchical: sample coarse → fine.
- Đối với VLM, hierarchical token cho phép LLM ưu tiên token coarse (nội dung chính) nếu context hạn hẹp.

### 5.5. dVAE và Gumbel-Softmax relaxation

- dVAE (DALL·E) thay VQ bằng distribution rời rạc với reparameterizable sample.
- Forward:
  - Encoder output logits $l \in \mathbb{R}^K$.
  - Sample token $k$ bằng Gumbel-Softmax:
    $$
    y_k = \frac{\exp((l_k + g_k)/\tau)}{\sum_j \exp((l_j + g_j)/\tau)}, \quad g_k \sim \text{Gumbel}(0, 1).
    $$
  - $z_q = \sum_k y_k e_k$ (soft combination).
- Khi $\tau \to 0$, $y_k$ gần one-hot.
- Advantage: gradient chính xác hơn STE.
- Drawback: training phức tạp, cần annealing $\tau$.

### 5.6. VQ-GAN: perceptual loss và discriminator

- VQ-GAN [Esser et al., 2021] thêm thành phần GAN để cải thiện chi tiết:
  $$
  \mathcal{L}_{\text{vq-gan}} = \lambda_{\text{rec}} \|x - \hat{x}\|_1 + \lambda_{\text{perc}} \sum_l \|\phi_l(x) - \phi_l(\hat{x})\|_2^2 + \mathcal{L}_{\text{GAN}} + \mathcal{L}_{\text{commit}}.
  $$
- $\phi_l$ là feature VGG; $\mathcal{L}_{\text{GAN}}$ là loss logistic GAN (PatchGAN).
- Training alternating:
  - Update $D$ để phân biệt $x$ và $\hat{x}$.
  - Update $E, Q, D_\theta$ để đánh lừa $D$.
- Kết quả: ảnh tái tạo sắc nét, giữ textura.
- VQ-GAN là nền tảng cho Stable Diffusion (latent space).

### 5.7. MaskGIT và masked token modeling

- MaskGIT [Chang et al., 2022] xem token ảnh như sequence và đào tạo transformer autoregressive masked.
- Training:
  1. Encode ảnh → token.
  2. Random mask subset (tỷ lệ p).
  3. Transformer dự đoán token bị mask.
- Sampling song song:
  - Khởi tạo tất cả token = `[MASK]`.
  - Transformer dự đoán distribution cho từng vị trí.
  - Chọn top-$k$ token theo temperature schedule, điền vào.
  - Lặp cho tới khi không còn mask.
- Ưu điểm: generation nhanh (tận dụng song song), quality cao.
- Trong bối cảnh token integration, MaskGIT gợi ý cách fine-tune LLM để đoán token missing – hữu ích khi LLM cần sửa token bị lỗi.

### 5.8. MAE-VQ và token learning adaptive

- MAE-VQ kết hợp Masked AutoEncoder và VQ để học token adaptively.
- Encoder ViT -> mask patch -> reconstruct patch.
- Learning objective: minimize reconstruction + encourage token sparse.
- Lợi ích: drop patch redundant, giảm token number.
- Là bước đệm cho TokenLearner/TokenMerging (sẽ phân tích sâu trong Part II khi nói về integration với LLM).

### 5.9. Ví dụ định lượng: lượng tử hoá patch $2 \times 2$

Để trực giác hơn, hãy xét patch grayscale $P = \begin{bmatrix}10 & 12 \\ 11 & 9\end{bmatrix}$.

1. Biến patch thành vector $z = [10, 12, 11, 9]$.
2. Giả sử codebook $\mathcal{E}$ gồm hai codeword:
   - $e_1 = [9, 11, 10, 8]$.
   - $e_2 = [20, 22, 21, 19]$.
3. Tính khoảng cách:
   - $\|z - e_1\|_2^2 = (1)^2 + (1)^2 + (1)^2 + (1)^2 = 4$.
   - $\|z - e_2\|_2^2 = (10)^2 + (10)^2 + (10)^2 + (10)^2 = 400$.
4. Chọn $e_1$ ⇒ token = 1.
5. Distortion patch = $\frac{4}{4} = 1$ pixel intensity.

Ví dụ cho thấy codebook cần bao phủ dải giá trị lớn; patch brightness khác biệt cần codeword riêng. Trong thực tế, vector dimension cao (256) nên difference phức tạp hơn, nhưng logic tương tự.

### 5.10. Liên hệ với clustering và density estimation

- VQ có thể xem như clustering; tuy nhiên, mục tiêu VQ là minimize distortion có trọng số theo phân phối dữ liệu.
- Nếu thêm regularizer entropy, ta thực thi clustering mềm (soft assignments).
- Codebook phản ánh density cao: nhiều codeword tập trung vào vùng manifold mật độ lớn; density thấp → codeword ít.
- Điều này tương đương Kernel Density Estimation với kernel Dirac; do đó VQ-VAE có thể coi là estimator của phân phối latent.

### 5.11. So sánh với transform coding (DCT)

- JPEG dùng Discrete Cosine Transform (DCT) + scalar quantization.
- VQ-VAE học transform phi tuyến $E_\phi$ + vector quantization.
- Ưu điểm học sâu:
  - Adapt theo thống kê dataset, không cố định như DCT.
  - Chịu được variation phức tạp (texture, phong cách).
- Nhược điểm:
  - Cần training và GPU.
  - Bảo đảm chết code (dead blocks) phức tạp.

### 5.12. Regularization bổ sung cho codebook

- **Orthogonality penalty**: $\lambda \sum_{i \neq j} |e_i^\top e_j|$ giúp codeword phân tán.
- **Norm penalty**: $||e_k||_2$ trong khoảng [a, b] để tránh codeword norm quá lớn.
- **Diversity loss**: maximize determinant của Gram matrix, khuyến khích spanning.

### 5.13. Batch norm vs instance norm trong encoder

- BatchNorm giúp training ổn định nhưng phụ thuộc batch size.
- InstanceNorm giữ style invariance, phù hợp dataset đa dạng.
- Thực nghiệm: VQ-VAE cho ảnh natural -> BatchNorm OK; ảnh nghệ thuật (đa phong cách) -> InstanceNorm tốt hơn.

### 5.14. Gradient clipping và learning rate schedule

- Gradient clipping (global norm 5–10) tránh explosion khi codebook chưa ổn định.
- Learning rate:
  - Encoder/decoder: warmup 10k step, LR 3e-4.
  - Codebook EMA: implicit LR $(1-\gamma)$; gamma 0.99 → lr 0.01.

### 5.15. Phân tích độ phức tạp tính toán

- Encoder/decoder complexity: $\mathcal{O}(HWCd)$.
- Quantizer: distance computation $\mathcal{O}(BHWK)$ → bottleneck.
- Tối ưu:
  - Dùng `torch.cdist` (vectorized).
  - Dùng FAISS GPU index IVFPQ nếu $K$ lớn (≥ 16k).
  - Áp dụng k-means++ khởi tạo để giảm epoch.

### 5.16. Trực quan hóa codebook

- PCA 2D: giảm codeword 256-dim xuống 2D, plot scatter.
- t-SNE/UMAP: highlight cluster (màu).
- Có thể so sánh activation overlay: với ảnh, map token ID -> màu để xem segmentation implicit.

### 5.17. Ghi chú về VQ cho video

- Video latent: $Z \in \mathbb{R}^{T \times H \times W \times d}$.
- Lượng tử 3D: codebook index tuple $(k_t, k_h, k_w)$.
- Một số model (MAGVIT) flatten thời gian + không gian: token = (time*space).
- Distortion function: weighted sum (có thể weight temporal).

### 5.18. Lượng tử hoá bất đối xứng (asymmetric quantization)

- Encoder và decoder dùng codebook khác nhau.
- Được dùng trong Asymmetric Numeral Systems (ANS) khi encode: store index + residual.
- Cho phép decoder light-weight (ít codeword), encoder heavy (nhiều codeword).

### 5.19. Kết nối với học biểu diễn tự giám sát

- VQ-VAE latent có thể dùng cho downstream tasks.
- Pretraining VQ-VAE sau đó fine-tune ViT (mapping token -> embedding).
- Tokenization + contrastive learning (ví dụ, DINO-VQ).

### 5.20. Bài học từ thực nghiệm của cộng đồng

- Từ open-source (taming-transformers):
  - Codebook size 1024 cho ảnh 256².
  - Beta commitment 0.25.
  - EMA decay 0.99.
- Một số nhóm (Kakaobrain, Stability) dùng codebook 8192 cho dataset lớn (LAION) → better fidelity nhưng context nặng.

---

---

## 6. Đo lường chất lượng: distortion, perceptual, task-level

### 6.1. PSNR, SSIM, MS-SSIM: công thức và limitations

- **PSNR**:
  $$
  \text{PSNR} = 10 \log_{10} \left(\frac{MAX_I^2}{\text{MSE}}\right).
  $$
  - $MAX_I$ thường là 1 hoặc 255.
  - Nhược điểm: nhạy với sai số pixel-level, không phản ánh cảm nhận.

- **SSIM**:  
  $$
  \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}.
  $$
  - Đo tương đồng về độ sáng, tương phản, cấu trúc.
  - MS-SSIM: đa thang đo.

- Hạn chế: SSIM vẫn không tương quan hoàn hảo với perception; VQ-GAN thêm perceptual loss để bù.

### 6.2. LPIPS, DISTS, FID, KID: ý nghĩa với VLM

- **LPIPS** (Learned Perceptual Image Patch Similarity):  
  - Tính $\|\phi(x) - \phi(y)\|_2$ trên feature AlexNet/VGG đã hiệu chỉnh.
  - Rất hữu ích khi đánh giá tokenization vì capture cảm nhận.

- **DISTS**: kết hợp distortion + structure, hiệu quả cho dataset natural image.

- **FID** (Fréchet Inception Distance):
  $$
  \text{FID} = \|\mu_x - \mu_y\|^2 + \text{Tr}(\Sigma_x + \Sigma_y - 2(\Sigma_x \Sigma_y)^{1/2}).
  $$
  - Đánh giá phân phối ảnh tái tạo so với ảnh thật.

- **KID** (Kernel Inception Distance): MMD với polynomial kernel, unbiased hơn FID khi sample nhỏ.

### 6.3. CLIPScore, BLIPScore và Alignment metric

- CLIPScore = cosine similarity giữa embedding ảnh và text.
- Dùng khi tokenization để feed LLM: tái tạo xong, ghép caption gốc, tính score.
- BLIPScore: tương tự CLIP nhưng dùng BLIP-2; nhạy với semantics.
- Ngoài ra, LOGIT-ACC (từ LLaVA) đo ability align token-cum-text.

### 6.4. Token diversity metrics: perplexity, utilization

- **Perplexity**:  
  $$
  \text{PP} = \exp\left(-\sum_{k=1}^K \hat{p}_k \log \hat{p}_k\right).
  $$
  - PP càng gần $K$ ⇒ codebook sử dụng đồng đều.

- **Utilization**: $\frac{\text{Số codeword được dùng}}{K}$.
  - Theo dõi dead code.

- **Average distance**: $\frac{1}{L} \sum_i \|z_e^{(i)} - e_{k_i}\|^2$.
  - Phản ánh chất lượng lượng tử hoá.

---

## 7. Timeline nghiên cứu 2017–2021 chi tiết

| Năm | Công trình | Đóng góp chính | Kết quả định lượng |
|-----|------------|----------------|-------------------|
| 2017 | VQ-VAE | Giới thiệu vector quantization trong AutoEncoder | Bits-per-dim (CIFAR-10) 2.85 |
| 2018 | VQ-VAE-2 | Hierarchical latent cho ảnh 256² | FID 51.4 trên ImageNet |
| 2019 | dVAE (DALL·E) | Gumbel-softmax discrete latent | Generated image quality cao |
| 2020 | HiFiC | GAN-based compression | MS-SSIM 0.93 @0.14bpp |
| 2020 | Jukebox (OpenAI) | VQ-VAE cho audio nhiều tầng | 3 tầng codebook cho audio 44kHz |
| 2021 | VQ-GAN | Kết hợp GAN + perceptual loss | FID 7.4 @256² |
| 2021 | MaskGIT | Parallel token generation | FID 6.18 @256² |

- Timeline cho thấy xu hướng: từ mô hình tái tạo (VQ-VAE) sang mô hình generative (MaskGIT) – mỗi bước cải thiện fidelity và tốc độ.
- Những kết quả này đặt nền tảng cho các model VLM như PaLM-E, IDEFICS, Gemini – nơi token ảnh phải gọn nhẹ nhưng giữ semantics.

---

## 8. Hạ tầng triển khai encoder lượng tử hoá

### 8.1. Design hardware pipeline và throughput

- **Mục tiêu**: encode 10 ảnh/giây để phục vụ streaming cho khách.
- Pipeline GPU:
 1. Load batch 8 ảnh từ disk (JPEG → tensor).
 2. Chuẩn hóa + resize (CUDA kernel).
 3. Encoder forward (FP16) → $Z_e$.
 4. Lượng tử hoá (CUDA op, argmin).
 5. Mapping index → token string.
- Cần tối ưu:
  - Dùng `torch.compile` (PyTorch 2.x) cho encoder.
  - Chuyển codebook sang GPU constant memory để truy cập nhanh.
  - Batch size 16 cho throughput cao.

### 8.2. Data loading, augmentation, mixed precision

- DataLoader:
  - `num_workers` ≥ 8 (tuỳ CPU).
  - `prefetch_factor` 4.
- Augmentation:
  - Random crop, color jitter (đảm bảo alignment text).
  - Blur gaussian (chống overfit chi tiết nhỏ).
- Mixed precision:
  - `torch.cuda.amp.Autocast` cho encoder/decoder.
  - Đối với quantizer, tính khoảng cách ở FP32 để tránh sai sót.

### 8.3. Quản lý checkpoint, logging và reproducibility

- Checkpoint:
  - Lưu $E_\phi$, $D_\theta$, codebook $\mathcal{E}$, EMA stats ($N_k, m_k$).
  - Dùng `safetensors` để giảm nguy cơ hỏng file.
- Logging:
  - WandB/TensorBoard track loss, PSNR, LPIPS, perplexity.
  - Log distribution codeword (histogram).
- Reproducibility:
  - Set seed cho PyTorch, NumPy, CUDA.
  - Ghi config YAML (learning rate, $\lambda$, $K$, stride).

---

## 9. Ví dụ PyTorch chi tiết: xây dựng VQ-VAE từ đầu

Đoạn code dưới đây không dùng thư viện ngoài, tự cài đặt quantizer, EMA và training loop.
Phần II sẽ xây dựng tiếp pipeline tích hợp với LLM.

```python
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return F.relu(out + residual)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 128, z_channels: int = 256):
        super().__init__()
        layers = []
        channels = hidden_channels
        layers.append(nn.Conv2d(in_channels, channels, 4, stride=2, padding=1))
        layers.append(nn.ReLU())
        for _ in range(3):
            layers.append(nn.Conv2d(channels, channels, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
        for _ in range(2):
            layers.append(ResidualBlock(channels))
        layers.append(nn.Conv2d(channels, z_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, hidden_channels: int = 128, z_channels: int = 256):
        super().__init__()
        layers = []
        channels = hidden_channels
        layers.append(nn.Conv2d(z_channels, channels, 3, padding=1))
        for _ in range(2):
            layers.append(ResidualBlock(channels))
        for _ in range(3):
            layers.append(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(channels, out_channels, 4, stride=2, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z_q))


@dataclass
class VQConfig:
    embedding_dim: int = 256
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5


class VectorQuantizerEMA(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_embeddings = config.num_embeddings
        self.commitment_cost = config.commitment_cost
        self.decay = config.decay
        self.epsilon = config.epsilon

        embedding = torch.randn(self.num_embeddings, self.embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("embedding_ema", embedding.clone())

    def forward(self, z_e: torch.Tensor):
        # z_e: [B, C, H, W] -> [BHW, C]
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z.view(-1, self.embedding_dim)

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)

        quantized = encodings @ self.embedding
        quantized = quantized.view(z.shape)

        if self.training:
            encodings_sum = encodings.sum(dim=0)
            self.cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)

            embed_sum = encodings.t() @ flat
            self.embedding_ema.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embedding = self.embedding_ema / cluster_size.unsqueeze(1)

        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        embedding_loss = F.mse_loss(quantized, z.detach())

        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, commitment_loss, embedding_loss, encoding_indices.view(z_e.size(0), -1)


class VQVAE(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()
        self.encoder = Encoder()
        self.vector_quantizer = VectorQuantizerEMA(config)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, commit, embed, codes = self.vector_quantizer(z_e)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + commit + embed
        metrics = {
            "recon_loss": recon_loss.detach(),
            "commit_loss": commit.detach(),
            "embed_loss": embed.detach(),
        }
        return x_recon, loss, metrics, codes


def train_epoch(model, dataloader, optimizer, scaler=None, device="cuda"):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images = batch.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                _, loss, _, _ = model(images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss, _, _ = model(images)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)
```

Đoạn code trên tạo một VQ-VAE chuẩn với EMA codebook.
Bạn có thể ghép thêm DataLoader CIFAR-10 và chạy training.
Trong thực tế, thêm LPIPS loss và perceptual giúp chất lượng tốt hơn.

### 9.1. Chuẩn bị DataLoader CIFAR-10 chuẩn

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
```

### 9.2. Vòng huấn luyện hoàn chỉnh với logging

```python
import tqdm

config = VQConfig()
model = VQVAE(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(100):
    epoch_loss = 0.0
    model.train()
    for images, _ in tqdm.tqdm(dataloader):
        images = images.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            _, loss, metrics, codes = model(images)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * images.size(0)
    epoch_loss /= len(dataloader.dataset)
    print(f"Epoch {epoch} | loss={epoch_loss:.4f} | recon={metrics['recon_loss']:.4f}")
```

### 9.3. Đánh giá perplexity và token usage

```python
def evaluate_token_usage(model, dataloader):
    model.eval()
    counts = torch.zeros(model.vector_quantizer.num_embeddings, device=device)
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _, _, _, codes = model(images)
            counts.scatter_add_(0, codes.view(-1), torch.ones_like(codes.view(-1), dtype=torch.float32))
    probs = counts / counts.sum()
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    perplexity = torch.exp(entropy)
    utilization = (counts > 0).float().mean()
    return perplexity.item(), utilization.item()
```

### 9.4. Lưu checkpoint và resume

```python
state = {
    "encoder": model.encoder.state_dict(),
    "decoder": model.decoder.state_dict(),
    "embedding": model.vector_quantizer.embedding,
    "embedding_ema": model.vector_quantizer.embedding_ema,
    "cluster_size": model.vector_quantizer.cluster_size,
    "optimizer": optimizer.state_dict(),
}
torch.save(state, "vqvae_checkpoint.pt")
```

Khi resume, load lại buffer `embedding`, `embedding_ema`, `cluster_size` để EMA tiếp tục chính xác.

### 9.5. Xuất token và ánh xạ sang text

```python
def image_to_token_strings(model, image_tensor):
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(image_tensor.unsqueeze(0).to(device))
        _, _, _, codes = model.vector_quantizer(z_e)
    tokens = []
    for idx in codes.view(-1).cpu().tolist():
        tokens.append(f"<v{idx}>")
    return tokens
```

Sequence token thu được sẽ sử dụng trong Part II khi xây prompt LLM.

### 9.6. Visualize tái tạo

```python
import torchvision.utils as vutils

model.eval()
with torch.no_grad():
    for images, _ in dataloader:
        recon, _, _, _ = model(images.to(device))
        grid = torch.cat([images[:8], recon[:8]], dim=0)
        vutils.save_image(grid, "recon.png", nrow=8, normalize=True, range=(-1, 1))
        break
```

Ảnh `recon.png` chứa hàng đầu là ảnh gốc, hàng dưới là ảnh tái tạo.

### 9.7. Lưu token thành file JSON

```python
import json

def dump_tokens(model, dataloader, output_path="tokens.jsonl"):
    model.eval()
    with torch.no_grad(), open(output_path, "w") as f:
        for idx, (images, labels) in enumerate(dataloader):
            z_e = model.encoder(images.to(device))
            _, _, _, codes = model.vector_quantizer(z_e)
            for i in range(images.size(0)):
                token_ids = codes[i].view(-1).cpu().tolist()
                record = {
                    "id": f"{idx}_{i}",
                    "tokens": token_ids,
                    "label": int(labels[i]),
                }
                f.write(json.dumps(record) + "\n")
```

File JSONL này là đầu vào lý tưởng cho prompt builder ở Part II.

---

## 10. Checklist kỹ thuật và bài học thực nghiệm

1. **Chọn $K$ phù hợp**:  
   - 512 cho dataset nhỏ; 1024–4096 cho dataset lớn.  
   - Nếu perplexity thấp, tăng $K$ hoặc regularization.

2. **Stride encoder**:  
   - Stride 16 → 256 token.  
   - Stride 8 → 1024 token (chi tiết hơn nhưng context lớn).

3. **Commitment $\beta$**:  
   - 0.25 là mặc định.  
   - Nếu latent thay đổi chậm, tăng $\beta$ để khuyến khích bám codeword.

4. **Warm-up EMA**:  
   - Trong 1000 step đầu, update codebook bằng mini-batch mean (không EMA) để tránh dead code.

5. **Monitoring**:  
   - Track histogram distance `||z_e - e_k||`.  
   - Track reconstruction metrics: PSNR, LPIPS.  
   - Track token usage.

6. **Finite precision**:  
   - Lưu ý jitter khi compute argmin.  
   - Có thể thêm noise Gaussian nhỏ trước lượng tử để regularize.

7. **Bảo toàn màu sắc**:  
   - Thêm chroma loss (YUV space) nếu dataset nhiều màu.

8. **Đồng bộ với caption**:  
   - Khi train cho VLM, augment text tương ứng (chú thích).
   - Bảo đảm augmentation không làm text sai.

9. **Theo dõi thời gian encode**:  
   - Log `tokens_per_second`.  
   - So sánh CPU vs GPU quantization.

10. **Bảo toàn metadata**:  
    - Lưu mapping pixel (x, y) ↔ token ID.  
    - Dùng JSON schema chuẩn để LLM trích dẫn.

11. **Kiểm thử với câu hỏi mô phỏng**:  
    - Dù Part I chưa vào integration, bạn nên sớm kiểm xem token tái tạo caption đúng hay không.  
    - Tạo script CLIPScore early warning.

12. **Khởi tạo codebook bằng k-means**:  
    - Trước khi training, chạy k-means 10k sample để init.  
    - Giảm thời gian warmup.

13. **Phân tích sensitivity**:  
    - Thử thay stride, beta, learning rate → log metric.  
    - Tạo bảng sensitivity để Part II tối ưu integration.

14. **Áp dụng progressive training**:  
    - Huấn luyện trên ảnh 128² -> finetune 256².  
    - Codebook reuse, chỉ fine-tune decoder.

15. **Đảm bảo reproducibility**:  
    - Ghi script `requirements.txt`.  
    - Lưu seed, commit git.

---

## 11. Kết luận phần I và giới thiệu phần II

Trong Phần I, chúng ta đã:

- Thiết lập cơ sở lý thuyết rate–distortion, entropy, mutual information.
- Phân tích sâu vector quantization và các biến thể.
- Trình bày kiến trúc VQ-VAE, VQ-GAN, MaskGIT cùng từng công thức loss.
- Cung cấp code VQ-VAE đầy đủ và checklist thực nghiệm.

Phần II sẽ nối tiếp với:

- Neural codec hyperprior, entropy model nâng cao.
- PaLM-E, Kosmos-2, IDEFICS-2 – cách họ biến token ảnh thành ngôn ngữ.
- Chiến lược tích hợp token vào LLM, computation analysis.
- Training pipeline đa giai đoạn (SFT, RLHF), code LLM integration.
- Evaluation, benchmark, hướng nghiên cứu tương lai.

Bạn có thể chuyển sang [Compression & Token Integration (Phần II)](/posts/2025/vlm-compression-token-integration-part2) ngay sau khi hoàn thành phần này.

---

## 12. Phụ lục A: Ký hiệu và quy ước

- $X$: biến ngẫu nhiên ảnh gốc.
- $\hat{X}$: ảnh tái tạo.
- $Z_e$: latent encoder (liên tục).
- $Z_q$: latent lượng tử hoá.
- $\mathcal{E}$: codebook.
- $e_k$: codeword thứ $k$.
- $K$: số codeword.
- $d(\cdot, \cdot)$: distortion function.
- $\lambda$: hệ số trade-off rate–distortion.
- $E_\phi$, $D_\theta$: encoder và decoder tham số.
- $\text{sg}[\cdot]$: stop-gradient.
- $R$: bitrate (bit/token hoặc bit/pixel).
- $D$: distortion kỳ vọng.
- $L$: số token ảnh.
- $K_t$: hyper-parameter temperature.

---

## 13. Phụ lục B: Chứng minh bổ sung

### 13.1. Gradient của loss VQ-VAE với STE

Giả sử reconstruction loss $\ell(\hat{x}, x)$ có gradient $\nabla_{\hat{x}} \ell$.
Do $z_q = z_e + (q - z_e).detach()$, ta có:

$$
\frac{\partial z_q}{\partial z_e} = I.
$$

Do đó gradient từ decoder truyền về encoder như autoencoder chuẩn.
Commitment loss gradient:

$$
\nabla_{z_e} \beta \| z_e - \text{sg}[e_k] \|^2 = 2 \beta (z_e - e_k).
$$

Cộng với gradient reconstruction → tổng gradient cho encoder.

### 13.2. KL giữa uniform prior và empirical code distribution

Nếu prior $p(z)$ uniform trên $\{1, \ldots, K\}$ và empirical distribution $\hat{p}$, ta có:

$$
D_{\text{KL}}(\hat{p} \| p) = \sum_k \hat{p}_k \log \frac{\hat{p}_k}{1/K} = \log K - H(\hat{p}).
$$

Điều này cho thấy maximizing entropy $\hat{p}$ tương đương minimize KL.

### 13.3. Công thức FID chi tiết

Cho 2 tập feature $F_X, F_Y$ (Inception pool3), mean $\mu_X, \mu_Y$, covariance $\Sigma_X, \Sigma_Y$.
FID:

$$
\|\mu_X - \mu_Y\|_2^2 + \text{Tr}\left(\Sigma_X + \Sigma_Y - 2 (\Sigma_X \Sigma_Y)^{1/2}\right).
$$

Ma trận tích căn bậc hai tính bằng SVD.

---

## 14. Tài liệu tham khảo

1. Van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning.* NeurIPS.  
2. Razavi, A., Oord, A. V. D., & Vinyals, O. (2019). *Generating Diverse High-Fidelity Images with VQ-VAE-2.* NeurIPS.  
3. Esser, P., Rombach, R., & Ommer, B. (2021). *Taming Transformers for High-Resolution Image Synthesis.* CVPR.  
4. Mentzer, F., Tschannen, M., Agustsson, E., & Timofte, R. (2020). *High-Fidelity Image Compression.* IEEE TPAMI / CVPR.  
5. Chang, H., Bharadhwaj, H., Park, H., & Ramanan, D. (2022). *MaskGIT: Masked Generative Image Transformer.* CVPR.  
6. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.  
7. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory.* Wiley.  
8. Gray, R. (1984). *Vector Quantization.* IEEE ASSP Magazine.  
9. Balle, J., Laparra, V., & Simoncelli, E. P. (2017). *End-to-End Optimized Image Compression.* ICLR.  
10. Lee, Y., Kim, J., & Choi, S. (2022). *Vision Transformer based Vector Quantization.* arXiv preprint.  
11. Alex, X. et al. (2023). *MAE-VQ: Masked Autoencoder with Vector Quantization.* arXiv preprint.  
12. Goodfellow, I., et al. (2014). *Generative Adversarial Networks.* NeurIPS.  
13. Petrov, M., et al. (2023). *TokenLearner Revisited for Multimodal Fusion.* ECCV Workshop.  
14. Oord, A. V. D., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv.  
15. Rissanen, J. (1978). *Modeling by Shortest Data Description.* Automatica.  
16. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.  
17. Deng, J., et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR.  
18. Zhang, R., et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR.  
19. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS.  
20. Parmar, N., et al. (2018). *Image Transformer.* ICML.

---

<script src="/assets/js/katex-init.js"></script>
