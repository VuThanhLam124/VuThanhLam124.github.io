---
title: "SOTA Multimodal Reasoning: VLM thế hệ mới"
date: "2025-04-06"
category: "vision-language-models"
tags: ["vlm", "flamingo", "gpt-4v", "gflowvlm", "reasoning"]
excerpt: "Bài tổng hợp theo phong cách literature review: cô hướng dẫn viên bảo tàng đối chiếu Flamingo, GPT-4V, Gemini, LLaVA-1.5, Qwen-VL và GFlowVLM với chứng cứ từ NeurIPS, TPAMI, arXiv; kèm phân tích toán, code và chiến lược vận hành."
author: "ThanhLamDev"
readingTime: 34
featured: true
---

# SOTA Multimodal Reasoning: Giúp hướng dẫn viên suy luận đa bước

**Cô hướng dẫn viên tại Bảo tàng Giao Thoa đã căn chỉnh (alignment) và hòa trộn (fusion) thành công hai luồng thị giác – ngôn ngữ. Để chuẩn bị mùa triển lãm mới, cô dành cả tuần đọc hai bài tổng quan TPAMI 2024 về Multimodal Foundation Models [2][3], cùng liên hệ trực tiếp với các nhóm nghiên cứu tại NeurIPS, CVPR, ICML. Mục tiêu: thuyết phục ban giám đốc rằng hệ thống dẫn tour phải nâng cấp lên thế hệ reasoning mới – có bộ nhớ dài, chuỗi suy luận (chain-of-thought) minh bạch và khả năng gọi công cụ. Bài viết này trình bày kết quả “literature review” của cô: đối chiếu các mô hình Flamingo, GPT-4V, Gemini, LLaVA-1.5, Qwen-VL, GFlowVLM, phân tích kiến trúc, số liệu benchmark, bài học vận hành và ví dụ code triển khai.**

## Mục lục

1. [Bối cảnh: tại sao bảo tàng cần reasoning sâu?](#1-bối-cảnh-tại-sao-bảo-tàng-cần-reasoning-sâu)
2. [Bản đồ SOTA & literature review 2022–2025](#2-bản-đồ-sota--literature-review-20222025)
3. [Flamingo: Perceiver Resampler và gated cross-attention](#3-flamingo-perceiver-resampler-và-gated-cross-attention)
4. [GPT-4V và Gemini: Instruction-following đa vòng + RLHF](#4-gpt-4v-và-gemini-instruction-following-đa-vòng--rlhf)
5. [Open-source challengers: LLaVA-1.5, Qwen-VL, Kosmos-2](#5-open-source-challengers-llava-15-qwen-vl-kosmos-2)
6. [GFlowVLM: Flow matching cho chuỗi suy luận đa phương thức](#6-gflowvlm-flow-matching-cho-chuỗi-suy-luận-đa-phương-thức)
7. [Bộ nhớ, context dài và tool-use orchestration](#7-bộ-nhớ-context-dài-và-tool-use-orchestration)
8. [Recipe huấn luyện reasoning SOTA cho bảo tàng](#8-recipe-huấn-luyện-reasoning-sota-cho-bảo-tàng)
9. [Ví dụ PyTorch: Multimodal ReAct Agent có logging minh bạch](#9-ví-dụ-pytorch-multimodal-react-agent-có-logging-minh-bạch)
10. [Đánh giá, minh bạch và phân tích lỗi](#10-đánh-giá-minh-bạch-và-phân-tích-lỗi)
11. [Liên kết với các bài tiếp theo](#11-liên-kết-với-các-bài-tiếp-theo)
12. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. Bối cảnh: tại sao bảo tàng cần reasoning sâu?

Sau khi giải quyết bài toán alignment và fusion, cô hướng dẫn viên nhận ra những hạn chế rõ rệt:

- Khách nước ngoài muốn so sánh hoa văn ở **hai** bức tranh đặt ở hai phòng khác nhau → hệ thống phải giữ context xuyên suốt hành trình.
- Khách học sinh yêu cầu giải bài toán lịch sử dựa trên sơ đồ chiến trận được trưng bày → cần đọc biểu đồ, suy luận từng bước và diễn giải lại.
- Khách khiếm thị mong đợi lời giải thích tường minh: “Tại sao tác phẩm này biểu tượng cho mùa thu?” → chuỗi suy luận trung gian phải minh bạch, không chỉ có đáp án cuối.

Những yêu cầu này nằm ở tầng **reasoning đa bước** – vượt xa khả năng matching hoặc captioning đơn giản. Để đảm bảo tính đáng tin, cô cần dựa vào bằng chứng từ các hội nghị lớn (NeurIPS, TPAMI, arXiv) và các benchmark minh bạch. Literature review trở thành bước bắt buộc trước khi viết đề xuất nâng cấp hệ thống.

---

## 2. Bản đồ SOTA & literature review 2022–2025

Hai khảo sát trọng yếu trên TPAMI [2][3] gợi ý một “bản đồ” gồm bốn trụ cột: **kiến trúc**, **dữ liệu**, **phương pháp huấn luyện**, **đánh giá minh bạch**. Cô tổng hợp lại dưới dạng timeline, trích dẫn trực tiếp các kết quả được công bố.

| Năm | Công trình | Hội nghị / Tạp chí | Đóng góp chính | Số liệu công bố |
|-----|------------|--------------------|----------------|-----------------|
| 2022 | Flamingo (Alayrac et al.) | NeurIPS 2022 | Perceiver Resampler + Gated Cross-Attention cho few-shot | VQAv2 (4-shot) **82.6%**, OK-VQA (4-shot) **57.8%** [1] |
| 2023 | LLaVA-1.5 (Liu et al.) | arXiv 2310.01547 → CVPR 2024 Workshop | Visual Instruction Tuning mở, 158k đối thoại ảnh-text | ScienceQA test **78.5%**, TextVQA **68.1%** (CoT-free) [6] |
| 2023 | Qwen-VL (Bai et al.) | arXiv 2308.12966 | Pretrain 42B image-text pairs song ngữ, mixture of prefixes | MMBench dev **78.2**, OCRBench **68.3** [7] |
| 2023 | MMMU Benchmark (Yue et al.) | NeurIPS 2023 | Bộ kiểm tra 11 lĩnh vực học thuật | GPT-4V (2023-10) **59.5%**, Human **93.7%** [11] |
| 2024 | GFlowVLM (Geng et al.) | arXiv 2403.08268 | Flow matching + policy gradient cho CoT | ScienceQA +**6.3** điểm & MMMU +**3.9** điểm vs. LLaVA-1.5 [9] |
| 2024 | Gemini 1.0 (Reid et al.) | arXiv 2312.11805 | Multimodal RLHF, toolformer tích hợp | MMMU **62.1**, ChartQA **86.1**, MathVista **66.8** [5] |

**Điểm chắt lọc từ các survey [2][3]:**

1. **Architectural trend:** hầu hết mô hình SOTA đều dùng backbone ngôn ngữ lớn (PaLM, GPT, Qwen) kết hợp với cơ chế cross-attention hoặc adapter để trộn vision tokens. Flamingo mở đường cho các khối gated CA; Qwen-VL sử dụng two-stream attention + prefix tuning.
2. **Data governance:** các survey nhấn mạnh nguồn dữ liệu ảnh-văn bản chất lượng cao (LAION-2B, WebLi, MMC4). Sự minh bạch thể hiện qua việc công bố số lượng, tỷ lệ lọc, giấy phép.
3. **Evaluation gap:** 70% bài báo báo cáo benchmark nhưng chỉ ~35% mở mã nguồn tái hiện [3]. Các tác giả khuyến nghị dùng bộ chấm như MMMU, MathVista, MMBench và cung cấp seed tái hiện.
4. **Reasoning transparency:** Chain-of-thought (CoT) xuất hiện trong GPT-4V, Gemini, GFlowVLM, ReAct-based agent. Survey khuyến khích log thought + cited evidence để đáp ứng yêu cầu giải thích.

Phần còn lại của bài viết bám sát bản đồ này để phân tích chi tiết từng mô hình.

---

## 3. Flamingo: Perceiver Resampler và gated cross-attention

Flamingo là mô hình đầu tiên ở NeurIPS 2022 đặt ra tiêu chuẩn “few-shot multimodal reasoning” [1].

### 3.1 Kiến trúc tổng thể

1. **Vision backbone:** ViT-G/14 hoặc ConvNeXt, sinh ra $N_v$ patch embedding.
2. **Perceiver Resampler:** nén $N_v$ patch thành $M$ latent token cố định:

$$
Z_0 = \text{LatentInit}(M, d), \quad Z_{l+1} = Z_l + \text{CrossAttn}(Z_l, V) + \text{FFN}(Z_l)
$$

3. **Language backbone:** Chồng lên trên là transformer decoder (từ Chinchilla) được chèn các **Gated Cross-Attention (GCA)** block.

### 3.2 Gated cross-attention chi tiết

Với text token $T_{t-1}$ và latent vision $Z_L$:

$$
\begin{aligned}
U_t &= \text{LMBlock}(T_{t-1}) \\
C_t &= \text{CrossAttn}(Q = U_t, K = Z_L, V = Z_L) \\
g_t &= \sigma(W_g [U_t; C_t] + b_g) \\
T_t &= U_t + g_t \odot C_t
\end{aligned}
$$

Gradient theo $g_t$:

$$
\frac{\partial \mathcal{L}}{\partial g_t} = \frac{\partial \mathcal{L}}{\partial T_t} \odot C_t \cdot g_t(1 - g_t)
$$

→ cổng mềm này học được khi nào nên “nhìn” vào ảnh. Các head khác nhau nắm bắt màu sắc, vị trí, thuộc tính – quan sát trong visualization của [1].

### 3.3 Chứng cứ thực nghiệm

- **VQAv2 (4-shot):** Flamingo-80B đạt 82.6% top-1, vượt ViLT fine-tune 81.2% [1].
- **OK-VQA (4-shot):** 57.8%, cao hơn PaLI 54.4%.
- **VizWiz (4-shot):** 33.6% (điểm yếu của mô hình – input ảnh đời thường).

**Bài học cho bảo tàng:** Nếu phải triển khai few-shot (tức không fine-tune), Flamingo cho chất lượng rất cao. Tuy nhiên mô hình đóng, chi phí inference lớn và thiếu logging chi tiết. Do đó cô hướng dẫn viên sử dụng Flamingo như chuẩn đối sánh khi đánh giá các phương án open-source.

---

## 4. GPT-4V và Gemini: Instruction-following đa vòng + RLHF

### 4.1 Kiến trúc chia sẻ token

GPT-4V đưa ảnh thành chuỗi token qua ViT + Linear Adapter, ghép vào prompt text:

$$
X = [\text{VisionPrefix}(I); \text{TextTokens}(S); \text{SpecialMarkers}]
$$

Tất cả token đi qua decoder-only transformer cực lớn (ước tính >1T tham số). Do dùng chung không gian embed, attention có thể đồng thời truy cập cả chữ và ảnh:

$$
\text{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

### 4.2 RLHF đa phương thức

Pipeline huấn luyện (theo GPT-4 Technical Report [4] và Gemini 1.0 [5]):

1. **Pretraining:** trộn MMC4, LAION-2B, sách scan, screenshot UI → đảm bảo coverage biểu đồ, OCR.
2. **Supervised fine-tuning:** thu thập cặp (prompt, chain-of-thought, answer). Cô hướng dẫn viên đặc biệt chú ý ví dụ ChartQA, MathVista, evaluation rubric.
3. **Reward modeling:** xây $R_\phi(I,S,A)$ đánh giá độ chính xác, lịch sự, tuân thủ. 
4. **PPO:** cập nhật chính sách $\pi_\theta$:

$$
\max_\theta \mathbb{E}_{A \sim \pi_\theta} [R_\phi(I, S, A) - \beta \,\mathrm{KL}(\pi_\theta \| \pi_{\text{SFT}})]
$$

Gemini bổ sung **toolformer module** – các token đặc biệt kích hoạt API như `code_interpreter`, `search`.

### 4.3 Evidence từ benchmark

- Theo MMMU [11], GPT-4V (2023-10) đạt 59.5% Overall, vượt GPT-4 (text) 53.6%.
- Gemini Ultra [5] báo cáo MMMU 62.1%, MathVista 66.8%, ChartQA 86.1%.
- Human upper bound MMMU 93.7% → vẫn còn khoảng cách lớn, minh chứng cho nhu cầu giữ người giám sát (human-in-the-loop).

**Góc nhìn vận hành:** GPT-4V/Gemini có chất lượng cao nhất nhưng là dịch vụ thương mại. Chiến lược khả thi cho bảo tàng là dựng agent hybrid: dùng mô hình open-source nội bộ, fallback sang GPT-4V khi gặp câu hỏi khó và log chi tiết để audit.

---

## 5. Open-source challengers: LLaVA-1.5, Qwen-VL, Kosmos-2

### 5.1 LLaVA-1.5 (arXiv 2310.01547) [6]

- **Pipeline:** CLIP ViT-L/14 → linear projector → LLaMA-2 decoder. Visual instruction tuning với 158k đối thoại (cộng đồng tự đóng góp).
- **Điểm mạnh:** mã nguồn mở đầy đủ, script huấn luyện, benchmark reproducible với seed.
- **Kết quả:** ScienceQA test 78.5% (không cần CoT), TextVQA 68.1%, DocVQA 61.0%. Với CoT prompting, ScienceQA tăng lên 83.0%.
- **Phát hiện từ literature review:** so với Flamingo, LLaVA cần fine-tune nhưng inference nhẹ (~13B tham số). Cô hướng dẫn viên có thể fine-tune thêm với dữ liệu bảo tàng.

### 5.2 Qwen-VL & Qwen-VL-Max (2023) [7]

- **Pretraining:** 42B mã thông báo (image-text), song ngữ Anh–Trung, phích tắc multi-resolution.
- **Kiến trúc:** Vision encoder EVA-CLIP-G, text decoder Qwen-7B, two-stream attention + prefix network.
- **Metrics:** Tác giả báo cáo MMBench dev 78.2, OCRBench 68.3, DocVQA 91.3. Qwen-VL-Max (32B) đạt 82.1 MMBench, ngang tầm GPT-4V early release.
- **Ưu điểm cho bảo tàng:** đã hỗ trợ tiếng Trung và Anh; pipeline training công bố dataset ratio, script inference.

### 5.3 Kosmos-2 (Microsoft, 2023) [8]

- **Mục tiêu:** grounding + reasoning. Vision encoder Swin-L, text backbone DeBERTa-v3.
- **Chiến lược:** pretrain trên 1.6T token multimodal, alignment qua dual-stage objective.
- **Kết quả:** NLVR2 92.3, DocVQA 84.2, Visual Grounding (RefCOCOg) 67.2. 
- **Insight:** Kosmos-2 mạnh ở grounding explicit (tọa độ, bounding box), hữu ích khi phải chỉ rõ vùng tác phẩm trong ảnh.

---

## 6. GFlowVLM: Flow matching cho chuỗi suy luận đa phương thức

GFlowVLM [9] là nỗ lực kết hợp **flow matching** với CoT.

### 6.1 Flow matching cho text generation

Sinh chuỗi $Y$ được xem là dòng liên tục từ phân phối đơn giản $p_0$ đến $p_1$ (ground truth):

$$
\frac{d x_t}{d t} = f_\theta(x_t, t), \quad x_0 \sim p_0
$$

Loss flow matching:

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, x_t} \left[ \| f_\theta(x_t, t) - u_t \|_2^2 \right]
$$

với $u_t$ là velocity suy ra từ dữ liệu quan sát.

### 6.2 Policy improvement đa mục tiêu

Sau khi học flow, tác giả áp dụng policy gradient với reward:

$$
R = \lambda_{\text{acc}} R_{\text{accuracy}} + \lambda_{\text{faith}} R_{\text{faithfulness}} + \lambda_{\text{style}} R_{\text{style}}
$$

Phần “faithfulness” sử dụng CLIPScore và IoU với vùng ảnh được trích dẫn trong thought chain để phạt hallucination. 

### 6.3 Số liệu nổi bật

- ScienceQA: +6.3 điểm so với LLaVA-1.5 (từ 83.0 → 89.3 khi dùng CoT).
- MMMU: +3.9 điểm so với LLaVA-Next 13B.
- Trace logging: paper mở mã inference ghi lại thought + vùng ảnh (mask attention) → đáp ứng yêu cầu minh bạch của bảo tàng.

---

## 7. Bộ nhớ, context dài và tool-use orchestration

### 7.1 Memory token + retrieval

- **Memory token:** LongVLM, Flamingo++ dùng trạng thái $m_t = \text{GRU}(m_{t-1}, h_t)$ để duy trì thông tin đa vòng. 
- **Retrieval augmentation:** lưu embedding hội thoại vào vector store (FAISS). Khi khách hỏi lại, truy xuất top-$k$ bằng cosine sim, ghép vào prompt. Survey [3] nhấn mạnh retrieval giúp giảm drift contextual.

### 7.2 Tool-use như một Markov Decision Process

Cô mô hình hóa quá trình ReAct thành MDP $(\mathcal{S}, \mathcal{A}, P, R)$:

- **State $s_t$:** concat của prompt, history, observation từ công cụ.
- **Action $a_t$:** phát lệnh `THOUGHT`, `ACTION[tool(args)]`, `ANSWER`.
- **Transition $P$:** deterministic đối với Thought, stochastic khi action gọi công cụ (kết quả phụ thuộc dữ liệu).
- **Reward $R$:** +1 nếu câu trả lời đúng, +0.3 nếu thought trích dẫn nguồn hợp lệ, -0.5 nếu tool được gọi sai.

Điều này cho phép áp dụng kỹ thuật policy gradient hoặc offline RL trên log tương tác tại bảo tàng.

### 7.3 Policy graph vận hành

| Node | Điều kiện chuyển | Hành động |
|------|------------------|-----------|
| `Start` | Khách hỏi câu mới | Khởi tạo slide tóm tắt phòng hiện tại |
| `Perception` | Câu hỏi chứa từ khóa “đếm”, “so sánh”, “màu” | Sinh thought phân tách nhiệm vụ + đánh số vùng ảnh |
| `ToolCall` | Thought đề xuất công cụ (`count_objects`, `ocr_plate`) | Gọi API, log kết quả |
| `Reason` | Nhận Observation | Cập nhật memory token, tạo thought tiếp theo |
| `Answer` | Thought kết thúc hoặc đạt confidence > 0.75 | Sinh câu trả lời + citation |

Graph này giúp kiểm soát latency và tránh spam API.

---

## 8. Recipe huấn luyện reasoning SOTA cho bảo tàng

### 8.1 Chuẩn bị dữ liệu có chú thích chuỗi suy luận

- **Nguồn mở:** kết hợp VQA-hard (TextVQA, VizWiz), ChartQA [12], MathVista, MMMU (subset), ScienceQA [13].
- **Nội bộ:** cô ghi âm 1.2k đoạn hội thoại với khách, gắn nhãn thought chain (3–6 bước), trích dẫn ID tác phẩm.
- **Hard negative:** tráo thought step, đổi thứ tự observation để mô hình học phát hiện lỗi logic.

### 8.2 Lộ trình huấn luyện nhiều giai đoạn

1. **Stage 0 – Warm-up alignment:** khởi động từ checkpoint fusion (bài trước) với LR $1\mathrm{e}{-6}$, batch 256 (accumulated). 
2. **Stage 1 – Instruction tuning (SFT):**

   $$
   \mathcal{L}_{\text{SFT}} = -\sum_{t} \log p_\theta(y_t | y_{<t}, I, S)
   $$

   Bao gồm thought + answer tokens.

3. **Stage 2 – Tool-feedback RL:** áp dụng GRPO hoặc PPO với reward $R_{\text{tool}}$ cho các action đúng.
4. **Stage 3 – Self-consistency distillation:** sinh $K=5$ thought chains, lấy majority vote, distill về mô hình nhỏ hơn bằng KL.
5. **Stage 4 – Safety fine-tuning:** sử dụng reward model phát hiện nội dung nhạy cảm (từ dataset Anthropic HH [14]) để chặn câu hỏi không liên quan.

### 8.3 Các trick thực dụng

- **Vision dropout 0.2:** tránh overfit một góc chụp.
- **Thought length penalty $\alpha=0.6$:** khuyến khích thought gọn, tránh lan man.
- **Mixed precision (bfloat16) + gradient checkpointing:** giảm bộ nhớ GPU, quan trọng khi training 13B+.

---

## 9. Ví dụ PyTorch: Multimodal ReAct Agent có logging minh bạch

Đoạn code dưới đây mở rộng skeleton ở bài trước: bổ sung tracking thought chain, observation, citation và hook đánh giá.

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AgentLog:
    thoughts: list
    actions: list
    observations: list
    citations: list

class MultimodalReasoner(torch.nn.Module):
    def __init__(self, vision_encoder, text_decoder, tool_registry):
        super().__init__()
        self.vision_encoder = vision_encoder      # ViT + perceiver resampler
        self.text_decoder = text_decoder          # Decoder-only LM với GCA blocks
        self.tool_registry = tool_registry        # Dict[str, Callable]
        self.memory_state = None

    def encode_images(self, images):
        vision_tokens = self.vision_encoder(images)
        return F.normalize(vision_tokens, dim=-1)

    def forward(self, images, prompt_tokens):
        vision_tokens = self.encode_images(images)
        logits = self.text_decoder(
            prompt_tokens,
            vision_tokens,
            memory=self.memory_state
        )
        return logits

    @torch.no_grad()
    def react(self, images, prompt_tokens, max_actions=6):
        log = AgentLog(thoughts=[], actions=[], observations=[], citations=[])
        tokens = prompt_tokens

        for _ in range(max_actions):
            logits = self.forward(images, tokens)
            next_token = torch.argmax(logits[:, -1], dim=-1)
            token_id = next_token.item()
            decoded = self.text_decoder.tokenizer.decode([token_id])

            if decoded.startswith("[THOUGHT]"):
                log.thoughts.append(decoded)
            elif decoded.startswith("[ACTION]"):
                action_name, kwargs = self._parse_action(decoded)
                log.actions.append(decoded)
                observation = self._call_tool(action_name, kwargs)
                log.observations.append(observation)
                tokens = torch.cat([tokens, self._encode_text(observation)], dim=1)
                continue
            elif decoded.startswith("[CITE]"):
                log.citations.append(decoded)
            elif decoded.startswith("[ANSWER]"):
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                break

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        return tokens, log

    def _call_tool(self, name, args):
        fn = self.tool_registry.get(name)
        if fn is None:
            raise ValueError(f"Tool {name} không tồn tại")
        result = fn(**args)
        return f"[OBSERVATION] {result}"

    def load_memory(self, memory_tokens):
        self.memory_state = memory_tokens

    def _encode_text(self, text):
        ids = self.text_decoder.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=next(self.parameters()).device)
```

**Điểm nhấn:**

- `AgentLog` lưu thought/action/observation phục vụ audit.
- Decoder học các token đặc biệt `[THOUGHT]`, `[ACTION]`, `[CITE]`, `[ANSWER]`.
- Có thể tích hợp thêm evaluation hook:

```python
def evaluate_agent(agent, dataloader, metrics):
    scores = {name: [] for name in metrics.keys()}
    for batch in dataloader:
        images, prompt_tokens, labels = batch
        _, log = agent.react(images, prompt_tokens)
        for name, fn in metrics.items():
            scores[name].append(fn(log, labels))
    return {name: sum(vals) / len(vals) for name, vals in scores.items()}
```

---

## 10. Đánh giá, minh bạch và phân tích lỗi

### 10.1 Bảng so sánh benchmark chính

| Mô hình | VQAv2 | ScienceQA | MMMU | ChartQA | Ghi chú |
|---------|-------|-----------|------|---------|---------|
| Flamingo-80B [1] | 82.6 (4-shot) | 75.2 (4-shot) | – | – | Mạnh few-shot, không mở |
| LLaVA-1.5 13B [6] | 78.5 (fine-tune) | 83.0 (CoT) | 36.1 | 67.7 | Dễ tái hiện, cần fine-tune |
| Qwen-VL-Max [7] | 80.3 | 84.4 | 41.8 | 74.5 | Hỗ trợ đa ngôn ngữ |
| GFlowVLM 13B [9] | 81.2 | 89.3 | 40.0 | 71.6 | Thought chain minh bạch, flow matching |
| GPT-4V (2023-10) [11] | 85.0 (community eval) | 88.2 | 59.5 | 82.3 | Dịch vụ thương mại |
| Gemini Ultra [5] | 86.5 | 90.0 | 62.1 | 86.1 | Tích hợp toolformer |

> **Chú ý:** các số liệu lấy từ bảng trong từng bài báo được trích ở mục Tài liệu tham khảo. MMMU theo chuẩn Official Evaluation (seed fix). ChartQA dùng split test, metric exact match.

### 10.2 Quy trình phân tích lỗi ở bảo tàng

1. **Replay thought chain:** đối chiếu từng thought với observation. Nếu chain không trích dẫn ID tác phẩm → phạt $R_{\text{faith}}$.
2. **Tool audit:** log JSON gồm `{action, args, latency, result}`. Kiểm tra tool nào vượt 500ms để tối ưu hoặc cache.
3. **Human-in-the-loop:** cô dành 30 phút cuối ngày đọc top-10 câu trả lời có confidence <0.6, viết note cho batch fine-tune tiếp theo.
4. **Safety review:** dùng reward model (bài [14]) để flag nội dung nhạy cảm; nếu vi phạm, thought chain dừng và chuyển cho nhân viên thật.

### 10.3 Tiêu chí minh bạch

- **Citation bắt buộc:** câu trả lời phải chứa `[CITE:id_tac_pham]`.
- **Latency budget:** <1.5 giây/vòng, >3 vòng thì fallback GPT-4V.
- **Reproducibility:** công bố seed, version checkpoint, log tool-call → đáp ứng khuyến nghị trong survey [3].

---

## 11. Liên kết với các bài tiếp theo

Sau khi hoàn thiện reasoning và minh bạch thought chain, bước tiếp theo của cô hướng dẫn viên là **tối ưu chi phí triển khai**: quantization, batching, offloading bộ nhớ để chạy trên cụm GPU nội bộ. Bài kế tiếp sẽ đào sâu system design, kết nối với phần đánh giá (benchmark) trong bài này.

---

## 12. Tài liệu tham khảo

1. Alayrac, J.-B. et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS.
2. Huang, W. et al. (2024). *Large Multimodal Models: A Survey.* IEEE TPAMI.
3. Xu, R. et al. (2024). *Multimodal Foundation Models: A Survey.* IEEE TPAMI.
4. OpenAI (2023). *GPT-4 Technical Report.*
5. Reid, M. et al. (2024). *Gemini 1.0: Multimodal Foundation Models Built From the Ground Up.*
6. Liu, H. et al. (2023). *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5).*
7. Bai, J. et al. (2023). *Qwen-VL: A Frontier Large Vision-Language Model with Comprehensive Abilities.*
8. Zhai, X. et al. (2023). *Kosmos-2: Grounding Multimodal Large Language Models to the World.*
9. Geng, Z. et al. (2024). *GFlowVLM: Multimodal Reasoning via Generative Flow Networks.*
10. Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.*
11. Yue, X. et al. (2023). *MMMU: A Massive Multi-discipline Multimodal Understanding Benchmark.*
12. Masry, A. et al. (2022). *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning.*
13. Lu, P. et al. (2022). *Learn to Explain: Multimodal Reasoning via Thought Chains for Science QA.*
14. Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with RLHF.*

---

<script src="/assets/js/katex-init.js"></script>
