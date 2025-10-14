# LLaVA: Large Language and Vision Assistant

**Ngày đăng:** 02/11/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** VLM, Instruction Following

## Giới thiệu

LLaVA connects **CLIP vision encoder** với **Vicuna LLM** through simple projection layer, achieving impressive visual reasoning.

## Architecture

```
Image → CLIP ViT-L/14 → Linear Projection → Vicuna 13B
                              ↓
                        Vision Tokens
```

## Two-Stage Training

### Stage 1: Feature Alignment (Pretraining)
- Freeze vision encoder & LLM
- Train projection layer only
- Task: Image captioning
- Data: 595K image-caption pairs

```python
# Stage 1: Alignment
for batch in caption_data:
    image, caption = batch
    vision_features = clip_encoder(image)  # Frozen
    vision_tokens = projector(vision_features)  # Trainable
    
    # Predict caption
    loss = llm(vision_tokens, caption)  # LLM frozen
    loss.backward()
```

### Stage 2: Visual Instruction Tuning
- Freeze vision encoder
- Train projection + LLM
- Task: Multi-turn conversation, VQA, reasoning
- Data: 158K instruction-following samples

```python
# Stage 2: Instruction tuning
for batch in instruction_data:
    image, instruction, response = batch
    vision_features = clip_encoder(image)  # Frozen
    vision_tokens = projector(vision_features)  # Trainable
    
    # Combine vision + text
    inputs = concat([vision_tokens, tokenize(instruction)])
    loss = llm(inputs, response)  # LLM trainable
    loss.backward()
```

## Instruction Data Generation

Use GPT-4 to generate diverse instructions from COCO captions:

```python
prompt = f"""
Given image caption: "{caption}"
Generate 3 types of instructions:
1. Conversation: Natural multi-turn dialogue
2. Detailed description: Request comprehensive description
3. Complex reasoning: Questions requiring inference

Format as JSON with instruction and response.
"""

instructions = gpt4(prompt, caption)
```

## Implementation

```python
class LLaVA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = CLIPVisionModel()  # ViT-L/14
        self.projector = nn.Linear(1024, 5120)  # CLIP → Vicuna
        self.llm = VicunaModel()  # 13B parameters
    
    def forward(self, images, instructions):
        # Encode image
        vision_features = self.vision_encoder(images)[:, 0]  # CLS token
        
        # Project to LLM space
        vision_tokens = self.projector(vision_features).unsqueeze(1)
        
        # Tokenize instruction
        text_tokens = self.llm.tokenizer(instructions)
        
        # Concatenate vision + text
        inputs = torch.cat([vision_tokens, text_tokens], dim=1)
        
        # Generate response
        outputs = self.llm(inputs)
        return outputs
```

## Key Results

- **Strong VQA performance** (80%+ on VQAv2)
- **Multi-turn conversation** capability
- **Complex reasoning** about images
- **Efficient training** (15 hours on 8xA100)

## Extensions

**LLaVA-1.5:**
- Higher resolution (336x336)
- Better projector (MLP)
- More instruction data

**LLaVA-NeXT:**
- Any aspect ratio
- Dynamic resolution
- Video understanding

## Tài liệu

- Liu et al. "Visual Instruction Tuning" (LLaVA) NeurIPS 2023
- Liu et al. "Improved Baselines with Visual Instruction Tuning" (LLaVA-1.5)

**Tags:** #LLaVA #VLM #InstructionTuning #VisualReasoning
