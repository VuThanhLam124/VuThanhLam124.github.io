# Visual Instruction Tuning for VLMs

**Ngày đăng:** 30/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** VLM, Instruction Following

## Giới thiệu

Visual instruction tuning teaches VLMs to **follow instructions** involving images và text.

## LLaVA Approach

### Architecture
```
Image → Vision Encoder → Projection → LLM
Text Instructions → Tokenizer → LLM
```

### Training Stages

**Stage 1: Pretraining**
- Align vision-language representations
- Image captioning task

**Stage 2: Instruction Tuning**
- Multi-task visual Q&A
- Diverse instruction formats

## Dataset Construction

```python
instruction_formats = [
    "What is shown in this image?",
    "Describe the image in detail.",
    "Answer the question: {question}",
    "Identify objects in the image."
]

def create_instruction_sample(image, caption, qa_pairs):
    instruction = random.choice(instruction_formats)
    if "{question}" in instruction:
        q, a = random.choice(qa_pairs)
        instruction = instruction.format(question=q)
        response = a
    else:
        response = caption
    return {"image": image, "instruction": instruction, "response": response}
```

## Best Practices

1. **Diverse instructions** - vary question types
2. **Detailed responses** - encourage comprehensive answers
3. **Multi-turn dialogues** - contextual understanding
4. **Negative examples** - handle incorrect assumptions

## Tài liệu

- Liu et al. "Visual Instruction Tuning" (LLaVA) NeurIPS 2023

**Tags:** #InstructionTuning #VLM #LLaVA
