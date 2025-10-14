# CLIP Training: Contrastive Language-Image Pretraining

**Ngày đăng:** 01/11/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** VLM, Contrastive Learning

## Giới thiệu

CLIP learns vision-language representations through **contrastive pretraining** on 400M image-text pairs.

## Architecture

```
Image → Vision Transformer → Image Embedding (d)
Text  → Text Transformer  → Text Embedding (d)
```

## Contrastive Loss

```python
def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity matrix
    logits = (image_embeddings @ text_embeddings.T) / temperature
    
    # Symmetric cross-entropy loss
    labels = torch.arange(len(logits))
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2
```

## Training Details

- **Batch size:** 32,768
- **Datasets:** 400M image-text pairs from internet
- **Augmentation:** Random crop, color jitter
- **Architecture:** ViT-L/14 for vision, Transformer for text
- **Training time:** ~2 weeks on 256 V100s

## Zero-Shot Transfer

```python
def zero_shot_classify(model, image, class_names):
    # Encode image
    image_features = model.encode_image(image)
    
    # Encode class names as text prompts
    prompts = [f"a photo of a {c}" for c in class_names]
    text_features = model.encode_text(prompts)
    
    # Compute similarities
    similarities = (image_features @ text_features.T)
    
    return similarities.argmax()
```

## Tài liệu

- Radford et al. "Learning Transferable Visual Models from Natural Language Supervision" ICML 2021

**Tags:** #CLIP #ContrastiveLearning #ZeroShot

<script src="/assets/js/katex-init.js"></script>
