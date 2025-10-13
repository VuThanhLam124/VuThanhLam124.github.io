# 📚 Hướng Dẫn Thêm Nội Dung và Topics

## 📋 Mục lục
1. [Cấu trúc Website](#cấu-trúc-website)
2. [Thêm Bài Viết Mới](#thêm-bài-viết-mới)
3. [Thêm Topic Mới vào Content Page](#thêm-topic-mới-vào-content-page)
4. [Cập nhật API Posts](#cập-nhật-api-posts)
5. [Deploy lên GitHub Pages](#deploy-lên-github-pages)

---

## 🏗️ Cấu trúc Website

```
VuThanhLam124.github.io/
├── index.html                  # Trang chủ
├── content.html               # Trang nội dung (danh sách topics)
├── api/
│   └── posts.json            # Database của tất cả bài viết
├── posts/
│   └── 2025/
│       ├── flow-matching-theory.md
│       ├── flow-matching-readme.md
│       └── ...               # Các bài viết markdown
├── assets/
│   ├── css/
│   │   └── main.css          # CSS chính
│   ├── js/
│   │   └── main.js           # JavaScript chính
│   └── image/
│       └── My.png            # Ảnh avatar
└── code/
    └── flow-matching/
        └── flow_matching_pytorch.py
```

---

## ✍️ Thêm Bài Viết Mới

### Bước 1: Tạo File Markdown

Tạo file mới trong `/posts/2025/` với tên dạng `slug-name.md`

**Ví dụ:** `/posts/2025/diffusion-transformers.md`

```markdown
# Diffusion Transformers (DiT): Next-Gen Architecture

**Ngày đăng:** 13/10/2025  
**Tác giả:** ThanhLamDev  
**Thể loại:** Deep Learning, Diffusion Models

## Giới thiệu

Diffusion Transformers (DiT) đại diện cho sự kết hợp giữa Transformer architecture và Diffusion Models...

## 1. Architecture Overview

### 1.1 Core Components
...

## 2. Implementation

```python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        # Implementation
```

## Kết luận

...

## Tài liệu tham khảo
1. Paper Title - Authors (Year)
2. ...
```

### Bước 2: Cập nhật API Posts

Mở file `/api/posts.json` và thêm metadata của bài viết:

```json
{
  "posts": [
    {
      "id": 7,
      "title": "Diffusion Transformers (DiT): Next-Gen Architecture",
      "slug": "diffusion-transformers",
      "excerpt": "Khám phá architecture mới kết hợp Transformer và Diffusion Models cho high-quality image generation.",
      "content": "/posts/2025/diffusion-transformers.md",
      "author": "ThanhLamDev",
      "date": "2025-10-13",
      "readingTime": 18,
      "category": "Diffusion Models",
      "tags": ["Diffusion", "Transformers", "DiT", "Architecture", "Deep Learning"],
      "featured": false
    }
  ]
}
```

**Giải thích các fields:**
- `id`: Số thứ tự bài viết (tăng dần)
- `title`: Tiêu đề hiển thị
- `slug`: URL-friendly name (dùng trong URL)
- `excerpt`: Mô tả ngắn (1-2 câu)
- `content`: Đường dẫn đến file markdown
- `author`: Tên tác giả
- `date`: Ngày đăng (format: YYYY-MM-DD)
- `readingTime`: Thời gian đọc ước tính (đã bỏ hiển thị nhưng giữ lại cho tương lai)
- `category`: Chủ đề chính
- `tags`: Mảng các tags liên quan
- `featured`: true/false - hiển thị trong featured section

---

## 🎯 Thêm Topic Mới vào Content Page

### Bước 1: Mở file `content.html`

Tìm đến phần `<div class="content-section">` (khoảng dòng 350)

### Bước 2: Thêm Section Mới

Copy template sau và điền thông tin:

```html
<!-- Your New Topic Section -->
<section class="topic-section" id="your-topic-id">
    <h2 class="topic-title">
        🎨 Your Topic Name
        <span class="topic-badge">Badge Text</span>
    </h2>
    <p class="topic-description">
        Mô tả chi tiết về topic này. Giải thích tại sao topic này quan trọng,
        những gì người đọc sẽ học được, và applications thực tế.
        Nên dài khoảng 3-5 câu để đủ context.
    </p>
    <ol class="posts-list">
        <li>
            <a href="/posts/2025/post-slug-1" class="post-link">
                Post Title 1
                <span class="post-meta">Post Type/Category</span>
            </a>
        </li>
        <li>
            <a href="/posts/2025/post-slug-2" class="post-link">
                Post Title 2
                <span class="post-meta">Post Type/Category</span>
            </a>
        </li>
        <!-- Thêm các bài viết khác -->
    </ol>
</section>
```

**Ví dụ thực tế:**

```html
<!-- Generative Adversarial Networks Section -->
<section class="topic-section" id="gans">
    <h2 class="topic-title">
        🎭 Generative Adversarial Networks (GANs)
        <span class="topic-badge">Classic</span>
    </h2>
    <p class="topic-description">
        GANs revolutionized generative modeling với adversarial training paradigm.
        Từ vanilla GAN cho đến StyleGAN3, từ conditional generation đến image-to-image 
        translation - GANs đã mở ra countless applications trong computer vision và 
        creative AI. Series này explores architectures, training techniques và 
        modern alternatives.
    </p>
    <ol class="posts-list">
        <li>
            <a href="/posts/2025/gan-fundamentals" class="post-link">
                GAN Fundamentals: Generator vs Discriminator
                <span class="post-meta">Theory + Math</span>
            </a>
        </li>
        <li>
            <a href="/posts/2025/stylegan-architecture" class="post-link">
                StyleGAN: Architecture và Style Control
                <span class="post-meta">Advanced Architecture</span>
            </a>
        </li>
        <li>
            <a href="/posts/2025/gan-training-tricks" class="post-link">
                Training GANs: Stability và Best Practices
                <span class="post-meta">Practical Guide</span>
            </a>
        </li>
    </ol>
</section>
```

### Bước 3: Thêm vào Secondary Navigation

Nếu muốn topic xuất hiện trong secondary nav bar, cập nhật:

```html
<div class="secondary-nav">
    <div class="secondary-nav-container">
        <a href="#flow-based">🌊 Flow-based Models</a>
        <a href="#diffusion">✨ Diffusion Models</a>
        <a href="#vlm">👁️ Vision-Language</a>
        <a href="#gans">🎭 GANs</a>  <!-- Topic mới -->
        <a href="#advanced">🚀 Advanced Topics</a>
    </div>
</div>
```

### Bước 4: Cập nhật Stats (nếu cần)

Trong hero section, cập nhật số liệu:

```html
<div class="hero-stats">
    <div class="stat-item">
        <span class="stat-number" id="total-posts">30+</span>  <!-- Tăng số -->
        <div class="stat-label">Bài viết chuyên sâu</div>
    </div>
    <div class="stat-item">
        <span class="stat-number">5</span>  <!-- Tăng từ 4 lên 5 -->
        <div class="stat-label">Chủ đề chính</div>
    </div>
    <div class="stat-item">
        <span class="stat-number">40+</span>  <!-- Tăng số -->
        <div class="stat-label">Code examples</div>
    </div>
</div>
```

---

## 📝 Các Badge Types

Có thể dùng các badge sau cho topics:

- `<span class="topic-badge">Cutting-edge</span>` - Công nghệ mới nhất
- `<span class="topic-badge">Foundation</span>` - Kiến thức nền tảng
- `<span class="topic-badge">Multimodal</span>` - Đa phương thức
- `<span class="topic-badge">Research</span>` - Nghiên cứu
- `<span class="topic-badge">Classic</span>` - Kinh điển
- `<span class="topic-badge">Production</span>` - Production-ready
- `<span class="topic-badge">Experimental</span>` - Thử nghiệm

---

## 🎨 Emoji Icons cho Topics

Một số emoji phù hợp:

- 🌊 - Flow/Wave (cho Flow-based Models)
- ✨ - Sparkle (cho Diffusion)
- 👁️ - Eye (cho Vision)
- 🤖 - Robot (cho AI/ML general)
- 🧠 - Brain (cho Neural Networks)
- 🎯 - Target (cho Optimization)
- 🔬 - Microscope (cho Research)
- 💡 - Bulb (cho Insights)
- 🎭 - Theater (cho GANs)
- 🚀 - Rocket (cho Advanced/Fast methods)
- 📊 - Chart (cho Analysis)
- 🎨 - Palette (cho Generative/Creative)

---

## 🔄 Workflow Hoàn Chỉnh

### 1. Tạo nội dung mới

```bash
# 1. Tạo file markdown
cd posts/2025/
nano your-new-post.md

# 2. Viết nội dung với markdown format
```

### 2. Cập nhật metadata

```bash
# Mở và edit posts.json
nano api/posts.json

# Thêm object mới vào mảng "posts"
```

### 3. Cập nhật content.html (nếu thêm topic mới)

```bash
nano content.html

# Thêm section mới hoặc thêm link vào section có sẵn
```

### 4. Test local (optional)

```bash
# Nếu có Python
python -m http.server 8000

# Truy cập: http://localhost:8000
```

### 5. Commit và Deploy

```bash
# Add tất cả changes
git add .

# Commit với message rõ ràng
git commit -m "Add new post: Your Post Title"

# Push lên GitHub
git push origin main

# Đợi 2-3 phút để GitHub Pages deploy
```

---

## 📊 Cập nhật Stats trong Hero Section

### File: `content.html`

Tìm section `hero-stats` (khoảng dòng 338):

```html
<div class="hero-stats">
    <div class="stat-item">
        <span class="stat-number" id="total-posts">25+</span>
        <div class="stat-label">Bài viết chuyên sâu</div>
    </div>
    <div class="stat-item">
        <span class="stat-number">4</span>
        <div class="stat-label">Chủ đề chính</div>
    </div>
    <div class="stat-item">
        <span class="stat-number">30+</span>
        <div class="stat-label">Code examples</div>
    </div>
</div>
```

**Cập nhật khi:**
- Thêm 5-10 bài mới → tăng "Bài viết chuyên sâu"
- Thêm topic mới → tăng "Chủ đề chính"
- Thêm code examples → tăng "Code examples"

---

## 🎯 Best Practices

### 1. **Naming Convention**

- **File names:** lowercase, dấu gạch ngang
  - ✅ `diffusion-transformers.md`
  - ❌ `Diffusion_Transformers.md`

- **Slugs:** giống file name (không cần .md)
  - ✅ `"slug": "diffusion-transformers"`

### 2. **Markdown Format**

```markdown
# Tiêu đề chính (H1) - chỉ dùng 1 lần ở đầu

## Section chính (H2)

### Subsection (H3)

#### Sub-subsection (H4)

**Bold text**
*Italic text*
`inline code`

```python
# Code block với syntax highlighting
def example():
    pass
```

- Bullet point
1. Numbered list

[Link text](https://example.com)

![Image alt text](image-url)
```

### 3. **Content Structure**

Mỗi bài viết nên có:

1. **Introduction** - Giới thiệu vấn đề
2. **Background** - Context và motivation
3. **Main Content** - Nội dung chính với sections
4. **Code Examples** - Code minh họa (nếu có)
5. **Conclusion** - Tổng kết
6. **References** - Tài liệu tham khảo

### 4. **Tags Guidelines**

- Dùng 3-7 tags per post
- Tag chính (category) đặt đầu tiên
- Bao gồm: technique, framework, application
- Ví dụ: `["Diffusion", "PyTorch", "Image Generation", "DDPM", "Theory"]`

### 5. **Excerpt Writing**

- Dài 1-2 câu (khoảng 20-30 từ)
- Highlight điểm chính của bài
- Tránh dùng "Trong bài này..." - đi thẳng vào vấn đề
- ✅ "Khám phá DiT architecture kết hợp Transformer và Diffusion cho high-quality generation."
- ❌ "Trong bài này chúng ta sẽ tìm hiểu về DiT."

---

## 🚀 Quick Reference Commands

```bash
# Tạo bài viết mới
cd posts/2025/
touch new-post.md
nano new-post.md

# Check git status
git status

# Add và commit
git add .
git commit -m "Add: [Topic] - Post title"

# Push và deploy
git push origin main

# Xem log
git log --oneline -5

# Check remote
git remote -v

# Force refresh sau deploy
# Ctrl + Shift + R (Windows/Linux)
# Cmd + Shift + R (Mac)
```

---

## 🔍 Troubleshooting

### Issue: Bài viết không hiển thị

**Giải pháp:**
1. Check file path trong `posts.json` đúng chưa
2. Check file markdown có tồn tại không
3. Check JSON syntax (dùng jsonlint.com)
4. Clear browser cache (Ctrl + Shift + R)

### Issue: Secondary nav không scroll đúng

**Giải pháp:**
1. Check ID trong section tag: `<section id="topic-id">`
2. Check link trong nav: `<a href="#topic-id">`
3. ID phải match chính xác

### Issue: GitHub Pages không update

**Giải pháp:**
1. Check GitHub Actions: `github.com/[user]/[repo]/actions`
2. Đợi 2-5 phút
3. Hard refresh browser
4. Trigger deploy: `git commit --allow-empty -m "Trigger deploy" && git push`

---

## 📚 Template Files

### Template: Bài viết mới

```markdown
# [Title]: [Subtitle]

**Ngày đăng:** DD/MM/YYYY  
**Tác giả:** ThanhLamDev  
**Thể loại:** [Category]

## 📋 Mục lục
1. [Section 1](#section-1)
2. [Section 2](#section-2)
3. [Conclusion](#conclusion)

---

## Giới thiệu

Brief introduction about the topic...

## 1. [Section 1]

Content...

### 1.1 [Subsection]

Details...

```python
# Code example
```

## 2. [Section 2]

Content...

## Kết luận

Summary and key takeaways...

## Tài liệu tham khảo

1. Paper/Book Title - Authors (Year) - [Link]
2. ...

---

**Tags:** #tag1 #tag2 #tag3
```

### Template: posts.json entry

```json
{
  "id": 0,
  "title": "Title of the Post",
  "slug": "url-friendly-slug",
  "excerpt": "Brief description in 1-2 sentences.",
  "content": "/posts/2025/file-name.md",
  "author": "ThanhLamDev",
  "date": "YYYY-MM-DD",
  "readingTime": 15,
  "category": "Main Category",
  "tags": ["Tag1", "Tag2", "Tag3", "Tag4"],
  "featured": false
}
```

---

## 💡 Tips & Tricks

1. **Markdown Preview**: Dùng VS Code với extension "Markdown Preview Enhanced"
2. **JSON Validation**: Validate `posts.json` tại jsonlint.com trước khi commit
3. **Image Optimization**: Nén ảnh trước khi upload (dùng tinypng.com)
4. **Code Formatting**: Dùng syntax highlighting với language tags
5. **Git Messages**: Viết commit message rõ ràng: "Add: [Topic] Post Title"

---

## 📞 Support

Nếu gặp vấn đề:
1. Check file này trước
2. Check console log trong browser (F12)
3. Check GitHub Actions logs
4. Review git commit history

---

**Last Updated:** October 13, 2025  
**Version:** 1.0  
**Author:** ThanhLamDev
