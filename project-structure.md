# Project Structure Overview

Đây là cấu trúc thư mục hoàn chình cho blog scalable AI Research Hub của bạn:

```
vuthanhlam124.github.io/
├── index.html                          # Main homepage (improved version)
├── assets/
│   ├── css/
│   │   ├── main.css                   # Core CSS với theme system
│   │   ├── syntax-highlighting.css    # Code highlighting styles  
│   │   └── print.css                  # Print-specific styles
│   ├── js/
│   │   ├── main.js                    # Core JavaScript functionality
│   │   ├── markdown-parser.js         # Markdown to HTML parser
│   │   ├── search.js                  # Search functionality
│   │   └── analytics.js               # Optional: Google Analytics
│   ├── images/
│   │   ├── logo.png                   # Blog logo
│   │   ├── avatar.jpg                 # Author avatar
│   │   └── posts/                     # Post featured images
│   └── fonts/                         # Custom fonts if needed
├── api/
│   ├── posts.json                     # Posts metadata và index
│   ├── categories.json                # Categories information
│   └── sitemap.json                   # Site structure for SEO
├── posts/                             # Markdown blog posts
│   ├── 2025/
│   │   ├── flow-matching-theory.md    # Example post
│   │   ├── ddpm-explained.md
│   │   ├── vlm-overview.md
│   │   └── generative-ai-survey.md
│   └── drafts/                        # Draft posts
├── code/                              # Code examples và tutorials
│   ├── flow-matching/
│   │   ├── flow_matching_pytorch.py   # Complete implementation
│   │   ├── requirements.txt
│   │   ├── README.md                  # Documentation
│   │   └── examples/
│   ├── ddpm/
│   │   ├── ddpm_implementation.py
│   │   ├── training_script.py
│   │   └── README.md
│   ├── vlm/
│   │   ├── clip_fine_tuning.py
│   │   ├── multimodal_eval.py
│   │   └── README.md
│   └── utils/
│       ├── visualization.py
│       ├── metrics.py
│       └── datasets.py
├── topics/                            # Topic-specific landing pages
│   ├── flow-matching.html
│   ├── ddpm.html
│   ├── generative-ai.html
│   └── vlm.html
├── about/
│   ├── index.html                     # About page
│   └── cv.pdf                         # Optional: CV/Resume
├── _config.yml                        # GitHub Pages configuration (optional)
├── robots.txt                         # SEO: Search engine instructions
├── sitemap.xml                        # SEO: Site structure
└── README.md                          # Repository documentation
```

## Workflow để sử dụng hệ thống:

### 1. Thêm bài viết mới:

**Tạo file:** `posts/2025/ten-bai-viet.md`
```markdown
---
title: "Title của bài viết"
date: "2025-10-13"
category: "Category Name"
tags: ["tag1", "tag2", "tag3"]
excerpt: "Mô tả ngắn gọn"
author: "ThanhLamDev"
readingTime: 15
featured: true
---

# Nội dung bài viết
...
```

**Update:** `api/posts.json` để thêm metadata

### 2. Thêm code examples:

**Tạo thư mục:** `code/topic-name/`
**Add files:**
- Python implementation
- README với hướng dẫn
- Requirements.txt
- Example notebooks

### 3. Deploy lên GitHub Pages:

```bash
git add .
git commit -m "Add new content"
git push origin main
```

GitHub Pages sẽ tự động serve tại `https://vuthanhlam124.github.io`

## Tính năng chính của hệ thống:

### ✅ **Scalable Architecture**
- JSON-based content management
- Dynamic loading với JavaScript
- Modular CSS với theme system
- Progressive enhancement

### ✅ **SEO Optimized** 
- Semantic HTML structure
- Meta tags cho mọi page
- Sitemap và robots.txt
- Open Graph tags

### ✅ **Performance**
- CSS/JS minification ready
- Image lazy loading
- Progressive loading
- Client-side caching

### ✅ **Developer Experience**
- Write trong Markdown
- Hot reload ready (với dev server)
- Easy content management
- Version control friendly

### ✅ **User Experience**
- Dark/Light theme toggle
- Search functionality
- Responsive design
- Fast navigation
- Reading progress indicator

## Next Steps để hoàn thiện:

1. **Replace index.html** với improved version [code_file:62]
2. **Add CSS file** `assets/css/main.css` [code_file:63]  
3. **Add JavaScript** `assets/js/main.js` [code_file:64]
4. **Setup API** với `api/posts.json` [code_file:65]
5. **Add sample post** `posts/2025/flow-matching-theory.md` [code_file:66]
6. **Add code examples** trong thư mục `code/`

Hệ thống này hoàn toàn static, compatible với GitHub Pages, và có thể scale từ vài posts tới hundreds of articles mà không cần database hay server backend!