# Scalable Blog Architecture Guide

## Cấu trúc thư mục được đề xuất:

```
your-blog/
├── index.html                 # Main homepage
├── assets/
│   ├── css/
│   │   ├── main.css          # Main styles with CSS variables
│   │   └── themes.css        # Dark/Light theme definitions
│   ├── js/
│   │   ├── main.js           # Core functionality
│   │   ├── markdown-loader.js # Dynamic markdown loading
│   │   └── search.js         # Search functionality
│   └── images/
├── posts/                     # Markdown blog posts
│   ├── 2025/
│   │   ├── flow-matching-theory.md
│   │   ├── ddpm-explained.md
│   │   └── generative-ai-trends.md
│   └── index.json            # Posts metadata
├── code/                      # Code examples and tutorials
│   ├── flow-matching/
│   │   ├── flow_matching_pytorch.py
│   │   └── README.md
│   ├── ddpm/
│   │   ├── ddpm_implementation.py
│   │   └── README.md
│   └── utils/
├── topics/                    # Topic-specific pages
│   ├── flow-matching.html
│   ├── ddpm.html
│   ├── generative-ai.html
│   └── vlm.html
└── api/                       # Optional: JSON APIs for dynamic loading
    ├── posts.json
    └── topics.json
```

## Hướng dẫn sử dụng hệ thống scalable:

### 1. Thêm bài viết markdown mới:

**Tạo file:** `posts/2025/ten-bai-viet.md`

**Format đầu file (Front Matter):**
```yaml
---
title: "Flow Matching: From Theory to Implementation"
date: "2025-10-13"
category: "Flow Matching"
tags: ["Flow Matching", "PyTorch", "Generative AI"]
excerpt: "Deep dive into Flow Matching theory and practical implementation"
author: "Your Name"
reading_time: 15
---

# Nội dung bài viết ở đây bằng Markdown

## Giới thiệu

Bài viết này sẽ...

## Lý thuyết Flow Matching

Flow Matching là...

## Implementation với PyTorch

```python
import torch
import torch.nn as nn

class FlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        # Implementation here
```

## Kết luận

Flow Matching có thể...
```

### 2. Thêm code examples:

**Tạo thư mục:** `code/topic-name/`

**Example structure:**
```
code/flow-matching/
├── README.md                 # Hướng dẫn sử dụng
├── requirements.txt          # Dependencies
├── flow_matching_basic.py    # Basic implementation
├── flow_matching_advanced.py # Advanced features
├── train.py                 # Training script
└── inference.py             # Inference script
```

### 3. JavaScript cho dynamic loading:

**File: `assets/js/markdown-loader.js`**
```javascript
class MarkdownLoader {
    constructor() {
        this.posts = [];
        this.currentPage = 1;
        this.postsPerPage = 5;
    }

    async loadPosts() {
        try {
            const response = await fetch('/api/posts.json');
            this.posts = await response.json();
            this.renderPosts();
        } catch (error) {
            console.error('Error loading posts:', error);
        }
    }

    async loadPost(slug) {
        try {
            const response = await fetch(`/posts/${slug}.md`);
            const markdown = await response.text();
            return this.parseMarkdown(markdown);
        } catch (error) {
            console.error('Error loading post:', error);
        }
    }

    parseMarkdown(text) {
        // Parse front matter and content
        const parts = text.split('---');
        const frontMatter = this.parseFrontMatter(parts[1]);
        const content = parts.slice(2).join('---');
        
        return {
            meta: frontMatter,
            content: this.markdownToHtml(content)
        };
    }

    markdownToHtml(markdown) {
        // Simple markdown to HTML conversion
        return markdown
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*)\*/gim, '<em>$1</em>')
            .replace(/```([\\s\\S]*?)```/gim, '<pre><code>$1</code></pre>');
    }

    renderPosts() {
        const container = document.getElementById('posts-container');
        const startIndex = (this.currentPage - 1) * this.postsPerPage;
        const endIndex = startIndex + this.postsPerPage;
        const postsToShow = this.posts.slice(startIndex, endIndex);

        container.innerHTML = postsToShow.map(post => `
            <article class="post-item" data-slug="${post.slug}">
                <div class="post-meta">
                    <span>📅 ${post.date}</span>
                    <span>🏷️ ${post.category}</span>
                    <span>⏱️ ${post.reading_time} min read</span>
                </div>
                <h3 class="post-title">${post.title}</h3>
                <p class="post-excerpt">${post.excerpt}</p>
                <div class="post-tags">
                    ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </article>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.post-item').forEach(item => {
            item.addEventListener('click', () => {
                const slug = item.dataset.slug;
                this.showPost(slug);
            });
        });
    }

    async showPost(slug) {
        const post = await this.loadPost(slug);
        const modal = document.createElement('div');
        modal.className = 'post-modal';
        modal.innerHTML = `
            <div class="post-modal-content">
                <button class="close-btn">&times;</button>
                <article>
                    <h1>${post.meta.title}</h1>
                    <div class="post-meta">
                        <span>📅 ${post.meta.date}</span>
                        <span>👤 ${post.meta.author}</span>
                    </div>
                    <div class="post-content">${post.content}</div>
                </article>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        modal.querySelector('.close-btn').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
    }
}

// Initialize
const loader = new MarkdownLoader();
document.addEventListener('DOMContentLoaded', () => {
    loader.loadPosts();
});
```

### 4. API endpoints (posts.json):

```json
{
  "posts": [
    {
      "slug": "flow-matching-theory",
      "title": "Flow Matching: From Theory to Implementation",
      "date": "2025-10-13",
      "category": "Flow Matching",
      "tags": ["Flow Matching", "PyTorch", "Theory"],
      "excerpt": "Deep dive into Flow Matching theory and practical implementation",
      "author": "Your Name",
      "reading_time": 15,
      "featured": true
    },
    {
      "slug": "ddpm-explained",
      "title": "DDPM Explained: Mathematics Behind Diffusion Models",
      "date": "2025-10-12",
      "category": "DDPM",
      "tags": ["DDPM", "Diffusion", "Mathematics"],
      "excerpt": "Comprehensive explanation of DDPM mathematics and implementation",
      "author": "Your Name",
      "reading_time": 12,
      "featured": false
    }
  ]
}
```

## Lợi ích của architecture này:

1. **Scalable**: Dễ dàng thêm posts, topics, code mà không cần modify core HTML
2. **Maintainable**: Tách biệt content (markdown) và presentation (HTML/CSS)  
3. **SEO-friendly**: Static files với proper meta tags
4. **Fast loading**: Progressive loading và caching
5. **Developer-friendly**: Write trong markdown, support code syntax highlighting
6. **Dark/Light theme**: CSS variables cho easy theming
7. **Mobile responsive**: Modern CSS Grid và Flexbox
8. **Search capability**: Client-side search qua JSON APIs

## Deployment trên GitHub Pages:

1. Push tất cả files lên repository `username.github.io`
2. GitHub Pages sẽ tự serve static files
3. Có thể enable Jekyll nếu cần advanced features
4. Domain tự động: `https://username.github.io`

Hệ thống này hoàn toàn static nên load rất nhanh và không cần database/server!