// Core JavaScript functionality for the blog
class BlogSystem {
    constructor() {
        this.posts = [];
        this.currentTheme = 'light';
        this.searchIndex = [];
        this.currentPage = 1;
        this.postsPerPage = 6;
        this.init();
    }

    async init() {
        await this.loadTheme();
        await this.loadPosts();
        this.setupEventListeners();
        this.setupProgressBar();
        this.setupAnimations();
    }

    // Theme Management
    async loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme);
    }

    setTheme(theme) {
        this.currentTheme = theme;
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.innerHTML = theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
        }
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    // Posts Management
    async loadPosts() {
        try {
            const response = await fetch('/api/posts.json');
            const data = await response.json();
            this.posts = data.posts || [];
            
            // Load actual content from markdown files
            await this.loadPostContents();
            
            this.buildSearchIndex();
            this.renderPosts();
        } catch (error) {
            console.log('Using placeholder posts data');
            this.loadPlaceholderPosts();
        }
    }

    async loadPostContents() {
        for (let post of this.posts) {
            try {
                // Try to load markdown content
                const markdownPath = `/posts/2025/${post.slug}.md`;
                const response = await fetch(markdownPath);
                if (response.ok) {
                    const content = await response.text();
                    post.content = this.parseMarkdown(content);
                    post.hasRealContent = true;
                }
            } catch (error) {
                console.log(`Could not load content for ${post.slug}`);
                post.hasRealContent = false;
            }
        }
    }

    parseMarkdown(content) {
        // Extract front matter and content
        const frontMatterRegex = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/;
        const match = content.match(frontMatterRegex);
        
        if (match) {
            const markdownContent = match[2];
            // Simple markdown parsing for excerpt
            const firstParagraph = markdownContent
                .split('\n\n')[0]
                .replace(/^#+\s+/gm, '') // Remove headers
                .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
                .replace(/\*(.*?)\*/g, '$1') // Remove italic
                .replace(/`(.*?)`/g, '$1') // Remove inline code
                .trim();
            
            return {
                full: markdownContent,
                excerpt: firstParagraph.substring(0, 200) + '...'
            };
        }
        
        return { full: content, excerpt: content.substring(0, 200) + '...' };
    }

    loadPlaceholderPosts() {
        this.posts = [
            {
                slug: 'flow-matching-theory',
                title: 'Flow Matching: Từ lý thuyết đến thực hành với PyTorch',
                date: '2025-10-13',
                category: 'Flow Matching',
                tags: ['Flow Matching', 'PyTorch', 'Generative AI'],
                excerpt: 'Deep dive vào Flow Matching theory, so sánh với Score-based Diffusion Models, và complete implementation từ scratch.',
                author: 'ThanhLamDev',
                readingTime: 15,
                featured: true,
                content: null
            },
            {
                slug: 'ddpm-explained',
                title: 'DDPM Explained: Toán học đằng sau Diffusion Models',
                date: '2025-10-12',
                category: 'DDPM',
                tags: ['DDPM', 'Diffusion', 'Mathematics'],
                excerpt: 'Phân tích chi tiết forward process, reverse process, training objective và sampling algorithms.',
                author: 'ThanhLamDev',
                readingTime: 12,
                featured: false,
                content: null
            },
            {
                slug: 'vlm-clip-llava',
                title: 'Vision-Language Models: CLIP, LLaVA và beyond',
                date: '2025-10-11',
                category: 'VLM',
                tags: ['VLM', 'CLIP', 'LLaVA', 'Multimodal'],
                excerpt: 'Khám phá kiến trúc và training strategies của VLMs hiện đại với hands-on examples.',
                author: 'ThanhLamDev',
                readingTime: 18,
                featured: true,
                content: null
            }
        ];
        this.buildSearchIndex();
        this.renderPosts();
    }

    buildSearchIndex() {
        this.searchIndex = this.posts.map(post => ({
            ...post,
            searchText: `${post.title} ${post.excerpt} ${post.tags.join(' ')} ${post.category}`.toLowerCase()
        }));
    }

    renderPosts(postsToRender = null) {
        const container = document.getElementById('posts-container');
        if (!container) return;

        const posts = postsToRender || this.posts;
        const startIndex = (this.currentPage - 1) * this.postsPerPage;
        const endIndex = startIndex + this.postsPerPage;
        const paginatedPosts = posts.slice(startIndex, endIndex);

        container.innerHTML = paginatedPosts.map(post => {
            // Use real content excerpt if available
            const displayExcerpt = post.hasRealContent && post.content ? 
                post.content.excerpt : post.excerpt;
            
            return `
            <article class="post-item" data-slug="${post.slug}">
                <div class="post-meta">
                    <span>📅 ${this.formatDate(post.date)}</span>
                    <span>🏷️ ${post.category}</span>
                    <span>⏱️ ${post.readingTime} min read</span>
                    ${post.featured ? '<span class="featured-badge">⭐ Featured</span>' : ''}
                    ${post.hasRealContent ? '<span class="content-badge">📄 Full Content</span>' : '<span class="preview-badge">👁️ Preview</span>'}
                </div>
                <h3 class="post-title">${post.title}</h3>
                <p class="post-excerpt">${displayExcerpt}</p>
                <div class="post-tags">
                    ${post.tags.map(tag => `<span class="tag" data-tag="${tag}">${tag}</span>`).join('')}
                </div>
                ${post.hasRealContent ? '<div class="read-more">Click to read full article →</div>' : '<div class="coming-soon">Coming soon...</div>'}
            </article>
            `;
        }).join('');

        // Add click handlers
        this.attachPostClickHandlers();
        this.renderPagination(posts.length);
    }

    attachPostClickHandlers() {
        document.querySelectorAll('.post-item').forEach(item => {
            item.addEventListener('click', () => {
                const slug = item.dataset.slug;
                this.showPost(slug);
            });
        });

        document.querySelectorAll('.tag').forEach(tag => {
            tag.addEventListener('click', (e) => {
                e.stopPropagation();
                this.filterByTag(tag.dataset.tag);
            });
        });
    }

    async showPost(slug) {
        const post = this.posts.find(p => p.slug === slug);
        if (!post) return;

        let content;
        if (post.hasRealContent && post.content) {
            content = this.markdownToHtml(post.content.full);
        } else {
            content = await this.loadPostContent(slug);
        }

        this.openModal(`
            <div class="post-modal-content">
                <button class="close-btn" onclick="blogSystem.closeModal()">&times;</button>
                <article class="full-post">
                    <header class="post-header">
                        <h1>${post.title}</h1>
                        <div class="post-meta">
                            <span>📅 ${this.formatDate(post.date)}</span>
                            <span>👤 ${post.author}</span>
                            <span>⏱️ ${post.readingTime} min read</span>
                            <span>🏷️ ${post.category}</span>
                        </div>
                        <div class="post-tags">
                            ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                        </div>
                    </header>
                    <div class="post-content">${content || this.getPlaceholderContent(post)}</div>
                </article>
            </div>
        `);
    }

    async loadPostContent(slug) {
        try {
            // Try different paths
            const paths = [
                `/posts/2025/${slug}.md`,
                `/posts/${slug}.md`,
                `/code/flow-matching/${slug}.py`
            ];
            
            for (const path of paths) {
                try {
                    const response = await fetch(path);
                    if (response.ok) {
                        const content = await response.text();
                        if (path.endsWith('.py')) {
                            return `<pre><code class="language-python">${this.escapeHtml(content)}</code></pre>`;
                        } else {
                            return this.markdownToHtml(content);
                        }
                    }
                } catch (e) {
                    continue;
                }
            }
            throw new Error('Content not found');
        } catch (error) {
            return this.getPlaceholderContent(this.posts.find(p => p.slug === slug));
        }
    }

    markdownToHtml(markdown) {
        // Remove front matter if present
        const contentWithoutFrontMatter = markdown.replace(/^---\n[\s\S]*?\n---\n/, '');
        
        // Simple markdown to HTML conversion
        let html = contentWithoutFrontMatter
            // Headers
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            // Bold and italic
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            // Line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
        
        return `<p>${html}</p>`;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    getPlaceholderContent(post) {
        return `
            <div class="placeholder-content">
                <p><strong>This post is coming soon!</strong></p>
                <p>${post.excerpt}</p>
                <p>Stay tuned for detailed content covering:</p>
                <ul>
                    ${post.tags.map(tag => `<li>In-depth analysis of ${tag}</li>`).join('')}
                    <li>Practical implementation examples</li>
                    <li>Code tutorials and best practices</li>
                    <li>Real-world applications and use cases</li>
                </ul>
                <p>Follow us on <a href="https://github.com/VuThanhLam124" target="_blank">GitHub</a> for updates!</p>
            </div>
        `;
    }

    markdownToHtml(markdown) {
        // Simple markdown parser - in production, use a proper markdown library
        return markdown
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/gim, '<em>$1</em>')
            .replace(/```([\\s\\S]*?)```/gim, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/gim, '<code>$1</code>')
            .replace(/\n/gim, '<br>');
    }

    // Search Functionality
    search(query) {
        if (!query.trim()) {
            this.renderPosts();
            return;
        }

        const results = this.searchIndex.filter(post => 
            post.searchText.includes(query.toLowerCase())
        );

        this.renderPosts(results);
        this.updateSearchStats(results.length, query);
    }

    updateSearchStats(count, query) {
        const statsContainer = document.getElementById('search-stats');
        if (statsContainer) {
            statsContainer.innerHTML = `Found ${count} posts matching "${query}"`;
        }
    }

    filterByTag(tag) {
        const filtered = this.posts.filter(post => post.tags.includes(tag));
        this.currentPage = 1; // Reset to first page
        this.renderPosts(filtered);
        this.updateSearchStats(filtered.length, `tag: ${tag}`);
    }

    // Search functionality
    search(query) {
        if (!query.trim()) {
            this.currentPage = 1;
            this.renderPosts();
            this.updateSearchStats(this.posts.length, '');
            return;
        }

        const searchTerms = query.toLowerCase().split(' ');
        const filtered = this.posts.filter(post => {
            const searchableText = (
                post.title + ' ' +
                post.excerpt + ' ' +
                post.category + ' ' +
                post.tags.join(' ') + ' ' +
                (post.content?.full || '')
            ).toLowerCase();

            return searchTerms.every(term => searchableText.includes(term));
        });

        this.currentPage = 1;
        this.renderPosts(filtered);
        this.updateSearchStats(filtered.length, `search: "${query}"`);
    }

    updateSearchStats(count, context) {
        const stats = document.getElementById('search-stats');
        if (stats) {
            if (context) {
                stats.textContent = `Found ${count} posts for ${context}`;
            } else {
                stats.textContent = `Showing ${count} posts`;
            }
        }
    }

    // Pagination
    renderPagination(totalPosts) {
        const container = document.getElementById('pagination-container');
        if (!container) return;

        const totalPages = Math.ceil(totalPosts / this.postsPerPage);
        if (totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        let paginationHTML = '<div class="pagination">';
        
        if (this.currentPage > 1) {
            paginationHTML += `<button onclick="blogSystem.goToPage(${this.currentPage - 1})">Previous</button>`;
        }

        for (let i = 1; i <= totalPages; i++) {
            const active = i === this.currentPage ? 'active' : '';
            paginationHTML += `<button class="${active}" onclick="blogSystem.goToPage(${i})">${i}</button>`;
        }

        if (this.currentPage < totalPages) {
            paginationHTML += `<button onclick="blogSystem.goToPage(${this.currentPage + 1})">Next</button>`;
        }

        paginationHTML += '</div>';
        container.innerHTML = paginationHTML;
    }

    goToPage(page) {
        this.currentPage = page;
        this.renderPosts();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Modal Management
    openModal(content) {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.innerHTML = content;
        document.body.appendChild(modal);
        document.body.style.overflow = 'hidden';

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal();
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', this.handleEscapeKey);
    }

    closeModal() {
        const modal = document.querySelector('.modal');
        if (modal) {
            modal.remove();
            document.body.style.overflow = '';
            document.removeEventListener('keydown', this.handleEscapeKey);
        }
    }

    handleEscapeKey = (e) => {
        if (e.key === 'Escape') {
            this.closeModal();
        }
    }

    // Progress Bar
    setupProgressBar() {
        window.addEventListener('scroll', this.updateProgressBar);
    }

    updateProgressBar() {
        const scrollTop = window.pageYOffset;
        const docHeight = document.body.offsetHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        const progressBar = document.getElementById('progress-bar');
        if (progressBar) {
            progressBar.style.width = Math.min(100, Math.max(0, scrollPercent)) + '%';
        }
    }

    // Event Listeners
    setupEventListeners() {
        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Search
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.search(e.target.value);
            });
        }

        // Modal close on backdrop click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal();
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });

        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    // Animations
    setupAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe elements with loading class
        document.querySelectorAll('.loading, .fade-in-delay').forEach(el => {
            observer.observe(el);
        });
    }

    // Utility Functions
    formatDate(dateString) {
        const options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        };
        return new Date(dateString).toLocaleDateString('vi-VN', options);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Modal functions
    openModal(content) {
        const modal = document.getElementById('post-modal') || this.createModal();
        modal.innerHTML = content;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    closeModal() {
        const modal = document.getElementById('post-modal');
        if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    createModal() {
        const modal = document.createElement('div');
        modal.id = 'post-modal';
        modal.className = 'modal';
        document.body.appendChild(modal);
        return modal;
    }
}

// Initialize the blog system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.blogSystem = new BlogSystem();
});

// Initialize blog system when DOM is loaded
let blogSystem;
document.addEventListener('DOMContentLoaded', () => {
    blogSystem = new BlogSystem();
});

// Export for global access
window.blogSystem = blogSystem;