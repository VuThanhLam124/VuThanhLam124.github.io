class BlogApp {
    constructor() {
        this.state = {
            posts: [],
            categories: [],
            tags: [],
            filteredPosts: [],
            activeCategory: 'all',
            activeTag: null,
            searchQuery: '',
            currentPage: 1,
            postsPerPage: 6
        };

        this.postIndex = new Map();
        this.contentCache = new Map();
        this.sidebarVisible = false;
        this.currentTheme = 'light';

        this.elements = {
            body: document.body,
            postsGrid: document.getElementById('posts-grid'),
            pagination: document.getElementById('pagination'),
            searchInput: document.getElementById('search-input'),
            searchStats: document.getElementById('search-stats'),
            featuredHighlight: document.getElementById('featured-highlight'),
            categoryFilter: document.getElementById('category-filter'),
            tagFilter: document.getElementById('tag-filter'),
            categoryGrid: document.getElementById('category-grid'),
            sidebar: document.getElementById('sidebar'),
            sidebarToggle: document.getElementById('sidebar-toggle'),
            sidebarLatest: document.getElementById('sidebar-latest'),
            sidebarTopics: document.getElementById('sidebar-topics'),
            themeToggle: document.getElementById('theme-toggle'),
            modal: document.getElementById('post-modal'),
            modalContent: document.getElementById('modal-content'),
            totalPosts: document.getElementById('total-posts'),
            totalReadingTime: document.getElementById('total-reading-time'),
            currentYear: document.getElementById('current-year')
        };

        this.init();
    }

    async init() {
        this.updateCurrentYear();
        this.setupTheme();
        this.bindGlobalEvents();
        await this.loadData();
        this.renderInitialView();
    }

    updateCurrentYear() {
        if (this.elements.currentYear) {
            this.elements.currentYear.textContent = new Date().getFullYear();
        }
    }

    setupTheme() {
        const savedTheme = localStorage.getItem('lam-theme');
        if (savedTheme) {
            this.currentTheme = savedTheme;
        } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            this.currentTheme = 'dark';
        }
        this.applyTheme();
    }

    applyTheme() {
        document.body.setAttribute('data-theme', this.currentTheme);
        if (this.elements.themeToggle) {
            this.elements.themeToggle.textContent =
                this.currentTheme === 'dark' ? '☀️' : '🌙';
            this.elements.themeToggle.setAttribute(
                'aria-label',
                this.currentTheme === 'dark'
                    ? 'Chuyển sang giao diện sáng'
                    : 'Chuyển sang giao diện tối'
            );
        }
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('lam-theme', this.currentTheme);
        this.applyTheme();
    }

    bindGlobalEvents() {
        if (this.elements.themeToggle) {
            this.elements.themeToggle.addEventListener('click', () =>
                this.toggleTheme()
            );
        }

        if (this.elements.searchInput) {
            const handler = this.debounce((event) => {
                this.state.searchQuery = event.target.value || '';
                this.state.currentPage = 1;
                this.renderPostsSection();
            }, 220);
            this.elements.searchInput.addEventListener('input', handler);
        }

        if (this.elements.sidebarToggle) {
            this.elements.sidebarToggle.addEventListener('click', () =>
                this.toggleSidebar()
            );
        }

        document
            .querySelectorAll('[data-close-modal]')
            .forEach((trigger) => {
                trigger.addEventListener('click', () => this.closeModal());
            });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                this.closeModal();
                this.closeSidebar();
            }
        });

        document.addEventListener('click', (event) => {
            if (!this.sidebarVisible) {
                return;
            }
            const sidebar = this.elements.sidebar;
            const toggle = this.elements.sidebarToggle;
            if (
                sidebar &&
                !sidebar.contains(event.target) &&
                toggle &&
                !toggle.contains(event.target)
            ) {
                this.closeSidebar();
            }
        });
    }

    toggleSidebar() {
        if (this.sidebarVisible) {
            this.closeSidebar();
        } else {
            this.openSidebar();
        }
    }

    openSidebar() {
        if (!this.elements.sidebar) return;
        this.elements.sidebar.classList.add('sidebar--visible');
        document.body.classList.add('has-sidebar-open', 'sidebar-overlay-visible');
        this.sidebarVisible = true;
    }

    closeSidebar() {
        if (!this.elements.sidebar) return;
        this.elements.sidebar.classList.remove('sidebar--visible');
        document.body.classList.remove(
            'has-sidebar-open',
            'sidebar-overlay-visible'
        );
        this.sidebarVisible = false;
    }

    async loadData() {
        try {
            const response = await fetch('/api/posts.json');
            if (!response.ok) {
                throw new Error('Không thể tải dữ liệu bài viết');
            }
            const data = await response.json();

            const posts = (data.posts || []).map((post) => {
                const published =
                    post.publishedAt || `${post.date || ''}T00:00:00Z`;
                const dateObject = new Date(published);
                const year = Number.isNaN(dateObject.getFullYear())
                    ? new Date().getFullYear()
                    : dateObject.getFullYear();
                const safeTags = Array.isArray(post.tags) ? post.tags : [];
                const readingTime = Number(post.readingTime) || 8;

                const decoratedPost = {
                    ...post,
                    dateObject,
                    publishedAt: published,
                    year,
                    tags: safeTags,
                    readingTime
                };

                this.postIndex.set(post.slug, decoratedPost);
                return decoratedPost;
            });

            posts.sort(
                (a, b) => b.dateObject.getTime() - a.dateObject.getTime()
            );

            this.state.posts = posts;

            const categoryCounts = this.countPostsByCategory(posts);

            this.state.categories = this.hydrateCategories(
                data.categories,
                categoryCounts
            );
            this.state.tags = data.tags || this.buildTags(posts);
        } catch (error) {
            console.error(error);
            this.state.posts = [];
            this.state.categories = [];
            this.state.tags = [];
        }
    }

    countPostsByCategory(posts) {
        const counts = new Map();
        posts.forEach((post) => {
            const category = post.category || 'Chưa phân loại';
            counts.set(category, (counts.get(category) || 0) + 1);
        });
        return counts;
    }

    hydrateCategories(source, counts) {
        if (Array.isArray(source) && source.length) {
            return source
                .map((category) => ({
                    ...category,
                    postCount:
                        counts.get(category.name) ??
                        category.postCount ??
                        0
                }))
                .sort((a, b) => (b.postCount || 0) - (a.postCount || 0));
        }

        return Array.from(counts.entries())
            .map(([name, postCount]) => ({
                name,
                slug: name.toLowerCase().replace(/\s+/g, '-'),
                description: '',
                color: '#0b73b7',
                postCount
            }))
            .sort((a, b) => b.postCount - a.postCount);
    }

    buildTags(posts) {
        const tagCounts = new Map();
        posts.forEach((post) => {
            post.tags.forEach((tag) => {
                tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
            });
        });

        return Array.from(tagCounts.keys());
    }

    renderInitialView() {
        this.renderFeatured();
        this.renderFilters();
        this.renderSidebar();
        this.renderTopicsSection();
        this.renderPostsSection();
        this.updateStats();
    }

    renderFilters() {
        this.renderCategoryFilter();
        this.renderTagFilter();
    }

    renderCategoryFilter() {
        if (!this.elements.categoryFilter) return;

        const categories = [
            { label: 'Tất cả', value: 'all' },
            ...this.state.categories.map((category) => ({
                label: category.name,
                value: category.name
            }))
        ];

        this.elements.categoryFilter.innerHTML = categories
            .map(
                (category) => `
                <button
                    type="button"
                    class="chip ${
                        this.state.activeCategory === category.value
                            ? 'is-active'
                            : ''
                    }"
                    data-action="filter-category"
                    data-category="${category.value}"
                >
                    ${category.label}
                </button>
            `
            )
            .join('');

        this.elements.categoryFilter
            .querySelectorAll('[data-action="filter-category"]')
            .forEach((chip) => {
                chip.addEventListener('click', () => {
                    const value = chip.dataset.category;
                    this.state.activeCategory = value;
                    this.state.currentPage = 1;
                    this.renderCategoryFilter();
                    this.renderPostsSection();
                });
            });
    }

    renderTagFilter() {
        if (!this.elements.tagFilter) return;

        const topTags = this.getTopTags(12);
        this.elements.tagFilter.innerHTML = topTags
            .map(
                (tag) => `
                <button
                    type="button"
                    class="chip ${
                        this.state.activeTag === tag ? 'is-active' : ''
                    }"
                    data-action="filter-tag"
                    data-tag="${tag}"
                >
                    ${tag}
                </button>
            `
            )
            .join('');

        this.elements.tagFilter
            .querySelectorAll('[data-action="filter-tag"]')
            .forEach((chip) => {
                chip.addEventListener('click', () => {
                    const tag = chip.dataset.tag;
                    this.state.activeTag =
                        this.state.activeTag === tag ? null : tag;
                    this.state.currentPage = 1;
                    this.renderTagFilter();
                    this.renderPostsSection();
                });
            });
    }

    getTopTags(limit = 10) {
        const counts = new Map();
        this.state.posts.forEach((post) => {
            post.tags.forEach((tag) => {
                counts.set(tag, (counts.get(tag) || 0) + 1);
            });
        });

        return Array.from(counts.entries())
            .sort(([, countA], [, countB]) => countB - countA)
            .slice(0, limit)
            .map(([tag]) => tag);
    }

    renderSidebar() {
        this.renderSidebarLatest();
        this.renderSidebarTopics();
    }

    renderSidebarLatest() {
        if (!this.elements.sidebarLatest) return;
        const latestPosts = this.state.posts.slice(0, 12);

        this.elements.sidebarLatest.innerHTML = latestPosts
            .map(
                (post) => `
                <li>
                    <a href="#posts"
                       data-action="open-post"
                       data-slug="${post.slug}">
                        ${post.title}
                    </a>
                </li>
            `
            )
            .join('');

        this.attachPostOpenHandlers(this.elements.sidebarLatest);
    }

    renderSidebarTopics() {
        if (!this.elements.sidebarTopics) return;
        const topCategories = this.state.categories.slice(0, 10);

        this.elements.sidebarTopics.innerHTML = topCategories
            .map(
                (category) => `
                <button
                    type="button"
                    data-action="sidebar-category"
                    data-category="${category.name}"
                >
                    ${category.name}
                </button>
            `
            )
            .join('');

        this.elements.sidebarTopics
            .querySelectorAll('[data-action="sidebar-category"]')
            .forEach((button) => {
                button.addEventListener('click', () => {
                    const value = button.dataset.category;
                    this.state.activeCategory = value;
                    this.closeSidebar();
                    this.renderCategoryFilter();
                    this.renderPostsSection();
                });
            });
    }

    renderTopicsSection() {
        if (!this.elements.categoryGrid) return;
        if (!this.state.categories.length) {
            this.elements.categoryGrid.innerHTML =
                '<p class="empty-state">Chưa có dữ liệu chủ đề.</p>';
            return;
        }

        this.elements.categoryGrid.innerHTML = this.state.categories
            .map(
                (category) => `
                <article class="category-card">
                    <header>
                        <h3>${category.name}</h3>
                        <p class="category-card__count">${category.postCount} bài viết</p>
                    </header>
                    <p>${category.description || 'Khám phá các bài viết liên quan.'}</p>
                    <button
                        type="button"
                        class="ghost-button"
                        data-action="filter-category"
                        data-category="${category.name}"
                    >
                        Xem bài viết
                    </button>
                </article>
            `
            )
            .join('');

        this.elements.categoryGrid
            .querySelectorAll('[data-action="filter-category"]')
            .forEach((button) => {
                button.addEventListener('click', () => {
                    const value = button.dataset.category;
                    this.state.activeCategory = value;
                    this.state.currentPage = 1;
                    this.renderCategoryFilter();
                    this.renderPostsSection();
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                });
            });
    }

    renderFeatured() {
        if (!this.elements.featuredHighlight) return;

        if (!this.state.posts.length) {
            this.elements.featuredHighlight.innerHTML =
                '<p class="empty-state">Chưa có bài viết nào được đăng tải.</p>';
            return;
        }

        const featuredPosts = this.state.posts.filter((post) => post.featured);
        const toRender = (featuredPosts.length
            ? featuredPosts
            : this.state.posts
        ).slice(0, 2);

        this.elements.featuredHighlight.innerHTML = toRender
            .map(
                (post) => `
                <article class="feature-card"
                         data-action="open-post"
                         data-slug="${post.slug}">
                    <div class="feature-card__meta">
                        <span>⭐ Featured</span>
                        <span>${this.formatDate(post.dateObject)}</span>
                    </div>
                    <h3 class="feature-card__title">${post.title}</h3>
                    <p class="feature-card__excerpt">${post.excerpt}</p>
                    <div class="post-tags">
                        ${post.tags
                            .slice(0, 4)
                            .map(
                                (tag) => `<span class="tag">${tag}</span>`
                            )
                            .join('')}
                    </div>
                    <div>
                        <button
                            type="button"
                            class="ghost-button"
                            data-action="open-post"
                            data-slug="${post.slug}"
                        >
                            Đọc chi tiết
                        </button>
                    </div>
                </article>
            `
            )
            .join('<hr>');

        this.attachPostOpenHandlers(this.elements.featuredHighlight);
    }

    renderPostsSection() {
        this.state.filteredPosts = this.applyFilters();
        this.renderPosts();
        this.renderPagination();
        this.updateSearchStats();
        this.updateStats();
    }

    applyFilters() {
        const { posts, activeCategory, activeTag, searchQuery } = this.state;
        const normalizedQuery = searchQuery.trim().toLowerCase();

        return posts.filter((post) => {
            const matchesCategory =
                activeCategory === 'all' ||
                (post.category || '').toLowerCase() ===
                    activeCategory.toLowerCase();

            const matchesTag = !activeTag || post.tags.includes(activeTag);

            const matchesQuery =
                !normalizedQuery ||
                [
                    post.title,
                    post.excerpt,
                    post.category,
                    post.description,
                    post.tags.join(' ')
                ]
                    .filter(Boolean)
                    .some((field) =>
                        field.toLowerCase().includes(normalizedQuery)
                    );

            return matchesCategory && matchesTag && matchesQuery;
        });
    }

    renderPosts() {
        if (!this.elements.postsGrid) return;

        const { currentPage, postsPerPage, filteredPosts } = this.state;
        const start = (currentPage - 1) * postsPerPage;
        const end = start + postsPerPage;
        const paginatedPosts = filteredPosts.slice(start, end);

        if (!paginatedPosts.length) {
            this.elements.postsGrid.innerHTML =
                '<div class="empty-state">Không tìm thấy bài viết nào. Hãy thử đổi từ khóa hoặc chủ đề khác.</div>';
            return;
        }

        this.elements.postsGrid.innerHTML = paginatedPosts
            .map((post) => this.renderPostCard(post))
            .join('');

        this.attachPostOpenHandlers(this.elements.postsGrid);
    }

    renderPostCard(post) {
        const category = post.category || 'Chưa phân loại';
        return `
            <article class="post-card"
                     data-action="open-post"
                     data-slug="${post.slug}">
                <div class="post-card__meta">
                    <span>📅 ${this.formatDate(post.dateObject)}</span>
                    <span>🏷️ ${category}</span>
                </div>
                <h3 class="post-card__title">${post.title}</h3>
                <p class="post-card__excerpt">${post.excerpt}</p>
                <div class="post-tags">
                    ${post.tags
                        .map((tag) => `<span class="tag">${tag}</span>`)
                        .join('')}
                </div>
            </article>
        `;
    }

    renderPagination() {
        if (!this.elements.pagination) return;

        const { filteredPosts, postsPerPage, currentPage } = this.state;
        const totalPages = Math.ceil(filteredPosts.length / postsPerPage);

        if (totalPages <= 1) {
            this.elements.pagination.innerHTML = '';
            return;
        }

        const buttons = [];

        if (currentPage > 1) {
            buttons.push(
                `<button type="button" data-action="paginate" data-page="${
                    currentPage - 1
                }">Trước</button>`
            );
        }

        for (let page = 1; page <= totalPages; page += 1) {
            buttons.push(
                `<button type="button" data-action="paginate" data-page="${page}" ${
                    page === currentPage ? 'class="active"' : ''
                }>${page}</button>`
            );
        }

        if (currentPage < totalPages) {
            buttons.push(
                `<button type="button" data-action="paginate" data-page="${
                    currentPage + 1
                }">Sau</button>`
            );
        }

        this.elements.pagination.innerHTML = buttons.join('');
        this.elements.pagination
            .querySelectorAll('[data-action="paginate"]')
            .forEach((button) => {
                button.addEventListener('click', () => {
                    const page = Number(button.dataset.page) || 1;
                    this.state.currentPage = page;
                    this.renderPosts();
                    this.renderPagination();
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                });
            });
    }

    updateSearchStats() {
        if (!this.elements.searchStats) return;
        const total = this.state.filteredPosts.length;

        const parts = [];
        if (this.state.activeCategory !== 'all') {
            parts.push(`chủ đề "${this.state.activeCategory}"`);
        }
        if (this.state.activeTag) {
            parts.push(`tag "${this.state.activeTag}"`);
        }
        if (this.state.searchQuery.trim()) {
            parts.push(`từ khóa "${this.state.searchQuery.trim()}"`);
        }

        if (parts.length) {
            this.elements.searchStats.textContent = `Tìm thấy ${total} bài viết cho ${parts.join(
                ', '
            )}.`;
        } else {
            this.elements.searchStats.textContent = `Đang hiển thị ${total} bài viết.`;
        }
    }

    updateStats() {
        if (this.elements.totalPosts) {
            this.elements.totalPosts.textContent = this.state.posts.length;
        }

        if (this.elements.totalReadingTime) {
            const totalMinutes = this.state.posts.reduce(
                (sum, post) => sum + (post.readingTime || 0),
                0
            );
            this.elements.totalReadingTime.textContent = `${totalMinutes} phút`;
        }
    }

    attachPostOpenHandlers(root) {
        if (!root) return;
        root.querySelectorAll('[data-action="open-post"]').forEach((element) => {
            element.addEventListener('click', (event) => {
                event.preventDefault();
                const slug = element.dataset.slug;
                if (slug) {
                    if (this.sidebarVisible) {
                        this.closeSidebar();
                    }
                    this.openPost(slug);
                }
            });
        });
    }

    async openPost(slug) {
        const post = this.postIndex.get(slug);
        if (!post) return;

        const content = await this.loadPostContent(post);
        this.renderModal(post, content);
        this.openModal();
    }

    async loadPostContent(post) {
        if (this.contentCache.has(post.slug)) {
            return this.contentCache.get(post.slug);
        }

        const year = post.year || new Date().getFullYear();
        const candidatePaths = [
            `/posts/${year}/${post.slug}.md`,
            `/posts/${post.slug}.md`,
            `/posts/${post.slug}/index.md`
        ];

        for (const path of candidatePaths) {
            try {
                const response = await fetch(path);
                if (response.ok) {
                    const raw = await response.text();
                    const content = this.parseMarkdown(raw);
                    this.contentCache.set(post.slug, content);
                    return content;
                }
            } catch (error) {
                console.warn(`Không thể tải nội dung từ ${path}`, error);
            }
        }

        return `<p>Nội dung chi tiết sẽ sớm được cập nhật. Bạn có thể xem code và tài liệu liên quan trong mục <strong>Code Lab</strong>.</p>`;
    }

    parseMarkdown(raw) {
        if (!raw) return '<p></p>';

        const withoutFrontMatter = raw.replace(/^---[\s\S]*?---\s*/, '').trim();
        const lines = withoutFrontMatter.split(/\r?\n/);

        let html = '';
        let paragraphBuffer = [];
        let listBuffer = [];
        let inCodeBlock = false;
        let inMathBlock = false;
        let codeBuffer = [];
        let mathBuffer = [];
        let codeLanguage = '';

        const flushParagraph = () => {
            if (!paragraphBuffer.length) return;
            const text = paragraphBuffer.join(' ');
            html += `<p>${this.inlineMarkdown(text)}</p>`;
            paragraphBuffer = [];
        };

        const flushList = () => {
            if (!listBuffer.length) return;
            const items = listBuffer
                .map((item) => `<li>${this.inlineMarkdown(item)}</li>`)
                .join('');
            html += `<ul>${items}</ul>`;
            listBuffer = [];
        };

        const flushCode = () => {
            if (!codeBuffer.length) return;
            const content = this.escapeHtml(codeBuffer.join('\n'));
            const lang = codeLanguage ? ` class="language-${codeLanguage}"` : '';
            html += `<pre><code${lang}>${content}</code></pre>`;
            codeBuffer = [];
            codeLanguage = '';
        };

        const flushMath = () => {
            if (!mathBuffer.length) return;
            // Keep math blocks as-is for KaTeX to process
            html += '\n$$\n' + mathBuffer.join('\n') + '\n$$\n';
            mathBuffer = [];
        };

        lines.forEach((line) => {
            const trimmed = line.trim();

            // Handle code blocks
            if (trimmed.startsWith('```')) {
                if (inCodeBlock) {
                    flushCode();
                    inCodeBlock = false;
                } else {
                    flushParagraph();
                    flushList();
                    flushMath();
                    inCodeBlock = true;
                    // Extract language
                    codeLanguage = trimmed.substring(3).trim();
                }
                return;
            }

            if (inCodeBlock) {
                codeBuffer.push(line);
                return;
            }

            // Handle display math blocks $$
            if (trimmed === '$$') {
                if (inMathBlock) {
                    flushMath();
                    inMathBlock = false;
                } else {
                    flushParagraph();
                    flushList();
                    inMathBlock = true;
                }
                return;
            }

            if (inMathBlock) {
                mathBuffer.push(line);
                return;
            }

            if (!trimmed) {
                flushParagraph();
                flushList();
                return;
            }

            const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
            if (headingMatch) {
                flushParagraph();
                flushList();
                const level = headingMatch[1].length;
                const headingText = headingMatch[2];
                // Preserve inline math in headings
                html += `<h${level}>${headingText}</h${level}>`;
                return;
            }

            if (/^[-*]\s+/.test(trimmed)) {
                flushParagraph();
                const itemText = trimmed.replace(/^[-*]\s+/, '');
                // Preserve inline math in list items
                listBuffer.push(itemText);
                return;
            }

            // Regular paragraph - preserve inline math
            paragraphBuffer.push(trimmed);
        });

        flushParagraph();
        flushList();
        flushCode();
        flushMath();

        return html || '<p></p>';
    }

    inlineMarkdown(text) {
        // Don't touch $ symbols - let KaTeX handle them
        // Only process other markdown
        return text
            .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>');
    }

    escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    renderModal(post, content) {
        if (!this.elements.modalContent) return;
        const author = post.author || 'ThanhLamDev';
        const category = post.category || 'Chưa phân loại';
        this.elements.modalContent.innerHTML = `
            <article class="full-post">
                <header>
                    <h1>${post.title}</h1>
                    <div class="full-post__meta">
                        <span>📅 ${this.formatDate(post.dateObject)}</span>
                        <span>👤 ${author}</span>
                        <span>🏷️ ${category}</span>
                    </div>
                    <div class="post-tags">
                        ${post.tags.map((tag) => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                </header>
                <div class="full-post__body">
                    ${content}
                </div>
            </article>
        `;
        
        // Render LaTeX/KaTeX with delay to ensure DOM is ready
        setTimeout(() => this.renderMath(), 100);
    }
    
    renderMath() {
        // Check if KaTeX is loaded
        if (typeof renderMathInElement === 'undefined') {
            console.warn('KaTeX auto-render not loaded yet, retrying...');
            // Retry after libraries load
            setTimeout(() => this.renderMath(), 500);
            return;
        }
        
        // Render math in modal content
        if (this.elements.modalContent) {
            try {
                renderMathInElement(this.elements.modalContent, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\[', right: '\\]', display: true},
                        {left: '\\(', right: '\\)', display: false}
                    ],
                    throwOnError: false,
                    errorColor: '#cc0000',
                    strict: false,
                    trust: true
                });
                console.log('KaTeX rendering completed');
            } catch (error) {
                console.error('KaTeX rendering error:', error);
            }
        }
    }

    openModal() {
        if (!this.elements.modal) return;
        this.elements.modal.classList.add('is-visible');
        this.elements.modal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
    }

    closeModal() {
        if (!this.elements.modal) return;
        this.elements.modal.classList.remove('is-visible');
        this.elements.modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
    }

    formatDate(date) {
        try {
            return new Intl.DateTimeFormat('vi-VN', {
                day: '2-digit',
                month: 'short',
                year: 'numeric'
            }).format(date);
        } catch (error) {
            return '';
        }
    }

    debounce(fn, delay = 200) {
        let timeoutId;
        return (...args) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn.apply(this, args), delay);
        };
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.blogApp = new BlogApp();
});
