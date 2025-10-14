(function bootstrapKaTeX() {
    const VERSION = '0.16.9';
    const CDN_BASE = `https://cdn.jsdelivr.net/npm/katex@${VERSION}/dist`;
    const STYLE_ID = 'katex-cdn-stylesheet';
    const SCRIPT_ID = 'katex-cdn-script';
    const AUTO_RENDER_ID = 'katex-auto-render-script';

    function ensureStylesheet() {
        if (document.getElementById(STYLE_ID)) {
            return;
        }

        const link = document.createElement('link');
        link.id = STYLE_ID;
        link.rel = 'stylesheet';
        link.href = `${CDN_BASE}/katex.min.css`;
        link.crossOrigin = 'anonymous';
        link.integrity =
            'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV';
        document.head.appendChild(link);
    }

    function loadScript(src, id) {
        return new Promise((resolve, reject) => {
            if (document.getElementById(id)) {
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.id = id;
            script.src = src;
            script.async = true;
            script.crossOrigin = 'anonymous';
            script.addEventListener('load', () => resolve());
            script.addEventListener('error', () =>
                reject(new Error(`Không thể tải script ${src}`))
            );
            document.head.appendChild(script);
        });
    }

    async function ensureLibraries() {
        ensureStylesheet();

        if (typeof window.katex === 'undefined') {
            await loadScript(
                `${CDN_BASE}/katex.min.js`,
                SCRIPT_ID
            );
        }

        if (typeof window.renderMathInElement === 'undefined') {
            await loadScript(
                `${CDN_BASE}/contrib/auto-render.min.js`,
                AUTO_RENDER_ID
            );
        }
    }

    function renderMath(root) {
        if (!root || typeof window.renderMathInElement === 'undefined') {
            return;
        }

        try {
            window.renderMathInElement(root, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false }
                ],
                throwOnError: false,
                errorColor: '#cc0000',
                strict: false,
                trust: true
            });
        } catch (error) {
            console.error('KaTeX rendering error:', error);
        }
    }

    async function init() {
        try {
            await ensureLibraries();
            renderMath(document.body);
        } catch (error) {
            console.error('Không thể khởi tạo KaTeX:', error);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();

