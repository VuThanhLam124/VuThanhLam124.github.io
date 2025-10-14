// Debug script - Add to browser console to check KaTeX status

console.log('=== KaTeX Debug Info ===');

// Check if KaTeX is loaded
console.log('KaTeX loaded:', typeof katex !== 'undefined');
console.log('renderMathInElement loaded:', typeof renderMathInElement !== 'undefined');

// Check modal content
const modalContent = document.getElementById('modal-content');
console.log('Modal content exists:', modalContent !== null);

if (modalContent) {
    // Check for raw LaTeX in content
    const html = modalContent.innerHTML;
    console.log('Has $$ blocks:', html.includes('$$'));
    console.log('Has $ inline:', /\$[^$]+\$/.test(html));
    console.log('Sample content (first 500 chars):', html.substring(0, 500));
    
    // Try manual render
    if (typeof renderMathInElement !== 'undefined') {
        console.log('Attempting manual KaTeX render...');
        renderMathInElement(modalContent, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false,
            trust: true
        });
        console.log('Manual render completed');
    }
}

// Check for .katex elements
const katexElements = document.querySelectorAll('.katex');
console.log('Number of rendered KaTeX elements:', katexElements.length);

console.log('=== End Debug ===');
