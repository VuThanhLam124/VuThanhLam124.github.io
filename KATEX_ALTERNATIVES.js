// Alternative delimiters if $ causes issues
renderMathInElement(element, {
    delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '\\(', right: '\\)', display: false},  // Use \(...\) for inline
        {left: '\\[', right: '\\]', display: true}
    ],
    throwOnError: false,
    trust: true
});

// Then in markdown, use:
// Inline: \(f: \mathbb{R}^d \rightarrow \mathbb{R}^d\)
// Display: \[ ... \] or $$...$$
