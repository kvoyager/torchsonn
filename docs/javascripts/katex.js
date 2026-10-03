// Render the math that pymdownx.arithmatex (generic mode) wraps in
// .arithmatex elements. document$ is Material's page-load observable, so this
// also runs after instant navigation swaps the page content.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
});
