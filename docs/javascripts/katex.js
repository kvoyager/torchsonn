// Render the math that pymdownx.arithmatex (generic mode) wraps in
// .arithmatex elements as \( ... \) and \[ ... \]. Only those delimiters are
// listed: a "$" delimiter would also typeset dollar amounts in ordinary text,
// such as "$100,000 ... $500k" in an included tutorial README. document$ is
// Material's page-load observable, so this also runs after instant
// navigation swaps the page content.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
});
