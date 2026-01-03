# arXiv Submission Instructions

## Files to Submit

For arXiv submission, you only need **one file**:

- `arxiv-version.tex` - The complete LaTeX source with embedded bibliography

The bibliography is now embedded directly in the `.tex` file, so no `.bib` or `.bbl` files are needed.

## Compilation (Local Testing)

```bash
cd venues
pdflatex arxiv-version.tex
pdflatex arxiv-version.tex
```

Two runs are sufficient since the bibliography is embedded.

## Submission to arXiv

1. **Create submission**:
   - Go to https://arxiv.org/submit
   - Upload only `arxiv-version.tex`

2. **arXiv Categories** (recommended):
   - **Primary**: cs.AI (Artificial Intelligence)
   - **Secondary**: cs.PL (Programming Languages), cs.LO (Logic in Computer Science)

3. **Verify the PDF** in arXiv's preview before finalizing

## Code Repository

The Lean 4 source code is available at:
- GitHub: https://github.com/fixthatbug/sigmagov-lean4

Consider adding this URL to the arXiv abstract as well.

## Pre-submission Checklist

- [x] Bibliography embedded (no external .bib needed)
- [x] All `\bind` and `\seq` commands defined
- [x] GitHub repository is public
- [ ] Verify LaTeX compiles without errors locally
- [ ] Check PDF renders correctly
- [ ] Review arXiv metadata fields

## Changes from Previous Version

The following issues were fixed:
1. Added missing `\bind` and `\seq` command definitions
2. Embedded bibliography directly (removed `\bibliography{references}`)
3. All 12 citations are now defined inline
