# arXiv Submission Instructions

## Files to Include

### Required Files
- `arxiv-version.tex`: The main LaTeX source file
- `references.bib`: The bibliography file

### Optional (Recommended)
- `arxiv-version.bbl`: Pre-compiled bibliography (recommended for reproducibility)

## Compilation Steps

1. Ensure you have a LaTeX distribution installed (e.g., TeX Live, MiKTeX).
2. Run the following commands:
   ```bash
   cd venues
   pdflatex arxiv-version.tex
   bibtex arxiv-version
   pdflatex arxiv-version.tex
   pdflatex arxiv-version.tex
   ```
3. This will generate `arxiv-version.pdf` and `arxiv-version.bbl`.

## Submission to arXiv

### Option A: Include .bbl file (Recommended)
1. Run compilation steps above to generate `.bbl` file
2. Create ZIP containing:
   - `arxiv-version.tex`
   - `arxiv-version.bbl`
3. Upload to arXiv

### Option B: Let arXiv compile
1. Create ZIP containing:
   - `arxiv-version.tex`  
   - `references.bib`
2. Upload to arXiv
3. Verify the generated PDF in arXiv's preview

## arXiv Categories

Recommended categories:
- **Primary**: cs.AI (Artificial Intelligence)
- **Secondary**: cs.PL (Programming Languages), cs.LO (Logic in Computer Science)

## Supplementary Materials

The Lean 4 source code is available at:
- GitHub: https://github.com/fixthatbug/sigmagov-lean4

Consider linking to the repository in the arXiv abstract or as ancillary files.

## Pre-submission Checklist

- [ ] Verify all LaTeX compiles without errors
- [ ] Check PDF renders correctly
- [ ] Confirm GitHub repository is public
- [ ] Update repository URL in paper if needed
- [ ] Review arXiv metadata fields
