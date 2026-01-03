# SigmaGov Paper - Venue Submission Guide

## Target Venues

### Tier 1: Formal Methods / Verification
| Venue | Deadline | Page Limit | Format | Fit |
|-------|----------|------------|--------|-----|
| **CAV** | Jan/Feb | 20 pages | LNCS | ⭐⭐⭐⭐⭐ |
| **POPL** | Jul | 25 pages | ACM | ⭐⭐⭐⭐ |
| **PLDI** | Nov | 12 pages | ACM | ⭐⭐⭐ |

### Tier 1: AI/ML Safety
| Venue | Deadline | Page Limit | Format | Fit |
|-------|----------|------------|--------|-----|
| **NeurIPS** | May | 9 pages | NeurIPS | ⭐⭐⭐⭐ |
| **ICML** | Jan | 9 pages | ICML | ⭐⭐⭐⭐ |
| **AAAI** | Aug | 7 pages | AAAI | ⭐⭐⭐⭐⭐ |

### Tier 1: NLP/Agents
| Venue | Deadline | Page Limit | Format | Fit |
|-------|----------|------------|--------|-----|
| **ACL** | Jan | 8 pages | ACL | ⭐⭐⭐⭐ |
| **EMNLP** | Jun | 8 pages | ACL | ⭐⭐⭐⭐ |
| **NAACL** | Dec | 8 pages | ACL | ⭐⭐⭐ |

### Preprint
| Venue | Timeline | Format |
|-------|----------|--------|
| **arXiv** | Immediate | Any LaTeX |

---

## Venue-Specific Adaptations

### arXiv (arxiv-version.tex)
- Full paper, no page limit
- Include all proofs and appendices inline
- Add arXiv identifier placeholder
- License: CC-BY 4.0

### AAAI (aaai-version.tex)  
- 7 pages + references
- AAAI Press format
- Emphasize: AI safety, agent governance, practical applicability
- Cut: Detailed Lean proofs (move to appendix)

### ACL (acl-version.tex)
- 8 pages + references
- ACL Anthology format
- Emphasize: NLP aspects (semantic alignment, embeddings)
- Cut: Formal methods background

### CAV (cav-version.tex)
- 20 pages LNCS
- Full formal details
- Emphasize: Verification, soundness, Lean proofs
- Include: Complete theorem statements

---

## Pitch Angles by Venue

### Formal Methods (CAV, POPL)
> "First machine-verified governance framework for LLM agents"
- Lead with Lean 4 formalization
- Emphasize zero-sorry achievement
- Focus on soundness and completeness

### AI Safety (NeurIPS, ICML, AAAI)
> "Formal specifications for verifiable LLM agent alignment"
- Lead with alignment problem
- Emphasize practical applicability
- Include empirical validation plan

### NLP (ACL, EMNLP)
> "Semantic convergence checking for LLM agent task completion"
- Lead with NPL and embeddings
- Emphasize linguistic grounding
- Connect to instruction following literature

---

## Submission Checklist

- [ ] Main paper (venue format)
- [ ] Supplementary materials (axiom appendix)
- [ ] Code archive (anonymized)
- [ ] Author response template
- [ ] Ethics statement (if required)
- [ ] Reproducibility checklist
