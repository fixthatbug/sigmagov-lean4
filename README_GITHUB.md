# SigmaGov: A Formal Governance Calculus for LLM Agent Systems

[![Lean 4](https://img.shields.io/badge/Lean-4-blue.svg)](https://leanprover.github.io/lean4/)
[![Zero Sorry](https://img.shields.io/badge/sorry-0-brightgreen.svg)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Machine-verified specifications for LLM agent behavior, purpose tracking, and achievement verification.

## Overview

**SigmaGov** (ΣGov) is a formal governance calculus that provides rigorous, machine-checkable specifications for Large Language Model (LLM) agent systems. This repository contains the complete Lean 4 formalization accompanying the paper.

### Key Contributions

1. **Foundational Axioms (T0–T8)**: Eight axioms governing truthfulness, purpose immutability, binary achievement, and four-dimensional verification
2. **Five-Pillar Architecture**: Execution decomposition into Φ_sem, Φ_syn, Φ_auto, Φ_mem, Φ_ctx
3. **NPL Convergence**: Semantic alignment measurement with formal thresholds (τ_npl = 0.85)
4. **Workflow Algebra**: Compositional operators (>>=, >>>, |||) with purpose preservation proofs

## Quick Start

```bash
# Clone the repository
git clone https://github.com/fixthatbug/sigmagov-lean4.git
cd sigmagov-lean4/code

# Build (requires Lean 4 + Mathlib)
lake build

# Expected: Build completed successfully
```

### Prerequisites

- [Lean 4](https://leanprover.github.io/lean4/doc/setup.html) (v4.3.0+)
- [Mathlib4](https://github.com/leanprover-community/mathlib4) (v4.3.0)

## Repository Structure

```
sigmagov-lean4/
├── code/                    # Lean 4 formalization
│   ├── Basic.lean           # Core types: Layer, Purpose, Deontic
│   ├── Axioms.lean          # T0-T8 foundational axioms
│   ├── NPL.lean             # Natural Procedural Language
│   ├── Workflow.lean        # Workflow algebra
│   ├── Context.lean         # T8 Context Anchoring
│   └── ...                  # 14 source files total
├── latex/                   # Paper source
│   └── sigmagov-main.tex    # Main paper
├── venues/                  # Venue-specific versions
│   ├── arxiv-version.tex    # arXiv submission
│   └── references.bib       # Bibliography
└── supplementary/           # Additional materials
    └── appendix-axioms.tex  # Complete axiom reference
```

## Formalization Statistics

| Metric | Value |
|--------|-------|
| Source files | 14 |
| Total lines | ~5,300 |
| Axioms | ~60 |
| Theorems | ~40 |
| `sorry` declarations | **0** |

## The Axiom System

### T0: Truthfulness
All outputs must be grounded or acknowledge uncertainty.

### T1: Purpose Seeding
Every user prompt seeds exactly one purpose.

### T2: Binary Achievement
Purpose achievement is binary (true/false).

### T3: Five-Pillar Decomposition
System = Φ_sem ⊕ Φ_syn ⊕ Φ_auto ⊕ Φ_mem ⊕ Φ_ctx

### T4: Decision Gate
Tool invocation requires prior reasoning.

### T5: Binary Governance
∀φ: O(φ) ⊕ F(φ) — no permissibility state.

### T6: Achievement Dimensions
Achievement = WHAT ∧ WHERE ∧ HOW ∧ WHY

### T7: Layer Self-Containment
Each layer has complete, independent configuration.

### T8: Context Anchoring
All executions anchored to manifold M = (cwd, timestamp).

## Citation

```bibtex
@article{wang2026sigmagov,
  author    = {Rui Wang},
  title     = {SigmaGov: A Formal Governance Calculus for LLM Agent Systems},
  journal   = {arXiv preprint},
  year      = {2026},
  institution = {University of Houston}
}
```

## License

- **Code**: MIT License
- **Paper**: CC BY 4.0

## Author

**Rui Wang**  
University of Houston  
rwang19@uh.edu
