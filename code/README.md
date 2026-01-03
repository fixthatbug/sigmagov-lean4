# SigmaGov Lean 4 Formalization - Code Archive

> Supplementary code for "SigmaGov: A Formal Governance Calculus for LLM Agent Systems"

## Quick Start

```bash
# Ensure Lean 4 is installed (https://leanprover.github.io/lean4/doc/setup.html)

# Build all files
lake build

# Expected output: "Build completed successfully (16 jobs)"
# Warnings: 3 unused variable warnings (linter only)
```

## Dependencies

- **Lean 4**: v4.3.0 or later
- **Mathlib**: v4.3.0 (commit `abc123...` - update with actual commit)

To update Mathlib:
```bash
lake update
lake build
```

## File Overview

| File | Lines | Description |
|------|-------|-------------|
| Basic.lean | 295 | Core types: Layer, Purpose, Goal, Deontic |
| Axioms.lean | 493 | T0-T8 foundational axioms |
| Primitives.lean | 268 | Phase, Timestamp, Agent primitives |
| Thresholds.lean | 237 | Constants: tau_min, tau_npl, tau_axiom |
| Thresholds_test.lean | 237 | Threshold validation tests |
| Context.lean | 288 | T8 Context Anchoring |
| Invariants.lean | 311 | Pi_1 through Pi_10 invariants |
| NPL.lean | 549 | Natural Procedural Language convergence |
| Workflow.lean | 350 | Workflow algebra (>>=, >>>, \|\|\|) |
| FractalInvariants.lean | 309 | Fractal purpose invariants |
| AxiomLoader.lean | 445 | Dynamic axiom projection |
| TestCorrespondence.lean | 757 | TypeScript test correspondence |
| Decomposer.lean | 763 | Purpose decomposition |
| Temporal.lean | 364 | Temporal governance |

**Total: 14 source files, ~5,300 lines, 0 sorry declarations**

Note: `lakefile.lean.example` is provided as a template. Rename to `lakefile.lean` when using standalone.

## Key Theorems

### T6 Achievement
```lean
-- Axioms.lean
theorem T6_achievement_requires_all_dimensions :
  ∀ (S : Session) (P : Purpose) (D : Dimensions S P),
    T6_achievement S P D ↔ (WHAT S P D.what ∧ WHERE S P D.where_ ∧ 
                            HOW S P D.how ∧ WHY S P D.why)
```

### Workflow Composition
```lean
-- Workflow.lean
theorem bind_preserves_purpose :
  ∀ (w1 : Workflow α) (f : α → Workflow β) (p : Purpose),
    (w1 >>= f) p = (w1 p >>= fun a => f a p)
```

### NPL Convergence
```lean
-- NPL.lean
def nplConverged (purpose result : NPL) (tau : Float) 
    (epsilon : Dimension → Float) : Prop :=
  nplSimilar purpose result tau ∧ 
  nplStructurallyAligned purpose result epsilon
```

## Dependencies

- Lean 4 (tested with v4.x)
- Mathlib (for standard library extensions)

## License

MIT License

Copyright (c) 2026 Rui Wang, University of Houston

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

## Citation

```bibtex
@software{sigmagov_lean4,
  author = {Wang, Rui},
  title = {SigmaGov: A Formal Governance Calculus for LLM Agent Systems},
  year = {2026},
  version = {0.3.0},
  institution = {University of Houston},
  url = {https://github.com/fixthatbug/sigmagov-lean4}
}
```
