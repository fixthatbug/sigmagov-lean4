# SigmaGov Paper Bundle

> Complete submission package for "SigmaGov: A Formal Governance Calculus for LLM Agent Systems"

**Version**: 0.3.0  
**Date**: January 2026  
**Status**: Paper Ready

---

## Bundle Contents

```
paper-bundle/
├── README.md                    # This file
├── latex/
│   └── sigmagov-main.tex        # Main paper (full version)
├── supplementary/
│   └── appendix-axioms.tex      # Complete axiom reference
├── code/
│   ├── README.md                # Code documentation
│   ├── lakefile.lean            # Build configuration
│   └── *.lean                   # All 14 Lean 4 source files
└── venues/
    ├── VENUE-GUIDE.md           # Submission strategy
    ├── arxiv-version.tex        # arXiv (full, no limit)
    ├── aaai-version.tex         # AAAI (7 pages)
    └── acl-version.tex          # ACL (8 pages)
```

---

## Quick Verification

```bash
# Navigate to code directory
cd code/

# Build all Lean files
lake build

# Expected: "Build completed successfully (16 jobs)"
# Warnings: 3 unused variables only
# Errors: 0
# Sorry: 0
```

---

## Key Claims (Verifiable)

| Claim | Verification |
|-------|--------------|
| Zero sorry declarations | `grep -r "sorry" *.lean` returns 0 matches |
| ~5,300 lines | `wc -l *.lean` ≈ 5,300 |
| 14 Lean files | `ls *.lean | wc -l` = 14 |
| Build succeeds | `lake build` exits 0 |

---

## Formalization Statistics

| Metric | Value |
|--------|-------|
| Total Lean files | 14 |
| Total lines | ~5,300 |
| Axioms declared | ~60 |
| Theorems proven | ~40 |
| sorry declarations | **0** |
| Build warnings | 3 (unused vars) |
| Grade | A- |

---

## Repository

**GitHub**: https://github.com/fixthatbug/sigmagov-lean4

---

## Paper Summary

**Title**: SigmaGov: A Formal Governance Calculus for LLM Agent Systems

**Abstract**: We present SigmaGov, a formal governance calculus providing
machine-checkable specifications for LLM agent behavior, purpose tracking,
and achievement verification. Our Lean 4 formalization establishes:

1. Eight foundational axioms (T0–T8) for agent governance
2. Five-pillar architecture (Φ_sem, Φ_syn, Φ_auto, Φ_mem, Φ_ctx)
3. NPL convergence for semantic alignment verification
4. Workflow algebra with purpose preservation proofs

The complete formalization (~5,300 lines) has **zero incomplete proofs**.

---

## Reviewer Notes

### What to Look For
- **Axiom Design**: Are T0–T8 axioms well-motivated and complete?
- **NPL Semantics**: Is semantic convergence checking sound?
- **Workflow Algebra**: Are composition operators correctly specified?
- **Correspondence**: Do Lean definitions match TypeScript behavior?

### Known Limitations
1. **Float vs Rat**: Uses Lean Float, not rational numbers
2. **No Lakefile Package**: Standalone files, not a Lake package
3. **Test Correspondence**: Axioms bridge Lean↔TypeScript (by design)

### Questions Welcome On
- Design decisions for binary governance (T5)
- NPL threshold selection (τ = 0.85)
- Axiom vs theorem classification

---

## License

This work is licensed under:
- **Paper**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0)
- **Code**: [MIT License](https://opensource.org/licenses/MIT)

---

## Contact

Rui Wang  
University of Houston  
rwang19@uh.edu
