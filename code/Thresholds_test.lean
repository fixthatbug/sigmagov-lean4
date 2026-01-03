import SigmaGov.Basic
import SigmaGov.NPL

/-
  SigmaGov.Thresholds - Formal Constants and Thresholds

  Lean 4 formalization of convergence thresholds from:
  - foundation/src/types.ts
  - foundation.gov v1.1.0
  - npl-convergence.gov v1.0.0

  All thresholds are formalized as axioms with invariant constraints.

  Note: Float arithmetic in Lean 4 is not decidable. Proofs involving
  concrete Float comparisons use IEEE 754 axioms that would need
  native verification.
-/

namespace SigmaGov.Thresholds

open SigmaGov
open SigmaGov.NPL

/-!
## Core Thresholds (tau)

Minimum similarity and coverage thresholds.
-/

/-- Minimum semantic conservation threshold (tau_min) -/
def tau_min : Float := 0.75

/-- NPL similarity threshold (tau_npl) -/
def tau_npl : Float := 0.85

/-- Axiom coverage threshold (tau_axiom) -/
def tau_axiom : Float := 0.85

/-- Coverage threshold for axiom projection -/
def COVERAGE_THRESHOLD : Float := 0.85

/-- Invariant: tau_npl >= tau_min -/
axiom tau_npl_ge_min : tau_npl >= tau_min

/-- Invariant: tau_axiom >= tau_min -/
axiom tau_axiom_ge_min : tau_axiom >= tau_min

/-- All thresholds in valid range [0, 1] -/
axiom thresholds_valid :
  0 <= tau_min && tau_min <= 1 &&
  0 <= tau_npl && tau_npl <= 1 &&
  0 <= tau_axiom && tau_axiom <= 1

end SigmaGov.Thresholds
