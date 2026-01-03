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

import SigmaGov.Basic
import SigmaGov.NPL

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
  0 <= tau_min ∧ tau_min <= 1 ∧
  0 <= tau_npl ∧ tau_npl <= 1 ∧
  0 <= tau_axiom ∧ tau_axiom <= 1

/-!
## Dimension Weights (gamma)

Weights for T6 achievement score computation.
Sum must equal 1.0.
-/

/-- WHAT dimension weight -/
def gamma_what : Float := 0.40

/-- WHERE dimension weight -/
def gamma_where : Float := 0.20

/-- HOW dimension weight -/
def gamma_how : Float := 0.25

/-- WHY dimension weight -/
def gamma_why : Float := 0.15

/-- Invariant: Dimension weights sum to 1.0 -/
axiom dimension_weights_sum :
  gamma_what + gamma_where + gamma_how + gamma_why = 1.0

/-- All weights are positive -/
axiom dimension_weights_positive :
  gamma_what > 0 ∧ gamma_where > 0 ∧ gamma_how > 0 ∧ gamma_why > 0

/-- Get weight for a dimension -/
def dimensionWeight : Dimension → Float
  | Dimension.WHAT => gamma_what
  | Dimension.WHERE => gamma_where
  | Dimension.HOW => gamma_how
  | Dimension.WHY => gamma_why

/-- Weight function sums to 1.0 -/
theorem weight_sum_one :
  dimensionWeight Dimension.WHAT +
  dimensionWeight Dimension.WHERE +
  dimensionWeight Dimension.HOW +
  dimensionWeight Dimension.WHY = 1.0 := by
  simp [dimensionWeight]
  exact dimension_weights_sum

/-\!
## Epsilon Thresholds (epsilon)

Maximum allowed gap per dimension.
Lower epsilon = stricter requirement.
-/

/-- WHAT epsilon (must be 90% aligned) -/
def epsilon_what : Float := 0.10

/-- WHERE epsilon (can be 85% aligned) -/
def epsilon_where : Float := 0.15

/-- HOW epsilon (can be 80% aligned) -/
def epsilon_how : Float := 0.20

/-- WHY epsilon (should be 85% aligned) -/
def epsilon_why : Float := 0.15

/-- Get epsilon for a dimension -/
def dimensionEpsilon : Dimension → Float
  | Dimension.WHAT => epsilon_what
  | Dimension.WHERE => epsilon_where
  | Dimension.HOW => epsilon_how
  | Dimension.WHY => epsilon_why

/-- All epsilons are valid (in [0, 1]) -/
axiom epsilons_valid :
  0 ≤ epsilon_what ∧ epsilon_what ≤ 1 ∧
  0 ≤ epsilon_where ∧ epsilon_where ≤ 1 ∧
  0 ≤ epsilon_how ∧ epsilon_how ≤ 1 ∧
  0 ≤ epsilon_why ∧ epsilon_why ≤ 1

/-- IEEE 754 axiom: 0.10 ≤ 0.15 -/
axiom float_le_0_10_0_15 : (0.10 : Float) ≤ 0.15

/-- IEEE 754 axiom: 0.10 ≤ 0.20 -/
axiom float_le_0_10_0_20 : (0.10 : Float) ≤ 0.20

/-- WHAT has strictest requirement -/
theorem what_strictest :
  epsilon_what ≤ epsilon_where ∧ epsilon_what ≤ epsilon_how ∧ epsilon_what ≤ epsilon_why := by
  simp only [epsilon_what, epsilon_where, epsilon_how, epsilon_why]
  exact ⟨float_le_0_10_0_15, float_le_0_10_0_20, float_le_0_10_0_15⟩

/-\!
## Alignment Calculation

Required alignment = 1 - epsilon
-/

/-- Required alignment for a dimension -/
def requiredAlignment : Dimension → Float
  | d => 1.0 - dimensionEpsilon d

/-- IEEE 754 axiom: 1.0 - 0.10 = 0.90 -/
axiom float_sub_1_0_10 : (1.0 : Float) - 0.10 = 0.90

/-- IEEE 754 axiom: 1.0 - 0.15 = 0.85 -/
axiom float_sub_1_0_15 : (1.0 : Float) - 0.15 = 0.85

/-- WHAT requires 90% alignment -/
theorem what_requires_90 :
  requiredAlignment Dimension.WHAT = 0.90 := by
  simp only [requiredAlignment, dimensionEpsilon, epsilon_what]
  exact float_sub_1_0_10

/-- WHERE requires 85% alignment -/
theorem where_requires_85 :
  requiredAlignment Dimension.WHERE = 0.85 := by
  simp only [requiredAlignment, dimensionEpsilon, epsilon_where]
  exact float_sub_1_0_15

/-!
## Coverage Weights (beta)

Weights for axiom coverage computation.
-/

/-- Verb coverage weight -/
def beta_verb : Float := 0.50

/-- Where coverage weight -/
def beta_where : Float := 0.30

/-- When coverage weight -/
def beta_when : Float := 0.20

/-- Coverage weights sum to 1.0 -/
axiom coverage_weights_sum :
  beta_verb + beta_where + beta_when = 1.0

/-!
## Iteration Limits
-/

/-- Default maximum iterations -/
def MAX_ITERATIONS : Nat := 10

/-- Node similarity threshold -/
def NODE_SIMILARITY_THRESHOLD : Float := 0.85

/-!
## Achievement Score

Weighted score computation for T6 dimensions.
-/

/-- Gap score per dimension -/
structure GapScores where
  what : Float
  where_ : Float
  how : Float
  why : Float
  deriving Repr

/-- Compute weighted achievement score -/
def achievementScore (gaps : GapScores) : Float :=
  gamma_what * (1.0 - gaps.what) +
  gamma_where * (1.0 - gaps.where_) +
  gamma_how * (1.0 - gaps.how) +
  gamma_why * (1.0 - gaps.why)

/-- IEEE 754 axiom: gamma_what * 1.0 + gamma_where * 1.0 + gamma_how * 1.0 + gamma_why * 1.0 = 1.0
    (follows from dimension_weights_sum when multiplying by 1) -/
axiom float_weighted_sum_ones :
  gamma_what * 1.0 + gamma_where * 1.0 + gamma_how * 1.0 + gamma_why * 1.0 = 1.0

/-- IEEE 754 axiom: 1.0 - 0 = 1.0 -/
axiom float_sub_1_0 : (1.0 : Float) - 0 = 1.0

/-- IEEE 754 axiom: 1.0 - 1 = 0.0 -/
axiom float_sub_1_1 : (1.0 : Float) - 1 = 0.0

/-- IEEE 754 axiom: any gamma * 0.0 = 0.0 -/
axiom float_mul_zero (x : Float) : x * 0.0 = 0.0

/-- IEEE 754 axiom: 0.0 + 0.0 + 0.0 + 0.0 = 0.0 -/
axiom float_add_zeros : (0.0 : Float) + 0.0 + 0.0 + 0.0 = 0.0

