/-
  SigmaGov.Primitives - Extended Type Primitives

  Formalization of the extended PRIMITIVES specification.
  Adds new types with mutability and pillar anchoring semantics.

  New primitives:
  - CWD: Immutable Φ_ctx anchor (physical boundary)
  - Timestamp: Immutable Φ_ctx anchor (temporal)
  - Knowledge: Dual anchoring (Φ_mem or Φ_ctx ∘ Φ_mem)
  - Protocol: Immutable Φ_ctx ∘ Φ_mem
  - Constraint: Immutable Φ_ctx ∘ Φ_mem
  - Execution: Mutable Phase → Trajectory
-/

import SigmaGov.Basic
import SigmaGov.Workflow

namespace SigmaGov.Primitives

open SigmaGov
open SigmaGov.Workflow

/-!
## Φ_ctx Anchors

Immutable context anchors that establish the execution boundary.
-/

/-- CWD: Immutable physical boundary of execution (Φ_ctx anchor) -/
structure CWD where
  path : String
  deriving DecidableEq, Repr

/-- Timestamp: Immutable temporal anchor (Φ_ctx anchor) -/
structure Timestamp where
  value : Nat
  deriving DecidableEq, Repr, Ord

/-- Context: Composition of Φ_ctx anchors -/
structure Context where
  cwd : CWD
  timestamp : Timestamp
  deriving Repr

/-- Context equality -/
instance : DecidableEq Context := fun c1 c2 =>
  if h1 : c1.cwd = c2.cwd then
    if h2 : c1.timestamp = c2.timestamp then
      isTrue (by cases c1; cases c2; simp_all)
    else
      isFalse (by intro h; cases h; exact h2 rfl)
  else
    isFalse (by intro h; cases h; exact h1 rfl)

/-!
## Knowledge Primitive

Dual anchoring based on tier:
- Fact/Official: Pure Φ_mem (immutable, global)
- Custom: Φ_ctx ∘ Φ_mem (mutable, session-scoped)
-/

/-- Knowledge tier determines mutability and anchoring -/
inductive KnowledgeTier
  /-- Immutable fact, pure Φ_mem -/
  | Fact
  /-- Immutable official knowledge, pure Φ_mem -/
  | Official
  /-- Mutable custom implementation, Φ_ctx ∘ Φ_mem -/
  | Custom
  deriving DecidableEq, Repr

/-- Check if tier is immutable -/
def KnowledgeTier.isImmutable : KnowledgeTier → Bool
  | Fact => true
  | Official => true
  | Custom => false

/-- Knowledge primitive with dual anchoring -/
structure Knowledge where
  id : Nat
  content : String
  tier : KnowledgeTier
  /-- Context anchor (None for Fact/Official, Some for Custom) -/
  anchor : Option Context
  deriving Repr

/-- Knowledge anchoring invariant: Custom requires anchor -/
def Knowledge.validAnchor (k : Knowledge) : Prop :=
  match k.tier with
  | KnowledgeTier.Custom => k.anchor.isSome
  | _ => k.anchor.isNone

/-- Theorem: Fact/Official knowledge has no anchor -/
theorem immutable_knowledge_no_anchor :
  ∀ (k : Knowledge), k.tier.isImmutable → k.validAnchor → k.anchor = none := by
  intro k himmut hvalid
  cases htier : k.tier with
  | Fact =>
    unfold Knowledge.validAnchor at hvalid
    simp only [htier] at hvalid
    cases h : k.anchor with
    | none => rfl
    | some _ => simp only [h, Option.isNone] at hvalid; exact absurd hvalid Bool.false_ne_true
  | Official =>
    unfold Knowledge.validAnchor at hvalid
    simp only [htier] at hvalid
    cases h : k.anchor with
    | none => rfl
    | some _ => simp only [h, Option.isNone] at hvalid; exact absurd hvalid Bool.false_ne_true
  | Custom => simp only [htier, KnowledgeTier.isImmutable] at himmut; exact absurd himmut Bool.false_ne_true

/-!
## Protocol Primitive

Immutable workflow rules anchored to Φ_ctx ∘ Φ_mem.
-/

/-- Protocol: Immutable workflow rules (Φ_ctx ∘ Φ_mem) -/
structure Protocol where
  id : Nat
  name : String
  /-- Workflow this protocol governs -/
  workflow : WorkflowId
  /-- Context anchor (required) -/
  anchor : Context
  deriving Repr

/-!
## Constraint Primitive

Immutable governance rules (O/F classification) anchored to Φ_ctx ∘ Φ_mem.
-/

/-- Constraint: Immutable governance rule (Φ_ctx ∘ Φ_mem) -/
structure Constraint where
  id : Nat
  /-- Behavior being constrained -/
  behavior : Behavior
  /-- Deontic classification: O or F only (T5) -/
  classification : Deontic
  /-- Context anchor (required) -/
  anchor : Context
  deriving Repr

/-- Constraint is valid if classification is binary -/
def Constraint.isValid (c : Constraint) : Prop :=
  c.classification = Deontic.Obligatory ∨ c.classification = Deontic.Forbidden

/-- Theorem: All constraints are valid (follows from T5) -/
theorem constraint_validity :
  ∀ (c : Constraint), c.isValid := by
  intro c
  unfold Constraint.isValid
  cases c.classification with
  | Obligatory => left; rfl
  | Forbidden => right; rfl

/-!
## Execution Primitive

Mutable temporal trajectory: Phase → Trajectory
-/

/-- Phase of execution (from CRERE) -/
inductive Phase
  | Boot      -- Session initialization
  | Orch      -- Orchestration (Σ_ORCH)
  | Plan      -- Planning (Σ_PLAN)
  | Exec      -- Execution (Σ_EXEC)
  | Valid     -- Validation (Σ_VALID)
  | Ops       -- Operations (Σ_OPS)
  deriving DecidableEq, Repr

/-- Execution step within a trajectory -/
structure Step where
  phase : Phase
  timestamp : Timestamp
  description : String
  deriving Repr

/-- Trajectory: Sequence of execution steps -/
abbrev Trajectory := List Step

/-- ExecutionTrace: Mutable Phase → Trajectory mapping
    Named to avoid conflict with Axioms.Execution -/
def ExecutionTrace := Phase → Trajectory

/-- Empty execution trace -/
def emptyExecutionTrace : ExecutionTrace := fun _ => []

/-- Get trajectory for a phase -/
def ExecutionTrace.getTrajectory (e : ExecutionTrace) (p : Phase) : Trajectory := e p

/-- Execution trace is monotonic if trajectories only grow -/
def ExecutionTrace.monotonic (e1 e2 : ExecutionTrace) : Prop :=
  ∀ p, List.IsPrefix (e1 p) (e2 p)

/-!
## Mutability Classification

Formal classification of primitives by mutability.
-/

/-- Mutability classification -/
inductive Mutability
  | Immutable
  | Mutable
  deriving DecidableEq, Repr

/-- Primitive type classification -/
inductive PrimitiveType
  | UserT
  | TimestampT
  | CWDT
  | PurposeT
  | LayerT
  | ExecutionT
  | OutputT
  | KnowledgeT (tier : KnowledgeTier)
  | ProtocolT
  | ConstraintT
  | ToolT
  deriving Repr

/-- Mutability of each primitive type -/
def PrimitiveType.mutability : PrimitiveType → Mutability
  | UserT => Mutability.Immutable
  | TimestampT => Mutability.Immutable
  | CWDT => Mutability.Immutable
  | PurposeT => Mutability.Mutable  -- Purpose.achieved is mutable
  | LayerT => Mutability.Immutable
  | ExecutionT => Mutability.Mutable
  | OutputT => Mutability.Immutable
  | KnowledgeT tier =>
    if tier.isImmutable then Mutability.Immutable else Mutability.Mutable
  | ProtocolT => Mutability.Immutable
  | ConstraintT => Mutability.Immutable
  | ToolT => Mutability.Immutable

/-- Theorem: Most primitives are immutable -/
theorem immutable_dominance :
  ∀ pt : PrimitiveType,
    pt ≠ PrimitiveType.PurposeT →
    pt ≠ PrimitiveType.ExecutionT →
    (∀ t, pt ≠ PrimitiveType.KnowledgeT t ∨ t.isImmutable) →
    pt.mutability = Mutability.Immutable := by
  intro pt hpurp hexec hknow
  cases pt with
  | UserT => rfl
  | TimestampT => rfl
  | CWDT => rfl
  | PurposeT => exact absurd rfl hpurp
  | LayerT => rfl
  | ExecutionT => exact absurd rfl hexec
  | OutputT => rfl
  | KnowledgeT tier =>
    simp [PrimitiveType.mutability]
    cases hknow tier with
    | inl h => exact absurd rfl h
    | inr h => simp [h]
  | ProtocolT => rfl
  | ConstraintT => rfl
  | ToolT => rfl

end SigmaGov.Primitives
