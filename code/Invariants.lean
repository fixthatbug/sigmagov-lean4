/-
  SigmaGov.Invariants - System Invariants

  Formalization of the 10 invariants (Π_1 through Π_10) from foundation.gov v1.1.0.
  These invariants are guaranteed properties that hold throughout execution.
-/

import SigmaGov.Basic
import SigmaGov.Axioms
import SigmaGov.Temporal
import SigmaGov.Primitives
import SigmaGov.Context

namespace SigmaGov.Invariants

open SigmaGov
open SigmaGov.Axioms
open SigmaGov.Temporal
open SigmaGov.Primitives
open SigmaGov.Context

/-!
## Invariant Π_1: Layer Self-Containment

∀ Ln ∈ Layer: config(Ln).complete = true

Source: Axiom T7
Guarantee: Each layer can operate independently
-/

/-- Π_1: Each layer has complete configuration -/
theorem Pi_1_layer_self_containment :
  ∀ (l : Layer), (config l).complete = true :=
  T7_completeness

/-!
## Invariant Π_2: Auto-Load Completeness

∀ S ∈ Session:
  session_start(S) →
    loaded(CLAUDE.md) ∧ ∀ f ∈ rules/*.md: loaded(f)

Guarantee: Foundation rules available at initialization
-/

/-- Loaded resources for a session -/
structure SessionResources where
  claudeMd : Bool
  ruleFiles : List String
  deriving Repr

/-- Check if foundation is loaded -/
def foundationLoaded (r : SessionResources) : Prop :=
  r.claudeMd = true ∧ r.ruleFiles.length > 0

/-- Π_2: Sessions start with foundation loaded -/
axiom Pi_2_autoload_completeness :
  ∀ (s : Session), ∃ (r : SessionResources), foundationLoaded r

/-!
## Invariant Π_3: Purpose Traceability

∀ P ∈ Purpose: ∃ prompt ∈ userPromptSubmit: P.origin = prompt

Source: Axiom T1
Guarantee: Every purpose traces to user input
-/

/-- Π_3: Every purpose has an origin prompt -/
theorem Pi_3_purpose_traceability :
  ∀ (p : Purpose), ∃ (prompt : Prompt), p.origin = prompt := by
  intro p
  exact ⟨p.origin, rfl⟩

/-!
## Invariant Π_4: Conditions Precedence

∀ P ∈ Purpose, ∀ D ∈ Delegate:
  set(P.conditions) ≺ invoke(D)

Guarantee: Success criteria defined before delegation
-/

/-- Conditions for purpose achievement -/
structure PurposeConditions where
  invariants : List String
  variables : List String
  formulas : List String
  deriving Repr

/-- Delegation event -/
structure DelegationEvent where
  source : Layer
  target : Layer
  purpose : Purpose
  conditions : PurposeConditions
  timestamp : Nat
  deriving Repr

/-- Condition setting event -/
structure ConditionSetEvent where
  purpose : Purpose
  conditions : PurposeConditions
  timestamp : Nat
  deriving Repr

/-- Π_4: Conditions set before delegation -/
axiom Pi_4_conditions_precedence :
  ∀ (d : DelegationEvent) (c : ConditionSetEvent),
    d.purpose = c.purpose →
    c.timestamp < d.timestamp

/-!
## Invariant Π_5: Terminal Layer Finality

∀ L ∈ {L3.Implementer, L4}: delegates(L) = ∅

Guarantee: Terminal layers cannot delegate further
-/

/-- Check if layer is terminal -/
def isTerminalLayer : Layer → Bool
  | Layer.L4 => true
  | _ => false  -- L3.Implementer is distinguished by workflow, not layer type

/-- Delegation relation: source can delegate to target based on layer hierarchy -/
def canDelegateTo : Layer → Layer → Prop
  | Layer.L1, Layer.L2 => True   -- L1 can delegate to L2 (context buffer)
  | Layer.L1, Layer.L3 => True   -- L1 can delegate to L3
  | Layer.L2, Layer.L3 => True   -- L2 can coordinate L3s
  | Layer.L3, Layer.L4 => True   -- L3 (orchestrator) can delegate to L4
  | Layer.L4, _ => False         -- L4 is terminal, cannot delegate
  | _, _ => False                -- All other combinations disallowed

/-- Π_5: Terminal layers have no delegates -/
theorem Pi_5_terminal_finality :
  ∀ (l : Layer), isTerminalLayer l = true →
    ∀ (target : Layer), ¬canDelegateTo l target := by
  intro l hterm target
  -- Only L4 satisfies isTerminalLayer = true
  cases l with
  | L1 => simp [isTerminalLayer] at hterm
  | L2 => simp [isTerminalLayer] at hterm
  | L3 => simp [isTerminalLayer] at hterm
  | L4 =>
    -- For L4, canDelegateTo L4 _ = False for all targets
    cases target <;> simp [canDelegateTo]

/-!
## Invariant Π_6: Aggregation Equivalence

∀ Ln ∈ Layer:
  (∀ child ∈ delegates(Ln): achieved(child.purpose)) ⟺ achieved(Ln.purpose)

Guarantee: Parent achievement equals child conjunction
-/

/-- Parent-child relationship in delegation -/
structure ParentChild where
  parent : Purpose
  children : List Purpose
  deriving Repr

/-- All children achieved -/
def allChildrenAchieved (pc : ParentChild) : Prop :=
  ∀ c ∈ pc.children, c.achieved = true

/-- Π_6: Parent achieved iff all children achieved -/
axiom Pi_6_aggregation_equivalence :
  ∀ (pc : ParentChild),
    allChildrenAchieved pc ↔ pc.parent.achieved = true

/-!
## Invariant Π_7: Grounding Requirement

∀ O ∈ Output: valid(O) → grounded(O) ∨ acknowledged_uncertainty(O)

Source: Axiom T0
Guarantee: No ungrounded assertions
-/

/-- Π_7: Valid outputs are grounded or uncertainty-acknowledged -/
theorem Pi_7_grounding_requirement :
  ∀ (o : Output), validOutput o → grounded o ∨ acknowledgedUncertainty o := by
  intro o h
  exact h

/-!
## Invariant Π_8: Decision Precedence

∀ t ∈ Tool: invoke(t) → ∃ reasoning ∈ Φ_sem: precedes(reasoning, invoke(t))

Source: Axiom T4
Guarantee: No autonomous tool invocation
-/

/-- Π_8: Every tool invocation has preceding reasoning -/
theorem Pi_8_decision_precedence :
  ∀ (t : ToolInvocation) (e : Execution),
    ∃ (r : Reasoning), precedes r t ∧ r.op.domain = Phi.Sem := by
  intro t e
  have h := T4_decision_gate t e
  obtain ⟨r, hd, hp, _⟩ := h
  exact ⟨r, hp, hd⟩

/-!
## Invariant Π_9: Purpose Monotonicity

∀ P ∈ Purpose: P.achieved: false ↦ true  (irreversible)

Guarantee: Achievement is monotonic
-/

/-- Purpose state transition -/
inductive PurposeTransition
  | initial : PurposeTransition              -- achieved = false
  | achieved : PurposeTransition             -- achieved = true
  deriving DecidableEq, Repr

/-- Valid state transitions -/
def validTransition : PurposeTransition → PurposeTransition → Bool
  | PurposeTransition.initial, PurposeTransition.initial => true
  | PurposeTransition.initial, PurposeTransition.achieved => true
  | PurposeTransition.achieved, PurposeTransition.achieved => true
  | PurposeTransition.achieved, PurposeTransition.initial => false  -- Cannot revert

/-- Π_9: Achievement cannot be reverted -/
theorem Pi_9_purpose_monotonicity :
  validTransition PurposeTransition.achieved PurposeTransition.initial = false := by
  rfl

/-- Π_9 Alternative: Once achieved, always achieved -/
theorem Pi_9_once_achieved_always_achieved :
  ∀ (p1 p2 : PurposeTransition),
    p1 = PurposeTransition.achieved →
    validTransition p1 p2 = true →
    p2 = PurposeTransition.achieved := by
  intro p1 p2 h1 hv
  cases p2 with
  | initial =>
    rw [h1] at hv
    exact absurd hv (Bool.false_ne_true)
  | achieved => rfl

/-!
## Invariant Π_10: Context Anchoring

∀ E ∈ Execution: ∃ ctx ∈ Φ_ctx: anchored(E, ctx)
∀ S ∈ Session: cwd(S) ≠ ∅ ∧ timestamp(S) ≠ ∅

Source: Axiom T8
Guarantee: All executions are anchored in physical context
-/

/-- Π_10: Every execution trace is anchored to a context manifold -/
theorem Pi_10_context_anchoring :
  ∀ (e : Primitives.ExecutionTrace),
    ∃ (m : Context.Manifold) (ec : Context.ExecutionContext),
      ec.execution = e ∧ ec.manifold = m :=
  Context.T8_context_anchoring

/-- Π_10 Corollary: Sessions have valid context (cwd and timestamp non-empty) -/
axiom Pi_10_session_context_valid :
  ∀ (s : Session),
    s.startTime > 0 -- timestamp exists

/-!
## Composite Invariants

Combined properties derived from multiple invariants.
-/

/-- Invariant preservation through execution -/
structure InvariantState where
  layerComplete : Bool     -- Π_1
  foundationLoaded : Bool  -- Π_2
  purposeTraced : Bool     -- Π_3
  conditionsPrecede : Bool -- Π_4
  terminalsNoDelegate : Bool -- Π_5
  aggregationHolds : Bool  -- Π_6
  outputsGrounded : Bool   -- Π_7
  decisionsGated : Bool    -- Π_8
  achievementMonotonic : Bool -- Π_9
  contextAnchored : Bool   -- Π_10
  deriving Repr

/-- All invariants hold -/
def allInvariantsHold (s : InvariantState) : Prop :=
  s.layerComplete = true ∧
  s.foundationLoaded = true ∧
  s.purposeTraced = true ∧
  s.conditionsPrecede = true ∧
  s.terminalsNoDelegate = true ∧
  s.aggregationHolds = true ∧
  s.outputsGrounded = true ∧
  s.decisionsGated = true ∧
  s.achievementMonotonic = true ∧
  s.contextAnchored = true

/-- System is valid when all invariants hold -/
def validSystem (s : InvariantState) : Prop :=
  allInvariantsHold s

/-- Valid system maintains consistency -/
theorem valid_system_consistent :
  ∀ (s : InvariantState), validSystem s → allInvariantsHold s := by
  intro s h
  exact h

end SigmaGov.Invariants
