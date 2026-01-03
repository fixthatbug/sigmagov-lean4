/-
  SigmaGov.Temporal - Temporal Logic

  Formalization of temporal operators for ΣGov governance.
  Implements the temporal aspects from Axiom T4 (DECISION_GATE)
  and general temporal ordering for the system.

  Operators:
  - Box (□): Always/Necessarily
  - Diamond (◇): Eventually/Possibly
  - Precedes (≺): Strict temporal ordering
  - Until (U): Holds until another condition
-/

import SigmaGov.Basic

namespace SigmaGov.Temporal

open SigmaGov

/-!
## Time Model

We use a discrete linear time model with natural number timestamps.
-/

/-- Time point in the execution trace -/
abbrev Time := Nat

/-- Timespan with start and end -/
structure Timespan where
  start : Time
  finish : Time
  valid : start ≤ finish
  deriving Repr

/-- Event with timestamp -/
structure Event (α : Type) where
  payload : α
  timestamp : Time
  deriving Repr

/-!
## Temporal Relations
-/

/-- Strict precedence: e1 happens before e2 -/
def precedes (e1 : Event α) (e2 : Event β) : Prop :=
  e1.timestamp < e2.timestamp

notation:50 e1 " ≺ " e2 => precedes e1 e2

/-- Immediate precedence: e1 happens directly before e2 with no events between -/
def immediatelyPrecedes (e1 e2 : Event α) (trace : List (Event α)) : Prop :=
  (e1 ≺ e2) ∧
  (∀ e3, e3 ∈ trace → ¬((e1 ≺ e3) ∧ (e3 ≺ e2)))

/-- Concurrent events: timespans overlap -/
def concurrent (t1 t2 : Timespan) : Prop :=
  t1.start ≤ t2.finish ∧ t2.start ≤ t1.finish

/-- Event occurs within timespan -/
def occursWithin (e : Event α) (t : Timespan) : Prop :=
  t.start ≤ e.timestamp ∧ e.timestamp ≤ t.finish

/-!
## Execution Traces

A trace is a sequence of events ordered by time.
-/

/-- Trace of events -/
structure Trace (α : Type) where
  events : List (Event α)
  ordered : ∀ (i j : Nat) (hi : i < events.length) (hj : j < events.length),
    i < j → (events.get ⟨i, hi⟩).timestamp ≤ (events.get ⟨j, hj⟩).timestamp

/-- Empty trace -/
def emptyTrace : Trace α :=
  { events := [], ordered := fun _ _ h _ _ => absurd h (Nat.not_lt_zero _) }

/-- Get event at index in trace -/
def getEvent (tr : Trace α) (i : Fin tr.events.length) : Event α :=
  tr.events.get i

/-!
## Temporal Logic Operators

Modal operators for reasoning about traces.
-/

/-- Predicate on traces -/
abbrev TracePred (α : Type) := Trace α → Prop

/-- State predicate: holds at a specific time point -/
abbrev StatePred (α : Type) := Event α → Prop

/-- Lift state predicate to trace predicate at specific index -/
def atIndex (p : StatePred α) (i : Nat) : TracePred α :=
  fun tr => ∃ (h : i < tr.events.length), p (tr.events.get ⟨i, h⟩)

/-- Box (□): Property holds at all time points -/
def box (p : StatePred α) : TracePred α :=
  fun tr => ∀ e ∈ tr.events, p e

notation:max "□" p => box p

/-- Diamond (◇): Property holds at some time point -/
def diamond (p : StatePred α) : TracePred α :=
  fun tr => ∃ e ∈ tr.events, p e

notation:max "◇" p => diamond p

/-- Until (U): p holds until q becomes true -/
def untilP (p q : StatePred α) : TracePred α :=
  fun tr => ∃ (i : Nat) (hi : i < tr.events.length),
    q (tr.events.get ⟨i, hi⟩) ∧
    ∀ (j : Nat) (hj : j < i), p (tr.events.get ⟨j, Nat.lt_trans hj hi⟩)

notation:40 p " U " q => untilP p q

/-- Next (X): Property holds at the next time point -/
def next (p : StatePred α) (i : Nat) : TracePred α :=
  fun tr => ∃ (h : i + 1 < tr.events.length), p (tr.events.get ⟨i + 1, h⟩)

/-- Eventually within bound: Property holds within n steps -/
def eventuallyWithin (p : StatePred α) (n : Nat) : TracePred α :=
  fun tr => ∃ (i : Nat) (hi : i < min n tr.events.length),
    p (tr.events.get ⟨i, Nat.lt_of_lt_of_le hi (Nat.min_le_right _ _)⟩)

/-!
## Temporal Axioms
-/

/-- Box distributes over conjunction -/
theorem box_and :
  ∀ (p q : StatePred α) (tr : Trace α),
    (□(fun e => p e ∧ q e)) tr ↔ (□p) tr ∧ (□q) tr := by
  intro p q tr
  constructor
  · intro h
    constructor
    · intro e he; exact (h e he).1
    · intro e he; exact (h e he).2
  · intro ⟨hp, hq⟩ e he
    exact ⟨hp e he, hq e he⟩

/-- Diamond distributes over disjunction -/
theorem diamond_or :
  ∀ (p q : StatePred α) (tr : Trace α),
    (◇(fun e => p e ∨ q e)) tr ↔ (◇p) tr ∨ (◇q) tr := by
  intro p q tr
  constructor
  · intro ⟨e, he, hpq⟩
    cases hpq with
    | inl hp => left; exact ⟨e, he, hp⟩
    | inr hq => right; exact ⟨e, he, hq⟩
  · intro h
    cases h with
    | inl hp =>
      obtain ⟨e, he, hp⟩ := hp
      exact ⟨e, he, Or.inl hp⟩
    | inr hq =>
      obtain ⟨e, he, hq⟩ := hq
      exact ⟨e, he, Or.inr hq⟩

/-- Box-Diamond duality: □p ↔ ¬◇¬p -/
theorem box_diamond_dual :
  ∀ (p : StatePred α) (tr : Trace α),
    (□p) tr ↔ ¬(◇(fun e => ¬p e)) tr := by
  intro p tr
  constructor
  · intro hbox ⟨e, he, hnp⟩
    exact hnp (hbox e he)
  · intro hndiamond e he
    match Classical.em (p e) with
    | Or.inl hp => exact hp
    | Or.inr hnp => exact absurd ⟨e, he, hnp⟩ hndiamond

/-- Diamond-Box duality: ◇p ↔ ¬□¬p -/
theorem diamond_box_dual :
  ∀ (p : StatePred α) (tr : Trace α),
    (◇p) tr ↔ ¬(□(fun e => ¬p e)) tr := by
  intro p tr
  constructor
  · intro ⟨e, he, hp⟩ hbox
    exact hbox e he hp
  · intro hnbox
    match Classical.em ((◇p) tr) with
    | Or.inl hdiamond => exact hdiamond
    | Or.inr hndiamond =>
      exfalso
      apply hnbox
      intro e he hp
      exact hndiamond ⟨e, he, hp⟩

/-- Box implies Diamond (on non-empty traces) -/
theorem box_implies_diamond :
  ∀ (p : StatePred α) (tr : Trace α),
    tr.events ≠ [] → (□p) tr → (◇p) tr := by
  intro p tr hne hbox
  match hevents : tr.events with
  | [] => exact absurd hevents hne
  | e :: es =>
    have hmem : e ∈ tr.events := by rw [hevents]; exact List.Mem.head es
    exact ⟨e, hmem, hbox e hmem⟩

/-!
## Ordering Properties
-/

/-- Precedence is transitive -/
theorem precedes_trans :
  ∀ (e1 : Event α) (e2 : Event β) (e3 : Event γ),
    (e1 ≺ e2) → (e2 ≺ e3) → (e1 ≺ e3) := by
  intro e1 e2 e3 h12 h23
  exact Nat.lt_trans h12 h23

/-- Precedence is irreflexive -/
theorem precedes_irrefl :
  ∀ (e : Event α), ¬(e ≺ e) := by
  intro e h
  exact Nat.lt_irrefl e.timestamp h

/-- Precedence is asymmetric -/
theorem precedes_asymm :
  ∀ (e1 e2 : Event α), (e1 ≺ e2) → ¬(e2 ≺ e1) := by
  intro e1 e2 h12 h21
  exact Nat.lt_asymm h12 h21

/-- Linear ordering: any two distinct events are ordered -/
theorem precedes_linear :
  ∀ (e1 e2 : Event α),
    e1.timestamp ≠ e2.timestamp →
    (e1 ≺ e2) ∨ (e2 ≺ e1) := by
  intro e1 e2 hne
  cases Nat.lt_trichotomy e1.timestamp e2.timestamp with
  | inl h => left; exact h
  | inr h =>
    cases h with
    | inl heq => exact absurd heq hne
    | inr hgt => right; exact hgt

/-!
## Decision Gate Temporal Formalization

From Axiom T4: Tool invocation requires precedent reasoning.
O(Sequential_Thinking) before tool_invocation
-/

/-- Tagged event for distinguishing reasoning from tool invocations -/
inductive TaggedEvent
  | reasoning (content : String)
  | toolInvoke (toolName : String)
  deriving Repr

/-- Check if event is a reasoning event -/
def isReasoning : TaggedEvent → Bool
  | TaggedEvent.reasoning _ => true
  | TaggedEvent.toolInvoke _ => false

/-- Check if event is a tool invocation -/
def isToolInvoke : TaggedEvent → Bool
  | TaggedEvent.reasoning _ => false
  | TaggedEvent.toolInvoke _ => true

/-- Decision gate property: every tool invocation has a preceding reasoning event -/
def decisionGateProperty : TracePred TaggedEvent :=
  fun tr => ∀ (i : Nat) (hi : i < tr.events.length),
    let event := tr.events.get ⟨i, hi⟩
    isToolInvoke event.payload →
    ∃ (j : Nat) (hj : j < i), isReasoning ((tr.events.get ⟨j, Nat.lt_trans hj hi⟩).payload)

/-- Strictly ordered trace: distinct indices have distinct timestamps -/
def strictlyOrdered (tr : Trace α) : Prop :=
  ∀ (i j : Nat) (hi : i < tr.events.length) (hj : j < tr.events.length),
    i < j →
    (tr.events.get ⟨i, hi⟩).timestamp < (tr.events.get ⟨j, hj⟩).timestamp

/-- All tool invocations in a valid trace have reasoning predecessors.

    Proof strategy:
    1. Convert List membership to index-based access via List.mem_iff_get
    2. Apply decisionGateProperty to get reasoning event at index j < i
    3. Convert back to membership and prove precedence

    Note: The precedence (r ≺ e) requires strict timestamp ordering.
    The Trace.ordered property gives ≤, so we add strictlyOrdered hypothesis.
    For traces where distinct events have distinct timestamps, this always holds. -/
theorem decision_gate_enforced :
  ∀ (tr : Trace TaggedEvent),
    decisionGateProperty tr →
    strictlyOrdered tr →
    ∀ e, e ∈ tr.events → isToolInvoke e.payload →
      ∃ r, r ∈ tr.events ∧ isReasoning r.payload ∧ (r ≺ e) := by
  intro tr hdg hstrict e he htool
  -- Step 1: Convert membership to index-based access
  rw [List.mem_iff_get] at he
  obtain ⟨⟨i, hi⟩, hget⟩ := he
  -- Step 2: Show the event at index i is a tool invocation
  have htool_i : isToolInvoke (tr.events.get ⟨i, hi⟩).payload := by
    rw [hget]; exact htool
  -- Step 3: Apply decisionGateProperty to get reasoning event index j < i
  obtain ⟨j, hji, hreason_j⟩ := hdg i hi htool_i
  -- Step 4: Construct the witness - the reasoning event at index j
  have hj_lt : j < tr.events.length := Nat.lt_trans hji hi
  -- The reasoning event
  let r := tr.events.get ⟨j, hj_lt⟩
  -- Step 5: Prove r satisfies all requirements
  refine ⟨r, ?mem, ?reas, ?prec⟩
  case mem =>
    exact List.get_mem tr.events ⟨j, hj_lt⟩
  case reas =>
    exact hreason_j
  case prec =>
    unfold precedes
    have h := hstrict j i hj_lt hi hji
    rw [hget] at h
    exact h

/-- Decision gate for traces with only weak ordering (≤).
    Returns the weaker conclusion that r.timestamp ≤ e.timestamp. -/
theorem decision_gate_weak :
  ∀ (tr : Trace TaggedEvent),
    decisionGateProperty tr →
    ∀ e, e ∈ tr.events → isToolInvoke e.payload →
      ∃ r, r ∈ tr.events ∧ isReasoning r.payload ∧ r.timestamp ≤ e.timestamp := by
  intro tr hdg e he htool
  rw [List.mem_iff_get] at he
  obtain ⟨⟨i, hi⟩, hget⟩ := he
  have htool_i : isToolInvoke (tr.events.get ⟨i, hi⟩).payload := by
    rw [hget]; exact htool
  obtain ⟨j, hji, hreason_j⟩ := hdg i hi htool_i
  have hj_lt : j < tr.events.length := Nat.lt_trans hji hi
  let r := tr.events.get ⟨j, hj_lt⟩
  refine ⟨r, List.get_mem tr.events ⟨j, hj_lt⟩, hreason_j, ?ord⟩
  case ord =>
    have h := tr.ordered j i hj_lt hi hji
    rw [hget] at h
    exact h

/-!
## Temporal Invariants
-/

/-- Purpose monotonicity: once achieved, always achieved -/
def purposeMonotonicity (achievedAt : Event Purpose → Bool) : TracePred Purpose :=
  fun tr => ∀ (i j : Nat) (hi : i < tr.events.length) (hj : j < tr.events.length),
    i < j →
    achievedAt (tr.events.get ⟨i, hi⟩) →
    achievedAt (tr.events.get ⟨j, hj⟩)

/-- Session boundaries: operations within session timespan -/
def withinSession (session : Timespan) : TracePred α :=
  fun tr => ∀ e ∈ tr.events, occursWithin e session

/-- Temporal consistency: trace respects causal ordering -/
def temporalConsistency : TracePred α :=
  fun tr => ∀ (i j : Nat) (hi : i < tr.events.length) (hj : j < tr.events.length),
    (tr.events.get ⟨i, hi⟩).timestamp < (tr.events.get ⟨j, hj⟩).timestamp →
    i < j

end SigmaGov.Temporal
