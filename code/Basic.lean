/-
  SigmaGov.Basic - Core Types and Structures

  Formalization of the ΣGov specification language primitives.
  Source: foundation.gov v1.1.0

  This module defines:
  - Agent and Layer types
  - Purpose and Goal structures
  - Deontic modalities (O, F)
  - Five Pillars (Φ_sem, Φ_syn, Φ_auto, Φ_mem, Φ_ctx)
  - Operation classifications per pillar
-/

namespace SigmaGov

/-- User: Human agent initiating interaction -/
structure User where
  id : Nat
  deriving DecidableEq, Repr

/-- Session: Temporal boundary of execution -/
structure Session where
  id : Nat
  user : User
  startTime : Nat := 0
  deriving Repr

/-- Layer: Execution strata {L1, L2, L3, L4} -/
inductive Layer
  | L1  -- User-facing LLM
  | L2  -- SDK Orchestrator - Context Buffer
  | L3  -- SDK Orchestrator/Implementer
  | L4  -- SDK Subagent
  deriving DecidableEq, Repr

/-- Power level associated with each layer -/
def Layer.powerLevel : Layer → Nat
  | L1 => 4
  | L2 => 3
  | L3 => 2
  | L4 => 1

/-- Layer ordering by power level -/
instance : LE Layer where
  le a b := a.powerLevel ≥ b.powerLevel

/-- Prompt: User input string -/
abbrev Prompt := String

/-- Purpose: Goal state derived from user intent -/
structure Purpose where
  /-- Unique identifier -/
  id : Nat
  /-- Owner of this purpose (requestor) -/
  owner : User
  /-- Originating prompt (traceability) -/
  origin : Prompt
  /-- Binary achievement state -/
  achieved : Bool := false
  deriving Repr

/-- Goal: Decomposed sub-purpose -/
structure Goal where
  id : Nat
  parentPurpose : Purpose
  description : String
  achieved : Bool := false
  deriving Repr

/-- Output: System-produced artifact -/
structure Output where
  id : Nat
  content : String
  session : Session
  deriving Repr

/-- Behavior: Executable action or operation -/
structure Behavior where
  id : Nat
  description : String
  deriving Repr

/-- Tool: Executable capability -/
structure Tool where
  id : Nat
  name : String
  deriving Repr

/-!
## Deontic Modalities

Binary governance: All behaviors are either Obligatory or Forbidden.
Permissibility is excluded to enforce determinism (Axiom T5).
-/

/-- Deontic classification of behaviors -/
inductive Deontic
  /-- O(φ): Action φ must occur -/
  | Obligatory
  /-- F(φ): Action φ must not occur -/
  | Forbidden
  deriving DecidableEq, Repr

/-- Deontic operator O: marks behavior as obligatory -/
def O : Behavior → Deontic := fun _ => Deontic.Obligatory

/-- Deontic operator F: marks behavior as forbidden -/
def F : Behavior → Deontic := fun _ => Deontic.Forbidden

/-- Governance function: classifies all behaviors
    Axiom: The specific classification rules are defined externally.
    This is intentionally axiomatic - governance rules come from foundation.gov -/
axiom governance : Behavior → Deontic

/-- Property: F(φ) ↔ O(¬φ) - Forbidden equals obligatory negation
    This is a semantic property of deontic logic.
    Axiom: By definition in binary governance, forbidden behaviors
    imply the existence of alternative obligatory behaviors. -/
axiom forbidden_equals_obligatory_negation :
  ∀ b : Behavior, (governance b = Deontic.Forbidden) ↔
    ∃ b' : Behavior, governance b' = Deontic.Obligatory ∧ b ≠ b'

/-- Property: ¬(O(φ) ∧ F(φ)) - No contradictions -/
theorem no_deontic_contradictions :
  ∀ b : Behavior, ¬(governance b = Deontic.Obligatory ∧
                    governance b = Deontic.Forbidden) := by
  intro b h
  cases h with
  | intro ho hf =>
    rw [ho] at hf
    exact Deontic.noConfusion hf

/-!
## Five Pillars: System Decomposition

Σ_System = Φ_sem ⊕ Φ_syn ⊕ Φ_auto ⊕ Φ_mem ⊕ Φ_ctx (Axiom T3)

The system partitions into five mutually exclusive, collectively exhaustive
operational domains:
- Φ_sem: Semantic (interpretation, meaning, indeterminacy)
- Φ_syn: Syntactic (formal rules, grammar, determinacy)
- Φ_auto: Autonomous (agency, driver, execution)
- Φ_mem: Memory (state, history, persistence)
- Φ_ctx: Context (spacetime, location, timestamp anchoring)
-/

/-- Operational domain classification (Five-Pillar System T3) -/
inductive Phi
  /-- Φ_sem: Semantic reasoning (deliberative thought, planning, indeterminacy) -/
  | Sem
  /-- Φ_syn: Syntactic structure (formal rules, grammar, determinacy) -/
  | Syn
  /-- Φ_auto: Autonomous execution (tool invocation, action, agency) -/
  | Auto
  /-- Φ_mem: Memory persistence (state continuity, recall, history) -/
  | Mem
  /-- Φ_ctx: Contextual anchoring (spacetime, location, timestamp) -/
  | Ctx
  deriving DecidableEq, Repr

/-- Operation type within the system -/
structure Operation where
  id : Nat
  domain : Phi
  description : String
  deriving Repr

/-- Semantic operations (Φ_sem) -/
inductive SemanticOp
  | Reasoning
  | Decomposition
  | Planning
  | Validation
  | Decision
  | Synthesis
  | Interpretation
  deriving DecidableEq, Repr

/-- Syntactic operations (Φ_syn) -/
inductive SyntacticOp
  | Parsing
  | Formatting
  | TypeChecking
  | GrammarValidation
  | StructuralTransform
  | Serialization
  deriving DecidableEq, Repr

/-- Autonomous operations (Φ_auto) -/
inductive AutoOp
  | ToolInvocation
  | Execution
  | Delegation
  | Coordination
  | Transformation
  | Agency
  deriving DecidableEq, Repr

/-- Memory operations (Φ_mem) -/
inductive MemOp
  | Persistence
  | Retrieval
  | Recall
  | Indexing
  | Search
  | History
  deriving DecidableEq, Repr

/-- Context operations (Φ_ctx) -/
inductive ContextOp
  | Anchoring
  | LocationBinding
  | TimestampBinding
  | ScopeResolution
  | BoundaryEnforcement
  deriving DecidableEq, Repr

/-- Classify an operation into its domain -/
def classifyOperation : Operation → Phi := fun op => op.domain

/-!
## Grounding and Epistemic Predicates

From Axiom T0 (TRUTHFULNESS): Outputs must be grounded or acknowledge uncertainty.
-/

/-- Evidence supporting an output -/
inductive Evidence
  | SourceMaterial (reference : String) (excerpt : String)
  | VerifiedFact (fact : String) (source : String)
  | LogicalDerivation (premises : List Evidence) (rule : String)
  | Observation (context : String) (timestamp : Nat)
  | Artifact (artifactType : String) (content : String)
  deriving Repr

/-- Evidence supports an output if the evidence content relates to output content.
    Axiom: The semantic relationship between evidence and output is domain-specific. -/
axiom Evidence.supportsOutput : Evidence → Output → Prop

/-- Predicate: Output is grounded (has verifiable evidence) -/
def grounded (o : Output) : Prop :=
  ∃ e : Evidence, Evidence.supportsOutput e o

/-- Predicate: Output acknowledges uncertainty -/
def acknowledgedUncertainty (o : Output) : Prop :=
  o.content.contains "uncertain" ∨
  o.content.contains "speculative" ∨
  o.content.contains "hypothetical"

/-- Predicate: Output is fabricated (ungrounded assertion) -/
def fabricated (o : Output) : Prop :=
  ¬grounded o ∧ ¬acknowledgedUncertainty o

/-- Output claims a fact if it uses assertive language without uncertainty markers -/
def Output.claimsFact (o : Output) : Prop :=
  ¬(o.content.contains "uncertain" ∨
    o.content.contains "possibly" ∨
    o.content.contains "might" ∨
    o.content.contains "hypothetical")

/-- Output is verifiable if it has source references or can be checked -/
def Output.verifiable (o : Output) : Prop :=
  o.content.contains "source:" ∨
  o.content.contains "reference:" ∨
  o.content.contains "verified:" ∨
  grounded o

/-- Predicate: Output is hallucinated (false assertion as fact) -/
def hallucinated (o : Output) : Prop :=
  Output.claimsFact o ∧ ¬Output.verifiable o

/-- Valid output: grounded or uncertainty acknowledged -/
def validOutput (o : Output) : Prop :=
  grounded o ∨ acknowledgedUncertainty o

/-!
## Achievement Dimensions (T6)

Achievement(S, P) ≡ WHAT(S, P) ∧ WHERE(S, P) ∧ HOW(S, P) ∧ WHY(S, P)
-/

/-- Achievement dimension predicate -/
structure AchievementDimension (S : Session) (P : Purpose) where
  what : Prop   -- Functional requirements satisfied
  where_ : Prop -- Context/location validated
  how : Prop    -- Method constraints met
  why : Prop    -- Intent/purpose fulfilled

/-- Full achievement: conjunction of all dimensions -/
def achievement (S : Session) (P : Purpose) (D : AchievementDimension S P) : Prop :=
  D.what ∧ D.where_ ∧ D.how ∧ D.why

/-!
## Layer Configuration

From Axiom T7: Each layer has complete, self-sufficient configuration.
-/

/-- Layer configuration -/
structure LayerConfig where
  layer : Layer
  complete : Bool
  independent : Bool
  deriving Repr

/-- Configuration is valid if complete and independent -/
def validConfig (c : LayerConfig) : Prop :=
  c.complete = true ∧ c.independent = true

end SigmaGov
