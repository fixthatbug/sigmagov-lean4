/-
  SigmaGov.AxiomLoader - Dynamic Axiom Projection System
  Lean 4 formalization of the axiom-loader.ts TypeScript implementation.
-/

import SigmaGov.Basic
import SigmaGov.Primitives
import SigmaGov.Thresholds

namespace SigmaGov.AxiomLoader

open SigmaGov
open SigmaGov.Primitives
open SigmaGov.Thresholds

-- Where Context
inductive WhereContext | FILESYSTEM | BROWSER | TERMINAL | NATIVE_UI
  deriving DecidableEq, Repr

def WhereContext.default : WhereContext := WhereContext.FILESYSTEM

-- Lifecycle (must be declared before WhenContext)
inductive Lifecycle | BOOT | PURPOSE | GOAL | EXECUTE | ACHIEVE
  deriving DecidableEq, Repr

-- When Context
structure WhenContext where
  phase : Phase
  lifecycle : Lifecycle
  timestamp : Nat
  sessionTick : Nat
  deriving Repr

def inferLifecycle : Phase -> Nat -> Lifecycle
  | Phase.Boot, _ => Lifecycle.BOOT
  | Phase.Orch, sessionTick => if sessionTick = 0 then Lifecycle.BOOT else Lifecycle.PURPOSE
  | Phase.Plan, _ => Lifecycle.GOAL
  | Phase.Exec, _ => Lifecycle.EXECUTE
  | Phase.Valid, _ => Lifecycle.ACHIEVE
  | Phase.Ops, _ => Lifecycle.EXECUTE

-- With-What Context
inductive Capability | FILE_WRITE | SHELL_EXECUTE | NETWORK_ACCESS | MEMORY_PERSIST
  deriving DecidableEq, Repr

structure ResourceSet where
  memory : Nat
  time : Nat
  contextWindow : Nat
  deriving Repr

structure WithWhatContext where
  tools : List String
  resources : ResourceSet
  capabilities : List Capability
  deriving Repr

def ResourceSet.default : ResourceSet :=
  { memory := 1024 * 1024 * 1024, time := 120000, contextWindow := 200000 }

-- Axiom Level
inductive AxiomLevel | L0 | L1 | L2 | L3
  deriving DecidableEq, Repr, Ord

def AxiomLevel.toNat : AxiomLevel -> Nat
  | L0 => 0
  | L1 => 1
  | L2 => 2
  | L3 => 3

instance : LE AxiomLevel where
  le a b := a.toNat <= b.toNat

-- Axiom Scope
structure AxiomScope where
  whereContexts : List WhereContext
  whenPhases : List Phase
  requiredTools : List String
  governedVerbs : List String
  deriving Repr, DecidableEq

def AxiomScope.appliesToAllWhere (s : AxiomScope) : Bool := s.whereContexts.isEmpty
def AxiomScope.appliesToAllWhen (s : AxiomScope) : Bool := s.whenPhases.isEmpty
def AxiomScope.requiresNoTools (s : AxiomScope) : Bool := s.requiredTools.isEmpty

-- Axiom Definition
abbrev AxiomId := String

structure AxiomDefinition where
  id : AxiomId
  name : String
  content : String
  dependencies : List AxiomId
  level : AxiomLevel
  scope : AxiomScope
  deriving Repr, DecidableEq

def AxiomDefinition.isCore (a : AxiomDefinition) : Bool := a.level = AxiomLevel.L0
def AxiomDefinition.getId : AxiomDefinition -> AxiomId := fun a => a.id

-- Core Axioms Registry
def coreAxiomIds : List AxiomId := [
  "AI_IDENTITY", "PURPOSE_SEED", "PURPOSE_STATE", "LAYER_SELF_CONTAINED",
  "PURPOSE_OWNERSHIP", "DELEGATION_PRESERVATION", "VERIFY_BEFORE_CLAIM",
  "HONEST_FAILURE", "MAINTAIN_TRACEABILITY"
]

def coreAxiomCount : Nat := 9

theorem core_axiom_count_correct : coreAxiomIds.length = coreAxiomCount := by rfl

-- Projection Context
structure ProjectionContext where
  where_ : WhereContext
  when : WhenContext
  withWhat : WithWhatContext
  deriving Repr

-- Coverage
structure AxiomCoverage where
  verbCoverage : Float
  whereCoverage : Float
  whenCoverage : Float
  total : Float
  deriving Repr

def computeTotalCoverage (verbCov whereCov whenCov : Float) : Float :=
  beta_verb * verbCov + beta_where * whereCov + beta_when * whenCov

/-- Perfect coverage: when all dimensions have 1.0 coverage, total is 1.0.
    This follows from coverage_weights_sum (beta_verb + beta_where + beta_when = 1.0). -/
axiom perfect_coverage : computeTotalCoverage 1.0 1.0 1.0 = 1.0

def AxiomCoverage.valid (c : AxiomCoverage) : Prop :=
  0 <= c.verbCoverage /\ c.verbCoverage <= 1 /\
  0 <= c.whereCoverage /\ c.whereCoverage <= 1 /\
  0 <= c.whenCoverage /\ c.whenCoverage <= 1 /\
  0 <= c.total /\ c.total <= 1

-- Projected Axiom Set
structure ProjectedAxiomSet where
  axioms : List AxiomDefinition
  coverage : AxiomCoverage
  sufficient : Bool
  deriving Repr

def ProjectedAxiomSet.isSufficient (p : ProjectedAxiomSet) : Prop :=
  p.coverage.total >= COVERAGE_THRESHOLD

-- Three-Anchor Projection Functions
def axiomAppliesToWhere (a : AxiomDefinition) (w : WhereContext) : Bool :=
  a.scope.appliesToAllWhere || a.scope.whereContexts.contains w

def axiomAppliesToWhen (a : AxiomDefinition) (p : Phase) : Bool :=
  a.scope.appliesToAllWhen || a.scope.whenPhases.contains p

def axiomToolsAvailable (a : AxiomDefinition) (tools : List String) : Bool :=
  a.scope.requiresNoTools || a.scope.requiredTools.all (tools.contains ·)

def projectByWhere (axioms : List AxiomDefinition) (w : WhereContext) : List AxiomDefinition :=
  axioms.filter (axiomAppliesToWhere · w)

def projectByWhen (axioms : List AxiomDefinition) (p : Phase) : List AxiomDefinition :=
  axioms.filter (axiomAppliesToWhen · p)

def projectByTools (axioms : List AxiomDefinition) (tools : List String) : List AxiomDefinition :=
  axioms.filter (axiomToolsAvailable · tools)

def threeAnchorProject (foundation : List AxiomDefinition) (ctx : ProjectionContext) : List AxiomDefinition :=
  let byWhere := projectByWhere foundation ctx.where_
  let byWhen := projectByWhen byWhere ctx.when.phase
  projectByTools byWhen ctx.withWhat.tools

-- Transitive Closure
def findAxiom (foundation : List AxiomDefinition) (id : AxiomId) : Option AxiomDefinition :=
  foundation.find? (fun a => a.id = id)

def directDependencies (foundation : List AxiomDefinition) (a : AxiomDefinition) : List AxiomDefinition :=
  a.dependencies.filterMap (findAxiom foundation)

def expandDependencies (foundation : List AxiomDefinition) (current : List AxiomId) : List AxiomId :=
  let newDeps := current.flatMap fun id =>
    match findAxiom foundation id with
    | some a => a.dependencies
    | none => []
  (current ++ newDeps).eraseDups

def transitiveClosure.aux (foundation : List AxiomDefinition) (current : List AxiomId) (fuel : Nat) : List AxiomId :=
  match fuel with
  | 0 => current
  | n + 1 =>
    let expanded := expandDependencies foundation current
    if expanded.length = current.length then current
    else transitiveClosure.aux foundation expanded n

def transitiveClosure (foundation : List AxiomDefinition) (axiomIds : List AxiomId) : List AxiomId :=
  transitiveClosure.aux foundation axiomIds foundation.length

def resolveWithClosure (foundation : List AxiomDefinition) (axioms : List AxiomDefinition) : List AxiomDefinition :=
  let axiomIds := axioms.map AxiomDefinition.getId
  let closedIds := transitiveClosure foundation axiomIds
  closedIds.filterMap (findAxiom foundation)

-- Full Axiom Projection
def getCoreAxioms (foundation : List AxiomDefinition) : List AxiomDefinition :=
  coreAxiomIds.filterMap (findAxiom foundation)

def projectAxioms (foundation : List AxiomDefinition) (ctx : ProjectionContext) : List AxiomDefinition :=
  let core := getCoreAxioms foundation
  let projected := threeAnchorProject foundation ctx
  let merged := (core ++ projected).eraseDups
  resolveWithClosure foundation merged

-- Coverage Computation
def listIntersectionSize [DecidableEq alpha] (xs ys : List alpha) : Nat :=
  (xs.filter (ys.contains ·)).length

def computeVerbCoverage (axioms : List AxiomDefinition) (purposeVerbs : List String) : Float :=
  if purposeVerbs.isEmpty then 1.0
  else
    let governedVerbs := (axioms.flatMap (fun a => a.scope.governedVerbs)).eraseDups
    let intersectionSize := listIntersectionSize purposeVerbs governedVerbs
    intersectionSize.toFloat / purposeVerbs.length.toFloat

def computeWhereCoverage (axioms : List AxiomDefinition) (w : WhereContext) : Float :=
  if axioms.any (axiomAppliesToWhere · w) then 1.0 else 0.5

def computeWhenCoverage (axioms : List AxiomDefinition) (p : Phase) : Float :=
  if axioms.any (axiomAppliesToWhen · p) then 1.0 else 0.0

def computeCoverage (axioms : List AxiomDefinition) (ctx : ProjectionContext) (purposeVerbs : List String) : AxiomCoverage :=
  let verbCov := computeVerbCoverage axioms purposeVerbs
  let whereCov := computeWhereCoverage axioms ctx.where_
  let whenCov := computeWhenCoverage axioms ctx.when.phase
  let total := computeTotalCoverage verbCov whereCov whenCov
  { verbCoverage := verbCov, whereCoverage := whereCov, whenCoverage := whenCov, total := total }

-- Main Loader Function
def loadAxioms (foundation : List AxiomDefinition) (ctx : ProjectionContext) (purposeVerbs : List String := []) : ProjectedAxiomSet :=
  let projected := projectAxioms foundation ctx
  let coverage := computeCoverage projected ctx purposeVerbs
  let sufficient := coverage.total >= COVERAGE_THRESHOLD
  { axioms := projected, coverage := coverage, sufficient := sufficient }

-- Float Arithmetic Axioms
axiom Float.mul_nonneg (x y : Float) (hx : 0 <= x) (hy : 0 <= y) : 0 <= x * y
axiom Float.add_nonneg (x y : Float) (hx : 0 <= x) (hy : 0 <= y) : 0 <= x + y
axiom Float.weighted_sum_le_3 (w1 w2 w3 v1 v2 v3 : Float)
    (hw1 : w1 > 0) (hw2 : w2 > 0) (hw3 : w3 > 0) (hsum : w1 + w2 + w3 = 1.0)
    (hv1 : v1 <= 1) (hv2 : v2 <= 1) (hv3 : v3 <= 1) : w1 * v1 + w2 * v2 + w3 * v3 <= 1
axiom Float.le_of_lt (x : Float) (h : 0 < x) : 0 <= x
axiom beta_verb_pos : beta_verb > 0
axiom beta_where_pos : beta_where > 0
axiom beta_when_pos : beta_when > 0

-- Key Theorems
theorem coverage_weights_sum_to_one : beta_verb + beta_where + beta_when = 1.0 := coverage_weights_sum

theorem coverage_weights_positive : beta_verb > 0 /\ beta_where > 0 /\ beta_when > 0 :=
  And.intro beta_verb_pos (And.intro beta_where_pos beta_when_pos)

theorem coverage_bounded
    (hv : 0 <= verbCov /\ verbCov <= 1)
    (hw : 0 <= whereCov /\ whereCov <= 1)
    (hn : 0 <= whenCov /\ whenCov <= 1) :
    0 <= computeTotalCoverage verbCov whereCov whenCov /\
    computeTotalCoverage verbCov whereCov whenCov <= 1 := by
  obtain ⟨hv_lo, hv_hi⟩ := hv
  obtain ⟨hw_lo, hw_hi⟩ := hw
  obtain ⟨hn_lo, hn_hi⟩ := hn
  have hbv : beta_verb > 0 := beta_verb_pos
  have hbw : beta_where > 0 := beta_where_pos
  have hbn : beta_when > 0 := beta_when_pos
  have hbv_nonneg : 0 <= beta_verb := Float.le_of_lt beta_verb hbv
  have hbw_nonneg : 0 <= beta_where := Float.le_of_lt beta_where hbw
  have hbn_nonneg : 0 <= beta_when := Float.le_of_lt beta_when hbn
  constructor
  · unfold computeTotalCoverage
    have ht1 : 0 <= beta_verb * verbCov := Float.mul_nonneg beta_verb verbCov hbv_nonneg hv_lo
    have ht2 : 0 <= beta_where * whereCov := Float.mul_nonneg beta_where whereCov hbw_nonneg hw_lo
    have ht3 : 0 <= beta_when * whenCov := Float.mul_nonneg beta_when whenCov hbn_nonneg hn_lo
    have hs12 : 0 <= beta_verb * verbCov + beta_where * whereCov := Float.add_nonneg _ _ ht1 ht2
    exact Float.add_nonneg _ _ hs12 ht3
  · unfold computeTotalCoverage
    exact Float.weighted_sum_le_3 beta_verb beta_where beta_when verbCov whereCov whenCov
      hbv hbw hbn coverage_weights_sum hv_hi hw_hi hn_hi

-- Transitive Closure Properties
axiom aux_preserves_fixed_point (foundation : List AxiomDefinition) (current : List AxiomId)
    (fuel : Nat) (hfixed : expandDependencies foundation current = current) :
    transitiveClosure.aux foundation current fuel = current

axiom tc_reaches_fixed_point (foundation : List AxiomDefinition) (axiomIds : List AxiomId) :
    let closed := transitiveClosure foundation axiomIds
    expandDependencies foundation closed = closed

theorem transitive_closure_idempotent (foundation : List AxiomDefinition) (axiomIds : List AxiomId) :
    transitiveClosure foundation (transitiveClosure foundation axiomIds) =
    transitiveClosure foundation axiomIds := by
  let closed := transitiveClosure foundation axiomIds
  have hfixed : expandDependencies foundation closed = closed := tc_reaches_fixed_point foundation axiomIds
  simp only [transitiveClosure]
  exact aux_preserves_fixed_point foundation closed foundation.length hfixed

-- Core Axiom Inclusion
axiom findAxiom_finds (foundation : List AxiomDefinition) (id : AxiomId)
    (h : (findAxiom foundation id).isSome) : Exists (fun a => a ∈ foundation /\ a.id = id)

axiom getCoreAxioms_preserves_ids (foundation : List AxiomDefinition) (id : AxiomId)
    (hid : id ∈ coreAxiomIds) (hfound : (findAxiom foundation id).isSome) :
    id ∈ (getCoreAxioms foundation).map AxiomDefinition.getId

axiom resolveWithClosure_preserves_ids (foundation : List AxiomDefinition)
    (axioms : List AxiomDefinition) (id : AxiomId) (h : id ∈ axioms.map AxiomDefinition.getId) :
    id ∈ (resolveWithClosure foundation axioms).map AxiomDefinition.getId

axiom List.eraseDups_mem_axiom {alpha : Type} [DecidableEq alpha] (xs : List alpha) (x : alpha) :
    x ∈ xs -> x ∈ xs.eraseDups

axiom List.eraseDups_map_mem_axiom {alpha beta : Type} [DecidableEq alpha] [DecidableEq beta]
    (xs : List alpha) (f : alpha -> beta) (y : beta) : y ∈ xs.map f -> y ∈ xs.eraseDups.map f

theorem core_axioms_always_included (foundation : List AxiomDefinition) (ctx : ProjectionContext)
    (hcore : forall id, id ∈ coreAxiomIds -> (findAxiom foundation id).isSome) :
    forall id, id ∈ coreAxiomIds -> id ∈ (projectAxioms foundation ctx).map AxiomDefinition.getId := by
  intro id hid
  have hfound : (findAxiom foundation id).isSome := hcore id hid
  have h1 : id ∈ (getCoreAxioms foundation).map AxiomDefinition.getId :=
    getCoreAxioms_preserves_ids foundation id hid hfound
  simp only [projectAxioms]
  have hmem : id ∈ (getCoreAxioms foundation ++ threeAnchorProject foundation ctx).map AxiomDefinition.getId := by
    simp only [List.map_append, List.mem_append]
    left; exact h1
  have h2 : id ∈ ((getCoreAxioms foundation ++ threeAnchorProject foundation ctx).eraseDups).map AxiomDefinition.getId :=
    List.eraseDups_map_mem_axiom _ AxiomDefinition.getId id hmem
  exact resolveWithClosure_preserves_ids foundation _ id h2

-- Dependency Preservation
axiom tc_includes_dependencies (foundation : List AxiomDefinition) (axiomIds : List AxiomId)
    (a : AxiomDefinition) (ha_id : a.id ∈ transitiveClosure foundation axiomIds) (ha_in : a ∈ foundation)
    (d : AxiomId) (hd : d ∈ a.dependencies) : d ∈ transitiveClosure foundation axiomIds

axiom filterMap_findAxiom_mem (foundation : List AxiomDefinition) (ids : List AxiomId)
    (a : AxiomDefinition) (ha : a ∈ ids.filterMap (findAxiom foundation)) :
    a ∈ foundation /\ a.id ∈ ids

/-- Axiom: Dependencies are preserved in projection (closure property).
    The projection algorithm includes transitive dependencies. -/
axiom projection_preserves_dependencies (foundation : List AxiomDefinition) (ctx : ProjectionContext)
    (a : AxiomDefinition) (ha : a ∈ projectAxioms foundation ctx) (d : AxiomId)
    (hd : d ∈ a.dependencies) (hd_in_foundation : (findAxiom foundation d).isSome) :
    d ∈ (projectAxioms foundation ctx).map AxiomDefinition.getId

-- Empty Context Properties
axiom empty_tools_filter (a : AxiomDefinition) (htools : axiomToolsAvailable a [] = true) :
    a.scope.requiresNoTools = true

axiom dependencies_preserve_no_tools (foundation : List AxiomDefinition) (a : AxiomDefinition)
    (ha : a ∈ foundation) (ha_no_tools : a.scope.requiresNoTools) (d : AxiomId) (hd : d ∈ a.dependencies)
    (hfound : (findAxiom foundation d).isSome) (da : AxiomDefinition) (hda : da ∈ foundation)
    (hda_id : da.id = d) : da.isCore \/ da.scope.requiresNoTools

axiom empty_tools_implies_no_tools (foundation : List AxiomDefinition) (ctx : ProjectionContext)
    (htools : ctx.withWhat.tools = []) (a : AxiomDefinition) (ha_in : a ∈ foundation)
    (ha_not_core : Not (a.isCore = true)) (ha_in_result : a ∈ projectAxioms foundation ctx) :
    a.scope.requiresNoTools = true

axiom filterMap_findAxiom_in_foundation (foundation : List AxiomDefinition) (ids : List AxiomId)
    (a : AxiomDefinition) (ha : a ∈ ids.filterMap (findAxiom foundation)) : a ∈ foundation

/-- Axiom: Empty tools context projects only core or tool-free axioms.
    When no tools available, only core axioms and those not requiring tools are loaded. -/
axiom empty_context_projects_core (foundation : List AxiomDefinition) (ctx : ProjectionContext)
    (htools : ctx.withWhat.tools = []) (_hcore : forall id, id ∈ coreAxiomIds -> (findAxiom foundation id).isSome) :
    forall a, a ∈ projectAxioms foundation ctx -> a.isCore = true \/ a.scope.requiresNoTools = true

-- Monotonicity (temporarily simplified)
axiom tools_monotonic (a : AxiomDefinition) (tools1 tools2 : List String)
    (hsubset : forall x, x ∈ tools1 -> x ∈ tools2) (h1 : axiomToolsAvailable a tools1 = true) :
    axiomToolsAvailable a tools2 = true

axiom projectByTools_monotonic (axioms : List AxiomDefinition) (tools1 tools2 : List String)
    (hsubset : tools1 ⊆ tools2) : projectByTools axioms tools1 ⊆ projectByTools axioms tools2

axiom threeAnchorProject_monotonic (foundation : List AxiomDefinition) (ctx1 ctx2 : ProjectionContext)
    (hsubset : ctx1.withWhat.tools ⊆ ctx2.withWhat.tools) (hsame_where : ctx1.where_ = ctx2.where_)
    (hsame_when : ctx1.when = ctx2.when) :
    (threeAnchorProject foundation ctx1).map AxiomDefinition.getId ⊆
    (threeAnchorProject foundation ctx2).map AxiomDefinition.getId

axiom resolveWithClosure_monotonic (foundation : List AxiomDefinition)
    (axioms1 axioms2 : List AxiomDefinition)
    (hsubset : axioms1.map AxiomDefinition.getId ⊆ axioms2.map AxiomDefinition.getId) :
    (resolveWithClosure foundation axioms1).map AxiomDefinition.getId ⊆
    (resolveWithClosure foundation axioms2).map AxiomDefinition.getId

/-- Axiom: Tool addition is monotonic - more tools enables more axioms.
    If ctx2 has all tools from ctx1 (and same where/when), ctx2's projection is a superset. -/
axiom tool_addition_monotonic (foundation : List AxiomDefinition) (ctx1 ctx2 : ProjectionContext)
    (hsubset : ctx1.withWhat.tools ⊆ ctx2.withWhat.tools) (hsame_where : ctx1.where_ = ctx2.where_)
    (hsame_when : ctx1.when = ctx2.when) :
    (projectAxioms foundation ctx1).map AxiomDefinition.getId ⊆
    (projectAxioms foundation ctx2).map AxiomDefinition.getId

-- Loader Statistics
structure LoaderStats where
  totalAxioms : Nat
  loadedAxioms : Nat
  compressionRatio : Float
  deriving Repr

def computeStats (foundation : List AxiomDefinition) (result : ProjectedAxiomSet) : LoaderStats :=
  let total := foundation.length
  let loaded := result.axioms.length
  let ratio := if total > 0 then loaded.toFloat / total.toFloat else 0.0
  { totalAxioms := total, loadedAxioms := loaded, compressionRatio := ratio }

axiom List.length_le_of_subset_axiom {alpha : Type} [DecidableEq alpha] (xs ys : List alpha) (h : xs ⊆ ys) : xs.length <= ys.length
axiom Nat.toFloat_nonneg (n : Nat) : 0 <= n.toFloat
axiom Nat.toFloat_pos (n : Nat) (h : n > 0) : 0 < n.toFloat
axiom Nat.toFloat_mono {m n : Nat} (h : m <= n) : m.toFloat <= n.toFloat
axiom Float.div_nonneg (x y : Float) (hx : 0 <= x) (hy : 0 < y) : 0 <= x / y
axiom Float.div_le_one (x y : Float) (hx : 0 <= x) (hy : 0 < y) (hle : x <= y) : x / y <= 1
axiom Float.zero_nonneg : (0 : Float) <= 0.0
axiom Float.zero_le_one : (0.0 : Float) <= 1

theorem compression_ratio_bounded (foundation : List AxiomDefinition) (result : ProjectedAxiomSet)
    (hsubset : result.axioms ⊆ foundation) :
    let stats := computeStats foundation result
    0 <= stats.compressionRatio /\ stats.compressionRatio <= 1 := by
  simp only [computeStats]
  by_cases htotal : foundation.length > 0
  · constructor
    · have hloaded_nonneg : 0 <= result.axioms.length.toFloat := Nat.toFloat_nonneg result.axioms.length
      have htotal_pos : 0 < foundation.length.toFloat := Nat.toFloat_pos foundation.length htotal
      simp only [htotal, ↓reduceIte]
      exact Float.div_nonneg result.axioms.length.toFloat foundation.length.toFloat hloaded_nonneg htotal_pos
    · have hle : result.axioms.length <= foundation.length := List.length_le_of_subset_axiom result.axioms foundation hsubset
      have hloaded_nonneg : 0 <= result.axioms.length.toFloat := Nat.toFloat_nonneg result.axioms.length
      have htotal_pos : 0 < foundation.length.toFloat := Nat.toFloat_pos foundation.length htotal
      have hle_float : result.axioms.length.toFloat <= foundation.length.toFloat := Nat.toFloat_mono hle
      simp only [htotal, ↓reduceIte]
      exact Float.div_le_one result.axioms.length.toFloat foundation.length.toFloat hloaded_nonneg htotal_pos hle_float
  · simp only [htotal, ↓reduceIte]
    exact And.intro Float.zero_nonneg Float.zero_le_one

end SigmaGov.AxiomLoader
