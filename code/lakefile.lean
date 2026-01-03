import Lake
open Lake DSL

package sigmagov where
  version := v!"0.3.0"
  description := "Lean 4 Formalization of SigmaGov (Five Pillars, T0-T8, NPL Convergence)"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.3.0"

@[default_target]
lean_lib SigmaGov where
  srcDir := "."
  globs := #[.submodules `SigmaGov]
