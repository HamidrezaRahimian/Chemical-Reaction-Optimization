# CHANGES

## Aufgabe 1 - Analyse und Korrektur der CRO-Implementierung

AI tool used: Codex

### Found Problems

- The implementation depended on Lombok and SLF4J although the target project has no build file that provides these dependencies.
- `Math.clamp` was used, which is not available in the locally installed JDK and makes the source less portable for grading.
- Synthesis could reduce the population to one molecule, after which `ensureMinimumPopulation` injected random new molecules with fresh energy.
- The injected replacement molecules changed total system energy and contradicted the CRO energy model.
- The collision counter was reset after every accepted move, so decomposition was triggered mainly by repeated failed moves instead of by the usual stagnation logic since the molecule's best state.
- The molecule's local minimum record was stored but not used consistently for global-best tracking and decomposition.
- Configuration validation missed finite-value checks and did not reject invalid search intervals such as `min >= max`.
- Objective functions received the internal structure array directly, so a caller could accidentally mutate optimizer state.
- The implementation contained claims about strict Lam-and-Li energy conservation that were not true because of population reinjection.

### Performed Corrections

- Replaced Lombok-generated builder, getters, setters and logging with explicit plain Java code.
  Reason: the submitted source must be understandable and compilable without extra framework setup.
- Replaced `Math.clamp` with a small local `clamp` helper.
  Reason: boundary handling is required, but it should not depend on unavailable APIs.
- Removed artificial population reinjection and prevented synthesis when it would shrink the population below two.
  Reason: CRO must not create new potential and kinetic energy outside a valid reaction.
- Kept the four CRO reactions: on-wall ineffective collision, decomposition, inter-molecular ineffective collision and synthesis.
  Reason: the implementation remains CRO and does not copy PSO behavior.
- Implemented acceptance checks based on available potential plus kinetic energy and explicit buffer use for decomposition.
  Reason: reaction acceptance must respect CRO's energy accounting.
- Tracked each molecule's best known structure, best potential energy, total collision count and best-hit count.
  Reason: decomposition should react to stagnation since the molecule's best state, not only to failed proposals.
- Updated global best from molecule-local best records.
  Reason: later accepted worsening moves must not lose the best solution found earlier.
- Added defensive copies for configuration arrays, returned best structures and objective-function evaluation input.
  Reason: mutable arrays should not allow external code to corrupt optimizer state.
- Strengthened configuration validation for dimensions, bounds, finite probabilities, energy parameters and step size.
  Reason: invalid parameters should fail early with a clear exception instead of producing undefined algorithm behavior.
- Kept the existing public usage style, including `CroConfig.builder()`, `optimize()`, `getBestPE()` and the demo class.
  Reason: the correction should preserve the project structure and avoid unnecessary API churn.
