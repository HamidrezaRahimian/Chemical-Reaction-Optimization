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

## Aufgabe 2 - Analyse und Verbesserung des CRO-Testszenarios

AI tool used: Codex

### Test Problems Found

- The previous tests were mostly flat and did not separate initialization, iteration behavior, reaction dynamics, boundaries and configuration validation.
- The convergence test used a high-dimensional stochastic run with an absolute threshold, making the assertion less focused on stable CRO behavior.
- Population behavior was only smoke-tested by checking that `optimize()` did not throw.
- Initialization was not checked directly; there was no assertion for configured population size or finite initial best energy.
- Edge cases for infeasible objective functions and non-finite candidate energies were missing.
- Defensive-copy behavior for result arrays, configuration bounds and objective-function inputs was not covered.
- Boundary tests only checked the final best from one scenario and did not deliberately stress clamping with a larger step size.
- Configuration validation missed finite-value cases and invalid intervals such as equal lower and upper bounds.

### Tests Changed or Added

- Reworked the test class into nested JUnit 5 groups: initialization, optimization behavior, CRO reaction dynamics, boundaries/result integrity, invalid fitness handling and configuration validation.
  Reason: the structure now mirrors the algorithm concerns required by the specification and makes failures easier to interpret.
- Added initialization tests for configured population size, finite initial best energy and infeasible objective functions.
  Reason: Aufgabe 2 explicitly asks whether initialization and edge cases are covered.
- Replaced the brittle convergence check with deterministic one-dimensional sphere improvement plus a moderate final threshold.
  Reason: this checks real optimizer progress without depending on uncontrolled random luck.
- Added energy-conservation coverage over a longer deterministic run.
  Reason: CRO reaction acceptance depends on potential energy, kinetic energy and buffer accounting.
- Added synthesis-specific population tests.
  Reason: CRO may change population size through decomposition/synthesis, and synthesis must not reduce the population below two.
- Added deterministic seed assertions for best solution, best potential energy and final population size.
  Reason: seeded stochastic algorithms must be reproducible for grading and debugging.
- Added boundary and defensive-copy tests for returned best positions, configuration bounds and objective evaluation input arrays.
  Reason: mutable arrays are a common source of hidden state corruption in optimization code.
- Added invalid-fitness handling tests.
  Reason: non-finite objective values must not become global bests or corrupt later optimization.
- Extended configuration validation tests for NaN/infinite parameters and invalid bound intervals.
  Reason: invalid inputs should fail early and consistently.
