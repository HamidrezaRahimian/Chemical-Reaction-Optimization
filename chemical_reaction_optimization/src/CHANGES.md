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

## Aufgabe 3 - Clean Coding und SOLID-Refactoring

AI tool used: Codex

### Violations Found

- `ChemicalReactionOptimization` used manual array management for `population` (`addMolecule`, `removeMolecule`, capacity checks).
  Violation: unnecessary complexity and KISS/DRY issue because Java already provides `List`.
- `ChemicalReactionOptimization.perturb`, `performSynthesis` and `createRandomMolecule` repeatedly called `config.minBounds()` and `config.maxBounds()` inside loops.
  Violation: duplicated access logic and avoidable defensive-copy churn.
- `ChemicalReactionOptimization.reactWithTwoMolecules` mixed index selection, reaction eligibility and reaction dispatch in one method.
  Violation: method responsibility was broader than necessary.
- `ChemicalReactionOptimization.performDecomposition` duplicated the construction of decomposition products in the direct-energy and buffer-energy branches.
  Violation: DRY issue and harder-to-read energy flow.
- `ChemicalReactionOptimization` accessed `Molecule` fields directly in most reaction methods.
  Violation: weak encapsulation; molecule state rules were spread across the optimizer instead of living behind intention-revealing methods.
- `ChemicalReactionOptimization.CroConfig` compact constructor contained all validation checks inline.
  Violation: long method and mixed validation responsibilities.
- Magic numbers appeared as raw literals, especially the minimum population size and maximum initialization attempts.
  Violation: unclear intent.

### Refactorings Performed

- Replaced manual molecule array storage with `List<Molecule>`.
  Reason: simpler population add/remove/set operations, less custom bookkeeping, same CRO behavior.
- Cached validated `minBounds` and `maxBounds` once in the optimizer constructor.
  Reason: keeps defensive copies at API boundaries while avoiding repeated cloning in inner loops.
- Added named constants `MINIMUM_POPULATION_SIZE` and `MAX_INITIALIZATION_ATTEMPTS`.
  Reason: removes magic numbers and clarifies algorithm constraints.
- Extracted small helper methods: `shouldUseBimolecularReaction`, `randomDifferentIndex`, `canSynthesize`, `randomStructure`, `hasEnoughEnergy` and `replaceWithDecompositionProducts`.
  Reason: separates reaction dispatch, random structure creation and energy checks without adding new abstractions.
- Moved molecule-related queries behind methods such as `totalEnergy`, `isStagnating`, `hasKineticEnergyAtMost`, `bestStructure` and `bestPotentialEnergy`.
  Reason: improves encapsulation and makes reaction code read in CRO terms.
- Split `CroConfig` validation into focused private helpers: core settings, probabilities, energy settings and bound intervals.
  Reason: keeps construction readable while preserving the existing nested `CroConfig.builder()` API.

### Justified Deviations

- `ChemicalReactionOptimization` remains a single public optimizer class with nested `CroConfig`, `ObjectiveFunction`, `Molecule` and demo class.
  Reason: splitting into many public files would add architecture without a clear grading benefit; the current structure stays compatible with the existing tests and assignment layout.
- No extra interfaces, factories or abstract base classes were added.
  Reason: OCP/DIP do not justify additional layers here; the existing `ObjectiveFunction` functional interface is sufficient dependency inversion for the objective function.
- LSP and ISP have no material violations.
  Reason: there is no inheritance hierarchy, and the only interface has one required method.

## Aufgabe 4 - Logging der Agentendynamik und Konvergenz

AI tool used: Codex

### Logging Concept

- CRO writes a semicolon-separated CSV file with the required header:
  `iteration;agentId;positionBefore;positionAfter;personalBest;personalBestFitness;globalBest;globalBestFitness;popAvgFitness;popStdDev;distToOptimum`.
- Each molecule has a stable numeric ID. Existing molecules keep their ID across accepted moves; decomposition keeps the source ID for one product and assigns a new ID to the second product; synthesis keeps the first merged molecule ID.
- Iteration `0` logs the initialized population before any reaction. Each later iteration logs the population after one CRO reaction, including the molecule's position before and after that iteration.
- Population statistics are computed from current finite potential energies after the reaction.
- Distance to optimum is computed from the configured `knownOptimum`; for the Ackley 2D sample this is `{0.0, 0.0}`.

### Files and Classes Added or Changed

- Added `AckleyFunction.java`.
  Reason: implements the required 2D-capable Ackley test function with `a = 20`, `b = 0.2`, `c = 2 * PI`, bounds `[-32.768, 32.768]`, and global minimum at the origin.
- Added `AlgorithmLogger.java`.
  Reason: small plain-Java CSV writer without external logging frameworks.
- Updated `ChemicalReactionOptimization.java`.
  Reason: integrated optional logging, stable molecule IDs, population statistics, distance-to-optimum calculation, and `AckleyLoggingExample`.
- Updated `ChemicalReactionOptimizationTest.java`.
  Reason: added tests for Ackley minimum, CSV header/columns, and disabled logging behavior.
- Generated `chemical_reaction_optimization/algorithm_run.log`.
  Reason: required sample logfile from a full CRO run on Ackley 2D.

### How Logging Is Disabled

- Logging is disabled by default because `CroConfig.builder().loggingEnabled(...)` defaults to `false`.
- To enable logging, set `.loggingEnabled(true)` and optionally `.logPath("algorithm_run.log")`.
- The Ackley sample run enables logging explicitly and writes `algorithm_run.log` in the target project directory.

### Confirmation

- `chemical_reaction_optimization/algorithm_run.log` was generated from `ChemicalReactionOptimization.AckleyLoggingExample`.
- The sample log contains the required CSV columns and records molecule ID, before/after position, personal best, global best, population average fitness, population standard deviation, and distance to `{0.0, 0.0}`.

## Aufgabe 5 - JavaDoc-Kommentierung

AI tool used: Codex

### Files and Classes Documented

- `ChemicalReactionOptimization.java`
  - Added JavaDoc for the constructor, initialization, optimization loop, CRO core reaction operators, decomposition product handling, perturbation, feasible molecule creation, objective evaluation, global-best update, logging snapshots, convergence statistics, distance-to-optimum calculation, `ObjectiveFunction`, `CroConfig`, molecule state transitions, and the Ackley logging example.
- `AlgorithmLogger.java`
  - Added JavaDoc for the logger purpose, file opening, CSV row writing, flushing, closing, and the `LogEntry` record fields.
- `AckleyFunction.java`
  - Added JavaDoc for the benchmark role, standard parameters, required bounds, global minimum, evaluation result, and invalid dimensionality exception.
- `ChemicalReactionOptimizationTest.java`
  - Added JavaDoc to the tests that document important algorithmic guarantees: initialization, monotonic global best memory, deterministic improvement, energy conservation, synthesis population behavior, boundary clamping, defensive objective evaluation, invalid-fitness rejection, Ackley minimum, CSV logging, and disabled logging.

### Review Confirmation

- Comments were reviewed against the final refactored code and CRO behavior.
- Trivial getters, simple builder setters and obvious assertions were intentionally not commented.
- JavaDoc uses `@param`, `@return`, and `@throws` where they add concrete information.
- The comments explain CRO-specific behavior instead of copying PSO terminology or generic descriptions.
