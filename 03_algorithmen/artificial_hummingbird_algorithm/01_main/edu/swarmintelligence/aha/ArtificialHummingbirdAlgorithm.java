package edu.swarmintelligence.aha;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * <b>Artificial Hummingbird Algorithm (AHA)</b> – strict implementation of
 * <i>Zhao, W., Wang, L., &amp; Mirjalili, S. “Artificial hummingbird algorithm:
 * A new bio‑inspired optimizer,” Computer Methods in Applied Mechanics and
 * Engineering, vol. 388, 2022, 114194.</i>
 *
 * <p>
 * A population‑based meta‑heuristic inspired by the foraging and flight
 * behavior of hummingbirds. Each hummingbird maintains a current food source and a
 * <i>visit table</i> that records how many times it has visited every other
 * food source. The algorithm balances exploitation and exploration through
 * three core mechanisms:
 *
 * <h3>Core mechanisms</h3>
 * <ul>
 *   <li><b>Flight skills:</b> axial (one dimension), diagonal (two), or
 *       omnidirectional (all). The direction vector <b>D</b> is chosen uniformly
 *       at random.</li>
 *   <li><b>Guided foraging (exploitation):</b> each hummingbird selects the
 *       <i>most visited</i> food source in its visit table (ties broken by
 *       best objective value) and flies towards it using
 *       {@code v = x_i + a·D⊙(x_i - x_j)}, where {@code a ~ N(0,1)}.
 *       If the candidate is better, the hummingbird moves and <b>resets the visit
 *       count of its former source to 0</b>; the visit count of the targeted
 *       source is always incremented by 1.</li>
 *   <li><b>Territorial foraging (exploration):</b> a hummingbird searches locally
 *       around its own position using {@code v = x_i + b·D⊙x_i}, where
 *       {@code b ~ N(0,1)}. If it improves, it moves and <b>resets the visit
 *       count of its former source to 0</b> (no other counters are modified).</li>
 *   <li><b>Migration:</b> a hummingbird whose current food source has not
 *       improved for {@code L = 2·n} iterations is reinitialized randomly,
 *       preserving diversity.</li>
 * </ul>
 *
 * <p>
 * <b>Parameters:</b> population size <i>n</i>, maximum iterations <i>T</i>,
 * problem dimensions <i>D</i>, search bounds.
 *
 * <p>
 * <b>Design:</b> Uses {@link RandomGenerator L128X256MixRandom} for
 * reproducibility and high statistical quality. The objective function is a
 * {@link FunctionalInterface}. Self‑contained; call {@link #optimise()}
 * to obtain the best‑found solution.
 *
 * @see <a href="https://doi.org/10.1016/j.cma.2021.114194">Zhao et al. (2022)</a>
 */
@Slf4j
public final class ArtificialHummingbirdAlgorithm {
    private static final String RNG_ALGORITHM = "L128X256MixRandom";

    private final AhaConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final int[] stagnationCounters;       // iterations since last food source improvement
    private double[][] positions;           // [bird][dim]
    private double[] fitnesses;
    private int[][] visitTable;             // [bird][target] = number of visits

    @Getter
    private double bestFitness = Double.POSITIVE_INFINITY;
    private double[] bestPosition;
    private double[] convergenceHistory;    // best fitness per iteration (index 0 = initial)

    private int guidedImprovements;
    private int territorialImprovements;
    private int migrations;

    // ────────────── Constructors ──────────────

    /**
     * Public constructor – creates the RNG internally from the config seed.
     */
    public ArtificialHummingbirdAlgorithm(final AhaConfig config,
                                          final ObjectiveFunction function) {
        this(config, function,
                RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed()));
    }

    /**
     * Package‑private constructor for testing – allows injection of a
     * custom {@link RandomGenerator}.
     */
    ArtificialHummingbirdAlgorithm(final AhaConfig config,
                                   final ObjectiveFunction function,
                                   final RandomGenerator random) {
        this.config = config;
        this.function = function;
        this.random = random;

        final int n = config.populationSize();
        final int dim = config.dimensions();
        positions = new double[n][dim];
        fitnesses = new double[n];
        stagnationCounters = new int[n];
        visitTable = new int[n][n];

        initialisePopulation();
    }

    // ────────────── Public API ──────────────

    private static String sci(final double val) {
        return String.format("%.6e", val);
    }

    /**
     * Demonstration with the 10‑D Sphere function using standard parameters.
     */
    static void main() {
        log.info("Starting Artificial Hummingbird Algorithm Demo");

        final ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();
        final int dim = 10;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, -5.12);
        Arrays.fill(maxB, 5.12);

        final AhaConfig cfg = AhaConfig.builder()
                .populationSize(40)
                .maxIterations(200)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .seed(42L)
                .build();

        final var aha = new ArtificialHummingbirdAlgorithm(cfg, sphere);
        final double[] solution = aha.optimise();

        log.info("Best Solution: {}", Arrays.toString(solution));
        log.info("Best Fitness:  {}", aha.getBestFitness());
    }

    /**
     * Runs the complete optimization loop.
     *
     * @return the best solution vector found
     */
    public double[] optimise() {
        final int maxIter = config.maxIterations();
        convergenceHistory = new double[maxIter + 1];
        convergenceHistory[0] = bestFitness;

        for (int iter = 1; iter <= maxIter; iter++) {
            log.debug("Iteration {} START", iter);
            guidedImprovements = 0;
            territorialImprovements = 0;
            migrations = 0;

            // Phase 1: Foraging (guided / territorial) for each bird
            for (int i = 0; i < positions.length; i++) {
                if (random.nextDouble() < 0.5) {
                    guidedForaging(i);
                } else {
                    territorialForaging(i);
                }
            }

            // Phase 2: Migration triggered by stagnation
            checkMigration();

            convergenceHistory[iter] = bestFitness;

            if (log.isDebugEnabled()) {
                log.debug("Iteration {} END | guidedImprov={} territorialImprov={} migrations={} best={}",
                        iter, guidedImprovements, territorialImprovements, migrations,
                        sci(bestFitness));
            }

            if (log.isInfoEnabled() &&
                    (iter == 1 || iter % Math.max(1, maxIter / 10) == 0 || iter == maxIter)) {
                log.info("Iteration {}/{} | Best Fitness: {}", iter, maxIter, bestFitness);
            }
        }
        return bestPosition.clone();
    }

    /**
     * Returns a copy of the per‑iteration best fitness history (index 0 = initial).
     */
    public double[] getConvergenceHistory() {
        return convergenceHistory.clone();
    }

    /**
     * Returns a deep copy of the current visit table (for testing).
     */
    int[][] getVisitTable() {
        final int n = visitTable.length;
        final int[][] copy = new int[n][n];
        for (int i = 0; i < n; i++) {
            copy[i] = visitTable[i].clone();
        }
        return copy;
    }

    /**
     * Returns a deep copy of the positions (for testing).
     */
    double[][] getPositions() {
        final double[][] copy = new double[positions.length][];
        for (int i = 0; i < positions.length; i++) {
            copy[i] = positions[i].clone();
        }
        return copy;
    }

    /**
     * Returns current fitness array (for testing).
     */
    double[] getFitnesses() {
        return fitnesses.clone();
    }

    // ────────────── Initialisation ──────────────

    /**
     * Returns the number of migrations that happened in the last iteration.
     */
    int getMigrationCountLastIteration() {
        return migrations;
    }

    // ────────────── Foraging phases ──────────────

    /**
     * Overwrites the entire internal state with the given arrays.
     * All invariants (global best, visit table, stagnation) are recalculated.
     * <b>Only for white‑box testing.</b>
     */
    void setStateForTesting(final double[][] pos, final double[] fit, final int[][] vt) {
        final int n = pos.length;
        final int dim = config.dimensions();
        if (fit.length != n || vt.length != n || vt[0].length != n) {
            throw new IllegalArgumentException("Dimensions mismatch");
        }
        positions = new double[n][dim];
        for (int i = 0; i < n; i++) {
            System.arraycopy(pos[i], 0, positions[i], 0, dim);
        }
        fitnesses = fit.clone();
        visitTable = new int[n][n];
        for (int i = 0; i < n; i++) {
            visitTable[i] = vt[i].clone();
        }

        // Re‑compute global best from scratch (no personal bests needed)
        bestFitness = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            if (fitnesses[i] < bestFitness) {
                bestFitness = fitnesses[i];
                bestPosition = positions[i].clone();
            }
        }
        // stagnation counters are left as 0 – set externally if needed
        Arrays.fill(stagnationCounters, 0);
    }

    private void initialisePopulation() {
        final int dim = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        for (int i = 0; i < positions.length; i++) {
            for (int d = 0; d < dim; d++) {
                positions[i][d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            final double obj = function.evaluate(positions[i]);
            fitnesses[i] = obj;

            if (obj < bestFitness) {
                bestFitness = obj;
                bestPosition = positions[i].clone();
            }
        }

        // Initial visit table: each hummingbird has visited its own food source once
        for (int i = 0; i < positions.length; i++) {
            Arrays.fill(visitTable[i], 0);
            visitTable[i][i] = 1;
        }

        log.info("Initial population built | best={} mean={}",
                sci(bestFitness), sci(meanFitness()));
    }

    // ────────────── Migration ──────────────

    /**
     * Guided foraging for bird {@code i}: selects the target food source with the
     * <b>highest</b> visit count (ties broken by best fitness), then flies toward it
     * using the paper’s formula {@code v = x_i + a·D⊙(x_i - x_j)}.
     */
    void guidedForaging(final int i) {
        final int j = selectTargetHighestVisit(i);
        final double[] dir = flightDirection();
        final double a = random.nextGaussian();
        final double[] newPos = new double[config.dimensions()];

        for (int d = 0; d < config.dimensions(); d++) {
            newPos[d] = positions[i][d]   // base is the bird’s own position
                    + a * dir[d] * (positions[i][d] - positions[j][d]);
        }
        clamp(newPos);
        evaluateAndUpdate(i, j, newPos, true);
    }

    // ────────────── Evaluation & state update ──────────────

    /**
     * Territorial foraging for bird {@code i}: local search around its own position
     * using {@code v = x_i + b·D⊙x_i}.
     */
    void territorialForaging(final int i) {
        final double[] dir = flightDirection();
        final double b = random.nextGaussian();
        final double[] newPos = new double[config.dimensions()];

        for (int d = 0; d < config.dimensions(); d++) {
            newPos[d] = positions[i][d] + b * dir[d] * positions[i][d];
        }
        clamp(newPos);
        evaluateAndUpdate(i, i, newPos, false);
    }

    // ────────────── Target selection ──────────────

    private void checkMigration() {
        final int n = positions.length;
        final int limit = 2 * n;           // L = 2·n
        for (int i = 0; i < n; i++) {
            if (stagnationCounters[i] >= limit) {
                final int dim = config.dimensions();
                final double[] min = config.minBounds();
                final double[] max = config.maxBounds();
                for (int d = 0; d < dim; d++) {
                    positions[i][d] = min[d] + random.nextDouble() * (max[d] - min[d]);
                }
                final double obj = function.evaluate(positions[i]);
                fitnesses[i] = obj;
                stagnationCounters[i] = 0;

                // Reset visit table for this bird: only self‑visit = 1
                Arrays.fill(visitTable[i], 0);
                visitTable[i][i] = 1;

                if (obj < bestFitness) {
                    bestFitness = obj;
                    bestPosition = positions[i].clone();
                }
                migrations++;
                if (log.isDebugEnabled()) {
                    log.debug("MIGRATION bird={} new fitness={}", i, sci(obj));
                }
            }
        }
    }

    // ────────────── Flight direction ──────────────

    /**
     * Evaluates a candidate position, updates the hummingbird’s current food source
     * (greedy), the global best, and the stagnation counter.
     *
     * <p>
     * <b>Visit table rules (paper exact):</b>
     * <ul>
     *   <li>If the hummingbird <em>moves</em> ({@code candidate < current fitness}),
     *       the visit count of its <b>former</b> source ({@code VT[i][i]}) is
     *       reset to 0.</li>
     *   <li>For <b>guided</b> foraging only, the visit count of the targeted
     *       source {@code j} ({@code VT[i][j]}) is always incremented by 1.</li>
     * </ul>
     *
     * @param i        index of the hummingbird
     * @param j        target food source index (i itself for territorial)
     * @param newPos   candidate position (already clamped)
     * @param isGuided {@code true} for guided foraging, {@code false} for territorial
     */
    void evaluateAndUpdate(final int i, final int j,
                           final double[] newPos, final boolean isGuided) {
        final double newObj = function.evaluate(newPos);

        final boolean improved = newObj < fitnesses[i];
        if (improved) {
            System.arraycopy(newPos, 0, positions[i], 0, config.dimensions());
            fitnesses[i] = newObj;
            visitTable[i][i] = 0;       // abandon previous source
            stagnationCounters[i] = 0;  // food source changed → reset stagnation
            if (isGuided) guidedImprovements++;
            else territorialImprovements++;
        } else {
            stagnationCounters[i]++;    // food source unchanged
        }

        // Global best update
        if (newObj < bestFitness) {
            bestFitness = newObj;
            bestPosition = newPos.clone();
        }

        // Guided foraging always records a visit to the targeted source
        if (isGuided) {
            visitTable[i][j]++;
        }
    }

    // ────────────── Helpers ──────────────

    /**
     * Selects a target food source {@code j ≠ i} with the <b>highest</b>
     * visit count in hummingbird {@code i}’s row of the visit table.
     * Ties are broken by choosing the source with the best (lowest) fitness.
     */
    int selectTargetHighestVisit(final int i) {
        final int n = positions.length;
        int bestJ = -1;
        int maxVisits = -1;
        double bestFit = Double.POSITIVE_INFINITY;

        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            final int visits = visitTable[i][j];
            if (visits > maxVisits || (visits == maxVisits && fitnesses[j] < bestFit)) {
                maxVisits = visits;
                bestFit = fitnesses[j];
                bestJ = j;
            }
        }
        return bestJ;
    }

    /**
     * Returns a direction vector {@code D} representing a random flight skill:
     * axial (one dimension = 1), diagonal (two = 1), or omnidirectional (all = 1).
     * Each type is equally probable.
     */
    private double[] flightDirection() {
        final int dim = config.dimensions();
        final double[] dir = new double[dim];
        final int mode = random.nextInt(3);   // 0 → axial, 1 → diagonal, 2 → omni

        switch (mode) {
            case 0 -> dir[random.nextInt(dim)] = 1.0;
            case 1 -> {
                final int d1 = random.nextInt(dim);
                int d2;
                do {
                    d2 = random.nextInt(dim);
                } while (d2 == d1);
                dir[d1] = 1.0;
                dir[d2] = 1.0;
            }
            default -> Arrays.fill(dir, 1.0);
        }
        return dir;
    }

    private void clamp(final double[] pos) {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        for (int d = 0; d < pos.length; d++) {
            pos[d] = Math.clamp(pos[d], min[d], max[d]);
        }
    }

    // ────────────── Demo entry point ──────────────

    private double meanFitness() {
        return Arrays.stream(fitnesses).average().orElse(Double.NaN);
    }

    // ────────────── Auxiliary types ──────────────

    /**
     * Objective function to minimize.
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable algorithm configuration.
     * Lombok {@code @Builder} supplies a fluent, type‑safe builder.
     */
    @Builder
    public record AhaConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            long seed
    ) {
        public AhaConfig {
            Objects.requireNonNull(minBounds);
            Objects.requireNonNull(maxBounds);
            if (populationSize < 2)
                throw new IllegalArgumentException("populationSize must be ≥ 2");
            if (maxIterations <= 0)
                throw new IllegalArgumentException("maxIterations must be positive");
            if (dimensions <= 0)
                throw new IllegalArgumentException("dimensions must be positive");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bounds length must equal dimensions");
        }
    }
}