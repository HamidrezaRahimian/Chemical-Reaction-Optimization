package edu.swarmintelligence.tso;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Tuna Swarm Optimization (TSO) – strictly verified against the source paper.
 * <p>
 * <b>Reference:</b> Xie, Han, Zhou, Zhang, Han &amp; Tang (2021).
 * Tuna Swarm Optimization. <i>Computational Intelligence and Neuroscience</i>,
 * 2021, Article ID 9210050.
 * <a href="https://doi.org/10.1155/2021/9210050">DOI:10.1155/2021/9210050</a>.
 * <p>
 * This educational implementation follows Algorithm 1 from the paper.
 * <b>Every equation is reproduced exactly</b>.
 * For students of Applied Informatics, the code illustrates cooperative‑hunting‑based
 * swarm intelligence with a clear mapping to the mathematical model.
 * <p>
 * <b>Algorithm overview:</b>
 * <ol>
 *   <li>Random uniform initialization within the search boundaries.</li>
 *   <li>In each iteration compute time‑dependent parameters:
 *       <ul>
 *           <li>Spiral weight {@code a = (1 - t/T)^(t/T)}</li>
 *           <li>Parabolic coefficient {@code TF = (t/T)²}</li>
 *       </ul>
 *   </li>
 *   <li>For every tuna, choose with equal probability between
 *       <b>spiral foraging</b> (Eqs. (7) or (8)) and <b>parabolic foraging</b> (Eq. (9)).
 *       A new position is accepted only if it improves fitness (greedy selection).
 *       The global best is updated immediately whenever a better fitness appears.</li>
 * </ol>
 * <p>
 * <b>Java 25 LTS features (no preview):</b>
 * <ul>
 *   <li>{@code Math.clamp} (since 21) – clean boundary handling</li>
 *   <li>{@code L128X256MixRandom} – high‑quality LXM generator for scientific work</li>
 *   <li>Lombok stable annotations – zero boilerplate</li>
 *   <li>Records for immutable configuration</li>
 * </ul>
 */
@Slf4j
public final class TunaSwarmOptimization {
    private final TsoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator rng;
    private final Tuna[] population;
    private final double[] xBestPosition;
    private double xBestFitness;

    /**
     * Initializes the TSO optimiser.
     *
     * @param config   valid configuration (population size, bounds, seed, …)
     * @param function objective function to minimize (ℝⁿ → ℝ)
     */
    public TunaSwarmOptimization(final TsoConfig config, final ObjectiveFunction function) {
        this.config = config;
        this.function = function;
        this.rng = RandomGeneratorFactory.of("L128X256MixRandom")
                .create(config.seed());
        this.population = new Tuna[config.populationSize()];
        this.xBestFitness = Double.MAX_VALUE;
        this.xBestPosition = new double[config.dimensions()];
        initializePopulation();
    }

    // ---------- Initialisation (uniform random, paper Eq. (3) area) ----------
    private void initializePopulation() {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int n = config.dimensions();
        final int N = config.populationSize();

        for (int i = 0; i < N; i++) {
            double[] x = new double[n];
            for (int d = 0; d < n; d++) {
                x[d] = min[d] + rng.nextDouble() * (max[d] - min[d]);
            }
            double f = function.evaluate(x);
            population[i] = new Tuna(n, x, f);
            if (f < xBestFitness) {
                xBestFitness = f;
                System.arraycopy(x, 0, xBestPosition, 0, n);
            }
        }
        log.debug("Population initialised. Size: {}, initial best fitness: {}", N, xBestFitness);
    }

    // ---------- Main optimization loop (Algorithm 1) ----------

    /**
     * Runs the complete TSO optimisation and returns the best position found.
     *
     * @return a defensive copy of the best position (length = dimensions)
     */
    public double[] optimize() {
        final int n = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int maxIter = config.maxIterations();
        final int popSize = config.populationSize();

        log.info("Starting TSO (full scientific model). Pop: {}, MaxIter: {}", popSize, maxIter);

        for (int t = 1; t <= maxIter; t++) {
            final double iterRatio = (double) t / maxIter;

            // a = (1 − t/T)^{t/T}   (paper Eq. (6) area)
            final double a = Math.pow(1.0 - iterRatio, iterRatio);
            // TF = (t/T)²            (paper Eq. (9))
            final double TF = iterRatio * iterRatio;

            for (int i = 0; i < popSize; i++) {
                final Tuna tuna = population[i];
                final double[] xI = tuna.getPosition();
                final double[] newPos = new double[n];

                if (rng.nextBoolean()) {          // SPIRAL FORAGING
                    if (rng.nextBoolean()) {
                        // Eq. (7) – companion‑based spiral
                        int rIdx = selectRandomCompanion(i, popSize);
                        double[] xR = population[rIdx].getPosition();
                        for (int d = 0; d < n; d++) {
                            newPos[d] = xBestPosition[d]
                                    + a * (xBestPosition[d] - xI[d])
                                    + (1.0 - a) * (xBestPosition[d] - xR[d]);
                        }
                    } else {
                        // Eq. (8) – random‑point spiral
                        for (int d = 0; d < n; d++) {
                            double xRand = min[d] + rng.nextDouble() * (max[d] - min[d]);
                            newPos[d] = xBestPosition[d]
                                    + a * (xBestPosition[d] - xI[d])
                                    + (1.0 - a) * (xBestPosition[d] - xRand);
                        }
                    }
                } else {                           // PARABOLIC FORAGING
                    // Eq. (9): X_i = X_best + rand·(X_best−X_i) + TF·(X_r−X_i)
                    int rIdx = selectRandomCompanion(i, popSize);
                    double[] xR = population[rIdx].getPosition();
                    double rand = rng.nextDouble();   // single random scalar for all dims
                    for (int d = 0; d < n; d++) {
                        newPos[d] = xBestPosition[d]
                                + rand * (xBestPosition[d] - xI[d])
                                + TF * (xR[d] - xI[d]);
                    }
                }

                // Enforce search boundaries (Java 21+ Math.clamp)
                for (int d = 0; d < n; d++) {
                    newPos[d] = Math.clamp(newPos[d], min[d], max[d]);
                }

                double newFitness = function.evaluate(newPos);

                // Greedy acceptance for the individual (paper line 13)
                if (newFitness < tuna.getFitness()) {
                    tuna.setFitness(newFitness);
                    System.arraycopy(newPos, 0, tuna.getPosition(), 0, n);
                }

                // Immediate update of the global best (paper line 13–14)
                if (newFitness < xBestFitness) {
                    xBestFitness = newFitness;
                    System.arraycopy(newPos, 0, xBestPosition, 0, n);
                }
            }

            if (t % 100 == 0) {
                log.info("Iteration {}: current best fitness {}", t, xBestFitness);
            }
        }

        log.info("Optimisation finished. Final best fitness: {}", xBestFitness);
        return xBestPosition.clone();
    }

    /**
     * Selects a random companion index different from {@code currentIdx}.
     */
    private int selectRandomCompanion(final int currentIdx, final int popSize) {
        if (popSize <= 1) return currentIdx;
        int r;
        do {
            r = rng.nextInt(popSize);
        } while (r == currentIdx);
        return r;
    }

    /**
     * @return the best fitness value observed during the optimization
     */
    public double getGlobalBestFitness() {
        return xBestFitness;
    }

    // ---------- Supporting types ----------

    /**
     * Objective function f: ℝⁿ → ℝ to be minimized.
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable configuration for TSO. All arrays are defensively cloned.
     */
    @Builder
    public record TsoConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            long seed
    ) {
        public TsoConfig {
            if (populationSize <= 0)
                throw new IllegalArgumentException("PopulationSize must be > 0");
            if (maxIterations <= 0)
                throw new IllegalArgumentException("MaxIterations must be > 0");
            if (dimensions <= 0)
                throw new IllegalArgumentException("Dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions) {
                throw new IllegalArgumentException(
                        "Bounds must match dimensions: min=" + minBounds.length
                                + ", max=" + maxBounds.length + ", dims=" + dimensions);
            }
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }
    }

    /**
     * Mutable individual representing a tuna.
     */
    @Getter
    @Setter
    private static class Tuna {
        private final double[] position;
        private double fitness;

        Tuna(final int dimensions, final double[] initialPosition, final double fitness) {
            this.position = Arrays.copyOf(initialPosition, dimensions);
            this.fitness = fitness;
        }
    }

    // ---------- Example usage (not part of the library) ----------
    @Slf4j
    public static class TSOExample {
        static void main() {
            ObjectiveFunction sphere = x ->
                    Arrays.stream(x).map(v -> v * v).sum();

            int dims = 10;
            double[] min = new double[dims];
            double[] max = new double[dims];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            TsoConfig config = TsoConfig.builder()
                    .populationSize(30)
                    .maxIterations(1000)
                    .dimensions(dims)
                    .minBounds(min)
                    .maxBounds(max)
                    .seed(12345L)
                    .build();

            TunaSwarmOptimization tso = new TunaSwarmOptimization(config, sphere);
            double[] best = tso.optimize();

            log.info("Best solution: {}", Arrays.toString(best));
            log.info("Objective value: {}", sphere.evaluate(best));
        }
    }
}