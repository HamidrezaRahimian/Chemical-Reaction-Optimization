package edu.swarmintelligence.woa;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Whale Optimization Algorithm (WOA) Implementation.
 * <p>
 * Based on: Mirjalili & Lewis (2016). The Whale Optimization Algorithm.
 * Strict adherence to the scientific definition provided in the knowledge base.
 * Designed for students in Applied Informatics to study spiral-based swarm intelligence.
 * </p>
 * <p>
 * <b>Extensions beyond the original paper:</b> All generated positions are clamped
 * to {@code [minBounds, maxBounds]} to enforce feasibility. The original algorithm
 * assumes an unconstrained search space.
 * </p>
 *
 * @see <a href="https://doi.org/10.1016/j.advengsoft.2016.01.008">Advances in Engineering Software, 95, 51–67</a>
 */
@Slf4j
public class WhaleOptimizationAlgorithm {
    private final WoaConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final Whale[] population;
    private double[] xBestPosition;
    private double xBestFitness;

    public WhaleOptimizationAlgorithm(WoaConfig config, ObjectiveFunction function) {
        this.config = config;
        this.function = function;
        // L128X256MixRandom is the modern standard for scientific/parallel algorithms
        this.random = RandomGeneratorFactory.of("L128X256MixRandom").create(config.seed());
        this.population = new Whale[config.populationSize()];
        this.xBestFitness = Double.MAX_VALUE;
        this.xBestPosition = null;

        initializePopulation();
    }

    /**
     * Phase 0: Initialization.
     * Randomly distribute whales within bounds.
     * Formula: X_i = x_min + r * (x_max - x_min)
     */
    private void initializePopulation() {
        var min = config.minBounds();
        var max = config.maxBounds();
        var n = config.dimensions();

        for (int i = 0; i < config.populationSize(); i++) {
            var x = new double[n];
            for (int d = 0; d < n; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            var fVal = function.evaluate(x);
            population[i] = new Whale(x, fVal);

            if (fVal < xBestFitness) {
                xBestFitness = fVal;
                xBestPosition = Arrays.copyOf(x, n);
            }
        }
        log.debug("Population initialized. Size: {}, Initial xBest Fitness: {}",
                config.populationSize(), xBestFitness);
    }

    /**
     * Executes the optimization process.
     * <p>
     * Each iteration performs three possible operations for every whale:
     * <ul>
     *   <li><b>Encircling prey</b> (p < 0.5 and |A| < 1) – exploitation</li>
     *   <li><b>Random search</b> (p < 0.5 and |A| ≥ 1) – exploration</li>
     *   <li><b>Spiral updating</b> (p ≥ 0.5) – exploitation</li>
     * </ul>
     * All newly generated positions are clamped to {@code [minBounds, maxBounds]}.
     *
     * @return the best position found (a defensive copy)
     */
    public double[] optimize() {
        var n = config.dimensions();
        var min = config.minBounds();
        var max = config.maxBounds();
        var maxIter = config.maxIterations();
        var b = config.spiralConstant();

        log.info("Starting WOA. Population: {}, Max Iterations: {}", config.populationSize(), maxIter);

        for (int t = 1; t <= maxIter; t++) {
            // Control parameter 'a' decreases linearly from 2 to 0
            var a = 2.0 - 2.0 * ((double) t / maxIter);

            for (int i = 0; i < config.populationSize(); i++) {
                var whale = population[i];
                var currentPos = whale.positionRef();
                var newPosition = new double[n];

                // Mirjalili 2016: A and C are scalars generated per whale
                var r1 = random.nextDouble();
                var r2 = random.nextDouble();
                var A = a * (2.0 * r1 - 1.0); // A in [-a, a]
                var C = 2.0 * r2;             // C in [0, 2]

                var p = random.nextDouble();

                if (p < 0.5) {
                    if (Math.abs(A) < 1.0) {
                        // OPERATOR 1: ENCIRCLING PREY (Exploitation)
                        // D = |C * X* - X_i|
                        // X_new = X* - A * D
                        for (int d = 0; d < n; d++) {
                            var D = Math.abs(C * xBestPosition[d] - currentPos[d]);
                            newPosition[d] = Math.clamp(xBestPosition[d] - A * D, min[d], max[d]);
                        }
                    } else {
                        // OPERATOR 3: RANDOM SEARCH (Exploration)
                        // D = |C * X_rand - X_i|
                        // X_new = X_rand - A * D
                        var randIdx = selectRandomWhaleIndex(); // any random whale, per original paper
                        var xRand = population[randIdx].positionRef();

                        for (int d = 0; d < n; d++) {
                            var D = Math.abs(C * xRand[d] - currentPos[d]);
                            newPosition[d] = Math.clamp(xRand[d] - A * D, min[d], max[d]);
                        }
                    }
                } else {
                    // OPERATOR 2: SPIRAL UPDATING POSITION (Exploitation)
                    // D' = |X* - X_i|
                    // X_new = D' * e^(b*l) * cos(2πl) + X*
                    var l = random.nextDouble(-1.0, 1.0);
                    var spiralFactor = Math.exp(b * l) * Math.cos(2.0 * Math.PI * l);

                    for (int d = 0; d < n; d++) {
                        var D_prime = Math.abs(xBestPosition[d] - currentPos[d]);
                        newPosition[d] = Math.clamp(D_prime * spiralFactor + xBestPosition[d], min[d], max[d]);
                    }
                }

                // Update position unconditionally (WOA has no individual greedy selection)
                System.arraycopy(newPosition, 0, currentPos, 0, n);
                whale.setFitness(function.evaluate(currentPos));
            }

            updateGlobalBest();

            if (t % 100 == 0) {
                log.info("Iteration {}: xBest Fitness {}", t, xBestFitness);
            }
        }

        log.info("Optimization finished. Final xBest Fitness: {}", xBestFitness);
        return Arrays.copyOf(xBestPosition, xBestPosition.length);
    }

    /**
     * Selects a random whale index from the whole population.
     * Per the original WOA paper, any whale (including the current one) may be chosen.
     */
    private int selectRandomWhaleIndex() {
        return random.nextInt(config.populationSize());
    }

    private void updateGlobalBest() {
        for (var whale : population) {
            if (whale.getFitness() < xBestFitness) {
                xBestFitness = whale.getFitness();
                xBestPosition = whale.getClonedPosition();
            }
        }
    }

    public double getGlobalBestFitness() {
        return xBestFitness;
    }

    /**
     * Returns the exploration‑exploitation control parameter {@code a} for iteration {@code t}.
     * {@code a} decreases linearly from 2 to 0 over {@code maxIter} iterations.
     */
    public double getControlParameter(int t, int maxIter) {
        return 2.0 - 2.0 * ((double) t / maxIter);
    }

    /**
     * Theoretical probability of exploration (|A| ≥ 1) given current {@code a}.
     * Derived from the uniform distribution of {@code r} in {@code A = 2*a*r - a}.
     * Not part of the original algorithm, but useful for analysis.
     */
    public double getExplorationProbability(int t, int maxIter) {
        var a = getControlParameter(t, maxIter);
        return (a > 1.0) ? (a - 1.0) / a : 0.0;
    }

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Configuration record for WOA parameters.
     * Immutable container with validation logic and defensive array copying.
     * Use the generated {@code builder()} for fluent construction:
     * <pre>{@code
     * var cfg = WoaConfig.builder()
     *     .populationSize(30).maxIterations(1000).dimensions(10)
     *     .minBounds(new double[]{-5,...}).maxBounds(new double[]{5,...})
     *     .spiralConstant(1.0).seed(12345L).build();
     * }</pre>
     *
     * @param populationSize Number of whales (N). Typical: 30–50.
     * @param maxIterations  Maximum number of iterations (T_max).
     * @param dimensions     Dimensionality of the problem (n).
     * @param minBounds      Lower bounds for search space (x_min).
     * @param maxBounds      Upper bounds for search space (x_max).
     * @param spiralConstant Spiral shape constant (b). Typical: 1.0.
     * @param seed           Random seed for strict reproducibility.
     */
    @Builder
    public record WoaConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            double spiralConstant,
            long seed
    ) {
        public WoaConfig {
            if (populationSize <= 0) throw new IllegalArgumentException("PopulationSize must be > 0");
            if (maxIterations <= 0) throw new IllegalArgumentException("MaxIterations must be > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("Dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions) {
                throw new IllegalArgumentException("Bounds must match dimensions");
            }
            if (spiralConstant <= 0.0) throw new IllegalArgumentException("SpiralConstant must be > 0");
            minBounds = Arrays.copyOf(minBounds, dimensions);
            maxBounds = Arrays.copyOf(maxBounds, dimensions);
        }

        public static WoaConfig createDefaultConfig(int populationSize, int maxIterations, int dimensions,
                                                    double[] min, double[] max, long seed) {
            return new WoaConfig(populationSize, maxIterations, dimensions, min, max, 1.0, seed);
        }
    }

    /**
     * Internal representation of a Whale individual.
     * Mutable for performance during optimization phases.
     * Note: WOA has NO individual memory (no personal best like PSO).
     * <p>
     * The position array is intentionally not exposed via a standard getter.
     * Use {@link #getClonedPosition()} for safe read access.
     * The optimization loop accesses the internal array directly via {@link #positionRef()}.
     * </p>
     */
    @AllArgsConstructor
    private static class Whale {
        // Internal mutable reference for tight‑loop performance
        private final double[] position;
        @Getter
        @Setter
        private double fitness;

        /**
         * Safely clones the position to prevent external mutation.
         */
        public double[] getClonedPosition() {
            return Arrays.copyOf(position, position.length);
        }

        /**
         * Returns the internal position array (mutable) – for use ONLY inside the optimizer loop.
         */
        double[] positionRef() {
            return position;
        }
    }

    /**
     * Example usage for students. Run the {@link #main()} method to see WOA in action.
     */
    @Slf4j
    public static class WOAExample {
        public static void main() {
            log.info("Starting WOA Optimization Example");

            // Sphere Function: f(x) = sum(x_i^2). Global minimum at x=0.
            ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();

            int dimensions = 10;
            double[] min = new double[dimensions];
            double[] max = new double[dimensions];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            var config = WoaConfig.createDefaultConfig(
                    30, 1000, dimensions, min, max, 12345L
            );

            var woa = new WhaleOptimizationAlgorithm(config, sphere);
            var bestSolution = woa.optimize();

            log.info("Best Solution: {}", Arrays.toString(bestSolution));
            log.info("Best Objective Value: {}", sphere.evaluate(bestSolution));
        }
    }
}