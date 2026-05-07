package edu.swarmintelligence.gwo;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Gray Wolf Optimisation (GWO) Algorithm Implementation.
 * <p>
 * Based on: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Gray Wolf Optimizer.
 * Advances in Engineering Software, 69, 46–61.
 * Designed for students in Applied Informatics to study swarm intelligence
 * inspired by gray wolf hunting behavior and social hierarchy.
 * </p>
 * <p>
 * Technical Architecture:
 * <ul>
 *   <li>Java 25 LTS Ready: Baseline Java 21+, manual bounds enforcement.</li>
 *   <li>Lombok Driven: Eliminates all getter / setter / logging / builder boilerplate.</li>
 *   <li>LXM PRNG: Uses {@code L128X256MixRandom} via {@code RandomGeneratorFactory} for
 *       statistically robust, scientific‑grade stochastic generation.</li>
 *   <li>Strict Immutability: {@code record} ensures bounds and configuration state cannot leak.</li>
 * </ul>
 * </p>
 * <p>
 * The algorithm models the social hierarchy of gray wolves (alpha, beta, delta)
 * and their encircling and attacking behaviors. The exploration/exploitation
 * balance is controlled by the linearly decreasing coefficient {@code a}.
 * Key steps:
 * <ol>
 *   <li>Initialise positions and evaluate.</li>
 *   <li>Determine alpha, beta, delta (top‑3 fitness).</li>
 *   <li>Update each wolf’s position using the three leaders.</li>
 *   <li>Re‑evaluate and refresh leaders.</li>
 *   <li>Decrease {@code a} linearly from 2 to 0 and repeat.</li>
 * </ol>
 * </p>
 *
 * @see <a href="https://doi.org/10.1016/j.advengsoft.2013.12.007">Mirjalili et al. (2014)</a>
 */
@Slf4j
public class GreyWolfOptimization {
    private final GwoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final List<Wolf> pack;
    private double[] alphaPos;
    private double alphaScore = Double.POSITIVE_INFINITY;
    private double[] betaPos;
    private double[] deltaPos;

    /**
     * Creates a new GWO optimizer with the given configuration and objective.
     * Immediately initializes the wolf pack and determines the first leaders.
     *
     * @param config   immutable configuration record (built via Lombok builder)
     * @param function the objective function to minimize (fitness)
     */
    public GreyWolfOptimization(final GwoConfig config, final ObjectiveFunction function) {
        this.config = Objects.requireNonNull(config);
        this.function = Objects.requireNonNull(function);
        this.random = RandomGeneratorFactory.of("L128X256MixRandom").create(config.seed());
        this.pack = new ArrayList<>(config.popSize());
        initializePack();
    }

    /* ------------------------------------------------------------------ */
    /*  Phase 0 – Initialisation                                           */
    /* ------------------------------------------------------------------ */

    /**
     * Generates random wolves, evaluates them, and extracts alpha, beta, delta.
     */
    private void initializePack() {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int dims = config.dimensions();

        for (int i = 0; i < config.popSize(); i++) {
            var wolf = createRandomWolf(dims, min, max);
            pack.add(wolf);
        }
        refreshLeaders();
        log.debug("Pack initialized. Alpha score: {}", alphaScore);
    }

    /* ------------------------------------------------------------------ */
    /*  Main optimisation loop                                             */
    /* ------------------------------------------------------------------ */

    /**
     * Runs the main iteration loop and returns the best position found.
     *
     * @return a <em>clone</em> of the alpha (global best) position
     */
    public double[] optimize() {
        final int dims = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int maxIter = config.maxIterations();

        for (int t = 1; t <= maxIter; t++) {
            // Linearly decreasing coefficient a from 2 to 0 (canonical GWO)
            double a = 2.0 * (1.0 - t / (double) maxIter);

            // Save current leaders for this iteration (all wolves use the same leaders)
            final double[] alpha = alphaPos.clone();
            final double[] beta = betaPos.clone();
            final double[] delta = deltaPos.clone();

            // Step 1: compute new positions using leaders from the *previous* iteration
            for (var wolf : pack) {
                double[] curPos = wolf.getPosition();
                var newPos = new double[dims];

                for (int d = 0; d < dims; d++) {
                    // Influence of Alpha
                    double r1 = random.nextDouble();
                    double r2 = random.nextDouble();
                    double A1 = 2.0 * a * r1 - a;
                    double C1 = 2.0 * r2;
                    double D_alpha = Math.abs(C1 * alpha[d] - curPos[d]);
                    double X1 = alpha[d] - A1 * D_alpha;

                    // Influence of Beta
                    r1 = random.nextDouble();
                    r2 = random.nextDouble();
                    double A2 = 2.0 * a * r1 - a;
                    double C2 = 2.0 * r2;
                    double D_beta = Math.abs(C2 * beta[d] - curPos[d]);
                    double X2 = beta[d] - A2 * D_beta;

                    // Influence of Delta
                    r1 = random.nextDouble();
                    r2 = random.nextDouble();
                    double A3 = 2.0 * a * r1 - a;
                    double C3 = 2.0 * r2;
                    double D_delta = Math.abs(C3 * delta[d] - curPos[d]);
                    double X3 = delta[d] - A3 * D_delta;

                    // Average the three leader‑guided steps
                    double raw = (X1 + X2 + X3) / 3.0;

                    // Manual clamping (Java does not have Math.clamp for double)
                    newPos[d] = Math.clamp(raw, min[d], max[d]);
                }

                wolf.setPosition(newPos);
            }

            // Step 2: evaluate all wolves with their new positions
            for (var wolf : pack) {
                double fit = function.evaluate(wolf.getPosition());
                if (!Double.isFinite(fit)) fit = Double.POSITIVE_INFINITY;
                wolf.setFitness(fit);
            }

            // Step 3: update leaders after the entire pack has moved
            refreshLeaders();

            if (t % 100 == 0 || t == maxIter) {
                log.info("Iteration {}/{}: Alpha score {}, Pop Size: {}",
                        t, maxIter, alphaScore, config.popSize());
            }
        }

        log.info("Optimization finished. Final Alpha score: {}", alphaScore);
        return alphaPos.clone();
    }

    /* ------------------------------------------------------------------ */
    /*  Leader management                                                  */
    /* ------------------------------------------------------------------ */

    /**
     * Re‑scans the entire pack and updates alpha, beta, delta.
     */
    private void refreshLeaders() {
        // Sort by fitness (ascending)
        pack.sort(Comparator.comparingDouble(Wolf::getFitness));

        alphaScore = pack.get(0).getFitness();
        alphaPos = pack.get(0).getPosition().clone();

        if (pack.size() > 1) {
            betaPos = pack.get(1).getPosition().clone();
        } else {
            betaPos = alphaPos.clone();
        }

        if (pack.size() > 2) {
            deltaPos = pack.get(2).getPosition().clone();
        } else {
            deltaPos = betaPos.clone();
        }
    }

    /**
     * Generates a random feasible wolf by resampling until a finite fitness is obtained.
     * After {@code MAX_RETRIES} attempts an exception is thrown to avoid infinite loops.
     */
    private Wolf createRandomWolf(final int dims, final double[] min, final double[] max) {
        final int MAX_RETRIES = 1000;
        for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
            var x = new double[dims];
            for (int d = 0; d < dims; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            double fit = function.evaluate(x);
            if (Double.isFinite(fit)) {
                return new Wolf(x, fit);
            }
        }
        throw new IllegalStateException(
                "Failed to generate a feasible wolf after " + MAX_RETRIES + " attempts.");
    }

    public double getBestFitness() {
        return alphaScore;
    }

    /* ------------------------------------------------------------------ */
    /*  Embedded types                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Functional interface for the objective function (Fitness).
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Configuration record for GWO parameters using Lombok Builder.
     * All arrays are defensively copied and validated in the compact constructor.
     */
    @Builder
    public record GwoConfig(
            int popSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            long seed
    ) {
        /**
         * Compact constructor validates all parameters and copies array-boundaries.
         */
        public GwoConfig {
            Objects.requireNonNull(minBounds, "minBounds must not be null");
            Objects.requireNonNull(maxBounds, "maxBounds must not be null");
            if (popSize < 2) throw new IllegalArgumentException("PopSize must be >= 2");
            if (maxIterations <= 0) throw new IllegalArgumentException("MaxIterations must be > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("Dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions) {
                throw new IllegalArgumentException("Bounds length must match dimensions");
            }
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }
    }

    /**
     * Internal wolf representation. Lombok {@code @Data} handles mutable state.
     */
    @Data
    private static class Wolf {
        private double[] position;
        private double fitness;

        Wolf(final double[] position, final double fitness) {
            this.position = position.clone();
            this.fitness = fitness;
        }
    }

    /**
     * Educational usage example demonstrating the builder pattern.
     */
    public static class GwoExample {
        public static void main() {
            log.info("Starting Grey Wolf Optimisation Example");

            final ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();

            final int dimensions = 10;
            final var min = new double[dimensions];
            final var max = new double[dimensions];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            final var config = GwoConfig.builder()
                    .popSize(30)
                    .maxIterations(1000)
                    .dimensions(dimensions)
                    .minBounds(min)
                    .maxBounds(max)
                    .seed(42L)
                    .build();

            final var gwo = new GreyWolfOptimization(config, sphere);
            final double[] bestSolution = gwo.optimize();

            log.info("Best Solution Found: {}", Arrays.toString(bestSolution));
            log.info("Best Objective Value (fitness): {}", gwo.getBestFitness());
        }
    }
}