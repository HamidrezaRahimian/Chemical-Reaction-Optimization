package edu.swarmintelligence.goa;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Grasshopper Optimisation Algorithm (GOA) Implementation.
 * <p>
 * Based on: Saremi, S., Mirjalili, S., & Lewis, A. (2017). Grasshopper Optimisation
 * Algorithm: Theory and application. Advances in Engineering Software, 105, 30–47.
 * Designed for students in Applied Informatics to study swarm intelligence
 * inspired by grasshopper swarming behavior.
 * </p>
 *
 * <h3>Technical Architecture</h3>
 * <ul>
 *   <li>Java 25 LTS Ready: Baseline Java 21+, fully leveraging {@code Math.clamp}.</li>
 *   <li>Lombok Driven: Eliminates all getter/setter/logging boilerplate.</li>
 *   <li>LXM PRNG: Uses {@code L128X256MixRandom} via {@code RandomGeneratorFactory}
 *       for statistically robust, scientific‑grade stochastic generation.</li>
 *   <li>Strict Immutability: {@code GoaConfig} is a record with defensive copies
 *       and validation; builder avoids Lombok experimental features.</li>
 * </ul>
 *
 * <p>
 * The algorithm mimics the swarming behavior of grasshoppers. Each individual
 * adjusts its position based on social interaction (attraction/repulsion), the
 * target (best‑found position), and a decreasing comfort‑zone coefficient {@code c}.
 * Inter‑grasshopper Euclidean distances are <strong>normalized to [1, 4]</strong>
 * in every iteration, as specified by the original pseudo‑code, to keep the
 * social force effective regardless of the search space scale.
 * </p>
 *
 * @see <a href="https://doi.org/10.1016/j.advengsoft.2017.01.004">Saremi et al. (2017)</a>
 */
@Slf4j
public class GrasshopperOptimization {
    private final GoaConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final List<Grasshopper> swarm;
    private double[] targetPosition;
    private double targetFitness = Double.POSITIVE_INFINITY;

    /**
     * Creates a new GOA optimizer with the given configuration and objective.
     * Immediately initializes the grasshopper swarm and records the best (target).
     *
     * @param config   immutable configuration (built via {@link GoaConfig.Builder})
     * @param function the objective function to minimize (fitness = potential energy)
     */
    public GrasshopperOptimization(final GoaConfig config, final ObjectiveFunction function) {
        this.config = Objects.requireNonNull(config);
        this.function = Objects.requireNonNull(function);
        this.random = RandomGeneratorFactory.of("L128X256MixRandom")
                .create(config.seed());
        this.swarm = new ArrayList<>(config.popSize());
        initializeSwarm();
    }

    /* ------------------------------------------------------------------ */
    /*  Phase 0 – Initialisation                                           */
    /* ------------------------------------------------------------------ */

    /**
     * Social interaction function (attraction / repulsion).
     * <p>
     * s(r) = f · exp(−r / l) − exp(−r)
     *
     * @param r distance scalar (typically the per‑dimension absolute difference)
     * @param f attraction intensity (default 0.5)
     * @param l attractive length scale (default 1.5)
     * @return social force value
     */
    public static double socialForce(final double r, final double f, final double l) {
        return f * Math.exp(-r / l) - Math.exp(-r);
    }

    /* ------------------------------------------------------------------ */
    /*  Main optimisation loop                                             */
    /* ------------------------------------------------------------------ */

    /**
     * Euclidean distance between two positions.
     */
    static double euclideanDist(final double[] a, final double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /* ------------------------------------------------------------------ */
    /*  Swarm mechanics                                                    */
    /* ------------------------------------------------------------------ */

    /**
     * Uniformly distributes initial grasshoppers and identifies the best (target).
     */
    private void initializeSwarm() {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int dims = config.dimensions();

        for (int i = 0; i < config.popSize(); i++) {
            var gh = createRandomGrasshopper(dims, min, max);
            swarm.add(gh);
            if (gh.getFitness() < targetFitness) {
                targetFitness = gh.getFitness();
                targetPosition = gh.getPosition().clone();
            }
        }
        log.debug("Swarm initialized. Start best fitness: {}", targetFitness);
    }

    /**
     * Runs the main iteration loop and returns the best position found.
     *
     * <p>The algorithm proceeds as described in the original paper:
     * <ol>
     *   <li>Update the comfort‑zone coefficient c linearly (Eq. 2.8).</li>
     *   <li>Normalize inter‑grasshopper Euclidean distances to [1, 4]
     *       using the current swarm’s maximum distance.</li>
     *   <li>For each grasshopper, compute the social interaction sum
     *       (Eq. 2.6) and add the target position, then clamp to bounds.</li>
     *   <li>Evaluate and update the target if a better fitness is found.</li>
     * </ol>
     *
     * @return a <em>clone</em> of the target (global best) position
     */
    public double[] optimize() {
        final int dims = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final double f = config.attractionIntensity();
        final double l = config.attractiveLengthScale();
        final double cmax = config.cMax();
        final double cmin = config.cMin();
        final int maxIter = config.maxIterations();
        final int pop = config.popSize();

        for (int t = 1; t <= maxIter; t++) {
            // Adaptive comfort‑zone coefficient (linear decrease, Eq. 2.8)
            double c = cmax - t * (cmax - cmin) / maxIter;

            // -------- Distance normalization (pseudo‑code step "Normalize the distance …") --------
            double maxDist = 0.0;
            for (int i = 0; i < pop; i++) {
                double[] pi = swarm.get(i).getPosition();
                for (int j = i + 1; j < pop; j++) {
                    maxDist = Math.max(maxDist, euclideanDist(pi, swarm.get(j).getPosition()));
                }
            }
            if (maxDist <= 0.0) maxDist = 1.0;   // avoid division by zero

            // Compute new positions for all grasshoppers (all use the same old swarm)
            var newPositions = new double[pop][dims];

            for (int i = 0; i < pop; i++) {
                double[] posI = swarm.get(i).getPosition();
                var sumVector = new double[dims];      // ∑_{j≠i} ...

                for (int j = 0; j < pop; j++) {
                    if (i == j) continue;
                    double[] posJ = swarm.get(j).getPosition();
                    double d_ij = euclideanDist(posI, posJ);
                    if (d_ij < 1e-10) continue;        // identical positions → skip

                    // Normalised distance in [1, 4]
                    double dNorm = 1.0 + (d_ij / maxDist) * 3.0;

                    for (int d = 0; d < dims; d++) {
                        double r = Math.abs(posJ[d] - posI[d]);          // |x_j^d - x_i^d|
                        double s = socialForce(r, f, l);                // s(r)
                        double scale = c * (max[d] - min[d]) / 2.0;     // inner c * (ub-lb)/2
                        sumVector[d] += scale * s * (posJ[d] - posI[d]) / dNorm;
                    }
                }

                // Position update with outer c and target (Eq. 2.6)
                for (int d = 0; d < dims; d++) {
                    newPositions[i][d] = Math.clamp(
                            c * sumVector[d] + targetPosition[d], min[d], max[d]);
                }
            }

            // Apply new positions and update target
            for (int i = 0; i < pop; i++) {
                var gh = swarm.get(i);
                gh.setPosition(newPositions[i]);
                double fit = function.evaluate(newPositions[i]);
                if (!Double.isFinite(fit)) fit = Double.POSITIVE_INFINITY;
                gh.setFitness(fit);

                if (fit < targetFitness) {
                    targetFitness = fit;
                    targetPosition = newPositions[i].clone();
                }
            }

            if (t % 100 == 0 || t == maxIter) {
                log.info("Iteration {}/{}: Best fitness {}, Pop Size: {}",
                        t, maxIter, targetFitness, pop);
            }
        }

        log.info("Optimization finished. Final Best fitness: {}", targetFitness);
        return targetPosition.clone();
    }

    /**
     * Generates a random feasible grasshopper by resampling until a finite fitness
     * is obtained. After {@code MAX_RETRIES} attempts an exception is thrown to
     * avoid infinite loops.
     */
    private Grasshopper createRandomGrasshopper(final int dims,
                                                final double[] min,
                                                final double[] max) {
        final int MAX_RETRIES = 1000;
        for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
            var x = new double[dims];
            for (int d = 0; d < dims; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            double fit = function.evaluate(x);
            if (Double.isFinite(fit)) {
                return new Grasshopper(x, fit);
            }
        }
        throw new IllegalStateException(
                "Failed to generate a feasible grasshopper after " + MAX_RETRIES + " attempts.");
    }

    public double getBestFitness() {
        return targetFitness;
    }

    /* ------------------------------------------------------------------ */
    /*  Embedded types                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Functional interface for the objective function (Fitness / Potential Energy).
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Configuration record for GOA parameters.
     * All arrays are defensively copied and validated in the compact constructor.
     * Constructed through the {@link Builder} to avoid Lombok experimental features.
     */
    public record GoaConfig(
            int popSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            double attractionIntensity,
            double attractiveLengthScale,
            double cMax,
            double cMin,
            long seed
    ) {
        /**
         * Compact constructor validates all parameters and copies array boundaries.
         */
        public GoaConfig {
            Objects.requireNonNull(minBounds, "minBounds must not be null");
            Objects.requireNonNull(maxBounds, "maxBounds must not be null");
            if (popSize < 2) throw new IllegalArgumentException("PopSize must be >= 2");
            if (maxIterations <= 0) throw new IllegalArgumentException("MaxIterations must be > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("Dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bounds length must match dimensions");
            if (attractionIntensity <= 0)
                throw new IllegalArgumentException("attractionIntensity must be > 0");
            if (attractiveLengthScale <= 0)
                throw new IllegalArgumentException("attractiveLengthScale must be > 0");
            if (cMax < cMin) throw new IllegalArgumentException("cMax must be >= cMin");
            if (cMin < 0) throw new IllegalArgumentException("cMin must be >= 0");
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }

        /**
         * Step‑builder to construct an immutable {@code GoaConfig}.
         * Mandatory fields: {@code minBounds} and {@code maxBounds} must be set before
         * calling {@link #build()}.
         */
        public static class Builder {
            private int popSize = 30;
            private int maxIterations = 1000;
            private int dimensions = 5;
            private double[] minBounds;
            private double[] maxBounds;
            private double attractionIntensity = 0.5;
            private double attractiveLengthScale = 1.5;
            private double cMax = 1.0;
            private double cMin = 1e-5;
            private long seed = 42L;

            public Builder popSize(int v) {
                this.popSize = v;
                return this;
            }

            public Builder maxIterations(int v) {
                this.maxIterations = v;
                return this;
            }

            public Builder dimensions(int v) {
                this.dimensions = v;
                return this;
            }

            public Builder minBounds(double[] v) {
                this.minBounds = v.clone();
                return this;
            }

            public Builder maxBounds(double[] v) {
                this.maxBounds = v.clone();
                return this;
            }

            public Builder attractionIntensity(double v) {
                this.attractionIntensity = v;
                return this;
            }

            public Builder attractiveLengthScale(double v) {
                this.attractiveLengthScale = v;
                return this;
            }

            public Builder cMax(double v) {
                this.cMax = v;
                return this;
            }

            public Builder cMin(double v) {
                this.cMin = v;
                return this;
            }

            public Builder seed(long v) {
                this.seed = v;
                return this;
            }

            public GoaConfig build() {
                return new GoaConfig(popSize, maxIterations, dimensions,
                        minBounds, maxBounds, attractionIntensity,
                        attractiveLengthScale, cMax, cMin, seed);
            }
        }
    }

    /**
     * Internal grasshopper representation. Lombok {@code @Data} handles mutable state.
     */
    @Data
    private static class Grasshopper {
        private double[] position;
        private double fitness;

        Grasshopper(final double[] position, final double fitness) {
            this.position = position.clone();
            this.fitness = fitness;
        }
    }

    /**
     * Educational usage example demonstrating the builder pattern.
     */
    public static class GoaExample {
        public static void main() {
            log.info("Starting Grasshopper Optimisation Algorithm Example");

            final ObjectiveFunction sphere = x ->
                    Arrays.stream(x).map(v -> v * v).sum();

            final int dimensions = 10;
            final var min = new double[dimensions];
            final var max = new double[dimensions];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            final var config = new GoaConfig.Builder()
                    .popSize(30)
                    .maxIterations(1000)
                    .dimensions(dimensions)
                    .minBounds(min)
                    .maxBounds(max)
                    .attractionIntensity(0.5)
                    .attractiveLengthScale(1.5)
                    .cMax(1.0)
                    .cMin(1e-5)
                    .seed(42L)
                    .build();

            final var goa = new GrasshopperOptimization(config, sphere);
            final double[] bestSolution = goa.optimize();

            log.info("Best Solution Found: {}", Arrays.toString(bestSolution));
            log.info("Best Objective Value (fitness): {}", goa.getBestFitness());
        }
    }
}