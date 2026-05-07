package edu.swarmintelligence.fwa;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Fireworks Optimization (FWA) Implementation.
 * <p>
 * Based on: Tan, Y., & Zhu, Y. (2010). Fireworks Algorithm for Optimization.
 * International Conference on Artificial Intelligence and Computational Intelligence.
 * Designed for students in Applied Informatics to study swarm intelligence
 * inspired by fireworks explosions.
 * </p>
 * <p>
 * Technical Architecture:
 * <ul>
 *   <li>Java 25 LTS Ready: Baseline Java 21+, fully leveraging {@code Math.clamp}.</li>
 *   <li>Lombok Driven: Eliminates all getter / setter / logging / builder boilerplate.</li>
 *   <li>LXM PRNG: Uses {@code L128X256MixRandom} via {@code RandomGeneratorFactory} for
 *       statistically robust, scientific‑grade stochastic generation.</li>
 *   <li>Strict Immutability: {@code record} ensures bounds and configuration state cannot leak.</li>
 * </ul>
 * </p>
 * <p>
 * The algorithm generates a swarm of fireworks, each producing a variable number of
 * explosion sparks whose amplitude depends on fitness. Diversity is maintained through
 * Gaussian sparks and distance‑based probabilistic selection.
 * </p>
 *
 * @see <a href="https://doi.org/10.1007/978-3-642-13278-0_52">Tan & Zhu (2010)</a>
 */
@Slf4j
public class FireworksOptimization {
    private final FwaConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final List<Firework> fireworks;
    private double[] globalBestStructure;
    private double globalBestPE = Double.POSITIVE_INFINITY;

    /**
     * Creates a new FWA optimizer with the given configuration and objective.
     * Immediately initializes the firework population and records the global best.
     *
     * @param config   immutable configuration record (built via Lombok builder)
     * @param function the objective function to minimize (potential energy)
     */
    public FireworksOptimization(final FwaConfig config, final ObjectiveFunction function) {
        this.config = Objects.requireNonNull(config);
        this.function = Objects.requireNonNull(function);
        this.random = RandomGeneratorFactory.of("L128X256MixRandom").create(config.seed());
        this.fireworks = new ArrayList<>(config.popSize());
        initializePopulation();
    }

    /* ------------------------------------------------------------------ */
    /*  Phase 0 – Initialisation                                           */
    /* ------------------------------------------------------------------ */

    /**
     * Uniformly distributes initial fireworks and sets the global optimum.
     */
    private void initializePopulation() {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int n = config.dimensions();

        for (int i = 0; i < config.popSize(); i++) {
            var fw = createRandomFirework(n, min, max);
            fireworks.add(fw);
            if (fw.getPotentialEnergy() < globalBestPE) {
                globalBestPE = fw.getPotentialEnergy();
                globalBestStructure = fw.getStructure().clone();
            }
        }
        log.debug("Fireworks initialized. Start Best PE: {}", globalBestPE);
    }

    /* ------------------------------------------------------------------ */
    /*  Main optimisation loop                                             */
    /* ------------------------------------------------------------------ */

    /**
     * Runs the main iteration loop (explosion of fireworks each generation)
     * and returns the best structure found.
     *
     * @return a <em>clone</em> of the global‑best firework structure
     */
    public double[] optimize() {
        final int dimensions = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final double m = config.sparkMultiplier();
        final double aHat = config.explosionAmplitudeConstant();
        final double epsilon = config.epsilon();
        final int g = config.gaussianSparksCount();

        for (int t = 1; t <= config.maxIterations(); t++) {
            // 1. Calculate fitness extremes and denominator
            double fMax = fireworks.stream()
                    .mapToDouble(Firework::getPotentialEnergy)
                    .max().orElseThrow();
            double denom = fireworks.stream()
                    .mapToDouble(fw -> fMax - fw.getPotentialEnergy() + epsilon)
                    .sum();

            // 2. Generate explosion sparks & Gaussian sparks
            var sparks = new ArrayList<Firework>();

            for (var fw : fireworks) {
                double weight = (fMax - fw.getPotentialEnergy() + epsilon) / denom;
                double amplitude = aHat * weight;
                int si = (int) Math.round(m * weight);
                si = clampSparkCount(si, m);

                for (int s = 0; s < si; s++) {
                    sparks.add(generateExplosionSpark(fw, amplitude, dimensions, min, max));
                }
            }

            // Gaussian sparks (diversity maintenance)
            for (int i = 0; i < g; i++) {
                var selectedFw = fireworks.get(random.nextInt(fireworks.size()));
                sparks.add(generateGaussianSpark(selectedFw, dimensions, min, max));
            }

            // 3. Evaluate sparks
            sparks.forEach(spark -> spark.setPotentialEnergy(function.evaluate(spark.getStructure())));

            // 4. Select next generation (elitism + distance‑based roulette)
            selectNextGeneration(sparks, dimensions, min, max);

            // 5. Update global best
            updateGlobalBest();

            if (t % 100 == 0 || t == config.maxIterations()) {
                log.info("Iteration {}/{}: Best PE {}, Pop Size: {}",
                        t, config.maxIterations(), globalBestPE, fireworks.size());
            }
        }

        log.info("Optimization finished. Final Best PE: {}", globalBestPE);
        return globalBestStructure.clone();
    }

    /* ------------------------------------------------------------------ */
    /*  Spark generation                                                   */
    /* ------------------------------------------------------------------ */

    /**
     * Generates an explosion spark by randomly perturbing a subset of dimensions
     * with a uniform displacement in [−amplitude, +amplitude].
     *
     * @param fw         parent firework
     * @param amplitude  explosion amplitude for this firework
     * @param dimensions problem dimension
     * @param min        lower bound
     * @param max        upper bound
     * @return a new unevaluated spark
     */
    private Firework generateExplosionSpark(final Firework fw, final double amplitude,
                                            final int dimensions, final double[] min, final double[] max) {
        var newStructure = fw.getStructure().clone();
        // Number of mutated dimensions: z ∈ [1, dimensions] uniformly
        int z = random.nextInt(dimensions) + 1;
        var dims = random.ints(0, dimensions).distinct().limit(z).toArray();
        for (int d : dims) {
            double displacement = (random.nextDouble() * 2 - 1) * amplitude;
            newStructure[d] = Math.clamp(newStructure[d] + displacement, min[d], max[d]);
        }
        return new Firework(newStructure, Double.POSITIVE_INFINITY);
    }

    /**
     * Generates a Gaussian spark around a selected firework using the distance to
     * the global best as scaling factor.  Formula (paper Eq. 4):
     * {@code x_i^k = x_i^k + g · (x_best^k − x_i^k)},  g ~ N(0,1).
     * Each coordinate has a 50 % chance of being mutated.
     *
     * @param selectedFw randomly chosen firework
     * @param dimensions problem dimension
     * @param min        lower bound
     * @param max        upper bound
     * @return a new unevaluated Gaussian spark
     */
    private Firework generateGaussianSpark(final Firework selectedFw, final int dimensions,
                                           final double[] min, final double[] max) {
        var newStructure = selectedFw.getStructure().clone();
        var best = globalBestStructure;
        for (int d = 0; d < dimensions; d++) {
            if (random.nextBoolean()) {
                double g = random.nextGaussian();                     // N(0,1)
                // x_i^k = x_i^k + g * (x_best^k - x_i^k)
                newStructure[d] = Math.clamp(
                        newStructure[d] + g * (best[d] - newStructure[d]),
                        min[d], max[d]);
            }
        }
        return new Firework(newStructure, Double.POSITIVE_INFINITY);
    }

    /* ------------------------------------------------------------------ */
    /*  Selection of next generation                                       */
    /* ------------------------------------------------------------------ */

    /**
     * Elitism + distance‑based probabilistic selection as described in Tan & Zhu (2010).
     * <p>
     * Complexity O(p³) in the worst case (distance matrix computation for roulette
     * wheel within a loop) – acceptable for the typical population sizes (≤ 100).
     * </p>
     *
     * @param sparks     the explosion and Gaussian sparks of the current generation
     * @param dimensions problem dimension (used only for generation of random refilling)
     * @param min        lower search bounds
     * @param max        upper search bounds
     */
    private void selectNextGeneration(final List<Firework> sparks, final int dimensions,
                                      final double[] min, final double[] max) {
        // Combine all candidates: current fireworks + all sparks
        var candidates = new ArrayList<Firework>(fireworks.size() + sparks.size());
        candidates.addAll(fireworks);
        candidates.addAll(sparks);

        // Discard any individual with non‑finite PE (unlikely but safe)
        candidates.removeIf(fw -> !Double.isFinite(fw.getPotentialEnergy()));

        if (candidates.isEmpty()) {
            // Degenerate case – repopulate with random fireworks
            fireworks.clear();
            for (int i = 0; i < config.popSize(); i++) {
                fireworks.add(createRandomFirework(dimensions, min, max));
            }
            return;
        }

        // Elitism: keep the best individual unconditionally
        var best = candidates.stream()
                .min(Comparator.comparingDouble(Firework::getPotentialEnergy))
                .orElseThrow();
        fireworks.clear();
        fireworks.add(best);
        candidates.remove(best);

        // Distance‑based roulette for the remaining (popSize − 1) fireworks
        int needed = config.popSize() - fireworks.size();
        while (needed > 0 && !candidates.isEmpty()) {
            // Total sum of all pairwise distances (denominator of the selection probability)
            double totalDistance = candidates.parallelStream()
                    .mapToDouble(ind -> candidates.stream()
                            .filter(o -> o != ind)
                            .mapToDouble(o -> euclideanDist(ind.getStructure(), o.getStructure()))
                            .sum())
                    .sum();

            if (totalDistance <= 0) {
                // All candidates are identical – pick randomly from the pool
                fireworks.add(candidates.remove(random.nextInt(candidates.size())));
                needed--;
                continue;
            }

            // Roulette‑wheel selection proportional to distance
            double randTarget = random.nextDouble() * totalDistance;
            double cumulative = 0.0;
            for (int i = 0; i < candidates.size(); i++) {
                var ind = candidates.get(i);
                double distSum = candidates.stream()
                        .filter(o -> o != ind)
                        .mapToDouble(o -> euclideanDist(ind.getStructure(), o.getStructure()))
                        .sum();
                cumulative += distSum;
                if (cumulative >= randTarget) {
                    fireworks.add(candidates.remove(i));
                    needed--;
                    break;
                }
            }
        }

        // Fill remaining slots with random fireworks if necessary
        while (fireworks.size() < config.popSize()) {
            fireworks.add(createRandomFirework(dimensions, min, max));
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Utility methods                                                    */
    /* ------------------------------------------------------------------ */

    /**
     * Clamps the raw spark count to the interval [a·m , b·m] where a and b are
     * the lower / upper bound factors (config parameters).
     */
    private int clampSparkCount(final int rawCount, final double m) {
        int lower = (int) Math.round(config.lowerBoundSparksFactor() * m);
        int upper = (int) Math.round(config.upperBoundSparksFactor() * m);
        return Math.clamp(rawCount, lower, upper);
    }

    private double euclideanDist(final double[] a, final double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Scans the current firework population and updates the global‑best record.
     */
    private void updateGlobalBest() {
        for (var fw : fireworks) {
            if (fw.getPotentialEnergy() < globalBestPE) {
                globalBestPE = fw.getPotentialEnergy();
                globalBestStructure = fw.getStructure().clone();
            }
        }
    }

    /**
     * Generates a random feasible firework by resampling until a finite PE is obtained.
     * After {@code MAX_RETRIES} attempts an exception is thrown to avoid infinite loops.
     */
    private Firework createRandomFirework(final int dimensions, final double[] min, final double[] max) {
        final int MAX_RETRIES = 1000;
        for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
            var x = new double[dimensions];
            for (int d = 0; d < dimensions; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            double pe = function.evaluate(x);
            if (Double.isFinite(pe)) {
                return new Firework(x, pe);
            }
        }
        throw new IllegalStateException(
                "Failed to generate a feasible firework after " + MAX_RETRIES + " attempts.");
    }

    public double getBestPE() {
        return globalBestPE;
    }

    /* ------------------------------------------------------------------ */
    /*  Embedded types                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Functional interface for the objective function (Potential Energy).
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Configuration record for FWA parameters using Lombok Builder.
     * All arrays are defensively copied and validated in the compact constructor.
     */
    @Builder
    public record FwaConfig(
            /* Population size (number of fireworks per generation). */
            int popSize,
            /* Maximum number of generations. */
            int maxIterations,
            /* Problem dimensionality. */
            int dimensions,
            /* Lower bounds for each dimension. */
            double[] minBounds,
            /* Upper bounds for each dimension. */
            double[] maxBounds,
            /*
             * Spark multiplier m. The raw spark count is
             * Sᵢ = round(m · (f_max − fᵢ + ε) / Σ(f_max − fⱼ + ε)).
             */
            double sparkMultiplier,
            /* Number of Gaussian sparks generated each generation. */
            int gaussianSparksCount,
            /*
             * Explosion amplitude constant Â. Amplitude Aᵢ = Â · (f_max − fᵢ + ε) / Σ(…).
             */
            double explosionAmplitudeConstant,
            /*
             * Lower bound factor a for spark count: Sᵢ ≥ round(a · m). Must be > 0.
             */
            double lowerBoundSparksFactor,
            /*
             * Upper bound factor b for spark count: Sᵢ ≤ round(b · m). Must be > a.
             */
            double upperBoundSparksFactor,
            /* Small constant ε to avoid division by zero. */
            double epsilon,
            /* Random seed for reproducibility. */
            long seed
    ) {
        /**
         * Compact constructor validates all parameters and copies array-boundaries.
         */
        public FwaConfig {
            Objects.requireNonNull(minBounds, "minBounds must not be null");
            Objects.requireNonNull(maxBounds, "maxBounds must not be null");
            if (popSize < 2) throw new IllegalArgumentException("PopSize must be >= 2");
            if (maxIterations <= 0) throw new IllegalArgumentException("MaxIterations must be > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("Dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions) {
                throw new IllegalArgumentException("Bounds length must match dimensions");
            }
            if (sparkMultiplier <= 0) throw new IllegalArgumentException("sparkMultiplier must be > 0");
            if (gaussianSparksCount < 0) throw new IllegalArgumentException("gaussianSparksCount must be >= 0");
            if (explosionAmplitudeConstant <= 0)
                throw new IllegalArgumentException("explosionAmplitudeConstant must be > 0");
            if (lowerBoundSparksFactor <= 0) throw new IllegalArgumentException("lowerBoundSparksFactor must be > 0");
            if (upperBoundSparksFactor <= lowerBoundSparksFactor)
                throw new IllegalArgumentException("upperBoundSparksFactor must be > lowerBoundSparksFactor");
            if (epsilon <= 0) throw new IllegalArgumentException("epsilon must be > 0");
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }
    }

    /**
     * Internal firework representation.  Lombok {@code @Data} handles mutable energetic state.
     */
    @Data
    private static class Firework {
        private final double[] structure;
        private double potentialEnergy;

        Firework(final double[] structure, final double potentialEnergy) {
            this.structure = structure.clone();
            this.potentialEnergy = potentialEnergy;
        }
    }

    /**
     * Educational usage example demonstrating the builder pattern.
     */
    public static class FWAExample {
        public static void main() {
            log.info("Starting Fireworks Optimization Example");

            final ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();

            final int dimensions = 10;
            final var min = new double[dimensions];
            final var max = new double[dimensions];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            final var config = FwaConfig.builder()
                    .popSize(20)
                    .maxIterations(1000)
                    .dimensions(dimensions)
                    .minBounds(min)
                    .maxBounds(max)
                    .sparkMultiplier(50)
                    .gaussianSparksCount(5)
                    .explosionAmplitudeConstant(10)
                    .lowerBoundSparksFactor(0.5)
                    .upperBoundSparksFactor(1.5)
                    .epsilon(1e-6)
                    .seed(42L)
                    .build();

            final var fwa = new FireworksOptimization(config, sphere);
            final double[] bestSolution = fwa.optimize();

            log.info("Best Solution Found: {}", Arrays.toString(bestSolution));
            log.info("Best Objective Value (PE): {}", fwa.getBestPE());
        }
    }
}