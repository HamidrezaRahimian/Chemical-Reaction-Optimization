package edu.swarmintelligence.eho;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Elephant Herding Optimization (EHO) Algorithm Implementation.
 * <p>
 * Based on: Wang, G.-G., Deb, S., &amp; Coelho, L. dos S. (2015).
 * Elephant Herding Optimization. 3rd International Symposium on
 * Computational and Business Intelligence (ISCBI), pp. 1–5.
 * </p>
 * <p>
 * The algorithm mimics the herding behavior of elephants:
 * <ol>
 *   <li><strong>Clan updating:</strong> each elephant moves towards the
 *       clan’s matriarch (best solution), while the matriarch herself
 *       moves towards the clan center. A new position is accepted
 *       <strong>only if it improves the elephant’s fitness</strong> –
 *       exactly the greedy selection prescribed in the original paper.</li>
 *   <li><strong>Separating:</strong> the worst elephant in every clan is
 *       replaced by a completely new random individual, introducing
 *       exploration.</li>
 * </ol>
 * </p>
 * <p>
 * <b>Technical Architecture:</b>
 * <ul>
 *   <li>Java 25 LTS Ready: baseline Java 21+, leveraging {@code Math.clamp}.</li>
 *   <li>Lombok Driven: eliminates all getter/setter/logging/builder boilerplate.</li>
 *   <li>LXM PRNG: uses {@code L128X256MixRandom} via
 *       {@code RandomGeneratorFactory} for statistically robust,
 *       scientific‑grade stochastic generation.</li>
 *   <li>Strict Immutability: {@code record} ensures bounds and configuration
 *       cannot leak.</li>
 * </ul>
 * </p>
 *
 * @see <a href="https://doi.org/10.1109/ISCBI.2015.8">Wang et al. (2015)</a>
 */
@Slf4j
public class ElephantHerdingOptimization {
    private final EhoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final List<List<Elephant>> clans;
    private double[] globalBestPosition;
    private double globalBestFitness = Double.POSITIVE_INFINITY;

    /**
     * Creates a new EHO optimizer, immediately initializing the elephant
     * population and recording the global best.
     *
     * @param config   validated, immutable configuration record
     * @param function fitness function to minimize
     */
    public ElephantHerdingOptimization(final EhoConfig config,
                                       final ObjectiveFunction function) {
        this.config = Objects.requireNonNull(config);
        this.function = Objects.requireNonNull(function);
        this.random = RandomGeneratorFactory.of("L128X256MixRandom")
                .create(config.seed());
        this.clans = new ArrayList<>(config.nClan());
        for (int i = 0; i < config.nClan(); i++) {
            clans.add(new ArrayList<>(config.elephantsPerClan()));
        }
        initializePopulation();
    }

    /* ------------------------------------------------------------------ */
    /*  Initialisation                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Randomly initializes all elephants within the search bounds, evenly
     * distributing them among the clans.
     */
    private void initializePopulation() {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int dim = config.dimensions();
        int total = config.nClan() * config.elephantsPerClan();

        for (int i = 0; i < total; i++) {
            Elephant e = createRandomElephant(dim, min, max);
            clans.get(i % config.nClan()).add(e);
            if (e.getFitness() < globalBestFitness) {
                globalBestFitness = e.getFitness();
                globalBestPosition = e.getPosition().clone();
            }
        }
        log.debug("Population initialised. Start best fitness: {}", globalBestFitness);
    }

    /* ------------------------------------------------------------------ */
    /*  Main optimisation loop                                             */
    /* ------------------------------------------------------------------ */

    /**
     * Runs the full EHO optimisation for {@code maxIterations} iterations
     * and returns the best decision vector found.
     *
     * @return a clone of the best‑known position
     */
    public double[] optimize() {
        final int dim = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        for (int iter = 1; iter <= config.maxIterations(); iter++) {
            // ---- 1. Clan updating operator (with greedy selection) ----
            for (var clan : clans) {
                Elephant matriarch = findBestInClan(clan);
                double[] bestPos = matriarch.getPosition();
                double[] center = computeClanCenter(clan);

                for (var elephant : clan) {
                    double[] newPos = (elephant == matriarch)
                            ? matriarchMove(center, dim, min, max)
                            : followerMove(elephant.getPosition(), bestPos, dim, min, max);
                    double newFit = function.evaluate(newPos);
                    // Accept only if the new position is strictly better
                    if (Double.isFinite(newFit) && newFit < elephant.getFitness()) {
                        elephant.setPosition(newPos);
                        elephant.setFitness(newFit);
                    }
                }
            }

            // ---- 2. Separating operator ----
            for (var clan : clans) {
                int worstIdx = findWorstIndex(clan);
                Elephant fresh = createRandomElephant(dim, min, max);
                clan.set(worstIdx, fresh);
            }

            updateGlobalBest();

            if (iter % 100 == 0 || iter == config.maxIterations()) {
                log.info("Iteration {}/{}: Best fitness {}",
                        iter, config.maxIterations(), globalBestFitness);
            }
        }

        log.info("Optimisation finished. Final best fitness: {}", globalBestFitness);
        return globalBestPosition.clone();
    }

    /* ------------------------------------------------------------------ */
    /*  Movement helpers (exactly as in the original paper)                */
    /* ------------------------------------------------------------------ */

    /**
     * Matriarch movement: x<sub>new</sub> = β · x<sub>center</sub>.
     */
    private double[] matriarchMove(final double[] center, final int dim,
                                   final double[] min, final double[] max) {
        var pos = new double[dim];
        double beta = config.beta();
        for (int d = 0; d < dim; d++) {
            pos[d] = Math.clamp(beta * center[d], min[d], max[d]);
        }
        return pos;
    }

    /**
     * Follower movement: x<sub>new</sub>[d] = x[d] + α · (x<sub>best</sub>[d] − x[d]) · r,
     * with r ∈ [0,1].
     */
    private double[] followerMove(final double[] current, final double[] best,
                                  final int dim, final double[] min, final double[] max) {
        var pos = new double[dim];
        double alpha = config.alpha();
        for (int d = 0; d < dim; d++) {
            pos[d] = Math.clamp(
                    current[d] + alpha * (best[d] - current[d]) * random.nextDouble(),
                    min[d], max[d]);
        }
        return pos;
    }

    /* ------------------------------------------------------------------ */
    /*  Clan‑level helpers                                                 */
    /* ------------------------------------------------------------------ */

    /**
     * Returns the elephant with the lowest fitness in a clan.
     */
    private Elephant findBestInClan(final List<Elephant> clan) {
        Elephant best = clan.getFirst();
        for (int i = 1; i < clan.size(); i++) {
            if (clan.get(i).getFitness() < best.getFitness()) {
                best = clan.get(i);
            }
        }
        return best;
    }

    /**
     * Computes the arithmetic center (mean position) of a clan.
     */
    private double[] computeClanCenter(final List<Elephant> clan) {
        int dim = config.dimensions();
        var center = new double[dim];
        for (var e : clan) {
            var pos = e.getPosition();
            for (int d = 0; d < dim; d++) {
                center[d] += pos[d];
            }
        }
        for (int d = 0; d < dim; d++) {
            center[d] /= clan.size();
        }
        return center;
    }

    /**
     * Returns the index of the elephant with the highest fitness (worst).
     */
    private int findWorstIndex(final List<Elephant> clan) {
        int worst = 0;
        double worstFit = clan.getFirst().getFitness();
        for (int i = 1; i < clan.size(); i++) {
            if (clan.get(i).getFitness() > worstFit) {
                worstFit = clan.get(i).getFitness();
                worst = i;
            }
        }
        return worst;
    }

    /**
     * Updates the global‑best position and fitness across all clans.
     */
    private void updateGlobalBest() {
        for (var clan : clans) {
            for (var e : clan) {
                if (e.getFitness() < globalBestFitness) {
                    globalBestFitness = e.getFitness();
                    globalBestPosition = e.getPosition().clone();
                }
            }
        }
    }

    /**
     * Creates a random feasible elephant, retrying until a finite fitness is
     * obtained (up to a maximum number of attempts).
     */
    private Elephant createRandomElephant(final int dim, final double[] min,
                                          final double[] max) {
        final int MAX_RETRIES = 1000;
        for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
            var pos = new double[dim];
            for (int d = 0; d < dim; d++) {
                pos[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            double fit = function.evaluate(pos);
            if (Double.isFinite(fit)) {
                return new Elephant(dim, pos, fit);
            }
        }
        throw new IllegalStateException(
                "Failed to generate a feasible elephant after " + MAX_RETRIES + " attempts.");
    }

    /**
     * Returns the current best fitness (for monitoring/testing).
     */
    public double getBestFitness() {
        return globalBestFitness;
    }

    /* ------------------------------------------------------------------ */
    /*  Embedded types                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Fitness function to be minimized.
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Configuration record for EHO parameters, built via Lombok.
     */
    @Builder
    public record EhoConfig(
            /* Number of clans (sub‑groups) */
            int nClan,
            /* Number of elephants per clan */
            int elephantsPerClan,
            /* Total number of iterations */
            int maxIterations,
            /* Problem dimensionality */
            int dimensions,
            /* Lower bounds per dimension (inclusive) */
            double[] minBounds,
            /* Upper bounds per dimension (inclusive) */
            double[] maxBounds,
            /* Influence factor of the matriarch (α > 0) */
            double alpha,
            /* Influence factor of the clan center (β > 0) */
            double beta,
            /* Seed for reproducible randomness */
            long seed
    ) {
        /**
         * Compact constructor validates all parameters and copies array‑bounds.
         */
        public EhoConfig {
            Objects.requireNonNull(minBounds, "minBounds must not be null");
            Objects.requireNonNull(maxBounds, "maxBounds must not be null");
            if (nClan <= 0)
                throw new IllegalArgumentException("nClan must be > 0");
            if (elephantsPerClan <= 0)
                throw new IllegalArgumentException("elephantsPerClan must be > 0");
            if (maxIterations <= 0)
                throw new IllegalArgumentException("maxIterations must be > 0");
            if (dimensions <= 0)
                throw new IllegalArgumentException("dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bounds length must match dimensions");
            if (alpha <= 0)
                throw new IllegalArgumentException("alpha must be > 0");
            if (beta <= 0)
                throw new IllegalArgumentException("beta must be > 0");

            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }
    }

    /**
     * Internal representation of a single elephant.
     * Mutable for performance; position and fitness are updated in‑place.
     */
    @Data
    private static class Elephant {
        private double[] position;
        private double fitness;

        Elephant(final int dimensions, final double[] position, final double fitness) {
            this.position = position.clone();   // defensive copy
            this.fitness = fitness;
        }
    }

    /**
     * Educational usage example demonstrating the Lombok builder and a
     * simple sphere‑function minimization.
     */
    @Slf4j
    public static class EHOExample {
        public static void main() {
            log.info("Starting EHO Optimisation Example");

            // Sphere function minimization
            ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();

            int dimensions = 10;
            var min = new double[dimensions];
            var max = new double[dimensions];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            var config = EhoConfig.builder()
                    .nClan(5)
                    .elephantsPerClan(10)
                    .maxIterations(1000)
                    .dimensions(dimensions)
                    .minBounds(min)
                    .maxBounds(max)
                    .alpha(0.5)
                    .beta(0.1)
                    .seed(12345L)
                    .build();

            var eho = new ElephantHerdingOptimization(config, sphere);
            double[] best = eho.optimize();

            log.info("Best solution found: {}", Arrays.toString(best));
            log.info("Best fitness: {}", eho.getBestFitness());
        }
    }
}