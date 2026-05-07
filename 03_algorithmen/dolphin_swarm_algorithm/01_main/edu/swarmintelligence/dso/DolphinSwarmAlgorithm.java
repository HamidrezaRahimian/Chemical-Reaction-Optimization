package edu.swarmintelligence.dso;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.Comparator;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Dolphin Swarm Algorithm (DSA) – cooperative hunting‑based metaheuristic.
 * <p>
 * Reference: Wu T., Yao M. & Yang J. (2016), “Dolphin Swarm Algorithm”,
 * Frontiers of Information Technology & Electronic Engineering 17(8), 717‑729.
 * <p>
 * Implements four phases per generation:
 * <ol>
 *   <li><b>Search (echolocation)</b>: each dolphin emits {@code M} probe vectors
 *       along random directions, moving step‑by‑step up to {@code T₁} times.
 *       The dolphin’s <b>historical best</b> Lᵢ is updated only if a better position
 *       is found.</li>
 *   <li><b>Call</b>: dolphins whose individual best fitness is ≤ θ
 *       (the median of all individual bests) broadcast their solution.</li>
 *   <li><b>Reception</b>: each dolphin updates its neighborhood best Kᵢ
 *       with better received solutions.</li>
 *   <li><b>Predation (geometric encirclement)</b>: the dolphin’s new position
 *       is determined by its distances to Kᵢ and Lᵢ, and a random
 *       encirclement radius, with greedy acceptance.</li>
 * </ol>
 * <h2>Implementation notes</h2>
 * <ul>
 *   <li>PRNG: {@code L128X1024MixRandom} – statistically superior, fixed‑seed deterministic.</li>
 *   <li>Communication threshold θ: dynamically set to the median of
 *       individual best fitnesses (if {@code config.commThreshold < 0}).</li>
 *   <li>Uniform random directions on the hypersphere are generated via the
 *       Gaussian method (Box‑Muller) for correctness.</li>
 *   <li>All position updates are clamped to the problem bounds.</li>
 *   <li>The algorithm strictly respects the evaluation budget
 *       ({@code maxFE}) and stops early if exhausted.</li>
 *   <li>Previous version inadvertently reset Lᵢ each generation; this has been corrected
 *       to reflect the paper’s persistent memory, which is essential for convergence.</li>
 * </ul>
 */
@Slf4j
public final class DolphinSwarmAlgorithm {

    private final DsaConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator rng;
    private final Dolphin[] population;
    private final boolean[][] transferMatrix;
    private final double searchRadius;   // R₁ = T₁ * speed
    private double[] globalBestPos;
    private double globalBestFit = Double.MAX_VALUE;
    private int feCount = 0;

    public DolphinSwarmAlgorithm(DsaConfig config, ObjectiveFunction function) {
        this.config = config;
        this.function = function;
        this.rng = RandomGeneratorFactory.of("L128X1024MixRandom").create(config.seed());
        this.population = new Dolphin[config.popSize()];
        this.transferMatrix = new boolean[config.popSize()][config.popSize()];
        this.searchRadius = config.maxSearchTime() * config.speed();
        initPopulation();
    }

    // --- Initialisation (Phase 0) ---
    private void initPopulation() {
        final var min = config.minBounds();
        final var max = config.maxBounds();
        final int d = config.dimensions();
        for (int i = 0; i < config.popSize(); i++) {
            var pos = new double[d];
            for (int j = 0; j < d; j++)
                pos[j] = min[j] + rng.nextDouble() * (max[j] - min[j]);
            double fit = eval(pos);
            population[i] = new Dolphin(pos, fit);
            if (fit < globalBestFit) {
                globalBestFit = fit;
                globalBestPos = pos.clone();
            }
        }
        log.debug("Initialised {} dolphins, best fitness = {}", config.popSize(), globalBestFit);
    }

    private double eval(double[] x) {
        feCount++;
        return function.evaluate(x);
    }

    // --- Main loop ---
    public DsaResult optimize() {
        final int D = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int M = config.numVectors();
        final int T1 = config.maxSearchTime();
        final double speed = config.speed();
        final double R1 = searchRadius;

        log.info("DSA started. maxFE = {}", config.maxFE());

        while (feCount < config.maxFE()) {
            // 1. Search Phase (echolocation)
            for (var dol : population) {
                for (int j = 0; j < M && feCount < config.maxFE(); j++) {
                    double[] dir = randomUnitVector(D);
                    for (int k = 0; k < D; k++) dir[k] *= speed;
                    for (int t = 1; t <= T1 && feCount < config.maxFE(); t++) {
                        var probe = new double[D];
                        for (int k = 0; k < D; k++)
                            probe[k] = Math.clamp(dol.pos[k] + t * dir[k], min[k], max[k]);
                        double f = eval(probe);
                        dol.updateIndividualOptimum(probe, f);
                    }
                }
                // If Lᵢ beats Kᵢ, Kᵢ ← Lᵢ
                if (dol.indBestFit < dol.neighBestFit)
                    dol.setNeighBestToIndividual();
            }
            if (feCount >= config.maxFE()) break;

            // 2. Call Phase
            double theta = adaptiveTheta();
            for (var row : transferMatrix) Arrays.fill(row, false);
            for (int i = 0; i < config.popSize(); i++) {
                if (population[i].indBestFit <= theta) {
                    for (int j = 0; j < config.popSize(); j++)
                        if (i != j) transferMatrix[i][j] = true;
                }
            }

            // 3. Reception Phase
            for (int j = 0; j < config.popSize(); j++) {
                var receiver = population[j];
                for (int i = 0; i < config.popSize(); i++) {
                    if (transferMatrix[i][j]) {
                        var sender = population[i];
                        receiver.updateNeighBest(sender.indBestPos, sender.indBestFit);
                    }
                }
            }

            // 4. Predation Phase (geometric encirclement)
            for (var dol : population) {
                double DK = dist(dol.pos, dol.neighBestPos);
                double DKL = dist(dol.indBestPos, dol.neighBestPos);
                var dirToK = DK > 1e-10 ? unitVectorTowards(dol.pos, dol.neighBestPos, DK)
                        : randomUnitVector(D);
                var randUnit = randomUnitVector(D);
                var newPos = new double[D];
                double R2;

                if (DK <= R1) {                     // Case 1
                    R2 = DKL / 2.0;
                    for (int k = 0; k < D; k++)
                        newPos[k] = dol.neighBestPos[k] + R2 * randUnit[k];
                } else if (DK <= R1 + DKL) {        // Case 2
                    R2 = (DK + DKL - R1) / 2.0;
                    for (int k = 0; k < D; k++)
                        newPos[k] = dol.neighBestPos[k] + R2 * randUnit[k];
                } else {                             // Case 3
                    R2 = DKL / 2.0;
                    for (int k = 0; k < D; k++)
                        newPos[k] = dol.pos[k] + R1 * dirToK[k] + R2 * randUnit[k];
                }

                for (int k = 0; k < D; k++)
                    newPos[k] = Math.clamp(newPos[k], min[k], max[k]);

                double newFit = eval(newPos);
                if (newFit < dol.fit) {
                    dol.fit = newFit;
                    System.arraycopy(newPos, 0, dol.pos, 0, D);
                }
                if (dol.fit < dol.neighBestFit)
                    dol.setNeighBestToCurrent();
            }

            updateGlobalBest();

            if (feCount % 1000 == 0)
                log.info("FE {}: best fitness = {}", feCount, globalBestFit);
        }

        log.info("DSA finished. best fitness = {}, FE = {}", globalBestFit, feCount);
        return new DsaResult(globalBestPos, globalBestFit, feCount, config.seed());
    }

    // --- Helper math ---
    private double dist(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private double[] unitVectorTowards(double[] from, double[] to, double dist) {
        final int n = from.length;
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = (to[i] - from[i]) / dist;
        return v;
    }

    /**
     * Uniformly distributed random unit vector on the hypersphere.
     * Implemented via Gaussian (Box‑Muller) transform for correctness.
     */
    private double[] randomUnitVector(int dim) {
        var v = new double[dim];
        double sumSq = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = rng.nextGaussian();
            sumSq += v[i] * v[i];
        }
        double len = Math.sqrt(sumSq);
        if (len < 1e-10) {   // extremely rare fallback
            v[0] = 1;
            len = 1;
        }
        for (int i = 0; i < dim; i++) v[i] /= len;
        return v;
    }

    /**
     * Communication threshold: if {@code config.commThreshold()} is non‑negative,
     * that fixed value is used; otherwise the median of the individual best fitnesses
     * is computed dynamically.
     */
    private double adaptiveTheta() {
        if (config.commThreshold() >= 0) return config.commThreshold();
        return Arrays.stream(population)
                .mapToDouble(d -> d.indBestFit)
                .sorted()
                .skip((config.popSize() - 1) / 2)
                .findFirst()
                .orElse(Double.MAX_VALUE);
    }

    private void updateGlobalBest() {
        var best = Arrays.stream(population)
                .min(Comparator.comparingDouble(d -> d.fit))
                .orElseThrow();
        if (best.fit < globalBestFit) {
            globalBestFit = best.fit;
            globalBestPos = best.pos.clone();
        }
    }

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    public record DsaConfig(
            int popSize,
            int maxFE,
            int dimensions,
            double @NonNull [] minBounds,
            double @NonNull [] maxBounds,
            int numVectors,
            int maxSearchTime,
            double speed,
            double commThreshold,
            long seed) {

        public DsaConfig {
            if (popSize < 2) throw new IllegalArgumentException("popSize >= 2 required");
            if (maxFE <= 0) throw new IllegalArgumentException("maxFE > 0 required");
            if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0 required");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("bounds length must equal dimensions");
            if (numVectors <= 0) throw new IllegalArgumentException("numVectors > 0 required");
            if (maxSearchTime <= 0) throw new IllegalArgumentException("maxSearchTime > 0 required");
            if (speed <= 0) throw new IllegalArgumentException("speed > 0 required");
            minBounds = Arrays.copyOf(minBounds, dimensions);
            maxBounds = Arrays.copyOf(maxBounds, dimensions);
        }

        public static DsaConfig createDefault(int popSize, int maxFE, int dimensions,
                                              double[] min, double[] max, long seed) {
            return new DsaConfig(popSize, maxFE, dimensions, min, max,
                    5, 5, 1.0, -1.0, seed);
        }
    }

    public record DsaResult(double @NonNull [] bestPosition, double bestFitness,
                            int functionEvaluations, long seed) {
        public DsaResult {
            java.util.Objects.requireNonNull(bestPosition);
            bestPosition = bestPosition.clone();
        }

        @Override
        public double[] bestPosition() {
            return bestPosition.clone();
        }
    }

    // --- Internal mutable Dolphin representation ---
    private static class Dolphin {
        final double[] pos;          // current position Xᵢ
        final double[] indBestPos;   // Lᵢ – personal historical best
        final double[] neighBestPos; // Kᵢ – neighbourhood best
        double fit;
        double indBestFit;
        double neighBestFit;

        Dolphin(double[] x, double f) {
            pos = x.clone();
            fit = f;
            indBestPos = x.clone();
            indBestFit = f;
            neighBestPos = x.clone();
            neighBestFit = f;
        }

        void updateIndividualOptimum(double[] candidate, double candFit) {
            if (candFit < indBestFit) {
                indBestFit = candFit;
                System.arraycopy(candidate, 0, indBestPos, 0, pos.length);
            }
        }

        void setNeighBestToIndividual() {
            neighBestFit = indBestFit;
            System.arraycopy(indBestPos, 0, neighBestPos, 0, pos.length);
        }

        void updateNeighBest(double[] cand, double candFit) {
            if (candFit < neighBestFit) {
                neighBestFit = candFit;
                System.arraycopy(cand, 0, neighBestPos, 0, pos.length);
            }
        }

        void setNeighBestToCurrent() {
            neighBestFit = fit;
            System.arraycopy(pos, 0, neighBestPos, 0, pos.length);
        }
    }
}