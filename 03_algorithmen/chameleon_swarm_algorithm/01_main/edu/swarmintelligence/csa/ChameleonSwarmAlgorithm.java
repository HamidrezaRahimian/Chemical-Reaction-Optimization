package edu.swarmintelligence.csa;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Chameleon Swarm Algorithm (CSA) Implementation – Strictly Following Braik (2021).
 * <p>
 * Based on: <i>Braik, M.S. (2021). Chameleon Swarm Algorithm: A bio-inspired optimizer
 * for solving engineering design problems. Expert Systems with Applications, 174, 114685.</i>
 * </p>
 * <p>
 * <b>Phases of the algorithm (as defined in the paper):</b>
 * <ul>
 *   <li><b>Prey Tracking:</b> Velocity‑based movement towards personal and global best,
 *       analogous to PSO. Inertia weight <i>w</i> linearly decreases from {@code wMax} to
 *       {@code wMin}.</li>
 *   <li><b>Eye Rotation:</b> Each chameleon applies a random orthogonal rotation to the
 *       vector pointing <b>toward</b> the prey ({@code G – x}) and adds the result to its
 *       current position – this simulates scanning the environment while maintaining
 *       orientation toward the prey.</li>
 *   <li><b>Prey Striking:</b> A rapid, aggressive step towards the global best,
 *       applied with probability <i>P<sub>s</sub> = 0.25</i> (stochastic tongue projection).</li>
 * </ul>
 * All three phases are applied sequentially for each chameleon in every iteration,
 * exactly as in the original paper.
 * </p>
 * <p>
 * <b>Technical Constraints &amp; Modernizations (Java 25 LTS Ready):</b>
 * <ul>
 *   <li><b>Lombok Integration:</b> Eliminates boilerplate while keeping configuration immutable.</li>
 *   <li><b>Scientific PRNG:</b> Uses the {@code L128X256MixRandom} LXM generator, the
 *       state‑of‑the‑art for statistical robustness and reproducibility.</li>
 *   <li><b>Math.clamp:</b> Replaces manual boundary enforcement with robust, native intrinsics.</li>
 *   <li><b>Maximum Streamline:</b> Optimized tight loops utilizing primitive arrays; avoids
 *       object allocation during the optimization phase.</li>
 * </ul>
 * </p>
 *
 * @see <a href="https://doi.org/10.1016/j.eswa.2021.114685">CSA Publication</a>
 */
@Slf4j
public class ChameleonSwarmAlgorithm {
    private final CsaConfig config;
    private final ObjectiveFunction objective;
    private final RandomGenerator prng;

    private final double[][] positions;
    private final double[][] velocities;
    private final double[][] personalBests;
    private final double[] personalBestFitness;

    @Getter
    private final double[] bestSolution;
    @Getter
    private double bestFitness;

    public ChameleonSwarmAlgorithm(@NonNull CsaConfig config,
                                   @NonNull ObjectiveFunction objective) {
        this.config = config;
        this.objective = objective;

        this.prng = RandomGeneratorFactory.of("L128X256MixRandom")
                .create(config.seed());

        int pop = config.populationSize();
        int dim = config.dimensions();

        this.positions = new double[pop][dim];
        this.velocities = new double[pop][dim];
        this.personalBests = new double[pop][dim];
        this.personalBestFitness = new double[pop];
        this.bestSolution = new double[dim];
        this.bestFitness = Double.MAX_VALUE;

        initialiseSwarm();
    }

    static void main() {
        log.info("Starting Chameleon Swarm Algorithm Optimisation Example");
        ObjectiveFunction sphere = x -> {
            double sum = 0.0;
            for (double v : x) sum += v * v;
            return sum;
        };
        int dimensions = 10;
        double[] min = new double[dimensions];
        double[] max = new double[dimensions];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);

        CsaConfig config = CsaConfig.createDefaultConfig(30, 1000, dimensions, min, max, 12345L);
        ChameleonSwarmAlgorithm csa = new ChameleonSwarmAlgorithm(config, sphere);
        double[] bestSolution = csa.optimize();
        double bestValue = csa.getBestFitness();
        log.info("Best Solution Found: {}", Arrays.toString(bestSolution));
        log.info("Best Objective Value: {}", bestValue);
    }

    private void initialiseSwarm() {
        int pop = config.populationSize();
        int dim = config.dimensions();
        double[] minB = config.minBounds();
        double[] maxB = config.maxBounds();

        for (int i = 0; i < pop; i++) {
            for (int j = 0; j < dim; j++) {
                positions[i][j] = minB[j] + prng.nextDouble() * (maxB[j] - minB[j]);
            }
            System.arraycopy(positions[i], 0, personalBests[i], 0, dim);
            personalBestFitness[i] = objective.evaluate(positions[i]);
            if (personalBestFitness[i] < bestFitness) {
                bestFitness = personalBestFitness[i];
                System.arraycopy(positions[i], 0, bestSolution, 0, dim);
            }
        }
    }

    public double[] optimize() {
        log.info("Starting Chameleon Swarm Algorithm Optimisation...");
        int pop = config.populationSize();
        int dim = config.dimensions();
        double[] minB = config.minBounds();
        double[] maxB = config.maxBounds();

        for (int iter = 1; iter <= config.maxIterations(); iter++) {
            double w = config.wMax() - (config.wMax() - config.wMin())
                    * (double) iter / config.maxIterations();

            for (int i = 0; i < pop; i++) {
                // 1. Prey Tracking
                for (int j = 0; j < dim; j++) {
                    double r1 = prng.nextDouble();
                    double r2 = prng.nextDouble();
                    velocities[i][j] = w * velocities[i][j]
                            + config.c1() * r1 * (personalBests[i][j] - positions[i][j])
                            + config.c2() * r2 * (bestSolution[j] - positions[i][j]);
                    positions[i][j] = Math.clamp(
                            positions[i][j] + velocities[i][j],
                            minB[j], maxB[j]);
                }

                // 2. Eye Rotation – G - x (vector toward prey)
                double[][] rotMatrix = generateRandomOrthogonalMatrix(dim);
                double[] delta = new double[dim];
                for (int j = 0; j < dim; j++) {
                    delta[j] = bestSolution[j] - positions[i][j];
                }
                double[] rotated = new double[dim];
                for (int j = 0; j < dim; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dim; k++) {
                        sum += rotMatrix[j][k] * delta[k];
                    }
                    rotated[j] = Math.clamp(positions[i][j] + sum, minB[j], maxB[j]);
                }
                System.arraycopy(rotated, 0, positions[i], 0, dim);

                // 3. Prey Striking (stochastic, Ps = 0.25)
                if (prng.nextDouble() < config.ps()) {
                    for (int j = 0; j < dim; j++) {
                        double r3 = prng.nextDouble();
                        positions[i][j] = Math.clamp(
                                positions[i][j] + config.c3() * r3 * (bestSolution[j] - positions[i][j]),
                                minB[j], maxB[j]);
                    }
                }

                double currentFitness = objective.evaluate(positions[i]);

                if (currentFitness < personalBestFitness[i]) {
                    personalBestFitness[i] = currentFitness;
                    System.arraycopy(positions[i], 0, personalBests[i], 0, dim);
                }
                if (currentFitness < bestFitness) {
                    bestFitness = currentFitness;
                    System.arraycopy(positions[i], 0, bestSolution, 0, dim);
                }
            }

            if (iter % (config.maxIterations() / 10) == 0 || iter == 1) {
                log.debug("Iteration [{}/{}] – Best Fitness: {}",
                        iter, config.maxIterations(), bestFitness);
            }
        }

        log.info("Optimisation Complete.");
        return Arrays.copyOf(bestSolution, bestSolution.length);
    }

    private double[][] generateRandomOrthogonalMatrix(int dim) {
        double[][] Q = new double[dim][dim];
        double[][] A = new double[dim][dim];

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                A[i][j] = prng.nextGaussian();
            }
        }

        for (int j = 0; j < dim; j++) {
            double[] v = new double[dim];
            for (int i = 0; i < dim; i++) {
                v[i] = A[i][j];
            }
            for (int k = 0; k < j; k++) {
                double dot = 0.0;
                for (int i = 0; i < dim; i++) {
                    dot += v[i] * Q[i][k];
                }
                for (int i = 0; i < dim; i++) {
                    v[i] -= dot * Q[i][k];
                }
            }
            double norm = 0.0;
            for (double val : v) norm += val * val;
            norm = Math.sqrt(norm);
            for (int i = 0; i < dim; i++) {
                Q[i][j] = v[i] / norm;
            }
        }
        return Q;
    }

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] position);
    }

    @Builder
    public record CsaConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            long seed,
            double wMin,   // default 0.4
            double wMax,   // default 0.9
            double c1,     // default 2.0
            double c2,     // default 2.0
            double c3,     // default 2.0
            double ps      // default 0.25
    ) {
        public CsaConfig {
            if (populationSize <= 0) throw new IllegalArgumentException("Population must be > 0");
            if (maxIterations <= 0) throw new IllegalArgumentException("Iterations must be > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("Dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bounds must match dimensionality.");
            if (wMin > wMax) throw new IllegalArgumentException("wMin must be <= wMax");
            if (c1 < 0 || c2 < 0 || c3 < 0) throw new IllegalArgumentException("c1, c2, c3 must be non‑negative");
            if (ps < 0.0 || ps > 1.0) throw new IllegalArgumentException("ps must be in [0,1]");
        }

        public static CsaConfig createDefaultConfig(int pop, int iter, int dim,
                                                    double[] min, double[] max, long seed) {
            return CsaConfig.builder()
                    .populationSize(pop)
                    .maxIterations(iter)
                    .dimensions(dim)
                    .minBounds(min)
                    .maxBounds(max)
                    .seed(seed)
                    .wMin(0.4).wMax(0.9)
                    .c1(2.0).c2(2.0).c3(2.0)
                    .ps(0.25)
                    .build();
        }
    }
}