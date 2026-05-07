package edu.swarmintelligence.zoa;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Zebra Optimization Algorithm (ZOA) – strict implementation.
 * <p>
 * Based on: Trojovská, E., Dehghani, M. & Trojovský, P. (2022).
 * Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm.
 * IEEE Access, 10, 49445–49473.
 * <a href="https://doi.org/10.1109/ACCESS.2022.3170366">DOI</a>
 * </p>
 *
 * <h3>Algorithm Overview</h3>
 * <ol>
 *   <li><b>Phase 1 – Foraging:</b> Movement towards the pioneer zebra (best solution).</li>
 *   <li><b>Phase 2 – Defense:</b> Dual‑strategy predator avoidance (zigzag escape or protective grouping).</li>
 * </ol>
 * Each zebra performs both phases in sequence, with greedy acceptance after each phase.
 * Total function evaluations per iteration = 2 × population size.
 *
 * <h3>Technical Constraints</h3>
 * <ul>
 *   <li>Java 25 LTS – stable features only.</li>
 *   <li>L128X256MixRandom for scientific‑grade reproducibility.</li>
 *   <li>Math.clamp() for optimal boundary enforcement.</li>
 *   <li>Lombok for logging and boilerplate reduction.</li>
 * </ul>
 *
 * @see <a href="https://doi.org/10.1109/ACCESS.2022.3170366">IEEE Access paper</a>
 */
@Slf4j
public class ZebraOptimizationAlgorithm {
    /**
     * Escape constant R – fixed per the original paper.
     */
    private static final double ESCAPE_CONSTANT = 0.01;

    private final ZoaConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final Zebra[] population;
    private double[] pioneerZebraPosition;

    @Getter
    private double pioneerZebraFitness;

    /**
     * Constructs the ZOA optimizer.
     *
     * @param config   algorithm configuration (population, iterations, bounds, seed)
     * @param function objective function to minimize
     */
    public ZebraOptimizationAlgorithm(ZoaConfig config, ObjectiveFunction function) {
        this.config = config;
        this.function = function;
        this.random = RandomGeneratorFactory.of("L128X256MixRandom").create(config.seed());
        this.population = new Zebra[config.populationSize()];
        this.pioneerZebraFitness = Double.MAX_VALUE;
        this.pioneerZebraPosition = null;

        initializePopulation();
    }

    /**
     * Phase 0: Random initialization within bounds.
     * X_i = x_min + r * (x_max - x_min) for each dimension.
     */
    private void initializePopulation() {
        double[] min = config.minBounds();
        double[] max = config.maxBounds();
        int n = config.dimensions();

        for (int i = 0; i < config.populationSize(); i++) {
            double[] x = new double[n];
            Arrays.setAll(x, d -> min[d] + random.nextDouble() * (max[d] - min[d]));
            double fVal = function.evaluate(x);
            population[i] = new Zebra(x, fVal);

            if (fVal < pioneerZebraFitness) {
                pioneerZebraFitness = fVal;
                pioneerZebraPosition = Arrays.copyOf(x, n);
            }
        }
        log.debug("Population initialized. Size: {}, Initial PZ Fitness: {}",
                config.populationSize(), pioneerZebraFitness);
    }

    /**
     * Executes the dual-phase optimization process.
     *
     * @return best found position (pioneer zebra)
     */
    public double[] optimize() {
        int maxIter = config.maxIterations();
        log.info("Starting ZOA. Population: {}, Max Iterations: {}", config.populationSize(), maxIter);

        for (int t = 1; t <= maxIter; t++) {
            runIteration(t);
            if (t % 100 == 0) {
                log.info("Iteration {}: PZ Fitness {}", t, pioneerZebraFitness);
            }
        }

        log.info("Optimization finished. Final PZ Fitness: {}", pioneerZebraFitness);
        return Arrays.copyOf(pioneerZebraPosition, pioneerZebraPosition.length);
    }

    /**
     * One full iteration (foraging + defense for all zebras, then updates pioneer).
     * Exposed for stepwise execution in testing/education.
     *
     * @param currentIteration current iteration number (1‑indexed)
     */
    public void runIteration(int currentIteration) {
        int n = config.dimensions();
        double[] min = config.minBounds();
        double[] max = config.maxBounds();
        int popSize = config.populationSize();

        for (int i = 0; i < popSize; i++) {
            Zebra zebra = population[i];

            // Phase 1: Foraging
            double[] foragingPos = performForaging(zebra, n, min, max);
            acceptIfBetter(zebra, foragingPos, n);

            // Phase 2: Defense (immediately after foraging)
            double[] defensePos = performDefense(zebra, i, currentIteration, n, min, max);
            acceptIfBetter(zebra, defensePos, n);
        }

        updatePioneerZebra();
    }

    /**
     * Greedy acceptance: replace if new position is better.
     */
    private void acceptIfBetter(Zebra zebra, double[] newPosition, int dimensions) {
        double newFitness = function.evaluate(newPosition);
        if (newFitness < zebra.getFitness()) {
            zebra.setFitness(newFitness);
            System.arraycopy(newPosition, 0, zebra.getPosition(), 0, dimensions);
        }
    }

    /**
     * Phase 1 – Foraging to Pioneer Zebra.
     * Formula: x_new[i,j] = x[i,j] + r * (PZ[j] - I * x[i,j])
     * r ~ U(0,1) is drawn per dimension, I ∈ {1,2} (one draw per zebra).
     */
    private double[] performForaging(Zebra zebra, int dimensions, double[] min, double[] max) {
        double[] newPosition = new double[dimensions];
        double[] zebraPos = zebra.getPosition();
        int I = random.nextInt(2) + 1;                      // I ∈ {1,2}

        Arrays.setAll(newPosition, d -> Math.clamp(
                zebraPos[d] + random.nextDouble() * (pioneerZebraPosition[d] - I * zebraPos[d]),
                min[d], max[d]));
        return newPosition;
    }

    /**
     * Phase 2 – Dual‑strategy Predator Avoidance.
     * <ul>
     *   <li><b>S1 (Escape):</b>   x_new = x + R * (2r - 1) * (1 - t/T) * x</li>
     *   <li><b>S2 (Group):</b>    x_new = x + r * (AZ[j] - I * x[j])</li>
     * </ul>
     * r per dimension, I ∈ {1,2} per zebra, AZ = randomly selected, different zebra.
     */
    private double[] performDefense(Zebra zebra, int currentIndex, int currentIteration,
                                    int dimensions, double[] min, double[] max) {
        double[] newPosition = new double[dimensions];
        double[] zebraPos = zebra.getPosition();

        if (random.nextDouble() <= 0.5) {
            // Strategy S1: Zigzag Escape
            double timeFactor = 1.0 - (double) currentIteration / config.maxIterations();

            Arrays.setAll(newPosition, d -> Math.clamp(
                    zebraPos[d] + ESCAPE_CONSTANT * (2.0 * random.nextDouble() - 1.0) * timeFactor * zebraPos[d],
                    min[d], max[d]));
        } else {
            // Strategy S2: Protective Group
            int attackedIdx = selectRandomZebraIndex(currentIndex);
            double[] attackedZebraPos = population[attackedIdx].getPosition();
            int I = random.nextInt(2) + 1;                  // I ∈ {1,2} per zebra

            Arrays.setAll(newPosition, d -> Math.clamp(
                    zebraPos[d] + random.nextDouble() * (attackedZebraPos[d] - I * zebraPos[d]),
                    min[d], max[d]));
        }
        return newPosition;
    }

    /**
     * Selects a random zebra index different from the current one.
     * If population size is 1, returns the same index (no alternative).
     */
    private int selectRandomZebraIndex(int currentIndex) {
        int popSize = config.populationSize();
        if (popSize <= 1) return currentIndex;

        int randomIdx;
        do {
            randomIdx = random.nextInt(popSize);
        } while (randomIdx == currentIndex);
        return randomIdx;
    }

    /**
     * Updates the pioneer zebra (global best) based on the current population.
     */
    private void updatePioneerZebra() {
        for (Zebra zebra : population) {
            if (zebra.getFitness() < pioneerZebraFitness) {
                pioneerZebraFitness = zebra.getFitness();
                pioneerZebraPosition = zebra.getClonedPosition();
            }
        }
    }

    /**
     * Total function evaluations for a given number of completed iterations.
     * 2N evaluations per iteration.
     */
    public long getTotalFunctionEvaluations(int iterationsCompleted) {
        return (long) iterationsCompleted * config.populationSize() * 2L;
    }

    /**
     * <b>Package‑private accessor for testing only.</b><br>
     * Returns the direct internal reference to the pioneer zebra position.
     * <strong>Do not mutate the returned array – it will corrupt the algorithm state.</strong>
     */
    double[] getPioneerPositionRaw() {
        return pioneerZebraPosition;
    }

    /**
     * Objective function to be minimized.
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Configuration record for ZOA.
     * Immutable, with validation and defensive copies of bounds arrays.
     */
    public record ZoaConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            long seed
    ) {
        public ZoaConfig {
            if (populationSize <= 0) throw new IllegalArgumentException("PopulationSize must be > 0");
            if (maxIterations <= 0) throw new IllegalArgumentException("MaxIterations must be > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("Dimensions must be > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions) {
                throw new IllegalArgumentException("Bounds must match dimensions");
            }
            // Defensive copy to preserve immutability
            minBounds = Arrays.copyOf(minBounds, dimensions);
            maxBounds = Arrays.copyOf(maxBounds, dimensions);
        }

        /**
         * Convenience factory method.
         */
        public static ZoaConfig createDefaultConfig(int populationSize, int maxIterations,
                                                    int dimensions, double[] min, double[] max, long seed) {
            return new ZoaConfig(populationSize, maxIterations, dimensions, min, max, seed);
        }
    }

    /**
     * Internal representation of a Zebra individual.
     * Mutable for performance – note that ZOA has no individual memory (no personal best like PSO).
     */
    @Getter
    @Setter
    @AllArgsConstructor
    private static class Zebra {
        private final double[] position;   // mutable contents, reference is final
        private double fitness;

        /**
         * Returns a safe copy of the current position.
         */
        public double[] getClonedPosition() {
            return Arrays.copyOf(position, position.length);
        }
    }

    /**
     * Educational example: minimization of the Sphere function.
     */
    @Slf4j
    public static class ZOAExample {
        static void main() {
            log.info("Starting ZOA Optimization Example");

            // Sphere function f(x) = sum(x_i^2), global minimum at x = 0
            ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();

            int dimensions = 10;
            double[] min = new double[dimensions];
            double[] max = new double[dimensions];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            ZoaConfig config = ZoaConfig.createDefaultConfig(30, 1000, dimensions, min, max, 12345L);
            ZebraOptimizationAlgorithm zoa = new ZebraOptimizationAlgorithm(config, sphere);
            double[] bestSolution = zoa.optimize();

            log.info("Best Solution: {}", Arrays.toString(bestSolution));
            log.info("Best Objective Value: {}", sphere.evaluate(bestSolution));
            log.info("Total Function Evaluations: {}", zoa.getTotalFunctionEvaluations(config.maxIterations()));
        }
    }
}