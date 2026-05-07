package edu.swarmintelligence.hho;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * <b>Harris Hawks Optimization (HHO) Algorithm</b> – strict implementation of
 * <i>Heidari et al. “Harris hawks optimization: Algorithm and applications.”
 * Future Generation Computer Systems 97 (2019): 849‑872.</i>
 *
 * <p>
 * A population‑based meta‑heuristic mimicking the cooperative surprise pounce
 * of Harris’s hawks. The algorithm transitions smoothly from global exploration
 * to local exploitation using a linearly decaying <i>escape energy E</i>.
 *
 * <h3>Equations</h3>
 * <ul>
 *   <li><b>Energy:</b> {@code E = 2 · E₀ · (1 – t / T)} where {@code E₀ ∈ (-1,1)}
 *        and t = 0,…,T-1</li>
 *   <li><b>Exploration (|E| ≥ 1):</b> perching on a random hawk <i>or</i> using the
 *       population mean (Eqs. 1‑2).</li>
 *   <li><b>Exploitation (|E| < 1):</b> four conditional strategies combining
 *       soft/hard besiege with or without rapid dives (Eqs. 4‑12).</li>
 *   <li><b>Progressive rapid dives:</b> Lévy‑flight perturbation to escape local optima
 *       (Eqs. 8‑11).</li>
 * </ul>
 *
 * <p>
 * <b>Design:</b> Uses {@link RandomGenerator L128X256MixRandom} for high‑quality
 * randomness. The objective function is a {@link FunctionalInterface}.
 * Configuration is immutable (Lombok {@code @Builder} on a record).
 * Simply instantiate and call {@link #optimise()}.
 *
 * @see <a href="https://doi.org/10.1016/j.future.2019.07.088">Heidari et al. (2019)</a>
 */
@Slf4j
public final class HarrisHawksOptimization {
    private final HhoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final Hawk[] population;
    private final double levySigma;           // Lévy scale σ for β = 1.5

    private double[] rabbitPosition;
    private double rabbitFitness = Double.POSITIVE_INFINITY;

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    /**
     * Creates an HHO optimiser, validates configuration, pre‑computes constants,
     * and initializes a uniformly distributed population.
     *
     * @param config   algorithm hyper‑parameters
     * @param function objective function to minimize
     */
    public HarrisHawksOptimization(final HhoConfig config, final ObjectiveFunction function) {
        this.config = config;
        this.function = function;

        this.random = RandomGeneratorFactory.of("L128X256MixRandom")
                .create(config.seed());
        this.population = new Hawk[config.populationSize()];

        // Pre‑compute Lévy scale parameter σ (Eq. for β = 1.5)
        final double beta = 1.5;
        this.levySigma = Math.pow(
                (gamma(1 + beta) * Math.sin(Math.PI * beta / 2)) /
                        (gamma((1 + beta) / 2) * beta * Math.pow(2, (beta - 1) / 2)),
                1.0 / beta
        );

        initialisePopulation();
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * High‑precision Lanczos approximation for Γ(z).
     */
    private static double gamma(final double z) {
        final double[] p = {
                676.5203681218851, -1259.1392167224028, 771.3234287776531,
                -176.6150291621405, 12.5073432786869, -0.1385710952657201,
                9.984369578019571e-6, 1.5056327351493116e-7
        };
        final double g = 7;
        if (z < 0.5) {
            return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
        }
        double zz = z - 1.0;
        double x = 0.9999999999998099;
        for (int i = 0; i < p.length; i++) {
            x += p[i] / (zz + i + 1.0);
        }
        final double t = zz + g + 0.5;
        return Math.sqrt(2.0 * Math.PI) * Math.pow(t, zz + 0.5) * Math.exp(-t) * x;
    }

    /**
     * Demonstration with the Sphere function (10‑D) using a standard configuration.
     * Usage: run this main method to observe convergence.
     */
    static void main() {
        log.info("Starting Harris Hawks Optimization Demo");

        final ObjectiveFunction sphere = x -> {
            var sum = 0.0;
            for (final double v : x) sum += v * v;
            return sum;
        };

        final int dim = 10;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, -5.12);
        Arrays.fill(maxB, 5.12);

        final HhoConfig cfg = HhoConfig.builder()
                .populationSize(30)
                .maxIterations(1000)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .seed(12345L)
                .build();

        final var hho = new HarrisHawksOptimization(cfg, sphere);
        final double[] solution = hho.optimise();

        log.info("Best Solution: {}", Arrays.toString(solution));
        log.info("Best Fitness:  {}", hho.getBestFitness());
    }

    // -------------------------------------------------------------------------
    // Exploration phase (|E| ≥ 1) – Eqs. 1 & 2
    // -------------------------------------------------------------------------

    /**
     * Runs the full optimization loop.
     *
     * @return the best solution vector found (the rabbit position)
     */
    public double[] optimise() {
        final int maxIter = config.maxIterations();
        final int popSize = config.populationSize();

        // Loop exactly as in original paper: t = 0,…,maxIter-1
        for (int t = 0; t < maxIter; t++) {
            final double e0 = -1.0 + 2.0 * random.nextDouble();          // E₀ ∈ (-1,1)
            final double e = 2.0 * e0 * (1.0 - (double) t / maxIter);  // Eq. (3)
            final double[] xMean = calculateMean();

            for (int i = 0; i < popSize; i++) {
                final Hawk hawk = population[i];
                double[] newPos;

                if (Math.abs(e) >= 1.0) {
                    newPos = performExploration(hawk.position(), xMean);
                } else {
                    newPos = performExploitation(hawk, e, xMean);
                }

                // Enforce search bounds before fitness evaluation
                clamp(newPos);
                final double newFit = function.evaluate(newPos);

                // Greedy update of hawk and rabbit
                if (newFit < hawk.fitness()) {
                    population[i] = new Hawk(newPos, newFit);
                    if (newFit < rabbitFitness) {
                        rabbitFitness = newFit;
                        rabbitPosition = Arrays.copyOf(newPos, config.dimensions());
                    }
                }
            }

            if (log.isInfoEnabled() && t % Math.max(1, maxIter / 10) == 0) {
                log.info("Iteration {}/{} | Best Fitness: {}", t + 1, maxIter, rabbitFitness);
            }
        }
        return rabbitPosition;
    }

    // -------------------------------------------------------------------------
    // Exploitation phase (|E| < 1) – Eqs. 3‑12
    // -------------------------------------------------------------------------

    /**
     * Returns the best fitness value found.
     */
    public double getBestFitness() {
        return rabbitFitness;
    }

    private double[] performExploration(final double[] currentPos, final double[] xMean) {
        final int dim = config.dimensions();
        final double q = random.nextDouble();
        final double[] next = new double[dim];

        if (q >= 0.5) {
            // Eq. (1): random perch on a tall tree
            final double r1 = random.nextDouble();
            final double r2 = random.nextDouble();
            final int randIdx = random.nextInt(config.populationSize()); // any hawk allowed
            final double[] xRand = population[randIdx].position();
            for (int d = 0; d < dim; d++) {
                next[d] = xRand[d] - r1 * Math.abs(xRand[d] - 2.0 * r2 * currentPos[d]);
            }
        } else {
            // Eq. (2): perch based on family members
            final double r3 = random.nextDouble();
            final double r4 = random.nextDouble();
            final double[] minB = config.minBounds();
            final double[] maxB = config.maxBounds();
            for (int d = 0; d < dim; d++) {
                final double range = maxB[d] - minB[d];
                next[d] = (rabbitPosition[d] - xMean[d]) - r3 * (minB[d] + r4 * range);
            }
        }
        return next;
    }

    private double[] performExploitation(final Hawk hawk, final double e, final double[] xMean) {
        final double r = random.nextDouble();                    // r ∈ (0,1)
        final double absE = Math.abs(e);
        final double j = 2.0 * (1.0 - random.nextDouble());     // jump strength J ∈ (0,2)

        if (absE >= 0.5 && r >= 0.5) return softBesiege(hawk.position(), e, j);
        if (absE < 0.5 && r >= 0.5) return hardBesiege(hawk.position(), e);
        if (absE >= 0.5) return softBesiegeWithDives(hawk.position(), e, j, hawk.fitness());
        return hardBesiegeWithDives(hawk.position(), e, j, xMean, hawk.fitness());
    }

    // Eq. (4) – Soft besiege
    private double[] softBesiege(final double[] current, final double e, final double j) {
        final int dim = config.dimensions();
        final double[] next = new double[dim];
        for (int d = 0; d < dim; d++) {
            next[d] = (rabbitPosition[d] - current[d]) -
                    e * Math.abs(j * rabbitPosition[d] - current[d]);
        }
        return next;
    }

    // Eq. (5) – Hard besiege
    private double[] hardBesiege(final double[] current, final double e) {
        final int dim = config.dimensions();
        final double[] next = new double[dim];
        for (int d = 0; d < dim; d++) {
            next[d] = rabbitPosition[d] - e * Math.abs(rabbitPosition[d] - current[d]);
        }
        return next;
    }

    // -------------------------------------------------------------------------
    // Progressive rapid dives – Eqs. 6‑11
    // -------------------------------------------------------------------------

    // Eq. (6) + dive logic (Eqs. 7‑9)
    private double[] softBesiegeWithDives(final double[] current, final double e,
                                          final double j, final double currentFitness) {
        final int dim = config.dimensions();
        final double[] y = new double[dim];
        for (int d = 0; d < dim; d++) {
            y[d] = rabbitPosition[d] - e * Math.abs(j * rabbitPosition[d] - current[d]);
        }
        return rapidDive(y, current, currentFitness);
    }

    // Eq. (8) + dive logic (Eqs. 9‑11)
    private double[] hardBesiegeWithDives(final double[] current, final double e,
                                          final double j, final double[] xMean,
                                          final double currentFitness) {
        final int dim = config.dimensions();
        final double[] y = new double[dim];
        for (int d = 0; d < dim; d++) {
            y[d] = rabbitPosition[d] - e * Math.abs(j * rabbitPosition[d] - xMean[d]);
        }
        return rapidDive(y, current, currentFitness);
    }

    // Common dive selection (Eqs. 7,9 vs. current position)
    private double[] rapidDive(final double[] y, final double[] current,
                               final double currentFitness) {
        clamp(y);
        if (function.evaluate(y) < currentFitness) {
            return y;
        }
        final double[] z = applyLevy(y);
        clamp(z);
        if (function.evaluate(z) < currentFitness) {
            return z;
        }
        return current;   // keep old position
    }

    // -------------------------------------------------------------------------
    // Lévy flight
    // -------------------------------------------------------------------------

    /**
     * Lévy flight perturbation (Eqs. 7,9):
     * {@code Z = Y + S · LF(D)} where
     * {@code LF(D) = 0.01 · (u·σ) / (|v|^(1/β))} with u,v ~ N(0,1)
     * and S is a vector of dimension‑wise uniform (0,1) multipliers.
     */
    private double[] applyLevy(final double[] input) {
        final int dim = config.dimensions();
        final double[] result = new double[dim];
        final double beta = 1.5;
        for (int d = 0; d < dim; d++) {
            final double u = random.nextGaussian();
            final double v = Math.abs(random.nextGaussian());
            final double lfStep = levySigma * u / Math.pow(v, 1.0 / beta);
            final double s = random.nextDouble();        // Sₙ ∈ (0,1)
            result[d] = input[d] + 0.01 * lfStep * s;
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Computes the population mean per dimension.
     */
    private double[] calculateMean() {
        final int dim = config.dimensions();
        final int popSize = config.populationSize();
        final double[] mean = new double[dim];
        for (final Hawk hawk : population) {
            final double[] pos = hawk.position();
            for (int d = 0; d < dim; d++) {
                mean[d] += pos[d];
            }
        }
        final double invPop = 1.0 / popSize;
        for (int d = 0; d < dim; d++) {
            mean[d] *= invPop;
        }
        return mean;
    }

    /**
     * Clamps a position inside the search bounds using {@link Math#clamp} (Java 21).
     */
    private void clamp(final double[] pos) {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        for (int d = 0; d < pos.length; d++) {
            pos[d] = Math.clamp(pos[d], min[d], max[d]);
        }
    }

    // -------------------------------------------------------------------------
    // Population initialization
    // -------------------------------------------------------------------------

    /**
     * Uniform random initialization, records the best solution as rabbit.
     */
    private void initialisePopulation() {
        final int dim = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        for (int i = 0; i < config.populationSize(); i++) {
            final double[] pos = new double[dim];
            for (int d = 0; d < dim; d++) {
                pos[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            final double fit = function.evaluate(pos);
            population[i] = new Hawk(pos, fit);
            if (fit < rabbitFitness) {
                rabbitFitness = fit;
                rabbitPosition = Arrays.copyOf(pos, dim);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Auxiliary types
    // -------------------------------------------------------------------------

    /**
     * Objective function to minimize.
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable algorithm configuration.
     * Use {@link #builder()} with {@code toBuilder = true} to create modified copies.
     */
    @Builder(toBuilder = true)
    public record HhoConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            long seed
    ) {
        public HhoConfig {
            Objects.requireNonNull(minBounds, "minBounds must not be null");
            Objects.requireNonNull(maxBounds, "maxBounds must not be null");
            if (populationSize < 2)
                throw new IllegalArgumentException("Population size must be ≥ 2");
            if (maxIterations <= 0)
                throw new IllegalArgumentException("maxIterations must be positive");
            if (dimensions <= 0)
                throw new IllegalArgumentException("dimensions must be positive");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bound array length must equal dimensions");
        }
    }

    /**
     * Immutable state of a hawk (candidate solution).
     */
    public record Hawk(double[] position, double fitness) {
        public Hawk {
            position = Arrays.copyOf(position, position.length); // defensive copy
        }
    }
}