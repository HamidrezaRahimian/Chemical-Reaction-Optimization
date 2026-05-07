 
 
 
 
 
 
package edu.swarmintelligence.pso.r01;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Particle Swarm Optimization (PSO) – global‑best topology.
 *
 * <p>References:
 * <ul>
 *   <li>Kennedy, J. &amp; Eberhart, R. (1995). <i>Particle Swarm Optimization</i>.
 *       Proc. ICNN'95, pp. 1942–1948.</li>
 *   <li>Shi, Y. &amp; Eberhart, R. (1998). <i>A modified particle swarm optimizer</i>.
 *       IEEE World Congress on Computational Intelligence, pp. 69–73.</li>
 * </ul>
 *
 * <h2>Update rules (per particle, per iteration)</h2>
 * <pre>
 *   vᵢ ← w·vᵢ + c₁·r₁·(pbestᵢ − xᵢ) + c₂·r₂·(gbest − xᵢ)
 *   vᵢ ← clamp(vᵢ, −vMax, vMax)     // per dimension
 *   xᵢ ← clamp(xᵢ + vᵢ, lb, ub)     // reflect into search space
 * </pre>
 *
 * <h2>Implementation notes</h2>
 * <ul>
 *   <li>PRNG: {@code L128X1024MixRandom} – deterministic with fixed seed.</li>
 *   <li>Global best is updated <b>immediately</b> when a particle improves its
 *       personal best beyond the current global best (asynchronous within the
 *       iteration).</li>
 *   <li>If a new position yields a non‑finite fitness, the particle stays at that
 *       position but its fitness is set to ∞ and personal/global bests are not updated.</li>
 *   <li>Convergence history stores the global best fitness <i>before</i> the first
 *       iteration (iteration 0) and after each iteration.</li>
 *   <li>Java 21‑ready: uses {@code Math.clamp} and modern {@code RandomGenerator} API.</li>
 * </ul>
 */
@Slf4j
public final class PsoAlgorithm {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";

    private final PsoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final Particle[] swarm;
    private final double[] vMax;                // per‑dimension velocity bound
    private double[] globalBestPosition;
    private double globalBestFitness = Double.POSITIVE_INFINITY;

    /**
     * Creates a new optimizer and immediately initialises the swarm.
     *
     * @param config   validated configuration (non‑null)
     * @param function objective to minimise (non‑null)
     */
    public PsoAlgorithm(@NonNull PsoConfig config, @NonNull ObjectiveFunction function) {
        this.config = config;
        this.function = function;
        this.random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.swarm = new Particle[config.populationSize()];
        this.vMax = new double[config.dimensions()];
        var min = config.minBounds();
        var max = config.maxBounds();
        for (int d = 0; d < config.dimensions(); d++) {
            vMax[d] = config.velocityClampFactor() * (max[d] - min[d]);
        }
        initializeSwarm();
    }

    // ─────────────────────────── Phase 0 : Initialisation ───────────────────────────

    private void initializeSwarm() {
        final var min = config.minBounds();
        final var max = config.maxBounds();
        final int n = config.dimensions();

        for (int i = 0; i < config.populationSize(); i++) {
            double[] x = new double[n];
            double[] v = new double[n];
            for (int d = 0; d < n; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
                v[d] = (random.nextDouble() * 2.0 - 1.0) * vMax[d];   // U(−vMax, vMax)
            }
            double fit = function.evaluate(x);
            if (!Double.isFinite(fit)) {
                log.warn("Particle {} initialised with invalid fitness → ∞", i);
                fit = Double.POSITIVE_INFINITY;
            }
            swarm[i] = new Particle(x, v, fit);

            if (fit < globalBestFitness) {
                globalBestFitness = fit;
                globalBestPosition = x.clone();
            }
        }
        log.debug("Swarm of {} particles started. Global best = {}", config.populationSize(), globalBestFitness);
    }

    // ────────────────────────── Main optimisation loop ─────────────────────────────

    /**
     * Runs the PSO for the configured number of iterations.
     *
     * @return an immutable summary of the run (best position, fitness, convergence history)
     */
    public Result optimize() {
        final int n = config.dimensions();
        final var min = config.minBounds();
        final var max = config.maxBounds();
        final double[] history = new double[config.maxIterations() + 1];
        history[0] = globalBestFitness;               // initial best

        for (int t = 1; t <= config.maxIterations(); t++) {
            for (final Particle p : swarm) {
                // Update velocity and position
                for (int d = 0; d < n; d++) {
                    double r1 = random.nextDouble();
                    double r2 = random.nextDouble();
                    double cognitive = config.cognitiveCoefficient() * r1
                            * (p.getPersonalBestPosition()[d] - p.getPosition()[d]);
                    double social = config.socialCoefficient() * r2
                            * (globalBestPosition[d] - p.getPosition()[d]);

                    double newV = config.inertia() * p.getVelocity()[d] + cognitive + social;
                    newV = Math.clamp(newV, -vMax[d], vMax[d]);
                    p.getVelocity()[d] = newV;

                    double newX = p.getPosition()[d] + newV;
                    p.getPosition()[d] = Math.clamp(newX, min[d], max[d]);
                }

                // Evaluate new position
                double fit = function.evaluate(p.getPosition());
                if (!Double.isFinite(fit)) {
                    p.setFitness(Double.POSITIVE_INFINITY);   // keep fitness consistent
                    continue;                                 // skip personal/global best update
                }
                p.setFitness(fit);

                // Update personal best
                if (fit < p.getPersonalBestFitness()) {
                    p.setPersonalBestFitness(fit);
                    System.arraycopy(p.getPosition(), 0, p.getPersonalBestPosition(), 0, n);
                }

                // Update global best immediately (asynchronous within iteration)
                if (p.getPersonalBestFitness() < globalBestFitness) {
                    globalBestFitness = p.getPersonalBestFitness();
                    globalBestPosition = p.getPersonalBestPosition().clone();
                }
            }
            history[t] = globalBestFitness;
            if (t % 100 == 0 || t == config.maxIterations()) {
                log.info("Iteration {}: best fitness = {}", t, globalBestFitness);
            }
        }
        log.info("Optimisation finished. Best fitness = {}", globalBestFitness);
        return Result.builder()
                .bestPosition(globalBestPosition.clone())
                .bestFitness(globalBestFitness)
                .convergenceHistory(history)
                .seed(config.seed())
                .build();
    }

    /**
     * Quick access to the best fitness found so far (after a call to {@link #optimize}).
     */
    public double getBestFitness() {
        return globalBestFitness;
    }

    // ──────────────────────── Diagnostic access (package‑private for tests) ─────────

    /**
     * Returns current positions of all particles (defensive copies).
     */
    double[][] getPositions() {
        return Arrays.stream(swarm).map(p -> p.getPosition().clone()).toArray(double[][]::new);
    }

    /**
     * Returns current velocities of all particles (defensive copies).
     */
    double[][] getVelocities() {
        return Arrays.stream(swarm).map(p -> p.getVelocity().clone()).toArray(double[][]::new);
    }

    /**
     * Returns personal best positions of all particles (defensive copies).
     */
    double[][] getPersonalBests() {
        return Arrays.stream(swarm).map(p -> p.getPersonalBestPosition().clone()).toArray(double[][]::new);
    }

    /**
     * Returns personal best fitnesses of all particles.
     */
    double[] getPersonalBestFitnesses() {
        return Arrays.stream(swarm).mapToDouble(Particle::getPersonalBestFitness).toArray();
    }

    // ──────────────────────── supporting types ─────────────────────────────────────

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable configuration for the PSO algorithm.
     */
    @Builder
    public record PsoConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double @NonNull [] minBounds,
            double @NonNull [] maxBounds,
            double inertia,                // w ∈ (0,1]   (correctly spelled "inertia")
            double cognitiveCoefficient,   // c1 ≥ 0
            double socialCoefficient,      // c2 ≥ 0
            double velocityClampFactor,    // k ∈ (0,1]
            long seed
    ) {
        public PsoConfig {
            Objects.requireNonNull(minBounds);
            Objects.requireNonNull(maxBounds);
            if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0");
            if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bounds length mismatch");
            if (inertia <= 0 || inertia > 1) throw new IllegalArgumentException("inertia ∈ (0,1]");
            if (cognitiveCoefficient < 0) throw new IllegalArgumentException("c1 ≥ 0");
            if (socialCoefficient < 0) throw new IllegalArgumentException("c2 ≥ 0");
            if (velocityClampFactor <= 0 || velocityClampFactor > 1)
                throw new IllegalArgumentException("velocityClampFactor ∈ (0,1]");
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }
    }

    /**
     * Outcome of a full optimisation run.
     */
    @Builder
    public record Result(
            double @NonNull [] bestPosition,
            double bestFitness,
            double @NonNull [] convergenceHistory,
            long seed) {
        public Result {
            Objects.requireNonNull(bestPosition);
            Objects.requireNonNull(convergenceHistory);
            bestPosition = bestPosition.clone();
            convergenceHistory = convergenceHistory.clone();
        }

        @Override
        public double[] bestPosition() {
            return bestPosition.clone();
        }

        @Override
        public double[] convergenceHistory() {
            return convergenceHistory.clone();
        }
    }

    /**
     * Internal mutable particle. Accessors are used inside the algorithm (Lombok).
     */
    @Getter
    @Setter
    private static class Particle {
        private final double[] position;
        private final double[] velocity;
        private double fitness;
        private double[] personalBestPosition;
        private double personalBestFitness;

        Particle(double[] position, double[] velocity, double fitness) {
            this.position = position.clone();
            this.velocity = velocity.clone();
            this.fitness = fitness;
            this.personalBestPosition = position.clone();
            this.personalBestFitness = fitness;
        }
    }

    // ──────────────────────── Demo with Ackley function ────────────────────────────

    /**
     * Minimal demo that optimises the Ackley function (10‑D).
     */
    @Slf4j
    public static class Demo {
        static void main(String... args) {
            log.info("=== PSO Demo – Ackley function ===");
            ObjectiveFunction ackley = x -> {
                int n = x.length;
                double sumSq = 0.0, sumCos = 0.0;
                for (double v : x) {
                    sumSq += v * v;
                    sumCos += Math.cos(2.0 * Math.PI * v);
                }
                return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                        - Math.exp(sumCos / n) + 20.0 + Math.E;
            };

            int d = 10;
            var min = new double[d];
            var max = new double[d];
            Arrays.fill(min, -32.768);
            Arrays.fill(max, 32.768);

            var cfg = PsoConfig.builder()
                    .populationSize(40)
                    .maxIterations(2000)
                    .dimensions(d)
                    .minBounds(min)
                    .maxBounds(max)
                    .inertia(0.7298)
                    .cognitiveCoefficient(1.49618)
                    .socialCoefficient(1.49618)
                    .velocityClampFactor(0.5)
                    .seed(12345L)
                    .build();

            var pso = new PsoAlgorithm(cfg, ackley);
            var res = pso.optimize();
            log.info("Best fitness = {}", res.bestFitness());
            log.info("Sample of best position = {}",
                    Arrays.toString(Arrays.copyOf(res.bestPosition(), 3)));
        }
    }
} 
 
package edu.swarmintelligence.pso.r02;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

@Slf4j
public class AckleyDemo {
    static void main(String... args) {
        log.info("=== PSO Demo – Ackley function ===");

        ObjectiveFunction ackley = x -> {
            int n = x.length;
            double sumSq = 0.0, sumCos = 0.0;
            for (double v : x) {
                sumSq += v * v;
                sumCos += Math.cos(2.0 * Math.PI * v);
            }
            return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                    - Math.exp(sumCos / n) + 20.0 + Math.E;
        };

        int d = 10;
        var min = new double[d];
        var max = new double[d];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);

        var cfg = PsoConfig.builder()
                .populationSize(40)
                .maxIterations(2000)
                .dimensions(d)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(12345L)
                .build();

        var optimizer = new PsoOptimizer(cfg, ackley);
        var result = optimizer.optimize();

        log.info("Best fitness = {}", result.bestFitness());
        log.info("Sample of best position = {}",
                Arrays.toString(Arrays.copyOf(result.bestPosition(), 3)));
    }
} 
 
package edu.swarmintelligence.pso.r02;

@FunctionalInterface
public interface ObjectiveFunction {
    double evaluate(double[] x);
} 
 
package edu.swarmintelligence.pso.r02;

import lombok.Getter;
import lombok.Setter;

import java.util.random.RandomGenerator;

@Getter
@Setter
class Particle {
    private final double[] position;
    private final double[] velocity;
    private final double[] personalBestPosition;
    private double fitness;
    private double personalBestFitness;

    Particle(double[] position, double[] velocity, double fitness) {
        this.position = position.clone();
        this.velocity = velocity.clone();
        this.fitness = fitness;
        this.personalBestPosition = position.clone();
        this.personalBestFitness = fitness;
    }

    void updateVelocityAndPosition(double inertia,
                                   double cognitiveCoeff,
                                   double socialCoeff,
                                   double[] globalBest,
                                   double[] vMax,
                                   double[] lb,
                                   double[] ub,
                                   RandomGenerator random) {
        for (int d = 0; d < position.length; d++) {
            double r1 = random.nextDouble();
            double r2 = random.nextDouble();
            double cognitive = cognitiveCoeff * r1 * (personalBestPosition[d] - position[d]);
            double social = socialCoeff * r2 * (globalBest[d] - position[d]);
            double newV = inertia * velocity[d] + cognitive + social;
            newV = Math.clamp(newV, -vMax[d], vMax[d]);
            velocity[d] = newV;
            position[d] = Math.clamp(position[d] + newV, lb[d], ub[d]);
        }
    }

    void updatePersonalBest() {
        System.arraycopy(position, 0, personalBestPosition, 0, position.length);
        personalBestFitness = fitness;
    }

    double getPersonalBestFitness() {
        return personalBestFitness;
    }

    double[] getPersonalBestPosition() {
        return personalBestPosition;
    }
} 
 
package edu.swarmintelligence.pso.r02;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoConfig(
        int populationSize,
        int maxIterations,
        int dimensions,
        double @NonNull [] minBounds,
        double @NonNull [] maxBounds,
        double inertia,                // w ∈ (0,1]   (historical spelling preserved)
        double cognitiveCoefficient,    // c1 ≥ 0
        double socialCoefficient,       // c2 ≥ 0
        double velocityClampFactor,     // k ∈ (0,1]
        long seed
) {
    public PsoConfig {
        Objects.requireNonNull(minBounds);
        Objects.requireNonNull(maxBounds);
        if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0");
        if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
        if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
        if (minBounds.length != dimensions || maxBounds.length != dimensions)
            throw new IllegalArgumentException("Bounds length mismatch");
        if (inertia <= 0 || inertia > 1) throw new IllegalArgumentException("inertia ∈ (0,1]");
        if (cognitiveCoefficient < 0) throw new IllegalArgumentException("c1 ≥ 0");
        if (socialCoefficient < 0) throw new IllegalArgumentException("c2 ≥ 0");
        if (velocityClampFactor <= 0 || velocityClampFactor > 1)
            throw new IllegalArgumentException("velocityClampFactor ∈ (0,1]");
        minBounds = minBounds.clone();
        maxBounds = maxBounds.clone();
    }
} 
 
package edu.swarmintelligence.pso.r02;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

@Slf4j
public final class PsoOptimizer {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";

    private final PsoConfig config;
    private final Swarm swarm;

    public PsoOptimizer(@NonNull PsoConfig config, @NonNull ObjectiveFunction function) {
        this.config = config;
        RandomGenerator random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.swarm = Swarm.createAndInitialize(config, function, random);
    }

    public PsoResult optimize() {
        final double[] history = new double[config.maxIterations() + 1];
        history[0] = swarm.getGlobalBestFitness();

        for (int t = 1; t <= config.maxIterations(); t++) {
            swarm.iterate();
            history[t] = swarm.getGlobalBestFitness();
            if (t % 100 == 0 || t == config.maxIterations()) {
                log.info("Iteration {}: best fitness = {}", t, swarm.getGlobalBestFitness());
            }
        }

        log.info("Optimisation finished. Best fitness = {}", swarm.getGlobalBestFitness());
        return PsoResult.builder()
                .bestPosition(swarm.getGlobalBestPosition())
                .bestFitness(swarm.getGlobalBestFitness())
                .convergenceHistory(history)
                .seed(config.seed())
                .build();
    }

    public double getBestFitness() {
        return swarm.getGlobalBestFitness();
    }

    Swarm getSwarm() {
        return swarm;
    }
} 
 
package edu.swarmintelligence.pso.r02;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoResult(
        double @NonNull [] bestPosition,
        double bestFitness,
        double @NonNull [] convergenceHistory,
        long seed
) {
    public PsoResult {
        Objects.requireNonNull(bestPosition);
        Objects.requireNonNull(convergenceHistory);
        bestPosition = bestPosition.clone();
        convergenceHistory = convergenceHistory.clone();
    }

    @Override
    public double[] bestPosition() {
        return bestPosition.clone();
    }

    @Override
    public double[] convergenceHistory() {
        return convergenceHistory.clone();
    }
} 
 
package edu.swarmintelligence.pso.r02;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.random.RandomGenerator;

@Slf4j
class Swarm {
    private final Particle[] particles;
    private final double[] vMax;
    private final PsoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;

    private double[] globalBestPosition;
    private double globalBestFitness;

    private Swarm(Particle[] particles,
                  double[] vMax,
                  PsoConfig config,
                  ObjectiveFunction function,
                  RandomGenerator random,
                  double[] globalBestPosition,
                  double globalBestFitness) {
        this.particles = particles;
        this.vMax = vMax;
        this.config = config;
        this.function = function;
        this.random = random;
        this.globalBestPosition = globalBestPosition.clone(); // defensive
        this.globalBestFitness = globalBestFitness;
    }

    static Swarm createAndInitialize(PsoConfig config,
                                     ObjectiveFunction function,
                                     RandomGenerator random) {
        int popSize = config.populationSize();
        int dims = config.dimensions();
        double[] min = config.minBounds();
        double[] max = config.maxBounds();

        double[] vMax = new double[dims];
        for (int d = 0; d < dims; d++) {
            vMax[d] = config.velocityClampFactor() * (max[d] - min[d]);
        }

        Particle[] particles = new Particle[popSize];
        double[] bestPos = null;
        double bestFit = Double.POSITIVE_INFINITY;

        for (int i = 0; i < popSize; i++) {
            double[] x = new double[dims];
            double[] v = new double[dims];
            for (int d = 0; d < dims; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
                v[d] = (random.nextDouble() * 2.0 - 1.0) * vMax[d];
            }
            double fit = function.evaluate(x);
            if (!Double.isFinite(fit)) {
                log.warn("Particle {} initialised with invalid fitness → ∞", i);
                fit = Double.POSITIVE_INFINITY;
            }
            particles[i] = new Particle(x, v, fit);
            if (fit < bestFit) {
                bestFit = fit;
                bestPos = x.clone();
            }
        }

        log.debug("Swarm of {} particles started. Global best = {}", popSize, bestFit);
        return new Swarm(particles, vMax, config, function, random, bestPos, bestFit);
    }

    void iterate() {
        final double inertia = config.inertia();
        final double cognitiveCoeff = config.cognitiveCoefficient();
        final double socialCoeff = config.socialCoefficient();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        for (Particle p : particles) {
            p.updateVelocityAndPosition(inertia, cognitiveCoeff, socialCoeff,
                    globalBestPosition, vMax, min, max, random);

            double fit = function.evaluate(p.getPosition());
            if (!Double.isFinite(fit)) {
                p.setFitness(Double.POSITIVE_INFINITY);
                continue;  // explicitly skip personal/global best update
            }
            p.setFitness(fit);

            if (fit < p.getPersonalBestFitness()) {
                p.updatePersonalBest();
            }
            if (p.getPersonalBestFitness() < globalBestFitness) {
                globalBestFitness = p.getPersonalBestFitness();
                globalBestPosition = p.getPersonalBestPosition().clone();
            }
        }
    }

    double[] getGlobalBestPosition() {
        return globalBestPosition.clone();  // defensive copy
    }

    double getGlobalBestFitness() {
        return globalBestFitness;
    }

    double[][] getPositions() {
        return Arrays.stream(particles).map(p -> p.getPosition().clone()).toArray(double[][]::new);
    }

    double[][] getVelocities() {
        return Arrays.stream(particles).map(p -> p.getVelocity().clone()).toArray(double[][]::new);
    }

    double[][] getPersonalBests() {
        return Arrays.stream(particles).map(p -> p.getPersonalBestPosition().clone()).toArray(double[][]::new);
    }

    double[] getPersonalBestFitnesses() {
        return Arrays.stream(particles).mapToDouble(Particle::getPersonalBestFitness).toArray();
    }
} 
 
package edu.swarmintelligence.pso.r03;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

@Slf4j
public class AckleyDemo {
    static void main(String... args) {
        log.info("=== PSO Demo – Ackley function ===");

        ObjectiveFunction ackley = x -> {
            int n = x.length;
            double sumSq = 0.0, sumCos = 0.0;
            for (double v : x) {
                sumSq += v * v;
                sumCos += Math.cos(2.0 * Math.PI * v);
            }
            return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                    - Math.exp(sumCos / n) + 20.0 + Math.E;
        };

        int d = 10;
        var min = new double[d];
        var max = new double[d];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);

        var cfg = PsoConfig.builder()
                .populationSize(40)
                .maxIterations(2000)
                .dimensions(d)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(12345L)
                .build();

        var optimizer = new PsoOptimizer(cfg, ackley);
        var result = optimizer.optimize();

        log.info("Best fitness = {}", result.bestFitness());
        log.info("Sample of best position = {}",
                Arrays.toString(Arrays.copyOf(result.bestPosition(), 3)));
    }
} 
 
package edu.swarmintelligence.pso.r03;

@FunctionalInterface
public interface ObjectiveFunction {
    double evaluate(double[] x);
} 
 
package edu.swarmintelligence.pso.r03;

import lombok.Getter;
import lombok.Setter;

import java.util.random.RandomGenerator;

@Getter
@Setter
class Particle {
    private final double[] position;
    private final double[] velocity;
    private final double[] personalBestPosition;
    private double fitness;
    private double personalBestFitness;

    Particle(double[] position, double[] velocity, double fitness) {
        this.position = position.clone();
        this.velocity = velocity.clone();
        this.fitness = fitness;
        this.personalBestPosition = position.clone();
        this.personalBestFitness = fitness;
    }

    void updateVelocityAndPosition(double inertia,
                                   double cognitiveCoeff,
                                   double socialCoeff,
                                   double[] globalBest,
                                   double[] vMax,
                                   double[] lb,
                                   double[] ub,
                                   RandomGenerator random) {
        for (int d = 0; d < position.length; d++) {
            double r1 = random.nextDouble();
            double r2 = random.nextDouble();
            double cognitive = cognitiveCoeff * r1 * (personalBestPosition[d] - position[d]);
            double social = socialCoeff * r2 * (globalBest[d] - position[d]);
            double newV = inertia * velocity[d] + cognitive + social;
            newV = Math.clamp(newV, -vMax[d], vMax[d]);
            velocity[d] = newV;
            position[d] = Math.clamp(position[d] + newV, lb[d], ub[d]);
        }
    }

    void updatePersonalBest() {
        System.arraycopy(position, 0, personalBestPosition, 0, position.length);
        personalBestFitness = fitness;
    }

    double getPersonalBestFitness() {
        return personalBestFitness;
    }

    double[] getPersonalBestPosition() {
        return personalBestPosition;
    }
} 
 
package edu.swarmintelligence.pso.r03;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoConfig(
        int populationSize,
        int maxIterations,
        int dimensions,
        double @NonNull [] minBounds,
        double @NonNull [] maxBounds,
        double inertia,                // w ∈ (0,1]   (historical spelling preserved)
        double cognitiveCoefficient,    // c1 ≥ 0
        double socialCoefficient,       // c2 ≥ 0
        double velocityClampFactor,     // k ∈ (0,1]
        long seed
) {
    public PsoConfig {
        Objects.requireNonNull(minBounds);
        Objects.requireNonNull(maxBounds);
        if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0");
        if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
        if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
        if (minBounds.length != dimensions || maxBounds.length != dimensions)
            throw new IllegalArgumentException("Bounds length mismatch");
        if (inertia <= 0 || inertia > 1) throw new IllegalArgumentException("inertia ∈ (0,1]");
        if (cognitiveCoefficient < 0) throw new IllegalArgumentException("c1 ≥ 0");
        if (socialCoefficient < 0) throw new IllegalArgumentException("c2 ≥ 0");
        if (velocityClampFactor <= 0 || velocityClampFactor > 1)
            throw new IllegalArgumentException("velocityClampFactor ∈ (0,1]");
        minBounds = minBounds.clone();
        maxBounds = maxBounds.clone();
    }
} 
 
package edu.swarmintelligence.pso.r03;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

@Slf4j
public final class PsoOptimizer {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";

    private final PsoConfig config;
    private final Swarm swarm;

    public PsoOptimizer(@NonNull PsoConfig config, @NonNull ObjectiveFunction function) {
        this.config = config;
        RandomGenerator random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.swarm = Swarm.createAndInitialize(config, function, random);
    }

    public PsoResult optimize() {
        final double[] history = new double[config.maxIterations() + 1];
        history[0] = swarm.getGlobalBestFitness();

        for (int t = 1; t <= config.maxIterations(); t++) {
            swarm.iterate();
            history[t] = swarm.getGlobalBestFitness();
            if (t % 100 == 0 || t == config.maxIterations()) {
                log.info("Iteration {}: best fitness = {}", t, swarm.getGlobalBestFitness());
            }
        }

        log.info("Optimisation finished. Best fitness = {}", swarm.getGlobalBestFitness());
        return PsoResult.builder()
                .bestPosition(swarm.getGlobalBestPosition())
                .bestFitness(swarm.getGlobalBestFitness())
                .convergenceHistory(history)
                .seed(config.seed())
                .build();
    }

    public double getBestFitness() {
        return swarm.getGlobalBestFitness();
    }

    Swarm getSwarm() {
        return swarm;
    }
} 
 
package edu.swarmintelligence.pso.r03;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoResult(
        double @NonNull [] bestPosition,
        double bestFitness,
        double @NonNull [] convergenceHistory,
        long seed
) {
    public PsoResult {
        Objects.requireNonNull(bestPosition);
        Objects.requireNonNull(convergenceHistory);
        bestPosition = bestPosition.clone();
        convergenceHistory = convergenceHistory.clone();
    }

    @Override
    public double[] bestPosition() {
        return bestPosition.clone();
    }

    @Override
    public double[] convergenceHistory() {
        return convergenceHistory.clone();
    }
} 
 
package edu.swarmintelligence.pso.r03;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.random.RandomGenerator;

@Slf4j
class Swarm {
    private final Particle[] particles;
    private final double[] vMax;
    private final PsoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;

    private double[] globalBestPosition;
    private double globalBestFitness;

    private Swarm(Particle[] particles,
                  double[] vMax,
                  PsoConfig config,
                  ObjectiveFunction function,
                  RandomGenerator random,
                  double[] globalBestPosition,
                  double globalBestFitness) {
        this.particles = particles;
        this.vMax = vMax;
        this.config = config;
        this.function = function;
        this.random = random;
        this.globalBestPosition = globalBestPosition.clone(); // defensive
        this.globalBestFitness = globalBestFitness;
    }

    static Swarm createAndInitialize(PsoConfig config,
                                     ObjectiveFunction function,
                                     RandomGenerator random) {
        int popSize = config.populationSize();
        int dims = config.dimensions();
        double[] min = config.minBounds();
        double[] max = config.maxBounds();

        double[] vMax = new double[dims];
        for (int d = 0; d < dims; d++) {
            vMax[d] = config.velocityClampFactor() * (max[d] - min[d]);
        }

        Particle[] particles = new Particle[popSize];
        double[] bestPos = null;
        double bestFit = Double.POSITIVE_INFINITY;

        for (int i = 0; i < popSize; i++) {
            double[] x = new double[dims];
            double[] v = new double[dims];
            for (int d = 0; d < dims; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
                v[d] = (random.nextDouble() * 2.0 - 1.0) * vMax[d];
            }
            double fit = function.evaluate(x);
            if (!Double.isFinite(fit)) {
                log.warn("Particle {} initialised with invalid fitness → ∞", i);
                fit = Double.POSITIVE_INFINITY;
            }
            particles[i] = new Particle(x, v, fit);
            if (fit < bestFit) {
                bestFit = fit;
                bestPos = x.clone();
            }
        }

        log.debug("Swarm of {} particles started. Global best = {}", popSize, bestFit);
        return new Swarm(particles, vMax, config, function, random, bestPos, bestFit);
    }

    void iterate() {
        final double inertia = config.inertia();
        final double cognitiveCoeff = config.cognitiveCoefficient();
        final double socialCoeff = config.socialCoefficient();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        for (Particle p : particles) {
            p.updateVelocityAndPosition(inertia, cognitiveCoeff, socialCoeff,
                    globalBestPosition, vMax, min, max, random);

            double fit = function.evaluate(p.getPosition());
            if (!Double.isFinite(fit)) {
                p.setFitness(Double.POSITIVE_INFINITY);
                continue;  // explicitly skip personal/global best update
            }
            p.setFitness(fit);

            if (fit < p.getPersonalBestFitness()) {
                p.updatePersonalBest();
            }
            if (p.getPersonalBestFitness() < globalBestFitness) {
                globalBestFitness = p.getPersonalBestFitness();
                globalBestPosition = p.getPersonalBestPosition().clone();
            }
        }
    }

    double[] getGlobalBestPosition() {
        return globalBestPosition.clone();  // defensive copy
    }

    double getGlobalBestFitness() {
        return globalBestFitness;
    }

    double[][] getPositions() {
        return Arrays.stream(particles).map(p -> p.getPosition().clone()).toArray(double[][]::new);
    }

    double[][] getVelocities() {
        return Arrays.stream(particles).map(p -> p.getVelocity().clone()).toArray(double[][]::new);
    }

    double[][] getPersonalBests() {
        return Arrays.stream(particles).map(p -> p.getPersonalBestPosition().clone()).toArray(double[][]::new);
    }

    double[] getPersonalBestFitnesses() {
        return Arrays.stream(particles).mapToDouble(Particle::getPersonalBestFitness).toArray();
    }
} 
 
package edu.swarmintelligence.pso.r04;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

@Slf4j
public class AckleyDemo {
    static void main(String... args) {
        log.info("=== PSO Demo – Ackley function ===");

        ObjectiveFunction ackley = x -> {
            int n = x.length;
            double sumSq = 0.0, sumCos = 0.0;
            for (double v : x) {
                sumSq += v * v;
                sumCos += Math.cos(2.0 * Math.PI * v);
            }
            return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                    - Math.exp(sumCos / n) + 20.0 + Math.E;
        };

        int d = 10;
        var min = new double[d];
        var max = new double[d];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        double[] knownOptimum = new double[d];   // optimum at origin

        var cfg = PsoConfig.builder()
                .populationSize(40)
                .maxIterations(2000)
                .dimensions(d)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(12345L)
                .knownOptimum(knownOptimum)
                .build();

        var optimizer = new PsoOptimizer(cfg, ackley);
        var result = optimizer.optimize();

        log.info("Best fitness = {}", result.bestFitness());
        log.info("Sample of best position = {}",
                Arrays.toString(Arrays.copyOf(result.bestPosition(), 3)));
    }
} 
 
package edu.swarmintelligence.pso.r04;

@FunctionalInterface
public interface ObjectiveFunction {
    double evaluate(double[] x);
} 
 
package edu.swarmintelligence.pso.r04;

import lombok.Getter;
import lombok.Setter;

import java.util.random.RandomGenerator;

@Getter
@Setter
class Particle {
    private final double[] position;
    private final double[] velocity;
    private final double[] personalBestPosition;
    private double fitness;
    private double personalBestFitness;

    Particle(double[] position, double[] velocity, double fitness) {
        this.position = position.clone();
        this.velocity = velocity.clone();
        this.fitness = fitness;
        this.personalBestPosition = position.clone();
        this.personalBestFitness = fitness;
    }

    void updateVelocityAndPosition(double inertia,
                                   double cognitiveCoeff,
                                   double socialCoeff,
                                   double[] globalBest,
                                   double[] vMax,
                                   double[] lb,
                                   double[] ub,
                                   RandomGenerator random) {
        for (int d = 0; d < position.length; d++) {
            double r1 = random.nextDouble();
            double r2 = random.nextDouble();
            double cognitive = cognitiveCoeff * r1 * (personalBestPosition[d] - position[d]);
            double social = socialCoeff * r2 * (globalBest[d] - position[d]);
            double newV = inertia * velocity[d] + cognitive + social;
            newV = Math.clamp(newV, -vMax[d], vMax[d]);
            velocity[d] = newV;
            position[d] = Math.clamp(position[d] + newV, lb[d], ub[d]);
        }
    }

    void updatePersonalBest() {
        System.arraycopy(position, 0, personalBestPosition, 0, position.length);
        personalBestFitness = fitness;
    }

    double getPersonalBestFitness() {
        return personalBestFitness;
    }

    double[] getPersonalBestPosition() {
        return personalBestPosition;   // defensive copying is done by callers
    }
} 
 
package edu.swarmintelligence.pso.r04;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoConfig(
        int populationSize,
        int maxIterations,
        int dimensions,
        double @NonNull [] minBounds,
        double @NonNull [] maxBounds,
        double inertia,
        double cognitiveCoefficient,
        double socialCoefficient,
        double velocityClampFactor,
        long seed,
        double[] knownOptimum                    // optional – null → distToOptimum = "NA"
) {
    public PsoConfig {
        Objects.requireNonNull(minBounds);
        Objects.requireNonNull(maxBounds);
        if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0");
        if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
        if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
        if (minBounds.length != dimensions || maxBounds.length != dimensions)
            throw new IllegalArgumentException("Bounds length mismatch");
        if (inertia <= 0 || inertia > 1) throw new IllegalArgumentException("inertia ∈ (0,1]");
        if (cognitiveCoefficient < 0) throw new IllegalArgumentException("c1 ≥ 0");
        if (socialCoefficient < 0) throw new IllegalArgumentException("c2 ≥ 0");
        if (velocityClampFactor <= 0 || velocityClampFactor > 1)
            throw new IllegalArgumentException("velocityClampFactor ∈ (0,1]");

        // defensive copies for mutable arrays
        minBounds = minBounds.clone();
        maxBounds = maxBounds.clone();

        // optional knownOptimum – validate length and defensive copy
        if (knownOptimum != null) {
            if (knownOptimum.length != dimensions)
                throw new IllegalArgumentException("knownOptimum length must equal dimensions");
            knownOptimum = knownOptimum.clone();   // re‑assign parameter, not field
        }
    }
} 
 
package edu.swarmintelligence.pso.r04;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

@Slf4j
public final class PsoOptimizer {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";

    private final PsoConfig config;
    private final Swarm swarm;
    private final Path logPath;   // null → no file logging

    /**
     * Creates an optimizer that writes the execution log to {@code algorithm_run.log}.
     */
    public PsoOptimizer(@NonNull PsoConfig config, @NonNull ObjectiveFunction function) {
        this(config, function, Path.of("algorithm_run.log"));
    }

    /**
     * Package‑private constructor for testing: {@code logPath} may be null to suppress file output.
     */
    PsoOptimizer(PsoConfig config, ObjectiveFunction function, Path logPath) {
        this.config = config;
        RandomGenerator random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.swarm = Swarm.createAndInitialize(config, function, random);
        this.logPath = logPath;
    }

    public PsoResult optimize() {
        final double[] history = new double[config.maxIterations() + 1];
        history[0] = swarm.getGlobalBestFitness();

        PrintWriter logWriter = null;
        try {
            if (logPath != null) {
                logWriter = new PrintWriter(
                        new BufferedWriter(new OutputStreamWriter(
                                new FileOutputStream(logPath.toFile()), StandardCharsets.UTF_8)));
                logWriter.println("iteration;agentId;positionBefore;positionAfter;personalBest;personalBestFitness;globalBest;globalBestFitness;popAvgFitness;popStdDev;distToOptimum");
            }

            // iteration 0 (initial state)
            if (logWriter != null) {
                List<Swarm.IterationLogEntry> initEntries = swarm.initialLogEntries();
                for (Swarm.IterationLogEntry e : initEntries) {
                    logWriter.println(e.toCsvLine());
                }
                logWriter.flush();
            }

            // main loop
            for (int t = 1; t <= config.maxIterations(); t++) {
                List<Swarm.IterationLogEntry> iterLog = swarm.iterate(t);
                history[t] = swarm.getGlobalBestFitness();

                if (logWriter != null) {
                    for (Swarm.IterationLogEntry e : iterLog) {
                        logWriter.println(e.toCsvLine());
                    }
                    logWriter.flush();
                }

                if (t % 100 == 0 || t == config.maxIterations()) {
                    log.info("Iteration {}: best fitness = {}", t, swarm.getGlobalBestFitness());
                }
            }
        } catch (IOException e) {
            log.error("Failed to write log file", e);
        } finally {
            if (logWriter != null) {
                logWriter.close();
            }
        }

        log.info("Optimisation finished. Best fitness = {}", swarm.getGlobalBestFitness());
        return PsoResult.builder()
                .bestPosition(swarm.getGlobalBestPosition())
                .bestFitness(swarm.getGlobalBestFitness())
                .convergenceHistory(history)
                .seed(config.seed())
                .build();
    }

    public double getBestFitness() {
        return swarm.getGlobalBestFitness();
    }

    Swarm getSwarm() {
        return swarm;
    }
} 
 
package edu.swarmintelligence.pso.r04;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoResult(
        double @NonNull [] bestPosition,
        double bestFitness,
        double @NonNull [] convergenceHistory,
        long seed
) {
    public PsoResult {
        Objects.requireNonNull(bestPosition);
        Objects.requireNonNull(convergenceHistory);
        bestPosition = bestPosition.clone();
        convergenceHistory = convergenceHistory.clone();
    }

    @Override
    public double[] bestPosition() {
        return bestPosition.clone();
    }

    @Override
    public double[] convergenceHistory() {
        return convergenceHistory.clone();
    }
} 
 
package edu.swarmintelligence.pso.r04;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

@Slf4j
class Swarm {
    private final Particle[] particles;
    private final double[] vMax;
    private final PsoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;

    private double[] globalBestPosition;
    private double globalBestFitness;

    private Swarm(Particle[] particles,
                  double[] vMax,
                  PsoConfig config,
                  ObjectiveFunction function,
                  RandomGenerator random,
                  double[] globalBestPosition,
                  double globalBestFitness) {
        this.particles = particles;
        this.vMax = vMax;
        this.config = config;
        this.function = function;
        this.random = random;
        this.globalBestPosition = globalBestPosition.clone();
        this.globalBestFitness = globalBestFitness;
    }

    static Swarm createAndInitialize(PsoConfig config,
                                     ObjectiveFunction function,
                                     RandomGenerator random) {
        int popSize = config.populationSize();
        int dims = config.dimensions();
        double[] min = config.minBounds();
        double[] max = config.maxBounds();

        double[] vMax = new double[dims];
        for (int d = 0; d < dims; d++) {
            vMax[d] = config.velocityClampFactor() * (max[d] - min[d]);
        }

        Particle[] particles = new Particle[popSize];
        double[] bestPos = null;
        double bestFit = Double.POSITIVE_INFINITY;

        for (int i = 0; i < popSize; i++) {
            double[] x = new double[dims];
            double[] v = new double[dims];
            for (int d = 0; d < dims; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
                v[d] = (random.nextDouble() * 2.0 - 1.0) * vMax[d];
            }
            double fit = function.evaluate(x);
            if (!Double.isFinite(fit)) {
                log.warn("Particle {} initialised with invalid fitness → ∞", i);
                fit = Double.POSITIVE_INFINITY;
            }
            particles[i] = new Particle(x, v, fit);
            if (fit < bestFit) {
                bestFit = fit;
                bestPos = x.clone();
            }
        }

        log.debug("Swarm of {} particles started. Global best = {}", popSize, bestFit);
        assert bestPos != null;
        return new Swarm(particles, vMax, config, function, random, bestPos, bestFit);
    }

    private static String formatArray(double[] arr) {
        return Arrays.stream(arr)
                .mapToObj(v -> String.format(Locale.ROOT, "%.16g", v))
                .collect(Collectors.joining(" "));
    }

    /**
     * Performs one iteration and returns per‑particle log entries for that iteration.
     */
    List<IterationLogEntry> iterate(int iteration) {
        final double inertia = config.inertia();
        final double cognitiveCoeff = config.cognitiveCoefficient();
        final double socialCoeff = config.socialCoefficient();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        List<IterationLogEntry> entries = new ArrayList<>(particles.length);

        for (int i = 0; i < particles.length; i++) {
            Particle p = particles[i];

            // snapshot before movement
            double[] posBefore = p.getPosition().clone();

            // update velocity & position
            p.updateVelocityAndPosition(inertia, cognitiveCoeff, socialCoeff,
                    globalBestPosition, vMax, min, max, random);

            // evaluate new position
            double fit = function.evaluate(p.getPosition());
            if (!Double.isFinite(fit)) {
                p.setFitness(Double.POSITIVE_INFINITY);
            } else {
                p.setFitness(fit);
                if (fit < p.getPersonalBestFitness()) {
                    p.updatePersonalBest();
                }
            }

            // asynchronous global best update (based on personal best)
            if (p.getPersonalBestFitness() < globalBestFitness) {
                globalBestFitness = p.getPersonalBestFitness();
                globalBestPosition = p.getPersonalBestPosition().clone();
            }

            String positionBeforeStr = formatArray(posBefore);
            String positionAfterStr = formatArray(p.getPosition());
            String personalBestStr = formatArray(p.getPersonalBestPosition().clone());
            double personalBestFit = p.getPersonalBestFitness();
            String globalBestStr = formatArray(globalBestPosition.clone());
            double globalBestFit = globalBestFitness;

            entries.add(new IterationLogEntry(iteration, i,
                    positionBeforeStr, positionAfterStr,
                    personalBestStr, personalBestFit,
                    globalBestStr, globalBestFit,
                    null, null, // avg, std to be filled after loop
                    computeDistToOptimum(p.getPosition())));
        }

        // compute population statistics (only finite current fitnesses)
        double[] currentFitnesses = Arrays.stream(particles).mapToDouble(Particle::getFitness).toArray();
        int finiteCount = 0;
        double sum = 0.0;
        for (double f : currentFitnesses) {
            if (Double.isFinite(f)) {
                sum += f;
                finiteCount++;
            }
        }
        double avgFit = finiteCount > 0 ? sum / finiteCount : Double.NaN;
        double varSum = 0.0;
        if (finiteCount > 0) {
            for (double f : currentFitnesses) {
                if (Double.isFinite(f)) {
                    double diff = f - avgFit;
                    varSum += diff * diff;
                }
            }
        }
        double stdDevFit = finiteCount > 0 ? Math.sqrt(varSum / finiteCount) : Double.NaN;

        // update all entries with population stats
        for (IterationLogEntry e : entries) {
            e.popAvgFitness = avgFit;
            e.popStdDev = stdDevFit;
        }

        return entries;
    }

    /**
     * Log entries for the initial state (iteration 0, no movement).
     */
    List<IterationLogEntry> initialLogEntries() {
        List<IterationLogEntry> entries = new ArrayList<>(particles.length);
        double[] initialFitnesses = Arrays.stream(particles).mapToDouble(Particle::getFitness).toArray();

        int finCnt = 0;
        double sum = 0.0;
        for (double f : initialFitnesses) {
            if (Double.isFinite(f)) {
                sum += f;
                finCnt++;
            }
        }
        double avgFit = finCnt > 0 ? sum / finCnt : Double.NaN;
        double varSum = 0.0;
        if (finCnt > 0) {
            for (double f : initialFitnesses) {
                if (Double.isFinite(f)) {
                    double diff = f - avgFit;
                    varSum += diff * diff;
                }
            }
        }
        double stdDevFit = finCnt > 0 ? Math.sqrt(varSum / finCnt) : Double.NaN;

        for (int i = 0; i < particles.length; i++) {
            Particle p = particles[i];
            String pos = formatArray(p.getPosition());
            String pBest = formatArray(p.getPersonalBestPosition());
            entries.add(new IterationLogEntry(0, i,
                    pos, pos,   // before and after identical for initial snapshot
                    pBest, p.getPersonalBestFitness(),
                    formatArray(globalBestPosition), globalBestFitness,
                    avgFit, stdDevFit,
                    computeDistToOptimum(p.getPosition())));
        }
        return entries;
    }

    private String computeDistToOptimum(double[] position) {
        double[] optimum = config.knownOptimum();
        if (optimum == null) return "NA";
        double sumSq = 0.0;
        for (int i = 0; i < position.length; i++) {
            double diff = position[i] - optimum[i];
            sumSq += diff * diff;
        }
        return Double.toString(Math.sqrt(sumSq));
    }

    // ──────────────────── accessors (defensive copies) ────────────────────

    double[] getGlobalBestPosition() {
        return globalBestPosition.clone();
    }

    double getGlobalBestFitness() {
        return globalBestFitness;
    }

    double[][] getPositions() {
        return Arrays.stream(particles).map(p -> p.getPosition().clone()).toArray(double[][]::new);
    }

    double[][] getVelocities() {
        return Arrays.stream(particles).map(p -> p.getVelocity().clone()).toArray(double[][]::new);
    }

    double[][] getPersonalBests() {
        return Arrays.stream(particles).map(p -> p.getPersonalBestPosition().clone()).toArray(double[][]::new);
    }

    double[] getPersonalBestFitnesses() {
        return Arrays.stream(particles).mapToDouble(Particle::getPersonalBestFitness).toArray();
    }

    // ──────────────────── log entry structure (package‑private) ──────────
    static class IterationLogEntry {
        final int iteration;
        final int agentId;
        final String positionBefore;
        final String positionAfter;
        final String personalBest;
        final double personalBestFitness;
        final String globalBest;
        final double globalBestFitness;
        Double popAvgFitness;   // filled after iteration loop
        Double popStdDev;
        String distToOptimum;

        IterationLogEntry(int iteration, int agentId,
                          String positionBefore, String positionAfter,
                          String personalBest, double personalBestFitness,
                          String globalBest, double globalBestFitness,
                          Double popAvgFitness, Double popStdDev,
                          String distToOptimum) {
            this.iteration = iteration;
            this.agentId = agentId;
            this.positionBefore = positionBefore;
            this.positionAfter = positionAfter;
            this.personalBest = personalBest;
            this.personalBestFitness = personalBestFitness;
            this.globalBest = globalBest;
            this.globalBestFitness = globalBestFitness;
            this.popAvgFitness = popAvgFitness;
            this.popStdDev = popStdDev;
            this.distToOptimum = distToOptimum;
        }

        String toCsvLine() {
            return String.format(Locale.ROOT,
                    "%d;%d;%s;%s;%s;%.16g;%s;%.16g;%.16g;%.16g;%s",
                    iteration, agentId,
                    positionBefore, positionAfter,
                    personalBest, personalBestFitness,
                    globalBest, globalBestFitness,
                    popAvgFitness != null ? popAvgFitness : Double.NaN,
                    popStdDev != null ? popStdDev : Double.NaN,
                    distToOptimum);
        }
    }
} 
 
package edu.swarmintelligence.pso.r05;

import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * A powerful data analytics engine that processes the PSO execution log
 * ({@code algorithm_run.log}) and generates a professional evaluation report
 * focused on algorithm quality metrics.
 *
 * <p>The report includes:
 * <ul>
 *   <li>Run overview (inferred configuration)</li>
 *   <li>Final solution quality</li>
 *   <li>Convergence speed &amp; stagnation analysis</li>
 *   <li>Swarm diversity (average pairwise distance)</li>
 *   <li>Exploration vs. exploitation balance (velocity magnitude)</li>
 *   <li>Global best update frequency</li>
 *   <li>Distance to optimum (if known)</li>
 *   <li>Population fitness statistics</li>
 *   <li>ASCII convergence &amp; diversity charts</li>
 * </ul>
 *
 * <p>All output uses only standard ASCII characters for maximum compatibility.
 */
@Slf4j
public final class PsoAnalytics {
    // ---------------------------------------------------------------------
    // Parsing helpers
    // ---------------------------------------------------------------------
    private static double[] parseDoubleArray(String s) {
        if (s == null || s.isBlank()) return new double[0];
        String[] parts = s.trim().split("\\s+");
        double[] arr = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            arr[i] = Double.parseDouble(parts[i]);
        }
        return arr;
    }

    private static List<LogEntry> parseLog(Path logPath) throws IOException {
        List<LogEntry> entries = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(logPath, StandardCharsets.UTF_8)) {
            String header = reader.readLine(); // discard header line
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] cols = line.split(";", -1);
                if (cols.length < 11) {
                    log.warn("Skipping malformed line ({} columns): {}", cols.length, line);
                    continue;
                }
                int iteration = Integer.parseInt(cols[0]);
                int agentId = Integer.parseInt(cols[1]);
                double[] posBefore = parseDoubleArray(cols[2]);
                double[] posAfter = parseDoubleArray(cols[3]);
                double[] pbest = parseDoubleArray(cols[4]);
                double pbestFit = Double.parseDouble(cols[5]);
                double[] gbest = parseDoubleArray(cols[6]);
                double gbestFit = Double.parseDouble(cols[7]);
                double popAvg = Double.parseDouble(cols[8]);
                double popStd = Double.parseDouble(cols[9]);
                String distRaw = cols[10];
                entries.add(new LogEntry(iteration, agentId,
                        posBefore, posAfter, pbest, pbestFit,
                        gbest, gbestFit, popAvg, popStd, distRaw));
            }
        }
        return entries;
    }

    private static RunMetrics computeMetrics(List<LogEntry> logEntries) {
        RunMetrics m = new RunMetrics();

        // Group by iteration (TreeMap ensures natural ordering)
        Map<Integer, List<LogEntry>> byIter = logEntries.stream()
                .collect(Collectors.groupingBy(l -> l.iteration, TreeMap::new, Collectors.toList()));

        // --- Basic configuration from the log ---
        List<LogEntry> iterZero = byIter.get(0);
        if (iterZero == null || iterZero.isEmpty()) {
            throw new IllegalStateException("Log does not contain iteration 0 (initial state).");
        }
        m.populationSize = iterZero.size();
        m.dimensions = iterZero.getFirst().positionBefore.length;
        m.maxIterations = byIter.keySet().stream().mapToInt(Integer::intValue).max().orElse(0);
        m.optimumKnown = !"NA".equals(iterZero.getFirst().distToOptimumRaw);

        Double lastGBest = null;
        for (int iter = 0; iter <= m.maxIterations; iter++) {
            List<LogEntry> lines = byIter.get(iter);
            if (lines == null || lines.isEmpty()) {
                log.warn("Missing entries for iteration {} – using last known values", iter);
                if (lastGBest != null) {
                    m.globalBestHistory.add(lastGBest);
                    m.popAvgHistory.add(m.popAvgHistory.isEmpty() ? Double.NaN : m.popAvgHistory.getLast());
                    m.popStdHistory.add(m.popStdHistory.isEmpty() ? Double.NaN : m.popStdHistory.getLast());
                    if (m.optimumKnown) {
                        m.distanceToOptimumHistory.add(m.distanceToOptimumHistory.isEmpty() ? Double.NaN : m.distanceToOptimumHistory.getLast());
                    }
                }
                continue;
            }
            LogEntry first = lines.getFirst();
            double gBest = first.globalBestFitness;
            m.globalBestHistory.add(gBest);
            lastGBest = gBest;
            m.popAvgHistory.add(first.popAvgFitness);
            m.popStdHistory.add(first.popStdDev);
            if (m.optimumKnown && !"NA".equals(first.distToOptimumRaw)) {
                m.distanceToOptimumHistory.add(Double.parseDouble(first.distToOptimumRaw));
            }
        }
        m.finalBestFitness = m.globalBestHistory.getLast();

        // Global best updates
        m.globalBestUpdates = 0;
        for (int i = 1; i < m.globalBestHistory.size(); i++) {
            if (m.globalBestHistory.get(i) < m.globalBestHistory.get(i - 1) - 1e-15) {
                m.globalBestUpdates++;
            }
        }

        // Convergence rate (geometric mean)
        double initialFitness = m.globalBestHistory.getFirst();
        if (initialFitness > m.finalBestFitness && m.finalBestFitness > 0) {
            m.convergenceRate = Math.pow(m.finalBestFitness / initialFitness, 1.0 / m.maxIterations);
        } else {
            m.convergenceRate = 1.0;
        }

        // Iterations to reach 1e-6 of initial fitness
        double threshold = initialFitness * 1e-6;
        for (int i = 0; i < m.globalBestHistory.size(); i++) {
            if (m.globalBestHistory.get(i) <= threshold) {
                m.iterationsToThreshold = i;
                break;
            }
        }

        // Stagnation detection (no improvement in the last 10% of iterations)
        int lookback = Math.max(1, (int) (m.maxIterations * 0.1));
        int lastImprovement = m.maxIterations;
        for (int i = m.maxIterations; i > m.maxIterations - lookback; i--) {
            if (i > 0 && m.globalBestHistory.get(i) < m.globalBestHistory.get(i - 1) - 1e-15) {
                lastImprovement = i;
                break;
            }
        }
        if (lastImprovement < m.maxIterations - lookback + 1) {
            m.stagnationStart = lastImprovement;
        }

        // Diversity & velocity magnitude per iteration
        for (int iter = 0; iter <= m.maxIterations; iter++) {
            List<LogEntry> lines = byIter.get(iter);
            if (lines == null || lines.isEmpty()) {
                m.avgDistanceHistory.add(m.avgDistanceHistory.isEmpty() ? 0.0 : m.avgDistanceHistory.getLast());
                m.avgVelocityMagHistory.add(m.avgVelocityMagHistory.isEmpty() ? 0.0 : m.avgVelocityMagHistory.getLast());
                continue;
            }
            List<double[]> positions = lines.stream().map(l -> l.positionAfter).toList();

            // Average pairwise distance
            double sumDist = 0.0;
            int pairs = 0;
            for (int i = 0; i < positions.size(); i++) {
                for (int j = i + 1; j < positions.size(); j++) {
                    sumDist += euclidean(positions.get(i), positions.get(j));
                    pairs++;
                }
            }
            m.avgDistanceHistory.add(pairs > 0 ? sumDist / pairs : 0.0);

            // Average velocity magnitude
            double sumVel = 0.0;
            for (LogEntry e : lines) {
                sumVel += euclidean(e.positionBefore, e.positionAfter);
            }
            m.avgVelocityMagHistory.add(sumVel / lines.size());
        }

        if (m.optimumKnown && !m.distanceToOptimumHistory.isEmpty()) {
            m.finalDistanceToOptimum = m.distanceToOptimumHistory.getLast();
        }

        return m;
    }

    private static double euclidean(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    // ---------------------------------------------------------------------
    // ASCII chart utility (only standard ASCII characters)
    // ---------------------------------------------------------------------
    private static void printAsciiChart(List<Double> values, int width, int height, String label) {
        if (values.isEmpty()) return;
        double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(1);
        if (max == min) max = min + 1; // avoid division by zero

        int steps = Math.min(values.size(), width);
        double[] display = new double[steps];
        for (int i = 0; i < steps; i++) {
            int idx = (int) ((long) i * values.size() / steps);
            display[i] = values.get(idx);
        }

        for (int row = height - 1; row >= 0; row--) {
            double threshold = min + (max - min) * row / (height - 1);
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("%10.4f |", threshold));
            for (double v : display) {
                if (v >= threshold) {
                    sb.append('#');   // ASCII block
                } else {
                    sb.append(' ');
                }
            }
            System.out.println(sb);
        }
        // x-axis
        System.out.println("           " + "-".repeat(steps));
        System.out.printf("           Iter: 0%" + (steps - 1) + "d\n", values.size() - 1);
        if (!label.isEmpty()) {
            System.out.println("           " + label);
        }
    }

    // ---------------------------------------------------------------------
    // Scientific formatting
    // ---------------------------------------------------------------------
    private static String scientific(double v) {
        if (Double.isNaN(v)) return "NaN";
        if (Double.isInfinite(v)) return v > 0 ? "Infinity" : "-Infinity";
        return String.format(Locale.US, "%.6e", v);
    }

    // ---------------------------------------------------------------------
    // Report generation
    // ---------------------------------------------------------------------
    public static void generateReport(Path logPath) {
        System.out.println("=".repeat(100));
        System.out.println("   PSO ALGORITHM QUALITY EVALUATION REPORT");
        System.out.println("=".repeat(100));
        System.out.println("Log file    : " + logPath.toAbsolutePath());
        System.out.println("Generated   : " + new Date());
        System.out.println();

        try {
            List<LogEntry> logEntries = parseLog(logPath);
            if (logEntries.isEmpty()) {
                System.out.println("ERROR: No valid data found in log file.");
                return;
            }

            RunMetrics m = computeMetrics(logEntries);
            String fmt = "%-35s %s%n";

            // 1. Run Overview
            System.out.println("--- 1. RUN OVERVIEW ---");
            System.out.printf(fmt, "Population size:", m.populationSize);
            System.out.printf(fmt, "Search dimensions:", m.dimensions);
            System.out.printf(fmt, "Maximum iterations:", m.maxIterations);
            System.out.printf(fmt, "Optimum known:", m.optimumKnown ? "Yes" : "No (distance = NA)");
            System.out.println();

            // 2. Final Solution Quality
            System.out.println("--- 2. FINAL SOLUTION QUALITY ---");
            System.out.printf(fmt, "Global best fitness:", scientific(m.finalBestFitness));
            if (m.optimumKnown) {
                System.out.printf(fmt, "Distance to optimum:", scientific(m.finalDistanceToOptimum));
            }
            System.out.println();

            // 3. Convergence Analysis
            System.out.println("--- 3. CONVERGENCE ANALYSIS ---");
            System.out.printf(fmt, "Convergence rate (per iter):", String.format("%.6f", m.convergenceRate));
            if (m.iterationsToThreshold >= 0) {
                System.out.printf(fmt, "Iterations to 1e-6 of initial:", m.iterationsToThreshold + " of " + m.maxIterations);
            } else {
                System.out.printf(fmt, "Iterations to 1e-6 of initial:", "not reached");
            }
            System.out.printf(fmt, "Global best updates:", m.globalBestUpdates);
            if (m.stagnationStart >= 0) {
                System.out.printf(fmt, "Stagnation detected from iter:", m.stagnationStart);
            } else {
                System.out.printf(fmt, "Stagnation:", "none (still improving)");
            }
            System.out.println("\nGlobal best fitness (log10 scale):");
            List<Double> logBest = m.globalBestHistory.stream()
                    .map(v -> Math.log10(Math.max(v, 1e-300)))
                    .collect(Collectors.toList());
            printAsciiChart(logBest, 70, 10, "Log10(global best fitness)");

            // 4. Swarm Diversity
            System.out.println("\n--- 4. SWARM DIVERSITY ---");
            double initDiv = m.avgDistanceHistory.getFirst();
            double finalDiv = m.avgDistanceHistory.getLast();
            System.out.printf(fmt, "Initial avg pairwise distance:", scientific(initDiv));
            System.out.printf(fmt, "Final avg pairwise distance:", scientific(finalDiv));
            System.out.printf(fmt, "Diversity preservation (%):", String.format("%.1f%%", finalDiv / initDiv * 100));
            printAsciiChart(m.avgDistanceHistory, 70, 8, "Average pairwise distance (diversity)");

            // 5. Exploration / Exploitation Balance
            System.out.println("\n--- 5. EXPLORATION / EXPLOITATION BALANCE ---");

            // Estimate search space diameter from initial positions
            List<LogEntry> initial = logEntries.stream().filter(l -> l.iteration == 0).toList();
            double[] minPos = new double[m.dimensions], maxPos = new double[m.dimensions];
            Arrays.fill(minPos, Double.POSITIVE_INFINITY);
            Arrays.fill(maxPos, Double.NEGATIVE_INFINITY);
            for (LogEntry e : initial) {
                for (int d = 0; d < m.dimensions; d++) {
                    if (e.positionBefore[d] < minPos[d]) minPos[d] = e.positionBefore[d];
                    if (e.positionBefore[d] > maxPos[d]) maxPos[d] = e.positionBefore[d];
                }
            }
            double searchDiameter = 0.0;
            for (int d = 0; d < m.dimensions; d++) {
                searchDiameter += Math.pow(maxPos[d] - minPos[d], 2);
            }
            searchDiameter = Math.sqrt(searchDiameter);

            double initVel = m.avgVelocityMagHistory.getFirst();
            double finalVel = m.avgVelocityMagHistory.getLast();
            System.out.printf(fmt, "Search space diameter (est.):", scientific(searchDiameter));
            System.out.printf(fmt, "Initial avg velocity magnitude:", scientific(initVel));
            System.out.printf(fmt, "Final avg velocity magnitude:", scientific(finalVel));
            System.out.printf(fmt, "Exploration ratio (init):", String.format("%.4f", initVel / searchDiameter));
            System.out.printf(fmt, "Exploration ratio (final):", String.format("%.4f", finalVel / searchDiameter));
            printAsciiChart(m.avgVelocityMagHistory, 70, 8, "Average velocity magnitude per iteration");

            // 6. Population Fitness Statistics
            System.out.println("\n--- 6. POPULATION FITNESS STATISTICS ---");
            System.out.printf(fmt, "Final mean fitness:", scientific(m.popAvgHistory.getLast()));
            System.out.printf(fmt, "Final std dev fitness:", scientific(m.popStdHistory.getLast()));
            System.out.println("\nPopulation mean fitness:");
            printAsciiChart(m.popAvgHistory, 70, 8, "Population mean fitness");
            System.out.println("\nPopulation fitness std dev:");
            printAsciiChart(m.popStdHistory, 70, 8, "Population fitness std dev");

            // 7. Distance to Optimum
            if (m.optimumKnown) {
                System.out.println("\n--- 7. DISTANCE TO OPTIMUM ---");
                printAsciiChart(m.distanceToOptimumHistory, 70, 8, "Euclidean distance to optimum");
            }

            System.out.println("\n" + "=".repeat(100));
            System.out.println("Report complete. All metrics are based solely on the execution log.");
            System.out.println("=".repeat(100));
        } catch (IOException e) {
            System.err.println("Failed to read log file: " + e.getMessage());
            log.error("Log read error", e);
        }
    }

    // ---------------------------------------------------------------------
    // Main entry point
    // ---------------------------------------------------------------------
    static void main(String[] args) {
        Path logPath;
        if (args.length > 0) {
            logPath = Paths.get(args[0]);
        } else {
            logPath = Paths.get("algorithm_run.log");
        }
        generateReport(logPath);
    }

    // ---------------------------------------------------------------------
    // Internal data structure for one parsed log line
    // ---------------------------------------------------------------------
    private static class LogEntry {
        final int iteration;
        final int agentId;
        final double[] positionBefore;
        final double[] positionAfter;
        final double[] personalBest;
        final double personalBestFitness;
        final double[] globalBest;
        final double globalBestFitness;
        final double popAvgFitness;
        final double popStdDev;
        final String distToOptimumRaw;   // "NA" or numeric string

        LogEntry(int iteration, int agentId,
                 double[] positionBefore, double[] positionAfter,
                 double[] personalBest, double personalBestFitness,
                 double[] globalBest, double globalBestFitness,
                 double popAvgFitness, double popStdDev,
                 String distToOptimumRaw) {
            this.iteration = iteration;
            this.agentId = agentId;
            this.positionBefore = positionBefore;
            this.positionAfter = positionAfter;
            this.personalBest = personalBest;
            this.personalBestFitness = personalBestFitness;
            this.globalBest = globalBest;
            this.globalBestFitness = globalBestFitness;
            this.popAvgFitness = popAvgFitness;
            this.popStdDev = popStdDev;
            this.distToOptimumRaw = distToOptimumRaw;
        }
    }

    // ---------------------------------------------------------------------
    // Metrics container and computation
    // ---------------------------------------------------------------------
    private static class RunMetrics {
        int populationSize;
        int dimensions;
        int maxIterations;
        boolean optimumKnown;
        double finalBestFitness;
        double finalDistanceToOptimum = Double.NaN;
        int stagnationStart = -1;        // iteration where stagnation began (-1 if none)
        int iterationsToThreshold = -1;  // iterations to reach 1e-6 of initial fitness
        double convergenceRate;          // per-iteration geometric mean improvement
        int globalBestUpdates;           // number of iterations where gBest improved
        List<Double> globalBestHistory = new ArrayList<>();
        List<Double> avgDistanceHistory = new ArrayList<>();
        List<Double> avgVelocityMagHistory = new ArrayList<>();
        List<Double> popAvgHistory = new ArrayList<>();
        List<Double> popStdHistory = new ArrayList<>();
        List<Double> distanceToOptimumHistory = new ArrayList<>();
    }
} 
 
package edu.swarmintelligence.pso.r01;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertAll;

@DisplayName("PSO – global‑best topology")
class PsoAlgorithmTest {
    // ──────────────────────── Helper objective functions ─────────────────────────

    static double sphere(double[] x) {
        return DoubleStream.of(x).map(v -> v * v).sum();
    }

    static double ackley(double[] x) {
        int n = x.length;
        double sumSq = 0.0, sumCos = 0.0;
        for (double v : x) {
            sumSq += v * v;
            sumCos += Math.cos(2.0 * Math.PI * v);
        }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                - Math.exp(sumCos / n) + 20.0 + Math.E;
    }

    // ──────────────────────── Configuration factories ────────────────────────────

    static PsoAlgorithm.PsoConfig easyConfig(long seed) {
        int dim = 2;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return PsoAlgorithm.PsoConfig.builder()
                .populationSize(50)
                .maxIterations(1000)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    static PsoAlgorithm.PsoConfig ackleyConfig(long seed) {
        int dim = 10;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        return PsoAlgorithm.PsoConfig.builder()
                .populationSize(50)
                .maxIterations(500)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    // ──────────────────────── Nested test groups ─────────────────────────────────

    @Nested
    @DisplayName("Correctness & convergence")
    class Convergence {
        @Test
        @DisplayName("Converges close to global optimum on 2‑D Sphere")
        void convergesOnSphere() {
            var cfg = easyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            assertThat(res.bestFitness()).isLessThan(0.01);
        }

        @Test
        @DisplayName("Converges near global optimum on 10‑D Ackley")
        void convergesOnAckley() {
            var cfg = ackleyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::ackley);
            var res = pso.optimize();
            assertThat(res.bestFitness()).isLessThan(0.5);
        }

        @Test
        @DisplayName("Convergence history is monotonically non‑increasing")
        void historyIsMonotonic() {
            var cfg = easyConfig(99);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            var h = res.convergenceHistory();
            for (int i = 1; i < h.length; i++) {
                assertThat(h[i]).isLessThanOrEqualTo(h[i - 1]);
            }
        }

        @Test
        @DisplayName("Convergence history length = maxIterations + 1")
        void historyLength() {
            var cfg = easyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            assertThat(res.convergenceHistory()).hasSize(cfg.maxIterations() + 1);
        }

        @Test
        @DisplayName("Best position remains within search bounds")
        void bestPositionWithinBounds() {
            var cfg = easyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            for (double v : res.bestPosition()) {
                assertThat(v).isBetween(-5.12, 5.12);
            }
        }
    }

    @Nested
    @DisplayName("Determinism")
    class Determinism {
        @RepeatedTest(3)
        @DisplayName("Same seed → identical fitness and convergence history")
        void sameSeed() {
            var cfg = easyConfig(777);
            var pso1 = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var pso2 = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res1 = pso1.optimize();
            var res2 = pso2.optimize();
            assertAll(
                    () -> assertThat(res1.bestFitness()).isEqualTo(res2.bestFitness()),
                    () -> assertThat(res1.convergenceHistory()).containsExactly(res2.convergenceHistory())
            );
        }

        @Test
        @DisplayName("Different seeds → different convergence histories")
        void differentSeedsDiffer() {
            var res1 = new PsoAlgorithm(easyConfig(1), PsoAlgorithmTest::sphere).optimize();
            var res2 = new PsoAlgorithm(easyConfig(2), PsoAlgorithmTest::sphere).optimize();
            assertThat(res1.convergenceHistory())
                    .isNotEqualTo(res2.convergenceHistory());
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class Config {
        @Test
        @DisplayName("Rejects invalid popSize (0)")
        void invalidPopSize() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects inertia outside (0,1]")
        void invalidInertia() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.0).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(1.5).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects negative cognitive/social coefficients")
        void invalidCoefficients() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(-0.1).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(-1.0)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects velocityClampFactor outside (0,1]")
        void invalidVelocityClamp() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.0).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(1.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensive copies of bounds arrays")
        void boundsDefensiveCopy() {
            var min = new double[]{0, 0};
            var max = new double[]{1, 1};
            var cfg = PsoAlgorithm.PsoConfig.builder()
                    .populationSize(2).maxIterations(10).dimensions(2)
                    .minBounds(min).maxBounds(max)
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(0).build();
            min[0] = 999;
            assertThat(cfg.minBounds()[0]).isEqualTo(0.0);
        }
    }

    @Nested
    @DisplayName("Result integrity")
    class ResultIntegrity {
        @Test
        @DisplayName("Result bestPosition is a defensive copy")
        void positionDefensiveCopy() {
            var res = new PsoAlgorithm(easyConfig(42), PsoAlgorithmTest::sphere).optimize();
            var p = res.bestPosition();
            p[0] = Double.MAX_VALUE;
            assertThat(res.bestPosition()[0]).isNotEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Result convergenceHistory is a defensive copy")
        void historyDefensiveCopy() {
            var res = new PsoAlgorithm(easyConfig(42), PsoAlgorithmTest::sphere).optimize();
            var h = res.convergenceHistory();
            h[0] = -999;
            assertThat(res.convergenceHistory()[0]).isNotEqualTo(-999);
        }

        @Test
        @DisplayName("getBestFitness matches Result.bestFitness after optimize()")
        void bestFitnessMatches() {
            var pso = new PsoAlgorithm(easyConfig(42), PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            assertThat(pso.getBestFitness()).isEqualTo(res.bestFitness());
        }
    }

    @Nested
    @DisplayName("Swarm dynamics (velocity & personal best)")
    class Dynamics {
        @Test
        @DisplayName("Initial velocities respect vMax")
        void initialVelocityWithinVMax() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            double[][] vels = pso.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("After iterations velocities remain within vMax (clamping effect)")
        void velocityClampedAfterUpdate() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            pso.optimize();
            double[][] vels = pso.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("Personal best fitness improves or stays the same over time")
        void personalBestNeverDegrades() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            double[] initialPBest = pso.getPersonalBestFitnesses();
            pso.optimize();
            double[] finalPBest = pso.getPersonalBestFitnesses();
            for (int i = 0; i < initialPBest.length; i++) {
                assertThat(finalPBest[i]).isLessThanOrEqualTo(initialPBest[i]);
            }
        }

        @Test
        @DisplayName("Global best fitness is the minimum personal best fitness")
        void globalBestMatchesMinPersonalBest() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            pso.optimize();
            double minPBest = Arrays.stream(pso.getPersonalBestFitnesses()).min().orElse(Double.NaN);
            assertThat(pso.getBestFitness()).isEqualTo(minPBest);
        }

        @Test
        @DisplayName("All particles' positions stay within bounds")
        void positionsWithinBounds() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            pso.optimize();
            double[][] pos = pso.getPositions();
            double lb = cfg.minBounds()[0];
            double ub = cfg.maxBounds()[0];
            for (double[] p : pos) {
                for (double pv : p) {
                    assertThat(pv).isBetween(lb, ub);
                }
            }
        }
    }
} 
 
package edu.swarmintelligence.pso.r02;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertAll;

@DisplayName("PSO – global‑best topology (refactored)")
class PsoOptimizerTest {
    static double sphere(double[] x) {
        return DoubleStream.of(x).map(v -> v * v).sum();
    }

    static double ackley(double[] x) {
        int n = x.length;
        double sumSq = 0.0, sumCos = 0.0;
        for (double v : x) {
            sumSq += v * v;
            sumCos += Math.cos(2.0 * Math.PI * v);
        }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                - Math.exp(sumCos / n) + 20.0 + Math.E;
    }

    static PsoConfig easyConfig(long seed) {
        int dim = 2;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return PsoConfig.builder()
                .populationSize(50)
                .maxIterations(1000)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    static PsoConfig ackleyConfig(long seed) {
        int dim = 10;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        return PsoConfig.builder()
                .populationSize(50)
                .maxIterations(500)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    @Nested
    @DisplayName("Correctness & convergence")
    class Convergence {
        @Test
        @DisplayName("Converges close to global optimum on 2‑D Sphere")
        void convergesOnSphere() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.01);
        }

        @Test
        @DisplayName("Converges near global optimum on 10‑D Ackley")
        void convergesOnAckley() {
            var optimizer = new PsoOptimizer(ackleyConfig(1), PsoOptimizerTest::ackley);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.5);
        }

        @Test
        @DisplayName("Convergence history is monotonically non‑increasing")
        void historyIsMonotonic() {
            var optimizer = new PsoOptimizer(easyConfig(99), PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            var h = result.convergenceHistory();
            for (int i = 1; i < h.length; i++) {
                assertThat(h[i]).isLessThanOrEqualTo(h[i - 1]);
            }
        }

        @Test
        @DisplayName("Convergence history length = maxIterations + 1")
        void historyLength() {
            var cfg = easyConfig(1);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            assertThat(result.convergenceHistory()).hasSize(cfg.maxIterations() + 1);
        }

        @Test
        @DisplayName("Best position remains within search bounds")
        void bestPositionWithinBounds() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            for (double v : result.bestPosition()) {
                assertThat(v).isBetween(-5.12, 5.12);
            }
        }
    }

    @Nested
    @DisplayName("Determinism")
    class Determinism {
        @RepeatedTest(3)
        @DisplayName("Same seed → identical fitness and convergence history")
        void sameSeed() {
            var cfg = easyConfig(777);
            var optimizer1 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            var optimizer2 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            var res1 = optimizer1.optimize();
            var res2 = optimizer2.optimize();
            assertAll(
                    () -> assertThat(res1.bestFitness()).isEqualTo(res2.bestFitness()),
                    () -> assertThat(res1.convergenceHistory()).containsExactly(res2.convergenceHistory())
            );
        }

        @Test
        @DisplayName("Different seeds → different convergence histories")
        void differentSeedsDiffer() {
            var res1 = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere).optimize();
            var res2 = new PsoOptimizer(easyConfig(2), PsoOptimizerTest::sphere).optimize();
            assertThat(res1.convergenceHistory())
                    .isNotEqualTo(res2.convergenceHistory());
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class Config {
        @Test
        @DisplayName("Rejects invalid popSize (0)")
        void invalidPopSize() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects inertia outside (0,1]")
        void invalidInertia() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.0).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(1.5).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects negative cognitive/social coefficients")
        void invalidCoefficients() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(-0.1).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(-1.0)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects velocityClampFactor outside (0,1]")
        void invalidVelocityClamp() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.0).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(1.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensive copies of bounds arrays")
        void boundsDefensiveCopy() {
            var min = new double[]{0, 0};
            var max = new double[]{1, 1};
            var cfg = PsoConfig.builder()
                    .populationSize(2).maxIterations(10).dimensions(2)
                    .minBounds(min).maxBounds(max)
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(0).build();
            min[0] = 999;
            assertThat(cfg.minBounds()[0]).isEqualTo(0.0);
        }
    }

    @Nested
    @DisplayName("Result integrity")
    class ResultIntegrity {
        @Test
        @DisplayName("Result bestPosition is a defensive copy")
        void positionDefensiveCopy() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere);
            var res = optimizer.optimize();
            var p = res.bestPosition();
            p[0] = Double.MAX_VALUE;
            assertThat(res.bestPosition()[0]).isNotEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Result convergenceHistory is a defensive copy")
        void historyDefensiveCopy() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere);
            var res = optimizer.optimize();
            var h = res.convergenceHistory();
            h[0] = -999;
            assertThat(res.convergenceHistory()[0]).isNotEqualTo(-999);
        }

        @Test
        @DisplayName("getBestFitness matches Result.bestFitness after optimize()")
        void bestFitnessMatches() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere);
            var res = optimizer.optimize();
            assertThat(optimizer.getBestFitness()).isEqualTo(res.bestFitness());
        }
    }

    @Nested
    @DisplayName("Swarm dynamics (velocity & personal best)")
    class Dynamics {
        @Test
        @DisplayName("Initial velocities respect vMax")
        void initialVelocityWithinVMax() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("After iterations velocities remain within vMax (clamping effect)")
        void velocityClampedAfterUpdate() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("Personal best fitness improves or stays the same over time")
        void personalBestNeverDegrades() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            Swarm swarm = optimizer.getSwarm();
            double[] initialPBest = swarm.getPersonalBestFitnesses();
            optimizer.optimize();
            double[] finalPBest = swarm.getPersonalBestFitnesses();
            for (int i = 0; i < initialPBest.length; i++) {
                assertThat(finalPBest[i]).isLessThanOrEqualTo(initialPBest[i]);
            }
        }

        @Test
        @DisplayName("Global best fitness is the minimum personal best fitness")
        void globalBestMatchesMinPersonalBest() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double minPBest = Arrays.stream(swarm.getPersonalBestFitnesses()).min().orElse(Double.NaN);
            assertThat(swarm.getGlobalBestFitness()).isEqualTo(minPBest);
        }

        @Test
        @DisplayName("All particles' positions stay within bounds")
        void positionsWithinBounds() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] pos = swarm.getPositions();
            double lb = cfg.minBounds()[0];
            double ub = cfg.maxBounds()[0];
            for (double[] p : pos) {
                for (double pv : p) {
                    assertThat(pv).isBetween(lb, ub);
                }
            }
        }
    }
} 
 
package edu.swarmintelligence.pso.r03;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertAll;

@DisplayName("PSO – global‑best topology (refactored)")
class PsoOptimizerTest {
    static double sphere(double[] x) {
        return DoubleStream.of(x).map(v -> v * v).sum();
    }

    static double ackley(double[] x) {
        int n = x.length;
        double sumSq = 0.0, sumCos = 0.0;
        for (double v : x) {
            sumSq += v * v;
            sumCos += Math.cos(2.0 * Math.PI * v);
        }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                - Math.exp(sumCos / n) + 20.0 + Math.E;
    }

    static PsoConfig easyConfig(long seed) {
        int dim = 2;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return PsoConfig.builder()
                .populationSize(50)
                .maxIterations(1000)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    static PsoConfig ackleyConfig(long seed) {
        int dim = 10;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        return PsoConfig.builder()
                .populationSize(50)
                .maxIterations(500)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    @Nested
    @DisplayName("Correctness & convergence")
    class Convergence {
        @Test
        @DisplayName("Converges close to global optimum on 2‑D Sphere")
        void convergesOnSphere() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.01);
        }

        @Test
        @DisplayName("Converges near global optimum on 10‑D Ackley")
        void convergesOnAckley() {
            var optimizer = new PsoOptimizer(ackleyConfig(1), PsoOptimizerTest::ackley);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.5);
        }

        @Test
        @DisplayName("Convergence history is monotonically non‑increasing")
        void historyIsMonotonic() {
            var optimizer = new PsoOptimizer(easyConfig(99), PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            var h = result.convergenceHistory();
            for (int i = 1; i < h.length; i++) {
                assertThat(h[i]).isLessThanOrEqualTo(h[i - 1]);
            }
        }

        @Test
        @DisplayName("Convergence history length = maxIterations + 1")
        void historyLength() {
            var cfg = easyConfig(1);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            assertThat(result.convergenceHistory()).hasSize(cfg.maxIterations() + 1);
        }

        @Test
        @DisplayName("Best position remains within search bounds")
        void bestPositionWithinBounds() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere);
            var result = optimizer.optimize();
            for (double v : result.bestPosition()) {
                assertThat(v).isBetween(-5.12, 5.12);
            }
        }
    }

    @Nested
    @DisplayName("Determinism")
    class Determinism {
        @RepeatedTest(3)
        @DisplayName("Same seed → identical fitness and convergence history")
        void sameSeed() {
            var cfg = easyConfig(777);
            var optimizer1 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            var optimizer2 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            var res1 = optimizer1.optimize();
            var res2 = optimizer2.optimize();
            assertAll(
                    () -> assertThat(res1.bestFitness()).isEqualTo(res2.bestFitness()),
                    () -> assertThat(res1.convergenceHistory()).containsExactly(res2.convergenceHistory())
            );
        }

        @Test
        @DisplayName("Different seeds → different convergence histories")
        void differentSeedsDiffer() {
            var res1 = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere).optimize();
            var res2 = new PsoOptimizer(easyConfig(2), PsoOptimizerTest::sphere).optimize();
            assertThat(res1.convergenceHistory())
                    .isNotEqualTo(res2.convergenceHistory());
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class Config {
        @Test
        @DisplayName("Rejects invalid popSize (0)")
        void invalidPopSize() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects inertia outside (0,1]")
        void invalidInertia() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.0).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(1.5).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects negative cognitive/social coefficients")
        void invalidCoefficients() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(-0.1).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(-1.0)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects velocityClampFactor outside (0,1]")
        void invalidVelocityClamp() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.0).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(1.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensive copies of bounds arrays")
        void boundsDefensiveCopy() {
            var min = new double[]{0, 0};
            var max = new double[]{1, 1};
            var cfg = PsoConfig.builder()
                    .populationSize(2).maxIterations(10).dimensions(2)
                    .minBounds(min).maxBounds(max)
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(0).build();
            min[0] = 999;
            assertThat(cfg.minBounds()[0]).isEqualTo(0.0);
        }
    }

    @Nested
    @DisplayName("Result integrity")
    class ResultIntegrity {
        @Test
        @DisplayName("Result bestPosition is a defensive copy")
        void positionDefensiveCopy() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere);
            var res = optimizer.optimize();
            var p = res.bestPosition();
            p[0] = Double.MAX_VALUE;
            assertThat(res.bestPosition()[0]).isNotEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Result convergenceHistory is a defensive copy")
        void historyDefensiveCopy() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere);
            var res = optimizer.optimize();
            var h = res.convergenceHistory();
            h[0] = -999;
            assertThat(res.convergenceHistory()[0]).isNotEqualTo(-999);
        }

        @Test
        @DisplayName("getBestFitness matches Result.bestFitness after optimize()")
        void bestFitnessMatches() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere);
            var res = optimizer.optimize();
            assertThat(optimizer.getBestFitness()).isEqualTo(res.bestFitness());
        }
    }

    @Nested
    @DisplayName("Swarm dynamics (velocity & personal best)")
    class Dynamics {
        @Test
        @DisplayName("Initial velocities respect vMax")
        void initialVelocityWithinVMax() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("After iterations velocities remain within vMax (clamping effect)")
        void velocityClampedAfterUpdate() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("Personal best fitness improves or stays the same over time")
        void personalBestNeverDegrades() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            Swarm swarm = optimizer.getSwarm();
            double[] initialPBest = swarm.getPersonalBestFitnesses();
            optimizer.optimize();
            double[] finalPBest = swarm.getPersonalBestFitnesses();
            for (int i = 0; i < initialPBest.length; i++) {
                assertThat(finalPBest[i]).isLessThanOrEqualTo(initialPBest[i]);
            }
        }

        @Test
        @DisplayName("Global best fitness is the minimum personal best fitness")
        void globalBestMatchesMinPersonalBest() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double minPBest = Arrays.stream(swarm.getPersonalBestFitnesses()).min().orElse(Double.NaN);
            assertThat(swarm.getGlobalBestFitness()).isEqualTo(minPBest);
        }

        @Test
        @DisplayName("All particles' positions stay within bounds")
        void positionsWithinBounds() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] pos = swarm.getPositions();
            double lb = cfg.minBounds()[0];
            double ub = cfg.maxBounds()[0];
            for (double[] p : pos) {
                for (double pv : p) {
                    assertThat(pv).isBetween(lb, ub);
                }
            }
        }
    }

    @Nested
    @DisplayName("Invalid fitness handling")
    class InvalidFitness {
        // Objective function that returns NaN for a region (x[0] > 2)
        static double partiallyInvalid(double[] x) {
            if (x[0] > 2.0) return Double.NaN;
            return x[0] * x[0] + x[1] * x[1]; // sphere
        }

        @Test
        @DisplayName("Non‑finite fitness does not update personal / global best and particle fitness becomes ∞")
        void nonFiniteFitnessSkipped() {
            var cfg = easyConfig(42); // deterministic seed
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.01);

            Swarm swarm = optimizer.getSwarm();
            double[] pBests = swarm.getPersonalBestFitnesses();
            // Version‑independent check: all personal bests must be finite
            assertThat(Arrays.stream(pBests)).allMatch(Double::isFinite);
        }

        @Test
        @DisplayName("Convergence history never contains NaN or Infinity")
        void historyNoNaN() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid);
            var result = optimizer.optimize();
            assertThat(Arrays.stream(result.convergenceHistory()))
                    .allMatch(Double::isFinite);
        }

        @Test
        @DisplayName("Global best is never updated from an invalid particle")
        void globalBestNeverInvalid() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid);
            optimizer.optimize();
            assertThat(optimizer.getBestFitness()).isFinite();
        }
    }
} 
 
package edu.swarmintelligence.pso.r04;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertAll;

@DisplayName("PSO – global‑best topology (refactored, with logging)")
class PsoOptimizerTest {
    static double sphere(double[] x) {
        return DoubleStream.of(x).map(v -> v * v).sum();
    }

    static double ackley(double[] x) {
        int n = x.length;
        double sumSq = 0.0, sumCos = 0.0;
        for (double v : x) {
            sumSq += v * v;
            sumCos += Math.cos(2.0 * Math.PI * v);
        }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                - Math.exp(sumCos / n) + 20.0 + Math.E;
    }

    static PsoConfig easyConfig(long seed) {
        int dim = 2;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return PsoConfig.builder()
                .populationSize(50)
                .maxIterations(1000)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    static PsoConfig ackleyConfig(long seed) {
        int dim = 10;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        return PsoConfig.builder()
                .populationSize(50)
                .maxIterations(500)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    // ─── existing tests (adapted to no‑logging constructor) ───────────────

    @Nested
    @DisplayName("Correctness & convergence")
    class Convergence {
        @Test
        @DisplayName("Converges close to global optimum on 2‑D Sphere")
        void convergesOnSphere() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.01);
        }

        @Test
        @DisplayName("Converges near global optimum on 10‑D Ackley")
        void convergesOnAckley() {
            var optimizer = new PsoOptimizer(ackleyConfig(1), PsoOptimizerTest::ackley, null);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.5);
        }

        @Test
        @DisplayName("Convergence history is monotonically non‑increasing")
        void historyIsMonotonic() {
            var optimizer = new PsoOptimizer(easyConfig(99), PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            var h = result.convergenceHistory();
            for (int i = 1; i < h.length; i++) {
                assertThat(h[i]).isLessThanOrEqualTo(h[i - 1]);
            }
        }

        @Test
        @DisplayName("Convergence history length = maxIterations + 1")
        void historyLength() {
            var cfg = easyConfig(1);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            assertThat(result.convergenceHistory()).hasSize(cfg.maxIterations() + 1);
        }

        @Test
        @DisplayName("Best position remains within search bounds")
        void bestPositionWithinBounds() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            for (double v : result.bestPosition()) {
                assertThat(v).isBetween(-5.12, 5.12);
            }
        }
    }

    @Nested
    @DisplayName("Determinism")
    class Determinism {
        @RepeatedTest(3)
        @DisplayName("Same seed → identical fitness and convergence history")
        void sameSeed() {
            var cfg = easyConfig(777);
            var optimizer1 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            var optimizer2 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            var res1 = optimizer1.optimize();
            var res2 = optimizer2.optimize();
            assertAll(
                    () -> assertThat(res1.bestFitness()).isEqualTo(res2.bestFitness()),
                    () -> assertThat(res1.convergenceHistory()).containsExactly(res2.convergenceHistory())
            );
        }

        @Test
        @DisplayName("Different seeds → different convergence histories")
        void differentSeedsDiffer() {
            var res1 = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere, null).optimize();
            var res2 = new PsoOptimizer(easyConfig(2), PsoOptimizerTest::sphere, null).optimize();
            assertThat(res1.convergenceHistory())
                    .isNotEqualTo(res2.convergenceHistory());
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class Config {
        @Test
        @DisplayName("Rejects invalid popSize (0)")
        void invalidPopSize() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects inertia outside (0,1]")
        void invalidInertia() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.0).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(1.5).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects negative cognitive/social coefficients")
        void invalidCoefficients() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(-0.1).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(-1.0)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects velocityClampFactor outside (0,1]")
        void invalidVelocityClamp() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.0).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(1.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects knownOptimum with wrong length")
        void invalidKnownOptimum() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0)
                    .knownOptimum(new double[]{0})   // length 1 ≠ 2
                    .build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensive copies of bounds arrays")
        void boundsDefensiveCopy() {
            var min = new double[]{0, 0};
            var max = new double[]{1, 1};
            var cfg = PsoConfig.builder()
                    .populationSize(2).maxIterations(10).dimensions(2)
                    .minBounds(min).maxBounds(max)
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(0).build();
            min[0] = 999;
            assertThat(cfg.minBounds()[0]).isEqualTo(0.0);
        }
    }

    @Nested
    @DisplayName("Result integrity")
    class ResultIntegrity {
        @Test
        @DisplayName("Result bestPosition is a defensive copy")
        void positionDefensiveCopy() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere, null);
            var res = optimizer.optimize();
            var p = res.bestPosition();
            p[0] = Double.MAX_VALUE;
            assertThat(res.bestPosition()[0]).isNotEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Result convergenceHistory is a defensive copy")
        void historyDefensiveCopy() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere, null);
            var res = optimizer.optimize();
            var h = res.convergenceHistory();
            h[0] = -999;
            assertThat(res.convergenceHistory()[0]).isNotEqualTo(-999);
        }

        @Test
        @DisplayName("getBestFitness matches Result.bestFitness after optimize()")
        void bestFitnessMatches() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere, null);
            var res = optimizer.optimize();
            assertThat(optimizer.getBestFitness()).isEqualTo(res.bestFitness());
        }
    }

    @Nested
    @DisplayName("Swarm dynamics (velocity & personal best)")
    class Dynamics {
        @Test
        @DisplayName("Initial velocities respect vMax")
        void initialVelocityWithinVMax() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("After iterations velocities remain within vMax (clamping effect)")
        void velocityClampedAfterUpdate() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("Personal best fitness improves or stays the same over time")
        void personalBestNeverDegrades() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            Swarm swarm = optimizer.getSwarm();
            double[] initialPBest = swarm.getPersonalBestFitnesses();
            optimizer.optimize();
            double[] finalPBest = swarm.getPersonalBestFitnesses();
            for (int i = 0; i < initialPBest.length; i++) {
                assertThat(finalPBest[i]).isLessThanOrEqualTo(initialPBest[i]);
            }
        }

        @Test
        @DisplayName("Global best fitness is the minimum personal best fitness")
        void globalBestMatchesMinPersonalBest() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double minPBest = Arrays.stream(swarm.getPersonalBestFitnesses()).min().orElse(Double.NaN);
            assertThat(swarm.getGlobalBestFitness()).isEqualTo(minPBest);
        }

        @Test
        @DisplayName("All particles' positions stay within bounds")
        void positionsWithinBounds() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] pos = swarm.getPositions();
            double lb = cfg.minBounds()[0];
            double ub = cfg.maxBounds()[0];
            for (double[] p : pos) {
                for (double pv : p) {
                    assertThat(pv).isBetween(lb, ub);
                }
            }
        }
    }

    @Nested
    @DisplayName("Invalid fitness handling")
    class InvalidFitness {
        static double partiallyInvalid(double[] x) {
            if (x[0] > 2.0) return Double.NaN;
            return x[0] * x[0] + x[1] * x[1];
        }

        @Test
        @DisplayName("Non‑finite fitness does not update personal/global best and particle fitness becomes ∞")
        void nonFiniteFitnessSkipped() {
            var cfg = easyConfig(42);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid, null);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.01);
            Swarm swarm = optimizer.getSwarm();
            double[] pBests = swarm.getPersonalBestFitnesses();
            assertThat(Arrays.stream(pBests)).allMatch(Double::isFinite);
        }

        @Test
        @DisplayName("Convergence history never contains NaN or Infinity")
        void historyNoNaN() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid, null);
            var result = optimizer.optimize();
            assertThat(Arrays.stream(result.convergenceHistory()))
                    .allMatch(Double::isFinite);
        }

        @Test
        @DisplayName("Global best is never updated from an invalid particle")
        void globalBestNeverInvalid() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid, null);
            optimizer.optimize();
            assertThat(optimizer.getBestFitness()).isFinite();
        }
    }

    // ─── new logging tests ────────────────────────────────────────────────
    @Nested
    @DisplayName("Execution log file")
    class Logging {

        @TempDir
        Path tempDir;

        private PsoConfig configWithKnownOptimum() {
            int dim = 2;
            var min = new double[dim];
            var max = new double[dim];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);
            double[] optimum = {0.0, 0.0};
            return PsoConfig.builder()
                    .populationSize(10)
                    .maxIterations(5)
                    .dimensions(dim)
                    .minBounds(min)
                    .maxBounds(max)
                    .inertia(0.7298)
                    .cognitiveCoefficient(1.49618)
                    .socialCoefficient(1.49618)
                    .velocityClampFactor(0.5)
                    .seed(42)
                    .knownOptimum(optimum)
                    .build();
        }

        @Test
        @DisplayName("Log file is created, contains header and correct number of lines")
        void logFileCreatedWithCorrectLineCount() throws IOException {
            Path logFile = tempDir.resolve("test_run.log");
            var config = configWithKnownOptimum();
            var optimizer = new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile);
            optimizer.optimize();

            assertThat(logFile).exists().isRegularFile();
            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            int expectedLineCount = 1 + (config.maxIterations() + 1) * config.populationSize();
            assertThat(lines).hasSize(expectedLineCount);
            assertThat(lines.getFirst()).isEqualTo(
                    "iteration;agentId;positionBefore;positionAfter;personalBest;personalBestFitness;globalBest;globalBestFitness;popAvgFitness;popStdDev;distToOptimum");
        }

        @Test
        @DisplayName("Log lines are correctly formatted and values are consistent")
        void logContentFormatAndConsistency() throws IOException {
            Path logFile = tempDir.resolve("format.log");
            var config = configWithKnownOptimum();
            new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile).optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            // first data line (iteration 0, agent 0)
            String[] parts = lines.get(1).split(";");
            assertThat(parts).hasSize(11);
            assertThat(Integer.parseInt(parts[0])).isEqualTo(0);
            assertThat(Integer.parseInt(parts[1])).isEqualTo(0);
            // positionBefore == positionAfter for initial snapshot
            assertThat(parts[2]).isEqualTo(parts[3]);
            double pBest = Double.parseDouble(parts[5]);
            double gBest = Double.parseDouble(parts[7]);
            assertThat(Double.isFinite(pBest)).isTrue();
            assertThat(Double.isFinite(gBest)).isTrue();
            assertThat(parts[10]).isNotEqualTo("NA");
            double dist = Double.parseDouble(parts[10]);
            assertThat(dist).isGreaterThanOrEqualTo(0);
        }

        @Test
        @DisplayName("Distance to optimum decreases over iterations for sphere function")
        void distanceDecreasesForSphere() throws IOException {
            Path logFile = tempDir.resolve("dist.log");
            var config = configWithKnownOptimum();
            new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile).optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            int pop = config.populationSize();
            double[] avgDistPerIter = new double[config.maxIterations() + 1];
            for (int i = 1; i < lines.size(); i++) {   // skip header
                String[] parts = lines.get(i).split(";");
                int iter = Integer.parseInt(parts[0]);
                double d = Double.parseDouble(parts[10]);
                avgDistPerIter[iter] += d;
            }
            for (int i = 0; i <= config.maxIterations(); i++) {
                avgDistPerIter[i] /= pop;
            }
            assertThat(avgDistPerIter[config.maxIterations()])
                    .isLessThan(avgDistPerIter[0]);
        }

        @Test
        @DisplayName("Distance field is 'NA' when knownOptimum is null")
        void distanceNAWhenOptimumNull() throws IOException {
            Path logFile = tempDir.resolve("na.log");
            var cfg = PsoConfig.builder()
                    .populationSize(5).maxIterations(2).dimensions(2)
                    .minBounds(new double[]{-1, -1}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(123)
                    .knownOptimum(null)
                    .build();
            new PsoOptimizer(cfg, PsoOptimizerTest::sphere, logFile).optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            for (int i = 1; i < lines.size(); i++) {
                String[] parts = lines.get(i).split(";");
                assertThat(parts[10]).isEqualTo("NA");
            }
        }

        @Test
        @DisplayName("Population statistics are correct (avg & std dev of finite fitnesses)")
        void populationStatsCorrect() throws IOException {
            Path logFile = tempDir.resolve("stats.log");
            var config = PsoConfig.builder()
                    .populationSize(3).maxIterations(2).dimensions(1)
                    .minBounds(new double[]{-5}).maxBounds(new double[]{5})
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(999)
                    .knownOptimum(new double[]{0})
                    .build();
            var optimizer = new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile);
            Swarm swarmBefore = optimizer.getSwarm();
            double[][] initPositions = swarmBefore.getPositions();
            double[] fit = new double[3];
            for (int i = 0; i < 3; i++) {
                fit[i] = initPositions[i][0] * initPositions[i][0];
            }
            double expectedAvg = (fit[0] + fit[1] + fit[2]) / 3.0;
            double var = 0.0;
            for (double f : fit) var += (f - expectedAvg) * (f - expectedAvg);
            var /= 3.0;
            double expectedStd = Math.sqrt(var);

            optimizer.optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            String[] parts = lines.get(1).split(";");   // iteration 0, agent 0
            double loggedAvg = Double.parseDouble(parts[8]);
            double loggedStd = Double.parseDouble(parts[9]);
            assertThat(loggedAvg).isCloseTo(expectedAvg, offset(1e-12));
            assertThat(loggedStd).isCloseTo(expectedStd, offset(1e-12));
        }

        @Test
        @DisplayName("Invalid fitness particles are excluded from population statistics")
        void invalidFitnessExcludedFromStats() throws IOException {
            Path logFile = tempDir.resolve("invalid.log");
            var config = PsoConfig.builder()
                    .populationSize(4).maxIterations(1).dimensions(1)
                    .minBounds(new double[]{-10}).maxBounds(new double[]{10})
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(111)
                    .knownOptimum(new double[]{0})
                    .build();

            ObjectiveFunction tricky = x -> {
                if (x[0] > 5.0) return Double.NaN;
                return x[0] * x[0];
            };
            var optimizer = new PsoOptimizer(config, tricky, logFile);
            optimizer.optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            // iteration 1 lines: indices 5..8 (header + 4 initial lines)
            double[] currentFits = new double[4];
            boolean[] valid = new boolean[4];
            for (int i = 0; i < 4; i++) {
                String[] parts = lines.get(5 + i).split(";");
                double pos = Double.parseDouble(parts[3].trim()); // 1D – positionAfter is a single number
                double fit = tricky.evaluate(new double[]{pos});
                currentFits[i] = fit;
                valid[i] = Double.isFinite(fit);
            }

            double sum = 0.0;
            int cnt = 0;
            for (int i = 0; i < 4; i++)
                if (valid[i]) {
                    sum += currentFits[i];
                    cnt++;
                }
            double expectedAvg = cnt > 0 ? sum / cnt : Double.NaN;
            double var = 0.0;
            for (int i = 0; i < 4; i++)
                if (valid[i]) var += (currentFits[i] - expectedAvg) * (currentFits[i] - expectedAvg);
            double expectedStd = cnt > 0 ? Math.sqrt(var / cnt) : Double.NaN;

            String[] firstLineIter1 = lines.get(5).split(";");
            double loggedAvg = Double.parseDouble(firstLineIter1[8]);
            double loggedStd = Double.parseDouble(firstLineIter1[9]);
            assertThat(loggedAvg).isCloseTo(expectedAvg, offset(1e-12));
            assertThat(loggedStd).isCloseTo(expectedStd, offset(1e-12));
        }
    }
} 
