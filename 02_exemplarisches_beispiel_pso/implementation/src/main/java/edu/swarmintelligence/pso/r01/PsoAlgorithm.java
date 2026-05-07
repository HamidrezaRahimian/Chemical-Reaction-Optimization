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