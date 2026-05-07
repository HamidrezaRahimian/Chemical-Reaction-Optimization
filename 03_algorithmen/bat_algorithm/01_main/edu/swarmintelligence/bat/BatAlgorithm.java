package edu.swarmintelligence.bat;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.DoubleStream;

/**
 * Bat Algorithm (BA) — echolocation‑based swarm metaheuristic.
 *
 * <p>Reference: Yang, X.-S. (2010). <i>A New Metaheuristic Bat‑Inspired Algorithm</i>.
 * In Nature Inspired Cooperative Strategies for Optimization (NICSO 2010),
 * pp. 65–74. DOI: 10.1007/978-3-642-16940-3_6.
 *
 * <h2>Algorithm steps (per generation)</h2>
 * <ol>
 *   <li><b>Frequency</b> ： fᵢ = fₘᵢₙ + (fₘₐₓ − fₘᵢₙ) · U(0,1)</li>
 *   <li><b>Velocity</b> ： vᵢ ← vᵢ + (xᵢ − x∗) · fᵢ</li>
 *   <li><b>Position</b> ： x′ᵢ = xᵢ + vᵢ (clamped to bounds)</li>
 *   <li><b>Local random walk</b> (exploitation)：
 *       if U(0,1) &gt; rᵢ then x′ᵢ = x∗ + ε · Ā， ε∈[−1,1]</li>
 *   <li><b>Acceptance</b> ：
 *       if U(0,1) &lt; Aᵢ ∧ f(x′ᵢ) &lt; f(x∗) then adopt x′ᵢ and update x∗</li>
 *   <li><b>Acoustic update</b> (only for accepted bats)：
 *       Aᵢ ← α·Aᵢ， rᵢ ← r⁰ᵢ·(1−exp(−γ·t))</li>
 * </ol>
 *
 * <h2>Implementation notes</h2>
 * <ul>
 *   <li>PRNG: {@code L128X1024MixRandom} — deterministic with fixed seed.</li>
 *   <li>The average loudness {@code Ā} is recomputed per iteration from the
 *       current swarm (real‑time values).</li>
 *   <li>Velocity is <b>not</b> reset after acceptance – the momentum term
 *       remains unchanged (Yang 2010, eq. 3).</li>
 *   <li>Initial pulse rate is always 0; the user‑supplied {@code pulseRateMax}
 *       is the asymptotic maximum r⁰ᵢ.</li>
 *   <li>Java 21‑ready ： uses {@code Math.clamp} and modern {@code RandomGenerator} API.</li>
 * </ul>
 */
@Slf4j
public final class BatAlgorithm {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";   // period ~ 2¹⁰²⁴

    private final BaConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final Bat[] population;
    private double[] globalBestPosition;
    private double globalBestFitness = Double.POSITIVE_INFINITY;

    /**
     * Creates a new optimizer. Immediately initializes the population.
     *
     * @param config   validated configuration (non‑null)
     * @param function objective to minimise (non‑null)
     */
    public BatAlgorithm(@NonNull BaConfig config, @NonNull ObjectiveFunction function) {
        this.config = config;
        this.function = function;
        this.random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.population = new Bat[config.populationSize()];
        initializePopulation();
    }

    // ───────────────────────── Phase 0 : Initialisation ─────────────────────────

    private void initializePopulation() {
        final var min = config.minBounds();
        final var max = config.maxBounds();
        final int n = config.dimensions();

        for (int i = 0; i < config.populationSize(); i++) {
            final double[] x = new double[n];
            for (int d = 0; d < n; d++)
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);

            double fVal = function.evaluate(x);
            if (!Double.isFinite(fVal)) {
                log.warn("Bat {} initialised with invalid fitness → ∞", i);
                fVal = Double.POSITIVE_INFINITY;
            }

            final double freq = config.fMin() + (config.fMax() - config.fMin()) * random.nextDouble();
            population[i] = new Bat(x, fVal, config.initialLoudness(), config.pulseRateMax(), freq);

            if (fVal < globalBestFitness) {
                globalBestFitness = fVal;
                globalBestPosition = x.clone();
            }
        }
        log.debug("Population {} started. Best fitness = {}", config.populationSize(), globalBestFitness);
    }

    // ────────────────────────── Main optimisation loop ──────────────────────────

    /**
     * Runs the BA for the configured number of iterations.
     *
     * @return an immutable summary of the run (best position, fitness, convergence history)
     */
    public Result optimize() {
        final int n = config.dimensions();
        final var min = config.minBounds();
        final var max = config.maxBounds();
        final double[] history = new double[config.maxIterations() + 1];
        history[0] = globalBestFitness;

        for (int t = 1; t <= config.maxIterations(); t++) {
            // Average loudness of the whole swarm (Ā in the paper)
            final double avgLoudness = Arrays.stream(population)
                    .mapToDouble(Bat::getLoudness)
                    .average()
                    .orElse(0.0);

            for (final Bat bat : population) {
                // 1. Frequency
                final double fi = config.fMin() + (config.fMax() - config.fMin()) * random.nextDouble();
                bat.setFrequency(fi);

                // 2. Velocity & 3. Position (candidate)
                final var vel = bat.getVelocity();
                final var pos = bat.getPosition();
                final var candidate = new double[n];
                for (int d = 0; d < n; d++) {
                    vel[d] += (pos[d] - globalBestPosition[d]) * fi;
                    candidate[d] = Math.clamp(pos[d] + vel[d], min[d], max[d]);
                }

                // 4. Local random walk (exploitation)
                if (random.nextDouble() > bat.getPulseRate()) {
                    final double epsilon = random.nextDouble(-1.0, 1.0);
                    for (int d = 0; d < n; d++)
                        candidate[d] = Math.clamp(globalBestPosition[d] + epsilon * avgLoudness,
                                min[d], max[d]);
                }

                // 5. Evaluation & acceptance (only if better than global best)
                final double candFit = function.evaluate(candidate);
                if (Double.isFinite(candFit) && candFit < globalBestFitness
                        && random.nextDouble() < bat.getLoudness()) {
                    // Accept the new solution
                    bat.setFitness(candFit);
                    System.arraycopy(candidate, 0, pos, 0, n);

                    // 6. Update acoustics
                    bat.setLoudness(config.alpha() * bat.getLoudness());
                    bat.setPulseRate(bat.getMaxPulseRate() * (1.0 - Math.exp(-config.gamma() * t)));

                    // 7. Update global best immediately (matches paper: f(x') < f(x*))
                    globalBestFitness = candFit;
                    globalBestPosition = pos.clone();   // defensive copy
                }
            }

            history[t] = globalBestFitness;
            if (t % 100 == 0 || t == config.maxIterations())
                log.info("Iteration {}: best fitness = {}", t, globalBestFitness);
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

    // ──────────────────────── Diagnostic access (package‑private for tests) ─────

    /**
     * Returns current pulse rates of all bats (for educational analysis).
     */
    double[] getPulseRates() {
        return Arrays.stream(population).mapToDouble(Bat::getPulseRate).toArray();
    }

    /**
     * Returns current loudness of all bats (for educational analysis).
     */
    double[] getLoudnessValues() {
        return Arrays.stream(population).mapToDouble(Bat::getLoudness).toArray();
    }

    // ──────────────────────── supporting types ────────────────────────

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable configuration for the Bat Algorithm.
     */
    @Builder
    public record BaConfig(
            int populationSize,
            int maxIterations,
            int dimensions,
            double @NonNull [] minBounds,
            double @NonNull [] maxBounds,
            double fMin,
            double fMax,
            double alpha,
            double gamma,
            double initialLoudness,
            /* Asymptotic maximum pulse rate r⁰ᵢ ∈ [0,1]. Initial pulse rate is always 0. */
            double pulseRateMax,
            long seed
    ) {
        public BaConfig {
            Objects.requireNonNull(minBounds);
            Objects.requireNonNull(maxBounds);
            if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0");
            if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bounds length mismatch");
            if (fMin < 0 || fMax <= fMin) throw new IllegalArgumentException("fMin/fMax invalid");
            if (alpha <= 0 || alpha >= 1) throw new IllegalArgumentException("alpha ∈ (0,1)");
            if (gamma <= 0) throw new IllegalArgumentException("gamma > 0");
            if (initialLoudness <= 0) throw new IllegalArgumentException("A⁰ > 0");
            if (pulseRateMax < 0 || pulseRateMax > 1)
                throw new IllegalArgumentException("pulseRateMax ∈ [0,1]");
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }
    }

    /**
     * Outcome of a full optimization run.
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
     * Internal mutable bat agent. Fields are accessed directly inside the algorithm.
     */
    @Getter
    @Setter
    private static class Bat {
        private final double[] position;
        private final double[] velocity;
        private final double maxPulseRate;      // r⁰ᵢ, asymptotic maximum
        private double frequency;
        private double loudness;
        private double pulseRate;               // current rᵢ, initially 0
        private double fitness;

        Bat(double[] position, double fitness, double loudness, double maxPulseRate, double freq) {
            this.position = position.clone();
            this.velocity = new double[position.length];
            this.fitness = fitness;
            this.loudness = loudness;
            this.maxPulseRate = maxPulseRate;
            this.pulseRate = 0.0;   // always start at 0 (Yang 2010)
            this.frequency = freq;
        }
    }

    /**
     * Minimal demo for students.
     */
    @Slf4j
    public static class Demo {
        static void main() {
            log.info("=== Bat Algorithm Demo ===");
            ObjectiveFunction sphere = x -> DoubleStream.of(x).map(v -> v * v).sum();
            int d = 10;
            var min = new double[d];
            var max = new double[d];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            var cfg = BaConfig.builder()
                    .populationSize(30)
                    .maxIterations(1000)
                    .dimensions(d)
                    .minBounds(min)
                    .maxBounds(max)
                    .fMin(0).fMax(2)
                    .alpha(0.9).gamma(0.9)
                    .initialLoudness(1.0)
                    .pulseRateMax(0.5)
                    .seed(12345)
                    .build();

            var ba = new BatAlgorithm(cfg, sphere);
            var res = ba.optimize();
            log.info("Best fitness = {}", res.bestFitness());
            log.info("Sample of best position = {}", Arrays.toString(Arrays.copyOf(res.bestPosition(), 3)));
        }
    }
}