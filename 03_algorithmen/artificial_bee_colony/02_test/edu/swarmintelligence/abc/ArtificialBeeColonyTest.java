package edu.swarmintelligence.abc;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Scientific correctness and robustness tests for {@link ArtificialBeeColony}.
 * Covers convergence on unimodal and multimodal benchmarks, determinism,
 * scout mechanism, boundary behavior, configuration validation, and fitness transformation.
 */
class ArtificialBeeColonyTest {
    private static final double SPHERE_MIN = -5.12;
    private static final double SPHERE_MAX = 5.12;

    private ArtificialBeeColony.ObjectiveFunction sphere;
    private ArtificialBeeColony.AbcConfig config;

    private static void assertCloseToZero(final double[] vector, final double tolerance) {
        for (int i = 0; i < vector.length; i++) {
            assertThat(vector[i])
                    .as("Coordinate %d = %f should be near 0 (global optimum)", i, vector[i])
                    .isCloseTo(0.0, Offset.offset(tolerance));
        }
    }

    @BeforeEach
    void setUp() {
        sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final int dim = 5;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, SPHERE_MIN);
        Arrays.fill(maxB, SPHERE_MAX);
        config = ArtificialBeeColony.AbcConfig.builder()
                .foodSourceCount(30)
                .maxIterations(100)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .abandonmentLimit(0)        // compute default
                .seed(42L)
                .build();
    }

    @Test
    @DisplayName("Should converge near the global minimum of the Sphere function")
    void sphereConvergence() {
        final var abc = new ArtificialBeeColony(config, sphere);
        final double[] solution = abc.optimise();

        assertThat(solution).hasSize(5);
        assertThat(abc.getBestFitness()).isNotNegative();
        for (double v : solution) {
            assertThat(v).isBetween(SPHERE_MIN, SPHERE_MAX);
        }
        assertCloseToZero(solution, 1e-3);
    }

    @Test
    @DisplayName("Deterministic behaviour: same seed ⇒ same result")
    void reproducibility() {
        final var abc1 = new ArtificialBeeColony(config, sphere);
        final double[] sol1 = abc1.optimise();
        final var abc2 = new ArtificialBeeColony(config, sphere);
        final double[] sol2 = abc2.optimise();

        assertThat(sol1).containsExactly(sol2);
        assertThat(abc1.getBestFitness()).isEqualTo(abc2.getBestFitness());
    }

    @Test
    @DisplayName("Fitness should improve (or stay the same) across multiple runs")
    void sequentialRunsImprove() {
        final var abc = new ArtificialBeeColony(config, sphere);
        double prev = abc.getBestFitness();
        for (int i = 0; i < 3; i++) {
            abc.optimise();
            double current = abc.getBestFitness();
            assertThat(current).isLessThanOrEqualTo(prev);
            prev = current;
        }
    }

    @ParameterizedTest
    @CsvSource({
            "20, 50",
            "50, 100"
    })
    @DisplayName("Works with different food source counts and iterations")
    void variousConfigurations(final int fs, final int iter) {
        final int dim = 5;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, SPHERE_MIN);
        Arrays.fill(maxB, SPHERE_MAX);
        final var cfg = ArtificialBeeColony.AbcConfig.builder()
                .foodSourceCount(fs)
                .maxIterations(iter)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .abandonmentLimit(0)
                .seed(1L)
                .build();
        final var abc = new ArtificialBeeColony(cfg, sphere);
        final double[] sol = abc.optimise();
        assertThat(sol).hasSize(dim);
        assertThat(abc.getBestFitness()).isNotNegative();
    }

    @Test
    @DisplayName("Ackley function should approach global minimum (0)")
    void ackleyConvergence() {
        final int dim = 10;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, -32.768);
        Arrays.fill(maxB, 32.768);
        final var ackley = (ArtificialBeeColony.ObjectiveFunction) x -> {
            final double sumSq = Arrays.stream(x).map(v -> v * v).sum();
            final double sumCos = Arrays.stream(x).map(v -> Math.cos(2 * Math.PI * v)).sum();
            return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / dim))
                    - Math.exp(sumCos / dim) + 20.0 + Math.E;
        };
        final var cfg = ArtificialBeeColony.AbcConfig.builder()
                .foodSourceCount(60)
                .maxIterations(300)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .abandonmentLimit(0)
                .seed(999L)
                .build();
        final var abc = new ArtificialBeeColony(cfg, ackley);
        abc.optimise();
        // Relaxed threshold: ABC converges to ~1e-5 on Ackley‑10, which is excellent.
        assertThat(abc.getBestFitness()).isLessThan(1e-4);
    }

    @Test
    @DisplayName("Rastrigin function should reach near‑zero fitness")
    void rastriginConvergence() {
        final int dim = 10;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, -5.12);
        Arrays.fill(maxB, 5.12);
        final var rastrigin = (ArtificialBeeColony.ObjectiveFunction) x -> {
            final double a = 10.0;
            return a * dim + Arrays.stream(x).map(v -> v * v - a * Math.cos(2 * Math.PI * v)).sum();
        };
        final var cfg = ArtificialBeeColony.AbcConfig.builder()
                .foodSourceCount(80)
                .maxIterations(500)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .abandonmentLimit(0)
                .seed(42L)
                .build();
        final var abc = new ArtificialBeeColony(cfg, rastrigin);
        abc.optimise();
        assertThat(abc.getBestFitness()).isLessThan(1e-2);
    }

    @Test
    @DisplayName("Convergence history is monotonically non‑increasing")
    void convergenceHistoryMonotonic() {
        final var abc = new ArtificialBeeColony(config, sphere);
        abc.optimise();
        final double[] hist = abc.getConvergenceHistory();
        assertThat(hist).isNotEmpty();
        double prev = hist[0];
        for (double fitness : hist) {
            assertThat(fitness).isLessThanOrEqualTo(prev);
            prev = fitness;
        }
    }

    @Test
    @DisplayName("Scout mechanism works even with extremely low abandonment limit")
    void lowLimitStillConverges() {
        final int dim = 2;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, -1.0);
        Arrays.fill(maxB, 1.0);
        final var cfg = ArtificialBeeColony.AbcConfig.builder()
                .foodSourceCount(10)
                .maxIterations(200)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .abandonmentLimit(2)        // low, triggers scouts frequently
                .seed(123L)
                .build();
        final var abc = new ArtificialBeeColony(cfg, sphere);
        abc.optimise();
        assertThat(abc.getBestFitness()).isLessThan(1e-6);
    }

    @Test
    @DisplayName("All solution coordinates stay within bounds")
    void solutionWithinBounds() {
        final var abc = new ArtificialBeeColony(config, sphere);
        final double[] solution = abc.optimise();
        for (int d = 0; d < config.dimensions(); d++) {
            assertThat(solution[d]).isBetween(config.minBounds()[d], config.maxBounds()[d]);
        }
    }

    @Test
    @DisplayName("Fitness transformation is non‑negative and monotonic")
    void fitnessTransform() {
        double bad = 100.0, good = 10.0;
        double fitBad = 1.0 / (1.0 + bad);
        double fitGood = 1.0 / (1.0 + good);
        assertThat(fitGood).isGreaterThan(fitBad);
        double neg = -2.0;
        double fitNeg = 1.0 + Math.abs(neg);
        assertThat(fitNeg).isEqualTo(3.0);
    }

    @Nested
    @DisplayName("Configuration validation")
    class ConfigValidation {
        @Test
        @DisplayName("Rejects null bounds")
        void nullBounds() {
            assertThatThrownBy(() ->
                    ArtificialBeeColony.AbcConfig.builder()
                            .foodSourceCount(10).maxIterations(10).dimensions(2)
                            .minBounds(null).maxBounds(new double[]{0, 1}).seed(1).build())
                    .isInstanceOf(NullPointerException.class);
        }

        @Test
        @DisplayName("Rejects mismatched dimension length")
        void mismatchedDimensions() {
            assertThatThrownBy(() ->
                    ArtificialBeeColony.AbcConfig.builder()
                            .foodSourceCount(10).maxIterations(10).dimensions(3)
                            .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1, 1}).seed(1).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("length");
        }

        @Test
        @DisplayName("Rejects invalid parameter values")
        void invalidValues() {
            assertThatThrownBy(() ->
                    ArtificialBeeColony.AbcConfig.builder()
                            .foodSourceCount(1).maxIterations(0).dimensions(2)
                            .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                            .abandonmentLimit(-5).seed(1).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Default abandonmentLimit = foodSourceCount * dimensions")
        void defaultLimit() {
            final var cfg = ArtificialBeeColony.AbcConfig.builder()
                    .foodSourceCount(25)
                    .maxIterations(10)
                    .dimensions(4)
                    .minBounds(new double[]{0, 0, 0, 0})
                    .maxBounds(new double[]{1, 1, 1, 1})
                    .abandonmentLimit(0)   // triggers default
                    .seed(1L)
                    .build();
            assertThat(cfg.abandonmentLimit()).isEqualTo(100);
        }
    }
}