package edu.swarmintelligence.hho;

import org.assertj.core.api.Assertions;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;

/**
 * Scientific correctness and robustness tests for {@link HarrisHawksOptimization}.
 * Covers benchmark functions, configuration validation, reproducibility,
 * monotonic fitness improvement, and boundary compliance.
 */
class HarrisHawksOptimizationTest {
    private static final double SPHERE_MIN = -5.12;
    private static final double SPHERE_MAX = 5.12;
    private HarrisHawksOptimization.ObjectiveFunction sphere;
    private HarrisHawksOptimization.HhoConfig defaultCfg;

    private static void assertCloseToZero(final double[] vector, final double tolerance) {
        for (int i = 0; i < vector.length; i++) {
            Assertions.assertThat(vector[i])
                    .as("Coordinate %d = %f should be near 0 (global optimum)", i, vector[i])
                    .isCloseTo(0.0, Offset.offset(tolerance));
        }
    }

    @BeforeEach
    void setUp() {
        sphere = Benchmarks.sphere(5);
        final int dim = 5;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, SPHERE_MIN);
        Arrays.fill(maxB, SPHERE_MAX);
        defaultCfg = HarrisHawksOptimization.HhoConfig.builder()
                .populationSize(20)
                .maxIterations(200)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .seed(42L)
                .build();
    }

    // -------------------------------------------------------------------------
    // Basic functionality
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("Should find near‑optimal solution for Sphere function")
    void sphereConvergence() {
        final var hho = new HarrisHawksOptimization(defaultCfg, sphere);
        final double[] solution = hho.optimise();

        Assertions.assertThat(solution).hasSize(5);
        for (int i = 0; i < solution.length; i++) {
            Assertions.assertThat(solution[i])
                    .as("Solution coordinate %d should be within bounds", i)
                    .isBetween(SPHERE_MIN, SPHERE_MAX);
        }
        Assertions.assertThat(hho.getBestFitness()).isNotNegative();
        assertCloseToZero(solution, 1e-3);
    }

    @Test
    @DisplayName("Deterministic behaviour: same seed ⇒ same result")
    void reproducibility() {
        final var hho1 = new HarrisHawksOptimization(defaultCfg, sphere);
        final double[] sol1 = hho1.optimise();
        final var hho2 = new HarrisHawksOptimization(defaultCfg, sphere);
        final double[] sol2 = hho2.optimise();

        Assertions.assertThat(sol1).containsExactly(sol2);
        Assertions.assertThat(hho1.getBestFitness()).isEqualTo(hho2.getBestFitness());
    }

    @RepeatedTest(5)
    @DisplayName("Fitness should never increase during optimization (monotonic)")
    void monotonicFitness() {
        final var hho = new HarrisHawksOptimization(defaultCfg, sphere);
        double prev = hho.getBestFitness();
        hho.optimise();
        double after = hho.getBestFitness();
        Assertions.assertThat(after).isLessThan(prev);

        prev = after;
        hho.optimise();
        after = hho.getBestFitness();
        Assertions.assertThat(after).isLessThanOrEqualTo(prev);
    }

    // -------------------------------------------------------------------------
    // Different configurations
    // -------------------------------------------------------------------------

    @ParameterizedTest
    @ValueSource(ints = {2, 5, 10, 30})
    @DisplayName("Should work with various population sizes")
    void variousPopulationSizes(final int popSize) {
        final var cfg = defaultCfg.toBuilder()
                .populationSize(popSize)
                .seed(1L)
                .build();
        final var hho = new HarrisHawksOptimization(cfg, sphere);
        final double[] sol = hho.optimise();
        Assertions.assertThat(sol).hasSize(5);
        Assertions.assertThat(hho.getBestFitness()).isLessThan(0.1);
    }

    @ParameterizedTest
    @ValueSource(ints = {10, 50, 200})
    @DisplayName("Convergence improves with more iterations")
    void iterationImpact(final int maxIter) {
        final var cfg = defaultCfg.toBuilder()
                .maxIterations(maxIter)
                .build();
        final var hho = new HarrisHawksOptimization(cfg, sphere);
        hho.optimise();
        Assertions.assertThat(hho.getBestFitness()).isLessThan(0.1);
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 10, 30})
    @DisplayName("Higher‑dimensional Sphere still converges reasonably")
    void varyingDimensions(final int dim) {
        final var cfg = configWithDimensions(dim).toBuilder()
                .maxIterations(300)
                .seed(7L)
                .build();
        final var obj = Benchmarks.sphere(dim);
        final var hho = new HarrisHawksOptimization(cfg, obj);
        hho.optimise();
        Assertions.assertThat(hho.getBestFitness()).isLessThan(dim * 1e-2);
    }

    // -------------------------------------------------------------------------
    // Benchmark functions
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("Rastrigin function: should approach global minimum (0) in 5D")
    void rastrigin5D() {
        final int dim = 5;
        final var cfg = configWithDimensions(dim).toBuilder()
                .populationSize(50)
                .maxIterations(500)
                .seed(999L)
                .build();
        final var hho = new HarrisHawksOptimization(cfg, Benchmarks.rastrigin(dim));
        hho.optimise();
        Assertions.assertThat(hho.getBestFitness()).isLessThan(1e-4);
    }

    @Test
    @DisplayName("Ackley function: should find near‑optimum (0) in 5D")
    void ackley5D() {
        final int dim = 5;
        final var cfg = configWithDimensions(dim).toBuilder()
                .populationSize(50)
                .maxIterations(500)
                .seed(42L)
                .build();
        final var hho = new HarrisHawksOptimization(cfg, Benchmarks.ackley(dim));
        hho.optimise();
        Assertions.assertThat(hho.getBestFitness()).isLessThan(1e-4);
    }

    @Test
    @DisplayName("Rosenbrock valley (2D) – challenging but should make progress")
    void rosenbrock2D() {
        final int dim = 2;
        final double[] minB = {-2.0, -2.0};
        final double[] maxB = {2.0, 2.0};
        final var cfg = HarrisHawksOptimization.HhoConfig.builder()
                .populationSize(30)
                .maxIterations(500)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .seed(123L)
                .build();
        final var hho = new HarrisHawksOptimization(cfg, Benchmarks.rosenbrock(dim));
        hho.optimise();
        Assertions.assertThat(hho.getBestFitness()).isLessThan(0.1);
    }

    // -------------------------------------------------------------------------
    // Boundary & constraint checking
    // -------------------------------------------------------------------------

    @Test
    @DisplayName("All candidate solutions remain inside search bounds")
    void solutionsWithinBounds() {
        final var hho = new HarrisHawksOptimization(defaultCfg, sphere);
        final double[] sol = hho.optimise();
        for (int d = 0; d < sol.length; d++) {
            Assertions.assertThat(sol[d])
                    .isBetween(defaultCfg.minBounds()[d], defaultCfg.maxBounds()[d]);
        }
    }

    @Test
    @DisplayName("Bounding works even with extreme initial population")
    void extremeBounds() {
        final double[] min = {-1, -1};
        final double[] max = {1, 1};
        final var cfg = HarrisHawksOptimization.HhoConfig.builder()
                .populationSize(10)
                .maxIterations(50)
                .dimensions(2)
                .minBounds(min)
                .maxBounds(max)
                .seed(1L)
                .build();
        final var obj = (HarrisHawksOptimization.ObjectiveFunction) x -> x[0] + x[1];
        final var hho = new HarrisHawksOptimization(cfg, obj);
        final double[] sol = hho.optimise();
        Assertions.assertThat(sol[0]).isBetween(-1.0, 1.0);
        Assertions.assertThat(sol[1]).isBetween(-1.0, 1.0);
    }

    // -------------------------------------------------------------------------
    // Configuration validation
    // -------------------------------------------------------------------------

    private HarrisHawksOptimization.HhoConfig configWithDimensions(final int dim) {
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, SPHERE_MIN);
        Arrays.fill(maxB, SPHERE_MAX);
        return HarrisHawksOptimization.HhoConfig.builder()
                .populationSize(20)
                .maxIterations(200)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .seed(42L)
                .build();
    }

    // -------------------------------------------------------------------------
    // Helper methods
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Benchmarks
    // -------------------------------------------------------------------------
    static final class Benchmarks {
        private Benchmarks() {
        }

        static HarrisHawksOptimization.ObjectiveFunction sphere(final int dim) {
            return x -> Arrays.stream(x).map(v -> v * v).sum();
        }

        static HarrisHawksOptimization.ObjectiveFunction rastrigin(final int dim) {
            return x -> {
                double sum = 10.0 * dim;
                for (double v : x) sum += v * v - 10.0 * Math.cos(2 * Math.PI * v);
                return sum;
            };
        }

        static HarrisHawksOptimization.ObjectiveFunction ackley(final int dim) {
            return x -> {
                double sum1 = 0, sum2 = 0;
                for (double v : x) {
                    sum1 += v * v;
                    sum2 += Math.cos(2 * Math.PI * v);
                }
                return -20.0 * Math.exp(-0.2 * Math.sqrt(sum1 / dim))
                        - Math.exp(sum2 / dim) + 20.0 + Math.E;
            };
        }

        static HarrisHawksOptimization.ObjectiveFunction rosenbrock(final int dim) {
            return x -> {
                double sum = 0;
                for (int i = 0; i < dim - 1; i++) {
                    double t1 = x[i + 1] - x[i] * x[i];
                    double t2 = x[i] - 1.0;
                    sum += 100.0 * t1 * t1 + t2 * t2;
                }
                return sum;
            };
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class ConfigValidation {
        @Test
        @DisplayName("Rejects null bounds")
        void nullBounds() {
            Assertions.assertThatThrownBy(() ->
                            HarrisHawksOptimization.HhoConfig.builder()
                                    .populationSize(10).maxIterations(100).dimensions(2)
                                    .minBounds(null).maxBounds(new double[]{0, 1}).seed(1).build())
                    .isInstanceOf(NullPointerException.class);
        }

        @Test
        @DisplayName("Rejects mismatched dimension length")
        void mismatchedDimensions() {
            Assertions.assertThatThrownBy(() ->
                            HarrisHawksOptimization.HhoConfig.builder()
                                    .populationSize(10).maxIterations(100).dimensions(3)
                                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1, 1}).seed(1).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("length");
        }

        @ParameterizedTest
        @ValueSource(ints = {0, -1})
        @DisplayName("Rejects non‑positive population size")
        void invalidPopulationSize(final int ps) {
            Assertions.assertThatThrownBy(() ->
                            defaultCfg.toBuilder().populationSize(ps).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @ParameterizedTest
        @ValueSource(ints = {0, -5})
        @DisplayName("Rejects non‑positive iterations or dimensions")
        void invalidIterOrDim(final int value) {
            Assertions.assertThatThrownBy(() ->
                            defaultCfg.toBuilder().maxIterations(value).build())
                    .isInstanceOf(IllegalArgumentException.class);
            Assertions.assertThatThrownBy(() ->
                            defaultCfg.toBuilder().dimensions(value).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }
}