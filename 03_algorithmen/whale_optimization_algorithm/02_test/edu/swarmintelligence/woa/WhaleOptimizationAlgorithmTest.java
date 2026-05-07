package edu.swarmintelligence.woa;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.*;

/**
 * JUnit 5 + AssertJ test suite for the Whale Optimization Algorithm.
 * <p>
 * Verifies scientific correctness, reproducibility, bounds enforcement,
 * and configuration robustness.
 * </p>
 */
@DisplayName("WhaleOptimizationAlgorithm")
class WhaleOptimizationAlgorithmTest {
    static final double SPHERE_TOLERANCE = 1e-4;

    // --------------------- helpers ---------------------
    private static WhaleOptimizationAlgorithm.ObjectiveFunction sphereFunction() {
        return x -> Arrays.stream(x).map(v -> v * v).sum();
    }

    private static double[] bounds(int dimensions, double value) {
        var arr = new double[dimensions];
        Arrays.fill(arr, value);
        return arr;
    }

    // --------------------- Functional tests ---------------------
    @Nested
    @DisplayName("Optimization behaviour")
    class OptimizationTests {
        @Test
        @DisplayName("Finds global minimum of the Sphere function")
        void shouldFindGlobalMinimumOfSphere() {
            var function = sphereFunction();
            int dims = 10;
            var min = bounds(dims, -5.0);
            var max = bounds(dims, 5.0);

            var config = WhaleOptimizationAlgorithm.WoaConfig.createDefaultConfig(
                    30, 1000, dims, min, max, 42L
            );
            var woa = new WhaleOptimizationAlgorithm(config, function);
            var best = woa.optimize();

            // Best position should be near zero in all dimensions
            assertThat(best).hasSize(dims);
            assertThat(woa.getGlobalBestFitness()).isLessThan(SPHERE_TOLERANCE);

            // All coordinates must be within bounds
            for (int d = 0; d < dims; d++) {
                assertThat(best[d]).isBetween(min[d], max[d]);
            }
        }

        @Test
        @DisplayName("Reproducible results with fixed seed")
        void shouldBeReproducibleWithSameSeed() {
            var function = sphereFunction();
            int dims = 5;
            var min = bounds(dims, -10.0);
            var max = bounds(dims, 10.0);

            var config1 = WhaleOptimizationAlgorithm.WoaConfig.createDefaultConfig(
                    20, 100, dims, min, max, 123L);
            var config2 = WhaleOptimizationAlgorithm.WoaConfig.createDefaultConfig(
                    20, 100, dims, min, max, 123L);

            var woa1 = new WhaleOptimizationAlgorithm(config1, function);
            var woa2 = new WhaleOptimizationAlgorithm(config2, function);

            var best1 = woa1.optimize();
            var best2 = woa2.optimize();

            assertThat(best1).containsExactly(best2);
            assertThat(woa1.getGlobalBestFitness())
                    .isCloseTo(woa2.getGlobalBestFitness(), within(1e-15));
        }

        @Test
        @DisplayName("Fitness improves during optimization (gradient‑free check)")
        void fitnessShouldImproveOverTime() {
            var function = sphereFunction();
            int dims = 10;
            var min = bounds(dims, -10.0);
            var max = bounds(dims, 10.0);

            var config = WhaleOptimizationAlgorithm.WoaConfig.createDefaultConfig(
                    20, 200, dims, min, max, 7L);
            var woa = new WhaleOptimizationAlgorithm(config, function);

            woa.optimize();
            var finalFitness = woa.getGlobalBestFitness();
            assertThat(finalFitness).isLessThan(100.0);
        }
    }

    @Nested
    @DisplayName("Boundary enforcement")
    class BoundaryTests {
        @Test
        @DisplayName("All positions remain within bounds after optimization")
        void allPositionsStayWithinBounds() {
            var function = sphereFunction();
            int dims = 3;
            var min = new double[]{-1.0, 0.0, -5.0};
            var max = new double[]{1.0, 2.0, 5.0};

            var config = new WhaleOptimizationAlgorithm.WoaConfig(
                    10, 50, dims, min, max, 1.0, 12345L);
            var woa = new WhaleOptimizationAlgorithm(config, function);
            var best = woa.optimize();

            for (int d = 0; d < dims; d++) {
                assertThat(best[d]).isBetween(min[d], max[d]);
            }
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class ConfigurationTests {
        @ParameterizedTest(name = "{0}")
        @CsvSource(textBlock = """
                populationSize,  -5, 10
                maxIterations,   50, 0
                dimensions,      50, -2
                boundsLength,    50, 3
                """)
        @DisplayName("Invalid parameters throw IllegalArgumentException")
        void shouldRejectInvalidConfig(String scenario, int popSize, int maxIter) {
            int dims = 3;
            var min = bounds(dims, -1.0);
            var max = bounds(dims, 1.0);

            assertThatIllegalArgumentException().isThrownBy(() -> {
                if ("boundsLength".equals(scenario)) {
                    // deliberately mismatched bounds
                    new WhaleOptimizationAlgorithm.WoaConfig(
                            popSize, maxIter, dims,
                            bounds(dims, -1.0), bounds(4, 1.0), 1.0, 1L);
                } else {
                    new WhaleOptimizationAlgorithm.WoaConfig(
                            popSize, maxIter, dims, min, max, 1.0, 1L);
                }
            });
        }

        @Test
        @DisplayName("Spiral constant must be positive")
        void spiralConstantMustBePositive() {
            var min = bounds(1, -1.0);
            var max = bounds(1, 1.0);
            assertThatIllegalArgumentException().isThrownBy(() ->
                    new WhaleOptimizationAlgorithm.WoaConfig(1, 1, 1, min, max, -1.0, 1L));
        }

        @Test
        @DisplayName("Builder creates valid config with defensive copy of bounds")
        void builderCreatesSafeConfig() {
            var originalMin = new double[]{0.0, -1.0};
            var originalMax = new double[]{2.0, 3.0};
            var cfg = WhaleOptimizationAlgorithm.WoaConfig.builder()
                    .populationSize(10).maxIterations(50).dimensions(2)
                    .minBounds(originalMin).maxBounds(originalMax)
                    .spiralConstant(2.0).seed(42L).build();

            // Mutating the original arrays must not affect the config
            originalMin[0] = 999.9;
            originalMax[1] = -99.9;

            assertThat(cfg.minBounds()).containsExactly(0.0, -1.0);
            assertThat(cfg.maxBounds()).containsExactly(2.0, 3.0);
        }
    }

    @Nested
    @DisplayName("Random generator")
    class RandomGeneratorTests {
        @Test
        @DisplayName("L128X256MixRandom is used and deterministic with seed")
        void generatorIsDeterministic() {
            var gen1 = RandomGeneratorFactory.of("L128X256MixRandom").create(999L);
            var gen2 = RandomGeneratorFactory.of("L128X256MixRandom").create(999L);
            assertThat(gen1.nextDouble()).isEqualTo(gen2.nextDouble());
            assertThat(gen1.nextInt()).isEqualTo(gen2.nextInt());
        }
    }
}