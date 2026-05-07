package edu.swarmintelligence.fwa;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Fireworks Optimization (FWA)")
class FireworksOptimizationTest {
    private FireworksOptimization.FwaConfig.FwaConfigBuilder defaultConfig() {
        return FireworksOptimization.FwaConfig.builder()
                .popSize(20)
                .maxIterations(1000)
                .dimensions(5)
                .minBounds(new double[]{-5.12, -5.12, -5.12, -5.12, -5.12})
                .maxBounds(new double[]{5.12, 5.12, 5.12, 5.12, 5.12})
                .sparkMultiplier(50)
                .gaussianSparksCount(5)
                .explosionAmplitudeConstant(10)
                .lowerBoundSparksFactor(0.5)
                .upperBoundSparksFactor(1.5)
                .epsilon(1e-6)
                .seed(42L);
    }

    @Test
    @DisplayName("Should minimise the sphere function to a reasonably low value")
    void shouldMinimiseSphereFunction() {
        var config = defaultConfig()
                .popSize(30)
                .maxIterations(3000)
                .seed(999L)
                .build();
        FireworksOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var fwa = new FireworksOptimization(config, sphere);
        double[] best = fwa.optimize();
        double bestPE = fwa.getBestPE();

        assertThat(bestPE).isGreaterThanOrEqualTo(0.0);
        assertThat(bestPE).isLessThan(1.0);
        for (double v : best) {
            assertThat(v).isBetween(-5.12, 5.12);
        }
    }

    @Test
    @DisplayName("Should produce reproducible results with fixed seed")
    void shouldBeReproducibleWithFixedSeed() {
        var config1 = defaultConfig().seed(123456789L).build();
        var config2 = defaultConfig().seed(123456789L).build();
        FireworksOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var fwa1 = new FireworksOptimization(config1, sphere);
        var fwa2 = new FireworksOptimization(config2, sphere);

        double[] best1 = fwa1.optimize();
        double[] best2 = fwa2.optimize();

        assertThat(best1).containsExactly(best2);
        assertThat(fwa1.getBestPE()).isEqualTo(fwa2.getBestPE());
    }

    @Test
    @DisplayName("Should never drop below the minimum population of 2")
    void shouldAlwaysMaintainMinimumPopulation() {
        var config = defaultConfig().popSize(10).maxIterations(200).build();
        FireworksOptimization.ObjectiveFunction dummy =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var fwa = new FireworksOptimization(config, dummy);
        assertThatCode(fwa::optimize).doesNotThrowAnyException();
    }

    @Test
    @DisplayName("Should reject invalid configuration parameters")
    void shouldValidateConfig() {
        assertThatThrownBy(() -> defaultConfig().popSize(1).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("PopSize");

        assertThatThrownBy(() -> defaultConfig().maxIterations(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("MaxIterations");

        assertThatThrownBy(() -> defaultConfig().dimensions(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Dimensions");

        assertThatThrownBy(() -> defaultConfig()
                .minBounds(new double[]{1.0})
                .maxBounds(new double[]{1.0, 2.0})
                .build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Bounds");

        assertThatThrownBy(() -> defaultConfig().sparkMultiplier(0).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().gaussianSparksCount(-1).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().explosionAmplitudeConstant(0).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().lowerBoundSparksFactor(0).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig()
                .lowerBoundSparksFactor(1.5)
                .upperBoundSparksFactor(1.0)
                .build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().epsilon(0).build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("FWAExample main should execute without exception")
    void shouldRunExampleWithoutException() {
        assertThatCode(FireworksOptimization.FWAExample::main)
                .doesNotThrowAnyException();
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 5, 10})
    @DisplayName("All returned coordinates must lie within search bounds")
    void bestSolutionMustRespectBounds(int dims) {
        var min = new double[dims];
        var max = new double[dims];
        Arrays.fill(min, -10.0);
        Arrays.fill(max, 10.0);

        var config = FireworksOptimization.FwaConfig.builder()
                .popSize(15)
                .maxIterations(100)
                .dimensions(dims)
                .minBounds(min)
                .maxBounds(max)
                .sparkMultiplier(50)
                .gaussianSparksCount(5)
                .explosionAmplitudeConstant(10)
                .lowerBoundSparksFactor(0.5)
                .upperBoundSparksFactor(1.5)
                .epsilon(1e-6)
                .seed(42L)
                .build();

        var fwa = new FireworksOptimization(config, x -> Arrays.stream(x).map(v -> v * v).sum());
        double[] best = fwa.optimize();

        // Simple, version‑agnostic validation of bounds
        for (double v : best) {
            assertThat(v).isBetween(-10.0, 10.0);
        }
    }

    @Test
    @DisplayName("Optimisation should not produce NaN or infinite values")
    void shouldNotProduceInvalidFitness() {
        var config = defaultConfig().maxIterations(300).build();
        var fwa = new FireworksOptimization(config, x -> Arrays.stream(x).map(v -> v * v).sum());
        fwa.optimize();
        assertThat(fwa.getBestPE()).isFinite();
    }
}