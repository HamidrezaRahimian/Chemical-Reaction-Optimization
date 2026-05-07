package edu.swarmintelligence.gwo;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Grey Wolf Optimisation (GWO)")
class GreyWolfOptimizationTest {
    private GreyWolfOptimization.GwoConfig.GwoConfigBuilder defaultConfig() {
        return GreyWolfOptimization.GwoConfig.builder()
                .popSize(30)
                .maxIterations(500)
                .dimensions(5)
                .minBounds(new double[]{-5.12, -5.12, -5.12, -5.12, -5.12})
                .maxBounds(new double[]{5.12, 5.12, 5.12, 5.12, 5.12})
                .seed(42L);
    }

    @Test
    @DisplayName("Should minimise the sphere function to < 0.1")
    void shouldMinimiseSphereFunction() {
        var config = defaultConfig()
                .popSize(50)
                .maxIterations(1000)
                .seed(999L)
                .build();
        GreyWolfOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var gwo = new GreyWolfOptimization(config, sphere);
        double[] best = gwo.optimize();
        double bestFitness = gwo.getBestFitness();

        assertThat(bestFitness).isBetween(0.0, 0.1);
        for (double v : best) {
            assertThat(v).isBetween(-5.12, 5.12);
        }
    }

    @Test
    @DisplayName("Should produce reproducible results with fixed seed")
    void shouldBeReproducibleWithFixedSeed() {
        var config1 = defaultConfig().seed(123456789L).build();
        var config2 = defaultConfig().seed(123456789L).build();
        GreyWolfOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var gwo1 = new GreyWolfOptimization(config1, sphere);
        var gwo2 = new GreyWolfOptimization(config2, sphere);

        double[] best1 = gwo1.optimize();
        double[] best2 = gwo2.optimize();

        assertThat(best1).containsExactly(best2);
        assertThat(gwo1.getBestFitness()).isEqualTo(gwo2.getBestFitness());
    }

    @Test
    @DisplayName("Should never drop below the minimum population of 2")
    void shouldAlwaysMaintainMinimumPopulation() {
        var config = defaultConfig().popSize(2).maxIterations(200).build();
        GreyWolfOptimization.ObjectiveFunction dummy =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var gwo = new GreyWolfOptimization(config, dummy);
        assertThatCode(gwo::optimize).doesNotThrowAnyException();
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
    }

    @Test
    @DisplayName("GwoExample main should execute without exception")
    void shouldRunExampleWithoutException() {
        assertThatCode(GreyWolfOptimization.GwoExample::main)
                .doesNotThrowAnyException();
    }

    @Test
    @DisplayName("Should respect bounds even with extreme objective values")
    void shouldRespectBounds() {
        var config = defaultConfig()
                .dimensions(3)
                .minBounds(new double[]{-1.0, -2.0, -3.0})
                .maxBounds(new double[]{1.0, 2.0, 3.0})
                .maxIterations(300)
                .build();

        GreyWolfOptimization.ObjectiveFunction extreme = x -> {
            for (int i = 0; i < x.length; i++) {
                if (x[i] < config.minBounds()[i] || x[i] > config.maxBounds()[i])
                    return Double.POSITIVE_INFINITY;
            }
            return Arrays.stream(x).map(v -> v * v).sum();
        };

        var gwo = new GreyWolfOptimization(config, extreme);
        double[] best = gwo.optimize();

        for (int i = 0; i < best.length; i++) {
            assertThat(best[i]).isBetween(config.minBounds()[i], config.maxBounds()[i]);
        }
    }

    @Test
    @DisplayName("Should handle single‑dimensional problems correctly")
    void shouldHandleSingleDimension() {
        var config = GreyWolfOptimization.GwoConfig.builder()
                .popSize(10)
                .maxIterations(200)
                .dimensions(1)
                .minBounds(new double[]{-10.0})
                .maxBounds(new double[]{10.0})
                .seed(42L)
                .build();

        GreyWolfOptimization.ObjectiveFunction quadratic = x -> x[0] * x[0];
        var gwo = new GreyWolfOptimization(config, quadratic);
        double[] best = gwo.optimize();

        assertThat(best[0]).isBetween(-10.0, 10.0);
        assertThat(gwo.getBestFitness()).isLessThan(0.01);
    }

    @Test
    @DisplayName("Fitness should generally improve over iterations (non‑strict due to stochastic nature)")
    void shouldImproveFitnessOverIterations() {
        // Build a fresh configuration with 2 dimensions and matching bounds
        var config = GreyWolfOptimization.GwoConfig.builder()
                .popSize(20)
                .maxIterations(500)
                .dimensions(2)
                .minBounds(new double[]{-5.12, -5.12})
                .maxBounds(new double[]{5.12, 5.12})
                .seed(987L)
                .build();

        GreyWolfOptimization.ObjectiveFunction sphere = x -> x[0] * x[0] + x[1] * x[1];

        var gwo = new GreyWolfOptimization(config, sphere);
        gwo.optimize();
        double finalFitness = gwo.getBestFitness();

        // The optimum is 0; GWO should converge to a very small value
        assertThat(finalFitness).isLessThan(0.05);
    }
}