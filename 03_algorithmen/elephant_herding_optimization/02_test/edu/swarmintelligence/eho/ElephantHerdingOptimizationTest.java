package edu.swarmintelligence.eho;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Elephant Herding Optimization (EHO)")
class ElephantHerdingOptimizationTest {
    private ElephantHerdingOptimization.EhoConfig.EhoConfigBuilder defaultConfig() {
        return ElephantHerdingOptimization.EhoConfig.builder()
                .nClan(5)
                .elephantsPerClan(10)
                .maxIterations(2000)
                .dimensions(5)
                .minBounds(new double[]{-5.12, -5.12, -5.12, -5.12, -5.12})
                .maxBounds(new double[]{5.12, 5.12, 5.12, 5.12, 5.12})
                .alpha(0.5)
                .beta(0.1)
                .seed(42L);
    }

    @Test
    @DisplayName("Should minimise the sphere function to a reasonably low value")
    void shouldMinimiseSphereFunction() {
        var config = defaultConfig()
                .nClan(8)
                .elephantsPerClan(15)
                .maxIterations(5000)
                .seed(123L)
                .build();
        ElephantHerdingOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var eho = new ElephantHerdingOptimization(config, sphere);
        double[] best = eho.optimize();
        double bestFit = eho.getBestFitness();

        assertThat(bestFit).isGreaterThanOrEqualTo(0.0);
        assertThat(bestFit).isLessThan(2.0);      // 5‑D sphere with strong convergence
        for (double v : best) {
            assertThat(v).isBetween(-5.12, 5.12);
        }
    }

    @Test
    @DisplayName("Should produce reproducible results with fixed seed")
    void shouldBeReproducibleWithFixedSeed() {
        var config1 = defaultConfig().seed(98765L).build();
        var config2 = defaultConfig().seed(98765L).build();
        ElephantHerdingOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var eho1 = new ElephantHerdingOptimization(config1, sphere);
        var eho2 = new ElephantHerdingOptimization(config2, sphere);

        double[] best1 = eho1.optimize();
        double[] best2 = eho2.optimize();

        assertThat(best1).containsExactly(best2);
        assertThat(eho1.getBestFitness()).isEqualTo(eho2.getBestFitness());
    }

    @Test
    @DisplayName("Should reject invalid configuration parameters")
    void shouldValidateConfig() {
        assertThatThrownBy(() -> defaultConfig().nClan(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("nClan");

        assertThatThrownBy(() -> defaultConfig().elephantsPerClan(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("elephantsPerClan");

        assertThatThrownBy(() -> defaultConfig().maxIterations(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("maxIterations");

        assertThatThrownBy(() -> defaultConfig().dimensions(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dimensions");

        assertThatThrownBy(() -> defaultConfig()
                .minBounds(new double[]{1.0})
                .maxBounds(new double[]{1.0, 2.0})
                .build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Bounds");

        assertThatThrownBy(() -> defaultConfig().alpha(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("alpha");

        assertThatThrownBy(() -> defaultConfig().beta(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("beta");
    }

    @Test
    @DisplayName("EHOExample main should execute without exception")
    void shouldRunExampleWithoutException() {
        assertThatCode(ElephantHerdingOptimization.EHOExample::main)
                .doesNotThrowAnyException();
    }
}