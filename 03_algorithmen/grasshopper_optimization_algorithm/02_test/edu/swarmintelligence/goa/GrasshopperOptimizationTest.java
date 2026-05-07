package edu.swarmintelligence.goa;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Grasshopper Optimisation Algorithm (GOA)")
class GrasshopperOptimizationTest {
    // Helper to create a default builder with valid 5‑D bounds
    private GrasshopperOptimization.GoaConfig.Builder defaultBuilder() {
        return new GrasshopperOptimization.GoaConfig.Builder()
                .popSize(30)
                .maxIterations(500)
                .dimensions(5)
                .minBounds(new double[]{-5.12, -5.12, -5.12, -5.12, -5.12})
                .maxBounds(new double[]{5.12, 5.12, 5.12, 5.12, 5.12})
                .attractionIntensity(0.5)
                .attractiveLengthScale(1.5)
                .cMax(1.0)
                .cMin(1e-5)
                .seed(42L);
    }

    @Test
    @DisplayName("Should minimise the sphere function to a low value")
    void shouldMinimiseSphereFunction() {
        var config = defaultBuilder()
                .popSize(50)
                .maxIterations(1000)
                .seed(999L)
                .build();
        var sphere = (GrasshopperOptimization.ObjectiveFunction)
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var goa = new GrasshopperOptimization(config, sphere);
        double[] best = goa.optimize();
        double bestFitness = goa.getBestFitness();

        // The optimum is 0; a value < 1.0 indicates strong convergence.
        assertThat(bestFitness).isGreaterThanOrEqualTo(0.0);
        assertThat(bestFitness).isLessThan(1.0);
        for (double v : best) {
            assertThat(v).isBetween(-5.12, 5.12);
        }
    }

    @Test
    @DisplayName("Should produce identical results with fixed seed (reproducibility)")
    void shouldBeReproducibleWithFixedSeed() {
        long seed = 123456789L;
        var config1 = defaultBuilder().seed(seed).build();
        var config2 = defaultBuilder().seed(seed).build();
        var sphere = (GrasshopperOptimization.ObjectiveFunction)
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var goa1 = new GrasshopperOptimization(config1, sphere);
        var goa2 = new GrasshopperOptimization(config2, sphere);

        double[] best1 = goa1.optimize();
        double[] best2 = goa2.optimize();

        assertThat(best1).containsExactly(best2);
        assertThat(goa1.getBestFitness()).isEqualTo(goa2.getBestFitness());
    }

    @Test
    @DisplayName("Should handle a minimal feasible population size (2)")
    void shouldMaintainMinimumPopulation() {
        var config = defaultBuilder().popSize(2).maxIterations(100).build();
        var sphere = (GrasshopperOptimization.ObjectiveFunction)
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var goa = new GrasshopperOptimization(config, sphere);
        assertThatCode(goa::optimize).doesNotThrowAnyException();
        assertThat(goa.getBestFitness()).isGreaterThanOrEqualTo(0.0);
    }

    @Test
    @DisplayName("Should enforce configuration validation")
    void shouldValidateConfig() {
        var b = defaultBuilder();
        assertThatThrownBy(() -> b.popSize(1).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("PopSize");

        assertThatThrownBy(() -> defaultBuilder().maxIterations(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("MaxIterations");

        assertThatThrownBy(() -> defaultBuilder().dimensions(0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Dimensions");

        assertThatThrownBy(() -> defaultBuilder()
                .minBounds(new double[]{1.0})
                .maxBounds(new double[]{1.0, 2.0})
                .build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Bounds");

        assertThatThrownBy(() -> defaultBuilder().attractionIntensity(0).build())
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> defaultBuilder().attractiveLengthScale(0).build())
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> defaultBuilder().cMax(0.5).cMin(1.0).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cMax");
        assertThatThrownBy(() -> defaultBuilder().cMin(-0.1).build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Should throw when mandatory bounds are not set")
    void shouldRejectMissingBounds() {
        // Builder without setting minBounds/maxBounds -> they remain null
        assertThatThrownBy(() -> new GrasshopperOptimization.GoaConfig.Builder()
                .dimensions(2)
                .build())
                .isInstanceOf(NullPointerException.class);   // record constructor checks null
    }

    @Test
    @DisplayName("Social force function should match formula s(r) = f*exp(-r/l) - exp(-r)")
    void shouldCalculateSocialForce() {
        double r = 2.0;
        double f = 0.5;
        double l = 1.5;
        double expected = f * Math.exp(-r / l) - Math.exp(-r);
        assertThat(GrasshopperOptimization.socialForce(r, f, l))
                .isCloseTo(expected, within(1e-12));
    }

    @ParameterizedTest(name = "f={0}, l={1}, r={2}  →  s(r) ≈ {3}")
    @CsvSource(delimiter = '|', textBlock = """
            0.5 | 1.5 | 0.0 | -0.5
            0.5 | 1.5 | 1.0 | -0.11117088
            1.0 | 1.5 | 2.0 |  0.12826185
            """)
    void socialForceParameterised(double f, double l, double r, double expected) {
        assertThat(GrasshopperOptimization.socialForce(r, f, l))
                .isCloseTo(expected, within(1e-5));
    }

    @Test
    @DisplayName("Optimised positions must stay within bounds")
    void positionsMustStayWithinBounds() {
        var config = defaultBuilder()
                .popSize(30)
                .maxIterations(200)
                .seed(123L)
                .build();
        var sphere = (GrasshopperOptimization.ObjectiveFunction)
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var goa = new GrasshopperOptimization(config, sphere);
        double[] best = goa.optimize();

        for (double v : best) {
            assertThat(v).isBetween(-5.12, 5.12);
        }
    }

    @Test
    @DisplayName("Should run the GoaExample without any exception")
    void shouldRunExampleWithoutException() {
        assertThatCode(GrasshopperOptimization.GoaExample::main)
                .doesNotThrowAnyException();
    }

    @Test
    @DisplayName("Best fitness should never increase over iterations (monotonic improvement)")
    void bestFitnessShouldBeNonIncreasing() {
        // We use a tiny run and check that the final best is not worse than initial.
        var config = defaultBuilder()
                .popSize(10)
                .maxIterations(50)
                .seed(7777L)
                .build();
        var sphere = (GrasshopperOptimization.ObjectiveFunction)
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var goa = new GrasshopperOptimization(config, sphere);
        // The initial targetFitness is set during constructor from the initial swarm.
        // After optimization the getBestFitness() must be <= that initial value,
        // because it is only replaced when a strictly better fitness is found.
        double initialBest = goa.getBestFitness();   // accessed after construction
        goa.optimize();
        assertThat(goa.getBestFitness()).isLessThanOrEqualTo(initialBest);
    }
}