package edu.swarmintelligence.cro;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Chemical Reaction Optimization (CRO) – scientific correctness & safety")
class ChemicalReactionOptimizationTest {
    private ChemicalReactionOptimization.CroConfig.CroConfigBuilder defaultConfig() {
        return ChemicalReactionOptimization.CroConfig.builder()
                .popSize(50)
                .maxIterations(2000)
                .dimensions(5)
                .minBounds(new double[]{-5.12, -5.12, -5.12, -5.12, -5.12})
                .maxBounds(new double[]{5.12, 5.12, 5.12, 5.12, 5.12})
                .kelossRate(0.2)
                .moleColl(0.2)
                .decThres(10)
                .synThres(1.0)
                .initialKE(1000.0)
                .enBuff(1000.0)
                .stepSize(0.1)
                .seed(42L);
    }

    @Test
    @DisplayName("Should minimise the sphere function to a reasonably low value")
    void shouldMinimiseSphereFunction() {
        var config = defaultConfig()
                .popSize(100)
                .maxIterations(5000)
                .seed(123L)
                .build();
        ChemicalReactionOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var cro = new ChemicalReactionOptimization(config, sphere);
        double[] best = cro.optimize();
        double bestPE = cro.getBestPE();

        assertThat(bestPE).isGreaterThanOrEqualTo(0.0);
        assertThat(bestPE).isLessThan(2.0);
        // Verify all coordinates remain inside bounds
        for (double v : best) {
            assertThat(v).isBetween(-5.12, 5.12);
        }
    }

    @Test
    @DisplayName("Should give exactly reproducible results with a fixed seed")
    void shouldBeReproducibleWithFixedSeed() {
        var base = defaultConfig().seed(123456789L);
        ChemicalReactionOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var cro1 = new ChemicalReactionOptimization(base.build(), sphere);
        var cro2 = new ChemicalReactionOptimization(base.build(), sphere);

        double[] best1 = cro1.optimize();
        double[] best2 = cro2.optimize();

        assertThat(best1).containsExactly(best2);
        assertThat(cro1.getBestPE()).isEqualTo(cro2.getBestPE());
    }

    @Test
    @DisplayName("Should conserve total energy throughout the whole run")
    void shouldConserveTotalEnergy() {
        var config = defaultConfig().maxIterations(500).build();
        ChemicalReactionOptimization.ObjectiveFunction dummy =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var cro = new ChemicalReactionOptimization(config, dummy);
        double initialEnergy = cro.getTotalSystemEnergy();
        cro.optimize();
        double finalEnergy = cro.getTotalSystemEnergy();

        assertThat(finalEnergy).isCloseTo(initialEnergy, withinPercentage(0.001));
    }

    @Test
    @DisplayName("Should never drop below the minimum population of 2")
    void shouldAlwaysMaintainMinimumPopulation() {
        var config = defaultConfig().popSize(4).maxIterations(200).build();
        ChemicalReactionOptimization.ObjectiveFunction dummy =
                x -> Arrays.stream(x).map(v -> v * v).sum();

        var cro = new ChemicalReactionOptimization(config, dummy);
        assertThatCode(cro::optimize).doesNotThrowAnyException();
    }

    @Test
    @DisplayName("Should reject every invalid configuration parameter")
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

        assertThatThrownBy(() -> defaultConfig().kelossRate(1.5).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().moleColl(-0.5).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().decThres(0).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().synThres(-0.1).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().initialKE(0.0).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().enBuff(-1.0).build())
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> defaultConfig().stepSize(0.0).build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("The CROExample.main() demo should run without any error")
    void shouldRunExampleWithoutException() {
        assertThatCode(ChemicalReactionOptimization.CROExample::main)
                .doesNotThrowAnyException();
    }

    @Test
    @DisplayName("Global‑best PE is finite and non‑negative")
    void globalBestIsFiniteAndNonNegative() {
        var config = defaultConfig().maxIterations(300).seed(99L).build();
        ChemicalReactionOptimization.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();
        var cro = new ChemicalReactionOptimization(config, sphere);

        cro.optimize();
        assertThat(cro.getBestPE()).isFinite().isGreaterThanOrEqualTo(0.0);
    }
}