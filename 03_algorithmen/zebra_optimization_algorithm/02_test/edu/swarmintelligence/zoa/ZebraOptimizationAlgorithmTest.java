package edu.swarmintelligence.zoa;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DisplayName("Zebra Optimization Algorithm – Comprehensive Test Suite")
class ZebraOptimizationAlgorithmTest {
    // Known global minimum at origin (value = 0)
    private static final ZebraOptimizationAlgorithm.ObjectiveFunction SPHERE =
            x -> Arrays.stream(x).map(v -> v * v).sum();

    // Rastrigin function – global minimum 0 at origin, many local minima
    private static final ZebraOptimizationAlgorithm.ObjectiveFunction RASTRIGIN = x -> {
        double sum = 0;
        for (double v : x) sum += v * v - 10 * Math.cos(2 * Math.PI * v);
        return 10 * x.length + sum;
    };

    // Helper: create valid config with symmetric bounds [-5.12, 5.12]
    private ZebraOptimizationAlgorithm.ZoaConfig createConfig(int pop, int iter, int dim, long seed) {
        double[] min = new double[dim];
        double[] max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return new ZebraOptimizationAlgorithm.ZoaConfig(pop, iter, dim, min, max, seed);
    }

    // ---------- Configuration validation ----------
    @Nested
    @DisplayName("Configuration parameters")
    class ConfigValidation {
        @Test
        @DisplayName("Rejects illegal population size")
        void rejectsZeroPopulation() {
            double[] min = {0, 0};
            double[] max = {1, 1};
            assertThatThrownBy(() -> new ZebraOptimizationAlgorithm.ZoaConfig(0, 10, 2, min, max, 1L))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("PopulationSize must be > 0");
        }

        @Test
        @DisplayName("Rejects zero iterations")
        void rejectsZeroIterations() {
            double[] min = {0, 0};
            double[] max = {1, 1};
            assertThatThrownBy(() -> new ZebraOptimizationAlgorithm.ZoaConfig(10, 0, 2, min, max, 1L))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("MaxIterations must be > 0");
        }

        @Test
        @DisplayName("Rejects zero dimensions")
        void rejectsZeroDimensions() {
            double[] min = {0, 0};
            double[] max = {1, 1};
            assertThatThrownBy(() -> new ZebraOptimizationAlgorithm.ZoaConfig(10, 10, 0, min, max, 1L))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("Dimensions must be > 0");
        }

        @Test
        @DisplayName("Rejects mismatched bounds length")
        void rejectsMismatchedBounds() {
            double[] min = {0, 0, 0}; // length 3, dim = 2
            double[] max = {1, 1};
            assertThatThrownBy(() -> new ZebraOptimizationAlgorithm.ZoaConfig(10, 10, 2, min, max, 1L))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("Bounds must match dimensions");
        }

        @Test
        @DisplayName("Defensive copy of bounds preserves immutability")
        void boundsAreDefensiveCopies() {
            double[] originalMin = {-5, -5};
            double[] originalMax = {5, 5};
            ZebraOptimizationAlgorithm.ZoaConfig config =
                    new ZebraOptimizationAlgorithm.ZoaConfig(10, 10, 2, originalMin, originalMax, 1L);
            // Mutate the original arrays
            originalMin[0] = 999;
            originalMax[1] = -999;
            // The config should still hold the original values
            assertThat(config.minBounds()).containsExactly(-5, -5);
            assertThat(config.maxBounds()).containsExactly(5, 5);
        }
    }

    // ---------- Algorithm behaviour ----------
    @Nested
    @DisplayName("Optimization behaviour")
    class OptimizationBehaviour {
        @Test
        @DisplayName("Improves initial solution on Sphere function")
        void improvesSphereFunction() {
            int dims = 5;
            ZebraOptimizationAlgorithm zoa =
                    new ZebraOptimizationAlgorithm(createConfig(30, 200, dims, 42L), SPHERE);

            double initialFitness = zoa.getPioneerZebraFitness();
            double[] best = zoa.optimize();
            double finalFitness = SPHERE.evaluate(best);

            assertThat(finalFitness).isLessThan(initialFitness);
            assertThat(finalFitness).isLessThan(0.01);   // converges close to global minimum
        }

        @Test
        @DisplayName("Improves on Rastrigin function (multimodal)")
        void improvesRastriginFunction() {
            int dims = 2;
            ZebraOptimizationAlgorithm zoa =
                    new ZebraOptimizationAlgorithm(createConfig(50, 300, dims, 777L), RASTRIGIN);

            double initialFitness = zoa.getPioneerZebraFitness();
            double[] best = zoa.optimize();
            double finalFitness = RASTRIGIN.evaluate(best);

            assertThat(finalFitness).isLessThan(initialFitness);
            // ZOA should significantly reduce the Rastrigin value from its initial range
            assertThat(finalFitness).isLessThan(5.0);
        }

        @Test
        @DisplayName("Best solution respects search space boundaries")
        void bestSolutionRespectsBoundaries() {
            int dims = 3;
            ZebraOptimizationAlgorithm zoa =
                    new ZebraOptimizationAlgorithm(createConfig(40, 100, dims, 123L), SPHERE);
            double[] best = zoa.optimize();

            double[] min = createConfig(40, 100, dims, 123L).minBounds();
            double[] max = createConfig(40, 100, dims, 123L).maxBounds();

            for (int d = 0; d < dims; d++) {
                assertThat(best[d]).isBetween(min[d], max[d]);
            }
        }

        @Test
        @DisplayName("Deterministic output with same seed")
        void deterministicWithSameSeed() {
            int dims = 2;
            long seed = 1234L;
            ZebraOptimizationAlgorithm zoa1 =
                    new ZebraOptimizationAlgorithm(createConfig(20, 50, dims, seed), RASTRIGIN);
            ZebraOptimizationAlgorithm zoa2 =
                    new ZebraOptimizationAlgorithm(createConfig(20, 50, dims, seed), RASTRIGIN);

            double[] best1 = zoa1.optimize();
            double[] best2 = zoa2.optimize();

            assertThat(best1).containsExactly(best2);
        }

        @Test
        @DisplayName("Pioneer fitness never worsens during stepwise execution")
        void pioneerFitnessMonotonicallyImproves() {
            int dims = 5;
            int maxIter = 200;
            ZebraOptimizationAlgorithm zoa =
                    new ZebraOptimizationAlgorithm(createConfig(25, maxIter, dims, 1L), SPHERE);

            double previous = zoa.getPioneerZebraFitness();
            for (int iter = 1; iter <= maxIter; iter++) {
                zoa.runIteration(iter);
                double current = zoa.getPioneerZebraFitness();
                assertThat(current).isLessThanOrEqualTo(previous);
                previous = current;
            }
        }

        @ParameterizedTest
        @DisplayName("Total function evaluations matches formula 2*N*iter")
        @CsvSource({"10,100", "20,50", "5,200"})
        void totalEvaluationsCount(int pop, int iter) {
            ZebraOptimizationAlgorithm zoa =
                    new ZebraOptimizationAlgorithm(createConfig(pop, iter, 3, 0L), SPHERE);
            long expected = (long) iter * pop * 2L;
            assertThat(zoa.getTotalFunctionEvaluations(iter)).isEqualTo(expected);
        }

        @Test
        @DisplayName("Works with minimal population size (1)")
        void worksWithMinimalPopulation() {
            double[] min = {-5, -5};
            double[] max = {5, 5};
            ZebraOptimizationAlgorithm.ZoaConfig config =
                    new ZebraOptimizationAlgorithm.ZoaConfig(1, 10, 2, min, max, 42L);
            ZebraOptimizationAlgorithm zoa = new ZebraOptimizationAlgorithm(config, SPHERE);

            double[] best = zoa.optimize();
            assertThat(best).hasSize(2);
            assertThat(SPHERE.evaluate(best)).isNotNull();
        }

        @Test
        @DisplayName("Handles high-dimensional problems without error")
        void handlesHighDimensions() {
            int dims = 100;
            ZebraOptimizationAlgorithm zoa =
                    new ZebraOptimizationAlgorithm(createConfig(20, 10, dims, 42L), SPHERE);
            double[] best = zoa.optimize();
            assertThat(best).hasSize(dims);
            assertThat(SPHERE.evaluate(best)).isGreaterThanOrEqualTo(0);
        }
    }

    // ---------- Internal position integrity (optional white‑box) ----------
    @Nested
    @DisplayName("Internal integrity")
    class InternalIntegrity {
        @Test
        @DisplayName("Pioneer position returned by optimize() is a defensive copy")
        void pioneerPositionIsIndependentCopy() {
            ZebraOptimizationAlgorithm instance =
                    new ZebraOptimizationAlgorithm(createConfig(10, 5, 2, 0L), SPHERE);

            double[] bestFromOptimize = instance.optimize();
            double[] internalReference = instance.getPioneerPositionRaw();

            // Values equal, but different object reference
            assertThat(bestFromOptimize).containsExactly(internalReference);
            assertThat(bestFromOptimize).isNotSameAs(internalReference);

            // Mutate the “safe” copy – internal pioneer must stay unchanged
            bestFromOptimize[0] = 9999.0;
            assertThat(internalReference[0]).isNotEqualTo(9999.0);
        }
    }
}