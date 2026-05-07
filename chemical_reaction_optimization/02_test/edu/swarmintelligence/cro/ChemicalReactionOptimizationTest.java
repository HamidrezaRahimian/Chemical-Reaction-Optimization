package edu.swarmintelligence.cro;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Chemical Reaction Optimization (CRO)")
class ChemicalReactionOptimizationTest {
    private static final double SPHERE_MIN = -5.12;
    private static final double SPHERE_MAX = 5.12;

    static double sphere(double[] x) {
        return DoubleStream.of(x).map(v -> v * v).sum();
    }

    static ChemicalReactionOptimization.CroConfig.CroConfigBuilder defaultConfig() {
        int dimensions = 5;
        double[] min = new double[dimensions];
        double[] max = new double[dimensions];
        Arrays.fill(min, SPHERE_MIN);
        Arrays.fill(max, SPHERE_MAX);

        return ChemicalReactionOptimization.CroConfig.builder()
                .popSize(30)
                .maxIterations(500)
                .dimensions(dimensions)
                .minBounds(min)
                .maxBounds(max)
                .kelossRate(0.2)
                .moleColl(0.2)
                .decThres(10)
                .synThres(1.0)
                .initialKE(100.0)
                .enBuff(100.0)
                .stepSize(0.1)
                .seed(42L);
    }

    static ChemicalReactionOptimization.CroConfig oneDimensionalConfig(long seed) {
        return ChemicalReactionOptimization.CroConfig.builder()
                .popSize(12)
                .maxIterations(300)
                .dimensions(1)
                .minBounds(new double[]{-2.0})
                .maxBounds(new double[]{2.0})
                .kelossRate(0.2)
                .moleColl(0.2)
                .decThres(8)
                .synThres(1.0)
                .initialKE(20.0)
                .enBuff(20.0)
                .stepSize(0.05)
                .seed(seed)
                .build();
    }

    @Nested
    @DisplayName("Initialization")
    class Initialization {
        @Test
        @DisplayName("Initializes the configured molecule population with finite best energy")
        void initializesPopulationAndBestEnergy() {
            var config = defaultConfig().build();
            var cro = new ChemicalReactionOptimization(config, ChemicalReactionOptimizationTest::sphere);

            assertThat(cro.getPopulationSize()).isEqualTo(config.popSize());
            assertThat(cro.getBestPE()).isFinite().isGreaterThanOrEqualTo(0.0);
            assertThat(cro.getTotalSystemEnergy())
                    .isGreaterThan(config.enBuff() + config.popSize() * config.initialKE());
        }

        @Test
        @DisplayName("Fails fast when no feasible initial molecule can be created")
        void rejectsObjectiveThatNeverProducesFiniteEnergy() {
            var config = defaultConfig().build();

            assertThatThrownBy(() -> new ChemicalReactionOptimization(config, x -> Double.NaN))
                    .isInstanceOf(IllegalStateException.class)
                    .hasMessageContaining("No feasible molecule");
        }
    }

    @Nested
    @DisplayName("Optimization behavior")
    class OptimizationBehavior {
        @Test
        @DisplayName("Best energy never worsens during optimization")
        void globalBestNeverWorsens() {
            var cro = new ChemicalReactionOptimization(defaultConfig().seed(7L).build(),
                    ChemicalReactionOptimizationTest::sphere);
            double initialBest = cro.getBestPE();

            double[] best = cro.optimize();

            assertThat(cro.getBestPE()).isLessThanOrEqualTo(initialBest);
            assertThat(sphere(best)).isEqualTo(cro.getBestPE());
        }

        @Test
        @DisplayName("Converges to a lower value on a simple 1-D sphere problem")
        void improvesOnSimpleSphereProblem() {
            var cro = new ChemicalReactionOptimization(oneDimensionalConfig(3L),
                    ChemicalReactionOptimizationTest::sphere);
            double initialBest = cro.getBestPE();

            cro.optimize();

            assertThat(cro.getBestPE()).isLessThan(initialBest);
            assertThat(cro.getBestPE()).isLessThan(0.1);
        }

        @Test
        @DisplayName("Preserves total CRO energy across accepted and rejected reactions")
        void conservesTotalEnergy() {
            var config = defaultConfig()
                    .maxIterations(1000)
                    .seed(123L)
                    .build();
            var cro = new ChemicalReactionOptimization(config, ChemicalReactionOptimizationTest::sphere);
            double initialEnergy = cro.getTotalSystemEnergy();

            cro.optimize();

            assertThat(cro.getTotalSystemEnergy()).isCloseTo(initialEnergy, offset(1e-8));
        }

        @Test
        @DisplayName("Uses the seed deterministically")
        void sameSeedProducesSameResult() {
            var config = defaultConfig().seed(123456789L).build();

            var first = new ChemicalReactionOptimization(config, ChemicalReactionOptimizationTest::sphere);
            var second = new ChemicalReactionOptimization(config, ChemicalReactionOptimizationTest::sphere);

            double[] firstBest = first.optimize();
            double[] secondBest = second.optimize();

            assertThat(firstBest).containsExactly(secondBest);
            assertThat(first.getBestPE()).isEqualTo(second.getBestPE());
            assertThat(first.getPopulationSize()).isEqualTo(second.getPopulationSize());
        }
    }

    @Nested
    @DisplayName("CRO reaction dynamics")
    class ReactionDynamics {
        @Test
        @DisplayName("Synthesis may shrink the population but never below two molecules")
        void synthesisNeverDropsBelowTwoMolecules() {
            var config = ChemicalReactionOptimization.CroConfig.builder()
                    .popSize(4)
                    .maxIterations(30)
                    .dimensions(2)
                    .minBounds(new double[]{-1.0, -1.0})
                    .maxBounds(new double[]{1.0, 1.0})
                    .kelossRate(0.2)
                    .moleColl(1.0)
                    .decThres(10)
                    .synThres(100.0)
                    .initialKE(1.0)
                    .enBuff(0.0)
                    .stepSize(0.1)
                    .seed(99L)
                    .build();
            var cro = new ChemicalReactionOptimization(config, x -> 0.0);

            cro.optimize();

            assertThat(cro.getPopulationSize()).isEqualTo(2);
            assertThat(cro.getTotalSystemEnergy())
                    .isCloseTo(config.popSize() * config.initialKE(), offset(1e-10));
        }

        @Test
        @DisplayName("Two-molecule population remains valid when synthesis is not allowed")
        void twoMoleculePopulationRemainsValid() {
            var config = ChemicalReactionOptimization.CroConfig.builder()
                    .popSize(2)
                    .maxIterations(100)
                    .dimensions(2)
                    .minBounds(new double[]{-1.0, -1.0})
                    .maxBounds(new double[]{1.0, 1.0})
                    .kelossRate(0.2)
                    .moleColl(1.0)
                    .decThres(10)
                    .synThres(100.0)
                    .initialKE(1.0)
                    .enBuff(0.0)
                    .stepSize(0.1)
                    .seed(5L)
                    .build();
            var cro = new ChemicalReactionOptimization(config, x -> 0.0);

            assertThatCode(cro::optimize).doesNotThrowAnyException();
            assertThat(cro.getPopulationSize()).isEqualTo(2);
        }
    }

    @Nested
    @DisplayName("Boundaries and result integrity")
    class BoundariesAndResultIntegrity {
        @Test
        @DisplayName("Best position remains within configured bounds")
        void bestPositionStaysInsideBounds() {
            var config = defaultConfig()
                    .popSize(50)
                    .maxIterations(1500)
                    .stepSize(1.0)
                    .seed(77L)
                    .build();
            var cro = new ChemicalReactionOptimization(config, ChemicalReactionOptimizationTest::sphere);

            double[] best = cro.optimize();

            assertThat(best).hasSize(config.dimensions());
            for (double coordinate : best) {
                assertThat(coordinate).isBetween(SPHERE_MIN, SPHERE_MAX);
            }
        }

        @Test
        @DisplayName("Returned best position is a defensive copy")
        void bestPositionResultIsDefensiveCopy() {
            var cro = new ChemicalReactionOptimization(oneDimensionalConfig(11L),
                    ChemicalReactionOptimizationTest::sphere);

            double[] firstBest = cro.optimize();
            firstBest[0] = 999.0;
            double[] secondBest = cro.optimize();

            assertThat(secondBest[0]).isNotEqualTo(999.0);
            assertThat(secondBest[0]).isBetween(-2.0, 2.0);
        }

        @Test
        @DisplayName("Objective evaluation cannot mutate internal molecule structures")
        void objectiveReceivesDefensiveCopy() {
            AtomicBoolean mutatedInput = new AtomicBoolean(false);
            ChemicalReactionOptimization.ObjectiveFunction mutatingSphere = x -> {
                double value = sphere(x);
                x[0] = 999.0;
                mutatedInput.set(true);
                return value;
            };
            var cro = new ChemicalReactionOptimization(oneDimensionalConfig(17L), mutatingSphere);

            double[] best = cro.optimize();

            assertThat(mutatedInput).isTrue();
            assertThat(best[0]).isBetween(-2.0, 2.0);
        }
    }

    @Nested
    @DisplayName("Invalid fitness handling")
    class InvalidFitnessHandling {
        @Test
        @DisplayName("Skips non-finite candidate energies without corrupting the best result")
        void nonFiniteCandidatesDoNotBecomeBest() {
            ChemicalReactionOptimization.ObjectiveFunction partiallyInvalid = x -> {
                if (x[0] > 0.5) {
                    return Double.NaN;
                }
                return sphere(x);
            };
            var config = ChemicalReactionOptimization.CroConfig.builder()
                    .popSize(20)
                    .maxIterations(500)
                    .dimensions(2)
                    .minBounds(new double[]{-2.0, -2.0})
                    .maxBounds(new double[]{2.0, 2.0})
                    .kelossRate(0.2)
                    .moleColl(0.2)
                    .decThres(8)
                    .synThres(1.0)
                    .initialKE(20.0)
                    .enBuff(20.0)
                    .stepSize(0.1)
                    .seed(8L)
                    .build();

            var cro = new ChemicalReactionOptimization(config, partiallyInvalid);
            double[] best = cro.optimize();

            assertThat(cro.getBestPE()).isFinite();
            assertThat(partiallyInvalid.evaluate(best)).isFinite();
            assertThat(best[0]).isLessThanOrEqualTo(0.5);
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class ConfigurationValidation {
        @Test
        @DisplayName("Rejects invalid population, dimensions and iteration settings")
        void rejectsInvalidCoreSettings() {
            assertThatThrownBy(() -> defaultConfig().popSize(1).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("PopSize");
            assertThatThrownBy(() -> defaultConfig().maxIterations(0).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("MaxIterations");
            assertThatThrownBy(() -> defaultConfig().dimensions(0).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("Dimensions");
        }

        @Test
        @DisplayName("Rejects invalid bounds")
        void rejectsInvalidBounds() {
            assertThatThrownBy(() -> defaultConfig()
                    .minBounds(new double[]{1.0})
                    .maxBounds(new double[]{1.0, 2.0})
                    .build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("Bounds");

            assertThatThrownBy(() -> ChemicalReactionOptimization.CroConfig.builder()
                    .popSize(2)
                    .maxIterations(1)
                    .dimensions(1)
                    .minBounds(new double[]{1.0})
                    .maxBounds(new double[]{1.0})
                    .kelossRate(0.2)
                    .moleColl(0.2)
                    .decThres(1)
                    .synThres(1.0)
                    .initialKE(1.0)
                    .enBuff(0.0)
                    .stepSize(0.1)
                    .seed(1L)
                    .build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("smaller than max");
        }

        @Test
        @DisplayName("Rejects invalid CRO control parameters")
        void rejectsInvalidCroParameters() {
            assertThatThrownBy(() -> defaultConfig().kelossRate(Double.NaN).build())
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
            assertThatThrownBy(() -> defaultConfig().stepSize(Double.POSITIVE_INFINITY).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensively copies bound arrays")
        void boundsAreDefensiveCopies() {
            double[] min = {-1.0};
            double[] max = {1.0};
            var config = ChemicalReactionOptimization.CroConfig.builder()
                    .popSize(2)
                    .maxIterations(1)
                    .dimensions(1)
                    .minBounds(min)
                    .maxBounds(max)
                    .kelossRate(0.2)
                    .moleColl(0.2)
                    .decThres(1)
                    .synThres(1.0)
                    .initialKE(1.0)
                    .enBuff(0.0)
                    .stepSize(0.1)
                    .seed(1L)
                    .build();

            min[0] = -999.0;
            max[0] = 999.0;
            double[] returnedMin = config.minBounds();
            returnedMin[0] = -500.0;

            assertThat(config.minBounds()).containsExactly(-1.0);
            assertThat(config.maxBounds()).containsExactly(1.0);
        }
    }

    @Test
    @DisplayName("CRO example runs without throwing")
    void exampleRunsWithoutThrowing() {
        assertThatCode(ChemicalReactionOptimization.CROExample::main)
                .doesNotThrowAnyException();
    }
}
