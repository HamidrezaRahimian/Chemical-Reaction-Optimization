package edu.swarmintelligence.tso;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

@DisplayName("Tuna Swarm Optimization – scientific validation")
class TunaSwarmOptimizationTest {
    // ---------- Standard benchmark functions ----------
    private static final TunaSwarmOptimization.ObjectiveFunction sphere =
            x -> Arrays.stream(x).map(v -> v * v).sum();

    private static final TunaSwarmOptimization.ObjectiveFunction rastrigin = x -> {
        double sum = 10.0 * x.length;
        for (double v : x) sum += v * v - 10.0 * Math.cos(2.0 * Math.PI * v);
        return sum;
    };

    private static final TunaSwarmOptimization.ObjectiveFunction rosenbrock = x -> {
        double sum = 0.0;
        for (int i = 0; i < x.length - 1; i++) {
            double t1 = x[i + 1] - x[i] * x[i];
            double t2 = x[i] - 1.0;
            sum += 100.0 * t1 * t1 + t2 * t2;
        }
        return sum;
    };

    private static final TunaSwarmOptimization.ObjectiveFunction ackley = x -> {
        int n = x.length;
        double sumSq = 0.0, sumCos = 0.0;
        for (double v : x) {
            sumSq += v * v;
            sumCos += Math.cos(2.0 * Math.PI * v);
        }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                - Math.exp(sumCos / n) + 20.0 + Math.E;
    };

    private static double[] fill(double value, int length) {
        double[] a = new double[length];
        Arrays.fill(a, value);
        return a;
    }

    static Stream<Arguments> easyProblems() {
        return Stream.of(
                Arguments.of(new TestProblem("Sphere 5D", 5,
                        fill(-5.12, 5), fill(5.12, 5), 0.0, 30, 200), "sphere"),
                Arguments.of(new TestProblem("Sphere 10D", 10,
                        fill(-100, 10), fill(100, 10), 0.0, 100, 2000), "sphere")
        );
    }

    static Stream<Arguments> multimodalProblems() {
        return Stream.of(
                Arguments.of(new TestProblem("Rastrigin 2D", 2,
                        fill(-5.12, 2), fill(5.12, 2), 0.0, 30, 500), "rastrigin"),
                Arguments.of(new TestProblem("Rosenbrock 2D", 2,
                        fill(-2.048, 2), fill(2.048, 2), 0.0, 30, 500), "rosenbrock"),
                Arguments.of(new TestProblem("Ackley 5D", 5,
                        fill(-32.768, 5), fill(32.768, 5), 0.0, 100, 2000), "ackley")
        );
    }

    // ---------- Parameterised test data ----------
    record TestProblem(String name, int dims, double[] min, double[] max,
                       double globalOptimum, int popsize, int iterations) {
    }

    // ---------- Configuration validation ----------
    @Nested
    @DisplayName("Configuration contract")
    class ConfigValidation {
        @Test
        @DisplayName("Rejects population size <= 0")
        void rejectZeroPopulation() {
            assertThatThrownBy(() -> TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1}).seed(1L).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("PopulationSize");
        }

        @Test
        @DisplayName("Rejects mismatched bounds dimensions")
        void rejectMismatchedBounds() {
            assertThatThrownBy(() -> TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(5).maxIterations(100).dimensions(3)
                    .minBounds(new double[]{-1, -1}).maxBounds(new double[]{1, 1, 1}).seed(1L).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("Bounds");
        }

        @Test
        @DisplayName("Rejects negative dimensions")
        void rejectNegativeDimensions() {
            assertThatThrownBy(() -> TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(5).maxIterations(10).dimensions(-1)
                    .minBounds(new double[]{0}).maxBounds(new double[]{1}).seed(1L).build())
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("Dimensions");
        }
    }

    // ---------- Optimisation behaviour ----------
    @Nested
    @DisplayName("Optimisation performance")
    class Optimization {

        @ParameterizedTest(name = "{0}")
        @MethodSource("edu.swarmintelligence.tso.TunaSwarmOptimizationTest#easyProblems")
        @DisplayName("Converges to machine zero on unimodal landscapes")
        void convergesOnEasyProblems(TestProblem prob, String funcName) {
            TunaSwarmOptimization.ObjectiveFunction f = switch (funcName) {
                case "sphere" -> sphere;
                default -> throw new IllegalArgumentException("Unknown: " + funcName);
            };

            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(prob.popsize)
                    .maxIterations(prob.iterations)
                    .dimensions(prob.dims)
                    .minBounds(prob.min)
                    .maxBounds(prob.max)
                    .seed(42L)
                    .build();

            var tso = new TunaSwarmOptimization(config, f);
            double fitness = f.evaluate(tso.optimize());
            assertThat(fitness).isLessThan(1e-4);
        }

        @ParameterizedTest(name = "{0}")
        @MethodSource("edu.swarmintelligence.tso.TunaSwarmOptimizationTest#multimodalProblems")
        @DisplayName("Massively reduces fitness on multimodal landscapes")
        void improvesOnMultimodalProblems(TestProblem prob, String funcName) {
            TunaSwarmOptimization.ObjectiveFunction f = switch (funcName) {
                case "rastrigin" -> rastrigin;
                case "rosenbrock" -> rosenbrock;
                case "ackley" -> ackley;
                default -> throw new IllegalArgumentException("Unknown: " + funcName);
            };

            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(prob.popsize)
                    .maxIterations(prob.iterations)
                    .dimensions(prob.dims)
                    .minBounds(prob.min)
                    .maxBounds(prob.max)
                    .seed(7L)
                    .build();

            var tso = new TunaSwarmOptimization(config, f);
            double fitness = f.evaluate(tso.optimize());

            // Near‑optimal thresholds for these instances
            String lower = funcName.toLowerCase();
            if (lower.contains("rastrigin")) {
                assertThat(fitness).isLessThan(2.0);
            } else if (lower.contains("rosenbrock")) {
                assertThat(fitness).isLessThan(1e-3);
            } else if (lower.contains("ackley")) {
                assertThat(fitness).isLessThan(0.5);
            }
        }

        @Test
        @DisplayName("Solution stays strictly within bounds")
        @Tag("boundary")
        void solutionWithinBounds() {
            int dims = 4;
            double[] min = fill(-2.0, dims);
            double[] max = fill(2.0, dims);
            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(30).maxIterations(200).dimensions(dims)
                    .minBounds(min).maxBounds(max).seed(99L).build();

            var tso = new TunaSwarmOptimization(config, sphere);
            double[] sol = tso.optimize();
            for (int d = 0; d < dims; d++) {
                assertThat(sol[d]).isBetween(min[d], max[d]);
            }
        }

        @Test
        @DisplayName("Deterministic output with identical seed")
        void deterministicWithSameSeed() {
            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(20).maxIterations(100).dimensions(3)
                    .minBounds(new double[]{-1, -1, -1}).maxBounds(new double[]{1, 1, 1})
                    .seed(123L).build();

            double f1 = sphere.evaluate(new TunaSwarmOptimization(config, sphere).optimize());
            double f2 = sphere.evaluate(new TunaSwarmOptimization(config, sphere).optimize());
            assertThat(f1).isCloseTo(f2, within(1e-12));
        }

        @Test
        @DisplayName("Optimisation completes without exceptions (smoke test)")
        void smokeTestNoException() {
            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(10).maxIterations(10).dimensions(2)
                    .minBounds(new double[]{-1, -1}).maxBounds(new double[]{1, 1})
                    .seed(42L).build();
            assertDoesNotThrow(() -> new TunaSwarmOptimization(config, sphere).optimize());
        }

        @Test
        @DisplayName("Final fitness is drastically better than worst‑case initial")
        void improvementOverInitial() {
            int dims = 2;
            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(30).maxIterations(150).dimensions(dims)
                    .minBounds(new double[]{-10, -10}).maxBounds(new double[]{10, 10})
                    .seed(111L).build();
            var tso = new TunaSwarmOptimization(config, sphere);
            double finalFitness = sphere.evaluate(tso.optimize());
            // Worst random start: dims * 10^2 = 200
            assertThat(finalFitness).isLessThan(2.0);
        }

        @Test
        @DisplayName("Handles single dimension (edge case)")
        void singleDimension() {
            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(20).maxIterations(50).dimensions(1)
                    .minBounds(new double[]{-10}).maxBounds(new double[]{10})
                    .seed(5L).build();
            var tso = new TunaSwarmOptimization(config, sphere);
            double[] sol = tso.optimize();
            assertThat(sol).hasSize(1);
            assertThat(sol[0]).isBetween(-10.0, 10.0);
            assertThat(sphere.evaluate(sol)).isLessThan(1e-2);
        }

        @Test
        @DisplayName("No NaN or infinite values appear")
        void noNanOrInfinity() {
            var config = TunaSwarmOptimization.TsoConfig.builder()
                    .populationSize(30).maxIterations(300).dimensions(5)
                    .minBounds(fill(-1, 5)).maxBounds(fill(1, 5))
                    .seed(123456L).build();
            var tso = new TunaSwarmOptimization(config, sphere);
            double[] sol = tso.optimize();
            for (double d : sol) {
                assertThat(d).isFinite();
            }
        }
    }
}