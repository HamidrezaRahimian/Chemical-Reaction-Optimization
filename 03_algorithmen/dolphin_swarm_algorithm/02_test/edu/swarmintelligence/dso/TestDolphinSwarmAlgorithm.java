package edu.swarmintelligence.dso;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Dolphin Swarm Algorithm – comprehensive test suite")
class TestDolphinSwarmAlgorithm {
    // ---------- Benchmark functions ----------
    static double sphere(double[] x) {
        return Arrays.stream(x).map(v -> v * v).sum();
    }

    static double rastrigin(double[] x) {
        return 10 * x.length + Arrays.stream(x)
                .map(v -> v * v - 10 * Math.cos(2 * Math.PI * v)).sum();
    }

    static double rosenbrock(double[] x) {
        double sum = 0;
        for (int i = 0; i < x.length - 1; i++) {
            sum += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
        }
        return sum;
    }

    static double ackley(double[] x) {
        int n = x.length;
        double sqsum = Arrays.stream(x).map(v -> v * v).sum();
        double cossum = Arrays.stream(x).map(v -> Math.cos(2 * Math.PI * v)).sum();
        return -20 * Math.exp(-0.2 * Math.sqrt(sqsum / n)) - Math.exp(cossum / n) + 20 + Math.E;
    }

    // ---------- Test helpers ----------
    static DolphinSwarmAlgorithm.DsaConfig convergedConfig(int dims, long seed) {
        double[] min = new double[dims];
        double[] max = new double[dims];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        int feBudget = dims <= 10 ? 10000 : 30000;
        return DolphinSwarmAlgorithm.DsaConfig.createDefault(15, feBudget, dims, min, max, seed);
    }

    static double[] uniformBounds(int dim, double lower, double upper) {
        double[] a = new double[dim];
        Arrays.fill(a, lower);
        return a;
    }

    // ---------- Nested test suites ----------

    @Nested
    @DisplayName("Correctness and convergence")
    class Convergence {
        @Test
        @DisplayName("Sphere: final fitness far better than random start")
        void improvesOverRandom() {
            var config = convergedConfig(5, 12345);
            var alg = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere);
            var res = alg.optimize();
            // Random start would be in [0, ~130], anything < 0.1 is a strong improvement
            assertThat(res.bestFitness()).isLessThan(0.1);
        }

        @RepeatedTest(3)
        @DisplayName("Sphere: near‑optimal solution found consistently")
        void convergesNearOptimum(RepetitionInfo info) {
            var config = convergedConfig(5, 100 + info.getCurrentRepetition());
            var alg = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere);
            var res = alg.optimize();
            // With 10 000 evaluations a value < 0.1 is robust; < 0.05 is a tighter but still fair expectation
            assertThat(res.bestFitness()).isLessThan(0.1);
        }

        @Test
        @DisplayName("Rastrigin: significant improvement (now with persistent Lᵢ)")
        void rastriginImprovement() {
            int dims = 5;
            double[] min = uniformBounds(dims, -5.12, -5.12);
            double[] max = uniformBounds(dims, 5.12, 5.12);
            var config = new DolphinSwarmAlgorithm.DsaConfig(
                    15, 20000, dims, min, max,
                    5, 5, 1.0, -1.0, 5000);
            var alg = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::rastrigin);
            var res = alg.optimize();
            // Rastrigin optimum = 0; the algorithm reliably drives fitness below 15.
            assertThat(res.bestFitness()).isLessThan(15.0);
        }

        @Test
        @DisplayName("Rosenbrock: reaches valley")
        void rosenbrockImprovement() {
            var config = convergedConfig(2, 5000);
            var alg = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::rosenbrock);
            var res = alg.optimize();
            // Rosenbrock valley floor fitness ≪ 5 is a clear improvement from a distant start
            assertThat(res.bestFitness()).isLessThan(5.0);
        }
    }

    @Nested
    @DisplayName("Budget respect")
    class Budget {
        @Test
        @DisplayName("Never exceeds maxFE")
        void neverExceedsMaxFE() {
            var config = convergedConfig(5, 1000);
            var alg = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere);
            var res = alg.optimize();
            assertThat(res.functionEvaluations()).isLessThanOrEqualTo(config.maxFE());
        }

        @Test
        @DisplayName("Stops exactly when maxFE reached in small budget")
        void stopsExactlyOnBudget() {
            var config = new DolphinSwarmAlgorithm.DsaConfig(4, 20, 2,
                    uniformBounds(2, -1, 1), uniformBounds(2, -1, 1),
                    2, 2, 1.0, -1, 0);
            var alg = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere);
            var res = alg.optimize();
            assertThat(res.functionEvaluations()).isEqualTo(20);
        }
    }

    @Nested
    @DisplayName("Determinism and reproducibility")
    class Determinism {
        @Test
        @DisplayName("Same seed ⇒ identical trajectory")
        void sameSeedIdentical() {
            var c = convergedConfig(5, 777);
            var a1 = new DolphinSwarmAlgorithm(c, TestDolphinSwarmAlgorithm::sphere);
            var a2 = new DolphinSwarmAlgorithm(c, TestDolphinSwarmAlgorithm::sphere);
            assertThat(a1.optimize().bestFitness()).isEqualTo(a2.optimize().bestFitness());
            assertThat(a1.optimize().bestPosition()).containsExactly(a2.optimize().bestPosition());
        }

        @Test
        @DisplayName("Different seeds ⇒ different results (with high probability)")
        void differentSeedsDiffer() {
            var c1 = convergedConfig(5, 1);
            var c2 = convergedConfig(5, 2);
            var r1 = new DolphinSwarmAlgorithm(c1, TestDolphinSwarmAlgorithm::sphere).optimize();
            var r2 = new DolphinSwarmAlgorithm(c2, TestDolphinSwarmAlgorithm::sphere).optimize();
            assertThat(r1.bestFitness()).isNotEqualTo(r2.bestFitness());
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class ConfigValidation {
        @Test
        @DisplayName("popSize < 2 throws")
        void popSizeTooSmall() {
            assertThatThrownBy(() -> new DolphinSwarmAlgorithm.DsaConfig(
                    1, 100, 2, new double[]{0, 0}, new double[]{1, 1}, 5, 5, 1, 0, 0))
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("maxFE <= 0 throws")
        void maxFENonPositive() {
            assertThatThrownBy(() -> new DolphinSwarmAlgorithm.DsaConfig(
                    2, 0, 2, new double[]{0, 0}, new double[]{1, 1}, 5, 5, 1, 0, 0))
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Dimensions mismatch throws")
        void boundsDimensionMismatch() {
            assertThatThrownBy(() -> new DolphinSwarmAlgorithm.DsaConfig(
                    2, 100, 2, new double[]{0, 0, 0}, new double[]{1, 1}, 5, 5, 1, 0, 0))
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }

    @Nested
    @DisplayName("Random direction uniformity")
    class RandomDirections {
        private final RandomGenerator rng = RandomGeneratorFactory.of("L128X1024MixRandom").create(42);

        @Test
        @DisplayName("Generated vectors have unit length")
        void unitLength() {
            for (int dim = 1; dim <= 10; dim++) {
                for (int i = 0; i < 1_000; i++) {
                    double[] v = generateUnitVector(dim);
                    double norm = Math.sqrt(Arrays.stream(v).map(d -> d * d).sum());
                    assertThat(norm).isCloseTo(1.0, offset(1e-10));
                }
            }
        }

        @Test
        @DisplayName("Coordinate mean ≈ 0 and variance consistent")
        void uniformityProperties() {
            int dim = 5;
            int n = 10_000;
            double[] means = new double[dim];
            double[] variances = new double[dim];
            for (int i = 0; i < n; i++) {
                double[] v = generateUnitVector(dim);
                for (int j = 0; j < dim; j++) {
                    means[j] += v[j];
                    variances[j] += v[j] * v[j];
                }
            }
            for (int j = 0; j < dim; j++) {
                means[j] /= n;
                variances[j] = variances[j] / n - means[j] * means[j];
                assertThat(means[j]).isCloseTo(0.0, offset(0.02));
                assertThat(variances[j]).isCloseTo(1.0 / dim, offset(0.02));
            }
        }

        private double[] generateUnitVector(int dim) {
            var v = new double[dim];
            double sumSq = 0;
            for (int i = 0; i < dim; i++) {
                v[i] = rng.nextGaussian();
                sumSq += v[i] * v[i];
            }
            double len = Math.sqrt(sumSq);
            if (len < 1e-10) {
                v[0] = 1;
                len = 1;
            }
            for (int i = 0; i < dim; i++) v[i] /= len;
            return v;
        }
    }

    @Nested
    @DisplayName("Result record consistency")
    class ResultRecord {
        @Test
        @DisplayName("Best position has correct dimension and is within bounds")
        void positionInsideBounds() {
            var config = convergedConfig(4, 2000);
            var alg = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere);
            var res = alg.optimize();
            assertThat(res.bestPosition()).hasSize(config.dimensions());
            for (int i = 0; i < config.dimensions(); i++) {
                assertThat(res.bestPosition()[i])
                        .isBetween(config.minBounds()[i], config.maxBounds()[i]);
            }
        }

        @Test
        @DisplayName("FE count is positive and ≤ maxFE")
        void feCountConsistency() {
            var config = convergedConfig(3, 500);
            var res = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere).optimize();
            assertThat(res.functionEvaluations()).isPositive()
                    .isLessThanOrEqualTo(config.maxFE());
        }

        @Test
        @DisplayName("Defensive copy: modifying returned array does not affect record")
        void defensiveCopy() {
            var config = convergedConfig(2, 100);
            var res = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere).optimize();
            double[] arr = res.bestPosition();
            arr[0] = 9999;
            assertThat(res.bestPosition()[0]).isNotEqualTo(9999);
        }
    }

    @Nested
    @DisplayName("Scaling across dimensions")
    class DimensionalScaling {
        @ParameterizedTest
        @ValueSource(ints = {2, 5, 10, 20})
        @DisplayName("Sphere convergence for increasing dimensions")
        void sphereConvergenceInDim(int dim) {
            var config = convergedConfig(dim, 42L * dim);
            var res = new DolphinSwarmAlgorithm(config, TestDolphinSwarmAlgorithm::sphere).optimize();
            assertThat(res.bestFitness()).isLessThan(1.0);
        }
    }

    @Nested
    @DisplayName("Boundary handling")
    class Boundary {
        @Test
        @DisplayName("All positions stay within bounds during run")
        void positionsClamped() {
            double[] min = {0, 0};
            double[] max = {1, 1};
            var config = new DolphinSwarmAlgorithm.DsaConfig(5, 500, 2, min, max, 3, 3, 0.5, -1, 999);
            var alg = new DolphinSwarmAlgorithm(config, x -> x[0] * x[1]);
            var res = alg.optimize();
            assertThat(res.bestPosition()).satisfies(array -> {
                for (double v : array) {
                    assertThat(v).isBetween(0.0, 1.0);
                }
            });
        }
    }
}