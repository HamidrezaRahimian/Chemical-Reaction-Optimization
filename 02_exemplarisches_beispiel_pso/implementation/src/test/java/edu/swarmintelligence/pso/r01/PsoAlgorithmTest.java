package edu.swarmintelligence.pso.r01;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertAll;

@DisplayName("PSO – global‑best topology")
class PsoAlgorithmTest {
    // ──────────────────────── Helper objective functions ─────────────────────────

    static double sphere(double[] x) {
        return DoubleStream.of(x).map(v -> v * v).sum();
    }

    static double ackley(double[] x) {
        int n = x.length;
        double sumSq = 0.0, sumCos = 0.0;
        for (double v : x) {
            sumSq += v * v;
            sumCos += Math.cos(2.0 * Math.PI * v);
        }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                - Math.exp(sumCos / n) + 20.0 + Math.E;
    }

    // ──────────────────────── Configuration factories ────────────────────────────

    static PsoAlgorithm.PsoConfig easyConfig(long seed) {
        int dim = 2;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return PsoAlgorithm.PsoConfig.builder()
                .populationSize(50)
                .maxIterations(1000)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    static PsoAlgorithm.PsoConfig ackleyConfig(long seed) {
        int dim = 10;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        return PsoAlgorithm.PsoConfig.builder()
                .populationSize(50)
                .maxIterations(500)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(seed)
                .build();
    }

    // ──────────────────────── Nested test groups ─────────────────────────────────

    @Nested
    @DisplayName("Correctness & convergence")
    class Convergence {
        @Test
        @DisplayName("Converges close to global optimum on 2‑D Sphere")
        void convergesOnSphere() {
            var cfg = easyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            assertThat(res.bestFitness()).isLessThan(0.01);
        }

        @Test
        @DisplayName("Converges near global optimum on 10‑D Ackley")
        void convergesOnAckley() {
            var cfg = ackleyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::ackley);
            var res = pso.optimize();
            assertThat(res.bestFitness()).isLessThan(0.5);
        }

        @Test
        @DisplayName("Convergence history is monotonically non‑increasing")
        void historyIsMonotonic() {
            var cfg = easyConfig(99);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            var h = res.convergenceHistory();
            for (int i = 1; i < h.length; i++) {
                assertThat(h[i]).isLessThanOrEqualTo(h[i - 1]);
            }
        }

        @Test
        @DisplayName("Convergence history length = maxIterations + 1")
        void historyLength() {
            var cfg = easyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            assertThat(res.convergenceHistory()).hasSize(cfg.maxIterations() + 1);
        }

        @Test
        @DisplayName("Best position remains within search bounds")
        void bestPositionWithinBounds() {
            var cfg = easyConfig(1);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            for (double v : res.bestPosition()) {
                assertThat(v).isBetween(-5.12, 5.12);
            }
        }
    }

    @Nested
    @DisplayName("Determinism")
    class Determinism {
        @RepeatedTest(3)
        @DisplayName("Same seed → identical fitness and convergence history")
        void sameSeed() {
            var cfg = easyConfig(777);
            var pso1 = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var pso2 = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            var res1 = pso1.optimize();
            var res2 = pso2.optimize();
            assertAll(
                    () -> assertThat(res1.bestFitness()).isEqualTo(res2.bestFitness()),
                    () -> assertThat(res1.convergenceHistory()).containsExactly(res2.convergenceHistory())
            );
        }

        @Test
        @DisplayName("Different seeds → different convergence histories")
        void differentSeedsDiffer() {
            var res1 = new PsoAlgorithm(easyConfig(1), PsoAlgorithmTest::sphere).optimize();
            var res2 = new PsoAlgorithm(easyConfig(2), PsoAlgorithmTest::sphere).optimize();
            assertThat(res1.convergenceHistory())
                    .isNotEqualTo(res2.convergenceHistory());
        }
    }

    @Nested
    @DisplayName("Configuration validation")
    class Config {
        @Test
        @DisplayName("Rejects invalid popSize (0)")
        void invalidPopSize() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects inertia outside (0,1]")
        void invalidInertia() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.0).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(1.5).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects negative cognitive/social coefficients")
        void invalidCoefficients() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(-0.1).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(-1.0)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects velocityClampFactor outside (0,1]")
        void invalidVelocityClamp() {
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.0).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoAlgorithm.PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(1.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensive copies of bounds arrays")
        void boundsDefensiveCopy() {
            var min = new double[]{0, 0};
            var max = new double[]{1, 1};
            var cfg = PsoAlgorithm.PsoConfig.builder()
                    .populationSize(2).maxIterations(10).dimensions(2)
                    .minBounds(min).maxBounds(max)
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(0).build();
            min[0] = 999;
            assertThat(cfg.minBounds()[0]).isEqualTo(0.0);
        }
    }

    @Nested
    @DisplayName("Result integrity")
    class ResultIntegrity {
        @Test
        @DisplayName("Result bestPosition is a defensive copy")
        void positionDefensiveCopy() {
            var res = new PsoAlgorithm(easyConfig(42), PsoAlgorithmTest::sphere).optimize();
            var p = res.bestPosition();
            p[0] = Double.MAX_VALUE;
            assertThat(res.bestPosition()[0]).isNotEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Result convergenceHistory is a defensive copy")
        void historyDefensiveCopy() {
            var res = new PsoAlgorithm(easyConfig(42), PsoAlgorithmTest::sphere).optimize();
            var h = res.convergenceHistory();
            h[0] = -999;
            assertThat(res.convergenceHistory()[0]).isNotEqualTo(-999);
        }

        @Test
        @DisplayName("getBestFitness matches Result.bestFitness after optimize()")
        void bestFitnessMatches() {
            var pso = new PsoAlgorithm(easyConfig(42), PsoAlgorithmTest::sphere);
            var res = pso.optimize();
            assertThat(pso.getBestFitness()).isEqualTo(res.bestFitness());
        }
    }

    @Nested
    @DisplayName("Swarm dynamics (velocity & personal best)")
    class Dynamics {
        @Test
        @DisplayName("Initial velocities respect vMax")
        void initialVelocityWithinVMax() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            double[][] vels = pso.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("After iterations velocities remain within vMax (clamping effect)")
        void velocityClampedAfterUpdate() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            pso.optimize();
            double[][] vels = pso.getVelocities();
            double vMax = cfg.velocityClampFactor() * (cfg.maxBounds()[0] - cfg.minBounds()[0]);
            for (double[] v : vels) {
                for (double vd : v) {
                    assertThat(vd).isBetween(-vMax, vMax);
                }
            }
        }

        @Test
        @DisplayName("Personal best fitness improves or stays the same over time")
        void personalBestNeverDegrades() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            double[] initialPBest = pso.getPersonalBestFitnesses();
            pso.optimize();
            double[] finalPBest = pso.getPersonalBestFitnesses();
            for (int i = 0; i < initialPBest.length; i++) {
                assertThat(finalPBest[i]).isLessThanOrEqualTo(initialPBest[i]);
            }
        }

        @Test
        @DisplayName("Global best fitness is the minimum personal best fitness")
        void globalBestMatchesMinPersonalBest() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            pso.optimize();
            double minPBest = Arrays.stream(pso.getPersonalBestFitnesses()).min().orElse(Double.NaN);
            assertThat(pso.getBestFitness()).isEqualTo(minPBest);
        }

        @Test
        @DisplayName("All particles' positions stay within bounds")
        void positionsWithinBounds() {
            var cfg = easyConfig(123);
            var pso = new PsoAlgorithm(cfg, PsoAlgorithmTest::sphere);
            pso.optimize();
            double[][] pos = pso.getPositions();
            double lb = cfg.minBounds()[0];
            double ub = cfg.maxBounds()[0];
            for (double[] p : pos) {
                for (double pv : p) {
                    assertThat(pv).isBetween(lb, ub);
                }
            }
        }
    }
}