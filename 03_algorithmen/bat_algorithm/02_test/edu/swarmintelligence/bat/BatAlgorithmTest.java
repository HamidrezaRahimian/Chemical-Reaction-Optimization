package edu.swarmintelligence.bat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertAll;

@DisplayName("Bat Algorithm – Yang (2010)")
class BatAlgorithmTest {
    static double sphere(double[] x) {
        return DoubleStream.of(x).map(v -> v * v).sum();
    }

    static BatAlgorithm.BaConfig easyConfig(long seed) {
        int dim = 2;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return BatAlgorithm.BaConfig.builder()
                .populationSize(50)
                .maxIterations(1000)
                .dimensions(dim)
                .minBounds(min)
                .maxBounds(max)
                .fMin(0).fMax(2)
                .alpha(0.9).gamma(0.9)
                .initialLoudness(1.0)
                .pulseRateMax(0.5)
                .seed(seed)
                .build();
    }

    @Nested
    @DisplayName("Correctness & convergence")
    class Convergence {
        @Test
        @DisplayName("Converges close to global optimum on 2‑D Sphere")
        void convergesOnSphere() {
            var cfg = easyConfig(1);
            var ba = new BatAlgorithm(cfg, BatAlgorithmTest::sphere);
            var res = ba.optimize();
            assertThat(res.bestFitness()).isLessThan(0.1);
        }

        @Test
        @DisplayName("Convergence history is monotonically non‑increasing")
        void historyIsMonotonic() {
            var cfg = easyConfig(99);
            var ba = new BatAlgorithm(cfg, BatAlgorithmTest::sphere);
            var res = ba.optimize();
            var h = res.convergenceHistory();
            for (int i = 1; i < h.length; i++)
                assertThat(h[i]).isLessThanOrEqualTo(h[i - 1]);
        }

        @Test
        @DisplayName("Convergence history length = maxIterations + 1")
        void historyLength() {
            var cfg = easyConfig(1);
            var ba = new BatAlgorithm(cfg, BatAlgorithmTest::sphere);
            var res = ba.optimize();
            assertThat(res.convergenceHistory()).hasSize(cfg.maxIterations() + 1);
        }

        @Test
        @DisplayName("All positions remain within search bounds")
        void positionsWithinBounds() {
            var cfg = easyConfig(1);
            var ba = new BatAlgorithm(cfg, BatAlgorithmTest::sphere);
            var res = ba.optimize();
            for (double v : res.bestPosition())
                assertThat(v).isBetween(-5.12, 5.12);
        }
    }

    @Nested
    @DisplayName("Determinism")
    class Determinism {
        @RepeatedTest(3)
        @DisplayName("Same seed → identical fitness and convergence history")
        void sameSeed() {
            var cfg = easyConfig(777);
            var ba1 = new BatAlgorithm(cfg, BatAlgorithmTest::sphere);
            var ba2 = new BatAlgorithm(cfg, BatAlgorithmTest::sphere);
            var res1 = ba1.optimize();
            var res2 = ba2.optimize();
            assertAll(
                    () -> assertThat(res1.bestFitness()).isEqualTo(res2.bestFitness()),
                    () -> assertThat(res1.convergenceHistory()).containsExactly(res2.convergenceHistory())
            );
        }

        @Test
        @DisplayName("Different seeds → different convergence histories")
        void differentSeedsDiffer() {
            var res1 = new BatAlgorithm(easyConfig(1), BatAlgorithmTest::sphere).optimize();
            var res2 = new BatAlgorithm(easyConfig(2), BatAlgorithmTest::sphere).optimize();
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
            assertThatThrownBy(() -> BatAlgorithm.BaConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .fMin(0).fMax(2).alpha(0.5).gamma(0.5)
                    .initialLoudness(1).pulseRateMax(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects alpha outside (0,1)")
        void invalidAlpha() {
            assertThatThrownBy(() -> BatAlgorithm.BaConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .fMin(0).fMax(2).alpha(1.0).gamma(0.5)
                    .initialLoudness(1).pulseRateMax(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects pulseRateMax outside [0,1]")
        void invalidPulseRateMax() {
            assertThatThrownBy(() -> BatAlgorithm.BaConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .fMin(0).fMax(2).alpha(0.9).gamma(0.9)
                    .initialLoudness(1).pulseRateMax(1.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> BatAlgorithm.BaConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .fMin(0).fMax(2).alpha(0.9).gamma(0.9)
                    .initialLoudness(1).pulseRateMax(-0.1).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensive copies of bounds arrays")
        void boundsDefensiveCopy() {
            var min = new double[]{0, 0};
            var max = new double[]{1, 1};
            var cfg = BatAlgorithm.BaConfig.builder()
                    .populationSize(2).maxIterations(10).dimensions(2)
                    .minBounds(min).maxBounds(max)
                    .fMin(0).fMax(2).alpha(0.5).gamma(0.5)
                    .initialLoudness(1).pulseRateMax(0.5).seed(0).build();
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
            var res = new BatAlgorithm(easyConfig(42), BatAlgorithmTest::sphere).optimize();
            var p = res.bestPosition();
            p[0] = Double.MAX_VALUE;
            assertThat(res.bestPosition()[0]).isNotEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Result convergenceHistory is a defensive copy")
        void historyDefensiveCopy() {
            var res = new BatAlgorithm(easyConfig(42), BatAlgorithmTest::sphere).optimize();
            var h = res.convergenceHistory();
            h[0] = -999;
            assertThat(res.convergenceHistory()[0]).isNotEqualTo(-999);
        }

        @Test
        @DisplayName("getBestFitness matches Result.bestFitness after optimize()")
        void bestFitnessMatches() {
            var ba = new BatAlgorithm(easyConfig(42), BatAlgorithmTest::sphere);
            var res = ba.optimize();
            assertThat(ba.getBestFitness()).isEqualTo(res.bestFitness());
        }
    }

    @Nested
    @DisplayName("Bat dynamics (pulse rate & loudness)")
    class Dynamics {
        @Test
        @DisplayName("All bats start with pulse rate = 0")
        void initialPulseRateIsZero() {
            // Before optimize, bats are initialized
            var ba = new BatAlgorithm(easyConfig(123), BatAlgorithmTest::sphere);
            double[] rates = ba.getPulseRates();
            assertThat(Arrays.stream(rates)).allMatch(r -> r == 0.0);
        }

        @Test
        @DisplayName("All bats start with initial loudness")
        void initialLoudnessMatchesConfig() {
            var cfg = easyConfig(123);
            var ba = new BatAlgorithm(cfg, BatAlgorithmTest::sphere);
            double[] loudness = ba.getLoudnessValues();
            assertThat(Arrays.stream(loudness))
                    .allMatch(l -> l == cfg.initialLoudness());
        }

        @Test
        @DisplayName("Pulse rate of some bats becomes > 0 after acceptance")
        void pulseRateIncreases() {
            var ba = new BatAlgorithm(easyConfig(123), BatAlgorithmTest::sphere);
            ba.optimize();
            var rates = ba.getPulseRates();
            assertThat(Arrays.stream(rates)).anyMatch(r -> r > 0.0);
        }

        @Test
        @DisplayName("Loudness of accepted bats decreases")
        void loudnessDecreasesForSome() {
            var ba = new BatAlgorithm(easyConfig(123), BatAlgorithmTest::sphere);
            ba.optimize();
            var loudness = ba.getLoudnessValues();
            assertThat(Arrays.stream(loudness)).anyMatch(l -> l < 1.0);
        }
    }
}