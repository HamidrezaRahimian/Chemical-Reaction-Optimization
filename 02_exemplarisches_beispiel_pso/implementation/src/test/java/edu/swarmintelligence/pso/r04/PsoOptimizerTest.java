package edu.swarmintelligence.pso.r04;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertAll;

@DisplayName("PSO – global‑best topology (refactored, with logging)")
class PsoOptimizerTest {
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

    static PsoConfig easyConfig(long seed) {
        int dim = 2;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return PsoConfig.builder()
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

    static PsoConfig ackleyConfig(long seed) {
        int dim = 10;
        var min = new double[dim];
        var max = new double[dim];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        return PsoConfig.builder()
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

    // ─── existing tests (adapted to no‑logging constructor) ───────────────

    @Nested
    @DisplayName("Correctness & convergence")
    class Convergence {
        @Test
        @DisplayName("Converges close to global optimum on 2‑D Sphere")
        void convergesOnSphere() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.01);
        }

        @Test
        @DisplayName("Converges near global optimum on 10‑D Ackley")
        void convergesOnAckley() {
            var optimizer = new PsoOptimizer(ackleyConfig(1), PsoOptimizerTest::ackley, null);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.5);
        }

        @Test
        @DisplayName("Convergence history is monotonically non‑increasing")
        void historyIsMonotonic() {
            var optimizer = new PsoOptimizer(easyConfig(99), PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            var h = result.convergenceHistory();
            for (int i = 1; i < h.length; i++) {
                assertThat(h[i]).isLessThanOrEqualTo(h[i - 1]);
            }
        }

        @Test
        @DisplayName("Convergence history length = maxIterations + 1")
        void historyLength() {
            var cfg = easyConfig(1);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            assertThat(result.convergenceHistory()).hasSize(cfg.maxIterations() + 1);
        }

        @Test
        @DisplayName("Best position remains within search bounds")
        void bestPositionWithinBounds() {
            var optimizer = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere, null);
            var result = optimizer.optimize();
            for (double v : result.bestPosition()) {
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
            var optimizer1 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            var optimizer2 = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            var res1 = optimizer1.optimize();
            var res2 = optimizer2.optimize();
            assertAll(
                    () -> assertThat(res1.bestFitness()).isEqualTo(res2.bestFitness()),
                    () -> assertThat(res1.convergenceHistory()).containsExactly(res2.convergenceHistory())
            );
        }

        @Test
        @DisplayName("Different seeds → different convergence histories")
        void differentSeedsDiffer() {
            var res1 = new PsoOptimizer(easyConfig(1), PsoOptimizerTest::sphere, null).optimize();
            var res2 = new PsoOptimizer(easyConfig(2), PsoOptimizerTest::sphere, null).optimize();
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
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(0).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects inertia outside (0,1]")
        void invalidInertia() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.0).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(1.5).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects negative cognitive/social coefficients")
        void invalidCoefficients() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(-0.1).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(-1.0)
                    .velocityClampFactor(0.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects velocityClampFactor outside (0,1]")
        void invalidVelocityClamp() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.0).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(1.5).seed(0).build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Rejects knownOptimum with wrong length")
        void invalidKnownOptimum() {
            assertThatThrownBy(() -> PsoConfig.builder()
                    .populationSize(10).maxIterations(100).dimensions(2)
                    .minBounds(new double[]{0, 0}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(0)
                    .knownOptimum(new double[]{0})   // length 1 ≠ 2
                    .build())
                    .isInstanceOf(IllegalArgumentException.class);
        }

        @Test
        @DisplayName("Defensive copies of bounds arrays")
        void boundsDefensiveCopy() {
            var min = new double[]{0, 0};
            var max = new double[]{1, 1};
            var cfg = PsoConfig.builder()
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
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere, null);
            var res = optimizer.optimize();
            var p = res.bestPosition();
            p[0] = Double.MAX_VALUE;
            assertThat(res.bestPosition()[0]).isNotEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Result convergenceHistory is a defensive copy")
        void historyDefensiveCopy() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere, null);
            var res = optimizer.optimize();
            var h = res.convergenceHistory();
            h[0] = -999;
            assertThat(res.convergenceHistory()[0]).isNotEqualTo(-999);
        }

        @Test
        @DisplayName("getBestFitness matches Result.bestFitness after optimize()")
        void bestFitnessMatches() {
            var optimizer = new PsoOptimizer(easyConfig(42), PsoOptimizerTest::sphere, null);
            var res = optimizer.optimize();
            assertThat(optimizer.getBestFitness()).isEqualTo(res.bestFitness());
        }
    }

    @Nested
    @DisplayName("Swarm dynamics (velocity & personal best)")
    class Dynamics {
        @Test
        @DisplayName("Initial velocities respect vMax")
        void initialVelocityWithinVMax() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
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
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] vels = swarm.getVelocities();
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
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            Swarm swarm = optimizer.getSwarm();
            double[] initialPBest = swarm.getPersonalBestFitnesses();
            optimizer.optimize();
            double[] finalPBest = swarm.getPersonalBestFitnesses();
            for (int i = 0; i < initialPBest.length; i++) {
                assertThat(finalPBest[i]).isLessThanOrEqualTo(initialPBest[i]);
            }
        }

        @Test
        @DisplayName("Global best fitness is the minimum personal best fitness")
        void globalBestMatchesMinPersonalBest() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double minPBest = Arrays.stream(swarm.getPersonalBestFitnesses()).min().orElse(Double.NaN);
            assertThat(swarm.getGlobalBestFitness()).isEqualTo(minPBest);
        }

        @Test
        @DisplayName("All particles' positions stay within bounds")
        void positionsWithinBounds() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, PsoOptimizerTest::sphere, null);
            optimizer.optimize();
            Swarm swarm = optimizer.getSwarm();
            double[][] pos = swarm.getPositions();
            double lb = cfg.minBounds()[0];
            double ub = cfg.maxBounds()[0];
            for (double[] p : pos) {
                for (double pv : p) {
                    assertThat(pv).isBetween(lb, ub);
                }
            }
        }
    }

    @Nested
    @DisplayName("Invalid fitness handling")
    class InvalidFitness {
        static double partiallyInvalid(double[] x) {
            if (x[0] > 2.0) return Double.NaN;
            return x[0] * x[0] + x[1] * x[1];
        }

        @Test
        @DisplayName("Non‑finite fitness does not update personal/global best and particle fitness becomes ∞")
        void nonFiniteFitnessSkipped() {
            var cfg = easyConfig(42);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid, null);
            var result = optimizer.optimize();
            assertThat(result.bestFitness()).isLessThan(0.01);
            Swarm swarm = optimizer.getSwarm();
            double[] pBests = swarm.getPersonalBestFitnesses();
            assertThat(Arrays.stream(pBests)).allMatch(Double::isFinite);
        }

        @Test
        @DisplayName("Convergence history never contains NaN or Infinity")
        void historyNoNaN() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid, null);
            var result = optimizer.optimize();
            assertThat(Arrays.stream(result.convergenceHistory()))
                    .allMatch(Double::isFinite);
        }

        @Test
        @DisplayName("Global best is never updated from an invalid particle")
        void globalBestNeverInvalid() {
            var cfg = easyConfig(123);
            var optimizer = new PsoOptimizer(cfg, InvalidFitness::partiallyInvalid, null);
            optimizer.optimize();
            assertThat(optimizer.getBestFitness()).isFinite();
        }
    }

    // ─── new logging tests ────────────────────────────────────────────────
    @Nested
    @DisplayName("Execution log file")
    class Logging {

        @TempDir
        Path tempDir;

        private PsoConfig configWithKnownOptimum() {
            int dim = 2;
            var min = new double[dim];
            var max = new double[dim];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);
            double[] optimum = {0.0, 0.0};
            return PsoConfig.builder()
                    .populationSize(10)
                    .maxIterations(5)
                    .dimensions(dim)
                    .minBounds(min)
                    .maxBounds(max)
                    .inertia(0.7298)
                    .cognitiveCoefficient(1.49618)
                    .socialCoefficient(1.49618)
                    .velocityClampFactor(0.5)
                    .seed(42)
                    .knownOptimum(optimum)
                    .build();
        }

        @Test
        @DisplayName("Log file is created, contains header and correct number of lines")
        void logFileCreatedWithCorrectLineCount() throws IOException {
            Path logFile = tempDir.resolve("test_run.log");
            var config = configWithKnownOptimum();
            var optimizer = new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile);
            optimizer.optimize();

            assertThat(logFile).exists().isRegularFile();
            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            int expectedLineCount = 1 + (config.maxIterations() + 1) * config.populationSize();
            assertThat(lines).hasSize(expectedLineCount);
            assertThat(lines.getFirst()).isEqualTo(
                    "iteration;agentId;positionBefore;positionAfter;personalBest;personalBestFitness;globalBest;globalBestFitness;popAvgFitness;popStdDev;distToOptimum");
        }

        @Test
        @DisplayName("Log lines are correctly formatted and values are consistent")
        void logContentFormatAndConsistency() throws IOException {
            Path logFile = tempDir.resolve("format.log");
            var config = configWithKnownOptimum();
            new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile).optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            // first data line (iteration 0, agent 0)
            String[] parts = lines.get(1).split(";");
            assertThat(parts).hasSize(11);
            assertThat(Integer.parseInt(parts[0])).isEqualTo(0);
            assertThat(Integer.parseInt(parts[1])).isEqualTo(0);
            // positionBefore == positionAfter for initial snapshot
            assertThat(parts[2]).isEqualTo(parts[3]);
            double pBest = Double.parseDouble(parts[5]);
            double gBest = Double.parseDouble(parts[7]);
            assertThat(Double.isFinite(pBest)).isTrue();
            assertThat(Double.isFinite(gBest)).isTrue();
            assertThat(parts[10]).isNotEqualTo("NA");
            double dist = Double.parseDouble(parts[10]);
            assertThat(dist).isGreaterThanOrEqualTo(0);
        }

        @Test
        @DisplayName("Distance to optimum decreases over iterations for sphere function")
        void distanceDecreasesForSphere() throws IOException {
            Path logFile = tempDir.resolve("dist.log");
            var config = configWithKnownOptimum();
            new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile).optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            int pop = config.populationSize();
            double[] avgDistPerIter = new double[config.maxIterations() + 1];
            for (int i = 1; i < lines.size(); i++) {   // skip header
                String[] parts = lines.get(i).split(";");
                int iter = Integer.parseInt(parts[0]);
                double d = Double.parseDouble(parts[10]);
                avgDistPerIter[iter] += d;
            }
            for (int i = 0; i <= config.maxIterations(); i++) {
                avgDistPerIter[i] /= pop;
            }
            assertThat(avgDistPerIter[config.maxIterations()])
                    .isLessThan(avgDistPerIter[0]);
        }

        @Test
        @DisplayName("Distance field is 'NA' when knownOptimum is null")
        void distanceNAWhenOptimumNull() throws IOException {
            Path logFile = tempDir.resolve("na.log");
            var cfg = PsoConfig.builder()
                    .populationSize(5).maxIterations(2).dimensions(2)
                    .minBounds(new double[]{-1, -1}).maxBounds(new double[]{1, 1})
                    .inertia(0.7).cognitiveCoefficient(1.5).socialCoefficient(1.5)
                    .velocityClampFactor(0.5).seed(123)
                    .knownOptimum(null)
                    .build();
            new PsoOptimizer(cfg, PsoOptimizerTest::sphere, logFile).optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            for (int i = 1; i < lines.size(); i++) {
                String[] parts = lines.get(i).split(";");
                assertThat(parts[10]).isEqualTo("NA");
            }
        }

        @Test
        @DisplayName("Population statistics are correct (avg & std dev of finite fitnesses)")
        void populationStatsCorrect() throws IOException {
            Path logFile = tempDir.resolve("stats.log");
            var config = PsoConfig.builder()
                    .populationSize(3).maxIterations(2).dimensions(1)
                    .minBounds(new double[]{-5}).maxBounds(new double[]{5})
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(999)
                    .knownOptimum(new double[]{0})
                    .build();
            var optimizer = new PsoOptimizer(config, PsoOptimizerTest::sphere, logFile);
            Swarm swarmBefore = optimizer.getSwarm();
            double[][] initPositions = swarmBefore.getPositions();
            double[] fit = new double[3];
            for (int i = 0; i < 3; i++) {
                fit[i] = initPositions[i][0] * initPositions[i][0];
            }
            double expectedAvg = (fit[0] + fit[1] + fit[2]) / 3.0;
            double var = 0.0;
            for (double f : fit) var += (f - expectedAvg) * (f - expectedAvg);
            var /= 3.0;
            double expectedStd = Math.sqrt(var);

            optimizer.optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            String[] parts = lines.get(1).split(";");   // iteration 0, agent 0
            double loggedAvg = Double.parseDouble(parts[8]);
            double loggedStd = Double.parseDouble(parts[9]);
            assertThat(loggedAvg).isCloseTo(expectedAvg, offset(1e-12));
            assertThat(loggedStd).isCloseTo(expectedStd, offset(1e-12));
        }

        @Test
        @DisplayName("Invalid fitness particles are excluded from population statistics")
        void invalidFitnessExcludedFromStats() throws IOException {
            Path logFile = tempDir.resolve("invalid.log");
            var config = PsoConfig.builder()
                    .populationSize(4).maxIterations(1).dimensions(1)
                    .minBounds(new double[]{-10}).maxBounds(new double[]{10})
                    .inertia(0.5).cognitiveCoefficient(1.0).socialCoefficient(1.0)
                    .velocityClampFactor(0.5).seed(111)
                    .knownOptimum(new double[]{0})
                    .build();

            ObjectiveFunction tricky = x -> {
                if (x[0] > 5.0) return Double.NaN;
                return x[0] * x[0];
            };
            var optimizer = new PsoOptimizer(config, tricky, logFile);
            optimizer.optimize();

            List<String> lines = Files.readAllLines(logFile, StandardCharsets.UTF_8);
            // iteration 1 lines: indices 5..8 (header + 4 initial lines)
            double[] currentFits = new double[4];
            boolean[] valid = new boolean[4];
            for (int i = 0; i < 4; i++) {
                String[] parts = lines.get(5 + i).split(";");
                double pos = Double.parseDouble(parts[3].trim()); // 1D – positionAfter is a single number
                double fit = tricky.evaluate(new double[]{pos});
                currentFits[i] = fit;
                valid[i] = Double.isFinite(fit);
            }

            double sum = 0.0;
            int cnt = 0;
            for (int i = 0; i < 4; i++)
                if (valid[i]) {
                    sum += currentFits[i];
                    cnt++;
                }
            double expectedAvg = cnt > 0 ? sum / cnt : Double.NaN;
            double var = 0.0;
            for (int i = 0; i < 4; i++)
                if (valid[i]) var += (currentFits[i] - expectedAvg) * (currentFits[i] - expectedAvg);
            double expectedStd = cnt > 0 ? Math.sqrt(var / cnt) : Double.NaN;

            String[] firstLineIter1 = lines.get(5).split(";");
            double loggedAvg = Double.parseDouble(firstLineIter1[8]);
            double loggedStd = Double.parseDouble(firstLineIter1[9]);
            assertThat(loggedAvg).isCloseTo(expectedAvg, offset(1e-12));
            assertThat(loggedStd).isCloseTo(expectedStd, offset(1e-12));
        }
    }
}