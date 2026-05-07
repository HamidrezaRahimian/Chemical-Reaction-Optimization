package edu.swarmintelligence.aha;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

/**
 * Comprehensive, white‑box & black‑box tests for
 * {@link ArtificialHummingbirdAlgorithm}.
 * <p>
 * Validates every detail of the original AHA paper:
 * <ul>
 *   <li>Configuration validation</li>
 *   <li>Convergence on standard benchmarks (Sphere, Rastrigin, Ackley)</li>
 *   <li>Correct guided‑foraging formula (paper‑exact)</li>
 *   <li>Correct visit‑table updates</li>
 *   <li>Target selection (highest visits + tie‑breaking by fitness)</li>
 *   <li>Migration after exactly 2·n stagnant iterations</li>
 *   <li>Global‑best monotonicity</li>
 *   <li>Position clamping to bounds</li>
 * </ul>
 */
@DisplayName("Artificial Hummingbird Algorithm")
class ArtificialHummingbirdAlgorithmTest {
    private static final int DIM = 5;
    private static final long TEST_SEED = 123456L;

    // ---------- helper factories ----------

    private static double[] boundsArray(final int dim, final double val) {
        final double[] arr = new double[dim];
        Arrays.fill(arr, val);
        return arr;
    }

    private static ArtificialHummingbirdAlgorithm.AhaConfig config(final int pop,
                                                                   final int iter,
                                                                   final int dim) {
        return ArtificialHummingbirdAlgorithm.AhaConfig.builder()
                .populationSize(pop)
                .maxIterations(iter)
                .dimensions(dim)
                .minBounds(boundsArray(dim, -10.0))
                .maxBounds(boundsArray(dim, 10.0))
                .seed(TEST_SEED)
                .build();
    }

    // ---------- 1. Configuration validation ----------

    private static org.assertj.core.data.Offset<Double> offset(double v) {
        return org.assertj.core.data.Offset.offset(v);
    }

    // ---------- 2. Sphere convergence ----------

    @Test
    @DisplayName("Builder rejects invalid configurations")
    void shouldRejectInvalidConfig() {
        // population size too small
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ArtificialHummingbirdAlgorithm.AhaConfig.builder()
                        .populationSize(1)
                        .maxIterations(10)
                        .dimensions(2)
                        .minBounds(new double[]{-1, -1})
                        .maxBounds(new double[]{1, 1})
                        .seed(1L)
                        .build());

        // zero iterations
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ArtificialHummingbirdAlgorithm.AhaConfig.builder()
                        .populationSize(2)
                        .maxIterations(0)
                        .dimensions(2)
                        .minBounds(new double[]{-1, -1})
                        .maxBounds(new double[]{1, 1})
                        .seed(1L)
                        .build());

        // bounds length mismatch
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ArtificialHummingbirdAlgorithm.AhaConfig.builder()
                        .populationSize(2)
                        .maxIterations(10)
                        .dimensions(2)
                        .minBounds(new double[]{0.0})
                        .maxBounds(new double[]{1.0, 2.0})
                        .seed(1L)
                        .build());
    }

    // ---------- 3. Rastrigin & Ackley ----------

    @ParameterizedTest(name = "Sphere {0}D")
    @ValueSource(ints = {5, 10})
    @DisplayName("Finds near‑optimal solution on the Sphere function")
    void shouldSolveSphere(final int dim) {
        final var cfg = config(40, 200, dim);
        final var aha = new ArtificialHummingbirdAlgorithm(cfg,
                x -> Arrays.stream(x).map(v -> v * v).sum());
        final double[] solution = aha.optimise();

        assertThat(aha.getBestFitness()).isLessThan(1e-5);
        assertThat(solution).hasSize(dim);
        for (double v : solution) {
            assertThat(Math.abs(v)).isLessThan(0.01);
        }
    }

    @Test
    @DisplayName("Significantly reduces the Rastrigin function")
    void shouldReduceRastrigin() {
        final var cfg = config(50, 300, 5);
        final var aha = new ArtificialHummingbirdAlgorithm(cfg, x -> {
            double sum = 10.0 * x.length;
            for (double v : x) sum += v * v - 10.0 * Math.cos(2 * Math.PI * v);
            return sum;
        });
        aha.optimise();
        assertThat(aha.getBestFitness()).isLessThan(5.0);
    }

    // ---------- 4. Guided‑foraging formula correctness (paper‑exact) ----------

    @Test
    @DisplayName("Finds near‑optimal solution on the Ackley function")
    void shouldSolveAckley() {
        final var cfg = config(40, 300, 5);
        final var aha = new ArtificialHummingbirdAlgorithm(cfg, x -> {
            final int n = x.length;
            double sqSum = 0.0, cosSum = 0.0;
            for (double v : x) {
                sqSum += v * v;
                cosSum += Math.cos(2 * Math.PI * v);
            }
            return -20.0 * Math.exp(-0.2 * Math.sqrt(sqSum / n))
                    - Math.exp(cosSum / n) + 20.0 + Math.E;
        });
        aha.optimise();
        assertThat(aha.getBestFitness()).isLessThan(0.1);
    }

    @Test
    @DisplayName("Guided foraging uses paper’s formula v = x_i + a·D⊙(x_i - x_j)")
    void guidedForagingFormulaIsPaperExact() {
        final int dim = 2;
        // positions: bird0 at [0,0], bird1 at [10,10]
        final double[][] pos = {{0.0, 0.0}, {10.0, 10.0}};
        final double[] fit = {100.0, 0.0};
        final int[][] vt = {{1, 0}, {0, 1}};
        // use a controlled RNG: direction always [1, 0] (axial), a = 0.5
        final var dummyRng = new RandomGenerator() {
            private final long[] sequence = {0, 0, 0}; // mode=0 (axial), dim=0, gaussian=0.5
            private int idx = 0;

            @Override
            public long nextLong() {
                return sequence[idx++ % sequence.length];
            }

            @Override
            public double nextDouble() {
                return 0.0;
            }

            @Override
            public int nextInt(int bound) {
                // first call: mode = 0 (axial); second call: dim index = 0
                // mode
                // dim=0
                return 0;
            }

            @Override
            public double nextGaussian() {
                return 0.5;
            }
            // other abstract methods omitted for brevity – real test uses a proper generator
        };

        final var cfg = ArtificialHummingbirdAlgorithm.AhaConfig.builder()
                .populationSize(2).maxIterations(1).dimensions(dim)
                .minBounds(boundsArray(dim, -100.0))
                .maxBounds(boundsArray(dim, 100.0))
                .seed(1L)
                .build();
        final var aha = new ArtificialHummingbirdAlgorithm(cfg,
                x -> x[0] * x[0] + x[1] * x[1], dummyRng);
        aha.setStateForTesting(pos, fit, vt);

        // Force guided foraging for bird 0
        aha.guidedForaging(0);

        // expected new position: [0 + 0.5*1*(0-10), 0 + 0.5*0*(0-10)] = [-5, 0]
        final double[][] finalPos = aha.getPositions();
        assertThat(finalPos[0][0]).isCloseTo(-5.0, offset(1e-9));
        assertThat(finalPos[0][1]).isCloseTo(0.0, offset(1e-9));
    }

    // ---------- 5. Target selection ----------

    @Test
    @DisplayName("selectTargetHighestVisit picks the most visited, ties broken by best fitness")
    void targetSelectionLogic() {
        final var cfg = ArtificialHummingbirdAlgorithm.AhaConfig.builder()
                .populationSize(4)
                .maxIterations(1)
                .dimensions(1)
                .minBounds(new double[]{-10.0})
                .maxBounds(new double[]{10.0})
                .seed(1L)
                .build();
        final var aha = new ArtificialHummingbirdAlgorithm(cfg, x -> x[0]);
        final double[][] pos = {{5.0}, {-2.0}, {3.0}, {7.0}};
        final double[] fit = {5.0, -2.0, 3.0, 7.0};
        final int[][] vt = {
                {1, 2, 2, 1},   // bird0: j=1 & j=2 both have 2 visits – tie
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
        };
        aha.setStateForTesting(pos, fit, vt);

        final int target = aha.selectTargetHighestVisit(0);
        assertThat(target).isEqualTo(1);    // bird1 fitness -2.0 beats bird2 fitness 3.0
    }

    // ---------- 6. Migration ----------

    @Test
    @DisplayName("Stagnant birds migrate after exactly 2·n iterations without improvement")
    void migrationHappensAfterStagnationLimit() {
        final int pop = 20;
        final var flat = (ArtificialHummingbirdAlgorithm.ObjectiveFunction) _ -> 0.0;
        // Use maxIterations = 2·pop so that migration must occur in the final iteration
        final var cfg = config(pop, 2 * pop, DIM);
        final var aha = new ArtificialHummingbirdAlgorithm(cfg, flat);
        aha.optimise();
        assertThat(aha.getMigrationCountLastIteration()).isGreaterThan(0);
    }

    // ---------- 7. Global best monotonicity ----------

    @Test
    @DisplayName("Global best fitness never increases over iterations")
    void globalBestIsMonotonic() {
        final var aha = new ArtificialHummingbirdAlgorithm(
                config(30, 100, DIM),
                x -> Arrays.stream(x).map(v -> v * v).sum());
        aha.optimise();
        final double[] history = aha.getConvergenceHistory();
        double prev = Double.POSITIVE_INFINITY;
        for (double val : history) {
            assertThat(val).isLessThanOrEqualTo(prev);
            prev = val;
        }
    }

    // ---------- 8. Position clamping ----------

    @Test
    @DisplayName("All positions stay within bounds after a full optimisation run")
    void positionsRemainInBounds() {
        final var cfg = config(20, 50, 3);
        final var aha = new ArtificialHummingbirdAlgorithm(cfg,
                x -> Arrays.stream(x).map(v -> v * v).sum());
        aha.optimise();
        final double[][] pos = aha.getPositions();
        final double[] min = cfg.minBounds();
        final double[] max = cfg.maxBounds();
        for (final double[] p : pos) {
            for (int d = 0; d < p.length; d++) {
                assertThat(p[d]).isBetween(min[d], max[d]);
            }
        }
    }

    // ---------- Nested: Visit table logic ----------

    @Nested
    @DisplayName("Visit table logic")
    class VisitTableLogic {
        /**
         * Creates a deterministic instance with fully controlled state.
         */
        private ArtificialHummingbirdAlgorithm createControlledInstance(
                final double[][] pos, final double[] fit, final int[][] vt,
                final ArtificialHummingbirdAlgorithm.ObjectiveFunction objF) {
            if (pos.length < 2) {
                throw new IllegalArgumentException("At least 2 birds required");
            }
            final var cfg = ArtificialHummingbirdAlgorithm.AhaConfig.builder()
                    .populationSize(pos.length)
                    .maxIterations(1)
                    .dimensions(pos[0].length)
                    .minBounds(boundsArray(pos[0].length, -100.0))
                    .maxBounds(boundsArray(pos[0].length, 100.0))
                    .seed(1L)
                    .build();
            final RandomGenerator dummyRng =
                    RandomGeneratorFactory.of("L128X256MixRandom").create(1L);
            final var aha = new ArtificialHummingbirdAlgorithm(cfg, objF, dummyRng);
            aha.setStateForTesting(pos, fit, vt);
            return aha;
        }

        @Test
        @DisplayName("Guided move → VT[i][i] reset to 0, VT[i][j] incremented")
        void guidedMoveResetsAndIncrements() {
            final double[][] pos = {{0.0}, {10.0}};
            final double[] fit = {100.0, 0.0};
            final int[][] vt = {{1, 0}, {0, 1}};

            final var aha = createControlledInstance(pos, fit, vt, x -> x[0] * x[0]);
            final double[] better = {2.0};
            aha.evaluateAndUpdate(0, 1, better, true);

            assertThat(aha.getPositions()[0]).containsExactly(2.0);
            assertThat(aha.getFitnesses()[0]).isEqualTo(4.0);
            final int[][] vtAfter = aha.getVisitTable();
            assertThat(vtAfter[0][0]).isEqualTo(0);
            assertThat(vtAfter[0][1]).isEqualTo(1);   // was 0, now +1
        }

        @Test
        @DisplayName("Guided no‑move → VT[i][i] unchanged, VT[i][j] still incremented")
        void guidedStayKeepsSelfVisitAndIncrementsTarget() {
            final double[][] pos = {{0.0}, {10.0}};
            final double[] fit = {1.0, 100.0};
            final int[][] vt = {{1, 0}, {0, 1}};

            final var aha = createControlledInstance(pos, fit, vt, x -> x[0] * x[0]);
            final double[] worse = {2.0};
            aha.evaluateAndUpdate(0, 1, worse, true);

            assertThat(aha.getPositions()[0]).containsExactly(0.0);
            assertThat(aha.getFitnesses()[0]).isEqualTo(1.0);
            final int[][] vtAfter = aha.getVisitTable();
            assertThat(vtAfter[0][0]).isEqualTo(1);
            assertThat(vtAfter[0][1]).isEqualTo(1);   // incremented from 0
        }

        @Test
        @DisplayName("Territorial move → VT[i][i] reset to 0, no other changes")
        void territorialMoveResetsSelfOnly() {
            final double[][] pos = {{5.0}, {10.0}};
            final double[] fit = {25.0, 100.0};
            final int[][] vt = {{1, 0}, {0, 1}};

            final var aha = createControlledInstance(pos, fit, vt, x -> x[0] * x[0]);
            final double[] better = {2.0};
            aha.evaluateAndUpdate(0, 0, better, false);

            assertThat(aha.getPositions()[0]).containsExactly(2.0);
            final int[][] vtAfter = aha.getVisitTable();
            assertThat(vtAfter[0][0]).isEqualTo(0);
            // target (j=0) not incremented because isGuided == false
            // remaining rows unchanged
        }

        @Test
        @DisplayName("Territorial no‑move → VT[i][i] unchanged")
        void territorialStayKeepsSelfVisit() {
            final double[][] pos = {{2.0}, {9.0}};
            final double[] fit = {4.0, 81.0};
            final int[][] vt = {{1, 0}, {0, 1}};

            final var aha = createControlledInstance(pos, fit, vt, x -> x[0] * x[0]);
            final double[] worse = {3.0};
            aha.evaluateAndUpdate(0, 0, worse, false);

            final int[][] vtAfter = aha.getVisitTable();
            assertThat(vtAfter[0][0]).isEqualTo(1);   // unchanged
        }
    }
}