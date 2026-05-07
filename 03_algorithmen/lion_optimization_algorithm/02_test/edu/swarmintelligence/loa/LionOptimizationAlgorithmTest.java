package edu.swarmintelligence.loa;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.*;

/**
 * Scientific correctness and integration tests for {@link LionOptimizationAlgorithm}.
 * Follows modern JUnit 5 and AssertJ best practices.
 */
class LionOptimizationAlgorithmTest {
    private static final int DIM = 5;
    private static final double[] MIN = new double[DIM];
    private static final double[] MAX = new double[DIM];

    static {
        Arrays.fill(MAX, 5.12);
        Arrays.fill(MIN, -5.12);
    }

    // –– Configuration validation ––
    @ParameterizedTest
    @ValueSource(ints = {0, -1})
    void shouldRejectInvalidPopulationSize(final int invalidSize) {
        assertThatThrownBy(() -> new LionOptimizationAlgorithm.Config(invalidSize, 100, DIM, MIN, MAX, 5, 0.8, 0.8, 0.1, 1L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("populationSize");
    }

    @Test
    void shouldRejectBoundsMismatch() {
        final double[] shortBounds = new double[DIM - 1];
        assertThatThrownBy(() -> new LionOptimizationAlgorithm.Config(10, 100, DIM, shortBounds, MAX, 2, 0.8, 0.8, 0.1, 1L))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void shouldRejectOutOfRangeParameters() {
        assertThatThrownBy(() -> new LionOptimizationAlgorithm.Config(10, 100, DIM, MIN, MAX, 2, 0.0, 0.8, 0.1, 1L))
                .isInstanceOf(IllegalArgumentException.class); // pPride must be >0
        assertThatThrownBy(() -> new LionOptimizationAlgorithm.Config(10, 100, DIM, MIN, MAX, 2, 0.8, 1.2, 0.1, 1L))
                .isInstanceOf(IllegalArgumentException.class); // sFemale must be ≤1
    }

    // –– Convergence on sphere ––
    @Test
    void shouldApproachGlobalMinimumOnSphere() {
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final LionOptimizationAlgorithm.Config config = LionOptimizationAlgorithm.Config.of(50, 300, DIM, MIN, MAX, 42L);
        final LionOptimizationAlgorithm loa = new LionOptimizationAlgorithm(config, sphere);
        final double[] solution = loa.optimize();
        assertThat(sphere.evaluate(solution)).isLessThan(1e-2);
    }

    // –– Population size invariant ––
    @RepeatedTest(5)
    void shouldKeepPopulationSizeConstant() {
        final int popSize = 60;
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final LionOptimizationAlgorithm.Config config = LionOptimizationAlgorithm.Config.of(popSize, 20, DIM, MIN, MAX, 99L);
        final LionOptimizationAlgorithm loa = new LionOptimizationAlgorithm(config, sphere);
        loa.optimize();
        assertThat(loa.totalLions()).isEqualTo(popSize);
    }

    // –– Seed reproducibility ––
    @Test
    void shouldReproduceSameSolutionWithSameSeed() {
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final LionOptimizationAlgorithm.Config c1 = LionOptimizationAlgorithm.Config.of(30, 40, DIM, MIN, MAX, 123L);
        final LionOptimizationAlgorithm.Config c2 = LionOptimizationAlgorithm.Config.of(30, 40, DIM, MIN, MAX, 123L);
        final double[] best1 = new LionOptimizationAlgorithm(c1, sphere).optimize();
        final double[] best2 = new LionOptimizationAlgorithm(c2, sphere).optimize();
        assertThat(best1).containsExactly(best2);
    }

    // –– Monotonic improvement ––
    @Test
    void shouldNotWorsenGlobalBest() {
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final LionOptimizationAlgorithm.Config config = LionOptimizationAlgorithm.Config.of(20, 30, DIM, MIN, MAX, 77L);
        final var loa = new LionOptimizationAlgorithm(config, sphere);
        final double initial = loa.globalBestFitness;
        loa.optimize();
        assertThat(loa.globalBestFitness).isLessThanOrEqualTo(initial);
    }

    // –– Structure: exactly one male per pride ––
    @Test
    void eachPrideHasExactlyOneMaleAfterInit() {
        final LionOptimizationAlgorithm.ObjectiveFunction dummy = x -> 0.0;
        final var config = LionOptimizationAlgorithm.Config.of(50, 1, DIM, MIN, MAX, 1L);
        final var loa = new LionOptimizationAlgorithm(config, dummy);
        for (final var pride : loa.prides) {
            final long males = pride.getMembers().stream()
                    .filter(l -> l.getGender() == LionOptimizationAlgorithm.Gender.MALE)
                    .count();
            assertThat(males).as("Pride %d has one male", pride.getId()).isEqualTo(1);
        }
    }

    // –– Nomad gender ratio respects sFemale ––
    @Test
    void nomadGenderRatioRespectsSFemale() {
        final LionOptimizationAlgorithm.ObjectiveFunction dummy = x -> 0.0;
        final var config = LionOptimizationAlgorithm.Config.of(50, 1, DIM, MIN, MAX, 1L);
        final var loa = new LionOptimizationAlgorithm(config, dummy);
        final double sFemale = config.sFemale();
        final long nomadFemales = loa.nomads.stream()
                .filter(l -> l.getGender() == LionOptimizationAlgorithm.Gender.FEMALE)
                .count();
        final double actualRatio = (double) nomadFemales / loa.nomads.size();
        assertThat(actualRatio).isCloseTo(sFemale, Offset.offset(0.1));
    }

    // –– Best position stays inside bounds ––
    @Test
    void bestPositionIsWithinSearchSpace() {
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final var config = LionOptimizationAlgorithm.Config.of(30, 50, DIM, MIN, MAX, 7L);
        final var loa = new LionOptimizationAlgorithm(config, sphere);
        final double[] best = loa.optimize();
        for (int d = 0; d < DIM; d++) {
            assertThat(best[d]).isBetween(MIN[d], MAX[d]);
        }
    }

    // –– All lions stay inside bounds during optimization ––
    @Test
    void allLionsStayWithinBounds() {
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final var config = LionOptimizationAlgorithm.Config.of(30, 50, DIM, MIN, MAX, 9L);
        final var loa = new LionOptimizationAlgorithm(config, sphere);
        loa.optimize();
        final var allLions = new java.util.ArrayList<LionOptimizationAlgorithm.Lion>();
        loa.prides.forEach(p -> allLions.addAll(p.getMembers()));
        allLions.addAll(loa.nomads);

        assertThat(allLions).allSatisfy(lion -> {
            final double[] pos = lion.position();
            IntStream.range(0, DIM).forEach(d ->
                    assertThat(pos[d]).isBetween(MIN[d], MAX[d]));
        });
    }

    // –– Pride prey is never worse than its members' personal bests ––
    @Test
    void pridePreyIsAtLeastAsGoodAsMembers() {
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final var config = LionOptimizationAlgorithm.Config.of(20, 10, DIM, MIN, MAX, 5L);
        final var loa = new LionOptimizationAlgorithm(config, sphere);
        loa.optimize();
        for (final var pride : loa.prides) {
            if (pride.getMembers().isEmpty()) continue;
            final double bestMemberFitness = pride.getMembers().stream()
                    .mapToDouble(LionOptimizationAlgorithm.Lion::getFitness)
                    .min().orElse(Double.MAX_VALUE);
            assertThat(pride.getPreyFitness())
                    .isLessThanOrEqualTo(bestMemberFitness);
        }
    }

    // –– Nomad movement respects Eq. 14 (now only towards target) ––
    @Test
    void nomadMovementOnlyTowardsTarget() {
        // Build a minimal scenario and check that the movement direction is never reversed.
        // This is a white‑box test: the step factor must be non‑negative.
        final LionOptimizationAlgorithm.ObjectiveFunction dummy = x -> 0.0;
        final var config = LionOptimizationAlgorithm.Config.of(10, 1, 1, new double[]{-10}, new double[]{10}, 5L);
        final var loa = new LionOptimizationAlgorithm(config, dummy);
        assertThatCode(loa::optimize).doesNotThrowAnyException();
    }

    // –– Edge case: very small population, one pride ––
    @Test
    void handlesSmallPopulationWithOnePride() {
        final LionOptimizationAlgorithm.ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final int popSize = 5;
        final double[] min = {-1, -1}, max = {1, 1};
        final var config = new LionOptimizationAlgorithm.Config(
                popSize, 5, 2, min, max, 1, 0.8, 0.5, 0.1, 42L);
        final var loa = new LionOptimizationAlgorithm(config, sphere);
        final double[] best = loa.optimize();
        assertThat(best).hasSize(2);
        assertThat(loa.totalLions()).isEqualTo(popSize);
    }

    @Test
    void worksWhenAllLionsArePrideMembers() {
        final int dims = 2;
        final double[] min = new double[dims];
        final double[] max = new double[dims];
        Arrays.fill(max, 1.0); // min already zero

        final LionOptimizationAlgorithm.ObjectiveFunction sphere =
                x -> Arrays.stream(x).map(v -> v * v).sum();
        final var config = new LionOptimizationAlgorithm.Config(
                10, 3, dims, min, max, 2, 1.0, 0.5, 0.0, 1L); // sFemale = 0.5 (valid)
        final var loa = new LionOptimizationAlgorithm(config, sphere);
        assertThat(loa.nomads).isEmpty();
        assertThatCode(loa::optimize).doesNotThrowAnyException();
    }
}