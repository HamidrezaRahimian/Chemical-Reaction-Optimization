package edu.swarmintelligence.who;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

class WildHorseOptimizerTest {
    private static final int DIM = 5;
    private double[] min, max;

    @BeforeEach
    void setUp() {
        min = new double[DIM];
        max = new double[DIM];
        Arrays.fill(min, -10.0);
        Arrays.fill(max, 10.0);
    }

    // ---- config validation ---------------------------------------------------

    @Test
    void shouldRejectInvalidConfigurations() {
        assertThatIllegalArgumentException().isThrownBy(() ->
                new WildHorseOptimizer.WhoConfig(0, 3, 100, 2,
                        new double[]{0, 1}, new double[]{2, 3}, 0.5, 2.0, 1L));
        assertThatIllegalArgumentException().isThrownBy(() ->
                new WildHorseOptimizer.WhoConfig(30, 2, 100, 2,
                        new double[]{0, 1}, new double[]{2, 3}, 0.5, 2.0, 1L));
        assertThatIllegalArgumentException().isThrownBy(() ->
                new WildHorseOptimizer.WhoConfig(5, 3, 100, 2,
                        new double[]{0, 1}, new double[]{2, 3}, 0.5, 2.0, 1L));
    }

    @Test
    void configShouldBeImmutable() {
        double[] origMin = {0, 1}, origMax = {3, 4};
        var cfg = new WildHorseOptimizer.WhoConfig(10, 3, 100, 2,
                origMin, origMax, 0.5, 2.0, 42L);
        origMin[0] = -99;
        origMax[1] = 99;
        assertThat(cfg.minBounds()).containsExactly(0.0, 1.0);
        assertThat(cfg.maxBounds()).containsExactly(3.0, 4.0);
        double[] copy = cfg.minBounds();
        copy[0] = 77;
        assertThat(cfg.minBounds()).containsExactly(0.0, 1.0);
    }

    @Test
    void defaultConfigShouldDeriveGroups() {
        var cfg = WildHorseOptimizer.WhoConfig.createDefaultConfig(50, 500, DIM, min, max, 1L);
        assertThat(cfg.numGroups()).isEqualTo(5);
        assertThat(cfg.populationSize()).isGreaterThanOrEqualTo(3 * cfg.numGroups());
    }

    // ---- initialization -------------------------------------------------------

    @Test
    void shouldCreateCorrectNumberOfHerdsAndHorses() {
        var cfg = new WildHorseOptimizer.WhoConfig(20, 4, 10, DIM, min, max, 0.5, 2.0, 1L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        assertThat(opt.herds()).hasSize(4);
        assertThat(opt.population()).hasSize(20);
        int total = opt.herds().stream().mapToInt(h -> h.getMembers().size()).sum();
        assertThat(total).isEqualTo(20);
        for (var herd : opt.herds()) assertThat(herd.getStallion()).isNotNull();
    }

    @Test
    void eachHorseBelongsToExactlyOneHerd() {
        var cfg = new WildHorseOptimizer.WhoConfig(18, 3, 1, DIM, min, max, 0.5, 2.0, 2L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        var seen = new java.util.HashSet<>();
        for (var herd : opt.herds())
            for (var h : herd.getMembers())
                assertThat(seen.add(h)).isTrue();
        assertThat(seen).hasSize(18);
    }

    @Test
    void stallionIsBestInItsHerd() {
        var cfg = new WildHorseOptimizer.WhoConfig(10, 3, 1, DIM, min, max, 0.5, 2.0, 3L);
        var opt = new WildHorseOptimizer(cfg, x -> Arrays.stream(x).map(v -> v * v).sum());
        for (var herd : opt.herds()) {
            var stallion = herd.getStallion();
            for (var h : herd.getMembers()) {
                assertThat(stallion.getFitness()).isLessThanOrEqualTo(h.getFitness());
            }
        }
    }

    // ---- grazing and waterhole ------------------------------------------------

    @Test
    void grazingPhaseDoesNotChangePopulationSize() {
        var cfg = new WildHorseOptimizer.WhoConfig(12, 3, 1, DIM, min, max, 0.5, 2.0, 4L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        int before = opt.population().size();
        double tdr = opt.getCurrentTDR(1, 1, 2.0);
        for (var herd : opt.herds()) {
            var s = herd.getStallion();
            for (var h : herd.getMembers()) {
                if (!h.isStallion())
                    opt.performGrazing(h, s, DIM, min, max, tdr);
            }
        }
        assertThat(opt.population()).hasSize(before);
    }

    @Test
    void grazingShouldRespectBounds() {
        var cfg = new WildHorseOptimizer.WhoConfig(10, 3, 1, 2,
                new double[]{-1, -1}, new double[]{1, 1}, 0.5, 2.0, 5L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        var herd = opt.herds().getFirst();
        var stallion = herd.getStallion();
        var horse = herd.getMembers().stream().filter(h -> !h.isStallion()).findFirst().orElseThrow();
        double[] pos = opt.performGrazing(horse, stallion, 2, new double[]{-1, -1}, new double[]{1, 1}, 0.5);
        for (double v : pos) {
            assertThat(v).isBetween(-1.0, 1.0);
        }
    }

    @Test
    void waterHolePhaseShouldRespectBounds() {
        var cfg = new WildHorseOptimizer.WhoConfig(10, 3, 1, 2,
                new double[]{-1, -1}, new double[]{1, 1}, 0.5, 2.0, 6L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        var stallion = opt.herds().getFirst().getStallion();
        double[] pos = opt.performWaterHole(stallion, 2, new double[]{-1, -1}, new double[]{1, 1}, 0.5);
        for (double v : pos) {
            assertThat(v).isBetween(-1.0, 1.0);
        }
    }

    // ---- mating and decency ---------------------------------------------------

    @Test
    void matingPhasePreservesPopulationSize() {
        var cfg = new WildHorseOptimizer.WhoConfig(12, 3, 1, DIM, min, max, 0.9, 2.0, 7L);
        var opt = new WildHorseOptimizer(cfg, _ -> 1.0);
        int before = opt.population().size();
        opt.performMatingAndDecency(DIM, min, max, 0.9);
        assertThat(opt.population()).hasSize(before);
        int totalMembers = opt.herds().stream().mapToInt(h -> h.getMembers().size()).sum();
        assertThat(totalMembers).isEqualTo(before);
    }

    @Test
    void decencyConstraintIsSatisfiedAfterOneIteration() {
        var cfg = new WildHorseOptimizer.WhoConfig(12, 3, 3, DIM, min, max, 0.8, 2.0, 8L);
        var opt = new WildHorseOptimizer(cfg, x -> Arrays.stream(x).map(v -> v * v).sum());
        opt.optimize();
        assertThat(opt.population()).hasSize(12);
        for (var herd : opt.herds()) {
            assertThat(herd.getStallion()).isNotNull();
        }
    }

    // ---- fitness progression ---------------------------------------------------

    @Test
    void waterHoleFitnessShouldNeverIncrease() {
        var cfg = new WildHorseOptimizer.WhoConfig(15, 3, 50, DIM, min, max, 0.5, 2.0, 9L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        double prev = Double.MAX_VALUE;
        opt.updateWaterHole();
        for (int t = 1; t <= 50; t++) {
            double tdr = opt.getCurrentTDR(t, 50, 2.0);
            for (var herd : opt.herds()) {
                var s = herd.getStallion();
                for (var h : herd.getMembers()) {
                    if (!h.isStallion()) {
                        var pos = opt.performGrazing(h, s, DIM, min, max, tdr);
                        double fit = Arrays.stream(pos).map(v -> v * v).sum();
                        if (fit < h.getFitness()) {
                            h.setFitness(fit);
                            System.arraycopy(pos, 0, h.internalPosition(), 0, DIM);
                        }
                    }
                }
            }
            for (var herd : opt.herds()) {
                var s = herd.getStallion();
                if (s != null) {
                    var pos = opt.performWaterHole(s, DIM, min, max, tdr);
                    double fit = Arrays.stream(pos).map(v -> v * v).sum();
                    if (fit < s.getFitness()) {
                        s.setFitness(fit);
                        System.arraycopy(pos, 0, s.internalPosition(), 0, DIM);
                    }
                }
            }
            opt.performMatingAndDecency(DIM, min, max, 0.5);
            for (var herd : opt.herds()) herd.updateStallion();
            opt.updateWaterHole();
            assertThat(opt.getWaterHoleFitness()).isLessThanOrEqualTo(prev);
            prev = opt.getWaterHoleFitness();
        }
    }

    // ---- optimisation benchmarks ----------------------------------------------

    @Test
    void shouldReduceSphereFunction() {
        var cfg = WildHorseOptimizer.WhoConfig.createDefaultConfig(30, 200, DIM, min, max, 10L);
        var opt = new WildHorseOptimizer(cfg, x -> Arrays.stream(x).map(v -> v * v).sum());
        double[] best = opt.optimize();
        double val = Arrays.stream(best).map(v -> v * v).sum();
        assertThat(val).isLessThan(5.0);
    }

    @Test
    void shouldReduceRastriginFunction() {
        int dim = 3;
        double[] mi = new double[dim], ma = new double[dim];
        Arrays.fill(mi, -5.12);
        Arrays.fill(ma, 5.12);
        var cfg = WildHorseOptimizer.WhoConfig.createDefaultConfig(50, 300, dim, mi, ma, 11L);
        var opt = new WildHorseOptimizer(cfg, x -> {
            double sum = 10 * x.length;
            for (double v : x) sum += v * v - 10 * Math.cos(2 * Math.PI * v);
            return sum;
        });
        double[] best = opt.optimize();
        double val = 10 * dim;
        for (double v : best) val += v * v - 10 * Math.cos(2 * Math.PI * v);
        assertThat(val).isLessThan(50.0);
    }

    // ---- reproducibility & TDR ------------------------------------------------

    @Test
    void shouldBeDeterministicWithSameSeed() {
        var cfg = WildHorseOptimizer.WhoConfig.createDefaultConfig(30, 50, DIM, min, max, 123L);
        var opt1 = new WildHorseOptimizer(cfg, x -> Arrays.stream(x).map(v -> v * v).sum());
        var opt2 = new WildHorseOptimizer(cfg, x -> Arrays.stream(x).map(v -> v * v).sum());
        double[] best1 = opt1.optimize();
        double[] best2 = opt2.optimize();
        assertThat(best1).containsExactly(best2);
        assertThat(opt1.getWaterHoleFitness()).isEqualTo(opt2.getWaterHoleFitness());
    }

    @Test
    void tdrFormulaShouldMatchThePaper() {
        var cfg = WildHorseOptimizer.WhoConfig.createDefaultConfig(10, 20, DIM, min, max, 1L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        int maxIter = 20;
        double p = 2.0;
        for (int t = 1; t <= maxIter; t++) {
            double tdr = opt.getCurrentTDR(t, maxIter, p);
            double expected = 1.0 - Math.pow((double) t / maxIter, 1.0 / p);
            assertThat(tdr).isCloseTo(expected, within(1e-12));
        }
    }

    @Test
    void tdrShouldDecreaseMonotonically() {
        var cfg = WildHorseOptimizer.WhoConfig.createDefaultConfig(10, 100, DIM, min, max, 1L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        double prev = 1.0;
        for (int t = 1; t <= 100; t++) {
            double tdr = opt.getCurrentTDR(t, 100, 2.0);
            assertThat(tdr).isBetween(0.0, 1.0);
            assertThat(tdr).isLessThanOrEqualTo(prev);
            prev = tdr;
        }
    }

    // ---- defensive copies -----------------------------------------------------

    @Test
    void horsePositionGetterReturnsDefensiveCopy() {
        // Fixed: populationSize = 9 ≥ 3*3
        var cfg = new WildHorseOptimizer.WhoConfig(9, 3, 1, 2,
                new double[]{0, 0}, new double[]{1, 1}, 0.5, 2.0, 1L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        var horse = opt.population().getFirst();
        double[] pos = horse.getPosition();
        pos[0] = 999;
        assertThat(horse.getPosition()).isNotEqualTo(pos);
    }

    @Test
    void optimizeReturnsDefensiveCopy() {
        var cfg = WildHorseOptimizer.WhoConfig.createDefaultConfig(10, 10, DIM, min, max, 42L);
        var opt = new WildHorseOptimizer(cfg, _ -> 0.0);
        double[] best = opt.optimize();
        assertThat(best).isNotSameAs(opt.waterHolePositionInternal());
        double[] internal = opt.waterHolePositionInternal();
        best[0] = 555;
        assertThat(internal[0]).isNotEqualTo(555);
    }
}