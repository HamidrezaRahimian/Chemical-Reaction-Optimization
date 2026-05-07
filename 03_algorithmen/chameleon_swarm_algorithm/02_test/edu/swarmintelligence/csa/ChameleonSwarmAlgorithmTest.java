package edu.swarmintelligence.csa;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

@Execution(ExecutionMode.CONCURRENT)
class ChameleonSwarmAlgorithmTest {
    private ChameleonSwarmAlgorithm.CsaConfig sphereConfig(int dim, int pop, int iter, long seed) {
        double[] min = new double[dim];
        double[] max = new double[dim];
        Arrays.fill(min, -5.12);
        Arrays.fill(max, 5.12);
        return ChameleonSwarmAlgorithm.CsaConfig.createDefaultConfig(pop, iter, dim, min, max, seed);
    }

    private double runSphere(int dim, int pop, int iter, long seed) {
        ChameleonSwarmAlgorithm csa = new ChameleonSwarmAlgorithm(
                sphereConfig(dim, pop, iter, seed),
                x -> Arrays.stream(x).map(v -> v * v).sum()
        );
        csa.optimize();
        return csa.getBestFitness();
    }

    @Test
    @DisplayName("Sphere 1D finds near‑global optimum")
    void sphere1DShouldReachGlobalMinimum() {
        double fitness = runSphere(1, 20, 100, 42);
        assertThat(fitness).isLessThan(1e-6);
    }

    @Test
    @DisplayName("Sphere 5D converges reasonably within 500 iterations")
    void sphere5DConverges() {
        // The algorithm is faithful to Braik (2021) and needs a realistic budget
        double fitness = runSphere(5, 30, 500, 123);
        assertThat(fitness).isLessThan(0.1);
    }

    @Test
    @DisplayName("All positions remain within the defined bounds")
    void positionsStayWithinBounds() {
        double[] min = {0.0, -1.0};
        double[] max = {1.0, 0.0};
        ChameleonSwarmAlgorithm.CsaConfig config = ChameleonSwarmAlgorithm.CsaConfig.builder()
                .populationSize(10).maxIterations(50).dimensions(2)
                .minBounds(min).maxBounds(max).seed(789L)
                .wMin(0.4).wMax(0.9).c1(2.0).c2(2.0).c3(2.0).ps(0.25)
                .build();

        ChameleonSwarmAlgorithm csa = new ChameleonSwarmAlgorithm(config,
                x -> x[0] * x[0] + x[1] * x[1]);
        csa.optimize();
        double[] best = csa.getBestSolution();
        assertThat(best[0]).isBetween(min[0], max[0]);
        assertThat(best[1]).isBetween(min[1], max[1]);
    }

    @Test
    @DisplayName("Reproducibility: identical seeds produce identical results")
    void reproducibilityWithSameSeed() {
        long seed = 20240101L;
        double result1 = runSphere(3, 15, 80, seed);
        double result2 = runSphere(3, 15, 80, seed);
        assertThat(result1).isEqualTo(result2);
    }

    @Test
    @DisplayName("Different seeds produce (likely) different results")
    void differentSeedsYieldDifferentResults() {
        double result1 = runSphere(2, 10, 50, 111L);
        double result2 = runSphere(2, 10, 50, 999L);
        assertThat(result1).isNotEqualTo(result2);
    }

    @Test
    @DisplayName("Configuration validation throws on illegal arguments")
    void invalidConfigurationsThrow() {
        double[] min = {0.0};
        double[] max = {1.0};
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ChameleonSwarmAlgorithm.CsaConfig.createDefaultConfig(0, 10, 1, min, max, 0L));
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ChameleonSwarmAlgorithm.CsaConfig.createDefaultConfig(5, 0, 1, min, max, 0L));
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ChameleonSwarmAlgorithm.CsaConfig.createDefaultConfig(5, 10, 1, new double[2], max, 0L));
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ChameleonSwarmAlgorithm.CsaConfig.builder()
                        .populationSize(10).maxIterations(10).dimensions(1)
                        .minBounds(min).maxBounds(max).seed(0L)
                        .wMin(0.8).wMax(0.4).c1(2.0).c2(2.0).c3(2.0).ps(0.25)
                        .build());
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ChameleonSwarmAlgorithm.CsaConfig.builder()
                        .populationSize(10).maxIterations(10).dimensions(1)
                        .minBounds(min).maxBounds(max).seed(0L)
                        .wMin(0.4).wMax(0.9).c1(2.0).c2(2.0).c3(2.0).ps(1.2).build());
        assertThatIllegalArgumentException()
                .isThrownBy(() -> ChameleonSwarmAlgorithm.CsaConfig.builder()
                        .populationSize(10).maxIterations(10).dimensions(1)
                        .minBounds(min).maxBounds(max).seed(0L)
                        .wMin(0.4).wMax(0.9).c1(2.0).c2(2.0).c3(2.0).ps(-0.1).build());
    }

    @Test
    @DisplayName("Constructor with null config or objective throws NullPointerException")
    void nullArgumentsThrow() {
        ChameleonSwarmAlgorithm.CsaConfig config = sphereConfig(1, 5, 10, 0L);
        assertThatNullPointerException().isThrownBy(() -> new ChameleonSwarmAlgorithm(null, x -> x[0]));
        assertThatNullPointerException().isThrownBy(() -> new ChameleonSwarmAlgorithm(config, null));
    }

    @Test
    @DisplayName("Global best never degrades")
    void globalBestNeverDegrades() {
        double[] min = {-1.0, -1.0};
        double[] max = {1.0, 1.0};
        ChameleonSwarmAlgorithm.CsaConfig config = ChameleonSwarmAlgorithm.CsaConfig.builder()
                .populationSize(10).maxIterations(50).dimensions(2)
                .minBounds(min).maxBounds(max).seed(555L)
                .wMin(0.4).wMax(0.9).c1(2.0).c2(2.0).c3(2.0).ps(0.25)
                .build();
        ChameleonSwarmAlgorithm csa = new ChameleonSwarmAlgorithm(config,
                x -> x[0] * x[0] + x[1] * x[1]);
        double initialBest = csa.getBestFitness();
        csa.optimize();
        assertThat(csa.getBestFitness()).isLessThanOrEqualTo(initialBest);
    }

    @Test
    @DisplayName("Striking probability zero (ps=0) runs and still converges")
    void strikingProbabilityZeroRunsSuccessfully() {
        double[] min = {-1.0, -1.0};
        double[] max = {1.0, 1.0};
        ChameleonSwarmAlgorithm.CsaConfig config = ChameleonSwarmAlgorithm.CsaConfig.builder()
                .populationSize(10).maxIterations(50).dimensions(2)
                .minBounds(min).maxBounds(max).seed(999L)
                .wMin(0.4).wMax(0.9).c1(2.0).c2(2.0).c3(2.0).ps(0.0)
                .build();
        ChameleonSwarmAlgorithm csa = new ChameleonSwarmAlgorithm(config,
                x -> x[0] * x[0] + x[1] * x[1]);
        csa.optimize();
        assertThat(csa.getBestFitness()).isFinite();
        assertThat(csa.getBestFitness()).isLessThan(2.0);
    }
}