package edu.swarmintelligence.pso.r03;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

@Slf4j
public final class PsoOptimizer {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";

    private final PsoConfig config;
    private final Swarm swarm;

    public PsoOptimizer(@NonNull PsoConfig config, @NonNull ObjectiveFunction function) {
        this.config = config;
        RandomGenerator random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.swarm = Swarm.createAndInitialize(config, function, random);
    }

    public PsoResult optimize() {
        final double[] history = new double[config.maxIterations() + 1];
        history[0] = swarm.getGlobalBestFitness();

        for (int t = 1; t <= config.maxIterations(); t++) {
            swarm.iterate();
            history[t] = swarm.getGlobalBestFitness();
            if (t % 100 == 0 || t == config.maxIterations()) {
                log.info("Iteration {}: best fitness = {}", t, swarm.getGlobalBestFitness());
            }
        }

        log.info("Optimisation finished. Best fitness = {}", swarm.getGlobalBestFitness());
        return PsoResult.builder()
                .bestPosition(swarm.getGlobalBestPosition())
                .bestFitness(swarm.getGlobalBestFitness())
                .convergenceHistory(history)
                .seed(config.seed())
                .build();
    }

    public double getBestFitness() {
        return swarm.getGlobalBestFitness();
    }

    Swarm getSwarm() {
        return swarm;
    }
}