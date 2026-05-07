package edu.swarmintelligence.pso.r04;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

@Slf4j
public final class PsoOptimizer {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";

    private final PsoConfig config;
    private final Swarm swarm;
    private final Path logPath;   // null → no file logging

    /**
     * Creates an optimizer that writes the execution log to {@code algorithm_run.log}.
     */
    public PsoOptimizer(@NonNull PsoConfig config, @NonNull ObjectiveFunction function) {
        this(config, function, Path.of("algorithm_run.log"));
    }

    /**
     * Package‑private constructor for testing: {@code logPath} may be null to suppress file output.
     */
    PsoOptimizer(PsoConfig config, ObjectiveFunction function, Path logPath) {
        this.config = config;
        RandomGenerator random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.swarm = Swarm.createAndInitialize(config, function, random);
        this.logPath = logPath;
    }

    public PsoResult optimize() {
        final double[] history = new double[config.maxIterations() + 1];
        history[0] = swarm.getGlobalBestFitness();

        PrintWriter logWriter = null;
        try {
            if (logPath != null) {
                logWriter = new PrintWriter(
                        new BufferedWriter(new OutputStreamWriter(
                                new FileOutputStream(logPath.toFile()), StandardCharsets.UTF_8)));
                logWriter.println("iteration;agentId;positionBefore;positionAfter;personalBest;personalBestFitness;globalBest;globalBestFitness;popAvgFitness;popStdDev;distToOptimum");
            }

            // iteration 0 (initial state)
            if (logWriter != null) {
                List<Swarm.IterationLogEntry> initEntries = swarm.initialLogEntries();
                for (Swarm.IterationLogEntry e : initEntries) {
                    logWriter.println(e.toCsvLine());
                }
                logWriter.flush();
            }

            // main loop
            for (int t = 1; t <= config.maxIterations(); t++) {
                List<Swarm.IterationLogEntry> iterLog = swarm.iterate(t);
                history[t] = swarm.getGlobalBestFitness();

                if (logWriter != null) {
                    for (Swarm.IterationLogEntry e : iterLog) {
                        logWriter.println(e.toCsvLine());
                    }
                    logWriter.flush();
                }

                if (t % 100 == 0 || t == config.maxIterations()) {
                    log.info("Iteration {}: best fitness = {}", t, swarm.getGlobalBestFitness());
                }
            }
        } catch (IOException e) {
            log.error("Failed to write log file", e);
        } finally {
            if (logWriter != null) {
                logWriter.close();
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