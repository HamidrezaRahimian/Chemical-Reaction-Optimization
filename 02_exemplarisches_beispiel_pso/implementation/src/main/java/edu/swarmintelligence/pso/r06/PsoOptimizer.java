package edu.swarmintelligence.pso.r06;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * PSO optimiser that runs the algorithm and optionally writes a detailed
 * per‑iteration log file (CSV) for later analysis.
 */
@Slf4j
public final class PsoOptimizer {
    private static final String RNG_ALGORITHM = "L128X1024MixRandom";

    private final PsoConfig config;
    private final Swarm swarm;
    /**
     * Path for the execution log; {@code null} suppresses file output.
     */
    private final Path logPath;

    /**
     * Creates an optimizer that writes the execution log to {@code algorithm_run.log}.
     *
     * @param config   validated configuration
     * @param function objective function to minimize
     */
    public PsoOptimizer(@NonNull PsoConfig config, @NonNull ObjectiveFunction function) {
        this(config, function, Path.of("algorithm_run.log"));
    }

    /**
     * Package‑private constructor for testing: if {@code logPath} is null, no file logging is performed.
     */
    PsoOptimizer(PsoConfig config, ObjectiveFunction function, Path logPath) {
        this.config = config;
        RandomGenerator random = RandomGeneratorFactory.of(RNG_ALGORITHM).create(config.seed());
        this.swarm = Swarm.createAndInitialize(config, function, random);
        this.logPath = logPath;
    }

    /**
     * Runs PSO for the configured number of iterations.
     *
     * @return an immutable result containing the best solution, fitness, and convergence history.
     */
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

    /**
     * Returns the best fitness found (after {@link #optimize()} has been called).
     *
     * @return global best fitness
     */
    public double getBestFitness() {
        return swarm.getGlobalBestFitness();
    }

    /**
     * Package‑private access to the swarm (for testing).
     */
    Swarm getSwarm() {
        return swarm;
    }
}