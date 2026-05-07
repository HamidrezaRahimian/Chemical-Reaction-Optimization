package edu.swarmintelligence.pso.r04;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

@Slf4j
class Swarm {
    private final Particle[] particles;
    private final double[] vMax;
    private final PsoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;

    private double[] globalBestPosition;
    private double globalBestFitness;

    private Swarm(Particle[] particles,
                  double[] vMax,
                  PsoConfig config,
                  ObjectiveFunction function,
                  RandomGenerator random,
                  double[] globalBestPosition,
                  double globalBestFitness) {
        this.particles = particles;
        this.vMax = vMax;
        this.config = config;
        this.function = function;
        this.random = random;
        this.globalBestPosition = globalBestPosition.clone();
        this.globalBestFitness = globalBestFitness;
    }

    static Swarm createAndInitialize(PsoConfig config,
                                     ObjectiveFunction function,
                                     RandomGenerator random) {
        int popSize = config.populationSize();
        int dims = config.dimensions();
        double[] min = config.minBounds();
        double[] max = config.maxBounds();

        double[] vMax = new double[dims];
        for (int d = 0; d < dims; d++) {
            vMax[d] = config.velocityClampFactor() * (max[d] - min[d]);
        }

        Particle[] particles = new Particle[popSize];
        double[] bestPos = null;
        double bestFit = Double.POSITIVE_INFINITY;

        for (int i = 0; i < popSize; i++) {
            double[] x = new double[dims];
            double[] v = new double[dims];
            for (int d = 0; d < dims; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
                v[d] = (random.nextDouble() * 2.0 - 1.0) * vMax[d];
            }
            double fit = function.evaluate(x);
            if (!Double.isFinite(fit)) {
                log.warn("Particle {} initialised with invalid fitness → ∞", i);
                fit = Double.POSITIVE_INFINITY;
            }
            particles[i] = new Particle(x, v, fit);
            if (fit < bestFit) {
                bestFit = fit;
                bestPos = x.clone();
            }
        }

        log.debug("Swarm of {} particles started. Global best = {}", popSize, bestFit);
        assert bestPos != null;
        return new Swarm(particles, vMax, config, function, random, bestPos, bestFit);
    }

    private static String formatArray(double[] arr) {
        return Arrays.stream(arr)
                .mapToObj(v -> String.format(Locale.ROOT, "%.16g", v))
                .collect(Collectors.joining(" "));
    }

    /**
     * Performs one iteration and returns per‑particle log entries for that iteration.
     */
    List<IterationLogEntry> iterate(int iteration) {
        final double inertia = config.inertia();
        final double cognitiveCoeff = config.cognitiveCoefficient();
        final double socialCoeff = config.socialCoefficient();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        List<IterationLogEntry> entries = new ArrayList<>(particles.length);

        for (int i = 0; i < particles.length; i++) {
            Particle p = particles[i];

            // snapshot before movement
            double[] posBefore = p.getPosition().clone();

            // update velocity & position
            p.updateVelocityAndPosition(inertia, cognitiveCoeff, socialCoeff,
                    globalBestPosition, vMax, min, max, random);

            // evaluate new position
            double fit = function.evaluate(p.getPosition());
            if (!Double.isFinite(fit)) {
                p.setFitness(Double.POSITIVE_INFINITY);
            } else {
                p.setFitness(fit);
                if (fit < p.getPersonalBestFitness()) {
                    p.updatePersonalBest();
                }
            }

            // asynchronous global best update (based on personal best)
            if (p.getPersonalBestFitness() < globalBestFitness) {
                globalBestFitness = p.getPersonalBestFitness();
                globalBestPosition = p.getPersonalBestPosition().clone();
            }

            String positionBeforeStr = formatArray(posBefore);
            String positionAfterStr = formatArray(p.getPosition());
            String personalBestStr = formatArray(p.getPersonalBestPosition().clone());
            double personalBestFit = p.getPersonalBestFitness();
            String globalBestStr = formatArray(globalBestPosition.clone());
            double globalBestFit = globalBestFitness;

            entries.add(new IterationLogEntry(iteration, i,
                    positionBeforeStr, positionAfterStr,
                    personalBestStr, personalBestFit,
                    globalBestStr, globalBestFit,
                    null, null, // avg, std to be filled after loop
                    computeDistToOptimum(p.getPosition())));
        }

        // compute population statistics (only finite current fitnesses)
        double[] currentFitnesses = Arrays.stream(particles).mapToDouble(Particle::getFitness).toArray();
        int finiteCount = 0;
        double sum = 0.0;
        for (double f : currentFitnesses) {
            if (Double.isFinite(f)) {
                sum += f;
                finiteCount++;
            }
        }
        double avgFit = finiteCount > 0 ? sum / finiteCount : Double.NaN;
        double varSum = 0.0;
        if (finiteCount > 0) {
            for (double f : currentFitnesses) {
                if (Double.isFinite(f)) {
                    double diff = f - avgFit;
                    varSum += diff * diff;
                }
            }
        }
        double stdDevFit = finiteCount > 0 ? Math.sqrt(varSum / finiteCount) : Double.NaN;

        // update all entries with population stats
        for (IterationLogEntry e : entries) {
            e.popAvgFitness = avgFit;
            e.popStdDev = stdDevFit;
        }

        return entries;
    }

    /**
     * Log entries for the initial state (iteration 0, no movement).
     */
    List<IterationLogEntry> initialLogEntries() {
        List<IterationLogEntry> entries = new ArrayList<>(particles.length);
        double[] initialFitnesses = Arrays.stream(particles).mapToDouble(Particle::getFitness).toArray();

        int finCnt = 0;
        double sum = 0.0;
        for (double f : initialFitnesses) {
            if (Double.isFinite(f)) {
                sum += f;
                finCnt++;
            }
        }
        double avgFit = finCnt > 0 ? sum / finCnt : Double.NaN;
        double varSum = 0.0;
        if (finCnt > 0) {
            for (double f : initialFitnesses) {
                if (Double.isFinite(f)) {
                    double diff = f - avgFit;
                    varSum += diff * diff;
                }
            }
        }
        double stdDevFit = finCnt > 0 ? Math.sqrt(varSum / finCnt) : Double.NaN;

        for (int i = 0; i < particles.length; i++) {
            Particle p = particles[i];
            String pos = formatArray(p.getPosition());
            String pBest = formatArray(p.getPersonalBestPosition());
            entries.add(new IterationLogEntry(0, i,
                    pos, pos,   // before and after identical for initial snapshot
                    pBest, p.getPersonalBestFitness(),
                    formatArray(globalBestPosition), globalBestFitness,
                    avgFit, stdDevFit,
                    computeDistToOptimum(p.getPosition())));
        }
        return entries;
    }

    private String computeDistToOptimum(double[] position) {
        double[] optimum = config.knownOptimum();
        if (optimum == null) return "NA";
        double sumSq = 0.0;
        for (int i = 0; i < position.length; i++) {
            double diff = position[i] - optimum[i];
            sumSq += diff * diff;
        }
        return Double.toString(Math.sqrt(sumSq));
    }

    // ──────────────────── accessors (defensive copies) ────────────────────

    double[] getGlobalBestPosition() {
        return globalBestPosition.clone();
    }

    double getGlobalBestFitness() {
        return globalBestFitness;
    }

    double[][] getPositions() {
        return Arrays.stream(particles).map(p -> p.getPosition().clone()).toArray(double[][]::new);
    }

    double[][] getVelocities() {
        return Arrays.stream(particles).map(p -> p.getVelocity().clone()).toArray(double[][]::new);
    }

    double[][] getPersonalBests() {
        return Arrays.stream(particles).map(p -> p.getPersonalBestPosition().clone()).toArray(double[][]::new);
    }

    double[] getPersonalBestFitnesses() {
        return Arrays.stream(particles).mapToDouble(Particle::getPersonalBestFitness).toArray();
    }

    // ──────────────────── log entry structure (package‑private) ──────────
    static class IterationLogEntry {
        final int iteration;
        final int agentId;
        final String positionBefore;
        final String positionAfter;
        final String personalBest;
        final double personalBestFitness;
        final String globalBest;
        final double globalBestFitness;
        Double popAvgFitness;   // filled after iteration loop
        Double popStdDev;
        String distToOptimum;

        IterationLogEntry(int iteration, int agentId,
                          String positionBefore, String positionAfter,
                          String personalBest, double personalBestFitness,
                          String globalBest, double globalBestFitness,
                          Double popAvgFitness, Double popStdDev,
                          String distToOptimum) {
            this.iteration = iteration;
            this.agentId = agentId;
            this.positionBefore = positionBefore;
            this.positionAfter = positionAfter;
            this.personalBest = personalBest;
            this.personalBestFitness = personalBestFitness;
            this.globalBest = globalBest;
            this.globalBestFitness = globalBestFitness;
            this.popAvgFitness = popAvgFitness;
            this.popStdDev = popStdDev;
            this.distToOptimum = distToOptimum;
        }

        String toCsvLine() {
            return String.format(Locale.ROOT,
                    "%d;%d;%s;%s;%s;%.16g;%s;%.16g;%.16g;%.16g;%s",
                    iteration, agentId,
                    positionBefore, positionAfter,
                    personalBest, personalBestFitness,
                    globalBest, globalBestFitness,
                    popAvgFitness != null ? popAvgFitness : Double.NaN,
                    popStdDev != null ? popStdDev : Double.NaN,
                    distToOptimum);
        }
    }
}