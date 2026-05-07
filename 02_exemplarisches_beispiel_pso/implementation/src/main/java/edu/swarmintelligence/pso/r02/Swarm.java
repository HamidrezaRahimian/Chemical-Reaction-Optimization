package edu.swarmintelligence.pso.r02;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.random.RandomGenerator;

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
        this.globalBestPosition = globalBestPosition.clone(); // defensive
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
        return new Swarm(particles, vMax, config, function, random, bestPos, bestFit);
    }

    void iterate() {
        final double inertia = config.inertia();
        final double cognitiveCoeff = config.cognitiveCoefficient();
        final double socialCoeff = config.socialCoefficient();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        for (Particle p : particles) {
            p.updateVelocityAndPosition(inertia, cognitiveCoeff, socialCoeff,
                    globalBestPosition, vMax, min, max, random);

            double fit = function.evaluate(p.getPosition());
            if (!Double.isFinite(fit)) {
                p.setFitness(Double.POSITIVE_INFINITY);
                continue;  // explicitly skip personal/global best update
            }
            p.setFitness(fit);

            if (fit < p.getPersonalBestFitness()) {
                p.updatePersonalBest();
            }
            if (p.getPersonalBestFitness() < globalBestFitness) {
                globalBestFitness = p.getPersonalBestFitness();
                globalBestPosition = p.getPersonalBestPosition().clone();
            }
        }
    }

    double[] getGlobalBestPosition() {
        return globalBestPosition.clone();  // defensive copy
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
}