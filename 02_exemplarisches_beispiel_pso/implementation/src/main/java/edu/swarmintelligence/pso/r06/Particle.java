package edu.swarmintelligence.pso.r06;

import lombok.Getter;
import lombok.Setter;

import java.util.random.RandomGenerator;

/**
 * Mutable particle in the swarm, holding its current position, velocity,
 * fitness, and personal best.
 */
@Getter
@Setter
class Particle {
    /**
     * Current position in the search space.
     */
    private final double[] position;
    /**
     * Current velocity (per‑dimension).
     */
    private final double[] velocity;
    /**
     * Best position visited by this particle so far.
     */
    private final double[] personalBestPosition;
    /**
     * Fitness (objective value) at the current position.
     */
    private double fitness;
    /**
     * Best fitness achieved so far.
     */
    private double personalBestFitness;

    /**
     * Creates a new particle.
     *
     * @param position initial position
     * @param velocity initial velocity
     * @param fitness  initial fitness
     */
    Particle(double[] position, double[] velocity, double fitness) {
        this.position = position.clone();
        this.velocity = velocity.clone();
        this.fitness = fitness;
        this.personalBestPosition = position.clone();
        this.personalBestFitness = fitness;
    }

    /**
     * Updates velocity and position according to the standard PSO equations,
     * then clamps both velocity and position to the allowed ranges.
     *
     * @param inertia        inertia weight w
     * @param cognitiveCoeff cognitive acceleration coefficient c1
     * @param socialCoeff    social acceleration coefficient c2
     * @param globalBest     current global best position
     * @param vMax           per‑dimension maximum velocity magnitude
     * @param lb             lower bound per dimension
     * @param ub             upper bound per dimension
     * @param random         random generator for the stochastic components
     */
    void updateVelocityAndPosition(double inertia,
                                   double cognitiveCoeff,
                                   double socialCoeff,
                                   double[] globalBest,
                                   double[] vMax,
                                   double[] lb,
                                   double[] ub,
                                   RandomGenerator random) {
        for (int d = 0; d < position.length; d++) {
            double r1 = random.nextDouble();
            double r2 = random.nextDouble();
            double cognitive = cognitiveCoeff * r1 * (personalBestPosition[d] - position[d]);
            double social = socialCoeff * r2 * (globalBest[d] - position[d]);
            double newV = inertia * velocity[d] + cognitive + social;
            newV = Math.clamp(newV, -vMax[d], vMax[d]);
            velocity[d] = newV;
            position[d] = Math.clamp(position[d] + newV, lb[d], ub[d]);
        }
    }

    /**
     * Updates the personal best position and fitness to the current state.
     */
    void updatePersonalBest() {
        System.arraycopy(position, 0, personalBestPosition, 0, position.length);
        personalBestFitness = fitness;
    }

    /**
     * Returns the personal best fitness.
     *
     * @return personal best fitness
     */
    double getPersonalBestFitness() {
        return personalBestFitness;
    }

    /**
     * Returns the personal best position (reference to internal array –
     * defensive copying is performed by callers).
     *
     * @return personal best position
     */
    double[] getPersonalBestPosition() {
        return personalBestPosition;
    }
}