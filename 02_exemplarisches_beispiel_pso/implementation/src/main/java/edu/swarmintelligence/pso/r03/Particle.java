package edu.swarmintelligence.pso.r03;

import lombok.Getter;
import lombok.Setter;

import java.util.random.RandomGenerator;

@Getter
@Setter
class Particle {
    private final double[] position;
    private final double[] velocity;
    private final double[] personalBestPosition;
    private double fitness;
    private double personalBestFitness;

    Particle(double[] position, double[] velocity, double fitness) {
        this.position = position.clone();
        this.velocity = velocity.clone();
        this.fitness = fitness;
        this.personalBestPosition = position.clone();
        this.personalBestFitness = fitness;
    }

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

    void updatePersonalBest() {
        System.arraycopy(position, 0, personalBestPosition, 0, position.length);
        personalBestFitness = fitness;
    }

    double getPersonalBestFitness() {
        return personalBestFitness;
    }

    double[] getPersonalBestPosition() {
        return personalBestPosition;
    }
}