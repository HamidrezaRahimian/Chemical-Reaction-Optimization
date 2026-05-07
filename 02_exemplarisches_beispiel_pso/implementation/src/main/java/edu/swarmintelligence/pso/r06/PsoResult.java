package edu.swarmintelligence.pso.r06;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

/**
 * Immutable result of a complete PSO run.
 */
@Builder
public record PsoResult(
        /* The best position found during the run. */
        double @NonNull [] bestPosition,
        /* The best (lowest) fitness value achieved. */
        double bestFitness,
        /* Fitness of the global best at iteration 0, 1, …, maxIterations. */
        double @NonNull [] convergenceHistory,
        /* The random seed used for the run. */
        long seed
) {
    public PsoResult {
        Objects.requireNonNull(bestPosition);
        Objects.requireNonNull(convergenceHistory);
        bestPosition = bestPosition.clone();
        convergenceHistory = convergenceHistory.clone();
    }

    /**
     * Returns a defensive copy of the best position.
     */
    @Override
    public double[] bestPosition() {
        return bestPosition.clone();
    }

    /**
     * Returns a defensive copy of the convergence history.
     */
    @Override
    public double[] convergenceHistory() {
        return convergenceHistory.clone();
    }
}