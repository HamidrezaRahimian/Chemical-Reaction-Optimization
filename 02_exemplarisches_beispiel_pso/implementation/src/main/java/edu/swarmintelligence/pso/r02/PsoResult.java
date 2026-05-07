package edu.swarmintelligence.pso.r02;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoResult(
        double @NonNull [] bestPosition,
        double bestFitness,
        double @NonNull [] convergenceHistory,
        long seed
) {
    public PsoResult {
        Objects.requireNonNull(bestPosition);
        Objects.requireNonNull(convergenceHistory);
        bestPosition = bestPosition.clone();
        convergenceHistory = convergenceHistory.clone();
    }

    @Override
    public double[] bestPosition() {
        return bestPosition.clone();
    }

    @Override
    public double[] convergenceHistory() {
        return convergenceHistory.clone();
    }
}