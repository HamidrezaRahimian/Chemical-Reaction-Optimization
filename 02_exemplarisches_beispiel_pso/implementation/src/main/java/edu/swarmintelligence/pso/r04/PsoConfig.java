package edu.swarmintelligence.pso.r04;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

@Builder
public record PsoConfig(
        int populationSize,
        int maxIterations,
        int dimensions,
        double @NonNull [] minBounds,
        double @NonNull [] maxBounds,
        double inertia,
        double cognitiveCoefficient,
        double socialCoefficient,
        double velocityClampFactor,
        long seed,
        double[] knownOptimum                    // optional – null → distToOptimum = "NA"
) {
    public PsoConfig {
        Objects.requireNonNull(minBounds);
        Objects.requireNonNull(maxBounds);
        if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0");
        if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
        if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
        if (minBounds.length != dimensions || maxBounds.length != dimensions)
            throw new IllegalArgumentException("Bounds length mismatch");
        if (inertia <= 0 || inertia > 1) throw new IllegalArgumentException("inertia ∈ (0,1]");
        if (cognitiveCoefficient < 0) throw new IllegalArgumentException("c1 ≥ 0");
        if (socialCoefficient < 0) throw new IllegalArgumentException("c2 ≥ 0");
        if (velocityClampFactor <= 0 || velocityClampFactor > 1)
            throw new IllegalArgumentException("velocityClampFactor ∈ (0,1]");

        // defensive copies for mutable arrays
        minBounds = minBounds.clone();
        maxBounds = maxBounds.clone();

        // optional knownOptimum – validate length and defensive copy
        if (knownOptimum != null) {
            if (knownOptimum.length != dimensions)
                throw new IllegalArgumentException("knownOptimum length must equal dimensions");
            knownOptimum = knownOptimum.clone();   // re‑assign parameter, not field
        }
    }
}