package edu.swarmintelligence.pso.r06;

import lombok.Builder;
import lombok.NonNull;

import java.util.Objects;

/**
 * Immutable configuration for a PSO run.
 * <p>All arrays are defensively copied; the optional {@code knownOptimum}
 * is validated for length if provided.</p>
 */
@Builder
public record PsoConfig(
        /* Number of particles in the swarm. */
        int populationSize,
        /* Maximum iterations of the optimization loop. */
        int maxIterations,
        /* Dimensionality of the search space. */
        int dimensions,
        /* Lower bounds per dimension. */
        double @NonNull [] minBounds,
        /* Upper bounds per dimension. */
        double @NonNull [] maxBounds,
        /* Inertia weight w, in (0,1]. */
        double inertia,
        /* Cognitive acceleration coefficient c1, c1 ≥ 0. */
        double cognitiveCoefficient,
        /* Social acceleration coefficient c2, c2 ≥ 0. */
        double socialCoefficient,
        /* Velocity clamping factor k, in (0,1]. */
        double velocityClampFactor,
        /* Random generator seed for reproducibility. */
        long seed,
        /* (Optional) known optimum position; if null distance tracking is disabled. */
        double[] knownOptimum
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

        if (knownOptimum != null) {
            if (knownOptimum.length != dimensions)
                throw new IllegalArgumentException("knownOptimum length must equal dimensions");
            knownOptimum = knownOptimum.clone();
        }
    }
}