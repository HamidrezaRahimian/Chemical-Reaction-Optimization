package edu.swarmintelligence.pso.r06;

/**
 * Represents a real‑valued objective function to be minimized.
 */
@FunctionalInterface
public interface ObjectiveFunction {
    /**
     * Evaluates the function at the given point.
     *
     * @param x the dimensional point
     * @return the function value
     */
    double evaluate(double[] x);
}