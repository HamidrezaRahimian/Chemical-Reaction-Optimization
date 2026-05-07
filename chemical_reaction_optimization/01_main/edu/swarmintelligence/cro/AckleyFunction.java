package edu.swarmintelligence.cro;

/**
 * Standard Ackley benchmark function used for the Aufgabe 4 logging scenario.
 * The global minimum is {@code f(0, ..., 0) = 0}; for the required 2D run the
 * configured search space is {@code [-32.768, 32.768]^2}.
 */
public final class AckleyFunction implements ChemicalReactionOptimization.ObjectiveFunction {
    public static final double LOWER_BOUND = -32.768;
    public static final double UPPER_BOUND = 32.768;

    private static final double A = 20.0;
    private static final double B = 0.2;
    private static final double C = 2.0 * Math.PI;

    /**
     * Evaluates the Ackley function with standard parameters
     * {@code a = 20}, {@code b = 0.2}, and {@code c = 2 * PI}.
     *
     * @param x candidate position, usually a 2D molecule structure for Aufgabe 4
     * @return Ackley function value; values closer to zero are better
     * @throws IllegalArgumentException if {@code x} has no dimensions
     */
    @Override
    public double evaluate(final double[] x) {
        int dimensions = x.length;
        if (dimensions == 0) {
            throw new IllegalArgumentException("Ackley function needs at least one dimension");
        }

        double sumSquared = 0.0;
        double sumCosine = 0.0;
        for (double value : x) {
            sumSquared += value * value;
            sumCosine += Math.cos(C * value);
        }

        return -A * Math.exp(-B * Math.sqrt(sumSquared / dimensions))
                - Math.exp(sumCosine / dimensions)
                + A
                + Math.E;
    }
}
