package edu.swarmintelligence.cro;

public final class AckleyFunction implements ChemicalReactionOptimization.ObjectiveFunction {
    public static final double LOWER_BOUND = -32.768;
    public static final double UPPER_BOUND = 32.768;

    private static final double A = 20.0;
    private static final double B = 0.2;
    private static final double C = 2.0 * Math.PI;

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
