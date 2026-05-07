package edu.swarmintelligence.pso.r04;

@FunctionalInterface
public interface ObjectiveFunction {
    double evaluate(double[] x);
}