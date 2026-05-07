package edu.swarmintelligence.pso.r02;

@FunctionalInterface
public interface ObjectiveFunction {
    double evaluate(double[] x);
}