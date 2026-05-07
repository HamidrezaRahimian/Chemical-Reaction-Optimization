package edu.swarmintelligence.pso.r03;

@FunctionalInterface
public interface ObjectiveFunction {
    double evaluate(double[] x);
}