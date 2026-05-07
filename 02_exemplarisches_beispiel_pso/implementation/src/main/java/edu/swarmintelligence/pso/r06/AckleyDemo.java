package edu.swarmintelligence.pso.r06;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

/**
 * Demonstrates PSO optimization on the 10‑dimensional Ackley function,
 * using the known global optimum at the origin.
 */
@Slf4j
public class AckleyDemo {
    static void main() {
        log.info("=== PSO Demo – Ackley function ===");

        ObjectiveFunction ackley = x -> {
            int n = x.length;
            double sumSq = 0.0, sumCos = 0.0;
            for (double v : x) {
                sumSq += v * v;
                sumCos += Math.cos(2.0 * Math.PI * v);
            }
            return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
                    - Math.exp(sumCos / n) + 20.0 + Math.E;
        };

        int d = 10;
        var min = new double[d];
        var max = new double[d];
        Arrays.fill(min, -32.768);
        Arrays.fill(max, 32.768);
        double[] knownOptimum = new double[d];   // optimum at origin

        var cfg = PsoConfig.builder()
                .populationSize(40)
                .maxIterations(2000)
                .dimensions(d)
                .minBounds(min)
                .maxBounds(max)
                .inertia(0.7298)
                .cognitiveCoefficient(1.49618)
                .socialCoefficient(1.49618)
                .velocityClampFactor(0.5)
                .seed(12345L)
                .knownOptimum(knownOptimum)
                .build();

        var optimizer = new PsoOptimizer(cfg, ackley);
        var result = optimizer.optimize();

        log.info("Best fitness = {}", result.bestFitness());
        log.info("Sample of best position = {}",
                Arrays.toString(Arrays.copyOf(result.bestPosition(), 3)));
    }
}