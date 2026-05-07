package edu.swarmintelligence.pso.r05;

import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * A powerful data analytics engine that processes the PSO execution log
 * ({@code algorithm_run.log}) and generates a professional evaluation report
 * focused on algorithm quality metrics.
 *
 * <p>The report includes:
 * <ul>
 *   <li>Run overview (inferred configuration)</li>
 *   <li>Final solution quality</li>
 *   <li>Convergence speed &amp; stagnation analysis</li>
 *   <li>Swarm diversity (average pairwise distance)</li>
 *   <li>Exploration vs. exploitation balance (velocity magnitude)</li>
 *   <li>Global best update frequency</li>
 *   <li>Distance to optimum (if known)</li>
 *   <li>Population fitness statistics</li>
 *   <li>ASCII convergence &amp; diversity charts</li>
 * </ul>
 *
 * <p>All output uses only standard ASCII characters for maximum compatibility.
 */
@Slf4j
public final class PsoAnalytics {
    // ---------------------------------------------------------------------
    // Parsing helpers
    // ---------------------------------------------------------------------
    private static double[] parseDoubleArray(String s) {
        if (s == null || s.isBlank()) return new double[0];
        String[] parts = s.trim().split("\\s+");
        double[] arr = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            arr[i] = Double.parseDouble(parts[i]);
        }
        return arr;
    }

    private static List<LogEntry> parseLog(Path logPath) throws IOException {
        List<LogEntry> entries = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(logPath, StandardCharsets.UTF_8)) {
            String header = reader.readLine(); // discard header line
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] cols = line.split(";", -1);
                if (cols.length < 11) {
                    log.warn("Skipping malformed line ({} columns): {}", cols.length, line);
                    continue;
                }
                int iteration = Integer.parseInt(cols[0]);
                int agentId = Integer.parseInt(cols[1]);
                double[] posBefore = parseDoubleArray(cols[2]);
                double[] posAfter = parseDoubleArray(cols[3]);
                double[] pbest = parseDoubleArray(cols[4]);
                double pbestFit = Double.parseDouble(cols[5]);
                double[] gbest = parseDoubleArray(cols[6]);
                double gbestFit = Double.parseDouble(cols[7]);
                double popAvg = Double.parseDouble(cols[8]);
                double popStd = Double.parseDouble(cols[9]);
                String distRaw = cols[10];
                entries.add(new LogEntry(iteration, agentId,
                        posBefore, posAfter, pbest, pbestFit,
                        gbest, gbestFit, popAvg, popStd, distRaw));
            }
        }
        return entries;
    }

    private static RunMetrics computeMetrics(List<LogEntry> logEntries) {
        RunMetrics m = new RunMetrics();

        // Group by iteration (TreeMap ensures natural ordering)
        Map<Integer, List<LogEntry>> byIter = logEntries.stream()
                .collect(Collectors.groupingBy(l -> l.iteration, TreeMap::new, Collectors.toList()));

        // --- Basic configuration from the log ---
        List<LogEntry> iterZero = byIter.get(0);
        if (iterZero == null || iterZero.isEmpty()) {
            throw new IllegalStateException("Log does not contain iteration 0 (initial state).");
        }
        m.populationSize = iterZero.size();
        m.dimensions = iterZero.getFirst().positionBefore.length;
        m.maxIterations = byIter.keySet().stream().mapToInt(Integer::intValue).max().orElse(0);
        m.optimumKnown = !"NA".equals(iterZero.getFirst().distToOptimumRaw);

        Double lastGBest = null;
        for (int iter = 0; iter <= m.maxIterations; iter++) {
            List<LogEntry> lines = byIter.get(iter);
            if (lines == null || lines.isEmpty()) {
                log.warn("Missing entries for iteration {} – using last known values", iter);
                if (lastGBest != null) {
                    m.globalBestHistory.add(lastGBest);
                    m.popAvgHistory.add(m.popAvgHistory.isEmpty() ? Double.NaN : m.popAvgHistory.getLast());
                    m.popStdHistory.add(m.popStdHistory.isEmpty() ? Double.NaN : m.popStdHistory.getLast());
                    if (m.optimumKnown) {
                        m.distanceToOptimumHistory.add(m.distanceToOptimumHistory.isEmpty() ? Double.NaN : m.distanceToOptimumHistory.getLast());
                    }
                }
                continue;
            }
            LogEntry first = lines.getFirst();
            double gBest = first.globalBestFitness;
            m.globalBestHistory.add(gBest);
            lastGBest = gBest;
            m.popAvgHistory.add(first.popAvgFitness);
            m.popStdHistory.add(first.popStdDev);
            if (m.optimumKnown && !"NA".equals(first.distToOptimumRaw)) {
                m.distanceToOptimumHistory.add(Double.parseDouble(first.distToOptimumRaw));
            }
        }
        m.finalBestFitness = m.globalBestHistory.getLast();

        // Global best updates
        m.globalBestUpdates = 0;
        for (int i = 1; i < m.globalBestHistory.size(); i++) {
            if (m.globalBestHistory.get(i) < m.globalBestHistory.get(i - 1) - 1e-15) {
                m.globalBestUpdates++;
            }
        }

        // Convergence rate (geometric mean)
        double initialFitness = m.globalBestHistory.getFirst();
        if (initialFitness > m.finalBestFitness && m.finalBestFitness > 0) {
            m.convergenceRate = Math.pow(m.finalBestFitness / initialFitness, 1.0 / m.maxIterations);
        } else {
            m.convergenceRate = 1.0;
        }

        // Iterations to reach 1e-6 of initial fitness
        double threshold = initialFitness * 1e-6;
        for (int i = 0; i < m.globalBestHistory.size(); i++) {
            if (m.globalBestHistory.get(i) <= threshold) {
                m.iterationsToThreshold = i;
                break;
            }
        }

        // Stagnation detection (no improvement in the last 10% of iterations)
        int lookback = Math.max(1, (int) (m.maxIterations * 0.1));
        int lastImprovement = m.maxIterations;
        for (int i = m.maxIterations; i > m.maxIterations - lookback; i--) {
            if (i > 0 && m.globalBestHistory.get(i) < m.globalBestHistory.get(i - 1) - 1e-15) {
                lastImprovement = i;
                break;
            }
        }
        if (lastImprovement < m.maxIterations - lookback + 1) {
            m.stagnationStart = lastImprovement;
        }

        // Diversity & velocity magnitude per iteration
        for (int iter = 0; iter <= m.maxIterations; iter++) {
            List<LogEntry> lines = byIter.get(iter);
            if (lines == null || lines.isEmpty()) {
                m.avgDistanceHistory.add(m.avgDistanceHistory.isEmpty() ? 0.0 : m.avgDistanceHistory.getLast());
                m.avgVelocityMagHistory.add(m.avgVelocityMagHistory.isEmpty() ? 0.0 : m.avgVelocityMagHistory.getLast());
                continue;
            }
            List<double[]> positions = lines.stream().map(l -> l.positionAfter).toList();

            // Average pairwise distance
            double sumDist = 0.0;
            int pairs = 0;
            for (int i = 0; i < positions.size(); i++) {
                for (int j = i + 1; j < positions.size(); j++) {
                    sumDist += euclidean(positions.get(i), positions.get(j));
                    pairs++;
                }
            }
            m.avgDistanceHistory.add(pairs > 0 ? sumDist / pairs : 0.0);

            // Average velocity magnitude
            double sumVel = 0.0;
            for (LogEntry e : lines) {
                sumVel += euclidean(e.positionBefore, e.positionAfter);
            }
            m.avgVelocityMagHistory.add(sumVel / lines.size());
        }

        if (m.optimumKnown && !m.distanceToOptimumHistory.isEmpty()) {
            m.finalDistanceToOptimum = m.distanceToOptimumHistory.getLast();
        }

        return m;
    }

    private static double euclidean(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    // ---------------------------------------------------------------------
    // ASCII chart utility (only standard ASCII characters)
    // ---------------------------------------------------------------------
    private static void printAsciiChart(List<Double> values, int width, int height, String label) {
        if (values.isEmpty()) return;
        double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(1);
        if (max == min) max = min + 1; // avoid division by zero

        int steps = Math.min(values.size(), width);
        double[] display = new double[steps];
        for (int i = 0; i < steps; i++) {
            int idx = (int) ((long) i * values.size() / steps);
            display[i] = values.get(idx);
        }

        for (int row = height - 1; row >= 0; row--) {
            double threshold = min + (max - min) * row / (height - 1);
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("%10.4f |", threshold));
            for (double v : display) {
                if (v >= threshold) {
                    sb.append('#');   // ASCII block
                } else {
                    sb.append(' ');
                }
            }
            System.out.println(sb);
        }
        // x-axis
        System.out.println("           " + "-".repeat(steps));
        System.out.printf("           Iter: 0%" + (steps - 1) + "d\n", values.size() - 1);
        if (!label.isEmpty()) {
            System.out.println("           " + label);
        }
    }

    // ---------------------------------------------------------------------
    // Scientific formatting
    // ---------------------------------------------------------------------
    private static String scientific(double v) {
        if (Double.isNaN(v)) return "NaN";
        if (Double.isInfinite(v)) return v > 0 ? "Infinity" : "-Infinity";
        return String.format(Locale.US, "%.6e", v);
    }

    // ---------------------------------------------------------------------
    // Report generation
    // ---------------------------------------------------------------------
    public static void generateReport(Path logPath) {
        System.out.println("=".repeat(100));
        System.out.println("   PSO ALGORITHM QUALITY EVALUATION REPORT");
        System.out.println("=".repeat(100));
        System.out.println("Log file    : " + logPath.toAbsolutePath());
        System.out.println("Generated   : " + new Date());
        System.out.println();

        try {
            List<LogEntry> logEntries = parseLog(logPath);
            if (logEntries.isEmpty()) {
                System.out.println("ERROR: No valid data found in log file.");
                return;
            }

            RunMetrics m = computeMetrics(logEntries);
            String fmt = "%-35s %s%n";

            // 1. Run Overview
            System.out.println("--- 1. RUN OVERVIEW ---");
            System.out.printf(fmt, "Population size:", m.populationSize);
            System.out.printf(fmt, "Search dimensions:", m.dimensions);
            System.out.printf(fmt, "Maximum iterations:", m.maxIterations);
            System.out.printf(fmt, "Optimum known:", m.optimumKnown ? "Yes" : "No (distance = NA)");
            System.out.println();

            // 2. Final Solution Quality
            System.out.println("--- 2. FINAL SOLUTION QUALITY ---");
            System.out.printf(fmt, "Global best fitness:", scientific(m.finalBestFitness));
            if (m.optimumKnown) {
                System.out.printf(fmt, "Distance to optimum:", scientific(m.finalDistanceToOptimum));
            }
            System.out.println();

            // 3. Convergence Analysis
            System.out.println("--- 3. CONVERGENCE ANALYSIS ---");
            System.out.printf(fmt, "Convergence rate (per iter):", String.format("%.6f", m.convergenceRate));
            if (m.iterationsToThreshold >= 0) {
                System.out.printf(fmt, "Iterations to 1e-6 of initial:", m.iterationsToThreshold + " of " + m.maxIterations);
            } else {
                System.out.printf(fmt, "Iterations to 1e-6 of initial:", "not reached");
            }
            System.out.printf(fmt, "Global best updates:", m.globalBestUpdates);
            if (m.stagnationStart >= 0) {
                System.out.printf(fmt, "Stagnation detected from iter:", m.stagnationStart);
            } else {
                System.out.printf(fmt, "Stagnation:", "none (still improving)");
            }
            System.out.println("\nGlobal best fitness (log10 scale):");
            List<Double> logBest = m.globalBestHistory.stream()
                    .map(v -> Math.log10(Math.max(v, 1e-300)))
                    .collect(Collectors.toList());
            printAsciiChart(logBest, 70, 10, "Log10(global best fitness)");

            // 4. Swarm Diversity
            System.out.println("\n--- 4. SWARM DIVERSITY ---");
            double initDiv = m.avgDistanceHistory.getFirst();
            double finalDiv = m.avgDistanceHistory.getLast();
            System.out.printf(fmt, "Initial avg pairwise distance:", scientific(initDiv));
            System.out.printf(fmt, "Final avg pairwise distance:", scientific(finalDiv));
            System.out.printf(fmt, "Diversity preservation (%):", String.format("%.1f%%", finalDiv / initDiv * 100));
            printAsciiChart(m.avgDistanceHistory, 70, 8, "Average pairwise distance (diversity)");

            // 5. Exploration / Exploitation Balance
            System.out.println("\n--- 5. EXPLORATION / EXPLOITATION BALANCE ---");

            // Estimate search space diameter from initial positions
            List<LogEntry> initial = logEntries.stream().filter(l -> l.iteration == 0).toList();
            double[] minPos = new double[m.dimensions], maxPos = new double[m.dimensions];
            Arrays.fill(minPos, Double.POSITIVE_INFINITY);
            Arrays.fill(maxPos, Double.NEGATIVE_INFINITY);
            for (LogEntry e : initial) {
                for (int d = 0; d < m.dimensions; d++) {
                    if (e.positionBefore[d] < minPos[d]) minPos[d] = e.positionBefore[d];
                    if (e.positionBefore[d] > maxPos[d]) maxPos[d] = e.positionBefore[d];
                }
            }
            double searchDiameter = 0.0;
            for (int d = 0; d < m.dimensions; d++) {
                searchDiameter += Math.pow(maxPos[d] - minPos[d], 2);
            }
            searchDiameter = Math.sqrt(searchDiameter);

            double initVel = m.avgVelocityMagHistory.getFirst();
            double finalVel = m.avgVelocityMagHistory.getLast();
            System.out.printf(fmt, "Search space diameter (est.):", scientific(searchDiameter));
            System.out.printf(fmt, "Initial avg velocity magnitude:", scientific(initVel));
            System.out.printf(fmt, "Final avg velocity magnitude:", scientific(finalVel));
            System.out.printf(fmt, "Exploration ratio (init):", String.format("%.4f", initVel / searchDiameter));
            System.out.printf(fmt, "Exploration ratio (final):", String.format("%.4f", finalVel / searchDiameter));
            printAsciiChart(m.avgVelocityMagHistory, 70, 8, "Average velocity magnitude per iteration");

            // 6. Population Fitness Statistics
            System.out.println("\n--- 6. POPULATION FITNESS STATISTICS ---");
            System.out.printf(fmt, "Final mean fitness:", scientific(m.popAvgHistory.getLast()));
            System.out.printf(fmt, "Final std dev fitness:", scientific(m.popStdHistory.getLast()));
            System.out.println("\nPopulation mean fitness:");
            printAsciiChart(m.popAvgHistory, 70, 8, "Population mean fitness");
            System.out.println("\nPopulation fitness std dev:");
            printAsciiChart(m.popStdHistory, 70, 8, "Population fitness std dev");

            // 7. Distance to Optimum
            if (m.optimumKnown) {
                System.out.println("\n--- 7. DISTANCE TO OPTIMUM ---");
                printAsciiChart(m.distanceToOptimumHistory, 70, 8, "Euclidean distance to optimum");
            }

            System.out.println("\n" + "=".repeat(100));
            System.out.println("Report complete. All metrics are based solely on the execution log.");
            System.out.println("=".repeat(100));
        } catch (IOException e) {
            System.err.println("Failed to read log file: " + e.getMessage());
            log.error("Log read error", e);
        }
    }

    // ---------------------------------------------------------------------
    // Main entry point
    // ---------------------------------------------------------------------
    static void main(String[] args) {
        Path logPath;
        if (args.length > 0) {
            logPath = Paths.get(args[0]);
        } else {
            logPath = Paths.get("algorithm_run.log");
        }
        generateReport(logPath);
    }

    // ---------------------------------------------------------------------
    // Internal data structure for one parsed log line
    // ---------------------------------------------------------------------
    private static class LogEntry {
        final int iteration;
        final int agentId;
        final double[] positionBefore;
        final double[] positionAfter;
        final double[] personalBest;
        final double personalBestFitness;
        final double[] globalBest;
        final double globalBestFitness;
        final double popAvgFitness;
        final double popStdDev;
        final String distToOptimumRaw;   // "NA" or numeric string

        LogEntry(int iteration, int agentId,
                 double[] positionBefore, double[] positionAfter,
                 double[] personalBest, double personalBestFitness,
                 double[] globalBest, double globalBestFitness,
                 double popAvgFitness, double popStdDev,
                 String distToOptimumRaw) {
            this.iteration = iteration;
            this.agentId = agentId;
            this.positionBefore = positionBefore;
            this.positionAfter = positionAfter;
            this.personalBest = personalBest;
            this.personalBestFitness = personalBestFitness;
            this.globalBest = globalBest;
            this.globalBestFitness = globalBestFitness;
            this.popAvgFitness = popAvgFitness;
            this.popStdDev = popStdDev;
            this.distToOptimumRaw = distToOptimumRaw;
        }
    }

    // ---------------------------------------------------------------------
    // Metrics container and computation
    // ---------------------------------------------------------------------
    private static class RunMetrics {
        int populationSize;
        int dimensions;
        int maxIterations;
        boolean optimumKnown;
        double finalBestFitness;
        double finalDistanceToOptimum = Double.NaN;
        int stagnationStart = -1;        // iteration where stagnation began (-1 if none)
        int iterationsToThreshold = -1;  // iterations to reach 1e-6 of initial fitness
        double convergenceRate;          // per-iteration geometric mean improvement
        int globalBestUpdates;           // number of iterations where gBest improved
        List<Double> globalBestHistory = new ArrayList<>();
        List<Double> avgDistanceHistory = new ArrayList<>();
        List<Double> avgVelocityMagHistory = new ArrayList<>();
        List<Double> popAvgHistory = new ArrayList<>();
        List<Double> popStdHistory = new ArrayList<>();
        List<Double> distanceToOptimumHistory = new ArrayList<>();
    }
}