package edu.swarmintelligence.cro;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Minimal CSV logger for CRO dynamics. It deliberately avoids external logging
 * frameworks because Aufgabe 4 requires a simple, machine-readable
 * {@code algorithm_run.log} that can be disabled through configuration.
 */
final class AlgorithmLogger implements AutoCloseable {
    private static final String HEADER = "iteration;agentId;positionBefore;positionAfter;"
            + "personalBest;personalBestFitness;globalBest;globalBestFitness;"
            + "popAvgFitness;popStdDev;distToOptimum";

    private BufferedWriter writer;

    private AlgorithmLogger(final BufferedWriter writer) {
        this.writer = writer;
    }

    /**
     * Opens a UTF-8 CSV log and writes the required header immediately.
     *
     * @param path file path for the algorithm log
     * @return enabled logger backed by the given file
     * @throws UncheckedIOException if the file cannot be opened or the header
     *                              cannot be written
     */
    static AlgorithmLogger open(final Path path) {
        try {
            BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
            writer.write(HEADER);
            writer.newLine();
            return new AlgorithmLogger(writer);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not open algorithm log: " + path, e);
        }
    }

    static AlgorithmLogger disabled() {
        return new AlgorithmLogger(null);
    }

    boolean isEnabled() {
        return writer != null;
    }

    /**
     * Writes one molecule state line. The entry contains both local memory
     * (personal best) and population-wide convergence values for the same
     * iteration.
     *
     * @param entry complete CSV row data
     * @throws UncheckedIOException if writing fails
     */
    void log(final LogEntry entry) {
        if (!isEnabled()) {
            return;
        }

        try {
            writer.write(entry.toCsvLine());
            writer.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException("Could not write algorithm log entry", e);
        }
    }

    /**
     * Flushes buffered CSV content so a long-running optimization leaves a
     * readable partial log.
     *
     * @throws UncheckedIOException if flushing fails
     */
    void flush() {
        if (!isEnabled()) {
            return;
        }

        try {
            writer.flush();
        } catch (IOException e) {
            throw new UncheckedIOException("Could not flush algorithm log", e);
        }
    }

    @Override
    /**
     * Closes the underlying file writer. The disabled logger and repeated close
     * calls are no-ops.
     *
     * @throws UncheckedIOException if closing the enabled writer fails
     */
    public void close() {
        if (!isEnabled()) {
            return;
        }

        try {
            writer.close();
            writer = null;
        } catch (IOException e) {
            throw new UncheckedIOException("Could not close algorithm log", e);
        }
    }

    private static String formatArray(final double[] values) {
        return Arrays.stream(values)
                .mapToObj(value -> String.format(Locale.ROOT, "%.16g", value))
                .collect(Collectors.joining(" "));
    }

    /**
     * Immutable data for one CSV row in the required Aufgabe 4 format.
     *
     * @param iteration CRO iteration number; {@code 0} denotes initialization
     * @param agentId stable molecule identifier
     * @param positionBefore molecule position before the reaction
     * @param positionAfter molecule position after the reaction
     * @param personalBest best position ever found by this molecule lineage
     * @param personalBestFitness potential energy at {@code personalBest}
     * @param globalBest best position known to the whole population
     * @param globalBestFitness potential energy at {@code globalBest}
     * @param popAvgFitness average current potential energy of finite molecules
     * @param popStdDev standard deviation of current finite potential energies
     * @param distToOptimum Euclidean distance from {@code positionAfter} to the
     *                      configured optimum
     */
    record LogEntry(int iteration,
                    long agentId,
                    double[] positionBefore,
                    double[] positionAfter,
                    double[] personalBest,
                    double personalBestFitness,
                    double[] globalBest,
                    double globalBestFitness,
                    double popAvgFitness,
                    double popStdDev,
                    double distToOptimum) {
        String toCsvLine() {
            return String.format(Locale.ROOT,
                    "%d;%d;%s;%s;%s;%.16g;%s;%.16g;%.16g;%.16g;%.16g",
                    iteration,
                    agentId,
                    formatArray(positionBefore),
                    formatArray(positionAfter),
                    formatArray(personalBest),
                    personalBestFitness,
                    formatArray(globalBest),
                    globalBestFitness,
                    popAvgFitness,
                    popStdDev,
                    distToOptimum);
        }
    }
}
