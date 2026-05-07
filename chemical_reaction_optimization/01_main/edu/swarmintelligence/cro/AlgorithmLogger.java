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

final class AlgorithmLogger implements AutoCloseable {
    private static final String HEADER = "iteration;agentId;positionBefore;positionAfter;"
            + "personalBest;personalBestFitness;globalBest;globalBestFitness;"
            + "popAvgFitness;popStdDev;distToOptimum";

    private BufferedWriter writer;

    private AlgorithmLogger(final BufferedWriter writer) {
        this.writer = writer;
    }

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
