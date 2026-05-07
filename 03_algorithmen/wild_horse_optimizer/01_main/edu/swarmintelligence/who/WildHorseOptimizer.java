package edu.swarmintelligence.who;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Wild Horse Optimizer (WHO) – strictly following
 * Naruei & Keynia, Engineering with Computers 38(Suppl 4), 3025‑3056 (2022).
 * <p>
 * Implements the hierarchical swarm intelligence algorithm with
 * grazing, water‑hole movement (with bi‑directional exploration),
 * mating, decency constraint, and dynamic TDR (time‑dependent randomness).
 * </p>
 * <p><b>Technical constraints (Java 25 LTS, stable API only):</b><br>
 * – Lombok for boilerplate reduction (no experimental features)<br>
 * – SLF4J logging<br>
 * – L128X256MixRandom – a state‑of‑the‑art scientific generator</p>
 */
@Slf4j
public final class WildHorseOptimizer {
    private final WhoConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final List<Herd> herds;
    private final List<Horse> population;
    private double[] waterHolePosition;
    @Getter
    private double waterHoleFitness;

    public WildHorseOptimizer(WhoConfig config, ObjectiveFunction function) {
        this.config = config;
        this.function = function;
        this.random = RandomGeneratorFactory.of("L128X256MixRandom").create(config.seed());
        this.herds = new ArrayList<>(config.numGroups());
        this.population = new ArrayList<>(config.populationSize());
        this.waterHoleFitness = Double.MAX_VALUE;
        initializeHerds();
        initializePopulation();
    }

    private static double clamp(double value, double min, double max) {
        return Math.clamp(value, min, max);
    }

    // ---- package‑private accessors for testing ---------------------------------
    List<Herd> herds() {
        return herds;
    }

    List<Horse> population() {
        return population;
    }

    // ---- public API ------------------------------------------------------------

    double[] waterHolePositionInternal() {
        return waterHolePosition;
    }

    /**
     * Runs the WHO algorithm and returns the best found position (water hole).
     */
    public double[] optimize() {
        int n = config.dimensions();
        var min = config.minBounds();
        var max = config.maxBounds();
        int maxIter = config.maxIterations();
        double p = config.tdrExponent();
        double pc = config.crossoverProb();

        log.info("WHO starting – population: {}, herds: {}, max iterations: {}",
                config.populationSize(), config.numGroups(), maxIter);

        for (int t = 1; t <= maxIter; t++) {
            // TDR decreases from 1 → 0 according to TDR = 1 – (t / T)^{1/p}
            double tdr = 1.0 - Math.pow((double) t / maxIter, 1.0 / p);

            // PHASE 1: GRAZING – each non‑stallion performs a spiral search around its stallion
            for (var herd : herds) {
                var stallion = herd.getStallion();
                for (var horse : herd.getMembers()) {
                    if (horse.isStallion()) continue;
                    double[] newPos = performGrazing(horse, stallion, n, min, max, tdr);
                    double newFit = function.evaluate(newPos);
                    if (newFit < horse.getFitness()) {
                        horse.setFitness(newFit);
                        System.arraycopy(newPos, 0, horse.internalPosition(), 0, n);
                    }
                }
            }

            // PHASE 2: WATER HOLE – stallions move toward or away from the global best
            for (var herd : herds) {
                var stallion = herd.getStallion();
                if (stallion == null) continue;
                double[] newPos = performWaterHole(stallion, n, min, max, tdr);
                double newFit = function.evaluate(newPos);
                if (newFit < stallion.getFitness()) {
                    stallion.setFitness(newFit);
                    System.arraycopy(newPos, 0, stallion.internalPosition(), 0, n);
                }
            }

            // PHASE 3: MATING + DECENCY – mean crossover, strict herd departure
            performMatingAndDecency(n, min, max, pc);

            // PHASE 4: update leaders
            for (var herd : herds) {
                herd.updateStallion();
            }
            updateWaterHole();

            if (t % 100 == 0) {
                log.info("Iteration {}: water hole fitness {}", t, waterHoleFitness);
            }
        }
        log.info("Optimisation finished. Final water hole fitness: {}", waterHoleFitness);
        return Arrays.copyOf(waterHolePosition, waterHolePosition.length);
    }

    /**
     * Current TDR value (for testing / analysis). Formula:
     * TDR = 1 – (t / maxIter)^{(1 / p)}
     */
    public double getCurrentTDR(int t, int maxIter, double p) {
        return 1.0 - Math.pow((double) t / maxIter, 1.0 / p);
    }

    // ---- internal phases -------------------------------------------------------

    public int getNumHerds() {
        return herds.size();
    }

    /**
     * Grazing: each horse performs a spiral search around its stallion.
     * <p>
     * Formula from the paper (dimension‑wise):<br>
     * X<sub>new</sub> = 2·Z·cos(2π·R·Z)·(Stallion – X) + Stallion<br>
     * where Z = R₂·(R₁ < TDR ? R₃ : 1)<br>
     * and R ∈ [‑2,2], R₁,R₂,R₃ ∈ [0,1] are independent random numbers.
     * </p>
     */
    double[] performGrazing(Horse horse, Horse stallion, int dim,
                            double[] min, double[] max, double tdr) {
        double[] newPos = new double[dim];
        double[] hPos = horse.internalPosition();
        double[] sPos = stallion.internalPosition();
        for (int d = 0; d < dim; d++) {
            var R = random.nextDouble(-2.0, 2.0);
            var R2 = random.nextDouble();
            var R1 = random.nextDouble();
            var R3 = random.nextDouble();
            double Z = R2 * (R1 < tdr ? R3 : 1.0);
            double spiral = 2.0 * Z * Math.cos(2.0 * Math.PI * R * Z);
            newPos[d] = clamp(spiral * (sPos[d] - hPos[d]) + sPos[d], min[d], max[d]);
        }
        return newPos;
    }

    /**
     * Water hole: the stallion moves toward (rand > 0.5) or away from
     * (rand ≤ 0.5) the global best position using the same spiral mechanism.
     * <p>
     * X<sub>new</sub> = 2·Z·cos(2π·R·Z)·(WH – Stallion) + WH&emsp;(if rand > 0.5)<br>
     * X<sub>new</sub> = WH – 2·Z·cos(2π·R·Z)·(WH – Stallion)&emsp;(otherwise)
     * </p>
     */
    double[] performWaterHole(Horse stallion, int dim,
                              double[] min, double[] max, double tdr) {
        double[] newPos = new double[dim];
        double[] sPos = stallion.internalPosition();
        double sign = random.nextDouble() > 0.5 ? 1.0 : -1.0;   // single decision for whole vector
        for (int d = 0; d < dim; d++) {
            var R = random.nextDouble(-2.0, 2.0);
            var R2 = random.nextDouble();
            var R1 = random.nextDouble();
            var R3 = random.nextDouble();
            double Z = R2 * (R1 < tdr ? R3 : 1.0);
            double spiral = 2.0 * Z * Math.cos(2.0 * Math.PI * R * Z);
            double diff = waterHolePosition[d] - sPos[d];
            newPos[d] = clamp(waterHolePosition[d] + sign * spiral * diff, min[d], max[d]);
        }
        return newPos;
    }

    /**
     * Mating and decency: each non‑stallion horse may mate (crossover probability PC)
     * with a random horse from a different herd via <b>mean crossover</b>. The resulting
     * foal is placed into a herd <i>g″ ≠ g, g′</i> (strict decency), replacing the
     * weakest member if it is better.
     */
    void performMatingAndDecency(int dim, double[] min, double[] max, double pc) {
        var foals = new ArrayList<FoalData>();
        for (var herd : herds) {
            for (var horse : herd.getMembers()) {
                if (horse.isStallion() || random.nextDouble() > pc) continue;
                var otherHerd = selectRandomHerdExcluding(herd.getId());
                if (otherHerd == null || otherHerd.getMembers().isEmpty()) continue;
                var otherHorse = otherHerd.getMembers().get(
                        random.nextInt(otherHerd.getMembers().size()));
                double[] foalPos = new double[dim];
                for (int d = 0; d < dim; d++) {
                    foalPos[d] = (horse.internalPosition()[d] + otherHorse.internalPosition()[d]) / 2.0;
                    foalPos[d] = clamp(foalPos[d], min[d], max[d]);
                }
                double fit = function.evaluate(foalPos);
                foals.add(new FoalData(new Horse(dim, foalPos, fit), herd.getId(), otherHerd.getId()));
            }
        }

        // decency constraint: foal must join group g″ ≠ g, g′
        for (var fd : foals) {
            var validHerds = herds.stream()
                    .filter(h -> h.getId() != fd.parent1() && h.getId() != fd.parent2())
                    .toList();
            if (validHerds.isEmpty()) continue;
            var target = validHerds.get(random.nextInt(validHerds.size()));
            var weakest = target.getWeakestMember();
            if (weakest != null && fd.foal().getFitness() < weakest.getFitness()) {
                target.removeMember(weakest);
                population.remove(weakest);
                target.addMember(fd.foal());
                population.add(fd.foal());
                log.trace("Decency: foal moved to herd {}", target.getId());
            }
        }
    }

    Herd selectRandomHerdExcluding(int excludeId) {
        var candidates = herds.stream().filter(h -> h.getId() != excludeId).toList();
        return candidates.isEmpty() ? null : candidates.get(random.nextInt(candidates.size()));
    }

    // ---- helpers ---------------------------------------------------------------

    void updateWaterHole() {
        for (var horse : population) {
            if (horse.getFitness() < waterHoleFitness) {
                waterHoleFitness = horse.getFitness();
                waterHolePosition = horse.getPosition(); // defensive copy
            }
        }
    }

    private void initializeHerds() {
        for (int i = 0; i < config.numGroups(); i++) {
            herds.add(new Herd(i));
        }
    }

    private void initializePopulation() {
        var min = config.minBounds();
        var max = config.maxBounds();
        int n = config.dimensions();
        for (int i = 0; i < config.populationSize(); i++) {
            double[] x = new double[n];
            for (int d = 0; d < n; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            population.add(new Horse(n, x, function.evaluate(x)));
        }
        int perHerd = config.populationSize() / config.numGroups();
        int rem = config.populationSize() % config.numGroups();
        int idx = 0;
        for (int g = 0; g < config.numGroups(); g++) {
            var herd = herds.get(g);
            int count = perHerd + (g < rem ? 1 : 0);
            for (int h = 0; h < count && idx < config.populationSize(); h++) {
                herd.addMember(population.get(idx++));
            }
            herd.updateStallion();
        }
        updateWaterHole();
    }

    // ---- inner types -----------------------------------------------------------

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable configuration for the WHO.
     * <p>All array components are defensively copied; accessors return copies.</p>
     */
    public record WhoConfig(
            int populationSize,
            int numGroups,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            double crossoverProb,
            double tdrExponent,
            long seed
    ) {
        public WhoConfig {
            if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0 required");
            if (numGroups < 3) throw new IllegalArgumentException("numGroups >= 3 (decency constraint)");
            if (populationSize < 3 * numGroups)
                throw new IllegalArgumentException("populationSize >= 3*numGroups");
            if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("minBounds/maxBounds must match dimensions");
            if (crossoverProb < 0.0 || crossoverProb > 1.0)
                throw new IllegalArgumentException("crossoverProb in [0,1]");
            if (tdrExponent <= 0.0) throw new IllegalArgumentException("tdrExponent > 0");

            minBounds = Arrays.copyOf(minBounds, dimensions);
            maxBounds = Arrays.copyOf(maxBounds, dimensions);
        }

        public static WhoConfig createDefaultConfig(int populationSize, int maxIterations,
                                                    int dimensions, double[] min, double[] max,
                                                    long seed) {
            int groups = (int) Math.floor(0.1 * populationSize);
            if (groups < 3) groups = 3;
            return new WhoConfig(populationSize, groups, maxIterations, dimensions,
                    min, max, 0.5, 2.0, seed);
        }

        @Override
        public double[] minBounds() {
            return Arrays.copyOf(minBounds, dimensions);
        }

        @Override
        public double[] maxBounds() {
            return Arrays.copyOf(maxBounds, dimensions);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof WhoConfig(
                    int ps, int ng, int mi, int d,
                    double[] mn, double[] mx, double cp, double te, long s
            )))
                return false;
            return populationSize == ps && numGroups == ng && maxIterations == mi
                    && dimensions == d && Double.compare(crossoverProb, cp) == 0
                    && Double.compare(tdrExponent, te) == 0 && seed == s
                    && Arrays.equals(minBounds, mn) && Arrays.equals(maxBounds, mx);
        }

        @Override
        public int hashCode() {
            return Arrays.deepHashCode(new Object[]{populationSize, numGroups, maxIterations,
                    dimensions, minBounds, maxBounds,
                    crossoverProb, tdrExponent, seed});
        }

        @Override
        @NonNull
        public String toString() {
            return "WhoConfig[populationSize=%d, numGroups=%d, maxIterations=%d, dimensions=%d, minBounds=%s, maxBounds=%s, crossoverProb=%.4f, tdrExponent=%.4f, seed=%d]".formatted(
                    populationSize, numGroups, maxIterations, dimensions,
                    Arrays.toString(minBounds), Arrays.toString(maxBounds),
                    crossoverProb, tdrExponent, seed);
        }
    }

    // ---- package‑private inner types (accessible for testing) ------------------

    /**
     * Links a foal to its parent herds for the decency check.
     */
    record FoalData(Horse foal, int parent1, int parent2) {
    }

    @Getter
    @Setter
    static final class Horse {
        @Getter(AccessLevel.NONE)
        private final double[] position;   // internal reference, defensive copy on getPosition()
        private double fitness;
        private int groupId = -1;
        private boolean stallion;

        public Horse(int dimensions, double[] initialPosition, double fitness) {
            this.position = Arrays.copyOf(initialPosition, dimensions);
            this.fitness = fitness;
        }

        public double[] getPosition() {
            return Arrays.copyOf(position, position.length);
        }

        double[] internalPosition() {
            return position;
        }   // package‑private for algorithm
    }

    @Getter
    static final class Herd {
        private final int id;
        private final List<Horse> members = new ArrayList<>();
        @Setter
        private Horse stallion;

        Herd(int id) {
            this.id = id;
        }

        void addMember(Horse horse) {
            members.add(horse);
            horse.setGroupId(id);
        }

        void removeMember(Horse horse) {
            members.remove(horse);
            horse.setGroupId(-1);
            horse.setStallion(false);
            if (horse.equals(stallion)) stallion = null;
        }

        void updateStallion() {
            if (members.isEmpty()) {
                stallion = null;
                return;
            }
            var best = members.getFirst();
            for (var h : members) if (h.getFitness() < best.getFitness()) best = h;
            if (stallion != null) stallion.setStallion(false);
            setStallion(best);
            best.setStallion(true);
        }

        Horse getWeakestMember() {
            if (members.isEmpty()) return null;
            var weakest = members.getFirst();
            for (var h : members) if (h.getFitness() > weakest.getFitness()) weakest = h;
            return weakest;
        }
    }

    // ---- example usage ---------------------------------------------------------

    @Slf4j
    public static final class WHOExample {
        static void main() {
            ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
            int dim = 10;
            double[] min = new double[dim], max = new double[dim];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);
            var cfg = WhoConfig.createDefaultConfig(50, 1000, dim, min, max, 12345L);
            var who = new WildHorseOptimizer(cfg, sphere);
            var best = who.optimize();
            log.info("Best: {}, fitness: {}", Arrays.toString(best), sphere.evaluate(best));
        }
    }
}