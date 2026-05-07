package edu.swarmintelligence.cro;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Chemical Reaction Optimization (CRO) for bounded continuous minimization
 * problems.
 *
 * <p>The implementation follows the standard CRO idea of molecules carrying a
 * structure, potential energy and kinetic energy. Reactions are accepted only
 * when the available energy can pay for the new potential energy. Lower
 * objective values are always considered better.</p>
 */
public class ChemicalReactionOptimization {
    private static final int MINIMUM_POPULATION_SIZE = 2;
    private static final int MAX_INITIALIZATION_ATTEMPTS = 1000;

    private final CroConfig config;
    private final ObjectiveFunction function;
    private final Random random;
    private final double[] minBounds;
    private final double[] maxBounds;
    private final double[] knownOptimum;
    private final AlgorithmLogger logger;

    private final List<Molecule> population;
    private long nextMoleculeId;
    private double buffer;
    private double[] globalBestStructure;
    private double globalBestPotentialEnergy;

    /**
     * Creates a CRO optimizer, prepares deterministic randomness, optionally
     * opens the CSV logger, and initializes the molecular population.
     *
     * @param config validated algorithm parameters, search bounds and logging options
     * @param function objective function interpreted as molecular potential energy
     * @throws NullPointerException if {@code config} or {@code function} is {@code null}
     * @throws IllegalStateException if no finite initial molecule can be sampled
     */
    public ChemicalReactionOptimization(final CroConfig config,
                                        final ObjectiveFunction function) {
        this.config = Objects.requireNonNull(config, "config");
        this.function = Objects.requireNonNull(function, "function");
        this.random = new Random(config.seed());
        this.minBounds = config.minBounds();
        this.maxBounds = config.maxBounds();
        this.knownOptimum = config.knownOptimum();
        this.logger = config.loggingEnabled()
                ? AlgorithmLogger.open(Path.of(config.logPath()))
                : AlgorithmLogger.disabled();
        this.population = new ArrayList<>(config.popSize());
        this.nextMoleculeId = 0L;
        this.buffer = config.enBuff();
        this.globalBestPotentialEnergy = Double.POSITIVE_INFINITY;
        initializePopulation();
    }

    /**
     * Samples the initial molecule population uniformly inside the configured
     * bounds and records the initial global best. Iteration {@code 0} is logged
     * here because no reaction has happened yet, so before/after positions are
     * identical for the initial state.
     *
     * @throws IllegalStateException if all sampled molecules have non-finite
     *                               potential energy
     */
    private void initializePopulation() {
        for (int i = 0; i < config.popSize(); i++) {
            population.add(createRandomMolecule(config.initialKE()));
        }
        updateGlobalBest();
        if (globalBestStructure == null) {
            throw new IllegalStateException("No feasible initial molecule found.");
        }
        writeLogEntries(0, snapshotPositions());
    }

    /**
     * Runs the configured number of CRO reactions and returns the best structure
     * found. The returned array is a defensive copy and can be changed by the
     * caller without affecting the optimizer state.
     *
     * @return best molecular structure found over the complete run
     * @throws java.io.UncheckedIOException if enabled CSV logging cannot write or close
     */
    public double[] optimize() {
        try {
            for (int iteration = 1; iteration <= config.maxIterations(); iteration++) {
                Map<Long, double[]> positionsBefore = snapshotPositions();
                if (shouldUseBimolecularReaction()) {
                    reactWithTwoMolecules();
                } else {
                    reactWithOneMolecule();
                }
                updateGlobalBest();
                writeLogEntries(iteration, positionsBefore);
            }
        } finally {
            logger.close();
        }
        return globalBestStructure.clone();
    }

    private void reactWithOneMolecule() {
        int index = random.nextInt(population.size());
        Molecule molecule = population.get(index);
        if (molecule.isStagnating(config.decThres())) {
            performDecomposition(index);
        } else {
            performOnWallCollision(molecule);
        }
    }

    private void reactWithTwoMolecules() {
        int firstIndex = random.nextInt(population.size());
        int secondIndex = randomDifferentIndex(firstIndex);

        Molecule first = population.get(firstIndex);
        Molecule second = population.get(secondIndex);

        if (canSynthesize(first, second)) {
            performSynthesis(firstIndex, secondIndex);
        } else {
            performInterMolecularCollision(first, second);
        }
    }

    private boolean shouldUseBimolecularReaction() {
        return population.size() >= MINIMUM_POPULATION_SIZE
                && random.nextDouble() < config.moleColl();
    }

    private int randomDifferentIndex(final int firstIndex) {
        int secondIndex = random.nextInt(population.size() - 1);
        return secondIndex >= firstIndex ? secondIndex + 1 : secondIndex;
    }

    private boolean canSynthesize(final Molecule first, final Molecule second) {
        return population.size() > MINIMUM_POPULATION_SIZE
                && first.hasKineticEnergyAtMost(config.synThres())
                && second.hasKineticEnergyAtMost(config.synThres());
    }

    /**
     * On-wall ineffective collision: one molecule is perturbed. If the new
     * structure is energetically feasible, part of the surplus energy remains as
     * kinetic energy and the rest is stored in the central buffer.
     *
     * @param molecule molecule selected for a unimolecular wall collision
     */
    private void performOnWallCollision(final Molecule molecule) {
        double[] candidateStructure = perturb(molecule.structure());
        double candidatePotentialEnergy = evaluate(candidateStructure);

        if (!Double.isFinite(candidatePotentialEnergy)) {
            molecule.registerCollision();
            return;
        }

        double availableEnergy = molecule.totalEnergy();
        if (hasEnoughEnergy(availableEnergy, candidatePotentialEnergy)) {
            double surplus = availableEnergy - candidatePotentialEnergy;
            double kineticRatio = config.kelossRate()
                    + random.nextDouble() * (1.0 - config.kelossRate());
            molecule.replaceState(candidateStructure, candidatePotentialEnergy,
                    surplus * kineticRatio);
            buffer += surplus * (1.0 - kineticRatio);
        } else {
            molecule.registerCollision();
        }
    }

    /**
     * Decomposition: one molecule is split into two perturbed molecules. Energy
     * missing from the source molecule may be borrowed from the central buffer.
     * This operator increases population size and helps escape stagnation around
     * a molecule's current local minimum.
     *
     * @param index index of the source molecule in the current population
     */
    private void performDecomposition(final int index) {
        Molecule source = population.get(index);
        double[] firstStructure = perturb(source.bestStructure());
        double[] secondStructure = perturb(source.structure());
        double firstPotentialEnergy = evaluate(firstStructure);
        double secondPotentialEnergy = evaluate(secondStructure);

        if (!Double.isFinite(firstPotentialEnergy) || !Double.isFinite(secondPotentialEnergy)) {
            source.registerCollision();
            return;
        }

        double availableEnergy = source.totalEnergy();
        double requiredPotentialEnergy = firstPotentialEnergy + secondPotentialEnergy;
        if (hasEnoughEnergy(availableEnergy, requiredPotentialEnergy)) {
            replaceWithDecompositionProducts(index, firstStructure, firstPotentialEnergy,
                    secondStructure, secondPotentialEnergy,
                    availableEnergy - requiredPotentialEnergy);
            return;
        }

        double deficit = requiredPotentialEnergy - availableEnergy;
        if (buffer >= deficit) {
            buffer -= deficit;
            replaceWithDecompositionProducts(index, firstStructure, firstPotentialEnergy,
                    secondStructure, secondPotentialEnergy, 0.0);
        } else {
            source.registerCollision();
        }
    }

    /**
     * Replaces a stagnating molecule with the two products of decomposition.
     * One product keeps the original molecule ID for log continuity; the second
     * product receives a new ID because it is a newly created molecule.
     *
     * @param index position of the source molecule in the population
     * @param firstStructure structure of the first product
     * @param firstPotentialEnergy potential energy of the first product
     * @param secondStructure structure of the second product
     * @param secondPotentialEnergy potential energy of the second product
     * @param surplusEnergy kinetic energy to split randomly between products
     */
    private void replaceWithDecompositionProducts(final int index,
                                                  final double[] firstStructure,
                                                  final double firstPotentialEnergy,
                                                  final double[] secondStructure,
                                                  final double secondPotentialEnergy,
                                                  final double surplusEnergy) {
        Molecule source = population.get(index);
        double firstKineticEnergy = random.nextDouble() * surplusEnergy;
        population.set(index, new Molecule(source.id(), firstStructure,
                firstPotentialEnergy, firstKineticEnergy));
        population.add(new Molecule(nextMoleculeId++, secondStructure, secondPotentialEnergy,
                surplusEnergy - firstKineticEnergy));
    }

    /**
     * Inter-molecular ineffective collision: two molecules are perturbed
     * independently and accepted when their combined energy can pay for both new
     * structures.
     *
     * @param first first selected molecule
     * @param second second selected molecule
     */
    private void performInterMolecularCollision(final Molecule first,
                                                final Molecule second) {
        double[] firstStructure = perturb(first.structure());
        double[] secondStructure = perturb(second.structure());
        double firstPotentialEnergy = evaluate(firstStructure);
        double secondPotentialEnergy = evaluate(secondStructure);

        if (!Double.isFinite(firstPotentialEnergy) || !Double.isFinite(secondPotentialEnergy)) {
            first.registerCollision();
            second.registerCollision();
            return;
        }

        double availableEnergy = first.totalEnergy() + second.totalEnergy();
        double requiredPotentialEnergy = firstPotentialEnergy + secondPotentialEnergy;
        if (hasEnoughEnergy(availableEnergy, requiredPotentialEnergy)) {
            double surplus = availableEnergy - requiredPotentialEnergy;
            double firstKineticEnergy = random.nextDouble() * surplus;
            first.replaceState(firstStructure, firstPotentialEnergy, firstKineticEnergy);
            second.replaceState(secondStructure, secondPotentialEnergy,
                    surplus - firstKineticEnergy);
        } else {
            first.registerCollision();
            second.registerCollision();
        }
    }

    /**
     * Synthesis: two low-energy molecules are merged into one molecule. This
     * reaction is skipped when it would shrink the population below two.
     *
     * @param firstIndex index of the first molecule; its ID is retained after a
     *                   successful synthesis
     * @param secondIndex index of the second molecule removed by a successful
     *                    synthesis
     */
    private void performSynthesis(final int firstIndex, final int secondIndex) {
        Molecule first = population.get(firstIndex);
        Molecule second = population.get(secondIndex);
        double[] candidateStructure = new double[config.dimensions()];
        for (int d = 0; d < candidateStructure.length; d++) {
            double ratio = random.nextDouble();
            candidateStructure[d] = clamp(
                    first.coordinate(d) + ratio * (second.coordinate(d) - first.coordinate(d)),
                    minBounds[d],
                    maxBounds[d]);
        }

        double candidatePotentialEnergy = evaluate(candidateStructure);
        if (!Double.isFinite(candidatePotentialEnergy)) {
            first.registerCollision();
            second.registerCollision();
            return;
        }

        double availableEnergy = first.totalEnergy() + second.totalEnergy();
        if (hasEnoughEnergy(availableEnergy, candidatePotentialEnergy)) {
            int highIndex = Math.max(firstIndex, secondIndex);
            int lowIndex = Math.min(firstIndex, secondIndex);
            population.remove(highIndex);
            population.set(lowIndex, new Molecule(first.id(), candidateStructure,
                    candidatePotentialEnergy, availableEnergy - candidatePotentialEnergy));
        } else {
            first.registerCollision();
            second.registerCollision();
        }
    }

    /**
     * Produces a local Gaussian mutation of a molecule and clamps every
     * coordinate to the search bounds. CRO reactions use this neighborhood
     * operator to explore without leaving the feasible search domain.
     *
     * @param structure source structure before mutation
     * @return bounded candidate structure
     */
    private double[] perturb(final double[] structure) {
        double[] result = new double[config.dimensions()];
        for (int d = 0; d < result.length; d++) {
            result[d] = clamp(structure[d] + random.nextGaussian() * config.stepSize(),
                    minBounds[d], maxBounds[d]);
        }
        return result;
    }

    /**
     * Samples a feasible molecule for the initial population. CRO needs finite
     * potential energy before it can apply energy-conserving reactions, so this
     * method retries random structures rather than accepting invalid objective
     * values.
     *
     * @param kineticEnergy initial kinetic energy assigned to the molecule
     * @return newly sampled molecule with finite potential energy
     * @throws IllegalStateException if no feasible structure is sampled within
     *                               the retry limit
     */
    private Molecule createRandomMolecule(final double kineticEnergy) {
        for (int attempt = 0; attempt < MAX_INITIALIZATION_ATTEMPTS; attempt++) {
            double[] structure = randomStructure();
            double potentialEnergy = evaluate(structure);
            if (Double.isFinite(potentialEnergy)) {
                return new Molecule(nextMoleculeId++, structure, potentialEnergy, kineticEnergy);
            }
        }
        throw new IllegalStateException("No feasible molecule found after "
                + MAX_INITIALIZATION_ATTEMPTS + " attempts.");
    }

    /**
     * Draws one molecular structure uniformly from the configured search box.
     *
     * @return position vector with one coordinate inside each configured bound
     */
    private double[] randomStructure() {
        double[] structure = new double[config.dimensions()];
        for (int d = 0; d < structure.length; d++) {
            structure[d] = minBounds[d] + random.nextDouble() * (maxBounds[d] - minBounds[d]);
        }
        return structure;
    }

    /**
     * Evaluates a structure defensively so user-supplied objective functions
     * cannot mutate molecule state through the passed array.
     *
     * @param structure candidate molecule structure
     * @return objective value interpreted as potential energy
     */
    private double evaluate(final double[] structure) {
        return function.evaluate(structure.clone());
    }

    private static boolean hasEnoughEnergy(final double availableEnergy,
                                           final double requiredPotentialEnergy) {
        return requiredPotentialEnergy <= availableEnergy;
    }

    /**
     * Updates the population-wide best record from each molecule's personal
     * best. This preserves discoveries even when a later accepted CRO reaction
     * moves a molecule to a worse current position.
     */
    private void updateGlobalBest() {
        for (Molecule molecule : population) {
            if (molecule.bestPotentialEnergy() < globalBestPotentialEnergy) {
                globalBestPotentialEnergy = molecule.bestPotentialEnergy();
                globalBestStructure = molecule.bestStructure();
            }
        }
    }

    /**
     * Captures molecule positions before a reaction so the logger can record
     * per-agent movement for the current iteration.
     *
     * @return map from stable molecule ID to its position before the reaction
     */
    private Map<Long, double[]> snapshotPositions() {
        Map<Long, double[]> positions = new HashMap<>();
        for (Molecule molecule : population) {
            positions.put(molecule.id(), molecule.structure());
        }
        return positions;
    }

    /**
     * Writes one CSV row per current molecule for the requested iteration.
     * Molecules created during decomposition use their after-position as the
     * before-position because they did not exist at the iteration start.
     *
     * @param iteration CRO iteration number, with {@code 0} representing the
     *                  initialized population
     * @param positionsBefore positions captured before this iteration's reaction
     * @throws java.io.UncheckedIOException if the enabled logger cannot write
     */
    private void writeLogEntries(final int iteration,
                                 final Map<Long, double[]> positionsBefore) {
        if (!logger.isEnabled()) {
            return;
        }

        PopulationStats stats = populationStats();
        for (Molecule molecule : population) {
            double[] positionAfter = molecule.structure();
            double[] positionBefore = positionsBefore.getOrDefault(molecule.id(), positionAfter);
            logger.log(new AlgorithmLogger.LogEntry(
                    iteration,
                    molecule.id(),
                    positionBefore,
                    positionAfter,
                    molecule.bestStructure(),
                    molecule.bestPotentialEnergy(),
                    globalBestStructure,
                    globalBestPotentialEnergy,
                    stats.averageFitness(),
                    stats.standardDeviation(),
                    distanceToOptimum(positionAfter)
            ));
        }
        logger.flush();
    }

    /**
     * Computes convergence statistics from current finite molecule potential
     * energies. Non-finite energies are excluded so a rejected or invalid
     * candidate cannot distort the logged population summary.
     *
     * @return average and population standard deviation of current finite
     *         potential energies
     */
    private PopulationStats populationStats() {
        double sum = 0.0;
        int finiteCount = 0;
        for (Molecule molecule : population) {
            double fitness = molecule.potentialEnergy();
            if (Double.isFinite(fitness)) {
                sum += fitness;
                finiteCount++;
            }
        }

        if (finiteCount == 0) {
            return new PopulationStats(Double.NaN, Double.NaN);
        }

        double average = sum / finiteCount;
        double varianceSum = 0.0;
        for (Molecule molecule : population) {
            double fitness = molecule.potentialEnergy();
            if (Double.isFinite(fitness)) {
                double diff = fitness - average;
                varianceSum += diff * diff;
            }
        }
        return new PopulationStats(average, Math.sqrt(varianceSum / finiteCount));
    }

    /**
     * Computes Euclidean distance from a molecule to the configured global
     * optimum. The Ackley logging scenario sets the optimum to the origin.
     *
     * @param position molecule position after the current reaction
     * @return distance to optimum, or {@link Double#NaN} when no optimum is configured
     */
    private double distanceToOptimum(final double[] position) {
        if (knownOptimum == null) {
            return Double.NaN;
        }

        double sumSquared = 0.0;
        for (int d = 0; d < position.length; d++) {
            double diff = position[d] - knownOptimum[d];
            sumSquared += diff * diff;
        }
        return Math.sqrt(sumSquared);
    }

    private static double clamp(final double value, final double min, final double max) {
        return Math.max(min, Math.min(max, value));
    }

    public double getTotalSystemEnergy() {
        double total = buffer;
        for (Molecule molecule : population) {
            total += molecule.totalEnergy();
        }
        return total;
    }

    public double getBestPE() {
        return globalBestPotentialEnergy;
    }

    int getPopulationSize() {
        return population.size();
    }

    @FunctionalInterface
    public interface ObjectiveFunction {
        /**
         * Computes the potential energy of a molecule structure. Lower values
         * are better because this CRO implementation solves minimization
         * problems.
         *
         * @param x candidate structure; implementations should treat it as read-only
         * @return finite objective value for feasible structures, or non-finite
         *         value to reject a candidate
         */
        double evaluate(double[] x);
    }

    /**
     * Configuration for bounded CRO minimization. Array parameters are copied
     * defensively on construction and when accessed.
     *
     * @param popSize initial number of molecules; must be at least two
     * @param maxIterations number of CRO reactions to execute
     * @param dimensions dimensionality of each molecule structure
     * @param minBounds lower bound per dimension
     * @param maxBounds upper bound per dimension
     * @param kelossRate minimum retained kinetic-energy ratio in on-wall collisions
     * @param moleColl probability of attempting a bimolecular reaction
     * @param decThres stagnation threshold before decomposition is attempted
     * @param synThres kinetic-energy threshold for synthesis candidates
     * @param initialKE kinetic energy assigned to each initial molecule
     * @param enBuff initial energy in the central buffer
     * @param stepSize standard deviation of Gaussian perturbations
     * @param seed random seed for reproducible runs
     * @param loggingEnabled whether CSV logging is written during optimization
     * @param logPath target path for {@code algorithm_run.log} style CSV output
     * @param knownOptimum optional optimum used for logged distance values
     */
    public record CroConfig(
            int popSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            double kelossRate,
            double moleColl,
            int decThres,
            double synThres,
            double initialKE,
            double enBuff,
            double stepSize,
            long seed,
            boolean loggingEnabled,
            String logPath,
            double[] knownOptimum
    ) {
        public CroConfig {
            Objects.requireNonNull(minBounds, "minBounds");
            Objects.requireNonNull(maxBounds, "maxBounds");
            validateCoreSettings(popSize, maxIterations, dimensions, minBounds, maxBounds);
            validateFiniteProbability(kelossRate, "kelossRate");
            validateFiniteProbability(moleColl, "moleColl");
            validateEnergySettings(decThres, synThres, initialKE, enBuff, stepSize);

            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
            validateBoundIntervals(minBounds, maxBounds);
            logPath = validateLogPath(loggingEnabled, logPath);
            knownOptimum = validateKnownOptimum(knownOptimum, dimensions);
        }

        public double[] minBounds() {
            return minBounds.clone();
        }

        public double[] maxBounds() {
            return maxBounds.clone();
        }

        public double[] knownOptimum() {
            return knownOptimum == null ? null : knownOptimum.clone();
        }

        public static CroConfigBuilder builder() {
            return new CroConfigBuilder();
        }

        private static void validateFiniteProbability(final double value, final String name) {
            if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
                throw new IllegalArgumentException(name + " must be finite and in [0, 1]");
            }
        }

        private static void validateCoreSettings(final int popSize,
                                                 final int maxIterations,
                                                 final int dimensions,
                                                 final double[] minBounds,
                                                 final double[] maxBounds) {
            if (popSize < MINIMUM_POPULATION_SIZE) {
                throw new IllegalArgumentException("PopSize must be >= " + MINIMUM_POPULATION_SIZE);
            }
            if (maxIterations <= 0) {
                throw new IllegalArgumentException("MaxIterations must be > 0");
            }
            if (dimensions <= 0) {
                throw new IllegalArgumentException("Dimensions must be > 0");
            }
            if (minBounds.length != dimensions || maxBounds.length != dimensions) {
                throw new IllegalArgumentException("Bounds length must match dimensions");
            }
        }

        private static void validateEnergySettings(final int decThres,
                                                   final double synThres,
                                                   final double initialKE,
                                                   final double enBuff,
                                                   final double stepSize) {
            if (decThres <= 0) {
                throw new IllegalArgumentException("decThres must be > 0");
            }
            validateFiniteNonNegative(synThres, "synThres");
            validateFinitePositive(initialKE, "initialKE");
            validateFiniteNonNegative(enBuff, "enBuff");
            validateFinitePositive(stepSize, "stepSize");
        }

        private static void validateFiniteNonNegative(final double value,
                                                      final String name) {
            if (!Double.isFinite(value) || value < 0.0) {
                throw new IllegalArgumentException(name + " must be finite and >= 0");
            }
        }

        private static void validateFinitePositive(final double value,
                                                   final String name) {
            if (!Double.isFinite(value) || value <= 0.0) {
                throw new IllegalArgumentException(name + " must be finite and > 0");
            }
        }

        private static void validateBoundIntervals(final double[] minBounds,
                                                   final double[] maxBounds) {
            for (int d = 0; d < minBounds.length; d++) {
                if (!Double.isFinite(minBounds[d]) || !Double.isFinite(maxBounds[d])) {
                    throw new IllegalArgumentException("Bounds must be finite");
                }
                if (minBounds[d] >= maxBounds[d]) {
                    throw new IllegalArgumentException("Each min bound must be smaller than max bound");
                }
            }
        }

        private static String validateLogPath(final boolean loggingEnabled,
                                              final String logPath) {
            if (!loggingEnabled) {
                return logPath;
            }
            if (logPath == null || logPath.isBlank()) {
                throw new IllegalArgumentException("logPath must be set when logging is enabled");
            }
            return logPath;
        }

        private static double[] validateKnownOptimum(final double[] knownOptimum,
                                                     final int dimensions) {
            if (knownOptimum == null) {
                return null;
            }
            if (knownOptimum.length != dimensions) {
                throw new IllegalArgumentException("knownOptimum length must match dimensions");
            }

            double[] copy = knownOptimum.clone();
            for (double value : copy) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("knownOptimum values must be finite");
                }
            }
            return copy;
        }

        public static final class CroConfigBuilder {
            private int popSize;
            private int maxIterations;
            private int dimensions;
            private double[] minBounds;
            private double[] maxBounds;
            private double kelossRate;
            private double moleColl;
            private int decThres;
            private double synThres;
            private double initialKE;
            private double enBuff;
            private double stepSize;
            private long seed;
            private boolean loggingEnabled;
            private String logPath = "algorithm_run.log";
            private double[] knownOptimum;

            public CroConfigBuilder popSize(final int popSize) {
                this.popSize = popSize;
                return this;
            }

            public CroConfigBuilder maxIterations(final int maxIterations) {
                this.maxIterations = maxIterations;
                return this;
            }

            public CroConfigBuilder dimensions(final int dimensions) {
                this.dimensions = dimensions;
                return this;
            }

            public CroConfigBuilder minBounds(final double[] minBounds) {
                this.minBounds = minBounds;
                return this;
            }

            public CroConfigBuilder maxBounds(final double[] maxBounds) {
                this.maxBounds = maxBounds;
                return this;
            }

            public CroConfigBuilder kelossRate(final double kelossRate) {
                this.kelossRate = kelossRate;
                return this;
            }

            public CroConfigBuilder moleColl(final double moleColl) {
                this.moleColl = moleColl;
                return this;
            }

            public CroConfigBuilder decThres(final int decThres) {
                this.decThres = decThres;
                return this;
            }

            public CroConfigBuilder synThres(final double synThres) {
                this.synThres = synThres;
                return this;
            }

            public CroConfigBuilder initialKE(final double initialKE) {
                this.initialKE = initialKE;
                return this;
            }

            public CroConfigBuilder enBuff(final double enBuff) {
                this.enBuff = enBuff;
                return this;
            }

            public CroConfigBuilder stepSize(final double stepSize) {
                this.stepSize = stepSize;
                return this;
            }

            public CroConfigBuilder seed(final long seed) {
                this.seed = seed;
                return this;
            }

            public CroConfigBuilder loggingEnabled(final boolean loggingEnabled) {
                this.loggingEnabled = loggingEnabled;
                return this;
            }

            public CroConfigBuilder logPath(final String logPath) {
                this.logPath = logPath;
                return this;
            }

            public CroConfigBuilder knownOptimum(final double[] knownOptimum) {
                this.knownOptimum = knownOptimum;
                return this;
            }

            public CroConfig build() {
                return new CroConfig(popSize, maxIterations, dimensions, minBounds, maxBounds,
                        kelossRate, moleColl, decThres, synThres, initialKE, enBuff,
                        stepSize, seed, loggingEnabled, logPath, knownOptimum);
            }
        }
    }

    private static final class Molecule {
        private final long id;
        private double[] structure;
        private double potentialEnergy;
        private double kineticEnergy;
        private double[] minStructure;
        private double minPotentialEnergy;
        private int collisionCount;
        private int bestCollisionCount;

        Molecule(final long id,
                 final double[] structure,
                 final double potentialEnergy,
                 final double kineticEnergy) {
            this.id = id;
            this.structure = structure.clone();
            this.potentialEnergy = potentialEnergy;
            this.kineticEnergy = kineticEnergy;
            this.minStructure = structure.clone();
            this.minPotentialEnergy = potentialEnergy;
            this.collisionCount = 0;
            this.bestCollisionCount = 0;
        }

        /**
         * Accepts a reaction product as the molecule's current state and then
         * updates its personal best if the new potential energy is lower.
         *
         * @param newStructure accepted bounded structure
         * @param newPotentialEnergy objective value of the new structure
         * @param newKineticEnergy kinetic energy left after the reaction
         */
        void replaceState(final double[] newStructure,
                          final double newPotentialEnergy,
                          final double newKineticEnergy) {
            structure = newStructure.clone();
            potentialEnergy = newPotentialEnergy;
            kineticEnergy = Math.max(0.0, newKineticEnergy);
            registerCollision();
        }

        /**
         * Records that this molecule participated in a reaction attempt. The
         * counter is used to detect stagnation for decomposition, and personal
         * best state is refreshed when the current potential energy improves.
         */
        void registerCollision() {
            collisionCount++;
            if (potentialEnergy < minPotentialEnergy) {
                minPotentialEnergy = potentialEnergy;
                minStructure = structure.clone();
                bestCollisionCount = collisionCount;
            }
        }

        long id() {
            return id;
        }

        double[] structure() {
            return structure.clone();
        }

        double coordinate(final int dimension) {
            return structure[dimension];
        }

        double totalEnergy() {
            return potentialEnergy + kineticEnergy;
        }

        double potentialEnergy() {
            return potentialEnergy;
        }

        boolean hasKineticEnergyAtMost(final double threshold) {
            return kineticEnergy <= threshold;
        }

        boolean isStagnating(final int decompositionThreshold) {
            return collisionCount - bestCollisionCount > decompositionThreshold;
        }

        double[] bestStructure() {
            return minStructure.clone();
        }

        double bestPotentialEnergy() {
            return minPotentialEnergy;
        }
    }

    private record PopulationStats(double averageFitness, double standardDeviation) {
    }

    public static class CROExample {
        public static void main() {
            ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
            int dimensions = 10;
            double[] min = new double[dimensions];
            double[] max = new double[dimensions];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            CroConfig config = CroConfig.builder()
                    .popSize(30)
                    .maxIterations(2000)
                    .dimensions(dimensions)
                    .minBounds(min)
                    .maxBounds(max)
                    .kelossRate(0.2)
                    .moleColl(0.2)
                    .decThres(20)
                    .synThres(1.0)
                    .initialKE(100.0)
                    .enBuff(100.0)
                    .stepSize(0.1)
                    .seed(12345L)
                    .build();

            ChemicalReactionOptimization cro =
                    new ChemicalReactionOptimization(config, sphere);
            cro.optimize();
        }

        public static void main(String[] args) {
            main();
        }
    }

    public static class AckleyLoggingExample {
        /**
         * Runs the required Aufgabe 4 Ackley 2D scenario and writes
         * {@code algorithm_run.log} in the current working directory.
         *
         * @param args ignored command-line arguments
         * @throws java.io.UncheckedIOException if the log file cannot be written
         */
        public static void main(String[] args) {
            double[] min = {AckleyFunction.LOWER_BOUND, AckleyFunction.LOWER_BOUND};
            double[] max = {AckleyFunction.UPPER_BOUND, AckleyFunction.UPPER_BOUND};

            CroConfig config = CroConfig.builder()
                    .popSize(20)
                    .maxIterations(100)
                    .dimensions(2)
                    .minBounds(min)
                    .maxBounds(max)
                    .kelossRate(0.2)
                    .moleColl(0.25)
                    .decThres(20)
                    .synThres(1.0)
                    .initialKE(50.0)
                    .enBuff(50.0)
                    .stepSize(0.4)
                    .seed(20260507L)
                    .loggingEnabled(true)
                    .logPath("algorithm_run.log")
                    .knownOptimum(new double[]{0.0, 0.0})
                    .build();

            new ChemicalReactionOptimization(config, new AckleyFunction()).optimize();
        }
    }
}
