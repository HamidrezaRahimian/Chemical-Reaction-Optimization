package edu.swarmintelligence.cro;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

    private final List<Molecule> population;
    private double buffer;
    private double[] globalBestStructure;
    private double globalBestPotentialEnergy;

    public ChemicalReactionOptimization(final CroConfig config,
                                        final ObjectiveFunction function) {
        this.config = Objects.requireNonNull(config, "config");
        this.function = Objects.requireNonNull(function, "function");
        this.random = new Random(config.seed());
        this.minBounds = config.minBounds();
        this.maxBounds = config.maxBounds();
        this.population = new ArrayList<>(config.popSize());
        this.buffer = config.enBuff();
        this.globalBestPotentialEnergy = Double.POSITIVE_INFINITY;
        initializePopulation();
    }

    private void initializePopulation() {
        for (int i = 0; i < config.popSize(); i++) {
            population.add(createRandomMolecule(config.initialKE()));
        }
        updateGlobalBest();
        if (globalBestStructure == null) {
            throw new IllegalStateException("No feasible initial molecule found.");
        }
    }

    /**
     * Runs the configured number of CRO reactions and returns the best structure
     * found. The returned array is a defensive copy and can be changed by the
     * caller without affecting the optimizer state.
     */
    public double[] optimize() {
        for (int iteration = 0; iteration < config.maxIterations(); iteration++) {
            if (shouldUseBimolecularReaction()) {
                reactWithTwoMolecules();
            } else {
                reactWithOneMolecule();
            }
            updateGlobalBest();
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

    private void replaceWithDecompositionProducts(final int index,
                                                  final double[] firstStructure,
                                                  final double firstPotentialEnergy,
                                                  final double[] secondStructure,
                                                  final double secondPotentialEnergy,
                                                  final double surplusEnergy) {
        double firstKineticEnergy = random.nextDouble() * surplusEnergy;
        population.set(index, new Molecule(firstStructure, firstPotentialEnergy, firstKineticEnergy));
        population.add(new Molecule(secondStructure, secondPotentialEnergy,
                surplusEnergy - firstKineticEnergy));
    }

    /**
     * Inter-molecular ineffective collision: two molecules are perturbed
     * independently and accepted when their combined energy can pay for both new
     * structures.
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
            population.set(lowIndex, new Molecule(candidateStructure, candidatePotentialEnergy,
                    availableEnergy - candidatePotentialEnergy));
        } else {
            first.registerCollision();
            second.registerCollision();
        }
    }

    private double[] perturb(final double[] structure) {
        double[] result = new double[config.dimensions()];
        for (int d = 0; d < result.length; d++) {
            result[d] = clamp(structure[d] + random.nextGaussian() * config.stepSize(),
                    minBounds[d], maxBounds[d]);
        }
        return result;
    }

    private Molecule createRandomMolecule(final double kineticEnergy) {
        for (int attempt = 0; attempt < MAX_INITIALIZATION_ATTEMPTS; attempt++) {
            double[] structure = randomStructure();
            double potentialEnergy = evaluate(structure);
            if (Double.isFinite(potentialEnergy)) {
                return new Molecule(structure, potentialEnergy, kineticEnergy);
            }
        }
        throw new IllegalStateException("No feasible molecule found after "
                + MAX_INITIALIZATION_ATTEMPTS + " attempts.");
    }

    private double[] randomStructure() {
        double[] structure = new double[config.dimensions()];
        for (int d = 0; d < structure.length; d++) {
            structure[d] = minBounds[d] + random.nextDouble() * (maxBounds[d] - minBounds[d]);
        }
        return structure;
    }

    private double evaluate(final double[] structure) {
        return function.evaluate(structure.clone());
    }

    private static boolean hasEnoughEnergy(final double availableEnergy,
                                           final double requiredPotentialEnergy) {
        return requiredPotentialEnergy <= availableEnergy;
    }

    private void updateGlobalBest() {
        for (Molecule molecule : population) {
            if (molecule.bestPotentialEnergy() < globalBestPotentialEnergy) {
                globalBestPotentialEnergy = molecule.bestPotentialEnergy();
                globalBestStructure = molecule.bestStructure();
            }
        }
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
        double evaluate(double[] x);
    }

    /**
     * Configuration for bounded CRO minimization. Array parameters are copied
     * defensively on construction and when accessed.
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
            long seed
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
        }

        public double[] minBounds() {
            return minBounds.clone();
        }

        public double[] maxBounds() {
            return maxBounds.clone();
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

            public CroConfig build() {
                return new CroConfig(popSize, maxIterations, dimensions, minBounds, maxBounds,
                        kelossRate, moleColl, decThres, synThres, initialKE, enBuff,
                        stepSize, seed);
            }
        }
    }

    private static final class Molecule {
        private double[] structure;
        private double potentialEnergy;
        private double kineticEnergy;
        private double[] minStructure;
        private double minPotentialEnergy;
        private int collisionCount;
        private int bestCollisionCount;

        Molecule(final double[] structure,
                 final double potentialEnergy,
                 final double kineticEnergy) {
            this.structure = structure.clone();
            this.potentialEnergy = potentialEnergy;
            this.kineticEnergy = kineticEnergy;
            this.minStructure = structure.clone();
            this.minPotentialEnergy = potentialEnergy;
            this.collisionCount = 0;
            this.bestCollisionCount = 0;
        }

        void replaceState(final double[] newStructure,
                          final double newPotentialEnergy,
                          final double newKineticEnergy) {
            structure = newStructure.clone();
            potentialEnergy = newPotentialEnergy;
            kineticEnergy = Math.max(0.0, newKineticEnergy);
            registerCollision();
        }

        void registerCollision() {
            collisionCount++;
            if (potentialEnergy < minPotentialEnergy) {
                minPotentialEnergy = potentialEnergy;
                minStructure = structure.clone();
                bestCollisionCount = collisionCount;
            }
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
}
