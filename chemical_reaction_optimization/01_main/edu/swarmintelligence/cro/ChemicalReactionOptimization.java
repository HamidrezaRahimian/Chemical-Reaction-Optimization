package edu.swarmintelligence.cro;

import lombok.*;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Chemical Reaction Optimization (CRO) Algorithm – strict implementation
 * of Lam & Li (2010). Energy conservation is guaranteed in every reaction.
 * <p>
 * Architecture:
 * <ul>
 *   <li>Java 25 LTS: leverages {@code Math.clamp}, records, compact constructors.</li>
 *   <li>Lombok: eliminates boilerplate; no experimental features.</li>
 *   <li>PRNG: {@code L128X256MixRandom} for reproducible, high‑quality randomness.</li>
 *   <li>Immutability: configuration is a validated record.</li>
 * </ul>
 * </p>
 *
 * @see <a href="https://doi.org/10.1109/TEVC.2010.2045390">Lam & Li (2010)</a>
 */
@Slf4j
public class ChemicalReactionOptimization {
    private final CroConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final List<Molecule> population;
    private double buffer;
    private double[] globalBestStructure;
    private double globalBestPE = Double.POSITIVE_INFINITY;

    /**
     * Creates an optimizer, initializes the molecular population,
     * and immediately records the initial global best.
     */
    public ChemicalReactionOptimization(final CroConfig config,
                                        final ObjectiveFunction function) {
        this.config = Objects.requireNonNull(config);
        this.function = Objects.requireNonNull(function);
        this.random = RandomGeneratorFactory.of("L128X256MixRandom")
                .create(config.seed());
        this.population = new ArrayList<>(config.popSize());
        this.buffer = config.enBuff();
        initializePopulation();
    }

    /* ---------------------------------------------------------------- */
    /*  Phase 0 – Initialisation                                       */
    /* ---------------------------------------------------------------- */

    private void initializePopulation() {
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();
        final int n = config.dimensions();

        for (int i = 0; i < config.popSize(); i++) {
            var mol = createRandomMolecule(n, min, max, config.initialKE());
            population.add(mol);
            if (mol.getPotentialEnergy() < globalBestPE) {
                globalBestPE = mol.getPotentialEnergy();
                globalBestStructure = mol.getStructure().clone();
            }
        }
        log.debug("Population initialized. Start best PE = {}", globalBestPE);
    }

    /* ---------------------------------------------------------------- */
    /*  Main optimisation loop                                          */
    /* ---------------------------------------------------------------- */

    /**
     * Executes the CRO iteration loop.
     *
     * @return a clone of the global‑best structure after all iterations
     */
    public double[] optimize() {
        final int n = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        for (int t = 1; t <= config.maxIterations(); t++) {
            final boolean bimolecular = random.nextDouble() < config.moleColl();

            if (bimolecular && population.size() >= 2) {
                int i1 = random.nextInt(population.size());
                int i2 = random.nextInt(population.size());
                while (i2 == i1) {
                    i2 = random.nextInt(population.size());
                }
                var m1 = population.get(i1);
                var m2 = population.get(i2);

                if (m1.getKineticEnergy() <= config.synThres() &&
                        m2.getKineticEnergy() <= config.synThres()) {
                    performSynthesis(m1, m2, i1, i2, n);
                } else {
                    performInterMolecularCollision(m1, m2, n, min, max);
                }
            } else {
                int idx = random.nextInt(population.size());
                var mol = population.get(idx);

                if (mol.getNumHit() >= config.decThres()) {
                    performDecomposition(mol, idx, n, min, max);
                } else {
                    performOnWallCollision(mol, n, min, max);
                }
            }

            ensureMinimumPopulation(n, min, max);
            updateGlobalBest();   // includes freshly injected molecules

            if (t % 100 == 0 || t == config.maxIterations()) {
                log.info("Iteration {}/{} : best PE = {}, pop size = {}",
                        t, config.maxIterations(), globalBestPE, population.size());
            }
        }

        log.info("Optimisation finished. Final best PE = {}", globalBestPE);
        return globalBestStructure.clone();
    }

    /* ---------------------------------------------------------------- */
    /*  Reaction 1 – On‑Wall Ineffective Collision                     */
    /* ---------------------------------------------------------------- */

    /**
     * Energy conservation:
     * <pre>
     *   if PE' ≤ PE + KE  →  KE' = (PE + KE − PE')·q
     *                        buffer ← buffer + (PE + KE − PE')·(1 − q)
     *                        q ∈ [KELossRate, 1]
     *   else               →  hits++
     * </pre>
     */
    private void performOnWallCollision(final Molecule mol, final int n,
                                        final double[] min, final double[] max) {
        var newStruct = perturbStructure(mol.getStructure(), n, min, max);
        double newPE = function.evaluate(newStruct);

        if (!Double.isFinite(newPE)) {
            mol.setNumHit(mol.getNumHit() + 1);
            return;
        }

        if (newPE <= mol.getPotentialEnergy() + mol.getKineticEnergy()) {
            double delta = mol.getPotentialEnergy() + mol.getKineticEnergy() - newPE;
            double q = config.kelossRate() +
                    random.nextDouble() * (1.0 - config.kelossRate());
            updateMoleculeState(mol, newStruct, newPE, delta * q);
            buffer += delta * (1.0 - q);
        } else {
            mol.setNumHit(mol.getNumHit() + 1);
        }
    }

    /* ---------------------------------------------------------------- */
    /*  Reaction 2 – Decomposition                                      */
    /* ---------------------------------------------------------------- */

    /**
     * Lam & Li, Algorithm 2.
     * If PE + KE ≥ PE₁′ + PE₂′ → split surplus randomly.
     * Else, take deficit from buffer (or fail).
     */
    private void performDecomposition(final Molecule mol, final int idx,
                                      final int n, final double[] min,
                                      final double[] max) {
        var struct1 = perturbStructure(mol.getStructure(), n, min, max);
        var struct2 = perturbStructure(mol.getStructure(), n, min, max);

        double pe1 = function.evaluate(struct1);
        double pe2 = function.evaluate(struct2);

        if (!Double.isFinite(pe1) || !Double.isFinite(pe2)) {
            mol.setNumHit(mol.getNumHit() + 1);
            return;
        }

        final double peBefore = mol.getPotentialEnergy();
        final double keBefore = mol.getKineticEnergy();

        if (peBefore + keBefore >= pe1 + pe2) {
            double surplus = peBefore + keBefore - (pe1 + pe2);
            double ke1 = random.nextDouble() * surplus;
            population.set(idx, new Molecule(n, struct1, pe1, ke1));
            population.add(new Molecule(n, struct2, pe2, surplus - ke1));
        } else {
            double deficit = pe1 + pe2 - peBefore - keBefore;
            if (buffer >= deficit) {
                buffer -= deficit;
                population.set(idx, new Molecule(n, struct1, pe1, 0.0));
                population.add(new Molecule(n, struct2, pe2, 0.0));
            } else {
                mol.setNumHit(mol.getNumHit() + 1);
            }
        }
    }

    /* ---------------------------------------------------------------- */
    /*  Reaction 3 – Inter‑Molecular Ineffective Collision            */
    /* ---------------------------------------------------------------- */

    /**
     * Condition: PE₁′+PE₂′ ≤ PE₁+KE₁+PE₂+KE₂.
     * Surplus is split randomly; buffer unchanged.
     */
    private void performInterMolecularCollision(final Molecule m1,
                                                final Molecule m2,
                                                final int n,
                                                final double[] min,
                                                final double[] max) {
        var newStruct1 = perturbStructure(m1.getStructure(), n, min, max);
        var newStruct2 = perturbStructure(m2.getStructure(), n, min, max);

        double newPE1 = function.evaluate(newStruct1);
        double newPE2 = function.evaluate(newStruct2);

        if (!Double.isFinite(newPE1) || !Double.isFinite(newPE2)) {
            m1.setNumHit(m1.getNumHit() + 1);
            m2.setNumHit(m2.getNumHit() + 1);
            return;
        }

        double totalBefore = m1.getPotentialEnergy() + m1.getKineticEnergy()
                + m2.getPotentialEnergy() + m2.getKineticEnergy();
        double totalAfter = newPE1 + newPE2;

        if (totalAfter <= totalBefore) {
            double surplus = totalBefore - totalAfter;
            double ke1 = random.nextDouble() * surplus;
            updateMoleculeState(m1, newStruct1, newPE1, ke1);
            updateMoleculeState(m2, newStruct2, newPE2, surplus - ke1);
        } else {
            m1.setNumHit(m1.getNumHit() + 1);
            m2.setNumHit(m2.getNumHit() + 1);
        }
    }

    /* ---------------------------------------------------------------- */
    /*  Reaction 4 – Synthesis                                          */
    /* ---------------------------------------------------------------- */

    /**
     * Fuses two molecules: {@code ω′ = ω₁ + r ⊙ (ω₂ − ω₁)}}.
     * Energy condition: PE₁+KE₁+PE₂+KE₂ ≥ PE′.
     */
    private void performSynthesis(final Molecule m1, final Molecule m2,
                                  final int idx1, final int idx2, final int n) {
        var newStruct = new double[n];
        for (int d = 0; d < n; d++) {
            double r = random.nextDouble();
            newStruct[d] = Math.clamp(
                    m1.getStructure()[d] + r * (m2.getStructure()[d] - m1.getStructure()[d]),
                    config.minBounds()[d], config.maxBounds()[d]);
        }

        double newPE = function.evaluate(newStruct);
        if (!Double.isFinite(newPE)) {
            m1.setNumHit(m1.getNumHit() + 1);
            m2.setNumHit(m2.getNumHit() + 1);
            return;
        }

        double totalBefore = m1.getPotentialEnergy() + m1.getKineticEnergy()
                + m2.getPotentialEnergy() + m2.getKineticEnergy();
        double newKE = totalBefore - newPE;

        if (newKE >= 0) {
            // Remove the two molecules (larger index first to keep indices valid)
            int first = Math.max(idx1, idx2);
            int second = Math.min(idx1, idx2);
            population.remove(first);
            population.remove(second);
            population.add(new Molecule(n, newStruct, newPE, newKE));
        } else {
            m1.setNumHit(m1.getNumHit() + 1);
            m2.setNumHit(m2.getNumHit() + 1);
        }
    }

    /* ---------------------------------------------------------------- */
    /*  Utilities                                                        */
    /* ---------------------------------------------------------------- */

    private void updateMoleculeState(final Molecule mol,
                                     final double[] newStruct,
                                     final double newPE, final double newKE) {
        mol.setPotentialEnergy(newPE);
        mol.setKineticEnergy(newKE);
        System.arraycopy(newStruct, 0, mol.getStructure(), 0, newStruct.length);
        mol.setNumHit(0);
        mol.updateMinRecord();
    }

    private double[] perturbStructure(final double[] structure, final int n,
                                      final double[] min, final double[] max) {
        var newStructure = new double[n];
        double step = config.stepSize();
        Arrays.setAll(newStructure, d ->
                Math.clamp(structure[d] + random.nextGaussian() * step,
                        min[d], max[d]));
        return newStructure;
    }

    private void updateGlobalBest() {
        for (var mol : population) {
            if (mol.getPotentialEnergy() < globalBestPE) {
                globalBestPE = mol.getPotentialEnergy();
                globalBestStructure = mol.getStructure().clone();
            }
        }
    }

    private void ensureMinimumPopulation(final int n, final double[] min,
                                         final double[] max) {
        while (population.size() < 2) {
            population.add(createRandomMolecule(n, min, max, config.initialKE()));
        }
    }

    private Molecule createRandomMolecule(final int n,
                                          final double[] min,
                                          final double[] max,
                                          final double initialKE) {
        final int MAX_RETRIES = 1000;
        for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
            var x = new double[n];
            for (int d = 0; d < n; d++) {
                x[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            double pe = function.evaluate(x);
            if (Double.isFinite(pe)) {
                return new Molecule(n, x, pe, initialKE);
            }
        }
        throw new IllegalStateException(
                "No feasible molecule found after " + MAX_RETRIES + " attempts.");
    }

    public double getTotalSystemEnergy() {
        double total = buffer;
        for (var mol : population) {
            total += mol.getPotentialEnergy() + mol.getKineticEnergy();
        }
        return total;
    }

    public double getBestPE() {
        return globalBestPE;
    }

    /* ---------------------------------------------------------------- */
    /*  Embedded types                                                   */
    /* ---------------------------------------------------------------- */

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable CRO configuration.  All arrays are defensively copied.
     */
    @Builder
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
            if (popSize < 2) throw new IllegalArgumentException(
                    "PopSize must be ≥ 2");
            if (maxIterations <= 0) throw new IllegalArgumentException(
                    "MaxIterations > 0");
            if (dimensions <= 0) throw new IllegalArgumentException(
                    "Dimensions > 0");
            if (minBounds.length != dimensions ||
                    maxBounds.length != dimensions) throw new IllegalArgumentException(
                    "Bounds length must match dimensions");
            if (kelossRate < 0.0 || kelossRate > 1.0) throw new IllegalArgumentException(
                    "KELossRate in [0,1]");
            if (moleColl < 0.0 || moleColl > 1.0) throw new IllegalArgumentException(
                    "MoleColl in [0,1]");
            if (decThres <= 0) throw new IllegalArgumentException(
                    "DecThres > 0");
            if (synThres < 0.0) throw new IllegalArgumentException(
                    "SynThres ≥ 0");
            if (initialKE <= 0) throw new IllegalArgumentException(
                    "InitialKE > 0");
            if (enBuff < 0.0) throw new IllegalArgumentException(
                    "EnBuff ≥ 0");
            if (stepSize <= 0) throw new IllegalArgumentException(
                    "StepSize > 0");
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }
    }

    /**
     * A molecule with its structure and energy state.
     * Equals/HashCode ignore the large double[] fields to avoid performance hits.
     */
    @Getter
    @Setter
    @EqualsAndHashCode(exclude = {"structure", "minStructure"})
    @ToString(exclude = {"structure", "minStructure"})
    private static class Molecule {
        private final double[] structure;
        private final double[] minStructure;
        private double potentialEnergy;
        private double kineticEnergy;
        private double minPE;
        private int numHit;

        Molecule(final int dimensions, final double[] structure,
                 final double potentialEnergy, final double kineticEnergy) {
            this.structure = structure.clone();
            this.potentialEnergy = potentialEnergy;
            this.kineticEnergy = kineticEnergy;
            this.minStructure = structure.clone();
            this.minPE = potentialEnergy;
            this.numHit = 0;
        }

        void updateMinRecord() {
            if (potentialEnergy < minPE) {
                minPE = potentialEnergy;
                System.arraycopy(structure, 0, minStructure, 0, structure.length);
            }
        }
    }

    /* ---------------------------------------------------------------- */
    /*  Quick demonstration                                              */
    /* ---------------------------------------------------------------- */

    @Slf4j
    public static class CROExample {
        public static void main() {
            log.info("CRO Example – Sphere function minimisation");

            ObjectiveFunction sphere = x ->
                    Arrays.stream(x).map(v -> v * v).sum();

            int dim = 10;
            var min = new double[dim];
            var max = new double[dim];
            Arrays.fill(min, -5.12);
            Arrays.fill(max, 5.12);

            var config = CroConfig.builder()
                    .popSize(10)
                    .maxIterations(1000)
                    .dimensions(dim)
                    .minBounds(min)
                    .maxBounds(max)
                    .kelossRate(0.2)
                    .moleColl(0.2)
                    .decThres(10)
                    .synThres(1.0)
                    .initialKE(1000.0)
                    .enBuff(1000.0)
                    .stepSize(0.1)
                    .seed(12345L)
                    .build();

            var cro = new ChemicalReactionOptimization(config, sphere);
            double[] best = cro.optimize();

            log.info("Best solution: {}", Arrays.toString(best));
            log.info("Best PE: {}", cro.getBestPE());
        }
    }
}