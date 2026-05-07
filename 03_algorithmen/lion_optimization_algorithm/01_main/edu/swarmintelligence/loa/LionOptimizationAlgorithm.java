package edu.swarmintelligence.loa;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.IntStream;

/**
 * Lion Optimization Algorithm (LOA) – strictly according to
 * <i>Yazdani &amp; Jolai (2016)</i>.
 *
 * <p><b>Journal of Computational Design and Engineering 3(1), 24–36.</b></p>
 *
 * <h4>Algorithm structure</h4>
 * <ol>
 *   <li><b>Initialisation</b> – lions randomly placed; prides formed; one male per pride.</li>
 *   <li><b>Hunting</b> – females move towards pride prey and territory center (Eq. 5).</li>
 *   <li><b>Roaming</b> – resident male random walk (10 % of search range).</li>
 *   <li><b>Territorial defense</b> – random nomad male may take over pride.</li>
 *   <li><b>Mating</b> – two‑point crossover + mutation, two cubs per lioness.</li>
 *   <li><b>Cub growth</b> – cub moves towards mother or father with probability α (0.5).</li>
 *   <li><b>Nomad movement</b> – female towards best nomad (Eq. 14), male towards random nomad.</li>
 *   <li><b>Takeover</b> – weakest pride member replaced by best same‑gender nomad.</li>
 *   <li><b>Migration</b> – S % of females randomly exchanged between prides/nomads.</li>
 *   <li><b>Population control</b> – excess lions removed (weakest first).</li>
 * </ol>
 *
 * <p>All operators are greedy: a new position is accepted only if it yields a better
 * fitness. The algorithm returns the best solution found over all iterations.</p>
 *
 * <p>Uses Java 21+ features ({@code Math.clamp}, {@code RandomGenerator}) and
 * Lombok for readability. Designed for students of Applied Informatics.</p>
 */
@Slf4j
public final class LionOptimizationAlgorithm {
    /* –– Constants –– */
    private static final double ROAMING_PERCENT = 0.1;   // roaming step as percentage of range
    private static final double MATING_RATE = 0.4;       // fraction of females mated per male
    private static final double MUTATION_PROB = 0.1;     // per‑gene mutation probability
    private static final double CUB_GROWTH_PROB = 0.5;   // α: move towards mother

    final List<Pride> prides;          // package‑private for tests
    final List<Lion> nomads;           // package‑private for tests

    private final int dimensions;
    private final double[] minBounds, maxBounds;
    private final int populationSize;
    private final int maxIterations;
    private final int nPride;
    private final double pPride;
    private final double sFemale;
    private final double migrationRate;
    private final RandomGenerator rng;
    private final ObjectiveFunction objective;

    double globalBestFitness = Double.MAX_VALUE;
    double[] globalBestPosition;

    /* –– Construction –– */
    public LionOptimizationAlgorithm(final Config config, final ObjectiveFunction objective) {
        this.dimensions = config.dimensions;
        this.minBounds = config.minBounds;
        this.maxBounds = config.maxBounds;
        this.populationSize = config.populationSize;
        this.maxIterations = config.maxIterations;
        this.nPride = config.nPride;
        this.pPride = config.pPride;
        this.sFemale = config.sFemale;
        this.migrationRate = config.migrationRate;
        this.rng = RandomGeneratorFactory.of("L64X128MixRandom").create(config.seed);
        this.objective = objective;
        this.prides = new ArrayList<>(nPride);
        this.nomads = new ArrayList<>();

        initPrides();
        initPopulation();
    }

    // –– Demonstration ––
    static void demo() {
        log.info("LOA example – sphere function");
        final ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();
        final int dims = 10;
        final double[] maxB = new double[dims];
        Arrays.fill(maxB, 5.12);
        final double[] minB = Arrays.stream(maxB).map(v -> -v).toArray();

        final Config cfg = Config.of(50, 500, dims, minB, maxB, 12345L);
        final LionOptimizationAlgorithm loa = new LionOptimizationAlgorithm(cfg, sphere);
        final double[] best = loa.optimize();
        log.info("Best solution: {}", Arrays.toString(best));
        log.info("Best objective value: {}", sphere.evaluate(best));
    }

    // –– Initialisation ––
    private void initPrides() {
        IntStream.range(0, nPride).mapToObj(Pride::new).forEach(prides::add);
    }

    private void initPopulation() {
        final int n = dimensions;
        final List<Lion> all = new ArrayList<>(populationSize);
        for (int i = 0; i < populationSize; i++) {
            final double[] pos = new double[n];
            for (int d = 0; d < n; d++) {
                pos[d] = rng.nextDouble(minBounds[d], maxBounds[d]);
            }
            all.add(new Lion(pos, objective.evaluate(pos)));
        }
        all.sort(Comparator.comparingDouble(Lion::getFitness));

        final int prideCount = (int) Math.floor(populationSize * pPride);
        // round‑robin distribution to prides
        for (int i = 0; i < prideCount; i++) {
            final Lion lion = all.get(i);
            lion.role = Role.PRIDE;
            lion.prideId = i % nPride;
            prides.get(lion.prideId).members.add(lion);
        }
        for (int i = prideCount; i < populationSize; i++) {
            final Lion lion = all.get(i);
            lion.role = Role.NOMAD;
            nomads.add(lion);
        }

        // assign gender: each pride gets exactly one male (its fittest member)
        for (final Pride pride : prides) {
            final List<Lion> members = pride.members;
            if (!members.isEmpty()) {
                members.getFirst().gender = Gender.MALE;
                members.stream().skip(1).forEach(l -> l.gender = Gender.FEMALE);
            }
        }

        final int nomadFemales = (int) Math.ceil(nomads.size() * sFemale);
        for (int i = 0; i < nomads.size(); i++) {
            nomads.get(i).gender = (i < nomadFemales) ? Gender.FEMALE : Gender.MALE;
        }

        prides.forEach(Pride::updatePrey);
        updateGlobalBest();
        log.debug("Initialised {} prides, {} nomads; global best = {}", prides.size(), nomads.size(), globalBestFitness);
    }

    // –– Main loop ––
    public double[] optimize() {
        log.info("LOA optimisation start ({} iterations)", maxIterations);
        for (int iter = 1; iter <= maxIterations; iter++) {
            for (final Pride pride : prides) {
                final double[] centre = pride.territoryCentre();

                // 1. Hunting: females
                pride.members.stream()
                        .filter(l -> l.gender == Gender.FEMALE)
                        .forEach(l -> hunt(l, pride.prey, centre));
                pride.updatePrey();

                // 2. Roaming: resident male
                pride.members.stream()
                        .filter(l -> l.gender == Gender.MALE)
                        .findAny()
                        .ifPresent(this::roam);
                pride.updatePrey();

                // 3. Territorial defense
                defend(pride);
                pride.updatePrey();
            }

            // 4. Mating → cubs become nomads after growth
            final List<Cub> cubs = new ArrayList<>();
            for (final Pride pride : prides) {
                cubs.addAll(mate(pride));
            }
            cubs.forEach(this::growCub);
            cubs.forEach(c -> c.lion.fitness = objective.evaluate(c.lion.position()));
            nomads.addAll(cubs.stream().map(Cub::lion).toList());

            // 5. Nomad movement (Eq. 14)
            final Lion bestNomad = nomads.stream().min(Comparator.comparingDouble(Lion::getFitness)).orElse(null);
            nomads.forEach(nm -> moveNomad(nm, bestNomad));

            // 6. Equilibrium
            takeover();
            migrateFemales();
            populationControl();

            // Ensure pride historical bests are up‑to‑date after structural changes
            prides.forEach(Pride::updatePrey);
            updateGlobalBest();

            if (iter % 100 == 0) {
                log.info("Iteration {}: best fitness = {}", iter, globalBestFitness);
            }
        }
        log.info("Optimisation finished. Best fitness = {}", globalBestFitness);
        return globalBestPosition.clone();
    }

    // –– Hunting (Eq. 5) ––
    private void hunt(final Lion lion, final double[] prey, final double[] centre) {
        final double[] pos = lion.position();
        final double[] newPos = new double[dimensions];
        final double r1 = rng.nextDouble(), r2 = rng.nextDouble();
        for (int d = 0; d < dimensions; d++) {
            newPos[d] = Math.clamp(pos[d] + r1 * (prey[d] - pos[d]) + r2 * (centre[d] - pos[d]),
                    minBounds[d], maxBounds[d]);
        }
        tryImprove(lion, newPos);
    }

    // –– Roaming: resident male local random walk ––
    private void roam(final Lion male) {
        final double[] pos = male.position();
        final double[] newPos = new double[dimensions];
        for (int d = 0; d < dimensions; d++) {
            final double step = rng.nextDouble(-1.0, 1.0) * ROAMING_PERCENT * (maxBounds[d] - minBounds[d]);
            newPos[d] = Math.clamp(pos[d] + step, minBounds[d], maxBounds[d]);
        }
        tryImprove(male, newPos);
    }

    // –– Territorial defense: random nomad male challenges resident male ––
    private void defend(final Pride pride) {
        final Lion male = pride.members.stream().filter(l -> l.gender == Gender.MALE).findAny().orElse(null);
        if (male == null) return;
        final List<Lion> challengers = nomads.stream().filter(l -> l.gender == Gender.MALE).toList();
        if (challengers.isEmpty()) return;
        final Lion challenger = challengers.get(rng.nextInt(challengers.size()));

        if (challenger.fitness < male.fitness) {
            nomads.remove(challenger);
            pride.members.remove(male);
            male.role = Role.NOMAD;
            male.prideId = -1;
            nomads.add(male);
            challenger.role = Role.PRIDE;
            challenger.prideId = pride.id;
            pride.members.add(challenger);
        }
    }

    // –– Mating: two‑point crossover + uniform mutation ––
    private List<Cub> mate(final Pride pride) {
        final List<Cub> cubs = new ArrayList<>();
        final Lion male = pride.members.stream().filter(l -> l.gender == Gender.MALE).findAny().orElse(null);
        final List<Lion> females = pride.members.stream().filter(l -> l.gender == Gender.FEMALE).toList();
        if (male == null || females.isEmpty()) return cubs;

        final int numMothers = Math.max(1, (int) Math.ceil(MATING_RATE * females.size()));
        final List<Lion> shuffled = new ArrayList<>(females);
        Collections.shuffle(shuffled, rng);
        for (int k = 0; k < numMothers && k < shuffled.size(); k++) {
            final Lion mother = shuffled.get(k);
            final Lion cub1 = new Lion(new double[dimensions], Double.MAX_VALUE);
            final Lion cub2 = new Lion(new double[dimensions], Double.MAX_VALUE);

            int cp1 = rng.nextInt(dimensions);
            int cp2 = rng.nextInt(dimensions);
            if (cp1 > cp2) {
                final int tmp = cp1;
                cp1 = cp2;
                cp2 = tmp;
            }
            final double[] mPos = male.position(), fPos = mother.position();
            for (int d = 0; d < dimensions; d++) {
                if (cp1 <= d && d <= cp2) {
                    cub1.position()[d] = mPos[d];
                    cub2.position()[d] = fPos[d];
                } else {
                    cub1.position()[d] = fPos[d];
                    cub2.position()[d] = mPos[d];
                }
            }
            mutate(cub1);
            mutate(cub2);

            cub1.role = Role.NOMAD;
            cub1.gender = rng.nextBoolean() ? Gender.MALE : Gender.FEMALE;
            cub2.role = Role.NOMAD;
            cub2.gender = rng.nextBoolean() ? Gender.MALE : Gender.FEMALE;

            cubs.add(new Cub(cub1, mother, male));
            cubs.add(new Cub(cub2, mother, male));
        }
        return cubs;
    }

    private void mutate(final Lion cub) {
        final double[] pos = cub.position();
        for (int d = 0; d < dimensions; d++) {
            if (rng.nextDouble() < MUTATION_PROB) {
                pos[d] = rng.nextDouble(minBounds[d], maxBounds[d]);
            }
        }
    }

    // –– Cub growth ––
    private void growCub(final Cub cub) {
        final Lion lion = cub.lion;
        final Lion parent = rng.nextDouble() < CUB_GROWTH_PROB ? cub.mother : cub.father;
        final double[] pos = lion.position(), parPos = parent.position();
        final double[] newPos = new double[dimensions];
        for (int d = 0; d < dimensions; d++) {
            newPos[d] = pos[d] + rng.nextDouble() * (parPos[d] - pos[d]);
            newPos[d] = Math.clamp(newPos[d], minBounds[d], maxBounds[d]);
        }
        tryImprove(lion, newPos);
    }

    // –– Nomad movement (Eq. 14 with rand ∈ [0,1]) ––
    private void moveNomad(final Lion nomad, final Lion bestNomad) {
        final double[] pos = nomad.position();
        final double[] target;
        if (nomad.gender == Gender.FEMALE && bestNomad != null) {
            target = bestNomad.position();
        } else {
            final List<Lion> others = nomads.stream().filter(l -> l != nomad).toList();
            target = others.isEmpty() ? pos : others.get(rng.nextInt(others.size())).position();
        }
        final double[] newPos = new double[dimensions];
        for (int d = 0; d < dimensions; d++) {
            final double step = rng.nextDouble() * (target[d] - pos[d]); // strictly 0..1
            newPos[d] = Math.clamp(pos[d] + step, minBounds[d], maxBounds[d]);
        }
        tryImprove(nomad, newPos);
    }

    // –– Takeover ––
    private void takeover() {
        for (final Pride pride : prides) {
            if (pride.members.isEmpty()) continue;
            final Lion weakest = Collections.max(pride.members, Comparator.comparingDouble(Lion::getFitness));
            final Lion bestSameGender = nomads.stream()
                    .filter(n -> n.gender == weakest.gender)
                    .min(Comparator.comparingDouble(Lion::getFitness))
                    .orElse(null);
            if (bestSameGender != null && bestSameGender.fitness < weakest.fitness) {
                nomads.remove(bestSameGender);
                pride.members.remove(weakest);
                weakest.role = Role.NOMAD;
                weakest.prideId = -1;
                nomads.add(weakest);
                bestSameGender.role = Role.PRIDE;
                bestSameGender.prideId = pride.id;
                pride.members.add(bestSameGender);
            }
        }
    }

    // –– Migration (S % of all females randomly exchanged) ––
    private void migrateFemales() {
        final List<Lion> allFemales = new ArrayList<>();
        prides.stream().flatMap(p -> p.members.stream())
                .filter(l -> l.gender == Gender.FEMALE).forEach(allFemales::add);
        nomads.stream().filter(l -> l.gender == Gender.FEMALE).forEach(allFemales::add);

        final int count = (int) Math.round(migrationRate * allFemales.size());
        Collections.shuffle(allFemales, rng);
        for (int i = 0; i < Math.min(count, allFemales.size()); i++) {
            final Lion female = allFemales.get(i);
            if (female.role == Role.PRIDE) {
                final Pride oldPride = prides.get(female.prideId);
                oldPride.members.remove(female);
                if (rng.nextBoolean()) {
                    female.role = Role.NOMAD;
                    female.prideId = -1;
                    nomads.add(female);
                } else {
                    final Pride newPride = prides.get(rng.nextInt(nPride));
                    female.prideId = newPride.id;
                    female.role = Role.PRIDE;
                    newPride.members.add(female);
                }
            } else { // NOMAD
                nomads.remove(female);
                final Pride newPride = prides.get(rng.nextInt(nPride));
                female.role = Role.PRIDE;
                female.prideId = newPride.id;
                newPride.members.add(female);
            }
        }
    }

    // –– Population control ––
    private void populationControl() {
        int total = totalLions();
        while (total > populationSize) {
            final Lion worstNomad = nomads.stream().max(Comparator.comparingDouble(Lion::getFitness)).orElse(null);
            if (worstNomad != null) {
                nomads.remove(worstNomad);
                total--;
                continue;
            }
            final Lion weakestFemale = prides.stream()
                    .filter(p -> p.members.size() > 1)
                    .flatMap(p -> p.members.stream())
                    .filter(l -> l.gender == Gender.FEMALE)
                    .max(Comparator.comparingDouble(Lion::getFitness))
                    .orElse(null);
            if (weakestFemale != null) {
                prides.get(weakestFemale.prideId).members.remove(weakestFemale);
                total--;
            } else {
                break;
            }
        }
    }

    int totalLions() {
        return prides.stream().mapToInt(p -> p.members.size()).sum() + nomads.size();
    }

    private void updateGlobalBest() {
        final List<Lion> all = new ArrayList<>(nomads);
        prides.forEach(p -> all.addAll(p.members));
        all.stream().min(Comparator.comparingDouble(Lion::getFitness))
                .ifPresent(best -> {
                    if (best.fitness < globalBestFitness) {
                        globalBestFitness = best.fitness;
                        globalBestPosition = best.position().clone();
                    }
                });
    }

    // –– Greedy improvement ––
    private void tryImprove(final Lion lion, final double[] candidate) {
        final double f = objective.evaluate(candidate);
        if (f < lion.fitness) {
            lion.fitness = f;
            System.arraycopy(candidate, 0, lion.position(), 0, dimensions);
        }
    }

    // –– Enums and helper records ––
    enum Gender {MALE, FEMALE}

    enum Role {PRIDE, NOMAD}

    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable configuration record.
     */
    public record Config(
            int populationSize,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            int nPride,
            double pPride,
            double sFemale,
            double migrationRate,
            long seed
    ) {
        public Config {
            if (populationSize <= 0) throw new IllegalArgumentException("populationSize > 0");
            if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations > 0");
            if (dimensions <= 0) throw new IllegalArgumentException("dimensions > 0");
            if (minBounds.length != dimensions || maxBounds.length != dimensions) {
                throw new IllegalArgumentException("bounds length == dimensions");
            }
            if (pPride <= 0.0 || pPride > 1.0) throw new IllegalArgumentException("pPride in (0,1]");
            if (sFemale <= 0.0 || sFemale > 1.0) throw new IllegalArgumentException("sFemale in (0,1]");
            if (nPride <= 0) throw new IllegalArgumentException("nPride > 0");
            if (migrationRate < 0.0 || migrationRate > 1.0) {
                throw new IllegalArgumentException("migrationRate in [0,1]");
            }
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();
        }

        public static Config of(final int popSize, final int iters, final int dims,
                                final double[] min, final double[] max, final long seed) {
            return new Config(popSize, iters, dims, min, max, 5, 0.8, 0.8, 0.1, seed);
        }
    }

    private record Cub(Lion lion, Lion mother, Lion father) {
    }

    @Data
    static final class Lion {
        private final double[] position;
        private double fitness;
        private Gender gender;
        private Role role;
        private int prideId = -1;

        Lion(final double[] pos, final double fitness) {
            this.position = pos.clone();
            this.fitness = fitness;
        }

        double[] position() {
            return position;
        }
    }

    @Data
    static final class Pride {
        private final int id;
        private final List<Lion> members = new ArrayList<>();
        private double[] prey;
        private double preyFitness = Double.MAX_VALUE;

        Pride(final int id) {
            this.id = id;
        }

        void updatePrey() {
            if (prey == null) {
                prey = new double[0];
            }
            for (final Lion lion : members) {
                if (lion.fitness < preyFitness) {
                    preyFitness = lion.fitness;
                    if (prey.length != lion.position.length) {
                        prey = new double[lion.position.length];
                    }
                    System.arraycopy(lion.position, 0, prey, 0, prey.length);
                }
            }
        }

        double[] territoryCentre() {
            final int n = prey == null ? 0 : prey.length;
            final double[] centre = new double[n];
            if (members.isEmpty()) return centre;
            for (final Lion lion : members) {
                for (int d = 0; d < n; d++) {
                    centre[d] += lion.position[d];
                }
            }
            for (int d = 0; d < n; d++) {
                centre[d] /= members.size();
            }
            return centre;
        }
    }
}