package edu.swarmintelligence.abc;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.IntStream;

/**
 * <b>Artificial Bee Colony (ABC) algorithm</b> – strict implementation of
 * <i>Karaboga, D. (2005). An idea based on honey bee swarm for numerical
 * optimization. Technical Report TR06, Erciyes University.</i>
 *
 * <p>
 * A population‑based meta‑heuristic that mimics the intelligent foraging
 * behavior of a honey bee colony. The colony consists of three groups acting
 * on the same set of food sources (candidate solutions):
 * <ul>
 *   <li><b>Employed bees</b> exploit a dedicated source and share information.</li>
 *   <li><b>Onlooker bees</b> choose a source via fitness‑proportional roulette
 *       wheel and perform local refinement.</li>
 *   <li><b>Scout bees</b> replace exhausted sources with a random position to
 *       maintain diversity.</li>
 * </ul>
 *
 * <h3>Algorithm summary</h3>
 * <ol>
 *   <li>Initialise {@code SN} food sources uniformly in the search space,
 *       evaluate them, and set trial counters to 0.</li>
 *   <li><b>Employed bee phase:</b> For each source <i>i</i>, choose a random
 *       dimension <i>d</i> and a random partner <i>k</i> ≠ <i>i</i>, generate a
 *       candidate by {@code v_id = x_id + φ · (x_id – x_kd)} with φ ∈ [‑1,1].
 *       Greedily keep the better solution; increment the trial counter on
 *       failure.</li>
 *   <li><b>Onlooker bee phase:</b> Compute selection probabilities
 *       <i>p<sub>i</sub> = fit<sub>i</sub> / Σ fit<sub>j</sub></i> using
 *       Karaboga’s fitness transformation. Each of <i>SN</i> onlookers selects
 *       a source by roulette wheel and performs the same greedy update as an
 *       employed bee.</li>
 *   <li><b>Scout bee phase:</b> If any source has reached the abandonment
 *       limit <i>limit</i>, discard it and create a new random source. At most
 *       one source is reset per cycle (the one with the highest trial counter).</li>
 *   <li>Repeat until the maximum number of iterations is reached.</li>
 * </ol>
 *
 * <p>
 * <b>Parameters:</b> number of food sources <i>SN</i>, abandonment limit
 * <i>limit</i> (usually {@code SN·D}), maximum iterations.
 * The algorithm converges quickly on unimodal functions and handles
 * multimodal landscapes robustly.
 *
 * <p>
 * <b>Design:</b> Uses {@link RandomGenerator L128X256MixRandom} for
 * reproducibility and scientific quality. The configuration is an immutable
 * Lombok {@code @Builder} record. The objective function is a
 * {@link FunctionalInterface}. Self‑contained; call {@link #optimise()} to
 * obtain the best found solution.
 *
 * @see <a href="https://doi.org/10.1007/978-3-540-72950-0_77">
 * Karaboga &amp; Basturk (2007), A powerful and efficient algorithm
 * for numerical function optimization: artificial bee colony (ABC)
 * algorithm, Journal of Global Optimization.</a>
 */
@Slf4j
public final class ArtificialBeeColony {
    private static final String RNG_ALGORITHM = "L128X256MixRandom";

    private final AbcConfig config;
    private final ObjectiveFunction function;
    private final RandomGenerator random;
    private final FoodSource[] sources;

    @Getter
    private double bestFitness = Double.POSITIVE_INFINITY;
    private double[] bestPosition;
    private double[] convergenceHistory;            // best fitness per iteration (0 = initial)

    // Per‑iteration counters for DEBUG logging
    private int employedImprovements;
    private int onlookerImprovements;
    private int scoutEvents;

    /**
     * Creates an ABC optimizer, validates the configuration, and initializes
     * the food source population.
     *
     * @param config   algorithm hyper‑parameters
     * @param function objective function to minimize
     */
    public ArtificialBeeColony(final AbcConfig config, final ObjectiveFunction function) {
        this.config = config;
        this.function = function;

        this.random = RandomGeneratorFactory.of(RNG_ALGORITHM)
                .create(config.seed());
        this.sources = new FoodSource[config.foodSourceCount()];
        initialisePopulation();
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Karaboga's piecewise fitness transformation.
     */
    private static double fitnessTransform(final double objective) {
        return objective >= 0.0
                ? 1.0 / (1.0 + objective)
                : 1.0 + Math.abs(objective);
    }

    private static String sci(final double value) {
        return String.format("%.6e", value);
    }

    // -------------------------------------------------------------------------
    // Initialisation
    // -------------------------------------------------------------------------

    /**
     * Demonstration with the 10‑D Sphere function using standard parameters.
     */
    @SuppressWarnings("unused")
    static void main(final String... args) {
        log.info("Starting Artificial Bee Colony Demo");

        final ObjectiveFunction sphere = x -> Arrays.stream(x).map(v -> v * v).sum();

        final int dim = 10;
        final double[] minB = new double[dim];
        final double[] maxB = new double[dim];
        Arrays.fill(minB, -5.12);
        Arrays.fill(maxB, 5.12);

        final AbcConfig cfg = AbcConfig.builder()
                .foodSourceCount(50)
                .maxIterations(200)
                .dimensions(dim)
                .minBounds(minB)
                .maxBounds(maxB)
                .abandonmentLimit(0)               // 0 ⇒ compute default SN * D
                .seed(42L)
                .build();

        final var abc = new ArtificialBeeColony(cfg, sphere);
        final double[] solution = abc.optimise();

        log.info("Best Solution: {}", Arrays.toString(solution));
        log.info("Best Fitness:  {}", abc.getBestFitness());
    }

    // -------------------------------------------------------------------------
    // Bee phases
    // -------------------------------------------------------------------------

    /**
     * Runs the full optimization loop.
     *
     * @return the best solution vector found
     */
    public double[] optimise() {
        final int maxIter = config.maxIterations();
        convergenceHistory = new double[maxIter + 1];
        convergenceHistory[0] = bestFitness;

        for (int iter = 1; iter <= maxIter; iter++) {
            log.debug("Iteration {} START", iter);
            employedImprovements = 0;
            onlookerImprovements = 0;
            scoutEvents = 0;

            employedBeePhase();
            onlookerBeePhase();
            scoutBeePhase();

            // Update global best
            final FoodSource iterationBest = findBestSource();
            if (iterationBest.getObjectiveValue() < bestFitness) {
                bestFitness = iterationBest.getObjectiveValue();
                bestPosition = iterationBest.getPosition().clone();
            }
            convergenceHistory[iter] = bestFitness;

            if (log.isDebugEnabled()) {
                log.debug("Iteration {} END | employedImprov={}/{} onlookerImprov={}/{} scouts={} best={}",
                        iter, employedImprovements, sources.length,
                        onlookerImprovements, sources.length, scoutEvents,
                        sci(bestFitness));
            }

            if (log.isInfoEnabled() && iter % Math.max(1, maxIter / 10) == 0) {
                log.info("Iteration {}/{} | Best Fitness: {}", iter, maxIter, bestFitness);
            }
        }
        return bestPosition.clone();
    }

    /**
     * Returns a copy of the per‑iteration best fitness history (index 0 = initial).
     */
    public double[] getConvergenceHistory() {
        return convergenceHistory.clone();
    }

    private void initialisePopulation() {
        final int dim = config.dimensions();
        final double[] min = config.minBounds();
        final double[] max = config.maxBounds();

        for (int i = 0; i < config.foodSourceCount(); i++) {
            final double[] pos = new double[dim];
            for (int d = 0; d < dim; d++) {
                pos[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            final double obj = function.evaluate(pos);
            sources[i] = new FoodSource(pos, obj);
            if (obj < bestFitness) {
                bestFitness = obj;
                bestPosition = pos.clone();
            }
        }
        log.info("Initial population built | best={} mean={}",
                sci(bestFitness), sci(meanObjective()));
    }

    // -------------------------------------------------------------------------
    // Core update mechanism
    // -------------------------------------------------------------------------

    /**
     * Employed‑bee phase: each source is exploited once.
     */
    private void employedBeePhase() {
        log.debug("Employed phase START | bees={}", sources.length);
        for (int i = 0; i < sources.length; i++) {
            if (greedyUpdate(i, "EMPLOYED", i)) {
                employedImprovements++;
            }
        }
    }

    /**
     * Onlooker‑bee phase: each onlooker selects a source by roulette and exploits it.
     */
    private void onlookerBeePhase() {
        log.debug("Onlooker phase START | bees={}", sources.length);
        updateSelectionProbabilities();

        if (log.isTraceEnabled()) {
            logProbabilitySummary();
        }

        for (int onl = 0; onl < sources.length; onl++) {
            final int chosen = rouletteWheel(random.nextDouble());
            if (log.isTraceEnabled()) {
                log.trace("ONLOOKER bee={} selected source={} prob={}",
                        onl, chosen, sci(sources[chosen].getSelectionProbability()));
            }
            if (greedyUpdate(chosen, "ONLOOKER", onl)) {
                onlookerImprovements++;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Probability helpers
    // -------------------------------------------------------------------------

    /**
     * Scout‑bee phase: abandon at most one exhausted source.
     */
    private void scoutBeePhase() {
        log.debug("Scout phase START | abandonment_limit={}", config.abandonmentLimit());

        int worstIdx = 0;
        for (int i = 1; i < sources.length; i++) {
            if (sources[i].getTrials() > sources[worstIdx].getTrials()) {
                worstIdx = i;
            }
        }

        if (sources[worstIdx].getTrials() >= config.abandonmentLimit()) {
            final double oldObj = sources[worstIdx].getObjectiveValue();
            final double[] freshPos = new double[config.dimensions()];
            final double[] min = config.minBounds();
            final double[] max = config.maxBounds();
            for (int d = 0; d < config.dimensions(); d++) {
                freshPos[d] = min[d] + random.nextDouble() * (max[d] - min[d]);
            }
            final double newObj = function.evaluate(freshPos);
            sources[worstIdx] = new FoodSource(freshPos, newObj);
            scoutEvents++;
            if (log.isDebugEnabled()) {
                log.debug("SCOUT abandoned source={} trials={} oldObj={} newObj={}",
                        worstIdx, config.abandonmentLimit(), sci(oldObj), sci(newObj));
            }
        } else {
            log.trace("Scout phase: no source eligible | worst trials={}",
                    sources[worstIdx].getTrials());
        }
    }

    /**
     * Performs the neighborhood search for source {@code idx}:
     * <ul>
     *   <li>Select a random dimension {@code d} and a random partner {@code k} ≠ {@code idx}.</li>
     *   <li>Generate candidate {@code v = x_id + φ · (x_id – x_kd)} with {@code φ ∈ [-1,1]}.</li>
     *   <li>Clamp to search bounds.</li>
     *   <li>Greedy accept / reject and update trial counter accordingly.</li>
     * </ul>
     *
     * @param sourceIdx index of the food source to be exploited
     * @param role      "EMPLOYED" or "ONLOOKER" (for logging)
     * @param beeId     bee identifier for tracing
     * @return {@code true} if the new candidate improved the solution
     */
    private boolean greedyUpdate(final int sourceIdx, final String role, final int beeId) {
        final FoodSource current = sources[sourceIdx];
        final int dim = random.nextInt(config.dimensions());
        final int partnerIdx = pickPartner(sourceIdx);
        final double phi = random.nextDouble(-1.0, 1.0);
        final FoodSource partner = sources[partnerIdx];

        final double[] candidate = current.getPosition().clone();
        candidate[dim] = Math.clamp(
                candidate[dim] + phi * (candidate[dim] - partner.getPosition()[dim]),
                config.minBounds()[dim], config.maxBounds()[dim]);

        final double candObj = function.evaluate(candidate);
        final boolean accept = candObj < current.getObjectiveValue();

        if (log.isTraceEnabled()) {
            log.trace("{} bee={} source={} partner={} dim={} φ={} old={} new={} {}",
                    role, beeId, sourceIdx, partnerIdx, dim, sci(phi),
                    sci(current.getObjectiveValue()), sci(candObj),
                    accept ? "ACCEPT" : "REJECT trials=" + (current.getTrials() + 1));
        }

        if (accept) {
            System.arraycopy(candidate, 0, current.getPosition(), 0, candidate.length);
            current.setObjectiveValue(candObj);
            current.resetTrials();
            return true;
        } else {
            current.incrementTrials();
            return false;
        }
    }

    /**
     * Selects a random partner index different from {@code sourceIdx}.
     */
    private int pickPartner(final int sourceIdx) {
        int partner;
        do {
            partner = random.nextInt(sources.length);
        } while (partner == sourceIdx);
        return partner;
    }

    // -------------------------------------------------------------------------
    // Utility / logging
    // -------------------------------------------------------------------------

    /**
     * Computes selection probabilities using Karaboga's fitness transform.
     * Falls back to uniform probabilities if the total fitness sum is non‑finite.
     */
    private void updateSelectionProbabilities() {
        final double[] fitness = Arrays.stream(sources)
                .mapToDouble(fs -> fitnessTransform(fs.getObjectiveValue()))
                .toArray();
        final double sum = Arrays.stream(fitness).sum();

        if (sum == 0.0 || !Double.isFinite(sum)) {
            log.warn("Fitness sum invalid ({}); using uniform probabilities.", sum);
            final double uniform = 1.0 / sources.length;
            for (final FoodSource fs : sources) {
                fs.setSelectionProbability(uniform);
            }
            return;
        }
        for (int i = 0; i < sources.length; i++) {
            sources[i].setSelectionProbability(fitness[i] / sum);
        }
    }

    /**
     * Roulette‑wheel selection using the pre‑computed probabilities.
     */
    private int rouletteWheel(final double uniformSample) {
        double cumulative = 0.0;
        for (int i = 0; i < sources.length; i++) {
            cumulative += sources[i].getSelectionProbability();
            if (uniformSample < cumulative) {
                return i;
            }
        }
        return sources.length - 1;      // safeguard against rounding errors
    }

    private FoodSource findBestSource() {
        return Arrays.stream(sources)
                .min(Comparator.comparingDouble(FoodSource::getObjectiveValue))
                .orElseThrow(() -> new IllegalStateException("Empty population."));
    }

    private double meanObjective() {
        return Arrays.stream(sources)
                .mapToDouble(FoodSource::getObjectiveValue)
                .average()
                .orElse(Double.NaN);
    }

    private void logProbabilitySummary() {
        final int[] topIdx = IntStream.range(0, sources.length)
                .boxed()
                .sorted((a, b) -> Double.compare(
                        sources[b].getSelectionProbability(),
                        sources[a].getSelectionProbability()))
                .limit(3)
                .mapToInt(Integer::intValue)
                .toArray();
        for (int rank = 0; rank < topIdx.length; rank++) {
            final int idx = topIdx[rank];
            log.trace("Probability rank {}: source={} P={} objective={}",
                    rank + 1, idx,
                    sci(sources[idx].getSelectionProbability()),
                    sci(sources[idx].getObjectiveValue()));
        }
    }

    // -------------------------------------------------------------------------
    // Auxiliary types
    // -------------------------------------------------------------------------

    /**
     * Objective function to minimize.
     */
    @FunctionalInterface
    public interface ObjectiveFunction {
        double evaluate(double[] x);
    }

    /**
     * Immutable algorithm configuration for ABC.
     * {@code abandonmentLimit = 0} triggers the default value
     * {@code foodSourceCount * dimensions}.
     */
    @Builder
    public record AbcConfig(
            int foodSourceCount,
            int maxIterations,
            int dimensions,
            double[] minBounds,
            double[] maxBounds,
            int abandonmentLimit,   // 0 → compute foodSourceCount * dimensions
            long seed
    ) {
        public AbcConfig {
            Objects.requireNonNull(minBounds);
            Objects.requireNonNull(maxBounds);
            // Defensive copy to preserve immutability
            minBounds = minBounds.clone();
            maxBounds = maxBounds.clone();

            if (foodSourceCount < 2)
                throw new IllegalArgumentException("foodSourceCount must be ≥ 2");
            if (maxIterations <= 0)
                throw new IllegalArgumentException("maxIterations must be positive");
            if (dimensions <= 0)
                throw new IllegalArgumentException("dimensions must be positive");
            if (minBounds.length != dimensions || maxBounds.length != dimensions)
                throw new IllegalArgumentException("Bounds array length must equal dimensions");
            if (abandonmentLimit < 0)
                throw new IllegalArgumentException("abandonmentLimit must be ≥ 0");
            if (abandonmentLimit == 0) {
                abandonmentLimit = foodSourceCount * dimensions;
            }
        }
    }

    /**
     * Mutable food source (candidate solution). {@code position} is returned by
     * reference for performance – treat it as read‑only from outside.
     */
    @Getter
    @Setter
    static final class FoodSource {
        private final double[] position;
        @Setter
        private double objectiveValue;
        @Setter
        private double selectionProbability;
        private int trials;

        FoodSource(final double[] position, final double objectiveValue) {
            this.position = position;       // no defensive copy, owned internally
            this.objectiveValue = objectiveValue;
        }

        void incrementTrials() {
            trials++;
        }

        void resetTrials() {
            trials = 0;
        }

        double[] getPosition() {
            return position;                // mutable reference, be careful
        }
    }
}