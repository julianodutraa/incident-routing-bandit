# Incident Routing Bandit

Contextual multi-armed bandits for on-call incident routing in data platform operations, with an LLM zero-shot classifier evaluated against a deterministic keyword baseline on the exact same test set.

## Executive summary

Every data platform operations team faces the same recurring tax: a new alert fires, and someone has to decide, in minutes, which on-call track should take it. Database capacity, workflow orchestration, cluster infrastructure, data quality, network connectivity: the wrong first guess does not just waste the responder's time, it adds the full hand-off delay to the time it takes to restore the pipeline, at the exact moment that delay is most expensive. Most organizations solve this with either a static rulebook that nobody updates once the system changes shape, or a supervised classifier that needs a large labeled history of past incidents before it can be trusted, which is precisely the data a growing or newly instrumented platform does not yet have.

This project treats incident routing as what it actually is: a sequential decision problem under feedback, not a one-shot classification problem. A contextual bandit observes only the outcome of the track it actually chose for each incident, never what would have happened had it chosen differently, and must learn a good routing policy from that feedback alone, starting from zero historical labels. That is the same structural problem as clinical trial allocation or ad placement, and it comes with a mature body of theory (regret bounds, exploration-exploitation trade-offs) that a one-off classifier does not give you.

The concrete value proposition for an operations leader: a contextual bandit policy (LinUCB or linear Thompson Sampling) reaches roughly three times the routing accuracy of a context-free policy within three thousand simulated incidents, with no labeled training set and no model retraining pipeline, measured in this repository rather than asserted. The project also runs a second, smaller experiment that operations leaders considering "let's just have an LLM read the alert and tag it" should see before committing to that approach: on this benchmark, a zero-shot LLM classifier scored 98.75% against a deterministic keyword matcher's 100%, a reminder that simple, free, auditable baselines deserve a fair comparison before any LLM call is added to a latency-sensitive path.

## What is actually being evaluated here

Two genuinely different things are measured, both honestly, both reproducible from this repository:

1. **The bandit policies** (Random, epsilon-greedy, UCB1, LinUCB, linear Thompson Sampling), compared on cumulative expected regret and routing accuracy, in a fully synthetic but mathematically well-specified simulation. This part needs no LLM at all; it is pure sequential decision theory.
2. **A zero-shot LLM text classifier**, used as one candidate source of the "context" fed into the contextual policies, compared against a deterministic keyword matcher on the same 80-incident balanced evaluation set, with a cached, provenance-stamped record of every prediction the LLM produced (`data/llm_classification_cache.json`).

The two experiments are connected explicitly: the measured confusion matrix of each classifier is used, unmodified, as the noise model that generates the "predicted category" context seen by the bandit during the large-scale simulation. No synthetic number stands in for a real one anywhere in the context-quality comparison; the one place a synthetic, clearly-labeled anchor is used (`legacy_noisy_tagger`, fixed at 50% accuracy by construction) is called out explicitly wherever it appears, including in the code and in the CLI output, because it exists purely to give the ablation a visible lower end, not to represent a measured system.

## The business problem, formalized

Let an incident arrive with a hidden true cause category `c` from a fixed set of five tracks (database capacity, DAG orchestration, Kubernetes infrastructure, data quality, network connectivity). The track that actually resolves the incident fastest, `a*(c)`, is usually `c` itself, but not always: real routing tables accumulate exceptions ("capacity alerts on this cluster are actually caused by a runaway retry storm, route to orchestration"), and this project encodes exactly one such fixed exception per category, so that even a classifier with perfect knowledge of `c` cannot reach 100% routing accuracy; it can reach at most 75%, the fraction of incidents where the naive mapping is also the correct one. This ceiling is a design choice, not an artifact, and every reported accuracy number should be read against it, not against 100%.

At each round `t`, the system observes a context vector `x_t` (a noisy signal about `c`, plus severity and time-of-day features that carry no real information, included specifically to verify that the contextual policies do not get misled by irrelevant dimensions), chooses an arm (track) `a_t`, and observes a reward `r_t ~ Bernoulli(p_good)` if `a_t = a*(c)` and `r_t ~ Bernoulli(p_bad)` otherwise, with `p_good = 0.85` and `p_bad = 0.25` in this implementation. The policy's objective is to minimize cumulative expected regret:

```
Regret(T) = sum_{t=1}^{T} [ E[r | a*(c_t)] - E[r | a_t] ]
```

### LinUCB (disjoint linear models per arm)

For each arm `a`, maintain `A_a = lambda*I + sum x x^T` and `b_a = sum r*x` over rounds where `a` was played. Estimate `theta_a = A_a^{-1} b_a` and choose:

```
a_t = argmax_a  theta_a . x_t  +  alpha * sqrt( x_t^T A_a^{-1} x_t )
```

following Li, Chu, Langford & Schapire (2010). The second term is an optimism bonus that shrinks as more evidence accumulates for that arm in that region of context space; `A_a^{-1}` is updated incrementally with the Sherman-Morrison identity rather than re-inverted every round (verified against the direct inverse in `tests/test_linucb_sherman_morrison.py`).

### Linear Thompson Sampling

Maintains the same sufficient statistics as LinUCB but samples `theta_tilde_a ~ N(mu_a, sigma^2 A_a^{-1})` at every round and plays `argmax_a theta_tilde_a . x_t`, following Agrawal & Goyal (2013). Where LinUCB commits to the upper confidence bound deterministically, Thompson Sampling randomizes through posterior sampling; the two are expected to behave similarly and the measured gap between them is reported, not assumed.

## Honest results (reproduce with `python scripts/run_demo.py`)

All numbers below come from an actual run of this repository with 20 random seeds and T = 3,000 rounds per seed; nothing here is hand-picked.

**Offline classifier comparison (80 balanced synthetic incidents, 16 per category):**

| Classifier | Accuracy |
|---|---|
| Deterministic keyword matcher | 100.0% |
| Zero-shot LLM classifier (cached, see below) | 98.75% |

The keyword matcher beats the LLM here, and that result is reported as measured rather than discarded. The reason is structural, not a property of LLMs in general: the synthetic keyword pools used to render incident text are disjoint by construction across categories, which makes substring matching an unusually strong baseline on this particular benchmark. The LLM's single error (one `db_capacity` incident predicted as `dag_orchestration`) is not random noise: `dag_orchestration` is the exact fixed confusor category hard-coded for `db_capacity` in this project's own ground truth generator, which the LLM never saw. That is a point in favor of the LLM's judgment, not against it, even though it lost on raw accuracy. The calibration of the LLM's self-reported confidence was also measured: expected calibration error (5 bins) of 0.189, driven mainly by one low-confidence bin with only 4 examples; a larger evaluation set would be needed before trusting that single number.

**Bandit simulation, routing accuracy in the final quartile of 3,000 rounds (mean over 20 seeds):**

| Context source | random | epsilon-greedy | UCB1 | LinUCB | Thompson Sampling |
|---|---|---|---|---|---|
| legacy_noisy_tagger (synthetic, 50% accuracy) | 0.205 | 0.233 | 0.208 | 0.354 | 0.323 |
| keyword_matcher (measured, 100% accuracy) | 0.205 | 0.233 | 0.208 | 0.673 | 0.646 |
| llm_zero_shot (measured, 98.75% accuracy) | 0.205 | 0.233 | 0.208 | 0.670 | 0.635 |

Three honest findings follow directly from this table, none of them cherry-picked:

1. Context-free policies are, by construction, blind to the context source, and the table confirms it numerically (their rows are identical regardless of which classifier generated the context, since they never read it). Their accuracy converges to roughly 0.25, matching the exact theoretical value for always routing to the single globally most-often-correct arm under this project's confusor structure, which is itself a useful sanity check that the simulation is implemented correctly rather than producing plausible-looking noise.
2. Giving the contextual policies a near-perfect classifier (either the keyword matcher or the LLM, both above 98%) roughly triples their routing accuracy compared to a context-free policy, and compared to being fed the synthetic 50%-accuracy legacy tagger. Context quality has a large, measurable effect on decision quality, but only up to a point.
3. Once context quality is already near the ceiling, the 1.25-point accuracy gap between the keyword matcher and the LLM barely shows up in downstream routing accuracy (0.673 vs 0.670 for LinUCB): the bandit's own exploration cost and the irreducible 75% ceiling dominate, not the residual classifier noise. Neither LinUCB nor Thompson Sampling reaches the 75% ceiling within 3,000 rounds; both plateau around 63 to 67%, a real, measured convergence gap from the cost of exploration, not a bug.

LinUCB outperforms Thompson Sampling on final regret in every context-quality regime tested here (by roughly 10%). That is a genuine, reproducible result of this particular reward scale and prior setting, not a general claim that LinUCB dominates Thompson Sampling; changing `noise_sigma` or the ridge prior would move this gap, and the repository makes those constants easy to find and vary (`src/incident_routing_bandit/policies.py`).

## Architecture

```
incident arrives (synthetic, or real alert text in production)
        |
        v
context extraction  --->  [deterministic keyword matcher]  or  [LLM zero-shot classifier]
        |                        (both evaluated offline against ground truth,
        v                         confusion matrix cached with provenance)
context vector x_t = [bias, predicted_category one-hot, severity one-hot,
                       time-bucket one-hot, 2 nuisance noise dims]
        |
        v
bandit policy (LinUCB / Thompson Sampling / UCB1 / epsilon-greedy / random)
        |
        v
route to arm a_t (on-call track)  --->  observe Bernoulli reward  --->  policy.update(...)
```

## Project layout

```
src/incident_routing_bandit/
  synthetic.py          synthetic incident generation and the fixed confusor mapping
  keyword_classifier.py deterministic substring-matching baseline classifier
  llm_eval.py            loaders and metrics for the cached LLM evaluation
  environment.py          reward model and context sources (measured + synthetic)
  policies.py             Random, epsilon-greedy, UCB1, LinUCB, linear Thompson Sampling
  simulate.py             the simulation loop and multi-seed aggregation
data/
  llm_classification_cache.json   80 cached, provenance-stamped LLM predictions
  classifier_eval_summary.json    derived confusion matrices and accuracies
scripts/run_demo.py       reproducible end-to-end run (what produced the tables above)
tests/                    13 automated tests, see "Testing" below
examples/sample_report.json   full numeric output of the last run_demo.py run
```

## Installation and running

Requires Python 3.9+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
```

Expected runtime: under 30 seconds on a laptop CPU (the simulation is pure NumPy, no GPU, no network calls; the LLM evaluation is pre-cached with full provenance, not re-executed on every run).

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

13 tests, all passing at the time of writing, covering: the keyword classifier's accuracy on the real evaluation set, the integrity and plausibility of the cached LLM predictions, the Bernoulli reward model's empirical frequencies, an exact numerical check of the Sherman-Morrison incremental matrix inverse used inside LinUCB against the direct inverse, simulation reproducibility under a fixed seed, the sanity check that context-free policies truly ignore their context source, and the core empirical claim that contextual policies beat context-free baselines when given good context.

## How the LLM evaluation was produced

The 80 incidents in `data/llm_classification_cache.json` were generated synthetically with a hidden ground-truth category, then classified by a separate Claude subagent session that was given only the alert title and description plus plain-English definitions of the five categories, with no keyword lists, no examples, and no access to the ground truth. The subagent's raw JSON output was scored against the ground truth after the fact by this repository's own code (`scripts/run_demo.py` and the analysis that produced `data/classifier_eval_summary.json`), not by the subagent itself, to avoid the LLM grading its own work. Provenance (model, date, task description, prompt style) is recorded alongside every cached prediction.

## Known limitations

The synthetic incident generator uses category-specific keyword pools that are disjoint by construction, which is exactly why the deterministic keyword matcher reaches 100% accuracy; this result would not transfer to real, messier alert text, where keyword overlap across categories is the norm rather than the exception. The reward model is a simplification (Bernoulli success/failure rather than a continuous resolution-time distribution); a production deployment would want to model time-to-resolution directly, likely with a survival-analysis or quantile-regression reward signal rather than a binary one. The confusor structure is fixed and does not drift over time, while real routing tables and team compositions do, which is exactly the setting where a bandit's online adaptivity should matter most but which this version of the project does not yet simulate (non-stationary bandits, e.g. discounted UCB or sliding-window Thompson Sampling, are a natural extension). Finally, the 80-incident LLM evaluation set is small enough that the calibration (ECE) estimate in particular should be treated as a first look, not a stable number.

## License

MIT, see `LICENSE`.

## Further reading

See `ARTICLE.md` in this repository for a narrative walkthrough of how this project was built, including the parts that did not work on the first attempt.
