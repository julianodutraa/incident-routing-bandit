# Teaching an on-call router to learn from its own mistakes

Every data platform team eventually writes the same document: a routing table. Capacity alerts go to the database group. Scheduling alerts go to the orchestration group. Pod alerts go to infrastructure. It is a reasonable document, right up until the first time a capacity alert turns out to be caused by a runaway retry loop in the orchestrator, and the database team spends forty minutes confirming they are not the problem before anyone thinks to page orchestration. The routing table was not wrong, exactly. It was just a snapshot of how the system used to fail, applied to how it fails now.

This project started from a simple question: what if the router learned from its own outcomes instead of from a document someone wrote once and forgot to update? That question has an answer in the bandit literature, and it is a good answer, so this is a write-up of building it, measuring it honestly, and running into the one result that did not go the way I expected.

## The problem is not classification, it is decision-making under feedback

The obvious first instinct is to train a classifier: take a history of incidents, label each one with the team that actually resolved it, fit a model, done. Two things break that plan immediately. First, a growing or recently instrumented platform does not have that history yet, certainly not enough of it to cover every category combination. Second, even with history, a classifier trained on past routing decisions inherits the bias of whoever made those decisions; if the database team was always paged first out of habit, the training data will say database alerts resolve fastest with the database team, which is circular.

A contextual bandit sidesteps both problems. It starts with zero labeled history and learns online, and because it only ever sees the outcome of the arm it actually pulled, it has an honest, built-in notion of regret: the gap between what it did and what the best policy would have done, measured in the actual currency of resolution speed. That is the right abstraction for this problem, and it comes from a well-studied part of the literature (Li, Chu, Langford and Schapire's 2010 LinUCB paper, and Agrawal and Goyal's 2013 Thompson Sampling analysis), not something improvised for this write-up.

## Building a problem hard enough to be interesting

The easy way to build a demo bandit problem is to make one arm obviously best and watch the policy find it. That proves the code runs, not that the method is useful. So the simulation here has a deliberate twist: for every category, the track that actually resolves the incident fastest matches the surface category seventy five percent of the time, and some fixed, different track the other twenty five percent, representing exactly the kind of accumulated routing exception every real on-call rotation has. That means even a policy that perfectly identifies the category can never do better than seventy five percent routing accuracy. The ceiling is part of the design, not a bug to explain away later.

Context, in this simulation, comes from classifying the incident's alert text into one of five categories. Two classifiers were built and evaluated on the same eighty-incident balanced set: a deterministic keyword matcher, and a zero-shot LLM classifier run through a separate Claude subagent that never saw the ground truth labels or the keyword lists, only the alert text and plain category definitions.

## The result I did not expect

I expected the LLM to win. It is the more sophisticated tool, it can reason about which signal in a noisy alert is the dominant cause rather than just counting words, and that reasoning should matter on genuinely ambiguous incidents. It scored 98.75 percent. The keyword matcher scored 100 percent.

The honest explanation is almost embarrassing in its simplicity: the synthetic incident generator draws its keyword phrases from category-specific pools that do not overlap, which hands a substring matcher a nearly unfair advantage on this particular benchmark. Real alert text is messier than that, and this result would not necessarily survive contact with it. But I am not going to quietly bury a result just because it complicates the pitch. What I will point out, because it is also true and also measured, is that the LLM's single error was not a random slip: it misclassified a database capacity incident as an orchestration one, and orchestration is the exact fixed exception this project's own ground truth generator assigns to database capacity incidents twenty five percent of the time. The LLM never saw that mapping. It read an alert that genuinely looked ambiguous between two plausible causes, and picked the one that, by the project's own hidden design, actually is sometimes the right answer. That is a far more interesting kind of being wrong than missing a keyword.

## Where the LLM's accuracy actually mattered, and where it did not

The two classifiers' confusion matrices were fed, unmodified, into the bandit simulation as two different context-quality regimes, alongside a third, explicitly synthetic low-accuracy baseline included only to give the comparison a visible lower end. Running three thousand simulated incidents across twenty random seeds for each of five policies produced a cleaner story than I expected going in.

Context-free policies (plain UCB1, epsilon-greedy, random) converged to roughly twenty five percent routing accuracy regardless of which classifier generated the context, which makes sense since they never look at it, and matches the exact theoretical value for always picking the single globally best arm under this project's exception structure. That is a useful sanity check in its own right: when a simulation's empirical number lands exactly on the value you can derive by hand, you trust the rest of the simulation more.

Giving the contextual policies, LinUCB and linear Thompson Sampling, either near-perfect classifier roughly tripled their routing accuracy, up to the mid-sixties percent range. But between the two near-perfect classifiers, the 1.25 point accuracy gap barely moved the needle downstream: 67.3 percent routing accuracy with the keyword matcher's context versus 67.0 percent with the LLM's. Once context quality clears a fairly low bar, the bottleneck shifts entirely to the bandit's own exploration cost and to the seventy five percent ceiling imposed by the routing exceptions themselves. Neither contextual policy reached that ceiling within three thousand rounds; both plateaued a good eight to ten points under it, which is itself worth sitting with: these are the two best policies tested, and three thousand incidents still were not enough to converge, let alone an org's first three thousand incidents in production, which will carry real-world noise this simulation does not.

LinUCB beat Thompson Sampling on cumulative regret in every context-quality setting tested, by roughly ten percent. I do not think that generalizes as a rule; it is a consequence of the specific reward scale and prior used here, and the repository leaves both easy to change for anyone who wants to check.

## What this is actually good for

Not a production incident router, not yet. What it is: a working, tested demonstration that a routing policy can start from nothing, no labeled history, and reach a meaningfully better-than-static-rulebook accuracy within a few thousand incidents of feedback, with the specific trade-offs (exploration cost, context-quality ceiling, the gap between theoretical and achieved accuracy) measured rather than asserted. The honest limitations section in the README is not boilerplate; the keyword-pool disjointness, the binary reward simplification, and the lack of non-stationarity are the three things I would fix first before trusting this anywhere near a real pager rotation.

The full code, the thirteen automated tests, the cached and provenance-stamped LLM predictions, and the reproducible script that generated every number in this article are here:

https://github.com/julianodutraa/incident-routing-bandit
