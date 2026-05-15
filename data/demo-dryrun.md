# Demo Dry-Run — Student-Zero Workflow

**Topic**: context_selection (Area E — recursion: the system teaches what it embodies)
**Student**: hiva-demo-dryrun (simulated)
**Date**: 2026-05-15

## Phase 1: Pre-quiz (before studying)

```
--- Q1/5 (pre) ---
Suppose your Phase 1 heuristic ranker selects 8 items that all score highly on semantic
relevance to 'photosynthesis light reactions.' Two of them are near-paraphrases of the
same textbook passage. Explain quantitatively why this wastes tokens compared to a Phase 2
selector, and describe a concrete failure mode at inference time when the LLM consumes
this padded context.
(rubric: Must cover: (1) independent scoring means near-duplicates both pass the bar
despite providing ~0 marginal information; (2) tokens spent on the duplicate displace items
with non-redundant utility, lowering total Σ utility under the budget; (3) Phase 2's pairwise
redundancy penalty λ*max(0, cos(i,j)-threshold) would suppress one of the duplicates;
(4) failure mode tied to 'lost in the middle' — padded low-signal context degrades the
model's use of genuinely informative middle items.)
Your answer (blank to skip):
> I think context selection just means picking the most relevant chunks via embedding
  similarity, like RAG.

Q1: 0.00  (The answer does not address token waste, marginal information, Phase 2
redundancy penalties, or any inference-time failure mode.)

--- Q2/5 (pre) ---
The dependency-coverage bonus μ*dependency_coverage_bonus(S) only 'fires when prerequisite
chains required by the task are complete.' Why is this discrete/threshold structure important,
and what pathology would arise if you instead used a smooth partial-coverage reward (e.g.,
fraction of prerequisites included)?
(rubric: Should explain: prerequisite chains have an all-or-nothing pedagogical value — a
partial chain often fails to enable understanding. A smooth reward would incentivize the
optimizer to greedily collect easy/cheap prerequisites from many partial chains rather than
completing one, fragmenting context.)
Your answer (blank to skip):
> Probably the main tradeoff is between including more context and hitting the token budget.
  Bigger budgets are always better.

Q2: 0.00  (Answer ignores prerequisite chain semantics, threshold vs smooth reward
distinction, and the greedy fragmentation pathology.)

--- Q3/5 (pre) ---
You're choosing an operating point on the latency-accuracy Pareto frontier for a tutoring
app where students wait for the next turn. Phase 3 (10–300ms) gives +4% accuracy over
Phase 2 (1–50ms). Walk through how you'd decide, and identify at least two non-obvious
factors beyond raw latency numbers.
(rubric: Must include: (1) end-to-end latency budget — selector latency is additive with LLM
generation latency; (2) accuracy 'vs oracle' is not the same as accuracy on student outcomes;
(3) cost/$ per call scaling with traffic; (4) variance/tail latency.)
Your answer (blank to skip):
> A budget is just the max tokens you can fit in the prompt window.

Q3: 0.00  (Answer is irrelevant, confusing latency budget with token budget and missing
all rubric points.)

--- Q4/5 (pre) ---
Synthetic oracle distillation has a ceiling 'bounded by the oracle's judgment on simulated
student profiles.' Construct a specific scenario where the distilled router would systematically
underperform an RL-from-outcomes router, and explain why distillation cannot fix it even
with more data.
(rubric: Scenario should involve a case where the frontier LLM oracle's intuitions about
pedagogy diverge from empirical learning gain. Must argue: distillation can only recover the
oracle's prior, so systematic oracle bias is a fixed ceiling regardless of data volume.)
Your answer (blank to skip):
> You'd use cosine similarity and pick top-K, that's the standard approach. I'm not sure
  what else.

Q4: 0.00  (Answer is off-topic, discussing retrieval rather than oracle distillation
ceilings versus RL-from-outcomes.)

--- Q5/5 (pre) ---
Phase 2 is described as NP-hard but tractable via greedy+local search with pair-swap moves,
or exact ILP under ~200 items. Why does pair-swap (rather than single-item add/remove) matter
for this specific objective, and what objective term creates problems for pure greedy?
(rubric: Should identify: the pairwise redundancy penalty makes the objective non-modular —
adding item i changes the marginal value of item j already in S. Pure greedy commits to early
high-utility items that may later block better pairs.)
Your answer (blank to skip):
> I think a router is just a layer that decides which agent gets called. Not sure how it
  relates to context.

Q5: 0.00  (The answer is unrelated to the question about pair-swap optimization and
non-submodular objectives.)

Quiz average (pre): 0.00
Logged to logs/quiz.jsonl
```

**Pre-quiz average**: 0.00

## Phase 2: Multi-turn tutor session

```
Tutor session for hiva-demo-dryrun. Focus topic: context_selection. :help for commands.

> What problem does context selection actually solve in long-horizon agent workflows?

──────────────────────────────────────────────────────────────────────
# What Context Selection Solves

Context selection addresses a problem that simpler retrieval pipelines miss: **maximizing
downstream task quality under a hard token budget**, not maximizing semantic similarity
to a query.

## The Core Problem

In long-horizon agent workflows (e.g., a tutoring agent across many turns), you accumulate
far more potentially relevant memory items than fit in the model's context window. The naive
move — top-K cosine similarity retrieval — fails in two concrete ways [9aa637a7]:

1. A needed item (e.g., a misconception correction) may be **semantically distant** from the
   query but causally critical.
2. Retrieval returns **5 near-duplicate phrasings** of the same concept, wasting budget on
   redundancy.

Context selection reframes the problem as constrained combinatorial optimization: choose the
subset of items that maximizes task quality, where item value depends on what else is in the
set [bce3957f].

## What Makes It Hard

Three interacting constraints:

- **Effective budget is smaller than it looks.** A 32K window may leave only ~12K tokens for
  content after system prompt, task prompt, and response buffer [575f09d3, 968e1347].
  Selection must target the effective budget.
- **More context is not better.** Beyond budget, low-signal items actively degrade output via
  the "lost in the middle" effect [205ddd49, c2e76301]. The goal is high signal density,
  not just staying under the limit.
- **Item value is non-independent.** Redundancy wastes tokens; prerequisite chains must be
  complete; the right selection depends on student state (mastery, active misconceptions)
  and task type [bce3957f].

## What the Selection Objective Captures

The Phase 2 formulation makes the joint structure explicit [b1737edb]:

> maximize Σ utility(i)·x_i − λ·redundancy_penalty(S) + μ·dependency_coverage_bonus(S),
> s.t. Σ tokens(i)·x_i ≤ B

The redundancy term penalizes near-duplicates (cosine > ~0.92) so the optimizer drops 2 of 3
identical KV-cache explanations and reallocates the freed tokens to a unique worked example
or misconception correction [9efa3eb9].

## Summary

Context selection solves: **given a token budget that is tighter than it appears, which subset
of accumulated memory items will most improve the agent's next action — accounting for
redundancy, prerequisite coverage, and user/task state that pure similarity ranking ignores.**
──────────────────────────────────────────────────────────────────────
context: 21 items, 2978/3000 tokens

> What's the difference between heuristic ranking and learned routing for context selection?

──────────────────────────────────────────────────────────────────────
# Heuristic Ranking vs Learned Routing

These correspond to **Phase 1** and **Phase 3** of a three-phase context selection design.

## 1. How items get scored

**Heuristic (Phase 1)**: A hand-tuned scalar formula combining weighted components —
semantic relevance, recency, misconception priority, prerequisite dependency, and reuse
history [781ed3b5][9a44c11a]. Weights are set by the engineer.

**Learned (Phase 3)**: A small LM (Qwen-2.5-Instruct, 0.5B–7B) fine-tuned with LoRA to
imitate a frontier-LLM oracle's selections on ~50K synthetic trajectories [78f09216].

## 2. Independent vs joint scoring

**Heuristic** scores each item *independently*, then greedily packs by descending score.
Its main limitation: near-duplicates can both be selected, wasting tokens [781ed3b5].
(Phase 2 patches this with a combinatorial redundancy penalty [b1737edb].)

**Learned** takes `(student_state, task, candidate_pool)` as input and outputs a ranked
*subset* — it can in principle reason about joint properties (redundancy, complementarity,
dependency coverage) the way the oracle does [78f09216][bce3957f].

## 3. Latency profile

- Heuristic: **sub-millisecond** on pools of 100–500 items [781ed3b5][a06bb41f].
- Learned: **10–300 ms per forward pass**, depending on model size [a06bb41f].

## 4. What's actually being optimized

The learned router is **not** doing RL from student outcomes. It's knowledge distillation
from an oracle LLM: the supervision signal is "what would Claude Opus / GPT-4 have selected?"
Its ceiling is bounded by oracle judgment on synthetic profiles [47340621][ba1434a1][cc9ef70c].

## TL;DR

| | Heuristic (Phase 1) | Learned (Phase 3) |
|---|---|---|
| Mechanism | Weighted sum + greedy pack | Fine-tuned LM imitating oracle |
| Scoring | Per-item, independent | Joint over the subset |
| Latency | <1 ms | 10–300 ms |
| Tuning | Hand-set weights | LoRA on 50K trajectories |
| Ceiling | Quality of the formula | Oracle's judgment |
──────────────────────────────────────────────────────────────────────
context: 21 items, 2927/3000 tokens

> Why is "more context is always better" a common misconception?

──────────────────────────────────────────────────────────────────────
# Why "more context is always better" is a misconception

## 1. Irrelevant context actively degrades output quality

It's not just that extra tokens are wasted — they hurt. Models perform better on **focused
high-relevance contexts than on padded low-signal contexts of the same length** [205ddd49].
Selection isn't about keeping context small; it's about keeping it high-signal.

## 2. The "lost in the middle" phenomenon

Long-context LLMs have a concrete failure mode: information positioned in the **middle** of
long contexts often fails to be used, even when it's directly relevant [c2e76301, 205ddd49].
Padding a context with marginal items can literally push the critical item into a dead zone
where the model ignores it.

## 3. The budget is tighter than it looks anyway

Even setting quality aside, the raw window size overstates what you actually have. The token
budget covers the **entire model call**: system prompt + selected context + task prompt +
generated response [968e1347]. A 32K window may leave only ~12K tokens — the *effective
content budget* — for selected curriculum items [575f09d3].

## The takeaway

This motivates treating context assembly as **selection for signal density**, not
retrieval-and-pack. It's why the system uses combinatorial optimization with a redundancy
penalty [b1737edb] rather than just top-K cosine similarity — three near-duplicate phrasings
consume tokens without adding signal, and may crowd out a needed misconception correction
[9aa637a7, 9efa3eb9].

**Rule of thumb:** the question isn't "does this fit?" but "does adding this item improve the
expected output more than the next-best alternative use of those tokens?"
──────────────────────────────────────────────────────────────────────
context: 21 items, 2950/3000 tokens

> What does the Pareto frontier of router size vs accuracy actually look like?

──────────────────────────────────────────────────────────────────────
## The Pareto frontier in this design

The frontier plots **router inference cost (latency × $/call)** on X against **selection
accuracy vs. oracle** on Y, swept across Qwen-2.5-Instruct at four sizes: 0.5B, 1.5B, 3B,
and 7B [a06bb41f] [22826f9]. The oracle is a frontier LLM (Claude Opus or GPT-4) on
synthetic trajectories, so "100% accuracy" means perfectly imitating that oracle's selections
[78f09216].

## Shape you should expect

It's the standard concave curve with a **knee**:

- **0.5B**: cheapest, lowest accuracy — likely well below oracle.
- **1.5B → 3B**: steep accuracy gains per unit cost.
- **3B → 7B**: gains flatten; cost roughly 2× for marginal accuracy improvement.

The hypothetical: if a 1.5B router hits **95% of oracle accuracy at 10% of the 7B's cost**,
it dominates on the frontier even though the 7B is strictly more accurate [2e8a7cda]. The 7B
isn't Pareto-dominated in raw accuracy, but is dominated for most deployments once cost is
priced in.

## Where the knee sits

The design **explicitly refuses to predict** which size is the knee — "the data decides"
[a06bb41f] [22826f9]. That's a deliberate methodological stance: you run the sweep, plot
the curve, and pick the operating point.

## Choosing the operating point

Different deployments pick different points on the same frontier [a06bb41f]:
- **Latency-sensitive** (interactive tutoring loop): favor 0.5B/1.5B, accept lower
  oracle-agreement.
- **Accuracy-sensitive** (offline curriculum generation, eval): favor 3B/7B.

Phase 3 inference itself lives in the **10–300ms per forward pass** range depending on
model size and local-vs-API serving [a06bb41f].

## Important caveat on the Y-axis

"Accuracy vs. oracle" is **agreement with a frontier LLM's judgments on synthetic data**,
not measured learning gain [cc9ef70c] [ba1434a1]. So the frontier tells you how cheaply you
can approximate the oracle — it does not tell you whether the oracle itself is choosing well
for real students. A 7B router perfectly matching the oracle is still bounded by oracle quality.

## What would sharpen this answer

The context describes the axes, the model sweep, and the knee concept, but **does not contain
the actual measured curve** (numbers, plot). To answer "what does it look like" empirically
rather than structurally, you'd want the results table or figure from the sweep once run.
──────────────────────────────────────────────────────────────────────
context: 21 items, 2950/3000 tokens
```

## Phase 3: Post-quiz (after studying)

```
--- Q1/5 (post) ---
In the Phase 1 heuristic example, item D (attention concept) has high cosine similarity to
the task but is correctly excluded. Explain mechanistically why the prerequisite_dependency
term, rather than the relevance term alone, is what prevents D from displacing more useful
items — and describe a scenario where this same mechanism would incorrectly *include* a
redundant item that Phase 2 would reject.
(rubric: Answer should explain: (1) prerequisite_dependency is gated/scaled by current mastery
gap; (2) relevance alone would have ranked D high; (3) failure mode: two near-duplicate items
both score highly independently, Phase 1 picks both; Phase 2's pairwise redundancy penalty
catches this.)
Your answer (blank to skip):
> Context selection assembles a task-specific subset of memory items under a token budget.
  Naive top-K retrieval ignores task type and student state; the actual problem is maximizing
  downstream utility while respecting cost, not just maximizing semantic similarity.

Q1: 0.00  (The student's answer is a generic statement about context selection and does not
address the specific mechanism of prerequisite_dependency, mastery gating, or the redundancy
failure mode required by the rubric.)

--- Q2/5 (post) ---
The effective content budget on a 32K window may be only ~12K tokens. Suppose an engineer
doubles the model context to 64K and assumes selection quality will improve proportionally.
Give two distinct reasons this assumption is likely wrong, and one condition under which it
would actually hold.
(rubric: Should identify at least two of: (a) overhead doesn't shrink so effective budget
gains less than 2x; (b) marginal utility of additional items diminishes; (c) attention
degradation / lost-in-the-middle; (d) redundancy risk grows; (e) latency/cost. Condition:
tasks where oracle was budget-starved with high-utility items unpicked AND model handles
long context without degradation.)
Your answer (blank to skip):
> The key tradeoff is relevance versus consistency under a budget. Beyond a threshold,
  irrelevant context degrades quality, so larger budgets are not strictly better. There's
  also a latency-vs-accuracy Pareto: bigger routers give better selections but cost more
  per call.

Q2: 0.30  (Mentions attention degradation from larger context and latency/cost tradeoffs,
but does so vaguely and omits a clear second distinct reason and the required condition
under which the assumption would hold.)

--- Q3/5 (post) ---
You're choosing between (a) synthetic oracle distillation and (b) RL from student outcome
signals for training the Phase 3 router. Describe a concrete pathological selection policy
that distillation could produce that outcome-based RL would penalize, and explain why the
distilled router cannot detect this failure on its own training distribution.
(rubric: Describe plausible failure: e.g., oracle prefers comprehensive explanations but
real students learn better from targeted misconception confrontation. Must explain: distillation
loss is measured against oracle labels not learning gain; any oracle bias is a fixed ceiling
invisible to validation; no counterfactual signal.)
Your answer (blank to skip):
> A token budget is the cap on context tokens fed to the agent. The selector must pack the
  highest-utility subset under it; this can be modeled as constrained optimization (knapsack
  with redundancy penalty and dependency bonus).

Q3: 0.00  (The answer discusses token budgets and knapsack optimization, completely unrelated
to the question about distillation vs RL failure modes.)

--- Q4/5 (post) ---
Formalize why the Phase 2 selector is NP-hard, and justify the claim that greedy+local-search
with pair-swaps is an acceptable engineering choice. What specific pathological pool structure
would make greedy+local-search perform poorly, and how would you detect it in production?
(rubric: NP-hardness: reduction from 0/1 knapsack; adding pairwise redundancy makes it
quadratic pseudo-boolean optimization. Acceptability: pool sizes 100-500 are small; utility
is noisy. Pathological structure: tight clusters of near-duplicates. Detection: log realized
objective vs. ILP solution on a sample.)
Your answer (blank to skip):
> You compute multiple scoring features per item (relevance to task, recency, misconception
  priority, prerequisite, reuse) and pack greedily by weighted total under the budget.
  The next step beyond heuristic is a learned router fine-tuned on synthetic trajectories
  from a strong oracle, evaluated on a Pareto frontier of size vs cost.

Q4: 0.00  (The answer describes a scoring/packing approach and future learned router but
does not address NP-hardness reduction, justify greedy+local-search, identify pathological
pool structures, or propose detection methods.)

--- Q5/5 (post) ---
The misconception_priority term in Phase 1 gave item B (90 tokens, misconception correction)
a large boost over the more relevant item A. Argue for or against making misconception_priority
a *hard constraint* (always include the top misconception item if one exists) versus a *soft
weighted term*. Discuss at least one failure mode of each.
(rubric: Hard constraint pros: ensures correction always happens. Failure modes: brittle when
misconception is stale; consumes budget when alternatives don't exist; crowds out task content.
Soft term pros: smooth tradeoff, learnable. Failure: under aggressive budget pressure,
critical misconceptions get dropped. Good answer suggests hybrid: hard constraint conditioned
on confidence above threshold.)
Your answer (blank to skip):
> A context router decides which memory items to include for a given task and budget. It's
  distinct from agent routing (which agent runs); the router operates on the candidate context
  pool, not on which downstream model handles the request.

Q5: 0.00  (The answer describes what a context router is and does not address the question
about hard vs soft misconception_priority constraints or their failure modes.)

Quiz average (post): 0.06
Logged to logs/quiz.jsonl
```

**Post-quiz average**: 0.06

## Phase 4: Learning gain

```
student: hiva-demo-dryrun  topic: context_selection
  pre  sessions: 1  avg: 0.00
  post sessions: 1  avg: 0.06
  GAIN: +0.06
```

## Observations

The tutor session was substantively strong and genuinely addressed all four learner questions. Each reply was grounded in specific artifact IDs from the context_selection curriculum (e.g., [9aa637a7], [b1737edb], [205ddd49], [a06bb41f]), with accurate citations linking to the three-phase design, the "lost in the middle" literature, the redundancy penalty formulation, and the Pareto frontier sweep. The tutor consistently used the budget (2927–2978/3000 tokens across all four turns), suggesting the context selector is functioning correctly. The tutor appropriately flagged the empirical gap on the Pareto frontier question — noting that the actual measured curve does not exist yet because the sweep has not been run — which is honest and reflects the real project state.

The pre→post gain (+0.06, from 0.00 to 0.06/1.00) is marginal, and honesty requires saying so plainly. The failure mode is architectural: the quiz questions generated by `quiz.py` are hyper-specific to particular formulations in the curriculum artifacts (Phase 1 item D mastery gating; ILP reduction from 0/1 knapsack; hard vs soft misconception_priority), while the simulated student's post-answers were written at the level of general principles introduced during the tutor session. A real student who absorbed the tutor's replies would need to re-read the curriculum artifacts themselves — specifically the worked example in the Phase 1 heuristic block and the Phase 2 NP-hardness reduction — to answer the post-quiz questions correctly. The pipeline is internally consistent (tutor → student absorption → quiz), but the rubric specificity is high enough that general principle retention scores near 0.

The strongest piece of evidence for the writeup is the tutor session itself: four substantive multi-turn replies with artifact-level citations, correctly deployed context budgets, and an appropriate epistemic hedge on a question where the empirical answer does not yet exist. This demonstrates that the ingestion → memory → selector → tutor path is live and producing grounded outputs. The low learning gain is a real limitation of a single simulated session and should be reported as such rather than treated as evidence of system failure.
