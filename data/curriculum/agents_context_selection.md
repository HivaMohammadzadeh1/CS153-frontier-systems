# Context Selection Under Budgets

**Area E — Agent Systems & Frontier Framing | Learning Memory OS Curriculum**

---

## 1. The Core Problem

Long-horizon agent workflows accumulate context. Each tutoring session, every quiz attempt, every misconception correction, every worked example the student has seen — all of it is a candidate piece of information that might (or might not) belong in the context window when the tutor answers the student's next question. After 6 deep-eval sessions on a single topic, the set of candidate memory items can easily exceed 200 entries, each carrying a few hundred tokens. At a 4K–8K token budget, only a fraction can fit.

The naive solution is top-K retrieval: embed the current query, embed all candidate items, rank by cosine similarity, take the top K under budget. This is better than nothing. It is not sufficient.

Cosine similarity answers the question "which items are semantically close to the query?" It does not answer: "which items, jointly, will allow the tutor to produce the best response for this specific student at this specific moment in their learning arc?" Those are different questions. A student who has a persistent misconception about why gradient checkpointing trades compute for memory needs an item that directly corrects that misconception — even if the misconception correction item ranks 47th in raw cosine similarity to the query "explain transformer memory footprint."

Context selection is therefore a constrained combinatorial optimization problem, not a retrieval problem. The constraints are token budget and latency. The objective is downstream task quality: will the tutor response produced under this context selection be more correct, more targeted to this student's misconceptions, and less redundant than alternative selections?

This framing has three important implications:

1. The value of an item depends on which other items are in context with it (redundancy and complementarity).
2. The student's current state — mastery scores, active misconceptions, recent session history — is part of the input to the selection function, not just the query text.
3. The optimal selection depends on the task type: explanation needs prerequisite concepts; quiz generation needs weak-concept identification; misconception repair needs the specific misconception plus its correction.

---

## 2. Three-Phase Selector Design

Learning Memory OS implements context selection in three progressively more powerful phases, each building on the previous.

### 2.1 Phase 1 — Heuristic Ranker and Budgeted Packing

Phase 1 is the MVP. Every item in the candidate pool receives a scalar score, and items are greedily selected in score order until the token budget is exhausted.

The heuristic score for item i given student state s and task t is:

```
score(i, s, t) = w_rel  * cosine_sim(embed(i), embed(query(t)))
              + w_rec  * recency_decay(i, s)
              + w_misc * misconception_priority(i, s)
              + w_pre  * prerequisite_dependency(i, t)
              + w_reuse * reuse_success_history(i, s)
```

Each component has a concrete definition:

- **Semantic relevance** (`w_rel`): cosine similarity between the item's embedding and the task query embedding. This is what classical IR computes. It is necessary but not sufficient.
- **Recency** (`w_rec`): exponential decay based on how many sessions ago the item was last seen. Items seen 1 session ago score high; items from 10 sessions ago score near zero unless relevance is strong. This prevents stale context from crowding out fresh material.
- **Misconception priority** (`w_misc`): a binary-like boost applied when item i is a misconception correction that targets an active misconception recorded in the student's state. Active misconceptions are those flagged in the diagnostic layer but not yet resolved. The boost is large — enough to force these items into context even if their cosine similarity is mediocre.
- **Prerequisite dependency** (`w_pre`): boost applied when item i is a prerequisite concept for the current task's topic, and the student's mastery of that prerequisite is below a threshold. The prerequisite graph is encoded in the topic YAML. This ensures that explanations of advanced topics pull in foundational material when mastery is low.
- **Reuse history** (`w_reuse`): boost applied when item i has been included in successful past interactions (measured by positive LLM-judge scores or absence of student correction requests). Items that "worked" previously are preferred. Items that were selected but generated student confusion get a small penalty.

After scoring, packing is greedy: sort by score descending, add items until budget is exhausted or no item fits. This is O(n log n) and runs in sub-millisecond time on typical candidate pool sizes (100–500 items).

The limitation of Phase 1 is independence: each item is scored individually, ignoring what other items have already been selected. This means two near-duplicate concept definitions can both enter the context, consuming tokens without adding information.

### 2.2 Phase 2 — Combinatorial Selector

Phase 2 replaces greedy packing with a knapsack-style optimization that explicitly models the joint value of a selected set S:

```
maximize   Σ_i  utility(i) * x_i
           - λ * redundancy_penalty(S)
           + μ * dependency_coverage_bonus(S)

subject to  Σ_i  tokens(i) * x_i  ≤  B
            x_i ∈ {0, 1}
```

Here:

- `utility(i)` is the Phase 1 score for item i, providing a baseline per-item value.
- `redundancy_penalty(S)` penalizes pairs of items whose embeddings are highly similar. Concretely, for each pair (i, j) both in S, the penalty is proportional to max(0, cosine_sim(i, j) - threshold). This forces the optimizer to prefer a diverse set over a redundant one.
- `dependency_coverage_bonus(S)` adds a bonus when S contains prerequisite-dependency chains that are complete — i.e., if the task requires concept C, and C has prerequisite P, and both C and P are in S, the coverage bonus fires. Partial chains (C without P, when P mastery is low) get no bonus.
- B is the token budget.

The exact optimization is NP-hard in general, but the candidate pool sizes encountered in practice (≤500 items) make greedy local search with pair-swap moves tractable. An ILP solver (PuLP) can also be used for exact optimization on smaller pools; for pools above ~200 items under tight latency constraints, greedy + local search suffices.

Phase 2 adds meaningful quality improvement over Phase 1 when the candidate pool has high internal redundancy — which happens when many sessions have produced similar-phrasing explanations of the same concept. In these cases, Phase 1 might include 3 near-identical concept explanations; Phase 2 keeps 1 and uses the freed tokens for a unique worked example.

### 2.3 Phase 3 — Learned Router

Phase 3 is the headline empirical contribution. A small language model is fine-tuned to imitate an oracle selector, replacing the hand-tuned weight vector from Phase 1 and the hand-tuned penalty terms from Phase 2 with learned parameters.

**Model family**: Qwen-2.5-Instruct at sizes 0.5B, 1.5B, 3B, and 7B.

**Fine-tuning method**: LoRA (rank 16), not full fine-tuning. Full fine-tuning at 7B is computationally out of scope for the 17-day window.

**Input representation**: A serialized tuple `(student_state, task, candidate_pool)`. The student state includes per-concept mastery scores, active misconceptions with recency, and a summary of the last N sessions. The task includes its type (explain / quiz / lab / review) and the current topic. The candidate pool is a ranked list of item IDs with their text bodies.

**Output**: The model produces a subset of item IDs from the pool, ranked by recommended inclusion order.

**Training data**: Synthetic tutoring trajectories in the format `(student_state, task, pool, oracle_selection)`. The oracle is a strong frontier LLM (Claude Opus or GPT-4) given full memory access and asked to make the ideal selection. 50K+ such trajectories are generated across diverse student profiles, topic areas, task types, and mastery states.

**Training objective**: Supervised fine-tuning on the oracle selections. The model learns to predict which item IDs the oracle would have chosen.

**Contrastive hard negatives**: Each positive oracle selection is paired with a synthetically generated suboptimal selection — one where a key misconception correction is dropped, or where redundant items are included. The model sees both and learns to prefer the oracle's choice.

---

## 3. The Core Tradeoffs

### 3.1 Latency Versus Accuracy: The Pareto Frontier

Running Phase 1 takes sub-millisecond time. Running Phase 2 takes 1–50ms depending on pool size and whether exact ILP is used. Running Phase 3 requires a forward pass through a fine-tuned language model: 10–300ms depending on model size and whether the router runs locally or as an API call.

The empirical question is: how much accuracy does each phase buy, and at what latency and cost? The headline result is a Pareto frontier plot with X = router inference cost (latency × dollars per call) and Y = context-selection accuracy versus oracle. The expectation, prior to running experiments, is that Phase 3 at larger model sizes approaches frontier-API oracle performance but at lower cost, and that there is a model size at which returns diminish — the "Pareto knee."

The design explicitly does not assume which model size is the knee. The data decides.

### 3.2 The Redundancy-Coverage Tradeoff

Diversity in selected context reduces redundancy but risks dropping coverage. If the only two items that explain a critical prerequisite are near-duplicates, a strong redundancy penalty will exclude one — leaving the context with one explanation where two were needed to cover different angles of the concept.

In practice this tension is managed through the threshold parameter in the redundancy penalty. The threshold is set so that true near-duplicates (cosine similarity > 0.92) are penalized, but complementary perspectives on the same concept (similarity 0.7–0.85) are not. The threshold is a hyperparameter tuned on held-out synthetic trajectories.

### 3.3 Synthetic Oracle Training as Distillation

A critical framing issue: the learned router trained on oracle-supervised synthetic trajectories is not learning from the environment. It is learning to imitate a large model's decisions. This is knowledge distillation, not reinforcement from actual student outcomes.

The distinction matters for two reasons. First, the evaluation metric — how closely the small router matches the oracle — measures imitation quality, not downstream student learning gain. Second, the oracle itself can be wrong: a frontier LLM asked to select context items for a student it has never interacted with makes decisions based on the simulated student profile in the trajectory, not ground truth about the real student.

Post-CS153 (Phase B), the training signal should come from real student trajectories: did including item i in turn t lead to measurable learning gain in turn t+1? That is the correct RL signal. For the 17-day window, synthetic oracle distillation is the tractable substitute, and the writeup is explicit about this limitation.

---

## 4. Concrete Examples from Learning Memory OS

### 4.1 The Heuristic Scoring Formula in Practice

Suppose the student's state shows: mastery(kv_cache) = 0.45, mastery(attention) = 0.8; active misconception = "KV cache stores activations, not key-value projections"; last session 2 days ago. The task is "explain why KV cache reduces memory pressure in multi-turn generation."

Candidate pool contains, among others:

- Item A: concept definition of KV cache (tokens: 180)
- Item B: misconception correction "KV cache stores key-value projections, not activations" (tokens: 90)
- Item C: worked example of KV cache memory savings calculation (tokens: 220)
- Item D: concept definition of attention mechanism (tokens: 200, high cosine similarity)
- Item E: a different misconception about paged attention (tokens: 80, low cosine similarity)

Phase 1 scoring: Item B gets a large misconception_priority boost because it directly addresses the student's active misconception. Item A gets strong relevance + moderate recency. Item D gets strong relevance (high cosine to "explain attention") but no prerequisite boost because attention mastery is already 0.8. Item E gets low score across all dimensions. Item C gets moderate relevance + historical reuse success.

Under a 600-token budget, Phase 1 might select B + A + C (550 tokens), excluding D because attention mastery is high and E because it is irrelevant.

Phase 2 would additionally penalize if two concept definitions of KV cache were near-duplicates in the pool, preferring diversity.

### 4.2 The Synthetic Trajectory Format

Each training example for the learned router is a structured record:

```json
{
  "student_state": {
    "mastery": {"kv_cache": 0.45, "attention": 0.8, "transformer_architecture": 0.9},
    "active_misconceptions": ["KV cache stores activations, not key-value projections"],
    "recent_sessions": ["session_42_kv_cache_quiz", "session_41_kv_cache_explain"]
  },
  "task": {
    "type": "explain",
    "topic": "kv_cache",
    "query": "Why does KV cache reduce memory pressure in multi-turn generation?"
  },
  "candidate_pool": [
    {"id": "item_A", "type": "concept", "body": "KV cache stores...", "tokens": 180},
    {"id": "item_B", "type": "misconception", "body": "Common error: ...", "tokens": 90},
    ...
  ],
  "oracle_selection": ["item_B", "item_A", "item_C"],
  "oracle_reasoning": "Misconception correction is highest priority given active error. Concept + example cover explanation. Attention concept omitted — mastery already high."
}
```

The router is trained on the `(student_state + task + pool) → oracle_selection` mapping. The oracle reasoning is kept in the training data for debugging and can optionally be used in a chain-of-thought variant.

---

## 5. Common Misconceptions

### Misconception 1: "More context is always better."

**Correction**: Beyond the token budget, irrelevant context actively degrades output quality. Even within budget, including low-relevance items dilutes the signal for the language model generating the response. Studies on long-context LLMs consistently show that models perform better on focused, high-relevance contexts than on padded, low-signal contexts of the same total length. The "lost in the middle" phenomenon — where models fail to use information positioned in the middle of long contexts — is a concrete mechanism. Context selection is not about keeping context small for efficiency alone; it is about keeping context high-signal for quality.

### Misconception 2: "Retrieval is enough; selection is an overengineering."

**Correction**: Classical retrieval (top-K by cosine similarity) optimizes for semantic closeness to the query. Context selection optimizes for downstream task quality given the student's current state. The two objectives diverge precisely when a student needs a misconception correction (which may be semantically distant from the query) or when retrieval returns 5 near-duplicate phrasings of the same concept (wasteful under a budget). The three-phase selector design was motivated empirically by observing retrieval failures of exactly these kinds on real tutoring trajectories.

### Misconception 3: "Bigger router model is always better."

**Correction**: The Pareto frontier exists because router inference itself has a cost — latency and dollars per call. A 7B router may have marginally higher selection accuracy than a 1.5B router, but if the 1.5B router achieves 95% of the oracle's accuracy at 10% of the inference cost, the 1.5B model dominates on the Pareto frontier. The design explicitly does not assume the largest model wins. The empirical question is where diminishing returns set in — the "Pareto knee." Different deployment settings (cost-sensitive vs latency-insensitive) will prefer different operating points on this frontier.

### Misconception 4: "The learned router is doing true learning from experience."

**Correction**: A router fine-tuned on oracle-supervised synthetic trajectories is performing knowledge distillation, not reinforcement learning from outcomes. The supervision signal is "what would a large LLM have selected?" not "what selection led to improved student mastery?" Distillation is valuable and tractable, but its ceiling is bounded by the oracle's judgment on synthetic data, not by real student learning trajectories. True learning from experience requires online updates from real-user feedback — future work for Phase B.

### Misconception 5: "The token budget only constrains what the model can read."

**Correction**: The token budget constrains the entire model call: system prompt, selected context, task prompt, and generated response all count toward the context window. In practice, the tutor's fixed prompt overhead plus a generous output budget leaves a smaller effective window for selected context than the raw window size suggests. A 32K context window might leave only 12K tokens for selected curriculum items once the prompt template and response buffer are accounted for. Context selection must target the *effective* content budget, not the raw model limit.

---

## 6. Open Questions

**Mixing synthetic and real trajectories**: The current training pipeline is 100% synthetic. As the system accumulates real student sessions, how should real and synthetic trajectories be mixed? Real trajectories are scarce (n=1 student for student-zero) but carry ground-truth learning signal. Synthetic trajectories are abundant but are oracle-distillation. One approach: warm-start on synthetic, fine-tune further on real. The right mixing ratio is empirically unknown.

**Online updates to the router**: The trained router is static after fine-tuning. But students evolve: a student who struggled with KV cache in session 3 may be fluent by session 8. The router's learned context selection policy should update accordingly. Online LoRA adaptation from recent session logs is the natural mechanism, but it requires careful regularization to avoid forgetting general selection quality.

**Evaluating under distribution shift**: The router is trained on synthetic trajectories drawn from a particular distribution of student profiles, topics, and task types. Real students may have profiles outside this distribution. Measuring router performance on out-of-distribution students — and understanding which selection failures are distribution-shift failures versus fundamental limitations — requires a held-out real-user eval set, which student-zero alone cannot provide.

**The cost of wrong selections**: Current evaluation measures selection accuracy (Jaccard vs oracle) and downstream quality (LLM-judge score). Neither directly measures harm from wrong selections — e.g., including an item that introduces a new misconception while trying to correct an existing one. Adversarial evaluation of selection failures is an underexplored direction.

---

## Summary

Context selection under budgets is a constrained combinatorial optimization problem that retrieval alone does not solve. The three-phase design — heuristic ranking, combinatorial knapsack, and learned router — progressively improves selection quality at increasing computational cost. The central empirical result is the accuracy-vs-cost Pareto frontier across router model sizes, benchmarked against retrieval-only and full-context baselines. Learned routers trained on oracle-supervised data are performing distillation, not RL; the training signal is a strong LLM's judgment, not real student learning outcomes. The recursion is central to the project: Learning Memory OS uses the three-phase context selection architecture it teaches in this topic to select context for each tutoring turn it generates.
