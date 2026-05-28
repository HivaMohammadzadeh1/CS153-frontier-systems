# Learning Memory OS: A Context-Routed Tutor for ML Systems Engineers

**Design spec — 2026-05-12**
**Author**: Hiva Mohammadzadeh
**Course**: CS 153 (Stanford, Spring 2026)
**Submission deadline**: 2026-05-29 (17 days from spec date)

---

## 1. Project thesis

### Mission framing (long-term)

Train ML systems engineers. The 17-day CS153 project is the first wedge — a working system that successfully trains one engineer (the author) on the foundations of agent systems, frontier inference, frontier training, and end-to-end ML systems. Future work scales to cohorts, broader curriculum, and real user-trajectory training data.

### Project thesis (the 17-day artifact)

A context-orchestrated multi-agent tutoring system that maintains structured knowledge state for a learner studying ML systems engineering, with **fine-tuned small-model learned context routers benchmarked across model sizes** to find the accuracy-vs-cost Pareto frontier. The system assembles task-specific prompts (explanation, exercise, quiz, review) under fixed token and latency budgets.

### Technical claims

1. A learned context router fine-tuned on synthetic tutoring trajectories beats retrieval-only, full-history, heuristic-ranking, and combinatorial-selection baselines on (a) learning gain measured via pre/post quizzes (student-zero), (b) misconception resolution rate, (c) redundancy avoidance — under fixed token and latency budgets.
2. There exists a small-model size at which the fine-tuned router approaches frontier-API (Claude/GPT) performance on context-selection accuracy at materially lower inference cost. The Pareto frontier across model sizes is the headline empirical result; the exact crossover point is to be measured, not assumed.

### Known limitations (acknowledged upfront)

- **Synthetic-only training** for the router means the supervision signal is itself an LLM oracle. This is effectively distillation of large-model context judgment into a smaller model, not learning from real student outcomes. Real-trajectory training is Phase B (post-CS153).
- **Student-zero evidence is n=1.** Learning gain measured on the author is a case study, not a statistical claim. Cohort evidence (n=2–3) is qualitative supplementary evidence, not a second eval arm.
- **17-day window forces depth/breadth tradeoffs.** Deep evaluation runs on 6 topics; the other 14 are content-loaded but not deeply evaluated.

### Curriculum spine (authoritative sources)

- **CS153** (Stanford, Spring 2026) — frontier-systems framing (bottlenecks, AI factory loop, agents)
- **CS336** (Stanford, *Language Modeling from Scratch*) — training from scratch (architecture → kernels → distributed → alignment)
- **CS349D** (Stanford, *AI Inference Infrastructure*) — inference at scale (serving → batching → caching → disaggregation)

### The recursion (headline narrative)

The techniques the system teaches in Area E (Agent systems) — multi-tier memory, context selection under budgets, multi-agent orchestration — are the same techniques used to build it. The author is both architect and student-zero. The demo arc: *"The techniques this tutor teaches in Area E are the techniques I used to implement it. I am both its author and its first student. Here is my measured trajectory across the deep-eval topics."*

### Project track

Automation / Agent Systems.

### Resume bullet (the end state)

> Built Learning Memory OS — a context-orchestrated multi-agent tutoring system for ML systems engineering. Fine-tuned context routers from 0.5B–7B parameters on 50K+ synthetic tutoring trajectories; established an accuracy-vs-cost Pareto frontier vs. frontier-API and classical-IR baselines. Self-trained as student-zero on a 20-topic curriculum drawn from CS153, CS336, and CS349D.

---

## 2. Architecture

Five layers.

### 2.1 Ingestion layer

Parses heterogeneous inputs into structured artifacts.

**Inputs**:
- CS336 lecture videos (Whisper transcripts), slides, assignment PDFs
- CS349D lecture materials, paper reading list, serving-engine starter code
- CS153 lectures (frontier-systems framing)
- Canonical papers per topic (FlashAttention, vLLM/PagedAttention, GPTQ, DPO, etc.)
- Authoritative blogs (HuggingFace, vLLM, Together AI, EleutherAI)
- Student-generated: notes, quiz attempts, exercise/code submissions

**Output schema** (structured artifacts):
- `concept` — name + definition + deep explanation
- `prerequisite` — directed edges between concepts
- `example` — worked examples per concept
- `common_misconception` — paired with correction
- `exercise` — lab task + starter code + rubric
- `code_pattern` — reusable code idioms (e.g., "FSDP wrap policy")
- `paper_claim` — claim + source + evidence

Generation: strong-LLM (Claude Opus / GPT-5) with strict JSON schema prompts; human curation pass (~30 min/topic).

### 2.2 Multi-tier memory store

| Tier | Contents | Lifetime |
|---|---|---|
| Semantic | Stable course/topic facts, concept defs, paper claims | Persistent |
| Student | Per-concept mastery state, persistent misconceptions, skill traces | Persistent per student |
| Episodic | Recent tutoring sessions, quiz attempts, exercise outcomes | Rolling window |
| Intervention | What tutoring strategy was used for which misconception, did it work | Persistent per student |

Backend: Postgres + pgvector (embeddings on concept defs, episodic events, misconceptions).

### 2.3 Context routing engine (the centerpiece — three phases)

**Phase 1 — Heuristic ranker + budgeted packing** *(must-ship MVP)*

Score each candidate memory item by weighted sum of:
- Semantic relevance to current task (embedding cosine)
- Recency (decay function)
- Misconception-priority (boost if item resolves an active misconception)
- Prerequisite-dependency (boost if item is a prerequisite of the current topic)
- Reuse history (boost if previously cited successfully)

Pick top-K under token budget.

**Phase 2 — Combinatorial selector**

Knapsack-style optimization:

```
maximize  Σ utility_i · x_i  -  λ · redundancy(S)  +  μ · dependency_coverage(S)
subject to  Σ tokens_i · x_i  ≤  B
            x_i ∈ {0, 1}
```

Solved via greedy + local search, or small ILP (PuLP).

**Phase 3 — Fine-tuned learned router** *(the headline)*

Model: Qwen-2.5-Instruct family, fine-tuned at multiple sizes (0.5B, 1.5B, 3B, 7B).

Fine-tuning method: **LoRA** (rank 16, default Hugging Face PEFT config), not full fine-tuning. This is what makes 7B feasible in the time window; full FT at 7B is out of scope.

Input: serialized `(student_state, task, candidate_pool)`.
Output: subset of candidate items, ranked.

Trained on 50K+ synthetic tutoring trajectories with oracle-supervised target selections plus contrastive hard negatives.

### 2.4 Specialized agents

| Agent | Role |
|---|---|
| Tutor | Explains concepts conditioned on student state |
| Diagnostic | Locates weaknesses from quiz/exercise output |
| Quiz generator | Produces targeted quizzes for weak concepts |
| Lab generator | Produces hands-on exercises with starter code + rubric |
| Study planner | Decides next session content given mastery state |
| Critic / consistency | Catches contradictions with prior tutor outputs |

All agents call the routing engine to assemble their input context; none receive raw global state.

### 2.5 Evaluation harness

Instruments every system interaction:
- Task type, candidate pool, scores, selected items, dropped items, token cost, latency
- Final output + LLM-judge score against rubric
- Whether student corrected the output / asked for re-explanation
- Whether contradictions were detected by the critic

Drives both the metrics dashboard and the data flywheel for future router training.

---

## 3. Curriculum

**20 topics across 5 areas**, drawn from three authoritative Stanford courses + the project's own recursion area.

### Area A — Model fundamentals *(CS336 L1–L4)*
1. Tokenization (BPE, byte-level, vocab tradeoffs)
2. Transformer architecture & hyperparameters
3. Attention variants & MoE
4. Resource accounting (FLOPs, memory, arithmetic intensity)

### Area B — Training systems *(CS336 L5–L8, L11; Assignment 2)*
5. GPU/TPU hardware & kernels (Triton, FlashAttention2)
6. Data parallelism (DDP)
7. Sharded training (FSDP / ZeRO-1/2/3)
8. Model parallelism (tensor + pipeline)
9. Scaling laws

### Area C — Inference infrastructure *(CS349D + CS336 L10)*
10. KV cache & PagedAttention
11. Quantization (int8/int4, GPTQ/AWQ)
12. Speculative decoding
13. Continuous batching & request scheduling
14. Prefill-decode disaggregation & hierarchical caching

### Area D — Data & alignment *(CS336 L13–L17; Assignments 4–5)*
15. Pretraining data (collection, dedup, filtering)
16. SFT + RLHF / DPO
17. RL systems for reasoning

### Area E — Agent systems & frontier framing *(CS153 + the recursion)*
18. **Agent memory architectures** ← system embodies this
19. **Context selection under budgets** ← system embodies this
20. **Multi-agent orchestration & long-horizon execution** ← system embodies this

### Capstone exercise (post-CS153, not in scope)

**Build a mini-vLLM** — the CS349D headline deliverable. Out of scope for the 17-day window (multi-week effort to implement a serving engine end-to-end). Listed as the canonical Phase B capstone the system will guide future students through. Mentioned in the writeup's future-work section.

### Per-topic content

- Concept def + deep explanation
- 3 worked examples
- 5 common misconceptions w/ corrections
- 8 quiz questions (MC + short-answer)
- 1 lab/exercise w/ rubric (Areas A–D have hands-on; Area E labs = modify Learning Memory OS itself)
- Prerequisite links

Content generation: strong-LLM-drafted, ~30 min/topic curation, ~10 hours total effort, parallel to system build.

### Deep-evaluation subset

The full 20 topics are content-loaded. Deep pre/post quiz cycles and detailed router ablations run on **6 topics** to keep evaluation tractable: KV cache, quantization, speculative decoding (Area C); memory, context selection, orchestration (Area E). These map directly to the recursion narrative.

---

## 4. Evaluation framework & data strategy

### 4.1 Three data streams

| Stream | Source | Purpose |
|---|---|---|
| Corpus | CS336 + CS349D + CS153 materials + papers + blogs | Feeds semantic memory |
| Synthetic trajectories | LLM-generated tutoring sessions (50K+) | Trains the fine-tuned routers |
| Real user logs | Author (student-zero) + light cohort | Evaluation evidence |

### 4.2 Synthetic trajectory generation

Pipeline:
1. Sample student profile (background, weak topics, learning style)
2. Sample target topic + task type (explain / quiz / lab / review)
3. Sample current student-memory state (mastery vector + active misconceptions)
4. Oracle step: strong LLM with full memory access decides which subset of candidate items *should* be in context for this task
5. Record `(state, pool, task, oracle_selection, outcome)` tuple
6. Hard negatives: generate suboptimal selections and downstream failures for contrastive signal

Output format: SFT pairs `(state + pool + task) → selected_item_ids`. Fine-tune Qwen-2.5-Instruct across sizes on these.

Target: 50K+ trajectories.

### 4.3 Metrics

**Primary (synthetic eval set — has statistical power)**
- Context-selection accuracy vs oracle (precision, recall, Jaccard on selected item IDs)
- Downstream task quality (LLM-judge against rubric on tutor outputs)
- Token cost per task
- Latency per task

**Primary (student-zero — n=1 case study, descriptive only)**
- Learning gain: pre/post quiz delta per deep-eval topic
- Misconception resolution rate across the author's session arc (≥3 sessions for student-zero)
- Self-reported difficulty / clarity per session

**Secondary diagnostics**
- Redundancy of selected context
- Omission of critical context (oracle-vs-selected gap)
- Consistency with prior student state
- Tutor-vs-expert agreement on a small annotated subset

**Cohort (qualitative supplement, not a quantitative arm)**
- Qualitative session reports from 2–3 classmates; verbatim quotes in the writeup; *not* averaged into metrics.

### 4.4 Baselines (two ablation tables, separated for clarity)

**Strategy comparison** *(single number per cell — best variant of each strategy)*

| Strategy | Description |
|---|---|
| B0 Full-context | No selection; truncate when over budget |
| B1 Retrieval-only top-K | Embedding similarity, top-K |
| B2 Summary-only | Rolling summary + always-on slots |
| B3 Heuristic ranker | Phase 1 of the engine |
| B4 Combinatorial selector | Phase 2 of the engine |
| B5 Fine-tuned router (best size) | Phase 3 of the engine |

**Router size sweep** *(B5 only — drives the Pareto plot)*

| Variant | Notes |
|---|---|
| Qwen-2.5-0.5B LoRA | Cheapest |
| Qwen-2.5-1.5B LoRA | |
| Qwen-2.5-3B LoRA | |
| Qwen-2.5-7B LoRA | Largest in scope |
| Frontier-API (Claude / GPT) | Zero-shot upper bound |

All evaluated at token budgets: **4K, 8K, 16K, 32K**.

### 4.5 Headline result: accuracy-vs-cost Pareto frontier

**Figure 1 of the writeup**: X = router inference cost (latency × dollars-per-call), Y = context-selection accuracy vs oracle (or downstream tutor quality, whichever shows the cleaner separation).

The Pareto frontier across {B0, B1, B2, B3, B4, B5-{0.5B, 1.5B, 3B, 7B}, B5-frontier-API} is the headline empirical result. The writeup's story is *whichever finding the data supports*, drawn from this menu:

- **If the smallest model is competitive**: "A LoRA-tuned 0.5B router approaches frontier-API context selection at ≪ inference cost."
- **If a mid-size model is the sweet spot**: "The Pareto knee sits at 1.5–3B; beyond that, returns diminish."
- **If frontier-API wins decisively**: "Learned small-model routers close most but not all of the gap; we characterize the residual."

All three are publishable framings. The design does *not* assume which one wins.

### 4.6 Evidence streams for the writeup

1. Pareto-frontier plot (Figure 1)
2. Learning-gain vs token-budget curves, all baselines
3. Misconception-resolution curve over sessions
4. Student-zero trajectory: full progression across 6 deep-eval topics with mastery state visualized
5. Cohort evidence (if cohort survives cut policy): 2–3 classmates × 1–2 sessions, qualitative report
6. Failure analysis: top-3 cases where the router failed + why
7. *Optional*: CS336 Assignment 1 case study (author completes Assignment 1 with system assistance; time + misconceptions documented)

---

## 5. 17-day timeline

Day 0 = 2026-05-12. Submission = 2026-05-29 (Day 17).

### Phase 0 — Setup (Day 0–1)
- Repo, dev env, GPU access verified
- Pull CS336/CS349D/CS153 materials, Whisper transcripts of lecture videos
- Memory backend (Postgres + pgvector)
- **Checkpoint D1**: ingestion pipeline working on 1 lecture end-to-end

### Phase 1 — MVP system (Day 2–5)
- Ingestion → structured artifacts (LLM extraction, schema'd)
- Multi-tier memory (4 tiers)
- Heuristic ranker + budgeted packing (Phase 1 selector)
- Tutor agent + logging
- 4 topics loaded end-to-end (1 per area A/B/C/E)
- **Checkpoint D5**: ask system a question → context selected → tutor responds → log shows what was selected. *This is the TA's "Week 6 MVP."*

### Phase 2 — Curriculum content (Day 4–8, overlaps Phase 1)
- LLM-generate per-topic artifacts for all 20 topics
- Author curation pass (~30 min/topic, ~10 hours total)
- Pre/post quizzes finalized for 6 deep-eval topics
- **Checkpoint D8**: all 20 topics loaded; 6 deep-eval topics have full quiz materials

### Phase 3 — Synthetic data + router training (Day 6–12, overlaps Phase 2)
- Synthetic-trajectory generator (D6–7)
- Generate 50K+ trajectories (D7–9, mostly API wait time)
- Fine-tune Qwen-2.5 at 0.5B, 1.5B, 3B, 7B (D9–12, GPU-bound)
- Frontier-API-as-router baseline (D10)
- **Checkpoint D12**: all 5 router variants trained + evaluated on held-out synthetic set

### Phase 4 — Combinatorial selector + ablations (Day 10–13, overlaps Phase 3)
- Phase 2 selector (knapsack with redundancy/dependency)
- Full 6-baseline × 4-budget ablation on synthetic eval set
- **Checkpoint D13**: ablation table populated end-to-end

### Phase 5 — Real-user data (Day 8–15, background)
- Author starts using the system day 8, ≥1 session/day, logs everything (target ≥6 sessions for student-zero)
- Recruit 2–3 classmates D9–10; 1–2 sessions each across D10–15
- Pre/post quizzes administered to all real users (different question pools to avoid memorization)
- **Phase 2 dependency**: requires ≥4 of 6 deep-eval topics fully content-loaded by D8. If Phase 2 slips, student-zero start slips proportionally; the final eval window compresses, not the cohort.
- **Checkpoint D15**: real-user evidence captured

### Phase 6 — Writeup + demo + polish (Day 14–17)
- Paper-style technical report (problem → architecture → data → experiments → results → failure analysis → future work)
- Demo video (3–5 min): student-zero arc, system in action, headline plots
- Polished GitHub README + repo structure
- One-page project page (pitch, diagram, demo gif, key chart, repo link)
- **Submission Day 17 (2026-05-29)**

---

## 6. Graceful-degradation policy

If behind schedule at any checkpoint, cut in this order:

| If behind by Day… | Cut |
|---|---|
| D5 | Drop combinatorial selector (B4); keep B3 + B5 only |
| D8 | Drop areas A + B + D to 2 topics each (load content but no deep eval); deep eval already scoped to C + E per §3 |
| D10 | Drop 0.5B + 7B router sizes; benchmark only 1.5B + 3B |
| D12 | Drop classmate cohort entirely; student-zero + synthetic only |
| D14 | Drop CS336 Assignment 1 case study; pure synthetic + author eval |

**Never cut**: fine-tuned router (at least 1 size), 6-baseline ablation at 1 budget, student-zero trajectory, demo video, paper writeup.

The Pareto plot survives as long as ≥2 router sizes survive.

---

## 7. Open questions to resolve during execution

These were deferred from spec to implementation because they need empirical or external answers:

1. **Exact GPU budget**: total GPU-hours available via AMP sponsorship determines feasible LoRA fine-tune sizes (especially 7B). Confirm before D6 (router-training start). If sub-A100, drop to 0.5B + 1.5B only.
2. **Frontier-API budget for B5-frontier-API baseline**: thousands of Claude/GPT calls during eval are real money (estimated $50–$300 depending on call count and model). Confirm budget before D10. If unbudgeted, fall back to a cheaper frontier-API tier (Sonnet, GPT-4o-mini) and document choice.
3. **Memory-tier cardinality**: how many items live in each tier per student before retrieval is needed? Determines whether vector search is overkill for early MVP.
4. **Critic-agent integration timing**: critic adds latency. May get deferred to post-MVP if it doesn't fit cleanly into the routing pipeline.
5. **Real-cohort recruitment**: realistic count by D10. If zero classmates engage, student-zero + synthetic carries the eval.
6. **CS349D lecture availability**: course is Spring 2026 (current quarter); not all lectures may be published by D7. Fall back on the syllabus + reading list + canonical papers if needed.
7. **Stanford data-handling for cohort**: collecting classmates' quiz responses and tutoring transcripts for a class project is almost certainly fine, but confirm with course staff that no IRB-style consent process is required before D9 recruitment.

---

## 8. What this leaves room for (post-CS153, Phase B)

Phase B of the larger mission (not in scope for 17 days):
- Add deeper coverage of training systems with real distributed-compute labs
- Scale to 10–20 students across multiple cohorts
- Train larger routers on **real** user-trajectory data, not just synthetic — directly addresses the §1 limitation about distillation
- Build the "mini-vLLM" capstone exercise into a full graded curriculum module
- Add online learning: router updates from live student feedback
- Publish as a workshop paper / open-source benchmark

The CS153 writeup will explicitly name this future work and tie it to the mission of training ML systems engineers at scale.
