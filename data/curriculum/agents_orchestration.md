# Multi-Agent Orchestration and Long-Horizon Execution

**Area E — Agent Systems & Frontier Framing | Learning Memory OS Curriculum**

---

## 1. Why Orchestration Matters

A single monolithic agent with all the tools is the default design choice when building a system for the first time. It is easy to reason about, easy to debug, and easy to iterate on during prototyping. It is also the wrong architecture for long-horizon tasks.

Long-horizon tasks are sessions that unfold over many turns, track evolving state, and require qualitatively different capabilities in different turns: diagnosing what a student knows in turn 1, generating a targeted quiz in turn 2, checking consistency of a student's quiz answer against a misconception history in turn 3, generating a hands-on lab exercise in turn 4. No single monolithic prompt template handles all of these well. More importantly, a monolithic agent trying to handle all of them accumulates state across turns in an ad hoc way — pasting prior turns into the context window — and eventually runs out of budget, degrades from irrelevant context, or produces inconsistent outputs because the agent has no structured way to know what it previously said.

Multi-agent orchestration addresses this by decomposing a long-horizon workflow into specialized agents, each of which:

1. Has a narrow, well-defined role.
2. Receives a curated context assembled by the routing engine rather than global state.
3. Produces structured output that updates shared memory, not ad hoc text concatenation.

The design principle is: agents are stateless over their own calls; state is externalized into the memory store and retrieved on demand via the routing engine.

---

## 2. Specialist Agent Taxonomy

Learning Memory OS defines six specialist agents, each with a distinct task type and context profile.

### 2.1 Tutor Agent

**Role**: Generate an explanation of a concept conditioned on the student's current mastery state and the curated context assembled by the selector.

**Context it needs**: The concept definition and deep explanation, prerequisite concept summaries (if mastery is low), relevant worked examples, active misconceptions and their corrections, recent session summaries.

**Context it does not need**: Global student history going back to session 1, unrelated topic content, quiz templates, lab starter code.

**Output**: A targeted explanation that directly addresses the student's current gap and corrects any active misconceptions, without repeating material the student already knows well.

### 2.2 Diagnostic Agent

**Role**: Analyze a quiz attempt or exercise submission and locate weaknesses — specific concepts the student answered incorrectly, misconceptions the answer reveals, and prerequisite gaps implied by the errors.

**Context it needs**: The quiz questions and the student's answers, the rubric for each question, the concept definitions for the tested topics, known active misconceptions (to detect whether an answer reflects a pre-existing misconception or a new one).

**Context it does not need**: Tutor explanations from prior sessions (it is assessing the student's current state, not reviewing past tutoring), worked examples, lab starter code.

**Output**: A structured diagnostic report: per-concept mastery update, new misconceptions to flag, prerequisite gaps to address. This output is written directly to the student memory store.

### 2.3 Quiz Generator Agent

**Role**: Produce a set of quiz questions targeting the student's weakest concepts in the current topic.

**Context it needs**: The concept definitions for weak concepts, common misconception statements (to use as attractive distractors in multiple-choice questions), worked examples (to derive question scenarios from), the student's prior quiz history (to avoid repeating questions already answered correctly).

**Output**: A structured quiz with correct answers, distractors, and the concept each question targets.

### 2.4 Lab Generator Agent

**Role**: Produce a hands-on exercise with starter code and a rubric, targeting a specific concept.

**Context it needs**: The concept definition, prerequisite concepts (whose implementations the lab can build on), code patterns relevant to the concept, the exercise template for the topic.

**Output**: A lab specification: problem statement, starter code, expected behavior, rubric criteria.

### 2.5 Study Planner Agent

**Role**: Decide the order and pacing of topics for the student's next session given the current mastery vector.

**Context it needs**: The student's per-topic mastery scores, the prerequisite graph for all topics, the study plan history (what was scheduled vs what the student actually completed), time-to-deadline.

**Output**: A session plan: ordered list of topics to cover, estimated time per topic, recommended task type per topic (explain, quiz, or lab based on mastery thresholds).

### 2.6 Critic / Consistency Agent

**Role**: Detect contradictions between a newly generated tutor response and prior tutor outputs stored in episodic memory.

**Context it needs**: The newly generated tutor response, recent prior tutor explanations of the same and related concepts, the student's misconception history.

**Output**: A consistency report: flagged contradictions (if any), corrected phrasing, or a clean pass. If contradictions are found, the orchestrator routes the output back to the Tutor Agent for revision before delivery.

The Critic Agent is latency-sensitive: it adds one extra LLM call per turn. In the MVP, it is deployed only for explanation turns (not quiz generation, where contradictions are less dangerous), and only when the tutor is covering a concept it has explained before in the session arc.

---

## 3. The Base Agent Pattern

All six agents share a contract: they call the routing engine to assemble their input context; they do not access the raw memory store directly.

This is not just a software engineering boundary. It is a correctness guarantee. If agents were allowed to read global state directly, they would see items that are irrelevant to their current task, run over budget, and produce outputs conditioned on noise. The routing engine's job is to enforce the budget, prioritize items by task-specific relevance, and suppress redundant content. Agents that bypass the router bypass all of this.

The pattern in code looks like:

```python
# Tutor agent pseudocode
def explain(student_id: str, topic: str, query: str) -> str:
    student_state = student_store.load(student_id)
    candidate_pool = semantic_store.candidates_for_topic(topic)
    selected_items = routing_engine.select(
        student_state=student_state,
        task=ExplainTask(topic=topic, query=query),
        pool=candidate_pool,
        budget_tokens=settings.context_budget,
    )
    context_text = format_items_for_prompt(selected_items)
    response = llm.complete(system=TUTOR_SYSTEM_PROMPT, context=context_text, user=query)
    log_interaction(student_id=student_id, task="explain", topic=topic,
                    selected_ids=[i.id for i in selected_items], response=response)
    return response
```

The routing engine call is the single entry point to memory. The agent never reads from `semantic_store` directly.

---

## 4. Long-Horizon Execution: What to Remember, Summarize, Retrieve, and Discard

Long-horizon execution introduces a decision problem that does not arise in single-turn agents: what happens to information across turns?

### 4.1 The Four-Tier Memory Architecture

Learning Memory OS maintains four memory tiers with distinct lifetimes and access patterns:

| Tier | Contents | Lifetime |
|---|---|---|
| Semantic | Stable course/topic facts, concept definitions, paper claims | Persistent (curriculum load) |
| Student | Per-concept mastery state, persistent misconceptions, skill traces | Persistent per student |
| Episodic | Recent tutoring sessions, quiz attempts, exercise outcomes | Rolling window (~20 events) |
| Intervention | What tutoring strategy was used for which misconception, whether it worked | Persistent per student |

The semantic tier is written at ingestion time and never modified during a session. The student tier is updated at the end of each diagnostic pass. The episodic tier is a circular buffer; events older than the window fall out. The intervention tier accumulates a record of what the system tried and whether it worked.

### 4.2 Decision Logs as Training Signal

Every orchestrated turn produces a decision log entry: which agent ran, which items were selected, what scores and budget were used, what the LLM response was, and what (if any) student feedback or correction followed. These logs are stored in JSONL format.

Decision logs serve a dual purpose. In the short term, they are debugging artifacts: when the tutor produces a poor explanation, the log shows exactly which items were selected and why, enabling post-hoc analysis. In the long term, they are the training signal for future learned routers: a real student's correction ("that explanation was wrong") paired with the decision log that generated the bad response is a negative training example. A post-session quiz gain paired with the selection decisions that preceded it is a positive training example.

This is the data flywheel: every real tutoring session improves the training data for future router training. The logs are not for debugging — they are the raw material for learning.

### 4.3 Episodic Memory and Summarization

The episodic buffer stores events as structured records: session timestamp, task type, topic, response summary, outcome (correct / incorrect / corrected / no feedback). The rolling window has two roles.

First, it provides recency signals to the heuristic scorer. Items from recent sessions that produced successful interactions get a reuse-success boost. Items from recent sessions that produced corrections get a small penalty.

Second, it feeds the Critic Agent. When the Critic checks for consistency, it reads the episodic buffer's recent tutor explanations to detect contradictions with the current output. Without the episodic buffer, the Critic would have no memory of what was said before.

When the episodic buffer reaches capacity, old events are not silently discarded — they are summarized. A summary agent (or a structured compression of the event record) writes a compact representation of the early session arc into the student tier. This ensures that long-horizon effects (a student who misunderstood concept X in session 1 but corrected it by session 5) are preserved without the full session text consuming the memory budget.

---

## 5. The Recursion

Learning Memory OS teaches multi-agent orchestration in this topic. It is also a multi-agent system. The connection is not incidental — it is the narrative spine of the project.

The Tutor Agent that explains this document's content to a student is itself running the three-phase context selector described in the preceding topic (context_selection). Its explanation is checked by the Critic Agent for consistency with what it said in prior sessions. The diagnostic of the student's comprehension is handled by the Diagnostic Agent. The quiz to assess retention is generated by the Quiz Generator Agent. The decision of whether to move to the next topic or revisit this one is made by the Study Planner.

The student is interacting with a multi-agent system while learning what a multi-agent system is. This is the recursion: the techniques Area E teaches — multi-tier memory, context selection under budgets, specialist agent decomposition, intervention memory — are the same techniques used to build the system that delivers the teaching.

The demo arc for the CS153 presentation captures this directly: "The techniques this tutor teaches in Area E are the techniques I used to implement it. I am both its architect and its first student. Here is my measured trajectory across the 6 deep-eval topics."

---

## 6. Observable Orchestration

A recurring failure mode in multi-agent systems is opacity: agents run, produce outputs, update state, and the system is functionally correct but uninterpretable when something goes wrong. Debugging requires reconstructing decisions from opaque LLM calls.

Learning Memory OS addresses this through instrumented orchestration. Every agent call logs:

- The agent type and task
- The candidate pool size before and after selection
- The selected item IDs, their scores, their types (concept / misconception / example), and their token costs
- The budget used and the budget remaining
- The LLM response (truncated at 500 characters in the log)
- The critic outcome (pass / flagged contradiction)
- The student feedback event if one occurred (within the next turn)

The evaluation harness can reconstruct the full decision trace for any turn from the JSONL log. This enables ablation analysis: which selections led to high-quality outputs? Which misconception corrections were included but failed to resolve the misconception? Where did the router disagree most from the oracle?

Observable orchestration is not a nice-to-have; it is the prerequisite for the data flywheel. Without full decision traces, real-user trajectories cannot be converted into training examples for future router fine-tuning.

---

## 7. Common Misconceptions

### Misconception 1: "One big agent with all the tools is the best architecture."

**Correction**: Monolithic agents struggle on long-horizon tasks because they accumulate state as raw text concatenation, degrade as context fills, and apply the same reasoning pattern to qualitatively different subtasks (diagnosis vs explanation vs quiz generation). Specialized agents with curated context consistently outperform monolithic agents in long-horizon tutoring settings because each agent operates under a focused, high-signal context window rather than a growing dump of prior turns. The routing engine is what makes specialization practical: each agent gets exactly the memory it needs, not everything the system has ever stored.

### Misconception 2: "Agents should share global state to coordinate."

**Correction**: Agents sharing global state creates tight coupling: every agent must understand the full schema of the shared state, every write from one agent can corrupt the context of another, and debugging requires tracing interactions across all agents simultaneously. The correct architecture is shared memory with agent-local context: all agents read from and write to the same memory store, but they never see the raw store directly. The routing engine mediates every read. Writes are structured (typed memory items, not free-form text appended to a shared scratchpad). This gives coordination without coupling.

### Misconception 3: "Logs are for debugging."

**Correction**: In a tutoring system with a data flywheel, logs are the primary training signal for future learned routers. A decision log entry that records which items were selected for a turn, what response was generated, and whether the student subsequently corrected or asked for re-explanation is a labeled training example for router fine-tuning. Treating logs as throw-away debugging artifacts means discarding the only ground-truth signal available for learning from real user interactions. The logging infrastructure must be designed for training-data quality: structured, schema-validated, complete, and efficiently queryable.

### Misconception 4: "Hand-coded orchestration scales."

**Correction**: Hand-coded routing logic (if-else chains deciding which agent to call when) works for small systems with stable workflows. It does not scale to systems where task types, student profiles, and topic areas multiply. Hand-coded logic is also difficult to introspect: it does not produce the decision traces needed for ablation analysis or router training. The correct path is observable, data-driven orchestration: a flow-control layer that makes routing decisions based on student state and logs every decision for analysis. As the system grows, the orchestration logic can be learned (a routing meta-agent) rather than hand-coded.

### Misconception 5: "The Critic Agent is optional polish."

**Correction**: In a multi-session tutoring system, the Critic Agent is a correctness requirement. Without it, a tutor that explains concept X one way in session 2 and a contradictory way in session 6 will confuse the student and undermine trust. Contradiction detection is especially critical for misconception corrections: if the tutor corrects a misconception in session 3 but then inadvertently restates the misconception in session 6 (because the correction item was not selected), the student's progress is actively harmed. The Critic Agent is what prevents this regression. Its latency cost (one additional LLM call) is justified by the correctness guarantee it provides.

---

## 8. Worked Examples

### 8.1 A Single Tutor Turn, End-to-End

Student asks: "Can you explain why speculative decoding speeds up LLM inference without changing output quality?"

1. **Study Planner** (already ran at session start): speculative decoding is scheduled for this session; task type is "explain."
2. **Routing engine** selects context from semantic and student tiers:
   - Concept: speculative decoding definition (selected: high relevance + student mastery 0.3)
   - Misconception correction: "Speculative decoding does not change the output distribution — the verification step guarantees this" (selected: active misconception in student state)
   - Prerequisite concept: autoregressive generation (selected: mastery 0.55, below threshold)
   - Example: latency-vs-throughput tradeoff in speculative decoding (selected: high reuse success)
   - Omitted: KV cache internals (not prerequisite for this task), quiz templates (wrong task type)
3. **Tutor Agent** generates explanation from curated context (budget used: 480/800 tokens).
4. **Critic Agent** checks explanation against prior tutor outputs on speculative decoding: no contradiction found (clean pass).
5. **Log entry** recorded: task=explain, topic=speculative_decoding, selected=[concept, misconception_correction, prerequisite, example], budget_used=480, critic=pass.

### 8.2 What Gets Logged Per Turn

The JSONL log entry for the above turn:

```json
{
  "turn_id": "t_0041",
  "student_id": "student_zero",
  "timestamp": "2026-05-15T14:23:07Z",
  "agent": "tutor",
  "task_type": "explain",
  "topic": "speculative_decoding",
  "selected_items": [
    {"id": "sd_concept_001", "type": "concept", "score": 0.91, "tokens": 195},
    {"id": "sd_misconception_003", "type": "misconception", "score": 0.87, "tokens": 88},
    {"id": "ar_gen_concept_001", "type": "concept", "score": 0.72, "tokens": 120},
    {"id": "sd_example_002", "type": "example", "score": 0.68, "tokens": 240}
  ],
  "budget_tokens": 800,
  "budget_used": 643,
  "critic_outcome": "pass",
  "response_preview": "Speculative decoding separates token generation into two phases...",
  "feedback_event": null
}
```

This log entry is immediately usable as a training example: if the student's next quiz shows mastery(speculative_decoding) improved from 0.3 to 0.65, this selection decision gets a positive label. If the student asks for re-explanation in the next turn, it gets a negative label.

### 8.3 How the Eval Harness Reconstructs Decisions

The evaluation harness reads the full JSONL log and reconstructs, for each session:

- Per-turn selection quality: did the selection include the oracle's top items? (Jaccard vs oracle)
- Misconception coverage: for each active misconception at session start, was the correction item selected in at least one turn?
- Redundancy: what fraction of selected tokens were near-duplicates (cosine > 0.9)?
- Critic catch rate: what fraction of turns with actual contradictions (identified by post-hoc review) were flagged by the Critic?

The harness is driven entirely from JSONL — no live system access required. This makes ablation analysis (comparing Phase 1 vs Phase 2 vs Phase 3 selection decisions on the same session) possible retroactively.

---

## 9. Open Questions

**Planner versus reactive orchestration**: Learning Memory OS uses a session-start planner (Study Planner Agent decides the session agenda) combined with turn-level reactive routing (Routing Engine adapts to what was just asked). The boundary between these is not yet fully principled. If the student's question deviates from the planner's agenda, who wins? A more principled architecture might have the Study Planner run at the start of each turn, not just the session — but this adds latency.

**When to invoke the Critic Agent**: Currently the Critic runs only on explain turns. This is conservative. Quiz and lab tasks can also contain contradictions (e.g., a quiz question that implies a fact the tutor previously stated was false). A lightweight Critic that runs on all turns with a smaller model might be the right tradeoff.

**Evaluating long-horizon coherence**: Short-horizon metrics (per-turn LLM-judge score) do not capture whether a student's 6-session arc is coherent — whether misconceptions were progressively corrected, whether the student's mastery trajectory was appropriately paced, whether the system avoided reinforcing the same correct explanations while neglecting weak areas. Long-horizon coherence metrics require session-level evaluation, not turn-level evaluation, and remain an open research problem.

**The orchestrator as a learnable module**: Currently the orchestration logic — which agent to call in which order, when to invoke the Critic, when to escalate from explanation to quiz — is hand-coded. A meta-agent that learns orchestration patterns from successful session arcs would generalize better. This requires session-level outcome labels (did the student's mastery improve across the arc?) and is Phase B work.

---

## Summary

Multi-agent orchestration decomposes long-horizon tutoring into specialist agents (Tutor, Diagnostic, Quiz Generator, Lab Generator, Study Planner, Critic) that each receive curated context via the routing engine and never access global state directly. The base agent pattern — route-then-call, log everything — is the architectural primitive that makes both correctness and data flywheel possible. Decision logs are not debugging artifacts; they are labeled training data. The recursion is explicit: the multi-agent system described here is the same system used to teach the student reading this topic. Observable orchestration — structured logs, selection traces, critic outcomes — is the prerequisite for ablation analysis, router fine-tuning, and eventual online adaptation from real student trajectories.
