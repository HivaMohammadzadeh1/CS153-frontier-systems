# Memex — demo video script (target ~7–9 min, max 10)

A beat-by-beat script answering the four required questions, with what to show on
screen. Speak to the bolded differentiation — it's what wins the "why not just a
Claude Project?" question.

---

## 0. Cold open (20s)
> "This is Memex — **LeetCode for ML-systems design**. It's an interview simulator that teaches you to think like the engineer on call for a frontier-model serving system: a **staff-engineer AI judge** runs mock interviews, gives you a **calibrated hire-bar verdict**, and a router we **fine-tuned ourselves** keeps the tutor's context tight. We sell proof you'd clear the bar — not answer text."

Show: the app open on **🧭 Interview Readiness** — the gauge, the **hire-bar verdict card** (tier + blended score), area bars, gaps.

---

## Q1 — Why did you build this? (≈1 min)
- **The bottleneck:** ML-systems engineering (CS336/CS349D material — kernels, FSDP, KV cache, MoE, inference serving) is brutal to self-study, and interview prep for it is underserved. People fall back to ChatGPT, which **answers questions but doesn't model the learner** — no idea what you've mastered, what you keep getting wrong, or whether you'd actually pass.
- **Inspiration:** a great interviewer doesn't lecture — they probe, follow up on your weak spot, and decide hire/no-hire against a real bar. Memex turns that into a loop: track mastery, target gaps, confront misconceptions, and *prove* you got better — efficiently, by **selecting** the right context instead of dumping a giant window.

Show: briefly, the Chat answering an ML-systems question with the "Context for this answer" panel, then a **🎯 Test** click that auto-infers the topic from the question.

---

## Q2 — How does it work? (≈3.5–4 min) — the meat

### [1] Research — the learned context router
> "The research core: can a **small** model match a frontier oracle at **context selection** — choosing which memory items to put in the tutor's context under a token budget?"
- Show `config/router_sizes.yaml` + `data/trajectories/val.jsonl` (5,000 oracle trajectories).
- "We LoRA-fine-tuned **four Qwen2.5 sizes (0.5B→7B)** on a shared **32×H100 SLURM cluster**." Show `cluster/finetune_router.sbatch` + a `squeue`/`*.out` snippet.
- Show **`data/eval/pareto.png`**: selection Jaccard vs. the oracle — **0.33 → 0.39 → 0.55 → 0.81**; the 7B dominates the 3B. "Accuracy-vs-cost frontier — why a cheap self-hosted router is good unit economics." Point to **`docs/EVAL_METHODOLOGY.md`**: we measure the router on *two* axes (selection Jaccard **and** downstream learning outcome), and report the learning result honestly as n=1.

### [2] Application / product — the AI Judge (the differentiator)
> "The product is a staff-engineer **AI judge** with three exercise modes, all graded against a real hire bar."
- **Multi-turn design interview** — generate a question, answer it, then watch the interviewer ask a **follow-up probe that drills into your weakest answer** (show it forcing a KV-cache number after a hand-wavy reply). It grades the **whole conversation** across 10 rubric categories. Call out: **the overall score is the staff-interviewer *weighted* sum — communication is a 0.02 multiplier, not a driver.**
- **Production debugging** — a realistic incident with simulated logs/metrics; graded on the debugging *process* (hypotheses → evidence → root cause → fix).
- **Forward-deployed engineer** — a vague customer complaint ("our agent feels slow"); graded on the **7 sub-skills** that separate forward-deployed work from backend: framing, asking for the right metrics, localization, iteration, the fix, its cost/SLA tradeoff, and explaining it to a non-expert.
- Then the **readiness verdict**: show the card — tier (frontier / ready / borderline / not-ready / remediation), the blend (**60% interview avg + 20% trajectory + 20% consistency**), and the **critical-failure gate** (a weak score in any of the 4 load-bearing skills blocks "ready"). "This is a hire bar, not a vibe."
- Architecture aside: **FastAPI + Postgres/pgvector**, four-tier memory, **auth**. **Deployment:** `deploy/digitalocean/` — "the trained 7B serves via vLLM behind an OpenAI-compatible API; the product calls it remotely."

### [3] Agent / automation
- **TutorAgent:** mastery-aware prompting — show it skipping a mastered topic / targeting a weak one. Trigger an **adaptive quiz** — note it **auto-infers the topic from your question** and **personalizes to what you already know** (targets your logged misconceptions). If wrong, show the **diagnostic remediation** loop.
- Show the **Router** dropdown — heuristic vs. our fine-tuned router selecting context — and the scored "why these items" panel. **This observability is the opposite of a black-box context window.**

---

## Q3 — Use cases & impact (≈1.5 min)
- **Primary:** ML-systems **interview prep** for engineers — the three modes + readiness verdict tell you *exactly where you'd fail* and drill it. **Everything is free right now.**
- **B2B:** bootcamps / company L&D / university courses — the per-learner mastery model aggregates into a **cohort dashboard** ("where is the class stuck?") that a chatbot can't offer.
- **Impact:** turns passive Q&A into a measurable, calibrated hire-bar signal; makes expert ML-systems knowledge learnable and *provable*; the routing research lowers the cost of long-context tutoring at scale.

> **Why choose this over a Claude Project:** "A Claude Project answers and keeps notes. Memex gives you a **staff-engineer judge with follow-up probes, a structured mastery model, observable context routing, and a calibrated hire-bar verdict that provably goes up** — and you can self-host it. We're not selling answers (a commodity); we're selling **proof you'd pass.**"

---

## Q4 — What's next? (≈1 min)
- **Cohort/instructor dashboard** (the B2B wedge).
- **RL from real outcomes** — today the router is oracle-distilled (ceiling = the oracle); next, train on observed mastery gains to *exceed* it (3-phase objective in `docs/EVAL_METHODOLOGY.md`).
- **Multi-LoRA serving** (all four routers off one base) + **proactive spaced-repetition nudges**.
- Curriculum expansion beyond CS336/CS349D; deeper forward-deployed scenario trees.

---

## Closing (15s)
> "Memex: a staff-engineer judge that probes you, a mastery model that remembers you, a router we trained ourselves, and a hire-bar verdict that proves you're improving. Repo and setup are in the README. Thanks."

---

### Recording checklist
- [ ] Seed/log in as a student **with data** (e.g. `Hiva`) so views aren't empty.
- [ ] Have `data/eval/pareto.png` and a cluster `squeue`/`.out` ready to show.
- [ ] In the **multi-turn interview**, give a deliberately hand-wavy first answer so the **follow-up probe** visibly drills in — it's the best 15 seconds of the demo.
- [ ] Show all **three modes** (design / debugging / forward-deployed) and the **readiness verdict card**.
- [ ] Show **🎯 Test** with no topic selected → it infers the topic from your question.
- [ ] Keep it under 10:00. Lead with the differentiator (the judge + verdict); spend the most time on Q2[1] (research) and Q2[2] (the judge).
