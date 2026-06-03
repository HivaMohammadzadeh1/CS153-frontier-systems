# Memex — demo video script (target ~7–9 min, max 10)

A beat-by-beat script answering the four required questions, with what to show on
screen. Speak to the bolded differentiation — it's what wins the "why not just a
Claude Project?" question.

---

## 0. Cold open (20s)
> "This is Memex. The pitch in one line: **it's not a chatbot that remembers you — it's a learning OS that models what you know, routes only the context you need with a model we fine-tuned ourselves, and proves your interview-readiness improving over time.**"

Show: the app open on **🧭 Interview Readiness** — the 22% gauge, area bars, gaps.

---

## Q1 — Why did you build this? (≈1 min)
- **The bottleneck:** ML-systems engineering (CS336/CS349D material — kernels, FSDP, KV cache, MoE) is brutal to self-study, and interview prep for it is underserved. People fall back to ChatGPT, which **answers questions but doesn't model the learner** — no idea what you've mastered, what you keep getting wrong, or whether you're actually improving.
- **Inspiration:** a tutor should behave like a great TA: track your mastery, target your weak spots, confront misconceptions, and *prove* you got better. And it should do that **efficiently** — not by dumping a giant context window, but by *selecting* the right context.

Show: briefly, the Chat answering an ML-systems question with the "Context for this answer" panel.

---

## Q2 — How does it work? (≈3.5–4 min) — the meat

### [1] Research — the learned context router
> "The research core: can a **small** model match a frontier oracle at **context selection** — choosing which memory items to put in the tutor's context under a token budget?"
- Show `config/router_sizes.yaml` + `data/trajectories/val.jsonl` (5,000 oracle trajectories).
- "We LoRA-fine-tuned **four Qwen2.5 sizes (0.5B→7B)** on a shared **32×H100 SLURM cluster**." Show `cluster/finetune_router.sbatch` + a `squeue`/`*.out` snippet.
- Show **`data/eval/pareto.png`**: selection Jaccard vs. the oracle — **0.33 → 0.39 → 0.55 → 0.81**; the 7B dominates the 3B. "This is the accuracy-vs-cost frontier — and it's why a cheap self-hosted router is good unit economics."

### [2] Application / product
- Architecture diagram (or narrate): **FastAPI + Postgres/pgvector**, four-tier memory, **auth**, **Stripe billing**.
- Show the SPA views: **Home** (mastery ring, XP), **Readiness**, **Progress**, **Path**.
- **Deployment:** show `deploy/digitalocean/` — "the trained 7B serves via vLLM on a GPU droplet behind an OpenAI-compatible API, and the product can call it remotely."

### [3] Agent / automation
- **TutorAgent:** mastery-aware prompting — show it skipping a mastered topic / targeting a weak one. Trigger an **adaptive quiz** (difficulty scales with mastery). If wrong, show the **diagnostic remediation** loop.
- Show the **Router** dropdown — heuristic vs. our fine-tuned router selecting context — and the scored "why these items" panel. **This observability is the opposite of a black-box context window.**

---

## Q3 — Use cases & impact (≈1.5 min)
- **Primary:** ML-systems **interview / mastery prep** for engineers — the Readiness report tells you *exactly where you'd fail* and drills it.
- **B2B:** bootcamps / company L&D / university courses — the per-learner mastery model aggregates into a **cohort dashboard** ("where is the class stuck?") that a chatbot can't offer.
- **Impact:** turns passive Q&A into measurable skill gain; makes expert ML-systems knowledge learnable and *provable*; the routing research lowers the cost of long-context tutoring at scale.

Show: the **paywall** — "free shows your readiness + one gap; **Pro** unlocks the full gap analysis, drill plan, and your **mastery-over-time trend**." Click **Get Pro** → checkout.

> **Why choose this over a Claude Project:** "A Claude Project answers and keeps notes. Memex gives you a **structured mastery model, observable context routing, and a readiness number that provably goes up** — and you can self-host it. We're not selling answers (a commodity); we're selling **proof of progress**."

---

## Q4 — What's next? (≈1 min)
- **Onboarding diagnostic** (60-sec placement → a real starting readiness, better conversion).
- **Cohort/instructor dashboard** (the B2B wedge).
- **RL from real outcomes** — today the router is oracle-distilled (ceiling = the oracle); next, train on observed mastery gains to *exceed* it.
- **Multi-LoRA serving** (all four routers off one base) + **proactive spaced-repetition nudges**.
- Curriculum expansion beyond CS336/CS349D.

---

## Closing (15s)
> "Memex: model what you know, route only what you need, and prove you're improving. Repo and setup are in the README. Thanks."

---

### Recording checklist
- [ ] Seed/log in as a student **with data** (e.g. `Hiva`) so views aren't empty.
- [ ] Have `data/eval/pareto.png` and a cluster `squeue`/`.out` ready to show.
- [ ] Show the Readiness paywall **and** (flip `is_pro`) the unlocked Pro report + trend.
- [ ] Keep it under 10:00. Lead with the differentiator; spend the most time on Q2[1] (research).
