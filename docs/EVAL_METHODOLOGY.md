# Evaluation Methodology

How we measure whether Memex's two learned systems — the **context router** and the
**readiness verdict** — actually work, and how we keep ourselves honest about a project
with a single primary user (n=1). Written to satisfy the rubric's "evaluate your claims"
bar and to be defensible to a skeptical reader.

---

## 1. The router has *two* metrics, not one

A context router can look good on the wrong axis. We measure both:

| Metric | Question it answers | How |
|---|---|---|
| **Selection quality (Jaccard)** | Does the small router pick the *same* context items the frontier oracle would? | Jaccard overlap between the router's selected item-set and the oracle's, per turn, averaged over a held-out trajectory set. This is the headline number we report (0.33 → 0.81 across model sizes). |
| **Outcome quality** | Given the router's selection, does the *student learn more*? | Downstream reward: post-turn mastery delta + spaced-repetition retention at the next review − misconception recurrence. A router that matches the oracle but produces worse learning is not actually better. |

**Why both.** Jaccard is cheap and stable but only proxies the oracle's taste; it can't
see whether a *different* selection would have taught better. Outcome quality is what we
actually care about, but it's noisy and slow (needs real study sessions). We optimize on
Jaccard during distillation and validate on outcome quality before shipping a new router.
Reporting only Jaccard would overclaim — we flag any case where the two disagree.

---

## 2. The router objective evolves in three phases

We do not claim the router is trained end-to-end on learning outcomes. It is built in
phases of increasing ambition, and we are explicit about which phase produced each number:

1. **Phase 1 — Oracle distillation (BUILT).** Supervised imitation of a frontier model's
   context selections over ~5,000 synthetic trajectories. Metric: selection Jaccard vs.
   oracle. This is what the shipped LoRA routers do today. *Claim ceiling: "matches the
   oracle's selections," not "maximizes learning."*
2. **Phase 2 — Reward-weighted fine-tuning (DESIGNED).** Re-weight real captured
   trajectories by outcome reward (`reward_weight` upsamples turns that produced mastery
   gains, drops non-positive ones), then re-train. Metric: selection Jaccard *and* outcome
   quality on held-out sessions.
3. **Phase 3 — Policy optimization from outcomes (FUTURE).** Treat selection as an action
   and the mastery delta / retention as reward; optimize with policy-gradient or DPO on
   trace pairs. Metric: outcome quality is primary. Not yet run — listed so the claim
   boundary is unambiguous.

The continuous-improvement spec (`docs/superpowers/specs/2026-06-03-...`) is the full
design; this doc fixes which metric backs which claim.

---

## 3. The readiness verdict is calibrated, not vibes

The verdict (`/api/student/{id}/readiness` → `verdict`) is a deliberate hire-bar mapping,
not a raw mastery average:

- **Blend:** `0.6 × interview_avg + 0.2 × trajectory + 0.2 × consistency`, over the last
  up-to-5 interviews. Average dominates; trajectory rewards improvement; consistency
  penalizes a single lucky answer.
- **Critical-failure gate:** a score below 60 in any of the four load-bearing categories
  (`bottleneck_identification`, `memory_compute_reasoning`,
  `latency_throughput_reasoning`, `production_debugging`) blocks a "ready" verdict
  regardless of the blended number — mirroring a real interviewer's no-hire veto.
- **Tiers:** ≥90 frontier · ≥80 ready · ≥70 borderline · ≥60 not-ready · <60 remediation.
- **Confirmation rule:** "interview-ready" requires avg ≥ 80 over **≥ 3** interviews with
  no critical failure. One good interview is explicitly *not* enough.

The overall interview score itself is the **staff-interviewer weighted sum** of the rubric
categories (`weighted_overall` in `interview_prompts.py`), not the LLM's holistic guess —
this fixes the failure mode where a model over-weights polished communication. Communication
is a 0.02 weight: a multiplier, not a driver.

---

## 4. n=1 validity: what we can and cannot claim

Memex has one primary user (the author). We do **not** claim population-level efficacy.
What a single-subject design *can* legitimately show:

- **Within-subject pre/post.** Baseline interview battery before using the product, then the
  same battery after a study period. The student's own earlier self is the control. Reported
  as a delta with the raw scores, never as "X% of users improve."
- **Instrument validity, separately.** The judge's discrimination is testable independent of
  the user: it scores hand-wavy answers low (verified ~8/100 on deliberately vague answers)
  and catches the specced canonical misconceptions. That's an evaluation of the *grader*, and
  it generalizes better than the n=1 learning result.
- **Honest framing.** Per the project's non-negotiables: synthetic router training is
  *oracle distillation*; the learning result is *n=1*. Both are stated as such in the writeup.

Threats we name rather than hide: practice/familiarity effects (same battery seen twice),
grader–tutor shared-model bias (both Claude-backed, so the judge may favor the tutor's
framing), and small-sample noise in the verdict trajectory.

---

## 5. Reproducing the numbers

- Router Jaccard: `scripts/build_training_set.py` → train (see
  `docs/superpowers/plans/2026-05-13-plan-3-router-finetuning.md`) → eval harness reports
  per-size Jaccard vs. oracle on the held-out split.
- Judge discrimination: run a mock interview with a deliberately vague answer and confirm
  the overall score is low and the missed tradeoffs are named.
- Verdict calibration: `_readiness_verdict` is pure given the interview rows; unit-testable
  by feeding synthetic `category_scores` and asserting the tier + critical gate.
