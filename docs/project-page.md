# Learning Memory OS

> A context-routed tutor for ML systems engineers.
> CS 153 Frontier Systems — Spring 2026
> Author: Hiva Mohammadzadeh

## One-sentence pitch

A context-orchestrated multi-agent tutoring system with a fine-tuned learned context router, evaluated on long-horizon ML-systems-engineering workflows against retrieval, full-context, and frontier-API baselines under fixed token budgets.

## Mission framing

Long-term: train ML systems engineers at scale. This 17-day CS153 artifact is the first wedge — a working system that successfully tutors one engineer (the author, as student-zero) on the foundations of agent systems, frontier inference, frontier training, and end-to-end ML systems.

## Curriculum

20 topics across 5 areas, drawn from three Stanford courses:

- **CS336** (*Language Modeling from Scratch*) — training from scratch
- **CS349D** (*AI Inference Infrastructure*) — inference at scale
- **CS153** (*Frontier Systems*) — frontier framing

## Architecture

See [architecture.md](architecture.md).

## Headline result

[FILL IN AFTER ROUTERS TRAIN]

Accuracy-vs-cost Pareto frontier across:
- 4 fine-tuned routers (Qwen-2.5 0.5B, 1.5B, 3B, 7B via LoRA)
- 4 heuristic baselines (full-context, retrieval-only, summary-only, heuristic ranker)
- 1 zero-shot frontier-API baseline (Claude Sonnet)

[`Figure 1` will go here]

## Live demo

[Video link will go here]

The demo arc:
1. Author asks a question on KV cache → system selects context from semantic + student memory
2. Tutor reply cites inline IDs grounded in source material
3. Diagnostic agent flags a misconception in the answer
4. Quiz generator surfaces a targeted quiz; mastery score updates
5. The next session's tutor reply reflects updated student state

## Known limitations

- **Synthetic-only router training** = oracle distillation, not learning from environment feedback. Real-trajectory training is Phase B (post-CS153).
- **Student-zero evidence is n=1**, descriptive not statistical.
- **17-day window forces depth/breadth tradeoffs.** Deep evaluation runs on a 6-topic subset; the other 14 are content-loaded but not deeply evaluated.

## Code + data

- Repository: [link]
- Trajectory dataset (5K, oracle = Claude Sonnet): `data/trajectories/val.jsonl`
- Trained adapters: `data/router_checkpoints/<size_id>/adapter/`
- Interaction log: `logs/interactions.jsonl`
- Quality report: `data/quality_report.txt`

## Acknowledgments

CS153 staff (Anjney Midha, Michael Abbott, Adrian A, Ramya I). AMP for compute. CS336 (Hashimoto, Liang) and CS349D for canonical curriculum sources.
