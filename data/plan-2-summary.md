# Plan 2 — Curriculum Loaded

**Tag:** `curriculum-loaded`
**Branch:** `plan-1-mvp`
**Date:** 2026-05-13

## Coverage

- Topics defined in `config/topics.yaml`: 20
- Topics with ingested artifacts: 14
- Total semantic_items: 416
- Sources: CS336 spring 2025 lectures (8 of 17 available), 5 seed topics including CS336 L10 (curated)

## Topics fully loaded (count from quality report)

| Topic | Area | Artifacts |
|---|---|---|
| tokenization | A | 4 |
| resource_accounting | A | 46 |
| gpu_kernels | B | 38 |
| data_parallelism | B | 4 |
| sharded_training | B | 30 |
| model_parallelism | B | 30 |
| kv_cache | C | 52 |
| cs336_l10_inference | C | 52 |
| quantization | C | 19 |
| speculative_decoding | C | 16 |
| continuous_batching | C | 16 |
| pretraining_data | D | 72 |
| rl_systems | D | 29 |
| agent_memory | E | 8 |

## Open gaps (deferred — need user-supplied content)

These topics are defined in the YAML but have no source files yet. Listed with the missing files:

- `transformer_architecture` (Area A) — needs `data/curriculum/cs336_l03_architecture.md`
- `attention_moe` (Area A) — needs `data/curriculum/cs336_l04_attention_moe.md`
- `scaling_laws` (Area B) — needs `data/curriculum/cs336_l09_scaling.md`, `cs336_l11_scaling.md`
- `prefill_decode_disagg` (Area C) — needs `data/curriculum/cs349d_disaggregation.md`
- `sft_rlhf_dpo` (Area D) — needs `data/curriculum/cs336_l15_alignment.md`
- `context_selection` (Area E) — needs `data/curriculum/agents_context_selection.md`
- `multi_agent_orchestration` (Area E) — needs `data/curriculum/agents_orchestration.md`

These CS336 lecture files (L3, L4, L9, L11, L15) are not yet in the spring2025-lectures GitHub repo as of 2026-05-13. The CS349D and agent-recursion files need to be authored by the user (the project recursion topics are the user's own work).

Once content is provided, run `uv run python -m scripts.ingest_all` again — the runner is idempotent and will pick up new sources.

## Plan 3 readiness

Plan 3 (synthetic trajectories + LoRA router fine-tuning) can begin with current coverage. 14 topics across all 5 areas is sufficient for the synthetic-trajectory generator to produce a diverse training set. The 7 missing topics can be added later without invalidating earlier training runs.

## How to verify locally

```bash
uv run python -m scripts.quality_report
uv run python -m scripts.tutor_repl --student-id <you> --question "..." --topic-id <topic>
```
