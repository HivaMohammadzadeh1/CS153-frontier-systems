# CS153 Readings — Curriculum Enrichment Pointers

Source: CS153 "Readings and Materials" handout (filed 2026-05-15).

These are speaker-recommended readings from the CS153 lecture series. They map to existing curriculum topics and can be used to enrich content in a future ingestion pass. **Not yet ingested.**

## Speaker advice (Area E framing material)

- "If you're early, you're on time; if you're on time, you're late; if you're late, you're dead" — Anj Midha
- "Make friends in college" — Nikhyl Singhal
- "Everything is figureoutable" — Amit Jain
- "You can outsource your thinking, but you can't outsource your understanding" — Andrej Karpathy

## Canonical papers → curriculum topic map

### Area A — Model fundamentals
- **AlexNet** (2012): historical context for `transformer_architecture` / pre-Transformer era
- **Word2Vec** (2013): historical context for `tokenization` / embedding-era
- **Transformers — Attention is All You Need** (2017): canonical source for `transformer_architecture`
- **BERT** (2018): for `transformer_architecture` (encoder variant), `pretraining_data`
- **GPT-1/2/3** (2018–2020): for `transformer_architecture` (decoder-only), `pretraining_data`, `scaling_laws`

### Area B — Training systems
- **Scaling Laws for Neural Language Models** (Kaplan et al, 2020): canonical for `scaling_laws`
- **Chinchilla — Training Compute-Optimal Large Language Models** (Hoffmann et al, 2022): canonical for `scaling_laws`

### Area D — Data & alignment
- **InstructGPT** (Ouyang et al, 2022): canonical for `sft_rlhf_dpo`
- **DDPM — Denoising Diffusion Probabilistic Models** (Ho et al, 2020): could seed a new "diffusion alternatives" topic in Area C

### Other / frontier framing (Area E)
- **The Bitter Lesson** (Rich Sutton): canonical essay; Area E framing material
- **DQN — Playing Atari with Deep Reinforcement Learning** (DeepMind, 2013): historical RL context for `rl_systems`

## Lecturer-specific external content

- **Lecture 1 (Anj Midha)**: 20VC podcast on $300M Anthropic investment — bottlenecks/capital framing
- **Lecture 2 (Anj Midha)**: CLOUD Act overview — capital/regulatory framing
- **Lecture 3 (Mati Staniszewski / ElevenLabs)**: speech/audio AI ops
- **Lecture 4 (Andreas Blattmann)**: Latent Diffusion + Stable Video Diffusion papers
- **Lecture 5 (Amit Jain / Luma AI)**: world models, neural rendering, Luma Agents
- **Lecture 6 (Nikhyl Singhal)**: "The Skip" (career advice essays), nikhyl.ai

## How to use this pointer

For a future enrichment pass:
1. Fetch each paper's abstract + intro + conclusion from arxiv (most have arxiv links)
2. Save under `data/curriculum/papers/<paper_short_id>.md` with clean markdown
3. Update `config/topics.yaml` to add these as extra sources for the corresponding topic
4. Re-run `uv run python -m scripts.ingest_all --only <topic>` for each enriched topic
5. Quality report should show artifact counts increasing per topic

This is deferred work. Prioritize Plan 3 (routers) and the demo/writeup first.
