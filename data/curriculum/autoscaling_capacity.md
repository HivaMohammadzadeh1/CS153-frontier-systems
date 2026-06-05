# Autoscaling and Capacity Planning for LLM Serving

## Why it's hard for LLMs
GPU instances are expensive and slow to start (cold starts: pull the image, load tens of GB of weights, warm the cache — often minutes). Traffic is bursty. So you can't autoscale LLM serving like a stateless web app; you must plan capacity and scale ahead of demand.

## Capacity planning basics
Estimate the GPUs you need from first principles:
- Per-GPU **throughput** (tokens/s) at your target batch size, given the model and KV-cache limits.
- Offered load: requests/s × average output tokens.
- Headroom for the **tail** and for bursts, not just the mean.
KV-cache memory (weights + per-token KV × max concurrency) usually bounds how many concurrent sequences fit before you add hardware.

## Autoscaling signals
Scale on queue depth / time-to-first-token / KV-cache utilization rather than CPU%. These reflect real saturation. Scale up *before* the SLO breaks, and scale down conservatively (cold starts make flapping costly).

## Levers beyond more GPUs
- Right-size the model (distillation, quantization) to fit more concurrency per GPU.
- Use spot/preemptible capacity with fallback for batch work.
- Multi-tenant packing and prefix caching to raise effective throughput.

## Interview-relevant reasoning
"Just autoscale" is naive for LLMs because cold starts and weight-loading time mean reactive scaling arrives too late. Strong answers pair capacity planning (so steady-state is provisioned) with predictive/early autoscaling and admission control for the overflow.
