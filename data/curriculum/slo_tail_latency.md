# SLOs and Tail Latency in LLM Serving

## Why tail latency matters
Production LLM serving is judged on SLOs (service-level objectives), not averages. A p50 that looks great hides the p95/p99 requests that dominate user-perceived quality and trigger client timeouts. Always reason about the tail.

## TTFT vs TPOT
Two distinct latency metrics govern an LLM request:
- **TTFT (time to first token)** is dominated by the *prefill* phase — processing the prompt. It scales with prompt length and is compute-bound. Long prompts, large system prompts, and retrieved context blow up TTFT.
- **TPOT (time per output token)**, a.k.a. inter-token latency, is dominated by the *decode* phase — generating one token at a time. It is memory-bandwidth-bound and roughly constant per token, so total decode time scales with output length.

A request's total latency ≈ TTFT + (output_tokens × TPOT). Diagnosing "it's slow" starts with: is the tail in TTFT or in TPOT? They have completely different fixes.

## Sources of tail latency
- **Queueing delay**: requests wait for a batch slot. Shows up as TTFT inflation under load even when GPU execution is fast.
- **Long-context prefill**: as conversations grow, prefill cost grows; p99 TTFT regresses for the longest sessions.
- **Batch interference**: a few very long requests in a continuous batch slow everyone (head-of-line blocking).
- **KV-cache pressure / preemption**: when memory is tight the scheduler preempts and recomputes, spiking tail latency.

## How to debug
1. Separate TTFT from TPOT in your metrics.
2. Break TTFT into queue time vs prefill compute time.
3. Look at the request-size distribution — tails are usually the longest prompts/conversations.
4. Check whether p99 tracks load (queueing) or context length (prefill).

## Common misconception
"Average latency is fine, so we're fine." The SLO that matters is p95/p99; optimizing the mean while ignoring the tail is the classic mistake.
