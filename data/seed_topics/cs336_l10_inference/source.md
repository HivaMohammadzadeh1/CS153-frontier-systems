# CS336 Lecture 10: Inference

Inference: given a fixed model, generate responses given prompts.

## Understanding the inference workload

### Landscape

Inference shows up in many places:
- Actual use (chatbots, code completion, batch data processing)
- Model evaluation (e.g., on instruction following)
- Test-time compute (thinking requires more inference)
- Training via reinforcement learning (sample generation, then score)

Why efficiency matters: training is one-time cost, inference is repeated many times.

Metrics:
- Time-to-first-token (TTFT): how long user waits before any generation happens (matters for interactive applications)
- Latency (seconds/token): how fast tokens appear for a user (matters for interactive applications)
- Throughput (tokens/second): useful for batch processing applications

Key considerations in efficiency:
- Training (supervised): you see all tokens, can parallelize over sequence (matmul in Transformer)
- Inference: you have to generate sequentially, can't parallelize, so harder to fully utilize compute

Companies doing inference:
- Providers serving closed models (OpenAI, Anthropic, Google)
- Providers serving open-weight models (Together, Fireworks, DeepInfra)

Open-source packages: vLLM (Berkeley), Tensor-RT (NVIDIA), TGI (Hugging Face).

### Arithmetic intensity review

Setup: multiply X (B x D) and W (D x F) matrix. Steps: read X (2 B D bytes), read W (2 D F bytes), compute Y = X @ W (2 B D F FLOPs), write Y (2 B F bytes). Arithmetic intensity = FLOPs / bytes_transferred.

When B is much less than D and F, intensity simplifies to B.

Accelerator intensity of H100: FLOPs/sec = 989e12, memory bandwidth = 3.35e12, so accelerator intensity is roughly 295.

If computation intensity > accelerator intensity, compute-limited (good). If less, memory-limited (bad). Conclusion: compute-limited iff B > 295.

Extreme case B = 1 (matrix-vector product): intensity = 1, memory-limited. This is what happens during generation.

### Arithmetic intensity of inference

Naive inference: to generate each token, feed history into Transformer. Complexity: generating T tokens requires O(T^3) FLOPs.

KV cache: store key and value tensors of past tokens in HBM so they don't need recomputing. For every (sequence B, token S, layer L, head K), store an H-dimensional vector.

Two stages of inference:
1. Prefill: given a prompt, encode into vectors (parallelizable like training)
2. Generation: generate new response tokens (sequential, one at a time)

MLP layers FLOPs and bytes: with B*T small compared to D, F, arithmetic intensity simplifies to B*T. Prefill (T = S) is easy to make compute-limited by enlarging batch. Generation (T = 1) is hard because batch B (concurrent requests) must be large.

Attention layers (with FlashAttention): intensity is S*T / (S + T). Prefill (T = S) gives S/2 (good). Generation (T = 1) gives <1 (bad). Unlike MLPs, no dependence on B — batching does NOT help attention because every sequence has its own KV cache.

Summary: prefill is compute-limited, generation is memory-limited. MLP intensity scales with B (helped by concurrent requests). Attention intensity is fixed at 1 during generation.

### Throughput and latency

Inference is memory-limited. Theoretical max latency = memory / memory_bandwidth (read all params + KV cache each step). Throughput = B / latency (generating B tokens in parallel).

Tradeoff:
- Smaller batch sizes: better latency, worse throughput
- Larger batch sizes: better throughput, worse latency

Easy parallelism: M replicas → same latency, M× throughput. Harder parallelism: shard the model and KV cache.

TTFT is essentially a function of prefill. Use smaller batches during prefill for faster TTFT; larger batches during generation for throughput.

## Taking shortcuts (lossy)

### Reduce KV cache size

Grouped-query attention (GQA): N query heads, but only K key/value heads, each interacting with N/K query heads. MHA: K=N. MQA: K=1. GQA: K in between. Reduces KV cache by N/K. Same or better accuracy.

Multi-head latent attention (MLA): project key and value down from N*H to C dimensions. DeepSeek v2: 16384 → 512. MLA is not compatible with RoPE, so add 64 dimensions for RoPE → 576 total. Better accuracy than MHA at much lower cost.

Cross-layer attention (CLA): share KVs across layers (analogous to GQA across heads). Improves accuracy/KV-size Pareto frontier.

Local attention (Longformer, sparse Transformer, Mistral 7B): only attend to local context, most relevant for modeling. Effective context scales linearly with layers. KV cache independent of sequence length. Can hurt accuracy; mitigate by interleaving with full-attention layers. character.ai uses 1 global layer per 6 layers in addition to CLA.

### Alternatives to the Transformer

State-space models. S4: classic state space models, good at synthetic long-context tasks. Weakness: bad at associative recall. Mamba: input-dependent SSM parameters, matches Transformer at 1B scale. Jamba: interleaves Transformer-Mamba (1:7) with a 52B MoE. BASED: linear + local attention. MiniMax-01: linear + full attention (456B MoE). Linear + local attention can yield SOTA. Replaces O(T) KV cache with O(1) state.

Diffusion models: generate tokens in parallel (not autoregressively), iterative refinement. Inception Labs results show much faster coding benchmark throughput.

### Quantization

Reduce precision of numbers. Less memory → higher throughput because inference is memory-limited.

Precisions: fp32 (4 bytes, training), bf16 (2 bytes, default inference), fp8 (1 byte, e4m3 range [-240, 240] on H100), int8 (1 byte, [-128, 127], inference only), int4 (0.5 bytes, [-8, 7], cheaper, even less accurate).

Quantization-aware training (QAT): train with quantization, doesn't scale. Post-training quantization (PTQ): use sample data to set scale and zero point per layer/tensor.

LLM.int8(): standard quantization fails on large networks due to outliers. Solution: extract outliers, process in fp16, rest in int8. 15–23% slower than fp16 but works.

Activation-aware quantization (AWQ): keep 0.1–1% of weights in high precision, chosen by activations. fp16 → int3 yields 4× memory reduction, 3.2× speedup.

### Model pruning

Algorithm:
1. Identify important {layer, head, hidden dimension} on small calibration set (1024 samples)
2. Remove unimportant layers → smaller model
3. Distill original model into pruned model

NVIDIA paper; produces strong accuracy/cost tradeoff.

## Use shortcuts but double-check (lossless)

### Speculative sampling

Recall: prefill is compute-limited (we get probabilities for free), generation is memory-limited (sequential, one token at a time). Checking is faster than generation.

Use a cheaper draft model p to guess a few tokens (e.g., 4). Evaluate with target model q in parallel and accept if it looks good. Modification of rejection sampling: always generate at least one candidate so the loop terminates.

Key property: guaranteed to be an exact sample from the target distribution.

Two-vocabulary proof sketch: target q, draft p. Assume p(A) > q(A) so p(B) < q(B). Residual probabilities max(q − p, 0) = [0, 1]. P[sampling A] = p(A)*(q(A)/p(A)) + p(B)*1*0 = q(A). Symmetric for B. Exact.

In practice: 70B target with 8B draft, or 8B target with 1B draft. Make the draft model as close to the target as possible (distillation).

Extensions: Medusa (draft generates multiple tokens in parallel), EAGLE (draft uses high-level features from the target model).

## Handling dynamic workloads

Batching over sequences in live traffic is tricky:
1. Requests arrive at different times (waiting for batch starves early requests)
2. Sequences have shared prefixes (system prompts; sampling multiple completions)
3. Sequences have different lengths (padding is wasteful)

### Continuous batching (Orca)

Iteration-level scheduling: decode token-by-token, add new requests to the batch as they arrive. Don't wait for current generations to complete.

Selective batching: handle ragged shapes. For non-attention computation, concatenate all sequences into [sum_lengths, H]. For attention, process each sequence separately.

### PagedAttention (vLLM)

Previous status quo: allocate a section of KV cache for each request up to max length. Wastes memory: internal fragmentation (generate fewer tokens than allocated) and external fragmentation (unused gaps between sections).

Solution: divide the KV cache of a sequence into non-contiguous blocks (like OS virtual memory pages). Block tables map logical addresses to physical blocks.

Sharing types: shared system prompt across requests; multiple samples per prompt for program synthesis. Implement with shared prefixes + copy-on-write at block level.

Other vLLM optimizations: fused block-read + attention kernels (reduce launch overhead), latest kernels (FlashAttention, FlashDecoding), CUDA graphs to skip launch overhead.

Key insight: use ideas from operating systems (paging) to manage memory for dynamic inference workloads.

## Summary

- Inference is important (actual use, evaluation, RL sample generation)
- Different characteristics from training: memory-limited, dynamic workloads
- Techniques: new architectures (GQA/MLA/CLA, SSMs, diffusion), quantization, pruning/distillation, speculative decoding
- Borrow ideas from operating systems (speculative execution, paging)
- New architectures have huge potential for further inference improvement
