# CS349D — Prefill-Decode Disaggregation & Hierarchical KV-Cache Management

**Area:** Inference Infrastructure (Area C)
**Prerequisites:** KV cache & PagedAttention, Continuous batching & request scheduling

---

## The Fundamental Problem: Prefill and Decode Are Different Workloads

LLM inference is divided into two distinct computational phases. **Prefill** processes the entire input prompt in one forward pass, computing key-value (KV) tensors for every token simultaneously. **Decode** then generates new tokens autoregressively — one at a time — each requiring a forward pass over the model with the full accumulated KV cache.

These two phases have radically different compute profiles, and conflating them on the same hardware is one of the central inefficiencies in naive inference deployments.

### Prefill Is Compute-Bound

During prefill, the GPU executes large batched matrix multiplications over the prompt tokens. If a prompt is S tokens long, the attention computation involves an O(S²) work term, and the weight matrices are applied to a [S × d_model] activation tensor. The arithmetic intensity — the ratio of floating-point operations to bytes moved from memory — is high. A well-implemented prefill kernel on a modern A100 or H100 can push close to peak FLOP utilization. The hardware is doing useful computation nearly every cycle.

### Decode Is Memory-Bandwidth-Bound

During decode, each step generates exactly one token. The weight matrices are the same size, but they are now applied to a [1 × d_model] activation vector. Every weight element is loaded from HBM (high-bandwidth memory) but contributes only a few multiply-add operations. The arithmetic intensity is approximately 1 FLOP per byte — far below the roofline threshold needed to saturate GPU compute. The GPU is mostly stalling, waiting for data to arrive from memory. This is an intrinsic property of single-token generation, not a software deficiency.

Arithmetic intensity for a single decode step over a weight matrix W ∈ ℝ^{d_out × d_in}:

```
AI = (2 × d_in × d_out) / (2 × d_in × d_out × dtype_bytes)
   = 1 FLOP / byte    (for fp16/bf16, dtype_bytes = 2)
```

A100 SXM4 peak memory bandwidth is ~2 TB/s; peak FLOP is ~312 TFLOP/s (bf16). The roofline memory intensity threshold is 312e12 / 2e12 ≈ 156 FLOP/byte. Decode operates at ~1% of that threshold.

### Why Mixing Both Phases on the Same GPU Hurts

When prefill and decode requests are co-located on the same GPU, they interfere in two ways:

1. **Decode latency spikes during prefill:** A prefill step is compute-saturating and can take hundreds of milliseconds for a long prompt. Any in-flight decode requests must wait. This causes high tail latency and unpredictable time-to-first-token (TTFT) for queued requests.
2. **Prefill throughput is wasted by batch fragmentation:** The scheduler must interrupt high-throughput prefill jobs to service decode steps, reducing GPU utilization for both phases.

The result is a lose-lose situation: decode latency is worse than necessary, and prefill throughput is lower than hardware allows.

---

## Disaggregation: Dedicated Prefill and Decode Nodes

The natural solution is **disaggregation**: run prefill on a dedicated pool of nodes optimized for compute throughput, and run decode on a separate pool optimized for low latency. A router dispatches each request to a prefill node first, then routes the resulting KV cache to a decode node where token generation proceeds.

### Prefill Nodes

Prefill nodes can be tuned for compute throughput:
- Large batch sizes (many prompt tokens processed simultaneously)
- Possibly FP8 quantization for weights and activations to maximize TFLOP utilization
- Fewer total nodes than decode because prefill is fast and each node can handle many short prefill jobs per second
- The key metric is prefill throughput: tokens-prefilled per second per GPU

### Decode Nodes

Decode nodes can be tuned for low latency:
- Smaller batch sizes to reduce decode step latency (each step touches the full weight matrix once per request)
- PagedAttention for efficient KV cache memory management across concurrent sequences
- More total nodes because each decode job ties up a GPU slot for the entire generation duration
- The key metric is tokens-per-second-per-request (TPS) and tail latency (P99 TPOT, time-per-output-token)

### The New System Challenge: KV Cache Transfer

When a request completes prefill on a prefill node, its KV cache — the K and V tensors for all prompt tokens across all layers — must be sent to a decode node before generation can begin. This transfer is the central systems engineering challenge introduced by disaggregation.

---

## KV Cache Transfer: Mechanics and Latency

### How Much Data Moves?

For a model with L transformer layers, H KV heads, D head dimension, and S prompt tokens, the KV cache size is:

```
KV_size = 2 × L × H × D × S × dtype_bytes
```

For a 70B-parameter model in the Llama family: L=80 layers, H=8 KV heads (GQA), D=128 head dim, bf16 (2 bytes):

```
KV_size = 2 × 80 × 8 × 128 × S × 2 bytes
        = 327,680 × S bytes
        ≈ 320 KB × S
```

At S=4096 prompt tokens:
```
KV_size ≈ 320 KB × 4096 ≈ 1.31 GB
```

Wait — let's redo this with a more compact Llama-70B (which uses GQA with 8 KV heads, not 32). Actual Llama-3-70B: L=80, num_kv_heads=8, head_dim=128:

```
KV_size(4096 tokens) = 2 × 80 × 8 × 128 × 4096 × 2 bytes
                     = 2 × 80 × 8 × 128 × 4096 × 2
                     = 2 × 80 × 4,194,304
                     ≈ 671 MB
```

Over a 100 Gb/s NIC (12.5 GB/s effective throughput):
```
transfer_time = 671 MB / 12,500 MB/s ≈ 54 ms
```

This is the **latency tax of disaggregation** for a single sequence at 4096 tokens. Typical TTFT targets for production systems are 200–500 ms for a first-token SLA, so 54 ms is material but not prohibitive — provided the transfer happens concurrently with other work.

For shorter prompts (S=512 tokens), the transfer is ~6.7 MB and takes ~0.5 ms, essentially free. For very long contexts (S=32K), the transfer is ~5.2 GB and takes ~418 ms at 100 Gb/s — now it dominates TTFT. This gives a design rule: disaggregation is cheaper in the common case, but long-context requests impose real latency on transfer unless the network bandwidth is scaled accordingly.

### Transport Options

**RDMA (Remote Direct Memory Access) over InfiniBand or RoCE:** The preferred approach for multi-host disaggregation. RDMA bypasses the host CPU for memory copies, achieving close to wire-rate throughput with low CPU overhead. InfiniBand HDR delivers 200 Gb/s; NDR 400 Gb/s. Mooncake's production system uses RDMA for cross-host KV transfer.

**NVLink / NVSwitch for intra-host disaggregation:** On a single server with multiple GPUs, NVLink provides GPU-to-GPU bandwidth of up to 900 GB/s (H100 NVSwitch). When a prefill GPU and decode GPU are on the same host, this path is effectively free from a latency standpoint.

**Host memory mediation:** The KV cache is first transferred from GPU to host DRAM via PCIe (≈32 GB/s bidirectional), then over the network, then PCIe on the far end. This is lower cost to implement but adds two PCIe hops and is less efficient. Still usable for less latency-sensitive workloads.

### Production Systems That Implement Disaggregation

**Splitwise** (Patel et al., Microsoft Azure, 2023): The first widely-cited academic treatment of prefill-decode disaggregation. Shows that separating prefill and decode pools improves TTFT by 1.4×–2× while maintaining throughput, under realistic Azure traffic patterns. Introduces the term "split" to describe the compute characteristic difference.

**Mooncake** (Moonshot AI, 2024): Moonshot's production KVCache-centric serving system built on top of vLLM. Mooncake externalizes the KV cache into a distributed cache layer, enabling KV reuse across requests and nodes. It is open-sourced and implements RDMA-based KV transfer between prefill and decode pools. Mooncake treats the KV cache as a first-class distributed data structure, not a side effect of GPU memory.

**DeepSeek-V3 serving infrastructure:** DeepSeek's technical report on V3/R1 describes a disaggregated serving system where prefill and decode are handled by different nodes at their scale. The system uses pipeline parallelism within each pool and a shared KV cache for common prefixes (system prompts, tool definitions).

**vLLM v0.6+:** vLLM added experimental support for disaggregated prefill via its `disagg_prefill` connector. The connector serializes the KV cache from a prefill worker and sends it to a decode worker over the network.

---

## Hierarchical KV-Cache Management

Disaggregation reveals a deeper problem: KV cache memory is expensive and recomputing it is costly. If two requests share a common prefix (e.g., the same system prompt), recomputing KV tensors for that prefix twice wastes both compute and memory. The solution is a **hierarchical cache** that stores KV tensors at multiple memory tiers.

### The Memory Hierarchy

```
HBM (GPU on-chip)       ~80 GB     ~3.35 TB/s      fastest, most expensive
  ↓
Host DRAM               ~0.5–2 TB  ~100–200 GB/s   slower, 10–25x cheaper
  ↓
NVMe SSD                ~4–50 TB   ~7–14 GB/s      slow, persistent, cheap
  ↓
Remote storage / S3     unlimited  network-bound    slowest, cheapest
```

KV tensors for frequently-used prefixes live in HBM. Less-used prefixes are evicted to DRAM. Rarely-used ones go to NVMe. Very long-term reuse (across service restarts) can be backed by S3 or a distributed KV store.

### Prefix Sharing

When a new request arrives, the serving system computes a hash over its token prefix and checks whether matching KV tensors exist in the cache. If found, the prefill for those tokens is skipped entirely — the system fast-forwards to the new suffix. This is called **prefix caching** or **prompt caching** in production APIs (Anthropic and OpenAI both expose this).

The efficiency gain depends on the prompt structure:
- Single shared system prompt (common in many SaaS deployments): all users share the same ~1K-token prefix, eliminating prefill for that prefix universally
- Document QA with the same document asked multiple times: the document KV cache is reused across questions
- Multi-turn conversation with long history: each turn shares the prefix of all prior turns

### Eviction Policy

Since HBM is limited, eviction policy determines what stays in the fast tier. Options:

**LRU (Least Recently Used):** Evict the prefix that was accessed longest ago. Simple and cache-friendly for cyclic access patterns. Implemented in vLLM's RadixAttention.

**LFU (Least Frequently Used):** Evict the prefix accessed fewest times. Better for identifying globally popular prefixes (e.g., a shared system prompt that is reused millions of times).

**Size-weighted LRU:** Prefer evicting large low-frequency prefixes over small frequently-accessed ones, maximizing HBM utilization for the overall hit rate.

In practice, most production systems use LRU or a variant because it is cheap to implement with a doubly-linked list and a hash map.

### Prefix Tree (Trie) Indexing

vLLM's RadixAttention represents the cache as a **radix tree** (prefix trie) keyed by token sequences. Each node in the trie corresponds to a cached block of tokens. Cache lookup is O(S) in the number of prompt tokens. Insertion and eviction operate on subtrees, enabling clean block-level management. This structure naturally handles partial prefix matches — if the first 1024 tokens of a 2048-token prompt are cached, the system only prefills the remaining 1024 tokens.

---

## Tradeoffs: When Does Disaggregation Help?

### Benefits

1. **Independent scaling:** If traffic shifts toward long-prompt workloads, add prefill nodes. If decoding demand grows (longer output sequences), add decode nodes. Without disaggregation, you must scale both together.
2. **Elimination of prefill-decode interference:** Decode P99 latency improves substantially because no prefill steps interrupt the decode worker.
3. **Higher prefill throughput per GPU:** Prefill nodes can use larger, contiguous batches and more aggressive quantization.
4. **Simpler autoscaling:** TTFT (time-to-first-token) and TPOT (time-per-output-token) are separately controlled by prefill and decode pool sizes, respectively.

### Costs

1. **Network latency per request:** Every request pays at least one KV transfer. For short prompts over a fast network, this is negligible. For long prompts over a slow link, it can dominate TTFT.
2. **Operational complexity:** Two pool types to manage, monitor, autoscale, and load-balance. The router must be aware of pool state.
3. **KV serialization overhead:** Serializing and deserializing KV tensors consumes CPU cycles on both ends (unless RDMA is used with GPU-direct).
4. **Load imbalance between pools:** If the prefill pool finishes quickly but the decode pool is saturated, prefill nodes sit idle. Dynamic pool resizing or spill-over strategies are needed.

---

## Common Misconceptions

### Misconception: Disaggregation is only useful for very large-scale deployments

**Correction:** Even a two-GPU single-server setup benefits from splitting prefill and decode. When a single GPU runs both phases, decode latency spikes during any prefill step. Assigning one GPU to prefill and another to decode within the same server eliminates the interference. The benefit is not about absolute scale — it is about removing the interference between two workloads with different hardware requirements.

### Misconception: Hierarchical caching only helps with long-context requests

**Correction:** Prefix sharing benefits any workload where many short requests share a common prefix. A chatbot deployment where all users share a 512-token system prompt sees nearly 100% prefix hit rate on that prefix regardless of whether user queries are short or long. Short, high-volume workloads with shared prefixes can achieve massive prefill savings. Long context is one motivating case; shared prefixes across millions of short requests is another and often larger one in practice.

### Misconception: Disaggregation is what makes inference fast

**Correction:** Disaggregation removes the mutual interference between prefill and decode. The speedup comes from that interference removal, not from the act of disaggregating itself. If a single GPU could perfectly time-multiplex prefill and decode with no interference, disaggregation would add no benefit. The benefit is specifically about the asymmetry in compute profiles — compute-bound prefill blocking memory-bandwidth-bound decode. This distinction matters when reasoning about whether disaggregation helps for a specific workload (short prompts, small models, low concurrency).

### Misconception: RDMA is mandatory for KV cache transfer

**Correction:** RDMA is the preferred mechanism for multi-host transfer because it achieves near-wire bandwidth with low CPU overhead. But it is not mandatory. For intra-host disaggregation (prefill and decode on GPUs within the same server), NVLink provides GPU-to-GPU copy at hundreds of GB/s — far faster than any network. For prototype or low-traffic deployments across hosts, standard TCP sockets with host memory staging are functional, just slower. RDMA is a performance optimization, not an architectural requirement.

### Misconception: Cache hits are free

**Correction:** A cache hit at the DRAM tier still requires a copy from DRAM to HBM before the KV tensors can be used in attention computation. At ~100–200 GB/s PCIe bandwidth, loading 1 GB from DRAM takes 5–10 ms — not free, but far cheaper than recomputing the KV tensors. Similarly, an NVMe hit (7 GB/s) for 1 GB of KV data takes ~140 ms, which is comparable to recomputing on a fast GPU. The cache hierarchy offers a cost spectrum, not a binary "hit = zero cost" outcome. System designers must account for promotion latency when quoting TTFT SLAs.

### Misconception: The prefill and decode split is always 1:1 in terms of GPU count

**Correction:** The optimal ratio of prefill GPUs to decode GPUs depends on the traffic distribution (prompt length, output length, concurrency) and on the hardware. A deployment serving many short prompts with long outputs might need 1 prefill GPU for every 8–10 decode GPUs. A code-generation workload with long prompts and short outputs might need 4 prefill GPUs per decode GPU. Capacity planning for disaggregated systems requires profiling the request distribution and modeling each pool's utilization separately.

---

## Open Questions in Disaggregation Research

### When does disaggregation overhead exceed its benefit?

For very short prompts (< 128 tokens), the KV cache is small and the prefill step is fast. The network transfer cost and routing overhead may exceed the interference savings. Empirically, this crossover point depends on the network bandwidth, model size, and concurrency level. Characterizing this as a function of prompt length and request rate is an open empirical question.

### How to handle dynamic load imbalance between pools

Request distributions are non-stationary. A burst of long-prompt requests saturates prefill nodes while decode nodes sit underutilized, and vice versa. Approaches include: hybrid nodes that can switch mode, dynamic pool resizing with container orchestration, and work-stealing between pools. None of these is fully solved at production scale.

### Should the router be KV-cache-hierarchy-aware?

In a hierarchical cache system, routing a request to a decode node that already has a matching prefix in its HBM tier avoids a DRAM or NVMe promotion. This is **locality-aware routing**. Mooncake implements a form of this by routing requests to nodes that hold the matching KV prefix. But this introduces routing state that must be kept consistent at high QPS, creating a new distributed systems problem. The tradeoff between routing locality and routing overhead is not fully resolved.

### Should prefill happen incrementally (chunked prefill)?

Chunked prefill processes the prompt in fixed-size chunks (e.g., 512 tokens at a time), interleaving with decode steps to reduce maximum latency spikes. This is an alternative to full disaggregation that avoids the KV transfer overhead. vLLM's chunked prefill is this approach. The question of when chunked prefill is preferable to full disaggregation — and whether the two can be combined — is an active design space.

---

## Summary

Prefill-decode disaggregation addresses a fundamental compute-profile mismatch within LLM inference. Prefill is compute-bound (GPU-saturating matrix multiplications over many tokens); decode is memory-bandwidth-bound (one token per step, low arithmetic intensity). Running both on the same GPU causes mutual interference that raises tail latency and lowers throughput. Disaggregation assigns each phase to dedicated hardware, linked by a KV cache transfer over RDMA or NVLink.

Hierarchical caching extends this by recognizing that recomputing shared prompt prefixes is wasteful. A trie-indexed cache spanning HBM, DRAM, and NVMe allows frequently-used prefixes to be served from fast memory and rare prefixes from cheaper storage. Cache hits are not free — each tier below HBM requires a promotion transfer — but they are substantially cheaper than full recomputation.

The key production systems implementing these ideas are: Splitwise (Microsoft, 2023), Mooncake (Moonshot AI, 2024), DeepSeek-V3's serving infrastructure, and vLLM v0.6+. The open problems — optimal pool sizing, locality-aware routing, and the cost crossover point for short prompts — remain active research and engineering challenges.

---

## Key Numbers to Memorize

| Quantity | Value |
|---|---|
| H100 HBM bandwidth | ~3.35 TB/s |
| A100 HBM bandwidth | ~2 TB/s |
| PCIe Gen5 bidirectional bandwidth | ~128 GB/s |
| 100 Gb/s NIC effective throughput | ~12.5 GB/s |
| IB HDR effective throughput | ~25 GB/s |
| Decode arithmetic intensity | ~1 FLOP/byte |
| Prefill arithmetic intensity (S=4096) | ~4096 FLOP/byte |
| KV cache for 70B model at S=4096 (GQA, L=80, H=8, D=128, bf16) | ~671 MB |
| Transfer time at 100 Gb/s NIC | ~54 ms |
| Transfer time over NVLink (900 GB/s) | ~0.7 ms |

---

**References:**
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," Microsoft Azure, 2023.
- Zhong et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving," Moonshot AI, 2024.
- DeepSeek-V3 Technical Report, DeepSeek AI, 2024.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
- Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," OSDI 2024.
