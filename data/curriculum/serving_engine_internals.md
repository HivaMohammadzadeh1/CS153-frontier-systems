# Serving Engine Internals (vLLM, PagedAttention, scheduling)

## Why a dedicated serving engine
General inference loops waste memory and GPU cycles on LLM workloads. Modern engines (vLLM, TensorRT-LLM, SGLang) are built around the unique shape of autoregressive decoding: dynamic, per-request KV-cache growth and highly variable sequence lengths.

## PagedAttention
The KV cache for a sequence grows one token at a time and you don't know the final length. Naive contiguous allocation causes massive internal/external fragmentation and forces over-reservation. **PagedAttention** (vLLM) borrows OS virtual-memory paging: KV cache is split into fixed-size blocks allocated on demand and referenced through a block table. This near-eliminates fragmentation, enables high batch occupancy, and makes prefix sharing (copy-on-write blocks) cheap.

## Continuous batching
Instead of static batches that wait for the slowest member, the scheduler runs **continuous (in-flight) batching**: finished sequences leave the batch and new ones join every step. This keeps the GPU saturated and dramatically raises throughput vs. static batching.

## The scheduler
Each step the scheduler decides which running/waiting sequences to execute given the KV budget. It handles admission, preemption (swap or recompute when memory is tight), and prioritization. Scheduler behavior explains many tail-latency and throughput phenomena.

## Parallelism knobs
Engines expose tensor, pipeline, data, and (for MoE) expert parallelism, plus context parallelism for long sequences. Each adds communication overhead, so the right choice depends on model size, sequence length, and hardware topology.

## Interview-relevant reasoning
"High GPU utilization but low useful throughput" often points at scheduling/memory stalls or preemption, not raw compute. Knowing PagedAttention + continuous batching lets you reason about why a serving stack behaves the way it does under load.
