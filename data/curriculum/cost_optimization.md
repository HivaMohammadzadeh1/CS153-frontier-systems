# Cost Optimization for ML Systems: Spot Instances, Mixed Precision, GPU Utilization, and FinOps

**Area F — Production ML Systems | Learning Memory OS Curriculum**

---

## 1. Why Cost Is a First-Class Engineering Metric

The cost of running a large language model in production is dominated by GPU compute. A single H100 SXM5 GPU costs approximately $3.00-4.00/hour on major cloud providers (A100: $2.00-3.00/hour). Training a 70B parameter model from scratch requires approximately 1-2 million GPU-hours — a cost of $3-8M. Serving that model at 10,000 requests/hour, with each request consuming 1 GPU-second of compute, requires ~2.8 H100-hours/hour — $8-12/hour at peak traffic, or $70-100K/year for a small-scale deployment.

Cost optimization is not optional. A model that costs 3× more to serve than necessary either prices the product out of the market or eliminates the margin that funds the next iteration. **FinOps for ML** — the practice of applying financial accountability to cloud ML workloads — is a distinct discipline from model performance optimization.

The four levers of ML cost optimization:
1. **Instance efficiency**: Use spot/preemptible instances where possible.
2. **Model efficiency**: Reduce compute per inference (quantization, distillation, smaller models).
3. **Batch efficiency**: Pack requests into batches to maximize GPU utilization.
4. **Infrastructure efficiency**: Eliminate idle time, cold starts, and overprovisioning.

---

## 2. Spot Instances and Preemption Handling

**Spot instances** (AWS) / **preemptible VMs** (GCP) / **spot VMs** (Azure) provide GPU compute at 50-80% discount compared to on-demand instances, with the caveat that the cloud provider can reclaim them at any time with 2-5 minutes' notice (GCP) or at instance interruption (AWS).

### 2.1 Spot for Training

Training jobs on spot instances are the canonical cost optimization: a 70B model training run that costs $4M on on-demand instances can be reduced to $800K-$1.6M on spot. The catch: the job must handle preemptions.

**Preemption handling for training**:
1. **Checkpoint frequently** (every 15-30 minutes for cloud training, more frequently for preemption-heavy instances).
2. **Detect preemption gracefully**: AWS sends an interruption notice 2 minutes before reclaim; register a signal handler for `SIGTERM` that writes a final checkpoint before exit.
3. **Auto-restart on new spot capacity**: Use managed training services (AWS SageMaker Managed Training with automatic restart, Google Cloud Vertex AI with automatic preemption handling) or self-managed Kubernetes with `restartPolicy: OnFailure` and a checkpoint-aware training script.
4. **Spot Fleet + On-Demand fallback**: If spot capacity is unavailable, fall back to a small on-demand pool to avoid stalling the job indefinitely. Accept a higher cost for short periods to prevent job starvation.

### 2.2 Spot for Inference

Spot instances are harder to use for inference than training because serving requires continuous availability. Strategies:
- **Over-provision with spot**: Run 2× the required GPU count on spot instances. With typical spot interruption rates of 5-15%, the probability that both a spot instance and its replacement are simultaneously unavailable is very low.
- **Spot + on-demand buffer**: Route base traffic to spot instances; have a small on-demand pool that absorbs traffic during spot interruptions.
- **Stateless serving with load balancing**: Ensure serving pods are stateless (no in-memory KV cache state that is lost on preemption). vLLM's KV cache is in-process memory; preemption loses the cache, so the first request after restart has degraded TTFT. Design the routing layer to absorb this gracefully.

---

## 3. Mixed Precision for Cost Reduction

Training and inference in lower precision reduces memory and compute cost:

| Precision | Memory per parameter | Compute throughput (H100) | Typical use |
|-----------|---------------------|--------------------------|-------------|
| FP32 | 4 bytes | 67 TFLOPS | Legacy, avoid |
| BF16 | 2 bytes | 1,979 TFLOPS | Standard training |
| FP8 (e4m3/e5m2) | 1 byte | 3,958 TFLOPS | Training + inference |
| INT8 | 1 byte | 3,958 TOPS | Inference |
| INT4 | 0.5 bytes | ~2,000 TOPS* | Quantized inference |

*INT4 throughput depends heavily on hardware dequantization overhead.

### 3.1 BF16 for Training

BF16 is now the standard training precision for large models. It has the same exponent range as FP32 (8 exponent bits) with lower mantissa precision (7 bits vs 23 bits). The key advantage over FP16 is numerical stability: BF16 does not overflow for the gradient scales typical of large model training. Modern frameworks (PyTorch AMP with BF16, Megatron-LM) use BF16 for most computations and FP32 for optimizer states (the "master weights" in mixed precision training).

**BF16 training cost saving**: Using BF16 instead of FP32 halves weight memory and doubles the tensor core throughput, reducing training time by ~1.5-2x. On cloud pricing, this halves the cost of a training run (same walltime = half the GPU count, or same GPU count = half the walltime, both at the same cost).

### 3.2 FP8 for Training and Inference

FP8 (H100 supports FP8 via Transformer Engine) halves memory again vs BF16 and doubles tensor core throughput to 3,958 TFLOPS. FP8 training requires careful handling of numerical stability: the two FP8 formats (e4m3 for forward pass, e5m2 for backward pass) have different dynamic range characteristics. Transformer Engine from NVIDIA handles the scaling and casting automatically.

For inference, FP8 quantization reduces model weight size by 2× vs BF16 and increases throughput by ~1.5-2× for memory-bandwidth-bound workloads (typical at small batch sizes).

### 3.3 INT4 for Inference

INT4 (GPTQ, AWQ, or similar weight-only quantization) reduces weight storage by 4× vs FP32, 2× vs INT8. At small batch sizes (batch=1 or batch=4), LLM inference is memory-bandwidth-bound: the bottleneck is reading weight matrices from HBM. INT4 quantization reduces memory bandwidth requirements by 4× (vs FP32) or 2× (vs BF16), yielding proportional throughput improvements.

**Cost implication**: A 70B model serving INT4 on 2 A100 GPUs can match the throughput of BF16 on 4 A100 GPUs at identical request rates. For serving, INT4 at correct quality levels can save 40-50% of GPU cost.

---

## 4. Batch Packing and GPU Utilization

**GPU utilization** (the fraction of time the GPU executes kernels) is the most direct proxy for cost efficiency. A GPU that is 30% utilized is paying 100% of the hourly rate for 30% of the potential throughput — effectively a 3× cost premium.

### 4.1 Static Batching

Static batching groups requests by input length and creates batches of fixed size. The problem: padding waste. If batch size = 32 and one request has 1000 tokens while the rest have 100 tokens, the other 31 requests are padded to 1000 tokens — 31 × 900 tokens of wasted compute.

### 4.2 Dynamic Batching and Continuous Batching

**Dynamic batching** (used in vLLM, TGI) collects requests arriving within a time window and batches them together. **Continuous batching** (also called in-flight batching) goes further: it interleaves prefill and decode phases for different requests in the same iteration, keeping the GPU continuously occupied.

For cost optimization, continuous batching increases effective GPU utilization from ~40-60% (static batching with variable-length requests) to ~70-85%. This improvement alone can reduce serving cost by 1.5-2×.

### 4.3 Batch Packing Strategies

**Bin packing**: Group requests by sequence length into bins (e.g., 0-128 tokens, 128-512 tokens, 512-2048 tokens). Apply different batch sizes per bin (larger batches for shorter sequences, smaller for longer). This minimizes padding waste without the complexity of continuous batching.

**Sequence bucketing**: A variant of bin packing that pads requests within each bucket to the maximum length in that bucket, rather than the global maximum. Used in Megatron-LM training to reduce padding waste during training.

---

## 5. MFU as a Cost Lever

**MFU (Model FLOP Utilization)** = achieved FLOP/s / theoretical peak FLOP/s. For an H100 with BF16: peak = 1,979 TFLOPS.

MFU is the primary efficiency metric for training and compute-bound inference. Low MFU means the GPU is doing less useful work per second than it could — equivalent to wasting money.

| Scenario | Typical MFU | Cause |
|----------|------------|-------|
| Large model, large batch, BF16, full throughput | 45-55% | Memory bandwidth bottleneck, operator overhead |
| Large model, small batch (bs=1) | 5-15% | Memory-bandwidth-bound (weight loading dominates) |
| FlashAttention2 enabled | +5-10% MFU | Reduced memory bandwidth for attention |
| INT4 quantization (bs=1) | 25-40% | Better memory bandwidth utilization |
| Poorly configured NCCL (for training) | -10-20% MFU | Communication overhead |

Improving MFU from 35% to 55% reduces training compute time by 36%, with a proportional reduction in cost. The primary levers:
1. **Use FlashAttention**: Reduces attention memory bandwidth overhead.
2. **Increase batch size**: Shifts the model toward compute-bound operation.
3. **Overlap communication with computation**: In data-parallel training, overlap gradient all-reduce with the backward pass.
4. **Mixed precision**: Use BF16 or FP8 to access higher tensor core throughput.

---

## 6. DCGM: GPU Utilization Metrics in Practice

**DCGM (Data Center GPU Manager)** is NVIDIA's tool for GPU health monitoring and profiling. For cost optimization, the relevant DCGM metrics are:

- `DCGM_FI_DEV_GPU_UTIL`: Percent of time the GPU was active (kernel running). This is the "GPU utilization" shown in most dashboards. Target: > 80%.
- `DCGM_FI_DEV_MEM_COPY_UTIL`: Percent of time the GPU memory bus was active (memory transfers). High memory copy utilization relative to GPU utilization indicates memory-bandwidth-bound operation.
- `DCGM_FI_PROF_SM_ACTIVE`: Fraction of streaming multiprocessors (SMs) that were active. More accurate than `GPU_UTIL` for roofline analysis.
- `DCGM_FI_PROF_TENSOR_ACTIVE`: Fraction of cycles the tensor cores were active. For a BF16 matmul-dominated workload, this should be > 60%.

Exposing DCGM metrics via Prometheus + Grafana is the standard monitoring stack for GPU clusters. A dashboard showing `TENSOR_ACTIVE` < 30% for a training job is a direct indicator of wasted money.

---

## 7. FinOps for ML

**FinOps** (Financial Operations) applies DevOps-style practices to cloud cost management. For ML teams, FinOps includes:

**Tagging and attribution**: Every GPU instance is tagged with `project`, `team`, `experiment_id`, and `cost_center`. Cost reports broken down by team and project enable accountability.

**Budget alerts**: Cloud-level budget alerts trigger when monthly spend exceeds a threshold. ML-level budget enforcement (e.g., "this experiment has a $10,000 GPU budget") requires custom tooling — budget checks in the training script's health loop.

**Rightsizing**: Regularly audit whether deployed model instances are oversized. A model serving 100 requests/day does not need 8 H100s on-demand. Rightsizing tools (AWS Compute Optimizer, Google Cloud Recommender) provide recommendations based on observed utilization.

**Reserved instances**: For stable, predictable workloads (production serving), Reserved Instances (1-year or 3-year commitment) reduce cost by 30-60% vs on-demand. Do not over-reserve — unused reservations have no value.

---

## 8. Cold-Start Tradeoffs

**Cold start** is the latency penalty incurred when a serving pod starts from scratch: pulling the container image (~2-10 minutes for large ML images), loading model weights from storage (~1-5 minutes for 70B at 1 GB/s), and JIT-compiling kernels (~5-30 minutes for TensorRT). During cold start, the pod cannot serve requests.

**Cost-cold start tradeoff**: Maintaining a minimum number of warm pods ("minimum hot instances") eliminates cold start for new requests but pays for idle GPU time during low-traffic periods. Autoscaling to zero (scale the deployment to 0 pods during off-hours) saves significant cost but causes cold starts for the first requests after the off-hours window.

**Strategies**:
- **Quantized weight caching**: Cache dequantized weight tensors in instance memory between requests. Avoid reloading from disk on every request.
- **Container image optimization**: Store model weights separately from the container image; use a volume mount or pre-populated node local SSD to avoid pulling weights from S3 at pod start.
- **Graduated scale-down**: Scale down slowly (e.g., reduce capacity by 25% per 15 minutes during a traffic decrease) rather than immediately. This prevents oscillation and gives in-flight requests time to complete.
- **Request buffering**: During cold start, buffer incoming requests in a queue rather than immediately returning 503. This accepts latency increases in exchange for higher availability.

---

## Misconception: Lower precision always reduces cost

Lower precision reduces *memory and compute* cost per token, but the net effect on serving cost depends on whether the workload is compute-bound or memory-bandwidth-bound. At large batch sizes, transformer inference is compute-bound: INT4 offers 2× memory savings but the compute throughput for INT4 matmul may be limited by dequantization overhead. At small batch sizes (bs=1), inference is memory-bandwidth-bound: INT4 reduces HBM reads by 4× vs FP32, directly improving throughput. The cost benefit of quantization must be validated empirically for each serving configuration; do not assume it is free.

## Misconception: Spot instances are unsuitable for production ML serving

Spot instances are unsuitable for *stateful* production serving — if a pod is reclaimed, in-flight requests fail and KV caches are lost. But for *stateless* serving with load balancing and automatic request routing, spot instances can provide 99.9%+ availability at 50-70% cost reduction, using a mix of spot (70-80%) and on-demand (20-30%) instances. The key requirement is graceful pod termination: the load balancer must drain in-flight requests from a spot pod before it is reclaimed, which requires the application to handle `SIGTERM` with a drain period.

## Misconception: GPU utilization = MFU

GPU utilization (`DCGM_FI_DEV_GPU_UTIL`) measures whether any kernel is running, not whether that kernel is efficient. A GPU can show 100% utilization while the kernel is only achieving 20% of the theoretical FLOP/s (e.g., a memory-bandwidth-bound elementwise add that runs continuously). MFU measures how efficiently the GPU's compute units are used. A GPU running inefficient kernels at 100% utilization has the same cost as a GPU running optimal kernels at 100% utilization but produces far less work per dollar. Both utilization and MFU are necessary metrics; neither alone is sufficient.

## Misconception: Reserved instances are always better than on-demand for stable workloads

Reserved instances save 30-60% vs on-demand for the committed capacity, but unused reservations have zero salvage value. If traffic decreases by 40% (due to seasonal patterns, product changes, or a switch to a more efficient model), the reserved capacity that was right-sized for peak traffic becomes wasteful. Before committing to 1- or 3-year reservations for GPU capacity, validate utilization stability over at least 6 months, and reserve only the minimum baseline capacity while keeping burst capacity on-demand or spot.

## Misconception: Cost optimization is only relevant after model quality is finalized

Cost optimization and model quality are deeply intertwined. A 70B model that achieves quality score 0.87 at $0.50/request competes with a 7B model that achieves score 0.82 at $0.05/request. For many applications, the smaller, cheaper model's quality is "good enough." Engineering teams should run **cost-quality Pareto analyses** throughout model development, not just at deployment. Choosing the right model size for the serving requirement (and right quantization level) can reduce cost by 10-50× with acceptable quality degradation. This is a fundamentally different optimization than infrastructure FinOps.

---

## 9. Practical Example: Cost Analysis for Llama-3-70B Serving

**Setup**: 100K requests/day, average 512 input + 256 output tokens, p99 latency SLA < 500ms.

**Baseline (BF16, on-demand A100)**:
- Throughput per A100: ~400 output tok/s at batch=8 with vLLM
- Required capacity: 100K × 256 tokens/day / (86,400s × 400 tok/s) ≈ 0.74 A100s
- With 2× headroom: 2 A100 instances on-demand
- Cost: 2 × $3.00/hr × 24hr × 365 = **$52,560/year**

**Optimized (INT4, spot + on-demand mix)**:
- INT4 throughput improvement: ~2× (memory-bandwidth-bound at batch=8) → 800 tok/s per A100
- Required capacity: 0.37 A100s → 1 A100 with headroom
- Spot pricing (70% discount): $0.90/hr; 80% spot + 20% on-demand
- Cost: 1 GPU × (0.8 × $0.90 + 0.2 × $3.00) × 24 × 365 = **$8,380/year**

**Total savings: $44,180/year (84%)** from quantization (2×) + spot instances (3.3×).

---

## 10. Exercise

**Exercise**: You are deploying a 13B parameter LLM (BF16, 26 GB weights) for a document summarization API with the following requirements: 1,000 requests/hour, average 2,048 input + 512 output tokens, p99 latency < 2 seconds. GPU: A100 80GB ($2.50/hr on-demand). Design a cost-optimal serving configuration specifying: (1) quantization level and expected throughput, (2) number of GPU instances needed (with utilization target), (3) spot vs on-demand split and preemption handling strategy, (4) autoscaling policy (scale-out threshold, scale-in delay), and (5) annual cost estimate. Compare three configurations: BF16 on-demand, INT8 on-demand, INT8 with 80% spot.

---

## References

- DCGM documentation: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/
- GPTQ: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023)
- AWQ: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- NVIDIA Transformer Engine (FP8): https://docs.nvidia.com/deeplearning/transformer-engine/
- vLLM continuous batching: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- AWS Spot Instance advisor: https://aws.amazon.com/ec2/spot/instance-advisor/
- FinOps Foundation: https://www.finops.org
