# Multi-Tenant Inference: Queueing, Fairness, SLA Management, and Quotas

**Area F — Production ML Systems | Learning Memory OS Curriculum**

---

## 1. The Multi-Tenant Serving Problem

A single large language model deployment often serves multiple tenants simultaneously: different product teams, different business customers, or different user tiers within a single product. Multi-tenancy creates a resource contention problem: the GPUs and KV cache memory are shared, but tenants have different priority levels, different SLA requirements, and different request volumes. Without explicit fairness enforcement, a single tenant with high request volume can monopolize GPU capacity and starve other tenants — a problem called **head-of-line blocking** or **noisy neighbor** interference.

Multi-tenant inference requires four mechanisms:
1. **Request scheduling**: Decide which queued request to serve next.
2. **Fair queueing**: Ensure no tenant monopolizes resources indefinitely.
3. **Rate limiting and quotas**: Enforce per-tenant resource bounds.
4. **Isolation**: Prevent tenants from interfering with each other's latency, memory, or behavior.

---

## 2. Request Scheduling

At every inference step, the scheduler decides which requests to process in the current iteration. The naive approach — first-come-first-served (FCFS) — is simple but unfair and inefficient.

### 2.1 First-Come-First-Served (FCFS)

FCFS processes requests in arrival order. Problems:
- **Head-of-line blocking**: A single long request (e.g., 8,192 token generation) blocks all shorter requests behind it in the queue.
- **No priority enforcement**: A low-priority background task arriving before a high-priority interactive request will be served first.
- **Unfair across tenants**: A tenant that submits a burst of requests at time T monopolizes capacity until their burst is processed, regardless of other tenants' needs.

### 2.2 Shortest Job First (SJF)

SJF prioritizes requests with the smallest estimated completion time. For LLM serving, completion time is approximately proportional to output length — shorter expected outputs are processed first. SJF minimizes average waiting time but requires output length prediction, which is imperfect. SJF is also unfair to requests with long outputs (starvation).

### 2.3 Priority Queues

Priority queues assign each request a numerical priority. At each scheduling step, the highest-priority queued request is served next. Priority can be assigned based on:
- **Tenant tier**: Premium tier gets priority 1, standard tier gets priority 2, background tier gets priority 3.
- **Request urgency**: Requests flagged by the client as interactive (synchronous user waiting) get higher priority than batch requests.
- **Waiting time**: Priority increases with queue wait time (aging), preventing starvation of low-priority requests.

vLLM implements a priority queue via a custom `Scheduler` class. The `Policy` enum includes `FCFS`, `Priority`, and custom implementations.

### 2.4 Continuous Batching with Per-Tenant Awareness

vLLM's **PagedAttention + continuous batching** scheduler processes requests in iterations, interleaving prefill and decode for different requests. The scheduler maintains a running batch: at each step, it may add a new request (prefill), continue decoding existing requests, and remove completed requests. With per-tenant awareness, the scheduler tracks KV cache usage per tenant and avoids allocating all available KV cache to one tenant's requests.

---

## 3. Fair Queueing Across Tenants

**Fair queueing** ensures that over any time window, each tenant receives a fair share of resources proportional to their allocated quota. The classical algorithm for fair queueing is **Weighted Fair Queuing (WFQ)**.

### 3.1 Weighted Fair Queuing

WFQ assigns each tenant a weight `w_i` (proportional to their quota). The scheduler simulates an idealized fluid model where each tenant gets `w_i / Σ_j w_j` fraction of capacity at all times. Real packet (or request) scheduling approximates this by tracking a **virtual time** for each tenant's queue.

For ML inference:
- Each request has a "work size" proportional to its compute cost (input + output token count).
- Tenant i's virtual finish time for a request of size L is: `VF_i = max(VF_prev_i, virtual_time) + L / w_i`.
- The scheduler always serves the request with the smallest virtual finish time.

This ensures that over any long interval, tenant i receives `w_i / Σ w_j` fraction of GPU-time. A tenant with weight 2 receives twice as many GPU-time units as a tenant with weight 1, regardless of request burst patterns.

### 3.2 Max-Min Fairness

Max-min fairness allocates resources to maximize the minimum share received by any tenant. This is especially useful when some tenants are idle (their allocation can be shared with active tenants), but the moment an idle tenant becomes active, their fair share is restored. The algorithm: sort tenants by demand; allocate to the tenant with lowest current allocation first; if a tenant is below its fair share and has demand, serve it next.

---

## 4. Per-Tenant Rate Limits and SLAs

### 4.1 Token Bucket Rate Limiting

The standard algorithm for rate limiting is the **token bucket**. Each tenant has a bucket with capacity C tokens. Tokens refill at rate R per second. Each request consumes a number of tokens proportional to its cost (e.g., input + output token count). If the bucket has insufficient tokens, the request is queued or rejected.

```python
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()

    def consume(self, tokens: int) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity,
                          self.tokens + (now - self.last_refill) * self.refill_rate)
        self.last_refill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True  # request allowed
        return False  # request exceeds rate limit
```

For a tenant with a quota of 10,000 tokens/minute, set `refill_rate = 10000/60 ≈ 167 tokens/second` and `capacity = 10000` (allow short bursts up to 10,000 tokens).

### 4.2 Per-Tenant SLA Tiers

Multi-tenant systems typically have 2-4 SLA tiers:

| Tier | TTFT SLA | Throughput guarantee | Max queue wait | Use case |
|------|----------|---------------------|----------------|----------|
| Premium | p99 < 200ms | Guaranteed | 0ms (immediate) | Interactive, real-time |
| Standard | p99 < 1s | Best-effort | 5s | Standard API |
| Background | p99 < 10s | Best-effort | 60s | Batch processing |
| Async | No TTFT SLA | Best-effort | 300s | Offline jobs |

SLA enforcement requires the scheduler to prioritize premium-tier requests over standard-tier requests, even if standard-tier requests have been waiting longer. The aging mechanism prevents starvation: after a standard-tier request has waited more than `max_queue_wait`, it is elevated to premium priority.

### 4.3 Admission Control

**Admission control** rejects requests when the system is overloaded, rather than allowing unlimited queue growth. An overloaded queue causes all requests to violate their latency SLAs — it is better to reject some requests early (returning 429 Too Many Requests) than to delay all requests beyond their SLAs.

The admission criterion: admit a new request if the estimated queue drain time is below the SLA deadline. For a priority queue, the estimated drain time requires summing the work sizes of all higher-priority requests ahead of the new request. This is O(N) for a naïve implementation; approximate admission control based on current queue length and average request cost is used in practice.

---

## 5. Isolation Strategies

Multi-tenancy requires isolation at multiple levels:

### 5.1 Memory Isolation: KV Cache Quota

In a vLLM-style deployment, the KV cache is the primary shared memory resource. Without isolation, a single tenant's requests can fill the entire KV cache, causing other tenants' requests to be preempted (their KV pages swapped to CPU or evicted). This causes unbounded latency spikes for non-offending tenants.

**KV cache quota per tenant**: Allocate a maximum number of KV cache pages per tenant. If a tenant's active requests would exceed their quota, the scheduler delays their new requests rather than evicting other tenants' pages. This prevents KV cache starvation at the cost of queuing the offending tenant's requests.

### 5.2 Process-Level Isolation

The strongest isolation: each tenant's requests are served by a separate model instance in a separate process. Tenants cannot interfere with each other's memory or GPU time (the OS scheduler enforces process-level fairness). The cost: each process loads a full copy of the model weights, which is prohibitively expensive for large models (a 70B BF16 model uses 140 GB — most GPU servers can host only one copy).

Process-level isolation is practical for smaller models (7B, 13B) on multi-GPU servers where each process owns a subset of GPUs.

### 5.3 Container-Level Isolation

Kubernetes namespaces + resource quotas provide container-level isolation: each tenant's model server runs in a separate pod with CPU, memory, and GPU resource limits. The Kubernetes scheduler enforces resource quotas; a tenant's pod cannot use more than its allocated GPU capacity.

For multi-GPU servers, the **NVIDIA MIG (Multi-Instance GPU)** feature partitions a single physical GPU into isolated slices, each with its own memory and compute bandwidth. MIG provides hardware-level isolation suitable for multi-tenant serving of smaller models on shared GPU infrastructure.

### 5.4 Kubernetes Namespace Isolation

At the cluster level, separate Kubernetes namespaces per tenant provide network policy isolation, resource quota enforcement, and RBAC. Each namespace has:
- `ResourceQuota`: Maximum GPU count, CPU, memory for all pods in the namespace.
- `LimitRange`: Per-pod resource limits.
- `NetworkPolicy`: Prevents cross-tenant network communication.

This is the preferred isolation model for cloud ML platforms (Kubeflow, Vertex AI, SageMaker) where tenants share a cluster but must not interfere with each other.

---

## 6. vLLM-Style Continuous Batching with Per-Tenant Fairness

vLLM's scheduler (as of v0.4+) supports basic priority-based scheduling. A production multi-tenant deployment on top of vLLM requires additional per-tenant accounting:

**Enhanced scheduling loop**:
1. **Per-tenant queue**: Maintain separate request queues per tenant, each with its own priority and rate-limiting state.
2. **Virtual time tracking**: For each tenant, track virtual finish time (WFQ algorithm) to determine which tenant's request to schedule next.
3. **KV page accounting**: Track KV pages used per tenant; block new requests if a tenant exceeds their KV quota.
4. **SLA deadline tracking**: Annotate each request with its SLA deadline; promote requests approaching their deadline regardless of tenant tier.

A reference implementation: the **Llumnix** system (from the Llumnix paper) implements cross-instance request migration for multi-tenant LLM serving, allowing hot-spot tenants' requests to be migrated to less loaded instances while preserving KV cache state.

---

## Misconception: Multi-tenant serving is just load balancing

Load balancing distributes requests across instances for capacity but does not enforce per-tenant fairness or isolation. A load balancer will happily route all of tenant A's requests to the same instance as tenant B's requests, with no guarantee that either tenant receives their fair share of GPU compute. Multi-tenant serving requires fairness enforcement at the scheduling layer (within each instance) and isolation at the infrastructure layer (between instances). Both are necessary for SLA compliance.

## Misconception: Rate limiting at the API gateway is sufficient for multi-tenant fairness

API gateway rate limiting (e.g., enforcing 1,000 requests/minute per API key) prevents a tenant from exceeding a *request count* limit, but not a *resource consumption* limit. A tenant who sends 1,000 very long requests (each consuming 10,000 tokens) consumes 10× more GPU resources than a tenant who sends 1,000 short requests (1,000 tokens each). Fair rate limiting for LLM APIs must be token-based (input + output tokens consumed), not request-count-based. Token bucket rate limiting with per-request token counting is required.

## Misconception: KV cache preemption (swapping to CPU) is free

When a tenant's requests are preempted and their KV pages are swapped to CPU (as in vLLM's preemption handling), restarting that request requires re-transferring the KV pages from CPU memory to GPU HBM. At 96 GB/s (PCIe 4.0 bandwidth), swapping 10 GB of KV cache (a long context request) takes ~100ms. For interactive tenants with 200ms TTFT SLAs, a single preemption event blows the SLA. KV page swapping is a mechanism of last resort for preventing out-of-memory conditions, not a routine fairness mechanism. Fair KV cache allocation (per-tenant quotas) prevents the need for preemption in the first place.

## Misconception: Stronger isolation always leads to better tenant outcomes

Strict process-level or instance-level isolation eliminates noisy neighbor interference but also eliminates resource sharing. If tenant A is idle and tenant B has a burst, process-level isolation prevents tenant B from using tenant A's idle GPU capacity — wasting resources. The optimal isolation level depends on the workload: for tenants with predictable, stable workloads and strict SLAs, strong isolation is worth the resource waste. For tenants with bursty workloads and soft SLAs, shared-instance WFQ scheduling with KV quotas provides better average performance at the cost of some SLA variance.

## Misconception: Priority queues prevent all SLA violations under overload

Priority queues guarantee SLA compliance for high-priority tenants *only when the system is not overloaded*. If the total request rate exceeds system capacity, even high-priority requests will eventually violate their SLAs (the queue grows without bound). Admission control is the necessary companion to priority queueing: when the system is overloaded, reject low-priority requests rather than queuing them indefinitely. The combination of priority scheduling + admission control provides deterministic SLA guarantees for high-priority tenants even under extreme overload.

---

## 7. Practical Example: Multi-Tenant LLM API with Three Tiers

Setup: A shared 8-H100 cluster serves a public LLM API with three tiers — Enterprise (100 customers, p99 TTFT < 300ms), Pro (1,000 customers, p99 TTFT < 1s), and Free (10,000 customers, p99 TTFT < 10s).

**Rate limits**:
- Enterprise: 500,000 tokens/minute per customer
- Pro: 50,000 tokens/minute per customer
- Free: 10,000 tokens/minute per customer

**Scheduling policy**: WFQ with weights [Enterprise=10, Pro=3, Free=1]. A long-running Free-tier request does not block Enterprise-tier requests; the WFQ virtual time ensures Enterprise tenants receive 10/(10+3+1) = 71% of GPU time even during peak load.

**KV cache allocation**:
- Enterprise: Up to 40% of total KV pages (32 GB / 80 GB total KV budget)
- Pro: Up to 35% of total KV pages
- Free: Up to 25% of total KV pages

**Admission control**: Reject Free-tier requests with HTTP 429 when estimated queue wait > 10s. Reject Pro-tier requests when estimated queue wait > 1s. Enterprise requests are never rejected (they are queued up to 5 minutes before return 503).

This configuration ensures Enterprise SLAs are met even when Free-tier traffic is 100× Pro-tier traffic, while maximizing GPU utilization during off-peak periods by allowing any tier to consume unused capacity.

---

## 8. Exercise

**Exercise**: Implement a multi-tenant request scheduler for a vLLM-based LLM serving system. The scheduler should support: (1) token-bucket rate limiting per tenant (with configurable capacity and refill rate), (2) WFQ scheduling across tenants with configurable weights, (3) per-tenant KV cache page quotas, and (4) SLA-deadline-based request promotion (requests within 100ms of their SLA deadline are elevated to the highest priority). Test your scheduler with three simulated tenants: a Premium tenant sending 10 rps with 512-token requests, a Standard tenant sending 20 rps with 256-token requests, and a Background tenant sending 5 rps with 2048-token requests. Report: p95/p99 latency per tier, GPU utilization, and request rejection rates under 2× overload.

---

## References

- vLLM scheduler: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- Llumnix: https://github.com/AlibabaPAI/llumnix — live migration for multi-tenant LLM serving
- Weighted Fair Queuing: Demers et al., "Analysis and Simulation of a Fair Queueing Algorithm" (SIGCOMM 1989)
- NVIDIA MIG documentation: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- Kubernetes ResourceQuota: https://kubernetes.io/docs/concepts/policy/resource-quotas/
- "Site Reliability Engineering" (Google, 2016) — Chapter on handling overload and admission control
- Sarathi-Serve: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (OSDI 2024) — straggler-aware batching in multi-tenant settings
