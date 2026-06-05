# Admission Control and Load Shedding

## The problem
GPUs have finite KV-cache memory and throughput. Under a traffic burst, naively accepting every request causes KV-cache exhaustion, preemption/recompute storms, OOM, or unbounded queueing — and the whole system degrades for everyone. Admission control protects the system by deciding *which* requests to accept, queue, or reject.

## Techniques
- **Concurrency / KV-budget limits**: cap the number of in-flight sequences so the KV cache never overcommits. New requests queue until a slot frees.
- **Queue bounds + load shedding**: bound the queue; when full, reject (HTTP 429) or route to a fallback rather than letting latency grow without limit. A fast failure beats a request that times out anyway.
- **Prioritization / fairness**: protect interactive traffic over batch; enforce per-tenant quotas so one customer can't starve others.
- **Backpressure**: signal upstream to slow down instead of silently dropping.

## Graceful degradation
When saturated, degrade deliberately: shed the lowest-priority traffic, serve a smaller/cheaper model, cap max output length, or disable optional features — keep the core SLO intact for the requests you do accept.

## The key tradeoff
Admission control trades *availability for some* against *degraded service for all*. Accepting everything under overload is the failure mode that makes incidents worse, not better.

## Interview-relevant reasoning
"OOM / capacity collapse during a traffic spike" is usually an admission-control gap: the system accepted more concurrent sequences than KV cache could hold. The fix is a concurrency cap + bounded queue + load shedding, not "add more GPUs" (which just moves the cliff).
