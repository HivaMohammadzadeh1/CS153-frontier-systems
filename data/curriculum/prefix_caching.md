# Prefix and Context Caching

## The idea
Many LLM requests share a common prefix — a long system prompt, few-shot examples, or the earlier turns of a conversation. Prefix caching stores the KV-cache for those shared tokens so the prefill phase can be skipped (a cache hit), turning an expensive recompute into a cheap lookup. This is one of the biggest TTFT wins in production serving.

## How it works
The serving engine hashes the token prefix and keys the cached KV blocks by that hash. When a new request arrives whose prefix matches, the engine reuses the cached KV blocks (often via block-level / PagedAttention allocation) and only prefills the new suffix. vLLM's automatic prefix caching and "RadixAttention" (SGLang) are well-known implementations.

## Cache-hit rate is the metric
The lever is **cache-hit rate**. A high hit rate slashes prefill compute and TTFT; a low hit rate means you're paying full prefill every time.

## What breaks it
- **Prompt-template changes**: changing even one token at the start of the shared prefix invalidates the cache for everyone downstream. A p99 TTFT regression right after a prompt-template deploy is a classic incident — the prefix-cache hit rate dropped.
- **Per-request variation early in the prompt** (e.g. injecting a timestamp or user id at the top) prevents sharing — put variable content at the end.
- **Eviction under memory pressure**: caches are bounded; LRU/▒eviction lowers hit rate during bursts.

## Interview-relevant reasoning
If long conversations are slow but decode is fine, and a recent change preceded it, suspect prefix-cache invalidation. Fix: restore template stability, move variable tokens to the suffix, and size the cache for the working set.
