# KV Cache (Area C — Inference)

During autoregressive decoding, a transformer recomputes attention over every prior token at each step. The KV cache stores the key (K) and value (V) projections of past tokens so they don't need to be recomputed. This trades memory for compute. PagedAttention (vLLM) extends this by managing the KV cache in fixed-size blocks similar to OS virtual memory, enabling efficient memory use across concurrent requests. A common misconception is that the KV cache stores raw token ids — it actually stores K and V tensors per layer and per head.
