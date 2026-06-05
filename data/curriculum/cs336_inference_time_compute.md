# Inference-Time Scaling (Test-Time Compute)

## The idea
Instead of only scaling training, you can spend more compute *at inference* to get better answers — "test-time compute." For reasoning tasks this can beat a larger model that thinks less.

## Techniques
- **Chain-of-thought / longer generations**: let the model reason in tokens before answering.
- **Best-of-n / sampling + selection**: generate n candidates and pick the best with a verifier or majority vote (self-consistency).
- **Search**: tree/beam search over reasoning steps, guided by a process or outcome reward model.
- **Verifier / reward models**: score candidate solutions; trade extra generations for accuracy.
- **Reasoning models**: trained (often with RL) to produce long internal reasoning traces before the final answer.

## The systems cost
Test-time compute is a latency/throughput/cost tradeoff: best-of-n multiplies token generation by n; long reasoning traces inflate output length and therefore TPOT-driven latency and KV-cache usage. Serving reasoning models stresses decode and KV memory far more than short-answer models, which changes capacity planning and batching.

## The scaling tradeoff
There's a frontier between *training-time* compute (bigger/better-trained model) and *inference-time* compute (more thinking per query). The right point depends on how often you query and how much each correct answer is worth.

## Interview-relevant reasoning
"How does serving a reasoning model differ?" — much longer outputs → decode-bound, large KV growth, worse tail latency; you budget for output length and may cap reasoning. "When is best-of-n worth it?" — when a verifier is reliable and accuracy matters more than per-query cost/latency.
