# CS336 Lecture 4 — Mixtures of Experts

**Source:** Stanford CS336 Spring 2025, Tatsu H.

## Mixture of Experts

GPT4 allegedly uses MoE.

## What's a MoE?

Replace the big feedforward network with many large feedforward networks and a selector layer [Fedus et al 2022]. You can increase the number of experts without affecting FLOPs.

## Why Are MoEs Getting Popular?

1. **Same FLOP, more parameters does better** [Fedus et al 2022]
2. **Faster to train MoEs** [OlMoE]
3. **Highly competitive vs dense equivalents**
4. **Parallelizable to many devices**

MoEs are most of the highest-performance open models, and are quite quick.

**Earlier MoE results from Chinese groups – Qwen:** Chinese LLM companies are also doing quite a bit of MoE work on the smaller end.

**Earlier MoE results from Chinese groups – DeepSeek:** Good recent ablation work on MoEs showing they're generally good.

## Why Haven't MoEs Been More Popular?

- Infrastructure is complex / advantages on multi-node
- Training objectives are somewhat heuristic (and sometimes unstable) [Fedus et al 2022, Zoph et al 2022]

## What MoEs Generally Look Like

- Typical: replace MLP with MoE layer
- Less common: MoE for attention heads [ModuleFormer, JetMoE]

## MoE – What Varies?

- Routing function
- Expert sizes
- Training objectives

## Routing Function – Overview

Many routing algorithms boil down to 'choose top k':

- Token chooses expert
- Expert chooses token
- Global routing via optimization

[Fedus et al 2022]

### Routing Type

Almost all MoEs do a standard 'token choice topk' routing.

**Common routing variants in detail:**

| Method | Models |
|---|---|
| Top-k | Switch Transformer (k=1), Gshard (k=2), Grok (2), Mixtral (2), Qwen (4), DBRX (4), DeepSeek (7) |
| Hashing | Common baseline [Fedus et al 2022] |

**Other routing methods:**

- RL to learn routes: Bengio 2013, not common now.
- Solve a matching problem: Linear assignment for routing, used in various papers like Clark '22.

### Top-K Routing in Detail

Most papers do the old and classic top-k routing. Gates selected by a logistic regressor.

- DeepSeek (V1-2) router (also Grok, Qwen): softmax before TopK.
- Mixtral, DBRX, DeepSeek v3: softmax after the TopK.

**Recent variations from DeepSeek and other Chinese LMs:** Smaller, larger number of experts + a few shared experts that are always on. Used in DeepSeek / Qwen, originally from DeepSpeed MoE.

More experts and shared experts all seem to generally help. Gains from fine-grained experts, but OlMoE finds none from shared experts.

### Expert Routing Setups for Recent MoEs

| Model | Routed | Active | Shared | Fine-grained ratio |
|---|---|---|---|---|
| GShard | 2048 | 2 | 0 | — |
| Switch Transformer | 64 | 1 | 0 | — |
| Mixtral | 8 | 2 | 0 | — |
| DBRX | 16 | 4 | 0 | — |
| Grok | 8 | 2 | 0 | — |
| DeepSeek v1 | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1/8 |
| DeepSeek v3 | 256 | 8 | 1 | 1/14 |
| OlMoE | 64 | 8 | 0 | 1/8 |
| MiniMax | 32 | 2 | 0 | ~1/4 |
| Llama 4 (maverick) | 128 | 1 | 1 | 1/2 |

## How Do We Train MoEs?

Major challenge: we need sparsity for training-time efficiency, but sparse gating decisions are not differentiable!

Solutions:

1. Reinforcement learning to optimize gating policies
2. Stochastic perturbations
3. Heuristic 'balancing' losses (what people use in practice)

### RL for MoEs

RL via REINFORCE does work, but not so much better that it's a clear win [Clark et al 2020]. RL is the 'right solution' but gradient variances and complexity means it's not widely used.

### Stochastic Approximations

From Shazeer et al 2017: routing decisions are stochastic with Gaussian perturbations.

1. This naturally leads to experts that are a bit more robust.
2. The softmax means that the model learns how to rank K experts.

Stochastic jitter in Fedus et al 2022 does a uniform multiplicative perturbation for the same goal. This was later removed in Zoph et al 2022.

### Heuristic Balancing Losses

Another key issue: systems efficiency requires that we use experts evenly.

From the Switch Transformer [Fedus et al 2022]: α N; the derivative with respect to p_i(x) means more frequent use = stronger downweighting.

**Example from DeepSeek (v1-2):**

- Per-expert balancing – same as the switch transformer
- Per-device balancing – the objective above, but aggregated by device

**DeepSeek v3 variation – per-expert biases:** Set up a per-expert bias (making it more likely to get tokens) and use online learning. They call this 'auxiliary loss free balancing' (but the approach is not fully aux loss free).

### What Happens When Removing Load Balancing Losses?

Training without balancing losses degrades performance.

## Training MoEs – the Systems Side

MoEs parallelize nicely: each FFN can fit in a device, enabling additional kinds of parallelism.

MoE routing allows for parallelism, but also some complexities. Modern libraries like MegaBlocks (used in many open MoEs) use smarter sparse matrix multiplications.

**Fun side issue – stochasticity of MoE models:** There was speculation that GPT-4's stochasticity was due to MoE. Why would a MoE have additional randomness? Token dropping from routing happens at a batch level – this means that other people's queries can drop your token!

## Issues with MoEs

### Stability

Solution: use Float32 just for the expert router (sometimes with an aux z-loss).

**Z-loss stability for the router:** What happens when we remove the z-loss? Training becomes unstable [Zoph 2022].

### Fine-Tuning

Sparse MoEs can overfit on smaller fine-tuning data.

- Zoph et al solution: finetune non-MoE MLPs.
- DeepSeek solution: use lots of data (1.4M SFT examples).

## Other Training Methods – Upcycling

Can we use a pre-trained LM to initialize a MoE?

**Upcycling example – MiniCPM:** Uses the MiniCPM model (topk=2, 8 experts, ~4B active params). Simple MoE, shows gains from the base model with ~520B tokens for training.

**Upcycling example – Qwen MoE:** Initialized from the Qwen 1.8B model, top-k=4, 60 experts with 4 shared. Similar architecture to DeepSeekMoE, but one of the first (confirmed) upcycling successes.

## DeepSeek MoE v1-v2-v3

### V1 (16B – 2.8B active)

- Shared (2) + Fine-grained (64/4) experts
- Standard, top-k routing
- Standard Aux-loss balancing (Expert + Device)

### V2 (236B – 21B active)

- Shared (2) + Fine-grained (160/10) experts, 6 active
- New: Top-M device routing
- Communication balancing loss – balancing both communication in and out

### V3 (671B – 37B active)

- Shared (1) + Fine-grained (258) experts, 8 active
- New: Sigmoid+Softmax topK + topM
- Aux-loss-free + seq-wise aux

## Bonus: What Else Do You Need to Make DeepSeek MoE v3?

### MLA: Multi-head Latent Attention

Basic idea: express the Q, K, V as functions of a lower-dimensional 'latent' activation.

Benefits: when KV-caching, we only need to store c_t^KV, which can be much smaller. W^UK can be merged into the Q projection (they also compress queries for memory savings during training).

Complexity: RoPE conflicts with MLA-style caching. Solution: have a few non-latent key dimensions that can be rotated.

### MTP: Multi-Token Prediction

Have small, lightweight models that predict multiple steps ahead (but they only do MTP with one token ahead). [DeepSeek v3, EAGLE]

## MoE Summary

- MoEs take advantage of sparsity – not all inputs need the full model
- Discrete routing is hard, but top-k heuristics seem to work
- Lots of empirical evidence now that MoEs work, and are cost-effective
