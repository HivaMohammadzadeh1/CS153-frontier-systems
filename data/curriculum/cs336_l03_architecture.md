# CS336 Lecture 3 — Everything You Didn't Want to Know About LM Architecture and Training

**Source:** Stanford CS336 Spring 2025, Tatsu H.

## Outline and Goals

- Quick recap of the 'standard' transformer (what you implement)
- What do most of the large LMs have in common?
- What are common variations to the architecture / training process?

Today's theme: the best way to learn is hands-on experience; the second best way is to try to learn from others' experience.

## Starting Point: The 'Original' Transformer

Review: choices in the standard transformer:

- Position embedding: sines and cosines
- FFN: ReLU
- Norm type: post-norm, LayerNorm

**What you implemented – simple, modern variant:**

Differences from the original:

- LayerNorm is in front of the block (pre-norm)
- Rotary position embeddings (RoPE)
- FF layers use SwiGLU, not ReLU
- Linear layers (and layernorm) have no bias (constant) terms

## How Should We Think About Architectures?

Lots of architecture variation. Just in the last year since last 336: over 19 new dense model releases, many of them with minor architecture tweaks.

We will talk through many major architecture and hyperparameter variants:

- What do all these models have in common?
- What parts vary?
- What can we learn from this?

## What Are We Going to Cover?

Common architecture variations:

- Activations, FFN
- Attention variants
- Position embeddings

Hyperparameters that (do or don't) matter:

- What is ff_dim? Do multi_head dims always sum to model_dim?
- How many vocab elements?

Stability tricks.

## Architecture Variations

### Pre-vs-Post Norm

The one thing everyone agrees on (in 2024): set up LayerNorm so that it doesn't affect the main residual signal path.

Almost all modern LMs use pre-norm (but BERT was post-norm). One somewhat funny exception: OPT350M.

**Pre-vs-post norm, explanations:**

- Gradient attenuation [Xiong 2020]
- Gradient spikes [Salazar and Ngyuen]
- Original stated advantage: removing warmup
- Today: stability and larger LRs for large networks

New things – 'double' norm: if putting LayerNorms in residual streams is bad, why not post-norm outside the stream? Recent models: Grok, Gemma 2, OlMo 2 (only does non-residual post norm).

### LayerNorm vs RMSNorm

Original transformer: LayerNorm – normalizes the mean and variance across d_model. Notable models: GPT3/2/1, OPT, GPT-J, BLOOM.

Many modern LMs: RMSNorm – does not subtract mean or add a bias term. Notable models: LLaMA-family, PaLM, Chinchilla, T5.

Formula: y = (x / sqrt(x^2 + ε)) * γ

**Why RMSNorm?**

Modern explanation: it's faster (and just as good).

- Fewer operations (no mean calculation)
- Fewer parameters (no bias term to store)

Important lesson: FLOPS are not runtime! RMSNorm can still matter due to the importance of data movement.

**More generally: dropping bias terms.** Most modern transformers don't have bias terms. Reasons: memory (similar to RMSNorm) and optimization stability.

**LayerNorm recap:**

- Basically everyone does pre-norm. Intuition: keep the good parts of residual connections. Observations: nicer gradient propagation, fewer spikes.
- Most people do RMSNorm. In practice, works as well as LayerNorm. But fewer parameters to move around, which saves wallclock time. People more generally drop bias terms since compute/param tradeoffs are not great.

## Activations

A whole zoo of activations: ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU.

**Common activations:**

- ReLU: FF(x) = max(0, xW1)W2. Notable models: Original transformer, T5, Gopher, Chinchilla, OPT.
- GeLU: FF(x) = GELU(xW1)W2. GELU(x) := x * Φ(x). Notable models: GPT1/2/3, GPTJ, GPT-Neox, BLOOM.
- SwiGLU / GeGLU: Notable models: Llama, PaLM, T5 v1.1, most models post 2023.

**Gated activations (*GLU):** GLUs modify the 'first part' of a FF layer. Instead of linear + ReLU, augment with an (entrywise) linear term:

max(0, xW1) → max(0, xW1) ⊗ (xV)

This gives ReGLU – note extra parameter V. FFReGLU(x) = (max(0, xW1) ⊗ xV) W2.

**Gated variants:**

- GeGLU: T5 v1.1, mT5, LaMDA, Phi3, Gemma 2, Gemma 3
- SwiGLU (swish is x * sigmoid(x)): LLaMa 1/2/3, PaLM, Mistral, OlMo, most models post 2023

Note: Gated models use smaller dimensions for d_ff by 2/3.

Do gated linear units work? Yes, fairly consistently so [Shazeer 2020, Narang et al 2020].

**Gating summary:** Many variations (ReLU, GeLU, *GLU). *GLU isn't necessary for a good model (see GPT3), but it's probably helpful. Evidence points towards somewhat consistent gains from SwiGLU/GeGLU.

## Serial vs Parallel Layers

Normal transformer blocks are serial – compute attention, then the MLP. Could we parallelize the transformer block?

A few models (GPTJ, PaLM, GPT-NeoX) do parallel layers. If implemented right, LayerNorm can be shared, and matrix multiplies can be fused. Recent models: Cohere Command A, Falcon 2 11B, Command R+.

**Summary architectures:**

- Pre-vs-post norm: Everyone does pre-norm (except OPT350M), likely with good reason.
- Layer vs RMSNorm: RMSNorm has clear compute wins, sometimes even performance.
- Gating: GLUs seem generally better, though differences are small.
- Serial vs parallel layers: No extremely serious ablations, but has a compute win.

## Position Embeddings

Many variations:

- **Sine embeddings:** Add sines and cosines. Notable models: Original transformer.
- **Absolute embeddings:** Add position vector. Notable models: GPT1/2/3, OPT.
- **Relative embeddings:** Add vector to attention computation. Notable models: T5, Gopher, Chinchilla.
- **RoPE embeddings:** Notable models: GPTJ, PaLM, LLaMA, most 2024+ models.

### RoPE: Rotary Position Embeddings

High level thought: a relative position embedding should be some f(x, i) s.t. ⟨f(x,i), f(y,j)⟩ = g(x, y, i-j). That is, the attention function only gets to depend on the relative position (i-j).

Existing embeddings don't fulfill this goal:

- Sine: Has various cross-terms that are not relative.
- Absolute: obviously not relative.
- Relative embeddings: not an inner product.

**How can we solve this problem?** We want embeddings invariant to absolute position. Inner products are invariant to arbitrary rotation. Just pair up the coordinates and rotate them in 2d (motivation: complex numbers).

**Implementation:** Multiply with sines and cosines. Difference from sine embeddings: not additive, no cross terms. Note: embedding at each attention operation to enforce position invariance.

## Hyperparameters

### Feedforward–Model Dimension Ratio

d_ff = 4 * d_model is almost always true.

**Exception #1 – GLU variants:** Remember that GLU variants scale down by 2/3. This means most GLU variants have d_ff = (8/3) * d_model.

| Model | d_ff/d_model |
|---|---|
| PaLM | 4 |
| Mistral 7B | 3.5 |
| LLaMA-2 70B | 3.5 |
| LLaMA 70B | 2.68 |
| Qwen 14B | 2.67 |
| DeepSeek 67B | 2.68 |

**Exception #2 – T5:** For the 11B model, T5 sets d_ff = 65,536 and d_model = 1,024 for a 64× multiplier. Other recent exceptions: Gemma 2 (8×), SmolLM/Gemma 3 (4×, GLU).

Why this range? Empirically, there's a basin between 1–10 where this hyperparameter is near-optimal [Kaplan+ 2020].

### Head-dim × num-heads to model-dim ratio

Most models have ratios around 1 – notable exceptions by some Google models.

Evidence for 1-1 ratio? Papers written against low-rank bottlenecks [Bhojanapalli et al 2020], but we don't seem to see significant bottlenecks in practice.

### Aspect Ratios

Should my model be deep or wide? Most models are surprisingly consistent:

| Model | d_model/n_layer |
|---|---|
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM (540B) | 156 |
| GPT3/OPT/Mistral/Qwen | 128 |
| LLaMA/LLaMA2/Chinchilla | 102 |
| T5 (11B) | 43 |
| GPT2 | 33 |

Extremely deep models are harder to parallelize and have higher latency [Tay et al 2021].

### Vocabulary Sizes

- Monolingual models: 30–50k vocab
- Multilingual / production systems: 100–250k

Notable examples: GPT (40,257), GPT2/3 (50,257), LLaMA (32,000), GPT4 (100,276), PaLM (256,000), Command A (255,000).

### Dropout and Regularization

Do we need regularization during pretraining?

Arguments against: lots of data (trillions of tokens), more than parameters; SGD only does a single pass on a corpus (hard to memorize).

In practice: many older models used dropout during pretraining. Newer models (except Qwen) rely only on weight decay.

**Why weight decay LLMs?** [Andriushchenko et al 2023] finds it's not to control overfitting; weight decay interacts with learning rates (cosine schedule).

**Hyperparameters summary:**

- Feedforward: Factor-of-4 rule of thumb (8/3 for GLUs) is standard (with some evidence).
- Head dim: Head dim × Num head = D model is standard – but low to no validation.
- Aspect ratio: Wide range of 'good' values (100–200). Systems concerns dictate the value.
- Regularization: You still 'regularize' LMs but its effects are primarily on optimization dynamics.

## Stability Tricks

Recently, lots of attention on stable training. Don't train models that look like the blue curve!

### Output Softmax Stability – The 'Z-Loss'

Softmaxes can be ill-behaved due to exponentials / division by zero. PaLM pioneered a 'z loss' trick for stability. Other examples: Baichuan 2 (2023), DCLM (2024), OLMo 2 (2025).

### Attention Softmax Stability – The 'QK Norm'

The query and keys are Layer (RMS) normed before going into the softmax operation. Other examples: DCLM, OLMo2, Gemma 2. Originally from vision and multimodal models [Dehghani 2023, Idefics, Chameleon].

### Logit Soft-Capping

Soft-capping the logits to some maximum value via Tanh. Prevents logits from blowing up, but may have perf issues.

## Attention Heads

Most models don't touch the attention heads much at all.

**GQA / MQA:** Saving inference costs by reducing the number of heads.

**Sparse or sliding window attention (GPT4/Mistral):** Restricting the attention pattern to reduce compute cost.

### GQA/MQA – Reducing Attention Head Cost

During text generation, we need to incrementally update attention via the 'KV cache'. Arithmetic intensity is not good – need large batches + short seq length or big model dimensions.

**MQA:** Have multiple queries, but just one dimension for keys and values. Much fewer items to move in and out of memory.

**GQA:** Don't go all the way to one KV dimension – have fewer dims. Simple knob to control expressiveness (key-query ratio) and inference efficiency.

Does MQA hurt? Small PPL hit with MQA [Shazeer 2019]; low/no hit with GQA [Ainslie 2023].

### Sparse / Sliding Window Attention

Attending to the entire context can be expensive (quadratic). Build sparse / structured attention that trades off expressiveness vs runtime [Child et al 2019, Mistral].

Current standard trick: interleave 'full' and local-range attention. From Cohere Command A: every 4th layer is full attention. Long-range info via NoPE, short-range info via RoPE + SWA. Other models: LLaMA 4, Gemma.

## Recap and Conclusion

Many aspects (arch, hparams) of transformers are in common across the big LMs.

Major differences: position embeddings, activations, tokenization.
