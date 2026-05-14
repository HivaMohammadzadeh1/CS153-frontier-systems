# CS336 Lecture 11 — Scaling: Case Study and Details

**Source:** Stanford CS336 Spring 2025.

## Motivation Today

What are the best practices for scaling and hparam tuning LMs?

- Does chinchilla's approach to scaling actually work?
- Can we save compute when training and fitting these things?
- Should we be picking particular architectures / parametrizations to scale nicely?

## Scaling in Practice

The newest model we talked about with scaling details: 2022. Many more models with scaling details in 2023, 2024, 2025.

## Maximum Update Parametrization (muP) – In Depth

Recall: the maximum update parametrization (muP) makes appealing claims. Scale-invariant hyperparameter tuning would be very nice.

## Recent Models with Detailed, Public Scaling Recipes

1. CerebrasGPT
2. MiniCPM
3. DeepSeek

## CerebrasGPT

CerebrasGPT: 0.1 to 13B models trained with the Chinchilla recipe.

**Core finding:** using muP parametrization makes scaling more stable.

**Hyperparam scaling strategy:** CerebrasGPT authors find more predictable scaling from muP parametrization.

**muP parametrization:** Appendix contains a very clear set of differences in parametrizing the model for scaling.

**Setting the empirical values:** muP is combined with aggressive scaling for hyperparameter optimization. Generally stable hyperparameters.

## MiniCPM

MiniCPM (2024): new small, high-performance LM from Tsinghua group. Careful, extensive scaling computations + muP to stabilize and simplify scaling.

High performance 1–2.5B parameter models. These models beat most 2Bs and match many modern 7B models.

**Technique 1: muP to stabilize scaling.** Scale_emb = 12, scale_depth = 1.4, init_std = 0.1, lr = 0.01. Cf. CerebrasGPT: Scale_emb = 10, lr=6e-3, init_base = 0.08.

**Scaling recipe / strategy:** Use muP for initialization, fix the aspect ratio, scale up the overall model size. Note that the gap between the largest model here and the actual model they train is ~5x. Optimal batch, LR, token-to-size ratios are directly fitted via scaling analysis.

### Optimal Batch

Three model sizes (9m, 30m, 170m) as a function of data size (y), batch (x) and loss (col). Vertical columns of points represent a single training curve. Red line attempts to identify minimum loss points for each y-value – this is the 'optimal batch size' for a model size / dataset size combination.

We can then follow the Kaplan 2020 analysis and plot optimal batch size vs final loss. Fairly clean trend: polynomially increase the batch size as loss decreases.

### Optimal LR

According to muP, optimal learning rate should be (roughly) stable.

### What Remains – Model Size vs Data Tradeoffs

From chinchilla: to fit a scaling law, we need to train from scratch, not just early stop. This turns the cost of fitting a scaling law from n to n^2. Can we avoid this?

**(Partial) solution in MiniCPM – WSD Learning Rate:** Instead of cosine, split learning rate into warmup, stable, and decay phases. For chinchilla-style analysis, can restart the run at the end of the stable phase.

**WSD learning rates work well in MiniCPM:** Slower during the stable phase, rapid loss decay during decay phase. Decay ~10%.

**Side note – other ways of estimating Chinchilla curves:** Gadre et al propose other, curve-fitting-based ways of doing similar things. Core idea: the 'penalty' from overtraining remains stable.

### Chinchilla-Type Analysis

Equipped with the WSD learning rate, we can now try to find the optimal data-to-model size ratio.

MiniCPM authors choose method 1 (lower envelope) and method 3 (joint fit).

**Chinchilla method 1:** Fairly clear (though maybe not linear?) trends. Different colors indicate different models. Their runs suggest relatively low diminishing returns due to data.

**Chinchilla method 3:** Their primary scaling approach is the joint fit – they find very high data-model ratios.

**Tiny models with lots of data.** The overall data-to-model ratio is very high (192), though they argue LLaMA architectures should have a higher ratio. Note that recent models like LLaMA 3 has significantly higher data-to-model ratios, suggesting that with more careful optimization, we might be able to go far beyond the 20×model_size rule of thumb.

**Scaling curve fits are (generally) good.** Overall fits and predictions of models across a large range of sizes are fairly good. X-axis: number of tokens in billions.

## DeepSeek

DeepSeek (2024): another LM with careful scaling analysis. 7B and 67B param models – generally high performance compared to other open LMs.

**Performance:** Roughly comparable to LLaMA 2 models of equivalent size.

**Scaling strategy – batch + LR:** Don't use any muP; directly estimate optimal batch / LR.

**Scaling analysis of learning rates:** Small scale runs + collect 'near optimal' (within 0.25% of min) models. Learning rate fit looks a bit questionable.

**For chinchilla analysis: WSD-style learning rate.** DeepSeek uses WSD-style learning rate – fast warmup + two decay steps of 10% each. Generally seems to match performance of cosine learning rates.

**Data-size tradeoff analysis: Chinchilla method 2.** Straightforward isoflop-style analysis for selecting the model size tradeoffs.

**Scaling predicts final model loss.** The fitted scaling models generally accurately predict the final model losses.

## LLaMA 3 (2024) Scaling Laws

- Isoflops-style scaling (39-1 ratio)
- Compute-to-downstream scaling

## Hunyuan-1 (2024) Large Scaling Laws

Yet more isoflops-style scaling (but this time for MoE parameter sizes). Optimal ratio: 96-1 (data to active param).

## MiniMax-01 (2025)

Architecture scaling laws + Chinchilla method 1.

## Recent Scaling Law Recipes Summary

**CerebrasGPT:**

- Use muP to make hyperparams invariant to scale.
- Directly use the chinchilla scaling formula.

**DeepSeek recipe:**

- Assume most transformer hypers are invariant to scale.
- Do a scaling analysis on batch / LR to figure out optimal scaling.
- IsoFLOP analysis to figure out model sizing.
- Use a piecewise-linear schedule to make chinchilla scaling cheap.

**MiniCPM recipe:**

- Use muP to make transformer + LR invariant to scale.
- Use a piecewise linear schedule to get sample for Chinchilla method 3 (curve fitting).

**Recent (late 2024+) but less detailed:**

- LLaMA 3 / Hunyuan: Just isoflops (no other scaling details).
- MiniMax: Architecture choice / decision scaling.

## Validating and Understanding muP

"Scale invariant" hyperparameter tuning would be very useful. CerebrasGPT and MiniCPM also use muP – is it actually useful?

### What Is muP, Anyway?

muP is based off the following assertions. As a function of the width of the network n_l:

- **A1:** The activations at initialization should remain Θ(1).
- **A2:** After one gradient step, the change in activation should be Θ(1).

Note: if individual activations are Θ(1), then the norm should be Θ(√n_l).

### Deriving muP (Condition A1)

Suppose we have a simple, deep linear network (h_l = W_l h_{l-1}) and we initialize W_l ~ N(0, σI_{n_l × n_{l-1}}).

By basic matrix concentration, W_l_* → σ(√n_{l-1} + √n_l).

Picking σ = 1/min(n_{l-1}, (n_l/n_{l-1})). Inductive case: ‖h_l‖_2 = √n_l + o(√n_l).

### Deriving muP (Condition A2)

For SGD, on a linear layer, the update looks like a rank-one loss-activation outer product: ΔW_l = -η_l ∇_{h_l} ℓ h_{l-1}^⊤.

Assuming that the leading order terms don't cancel:

- W_l Δh_{l-1} = Θ(√n_l) from induction + condition A1.
- ΔW_l h_{l-1} = ‖ΔW_l‖_* √n_{l-1}, thus ‖ΔW_l‖_* = Θ(√n_l/n_{l-1}).

The key is to pick LR such that ‖ΔW_l‖_* √n_{l-1} = Θ(√n_l). For Adam: η_l ∝ n_{l-1}.

### muP Mini Recap

What is (baby) muP about? Controlling activations (and changes) via W and ΔW.

**muP:**

- Initialization: Set to Θ(1/min(n_{l-1}, (n_l/n_{l-1})^(1/2)))
- Learning rates: Set to n_l/n_{l-1} (for Adam: n_{l-1})

**Standard parametrizations:**

- Initialization: Set to 1/√n_{l-1}
- Learning rates: Set to Θ(1)

Differences: LR changes for Adam, also init diffs when fanout n_l < fanin.

### Implementation in CerebrasGPT

Architecture components with different muP scaling:

- Embedding
- Attention params
- Input/output MLP matrix multiply
- Softmax linear

Important limitation of the work: only width scaling.

### Replicating muP

Q1: Does muP work as claimed? When we scale widths, is optimal LR constant?

### What Is muP Robust To?

Modern LMs have many components that deviate from muP's theory:

- Activations – SwiGLU and squared ReLU
- Batch sizes – large / small
- Initialization variations – zero attention, etc.
- RMS norm gains
- Exotic optimizers (Lion)
- Regularizers

**Nonlinearities:** SwiGLU, Squared ReLU have the same optimal LR (and both provide minor gains).

**Batch size:** Larger and smaller batches. The original derivation doesn't handle batch size considerations.

**Initialization:** There are new initializations sometimes used:

- SP Unembedding: pre-softmax linear layer. 1/M (SP) vs 1/M^2 (muP)
- Zero Query: Set the query matrix to zero (so that all items get uniform attention)

### What Is muP NOT Robust To?

**RMSNorm gain:** In our arch, RMSNorm has learnable gains. This turns out to break muP. But these gains can be removed with little loss of performance.

**Exotic optimizers:** Other, exotic optimizers based on just gradient signs.

**(Strong) weight decay:** What about strong (0.1) weight decay? This is maybe the only significant muP failure.

### Is muP Useful?

At least to some extent: SP is quite a bit more unstable. Current evidence suggests that muP parametrization / initialization may be easier to tune. Overall, muP generally seems useful.

## Recap: Scaling in the Wild

**Challenges in scaling 'in practice':**

1. Setting model arch hyperparameters (width, etc.)
2. Setting optimizer hyperparameters (LR, batch)
3. Compute needed to fit the big chinchilla sweep

**Some solutions:**

1. Assume stability (or use muP)
2. Search for optimal LR / batch in small scale, either keep fixed or predict scaling
3. Use alternative learning schedules (WSD-like)
