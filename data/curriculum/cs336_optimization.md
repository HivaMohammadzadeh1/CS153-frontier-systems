# Optimization and Training Stability

## The optimizer
LLMs are trained with **AdamW** (Adam + decoupled weight decay). Adam keeps per-parameter first and second moment estimates, which costs ~2× the model size in optimizer state — a major memory driver that ZeRO/FSDP shard. Betas (≈0.9, 0.95) and epsilon matter at scale.

## Learning-rate schedule
The standard recipe is **warmup → cosine (or linear) decay**. Warmup avoids early instability when Adam's moment estimates are noisy; decay improves final convergence. Peak LR is tuned to model size (larger models use smaller peak LRs). Batch size, LR, and warmup interact.

## Training stability
Large-model pretraining is fragile:
- **Loss spikes**: sudden divergence, often from a bad batch, attention-logit overflow, or too-high LR. Mitigations: gradient clipping, lower LR, skip/replay the batch, qk-layernorm, careful init.
- **Gradient clipping** (global norm) is standard to bound update size.
- **Init and normalization** (pre-norm, RMSNorm) keep activations well-scaled.
- Reproducibility/checkpointing so a run can resume from before a divergence.

## Scaling considerations
As you scale, hyperparameters don't transfer naively; techniques like µP (maximal update parameterization) aim to make LR transfer across widths. Resource accounting (memory for params + grads + optimizer state + activations) determines what fits.

## Interview-relevant reasoning
"Your loss spiked at step 40k — what do you check?" Strong answer: LR/warmup, gradient norm and clipping, a bad data shard, precision/overflow in attention logits; mitigate with clipping, LR reduction, and resume-from-checkpoint.
