# Mixed Precision and Numerics

## Why mixed precision
Training and serving in full fp32 wastes memory and bandwidth. Mixed precision keeps most math in a low-precision format (fp16, bf16, or fp8) while preserving accuracy where it matters, cutting memory and roughly doubling throughput on tensor-core hardware.

## bf16 vs fp16
- **fp16** has a small exponent range and overflows/underflows easily, so it needs **loss scaling** (scale the loss up before backward, unscale before the optimizer step) to keep gradients in range.
- **bf16** has the same exponent range as fp32 (just fewer mantissa bits). It rarely overflows, so it usually needs no loss scaling — which is why bf16 is the default for modern LLM training on hardware that supports it.

## fp8 and quantized inference
fp8 (e4m3 / e5m2) pushes further for both training and inference, trading mantissa bits for speed/memory. It demands careful scaling (per-tensor or per-block) to control error. For inference, int8/int4 weight quantization (GPTQ/AWQ) cuts memory bandwidth — the decode bottleneck — but can affect quality, so it must be evaluated, not assumed free.

## Numerical stability
- Keep a master fp32 copy of weights for the optimizer; accumulate in higher precision.
- Softmax, layernorm, and reductions are sensitive — often kept in higher precision.
- Watch for **loss spikes / NaNs**: often a precision/scaling issue, a bad LR, or attention-logit overflow.

## Interview-relevant reasoning
"Why bf16 over fp16?" — dynamic range, no loss scaling. "Does quantization only affect quality?" — no: it changes memory bandwidth, cost, and kernel behavior, and the quality impact must be measured.
