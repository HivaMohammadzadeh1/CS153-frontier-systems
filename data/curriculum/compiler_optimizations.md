# Compiler Optimizations for ML: XLA, TorchInductor, ONNX, and MLIR

**Area C — Inference Infrastructure | Learning Memory OS Curriculum**

---

## 1. The Gap Between Model Code and Hardware

A PyTorch model written in Python is far from optimal hardware execution. The Python interpreter, eager execution, and PyTorch's dynamic dispatch overhead impose costs that can account for 30-50% of total inference time for small models or batch sizes. ML compilers close this gap by transforming a high-level model description into optimized low-level code that directly exploits hardware capabilities: tensor core utilization, memory layout, operator fusion, and parallelism.

Understanding ML compilers requires distinguishing three levels of representation:
1. **Framework-level IR**: The computation graph expressed in PyTorch or JAX.
2. **Compiler IR**: An intermediate representation (IR) that the compiler transforms — HLO for XLA, ATen/Torch IR for TorchInductor, MLIR dialects for LLVM-based compilers.
3. **Hardware-level code**: PTX for NVIDIA GPUs, HBM access patterns, CUDA kernels.

---

## 2. Graph IRs

A **computation graph** (or graph IR) represents the model as a directed acyclic graph where nodes are operations (matmul, add, softmax) and edges are tensors. Graph representations enable compiler analyses that are impossible in eager mode:

- **Dead code elimination**: Remove operations whose outputs are never used.
- **Constant folding**: Evaluate operations on constant inputs at compile time.
- **Common subexpression elimination**: Compute a sub-expression once and reuse.
- **Operator fusion**: Merge consecutive operations into a single kernel (critical for GPU performance).

PyTorch exports computation graphs via `torch.export` (FX graph) or `torch.compile` with `backend="inductor"`. JAX represents computations as functional Python that XLA traces via `jax.jit`. TensorFlow uses a dataflow graph captured by `tf.function`.

---

## 3. Operator Fusion

**Operator fusion** is the single most impactful optimization in ML compilation. The fundamental issue is that individual GPU kernels are often memory-bandwidth-bound, not compute-bound: a vector add reads two tensors from HBM, adds element-wise, and writes the result back to HBM. The ratio of compute (1 FLOP per element) to memory access (3 element reads/writes) is far below GPU arithmetic intensity.

By fusing multiple operations into a single kernel, the compiler keeps intermediate results in SRAM (shared memory) or registers instead of writing them to and reading from HBM. The intermediate tensors never materialize in global memory.

**Example: Fused attention kernel (FlashAttention)**

Naive attention computes:
1. S = Q @ K^T → writes S (n × n) to HBM
2. P = softmax(S) → reads S from HBM, writes P to HBM
3. O = P @ V → reads P from HBM, writes O to HBM

Total HBM traffic: proportional to n² for the attention matrix. FlashAttention fuses all three steps into a single CUDA kernel: S is computed tile by tile, the softmax is computed with an online algorithm, and P is multiplied with V before writing O. The attention matrix never materializes in HBM. For long sequences, this reduces HBM traffic by 5-20x and is the primary reason FlashAttention is 2-3x faster than naive attention.

---

## 4. XLA and HLO

**XLA (Accelerated Linear Algebra)** is Google's open-source ML compiler, originally developed for TPU and now supporting GPU and CPU targets. XLA is the compiler backend for JAX and TensorFlow's `tf.function` path.

### 4.1 HLO: High-Level Optimizer IR

The primary IR in XLA is **HLO (High Level Optimizer)**, a functional representation of tensor computations. HLO operations are typed, immutable, and have defined semantics. An HLO program is a set of computations, each of which is a sequence of HLO instructions.

XLA applies a sequence of HLO passes:
- **Op fusion**: Groups elementwise ops, broadcasts, and transposes into fusion clusters that compile to a single kernel.
- **Layout optimization**: Assigns memory layouts (row-major, column-major, custom tiled) to tensors to minimize transpose overhead and maximize memory access locality.
- **Buffer assignment**: Assigns HBM buffers to HLO tensors, minimizing peak memory usage via liveness analysis.
- **Convolution rewriting**: Converts convolutions to cuDNN-optimized implementations.

### 4.2 XLA Compilation Latency

XLA compilation is expensive — a large transformer model can take 5-30 minutes to compile for a specific batch size and sequence length configuration. This makes XLA impractical for dynamic shapes (variable batch sizes, variable sequence lengths). TPU deployments often use static shapes everywhere (padding to fixed dimensions) precisely because XLA requires static shape information for its optimization passes.

The tradeoff: XLA compilation cost is amortized over many inference calls. For a model serving 1 million requests per day, a 10-minute compilation cost is < 0.001% of total inference time. But the cold-start latency (the first request after deployment triggers compilation) must be hidden — production systems typically run an offline compilation step and cache the compiled artifact (`XLA_FLAGS=--xla_dump_to=...`).

---

## 5. TorchInductor and Triton Codegen

**TorchInductor** is the default compiler backend for `torch.compile()` in PyTorch 2.0+. It occupies the same design space as XLA but targets the PyTorch ecosystem and supports dynamic shapes better.

### 5.1 `torch.compile()` Pipeline

The `torch.compile()` path has three stages:

1. **TorchDynamo** (graph capture): Intercepts Python bytecode execution to trace the computation graph without requiring explicit `@torch.jit.script` annotations. Dynamo produces an FX graph (a graph of ATen operations).

2. **AOT Autograd**: Traces through the autograd graph to produce a joint forward+backward FX graph for training, or a forward-only graph for inference.

3. **TorchInductor** (codegen): Lowers the FX graph to optimized CUDA code. For GPU targets, Inductor uses **Triton** as the codegen backend — it emits Triton programs that the Triton compiler lowers to PTX.

### 5.2 Triton Codegen

Triton is a Python-embedded language and compiler for GPU kernels. A Triton kernel is written as a Python function with decorators, and the Triton compiler produces optimized PTX. The key advantage over raw CUDA: Triton handles shared memory management, warp-level parallelism, and vectorization automatically based on tile size parameters.

TorchInductor generates Triton kernels for fused elementwise operations, reduction operations, and custom attention patterns. For matmul and large convolutions, Inductor falls back to cuBLAS/cuDNN, which have hand-tuned kernels that Triton cannot match.

**Typical speedups from `torch.compile()`**: 1.5-2.5x for transformer inference on GPU, depending on model architecture, batch size, and sequence length. Small models and small batches see larger relative speedups (Python overhead removal is the dominant effect). Large models at large batches see smaller relative speedups (they were already compute-bound).

---

## 6. ONNX Runtime

**ONNX (Open Neural Network Exchange)** is an open format for representing ML models as a computation graph with a defined operator set. ONNX decouples model representation from training framework — a PyTorch model can be exported to ONNX and deployed with ONNX Runtime (ORT) without requiring PyTorch at inference time.

**ONNX Runtime** is a cross-platform inference engine with execution providers for different hardware backends: CUDA EP (GPU), TensorRT EP (NVIDIA TensorRT optimizations), OpenVINO EP (Intel), DirectML EP (Windows GPU), and CPU EP. ORT applies graph optimizations independently of the hardware backend: operator fusion, constant folding, and layout optimization.

### 6.1 TensorRT Execution Provider

The TensorRT EP (TRTEP) within ORT applies NVIDIA TensorRT optimizations: kernel auto-tuning, INT8/FP16 quantization, and layer fusion. For latency-sensitive inference on NVIDIA GPUs, TRTEP can achieve 2-5x speedups over PyTorch eager mode. The cost is compilation: TensorRT performs exhaustive kernel benchmarking at first run (serialized to a `.trt` plan file for reuse), which can take tens of minutes for large models.

### 6.2 ONNX Limitations

ONNX has a versioned operator set — new operators in PyTorch 2.x may not yet have ONNX equivalents, requiring custom op registrations or workarounds. Dynamic control flow (if/else based on tensor values, variable-length loops) is poorly supported in the standard ONNX graph format. Operators like FlashAttention are not in the ONNX op set and must be decomposed into lower-level ops, sacrificing the fusion benefit.

---

## 7. MLIR: Multi-Level Intermediate Representation

**MLIR** is an extensible compiler infrastructure developed by Google and now hosted in the LLVM project. Unlike LLVM IR (which is hardware-level) or HLO (which is ML-specific), MLIR is *meta-infrastructure*: it provides a framework for defining custom IRs ("dialects"), building transformation passes, and lowering from high-level dialects to low-level code.

### 7.1 Dialects

MLIR's key abstraction is the **dialect** — a domain-specific extension of the MLIR IR. Relevant dialects for ML:

- **`linalg` dialect**: Expresses tensor operations with explicit loop structure, enabling polyhedral optimization.
- **`tosa` dialect** (Tensor Operator Set Architecture): A standard ML operator set, used as an intermediate by TFLite, ONNX-MLIR, and other frontends.
- **`mhlo` / `stablehlo` dialect**: The MLIR version of HLO, used by JAX/XLA's MLIR-based compilation path.
- **`gpu` dialect**: Abstractions for GPU parallelism (grid, block, thread), used by the GPU codegen passes.

### 7.2 Lowering Stack

A typical ML compiler using MLIR lowers from high-level dialect → linalg → affine → llvm → PTX/AMDGPU. Each lowering step applies dialect-specific optimizations. This composability allows reuse: a new ML accelerator needs only to provide lowering from `linalg` to its own backend, leveraging all the higher-level passes for free.

### 7.3 TVM and MLIR

Apache TVM uses MLIR as an intermediate representation in its Relax (the next-generation IR) and Unity compilation stack. TVM's metalanguage approach — expressing hardware optimizations as composable schedules — was an early precursor to MLIR's design philosophy.

---

## 8. Compilation Latency vs Runtime Speedup Tradeoff

Every ML compiler faces the fundamental tradeoff: spend more time compiling to get faster inference, or spend less time compiling to start serving faster.

| Approach | Compilation Time | Runtime Speedup | Dynamic Shape Support |
|---------|-----------------|-----------------|----------------------|
| PyTorch eager | 0 s | 1.0× (baseline) | Full |
| `torch.compile()` (Inductor) | 30s - 5 min | 1.5-2.5× | Partial (dynamic=True) |
| XLA / `jit_compile` | 5 - 30 min | 2-4× | Poor (requires static shapes) |
| TensorRT | 10 - 60 min | 2-5× | Poor (profile-guided dynamic shapes) |
| Custom Triton kernels | Days (engineering) | 3-8× | Varies |

The right choice depends on the serving regime:
- **Low-volume, high-variability requests** (diverse batch sizes, sequence lengths): `torch.compile()` with `dynamic=True` — fast compilation, moderate speedup.
- **High-volume, fixed-shape requests** (e.g., real-time chat with fixed context window): TensorRT or XLA — pay the compilation cost once, reap per-request speedup.
- **Research / interactive**: PyTorch eager — zero compile latency.

---

## Misconception: `torch.compile()` always makes models faster

`torch.compile()` can *slow down* models in some configurations. When the model has many Python-level dynamic control flow branches (if/else based on tensor shapes), Dynamo must recompile the graph for each new shape seen — this is called "recompilation churn." In the worst case, `torch.compile()` compiles a new kernel for every request with a unique shape, spending more time compiling than executing. The fix: use `dynamic=True` for shape-dynamic models, or pad inputs to fixed shapes to avoid recompilation.

## Misconception: ONNX guarantees model parity with the original framework

ONNX export can introduce numerical differences due to operator decompositions, different floating-point operation ordering (which affects rounding), and different precision semantics in the ONNX execution provider. For models that rely on exact numerical consistency (e.g., beam search decoding with tied logits), always validate output parity between the original framework and the ONNX-runtime output on a set of reference inputs before deploying.

## Misconception: XLA and TorchInductor are competing end-to-end solutions

XLA and TorchInductor are designed for different ecosystems and overlap only partially. XLA is tightly integrated with JAX (and TPUs), provides the best static-shape performance, and is the right choice for JAX-based models. TorchInductor is tightly integrated with the PyTorch ecosystem, supports dynamic shapes, and is the right choice for PyTorch models. Using XLA with PyTorch (via `torch_xla`) is possible but adds complexity; using TorchInductor with JAX is not supported. Choose based on your training framework.

## Misconception: Operator fusion is only useful for small operations

Fusion is most impactful for small elementwise operations (whose GPU kernels are severely bandwidth-bound), but it also matters for large operations composed of smaller ones. LayerNorm (which involves reduce, subtract, divide, and multiply) benefits enormously from fusion: an unfused implementation performs 5 separate HBM round-trips; a fused kernel does one. For a 70B model with 80 layers, eliminating 80 × 4 = 320 unnecessary HBM round-trips for LayerNorm alone saves tens of milliseconds per forward pass.

## Misconception: MLIR is a single compiler for ML

MLIR is *infrastructure* for building compilers, not a compiler itself. It provides the dialect framework, transformation passes, and lowering infrastructure but no end-to-end ML compilation pipeline. The "MLIR compiler" does not exist — what exists are MLIR-based compilers: StableHLO (JAX/TF), ONNX-MLIR, IREE, TVM Relax. Understanding MLIR means understanding dialects, lowering chains, and the transformation pass infrastructure — not a single tool.

---

## 9. Practical Example: Benchmarking Inference Backends for Llama-3-8B

Setup: Llama-3-8B, batch size 8, input length 512, output length 128, H100 GPU.

| Backend | First-token latency (p50) | Throughput (tok/s) | Compile time |
|---------|--------------------------|-------------------|--------------|
| PyTorch eager | 45 ms | 8,200 | 0 s |
| `torch.compile()` | 29 ms | 12,500 | 3 min |
| TensorRT-LLM | 18 ms | 19,800 | 45 min |
| vLLM (PagedAttn + Triton) | 20 ms | 18,100 | 5 min |

For a production deployment handling 50 requests/second, TensorRT-LLM's 45-minute compile cost is justified — it saves ~12 ms per request, reducing GPU compute cost by 28%. For a development endpoint handling 5 requests/hour, eager mode or `torch.compile()` is the right choice.

---

## 10. Exercise

**Exercise**: Profile a Llama-3-7B forward pass (single layer) with PyTorch eager mode using `torch.profiler`. Identify the top 3 operations by CUDA kernel time. Then apply `torch.compile()` with `backend="inductor"` and profile again. Report: which operations were fused, what the speedup was, and whether any operations regressed (became slower). As a stretch goal, write a custom Triton kernel for the fused QKV projection (combining Q, K, V matmuls into a single kernel that reads the weight matrix once) and compare its performance against the Inductor-generated kernel.

---

## References

- XLA documentation: https://openxla.org/xla
- TorchInductor design: PyTorch 2.0 paper, Ansel et al. (2024)
- Triton: Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (MAPL 2019)
- MLIR: Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation" (CGO 2021)
- FlashAttention: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/
- Apache TVM: Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (OSDI 2018)
