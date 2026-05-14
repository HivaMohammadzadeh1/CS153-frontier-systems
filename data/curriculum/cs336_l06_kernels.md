Last lecture: high-level overview of GPUs and performance

This lecture: benchmarking/profiling + write kernels

You should run this lecture on a GPU to get the full experience.

## Summary

Gap between the programming model (PyTorch, Triton, PTX) and hardware => performance mysteries

Benchmarking for understanding scaling

Profiling for understanding internals of PyTorch functions (bottoms out with kernels)

Looking at PTX assembly to understand internals of CUDA kernels

5 ways to write a function: manual, PyTorch, compiled, CUDA, Triton

GeLU (element-wise), softmax (row-wise), matmul (complex aggregation)

Key principle: organize computation to minimize reads/writes

Key ideas: kernel fusion (warehouse/factory analogy), tiling (shared memory)

Automatic compilers (Triton, torch.compile) will get better over time

Assignment 1 leaderboard 

Assignment 2 is out 

## Hardware

Compute: streaming multiprocessors (SMs) [A100: 108]

Memory:

- DRAM [A100: 80GB] - big, slow

- L2 cache [A100: 40MB]

- L1 cache [A100: 192KB per SM] - small, fast

You can look at the specs on your actual GPU.

Basic structure: run f(i) for all i = 0, ..., N-1

## Execution model

- *Thread*: process individual index (i.e., f(i))

- *Thread block* (a.k.a. concurrent thread arrays): scheduled on a single SM

- *Grid*: collection of thread blocks

Why thread blocks? Shared memory.

- Intuition: group f(i)'s that read similar data together

- Threads within a thread block have shared memory (as fast as L1 cache) [A100: 164KB]

- Can synchronize threads (for reading/writing) within a block (but not across blocks)

### Hardware and execution interact.

Thread blocks scheduled onto SMs in waves.

Problem: last wave has fewer thread blocks, leaving some SMs idle (low occupancy).

Wave quantization: make number of thread blocks divide # SMs.

Rule of thumb: number of thread blocks should be >= 4x # SMs

Challenge: some aspects of hardware are hidden from the execution model (e.g., scheduling, # SMs).

### Arithmetic intensity: # FLOPs / # bytes

- If high, operation is compute-bound (good)

- If low, operation is memory-bound (bad)

General rule: matrix multiplication is compute-bound, everything else is memory-bound

IMPORTANT: benchmark/profile your code!

You can read spec sheets (marketing material) and papers

...but performance depends on your library version, your hardware, your workload

...so there is no substitute for benchmarking/profiling your code.

Example computation: running forward/backward passes on an MLP.

Every time you make a change, benchmark/profile!

Benchmarking measures the wall-clock time of performing some operation.

It only gives you end-to-end time, not where time is spent (profiling).

It is still useful for:

- comparing different implementations (which is faster?), and

- understanding how performance scales (e.g., with dimension).

Let's define a convenient function for benchmarking an arbitrary function.

### Benchmarking matrix multiplication

First, let us benchmark matrix multiplication of square matrices.

Let us benchmark our MLP!

Scale the number of steps.

Scale the number of layers.

Scale the batch size.

Scale the dimension.

The timings are not always predictable due to the non-homogenous nature of CUDA kernels, hardware, etc.

You can also use `torch.utils.benchmark`, which provides more amenities. 

We did not use this to make benchmarking more transparent.

While benchmarking looks at end-to-end time, profiling looks at where time is spent.

Obvious: profiling helps you understand where time is being spent.

Deeper: profiling helps you understand (what is being called).

PyTorch has a nice built-in profiler 

Let's profile some code to see what is going on under the hood.

Let's start with some basic operations.

Observations

- You can see what CUDA kernels are actually being called.

- Different CUDA kernels are invoked depending on the tensor dimensions.

Name of CUDA kernel tells us something about the implementation.

Example: cutlass_80_simt_sgemm_256x128_8x4_nn_align1

- cutlass: NVIDIA's CUDA library for linear algebra

- 256x128: tile size

Let's now look at some composite operations.

Now let's profile our MLP.

We will also visualize our stack trace using a flame graph, which reveals where time is being spent.

Horace He's blog post 

Analogy: warehouse : DRAM :: factory : SRAM

Each operation needs to read/compute/write:

If we *fuse* the operations, only need to read/write once:

To see the effect of fusion, let's consider the GeLU activation function. 

Let's consider two ways to compute GeLU:

1. The default PyTorch implementation (fused):

2. We can also write our own by hand (not fused):

Let's benchmark.

Could not compare times - benchmark results were None

Let's look under the hood.

The PyTorch just calls one kernel whereas the others are atomic (remember the warehouse/factory) 

Now let's open the box to understand what's going on inside a CUDA kernel by writing our own.

Let's write the GeLU function in CUDA.

Check correctness of our implementation.

Benchmark our CUDA version.

Our CUDA implementation is faster than manual, but not as good as PyTorch.

Elementwise operations are easy in CUDA (though you can still be smarter).

But most interesting operations (e.g., matmul, softmax, RMSNorm) require reading multiple values.

For that, you have to think about managing shared memory, etc.

CUDA is an extension of C/C++ with APIs for managing GPUs.

Simplified picture: write f(i), CUDA kernel computes f(i) for all i.

Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8)

Thread block: collection of threads: blockIdx = (0, 1)

Thread: single unit of operation: threadIdx = (0, 3).

You write code that a thread execute, using (blockIdx, blockDim, threadIdx) to determine what to do.

Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong.

The `load_inline` function makes it convenient to write CUDA code and bind it to a Python module for immediate use.

Compile the CUDA code and bind it to a Python module.

Developed by OpenAI in 2021 

Make GPU programming more accessible

- Write in Python

- Think about thread blocks rather than threads

What does Triton offer?

                                             CUDA      Triton

- Memory coalescing (transfer from DRAM)     manual    automatic

- Shared memory management                   manual    automatic

- Scheduling within SMs                      manual    automatic

- Scheduling across SMs                      manual    manual

Compiler does more work, can actually outperform PyTorch implementations!

One big advantage of Triton is that you can step through the Python code.

Let's step through a Triton kernel.

Check that it's correct.

Let's now benchmark it compared to the PyTorch and CUDA implementations.

Remember to set TRITON_INTERPRET=0 for good performance.

Our Triton implementation (triton_gelu):

- is almost as good as the PyTorch implementation (pytorch_gelu).

- is actually slower than our naive CUDA implementation (cuda_gelu).

Triton operates on blocks, CUDA operates on threads.

Blocks allows Triton compiler to do other optimizations (e.g., thread coarsening).

Everything is way faster than the manual implementation (manual_gelu).

PTX (parallel thread execution) is like an assembly language for GPUs.

We can see the PTX code generated by Triton.

Observations:

- ld.global.* and st.global.* reads and writes from global memory

- %ctaid.x is block index, %tid.x is thread index

- %f* are floating point registers, %r* are integer registers

- One thread processes 8 elements at the same time (thread coarsening)

PTX is not generated when in interpret mode.

Let's go poke around at the PTX code.

So far, we have seen three ways to write GeLU:

- Use the default PyTorch function

- Write it in Python 

- Write it in CUDA 

- Write it in Triton 

- Write it in Python and compile it into Triton

Check correctness of our implementation.

Let's benchmark and profile it!

Let's look under the hood

So far, we've looked at elementwise operations in Triton (e.g., GeLU).

Now let us look at operations that aggregate over multiple values.

We will roughly follow the Triton fused softmax tutorial: 

Recall the softmax operation is used in attention and generating probabilities.

Normalize each row of a matrix:

[A1 A2 A3]   =>   [A1/A A2/A A3/A]

[B1 B2 B3]   =>   [B1/B B2/B B3/B]

Let's first start with the naive implementation and keep track of reads/writes.

Now let us write the Triton kernel.

Check our implementations are correct.

Now let's benchmark everything.

Look under the hood using the profiler.

Let's end by looking at the PTX code.

Matrix multipliction is perhaps the most optimized algorithm ever.

If you write matrix multiplication in CUDA, there's all sorts of crazy things you have to do.

It's much easier in Triton.

       k                  j                     

  [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]

i [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]

  [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]

Naively: need MKN reads, MN writes

Computing C4 and C5 both need A4, A5, A6.

Can we read A4, A5, A6 from DRAM once to compute both?

Answer: yes, using shared memory!

## Tiling (leveraging shared memory)

Recall that shared memory is:

- fast (10x faster) and small(~100KB)

- shared between all the threads in a block.

Trivial: for small matrices, load all of A and B into shared memory, then could compute C.

Now we get MK + KN reads, MN writes

But what if we have big matrices...

Key idea: divide the matrix into blocks.

For each block of A and block of B:

- load into shared memory,

- do mini-matrix multiplication,

- write the partial sum.

Animation of tiled matrix multiplication 

## Leveraging L2 cache

Two ways of computing 9 elements of a matrix:

1. Loads 9 + 81 = 90 blocks

1. Loads 27 + 27 = 54 blocks

Process the blocks in an order that minimizes the reads.

Why write your own kernel for matrix multiplication (e.g., A @ B)?

Answer: fusion with another operation (e.g., gelu(A @ B))

Let's try it!

Horace He's blog post 

CUDA MODE Lecture 1: how to profile CUDA kernels in PyTorch 

CUDA MODE Lecture 2: Chapters 1-3 of PPMP book 

CUDA MODE Lecture 3: Getting started with CUDA for Python Programmers 

CUDA MODE Lecture 4: Compute and memory basics 

CUDA MODE Lecture 8: CUDA performance checklist 

HetSys Course: Lecture 1: Programming heterogenous computing systems with GPUs 

HetSys Course: Lecture 2: SIMD processing and GPUs 

HetSys Course: Lecture 3: GPU Software Hierarchy 

HetSys Course: Lecture 4: GPU Memory Hierarchy 

HetSys Course: Lecture 5: GPU performance considerations
