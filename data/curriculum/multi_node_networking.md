# Multi-Node Networking: InfiniBand, NVLink, RDMA, and Collective Communications

**Area B — Training Systems | Learning Memory OS Curriculum**

---

## 1. Why the Network Is the Bottleneck

Modern GPU clusters can deliver 3,000+ TFLOPS of compute per node with H100 SXM5. That compute is useless if the GPUs spend most of their time waiting for gradient tensors to arrive from other nodes. The fundamental challenge of distributed training is keeping the GPUs busy with compute while the network carries gradients and activations between ranks.

The key ratio is **compute-to-communication ratio**: the amount of FLOPs per gradient byte. For a standard 70B dense model with data parallelism, each all-reduce communicates ~140 GB of gradient data per step, while the forward+backward pass performs ~14 PFLOP of compute per GPU. An H100 at 3 PFLOPS peak takes ~4.7 ms for the compute. A 400 Gbps InfiniBand link requires ~2.8 ms for the all-reduce (at 50% efficiency). The compute and communication are nearly balanced — any degradation in network performance directly extends step time.

---

## 2. NVLink and NVSwitch: Intra-Node Fabric

Within a node, GPUs are interconnected via **NVLink**, NVIDIA's proprietary high-bandwidth interconnect. NVLink is strictly superior to PCIe for GPU-GPU communication within a server.

### 2.1 NVLink Generations

| Generation | Bandwidth (bidirectional) | Introduced |
|-----------|--------------------------|------------|
| NVLink 2.0 | 300 GB/s | V100 |
| NVLink 3.0 | 600 GB/s | A100 |
| NVLink 4.0 | 900 GB/s | H100 |

An H100 SXM5 node (DGX H100) has all 8 GPUs connected via **NVSwitch 3.0**, which provides a full all-to-all bandwidth of 900 GB/s per GPU. An all-reduce across 8 H100s within a node on a 1 GB tensor completes in approximately 1.1 ms, constrained entirely by NVSwitch bandwidth.

### 2.2 NVSwitch Topology

NVSwitch is a non-blocking crossbar switch implemented in dedicated ASICs on the DGX server board. In a DGX H100, 4 NVSwitch chips each connect to all 8 GPUs, providing 4x redundant paths. The full-bisection bandwidth means any GPU can simultaneously send data to any other GPU at full bandwidth with no head-of-line blocking.

For tensor parallelism (TP), which requires all-gather and reduce-scatter at every transformer layer, NVLink bandwidth is the primary constraint. TP degree is typically limited to 8 (one node) precisely because inter-node bandwidth (even InfiniBand) is 3-5x lower than NVLink bandwidth.

---

## 3. InfiniBand vs RoCE

For inter-node communication, the two dominant fabrics are **InfiniBand (IB)** and **RoCE (RDMA over Converged Ethernet)**.

### 3.1 InfiniBand

InfiniBand is a purpose-built network fabric for HPC, with its own physical layer, data link layer, and transport protocol. It is not Ethernet. Key properties:

- **HDR InfiniBand**: 200 Gbps per port (per NVIDIA Mellanox HDR spec)
- **NDR InfiniBand**: 400 Gbps per port
- **Latency**: 0.5-1 µs MPI message latency (vs 5-10 µs for Ethernet)
- **Hardware offload**: RDMA operations are handled entirely in the NIC (HCA — Host Channel Adapter), bypassing the CPU and OS kernel
- **Lossless**: IB uses credit-based flow control; packets are never dropped (they are held at the sender until credits allow). This is critical for NCCL collective operations.

A modern H100 DGX SuperPOD uses NDR400 InfiniBand with 8 HDR ports per node (8 × 400 Gbps = 3.2 Tbps intra-pod bandwidth per node).

### 3.2 RoCE (RDMA over Converged Ethernet)

RoCE implements the InfiniBand transport protocol (IB verbs) over an Ethernet physical layer. RoCEv2 (the current version) uses UDP/IP. It allows existing Ethernet infrastructure to provide RDMA semantics.

The critical difference: Ethernet is lossy by default. RoCEv2 requires **DCQCN (Data Center Quantized Congestion Notification)** or **PFC (Priority Flow Control)** to approximate the lossless behavior of IB. PFC is a crude backpressure mechanism that can cause head-of-line blocking and "PFC storms" that degrade cluster-wide performance. DCQCN is more sophisticated but harder to tune.

**When to use IB vs RoCE**: Dedicated ML clusters with budget for Mellanox Quantum or Quantum-2 switches should use InfiniBand — the lossless fabric and lower latency justify the cost. Cloud providers (AWS, Azure, GCP) use RoCE internally (Azure uses RoCEv2 with DCQCN; Google uses a proprietary variant). If you are building on cloud VMs with SR-IOV ENA Enhanced Networking, you are using Ethernet-backed RDMA, not native IB.

---

## 4. RDMA Fundamentals

**RDMA (Remote Direct Memory Access)** allows one node to read or write directly to the memory of another node without involving the CPU or OS on the remote side. This is the key property that enables low-latency, high-throughput collective communications.

The RDMA primitives used by NCCL are:

- **SEND/RECV**: Traditional message passing — both sides are involved.
- **WRITE**: Initiator writes directly into a pre-registered buffer on the remote node. The remote CPU is not involved.
- **READ**: Initiator reads from a remote buffer. Less common in collectives.

NCCL primarily uses RDMA WRITE (via InfiniBand RC — Reliable Connection — transport) for inter-node transfers. The combination of hardware-level reliability, kernel bypass, and zero-copy transfers achieves bus utilization rates of 85-95% of theoretical bandwidth.

**Memory registration** is a prerequisite: buffers used for RDMA must be registered with the NIC. Registration pins the physical pages and records their physical addresses in the NIC's memory translation table. For large, long-lived training buffers (model weights, optimizer states), registration cost is a one-time overhead. For dynamic allocations (activations), registration can become a bottleneck if done repeatedly — NCCL maintains a registration cache to amortize this cost.

---

## 5. Collective Operations

Distributed training relies on five collective operations. Understanding their algorithmic complexity and optimal implementations is essential for choosing the right parallelism strategy.

### 5.1 All-Reduce

**Purpose**: Sum (or average) a tensor across all N ranks, with every rank receiving the result. Used for gradient synchronization in data parallelism.

**Ring all-reduce** (Baidu Deep Learning, 2017): Arranges N ranks in a logical ring. The operation proceeds in two phases:
1. **Reduce-scatter**: Each rank sends 1/N of its tensor to the next rank and accumulates a partial reduction. After N-1 steps, each rank has the full reduction of 1/N of the tensor.
2. **All-gather**: Each rank sends its reduced chunk to the next rank. After N-1 steps, every rank has the full result.

Total data sent per rank: `2 * (N-1)/N * tensor_size ≈ 2 * tensor_size` for large N.
Bandwidth: `2 * tensor_size * (N-1)/N / bandwidth_per_link`. For large N, this is approximately `2 * tensor_size / bandwidth_per_link`, independent of N. This is the key advantage: ring all-reduce scales perfectly.

**Tree all-reduce**: Reduces in a binary tree topology. Latency scales as O(log N) but bandwidth scales as O(N) (the root link is a bottleneck). Tree all-reduce is better for small tensors (latency-bound) but worse for large tensors (bandwidth-bound).

NCCL adaptively selects ring vs tree vs recursive halving based on message size and topology.

### 5.2 All-Gather

**Purpose**: Collect different chunks from all ranks and deliver the full tensor to every rank. Used in ZeRO-3 (collect the sharded parameters before the forward pass) and in tensor parallelism (collect activations after a column-parallel linear layer).

Data sent per rank: `tensor_size * (N-1)/N`, received: `tensor_size`.

### 5.3 Reduce-Scatter

**Purpose**: Reduce a tensor across all ranks and distribute the result shards back to each rank. Every rank receives 1/N of the final reduced tensor. Used in ZeRO-3 (scatter reduced gradients to the owning rank) and in tensor parallelism (reduce partial activations and scatter).

All-Reduce = All-Gather ∘ Reduce-Scatter (the ring algorithm implements both phases sequentially).

### 5.4 Broadcast and Barrier

**Broadcast**: Send a tensor from one root rank to all other ranks. Used for initialization (broadcast initial weights from rank 0).

**Barrier**: Synchronize all ranks at a point — no rank proceeds past the barrier until all ranks have reached it. Barriers are expensive because they require a round-trip through all ranks.

---

## 6. NCCL: NVIDIA Collective Communications Library

NCCL (pronounced "nickel") is NVIDIA's implementation of collective operations optimized for NVLink + InfiniBand topologies. Key design decisions:

- **Topology-aware ring construction**: NCCL builds logical rings that maximize NVLink utilization within a node before using inter-node IB links. Intra-node all-reduce uses NVSwitch; inter-node all-reduce uses IB.
- **Parallel channels**: NCCL opens multiple parallel rings/trees simultaneously to saturate available bandwidth. With 8 HDR ports per H100 node, NCCL uses 8 parallel RDMA channels.
- **Pipelining**: NCCL pipelines the reduce-scatter and all-gather phases to keep all links busy simultaneously.

The `NCCL_TOPO_FILE` environment variable specifies an XML topology file that describes the NVLink and IB fabric. Without this file, NCCL auto-detects topology via PCI probing, which can give suboptimal ring assignments on complex fabrics. For production training on dedicated clusters, always provide a topology file.

**NCCL_DEBUG=INFO** logs topology detection results, ring assignments, and per-channel bandwidth. This is the first debugging tool when collective performance is below expectations.

---

## 7. Topology-Aware Sharding

Tensor parallelism (TP) and pipeline parallelism (PP) place different communication demands on the fabric:

- **TP** requires all-gather and reduce-scatter at every transformer layer — high bandwidth, latency-sensitive. Must use NVLink (intra-node only).
- **PP** requires point-to-point transfers of activation tensors between pipeline stages — lower bandwidth, latency-sensitive for bubble ratio. Typically assigned to IB links connecting different nodes.
- **DP** requires all-reduce of gradients once per batch — high bandwidth, latency-tolerant (can overlap with the backward pass). Uses IB links.

The standard topology assignment for a 64-node H100 cluster training a 70B model with 3D parallelism (TP8 × PP4 × DP16) is:
- TP: within a node (8 GPUs, NVLink)
- PP: across 4 nodes (one pipeline stage per node, IB HDR)
- DP: across 16 pipeline replicas (IB HDR, overlapped with backward)

### Bandwidth Budget

For a 70B model with hidden size 8192, 80 layers:
- Per-layer all-gather (TP): `8192 * 4 * 4 bytes * 2 = 262 KB` every forward pass. At 900 GB/s NVLink: 0.3 µs per layer.
- Per-batch DP all-reduce: `280 GB` of gradients. At 400 Gbps IB (50 GB/s per port, 8 ports = 400 GB/s aggregate): ~0.7 seconds at ring efficiency.
- Per-step backward compute on one H100 at 3 PFLOPS: ~18 ms per layer × 80 layers = ~1.4 seconds.
- Communication-to-compute ratio: ~0.7 / 1.4 = 50% — can be hidden with gradient accumulation.

---

## Misconception: InfiniBand and Ethernet are just different speeds of the same thing

InfiniBand is not Ethernet. IB has its own protocol stack: a physical layer (serial lanes), a data link layer with credit-based flow control, and a transport layer with hardware-implemented reliability. The key difference is that IB is lossless by design — the flow control protocol prevents drops at the NIC level, not with retransmits. Ethernet can drop packets; RoCEv2 relies on ECN/PFC to approximate losslessness. For NCCL's ring all-reduce, a single dropped packet causes the entire collective to stall or fail, so losslessness is not a nice-to-have: it is a correctness requirement.

## Misconception: More GPU memory bandwidth means faster collectives

GPU memory bandwidth (HBM bandwidth, ~3.3 TB/s on H100) is irrelevant for inter-node collective performance. Collectives are bottlenecked by the NIC-to-NIC bandwidth (InfiniBand port speed) and by the NVLink crossbar for intra-node operations. HBM bandwidth matters for compute-bound kernels (matrix multiplications), not for communication-bound collectives.

## Misconception: NCCL automatically uses the fastest available topology

NCCL detects topology automatically but is not omniscient. On complex fabrics (multi-rail IB, asymmetric NVLink configurations, virtual topologies in cloud VMs), auto-detection can produce suboptimal ring assignments. On cloud instances with virtualized network adapters, NCCL may not detect RDMA capability at all and fall back to socket-based communication, which is 5-10x slower. Always verify with `NCCL_DEBUG=INFO` and confirm that `NCCL: Using IB RDMA` appears in the logs, not `NCCL: Using socket`.

## Misconception: Ring all-reduce scales linearly with the number of nodes

Ring all-reduce has nearly constant bandwidth cost per rank as N grows (sending ~2 × tensor_size of data), but latency scales as O(N) because of the ring traversal. For small tensors (< 1 MB), the O(N) latency term dominates and ring all-reduce degrades. NCCL switches to tree all-reduce or recursive halving for small tensors. For large tensors (> 10 MB), ring all-reduce effectively scales — adding more nodes does not increase the total data sent per rank.

## Misconception: NVLink is only useful for communication within a single GPU pair

NVLink via NVSwitch provides all-to-all connectivity among all 8 GPUs in a DGX node simultaneously, not just pairwise. The NVSwitch crossbar means any-to-any transfers happen at full 900 GB/s NVLink bandwidth without contention. This enables all-reduce operations among 8 GPUs to proceed at NVLink speeds, not PCIe speeds — the difference is roughly 10x (900 GB/s vs ~64 GB/s for PCIe 5.0 ×16).

---

## 8. Practical Example: NCCL All-Reduce Debugging on 32 Nodes

A training job on 32 H100 nodes shows step times 40% higher than expected. The profiler shows most of the excess time is in the all-reduce operation. Debugging steps:

1. `NCCL_DEBUG=INFO | grep -E "NCCL|ring|algo"` — confirm NCCL is using RDMA, not sockets; confirm the ring assignment.
2. `ibstat | grep Rate` — check that all 8 IB ports show 400 Gbps (not degraded to 200 Gbps due to link negotiation).
3. Check switch load: if two training jobs share the same IB switches, their all-reduces can contend for bandwidth. Slurm node allocation policies should ensure jobs get dedicated switch ports.
4. `nccl-tests/build/all_reduce_perf -b 1G -e 1G -t 1 -n 100` — measure raw all-reduce bandwidth. Compare against `N/2 * 2 * message_size / total_bandwidth` theoretical.

In this example, the IB ports were negotiated at 200 Gbps instead of 400 Gbps due to a firmware mismatch on the HCA. Updating the HCA firmware resolved the issue.

---

## 9. Exercise

**Exercise**: On a cluster with 8 nodes, each with 8 A100 GPUs (NVLink 3.0, 600 GB/s) and 2 HDR100 IB ports (200 Gbps each), compute:
1. The theoretical all-reduce time for a 1 GB gradient tensor using ring all-reduce, assuming perfect bandwidth utilization.
2. The NVLink all-reduce time for the same tensor across 8 GPUs within one node.
3. The ratio of inter-node to intra-node all-reduce bandwidth. Verify that TP degree = 8 (within one node) is the correct choice for this fabric.
4. What is the maximum tensor parallelism degree you could use before inter-node TP communication becomes more expensive than the compute savings? Express this as a function of model FLOPs per layer and IB bandwidth.

---

## References

- NCCL documentation: https://docs.nvidia.com/deeplearning/nccl/user-guide/
- Ring all-reduce: Baidu Research blog, "Bringing HPC techniques to deep learning" (2017)
- NVLink 4.0 and NVSwitch 3.0: NVIDIA H100 architecture whitepaper
- InfiniBand NDR specification: NVIDIA Mellanox NDR technical brief
- Megatron-LM: Shoeybi et al. (2019) — topology-aware 3D parallelism
- Massively parallel training with 3D parallelism: Smith et al., "Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B" (2022)
- RoCEv2 and DCQCN: Zhu et al., "Congestion Control for Large-Scale RDMA Deployments" (SIGCOMM 2015)
