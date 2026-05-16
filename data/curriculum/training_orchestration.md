# Training Orchestration: Ray, Slurm, and Kubernetes for Distributed ML

**Area B — Training Systems | Learning Memory OS Curriculum**

---

## 1. Why Orchestration Is a First-Class Problem

Writing a training script that runs on a single GPU is straightforward. Getting that script to run reliably on 512 GPUs across 64 nodes, restart cleanly after a preemption at step 47,000, and maintain close-to-linear scaling efficiency is a multi-week engineering effort. Orchestration is the layer that turns a working local script into a production training job: it handles job submission, resource allocation, process lifecycle, fault recovery, and checkpoint management.

The three dominant orchestration systems in industry are **Ray**, **Slurm**, and **Kubernetes** with job controllers. They address the same core problems but differ enormously in their operational model, failure semantics, and the infrastructure they assume. Understanding when to use each — and what each system cannot do — is a fundamental skill for ML systems engineers.

---

## 2. Slurm: The HPC Standard

Slurm (Simple Linux Utility for Resource Management) is the scheduling backbone of most academic and national supercomputing clusters. It is a batch job scheduler: you write a job script specifying the number of nodes, tasks per node, GPUs, memory, and walltime, then submit it with `sbatch`. Slurm maintains a queue, allocates resources when they become available, launches your processes, and tears them down when the job finishes or the walltime expires.

### 2.1 Job Submission

A minimal multi-node PyTorch DDP job script looks like:

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8        # 8 GPUs per node
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-large

srun --container-image=nvcr.io/nvidia/pytorch:24.01-py3 \
     --container-mounts=/scratch/checkpoints:/checkpoints \
     python train.py --ckpt-dir /checkpoints
```

The `srun` command here uses **Pyxis**, an NVIDIA-maintained Slurm plugin that enables containerized job steps. Pyxis is widely used in modern HPC clusters because it provides reproducible environments without root privileges.

### 2.2 Gang Scheduling

A critical property of distributed training is that all processes must be running simultaneously — a straggler or a failed process blocks the entire collective. Slurm gang-schedules a job: all requested nodes must be allocated before any processes start. This is correct for training but means a job requesting 64 nodes waits in the queue until all 64 are free simultaneously, which can create significant queue delays on a heavily loaded cluster.

### 2.3 Fault Tolerance in Slurm

Slurm's native fault model is coarse: if a node fails, the job fails. Users handle resilience at the application layer — training code writes checkpoints every N steps, and the job resubmits itself via a `--dependency=afternotok:$SLURM_JOB_ID` chain. This is fragile because a failure at step 47,000 means restarting from the last checkpoint (say, step 40,000), wasting 7,000 steps of compute.

More sophisticated deployments use **elastic training** (discussed below) and NFS or Lustre checkpointing to minimize recovery cost.

---

## 3. Ray and Ray Train

Ray is an open-source distributed computing framework built around a Python actor model. Ray Train is the high-level API for distributed deep learning on top of Ray. Unlike Slurm, Ray is designed as a dynamic, elastic cluster: workers can join and leave, tasks can be retried automatically, and resource requirements can be specified per-task rather than globally.

### 3.1 Ray Architecture

A Ray cluster has one head node (the GCS — Global Control Store) and any number of worker nodes. The GCS maintains the cluster state, scheduling decisions, and actor locations. Ray's distributed object store (Plasma) allows workers to share tensors without serialization when they are on the same node.

A Ray Train job is submitted programmatically:

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

trainer = TorchTrainer(
    train_loop_per_worker=my_train_fn,
    scaling_config=ScalingConfig(num_workers=64, use_gpu=True),
    run_config=RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=2)),
)
result = trainer.fit()
```

Ray handles process spawning, `MASTER_ADDR/MASTER_PORT` setup, and NCCL initialization. `my_train_fn` runs on each worker and calls `train.report(metrics)` to log metrics back to the driver.

### 3.2 Fault Tolerance in Ray Train

Ray Train has first-class fault tolerance: if a worker crashes, Ray can restart the `TorchTrainer` from the last reported checkpoint without requiring a full job resubmission. This is a major operational advantage over Slurm for jobs running on cloud spot instances, where preemptions are frequent.

The recovery flow is: (1) worker failure detected by the GCS heartbeat, (2) checkpoint restored from shared storage (S3, GCS, NFS), (3) workers respawned, (4) training resumes. The full recovery loop typically completes in 2-5 minutes for a 64-node job, depending on checkpoint size and storage bandwidth.

### 3.3 Elastic Training with Ray

Elastic training allows the number of workers to change mid-training — nodes can join or leave and training continues. Ray Train supports this via the `ElasticTrainingCallback`. The catch is that elastic training complicates learning rate schedules and gradient accumulation because the effective batch size changes dynamically. Most practitioners fix the worker count and use fault tolerance rather than true elasticity, reserving elasticity for jobs that need to shrink to free up resources for higher-priority workloads.

---

## 4. Kubernetes with Kubeflow and Volcano

Kubernetes (K8s) is the dominant container orchestration platform for cloud-native workloads. By itself, K8s does not know about ML training jobs — it knows about Pods and Services. The ML-specific job semantics are provided by custom resource definitions (CRDs) and controllers.

### 4.1 PyTorchJob via Kubeflow

Kubeflow's Training Operator defines the `PyTorchJob` CRD. A `PyTorchJob` spec describes master and worker pod templates, replica counts, and restart policies. The controller creates the pods, sets `MASTER_ADDR` and `RANK` environment variables, and restarts failed pods up to a configurable limit.

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training-image:latest
            resources:
              limits:
                nvidia.com/gpu: "8"
    Worker:
      replicas: 7
      restartPolicy: OnFailure
```

### 4.2 Volcano for Gang Scheduling

Standard Kubernetes does not gang-schedule: pods may start partially, leading to some workers waiting indefinitely for others. **Volcano** is a batch scheduling system that adds gang scheduling, fairness queues, and job preemption to Kubernetes. For distributed training, Volcano ensures that all pods in a job start atomically.

Volcano uses a `PodGroup` to express the gang constraint: the scheduler waits until `minMember` pods can be co-scheduled before starting any of them. This prevents the "partial allocation" deadlock.

### 4.3 Multi-Cluster vs Single-Cluster Tradeoffs

Single-cluster deployments simplify networking (all nodes share a flat L3 network, InfiniBand fabric is unified) and reduce scheduling latency. Multi-cluster deployments are used when workloads span geographic regions, when regulatory requirements mandate data locality, or when scale exceeds a single cluster's capacity (e.g., Pathways, described in the Google Pathways paper, spans multiple TPU pods across data centers).

The primary cost of multi-cluster training is inter-cluster bandwidth: a 400 Gbps backbone link between data centers is roughly 4x slower than an InfiniBand HDR100 fabric within a single cluster, making tensor-parallel and pipeline-parallel communication prohibitively expensive across clusters. In practice, multi-cluster training uses data-parallel communication only, with each cluster holding a complete model replica.

---

## 5. Checkpointing Strategies

Checkpointing is the primary resilience mechanism for training jobs. The key dimensions are frequency, storage target, and format.

**Frequency**: Checkpointing every 500-1000 steps at bf16 is typical for large models. For a 70B model, a checkpoint is roughly 140 GB in bf16. At 500-step intervals on a cluster with 300 GB/s NFS throughput, each checkpoint write takes ~0.5 seconds — negligible. If checkpoints are written to S3, bandwidth is lower (~50 GB/s for a well-optimized multi-part upload from 64 nodes), so the write takes ~3 seconds per checkpoint. The tradeoff between checkpoint frequency and storage cost is a classic FinOps decision.

**Async checkpointing**: Megatron-LM and other frameworks support asynchronous checkpointing: the optimizer state is copied to CPU memory (pinned) and written to disk by a background thread while training continues. The memory cost is one extra copy of the optimizer state (~2.5x model size in Adam). Async checkpointing can hide the I/O latency almost completely for reasonable storage bandwidths.

**Format**: PyTorch's `torch.save` produces a pickle-based format that is not always backward-compatible. For production systems, structured formats like Safetensors (used by Hugging Face) or model-specific sharded formats (Megatron-LM uses a custom shard-per-TP-rank format) are preferred. Safetensors supports memory-mapped loading, which reduces checkpoint load time from minutes to seconds for large models.

---

## 6. Failure Recovery Patterns

Production training clusters exhibit several classes of failures:

**GPU hardware failures**: NCCL hangs are the most common symptom. The training process appears alive but collective operations stall. Detection requires a watchdog timeout (e.g., `NCCL_TIMEOUT=1800`). Recovery: terminate all processes, restart from the last checkpoint. NVIDIA's `dcgmi` tool can blacklist the failed GPU.

**Network transient failures**: A dropped packet during an all-reduce causes NCCL to report an error and hang. With elastic training + checkpoint recovery, the job can resume within minutes. Without it, the entire job must be requeued.

**Storage failures**: Writing checkpoints to a Lustre file system with a failing OST (object storage target) silently produces corrupt checkpoint files. Defense: always verify checkpoint integrity with a hash after writing, and maintain N-1 checkpoints so a corrupt write does not destroy recovery state.

---

## Misconception: Ray and Slurm are equivalent and interchangeable

Ray and Slurm are not interchangeable. Slurm is a batch scheduler with coarse-grained fault semantics — it schedules jobs, not individual tasks, and job failure means full resubmission. Ray is a dynamic actor framework with fine-grained task retries and elastic scaling. Slurm is the right choice for managed HPC clusters with dedicated InfiniBand fabric and fixed job sizes. Ray is the right choice for cloud environments with spot instances, dynamic workloads, and Python-native workflows. Many organizations run Ray on top of Slurm using `ray up --slurm` to get dynamic task management within a Slurm allocation.

## Misconception: Gang scheduling eliminates deadlocks in distributed training

Gang scheduling ensures that all processes start simultaneously, preventing the "partial allocation" deadlock where some workers wait indefinitely. However, gang scheduling does not prevent deadlocks caused by application-level bugs (e.g., a barrier that only some ranks reach), NCCL communication graph misconfiguration, or asymmetric collective operations. NCCL hangs in production are usually application-level bugs, not scheduling bugs.

## Misconception: Elastic training is always better than fixed-size training

Elastic training adds significant complexity to learning rate schedulers, gradient accumulation logic, and checkpoint compatibility. If the training run changes worker count mid-job, the effective global batch size changes, which can destabilize training dynamics unless the LR is carefully rescaled. For most production pre-training runs (which operate at fixed scale on dedicated clusters), fixed-size training with checkpoint-based recovery is simpler and more reliable than elastic training.

## Misconception: Checkpointing is a solved problem

Checkpoint writes can silently corrupt if storage has partial failures. Checkpoint loading can fail if the framework version changes between write and read (pickle-based formats are fragile). Checkpoint files can be incomplete if the job is killed mid-write. Defense requires: hash verification after every write, atomic renames (write to temp, rename to final), at least two recent checkpoints, and a startup validation that loads and forward-passes the checkpoint before beginning training.

## Misconception: Kubernetes is better than Slurm for large-scale training

Kubernetes provides flexibility and a rich ecosystem but imposes overhead: pod startup times are higher than Slurm process launch times (10-60 seconds vs 1-5 seconds), the K8s API server can become a bottleneck at thousands of nodes, and InfiniBand RDMA requires special device plugins. Slurm, designed specifically for HPC, typically achieves lower scheduling latency, more predictable performance, and better support for high-speed interconnects on dedicated clusters. K8s + Volcano is preferred when the cluster is heterogeneous, cloud-native, or shares resources with non-ML workloads.

---

## 7. Practical Example: Training Llama-3-70B on 64 Nodes with Slurm + NCCL

A 70B parameter model in bf16 requires 140 GB for weights alone. With FSDP/ZeRO-3, the memory per GPU is roughly `140 GB / N_GPUs + optimizer overhead`. On 64 nodes × 8 H100s = 512 GPUs, each GPU holds ~280 MB of sharded weights plus ~2 GB of optimizer state — well within an H100's 80 GB HBM.

The NCCL topology file (`NCCL_TOPO_FILE=/etc/nccl_topo.xml`) tells NCCL the NVLink and InfiniBand topology. Without this file, NCCL auto-detects topology via PCI traversal, which is slower and sometimes wrong on heterogeneous fabric configurations. On an IB HDR100 fabric (200 Gbps per port), a 512-GPU all-reduce of a 1 GB tensor takes approximately 8 ms at the theoretical bandwidth limit. In practice, with NCCL ring all-reduce and real-world MPI-like overhead, latency is 12-15 ms.

For this job:
- Checkpoint frequency: every 500 steps
- Checkpoint size: ~280 GB (weights + optimizer + scheduler)
- Storage target: Lustre `/scratch` at 500 GB/s aggregate bandwidth
- Write time per checkpoint: ~0.56 seconds (async, background thread)
- Recovery time after failure: ~3 minutes (load checkpoint + respawn + NCCL init)

---

## 8. Exercise

**Exercise**: Set up a Ray Train job that trains a 7B parameter language model on 4 nodes, implements checkpoint-based fault recovery, and uses an exponential LR decay schedule that correctly adjusts for worker restarts (i.e., the LR at resume is consistent with the step count at the checkpoint, not reset to initial LR). Compare recovery time after simulated node failure (kill one worker process) vs a full Slurm job resubmission with the same model and checkpoint. Report: recovery latency, effective MFU before and after recovery, and checkpoint write overhead as a fraction of step time.

---

## References

- Ray Train documentation: https://docs.ray.io/en/latest/train/train.html
- Megatron-LM: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- Kubeflow Training Operator: https://github.com/kubeflow/training-operator
- Volcano batch system: https://volcano.sh
- Pyxis (Slurm container plugin): https://github.com/NVIDIA/pyxis
- The Pathways system paper (Dean et al., 2022) describes multi-cluster training across TPU pods
- NCCL documentation: https://docs.nvidia.com/deeplearning/nccl/user-guide/
