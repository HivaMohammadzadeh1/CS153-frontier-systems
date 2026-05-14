# Data Parallelism (Area B — Training)

In distributed training, data parallelism replicates the model across workers and splits each batch across them. Each worker computes gradients locally and synchronizes via all-reduce. DDP is PyTorch's standard implementation. The bottleneck is communication: large models have large gradient buffers, and slow interconnects make all-reduce dominate step time. A misconception is that DDP scales arbitrarily — beyond modest worker counts, communication overhead and stragglers cap throughput.
