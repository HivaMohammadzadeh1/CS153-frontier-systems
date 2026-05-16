# Feature Stores: Online/Offline Split, Embeddings, and Freshness

**Area F — Production ML Systems | Learning Memory OS Curriculum**

---

## 1. The Feature Engineering Problem at Scale

Machine learning models need features — numerical representations of entities (users, items, events) computed from raw data. For simple models, feature computation is done at training time and baked into the dataset. This breaks down in production when:

1. **Freshness requirements**: A recommendation model needs user behavior features from the last 10 minutes, not last week's training data.
2. **Consistency requirements**: The features used at serving time must be exactly reproducible at training time. Using different feature computation logic between training and serving causes **training-serving skew**, one of the most insidious production ML bugs.
3. **Scale requirements**: Serving 50,000 requests per second while computing complex features in real time is infeasible — features must be precomputed and cached.

A **feature store** solves all three: it maintains a centralized repository of feature definitions, precomputed feature values, and guarantees that the same feature values are available at training time (offline) and serving time (online) with defined freshness guarantees.

---

## 2. Online vs Offline Store Split

The fundamental architectural split in a feature store is between the **online store** and the **offline store**.

### 2.1 Online Store

The online store serves features at inference time. Requirements:
- **Low latency**: p99 < 5-10 ms (compatible with an overall serving SLA of 100-200ms)
- **High throughput**: 10,000s of feature lookups per second
- **Single-key lookup**: Fetch features for one entity (user_id, item_id) at a time

Implementation: A key-value store. **Redis** is the most common choice for its low latency (~0.5ms read) and atomic operations. DynamoDB, Google Cloud Bigtable, and Cassandra are alternatives for larger scale or different consistency requirements.

A typical online store entry:

```
key: "user:{user_id}:features:v3"
value: {
  "avg_click_rate_1d": 0.043,
  "session_count_7d": 12,
  "last_purchase_category": "electronics",
  "embedding": [0.12, -0.34, 0.09, ...]  // 128-dim
  "computed_at": "2025-01-15T10:20:00Z"
}
TTL: 3600  // 1-hour expiry
```

### 2.2 Offline Store

The offline store serves features at training time and for batch inference. Requirements:
- **High throughput**: Process billions of feature rows per day
- **Time-travel queries**: Fetch the feature values as they were at a specific point in the past (for point-in-time correct training)
- **Storage efficiency**: Store historical feature values at columnar layout

Implementation: A columnar data warehouse or data lake. **Parquet** files on S3/GCS, queried via Spark or BigQuery. Some feature stores use Hive tables; others use Delta Lake or Iceberg for ACID transactions and time-travel.

### 2.3 The Materialization Pipeline

Features are computed in a **batch pipeline** (e.g., daily Spark job) and written to both stores:
- **Offline**: Historical rows appended to the Parquet/Iceberg table.
- **Online**: Current values written to Redis with an appropriate TTL.

For features that need sub-minute freshness (streaming features), a **stream processor** (Flink, Spark Streaming, Kafka Streams) computes features from the real-time event stream and writes to the online store.

---

## 3. Point-in-Time Correctness

**Point-in-time correctness** (PIT correctness) is the requirement that, when generating training data for an event at time T, the feature values used must be the values that *were available* at time T — not features computed from data after T. Violating this creates **label leakage**: the model sees the future at training time but not at serving time.

**Example**: A churn prediction model is trained on user features as of July 1. User Alice churned on July 5. If the training pipeline uses Alice's features computed *after* July 5 (which may include her "churned=True" flag computed in a weekly job), the model sees a spurious signal that does not exist at serving time.

PIT-correct training data generation requires a snapshot of feature values at each label event's timestamp. This is expensive to compute naively (a full scan for every event) and is one of the core hard problems that feature stores solve.

**Implementation approaches**:
- **Temporal join**: Join the label events table with the feature table on `entity_id` and the condition `feature_time ≤ event_time`, taking the most recent feature row before each event.
- **Snapshot tables**: Maintain daily (or hourly) snapshots of all feature values, enabling fast PIT lookups.
- **Apache Iceberg time travel**: `SELECT * FROM features TIMESTAMP AS OF '2025-01-15 10:00:00'` — built-in PIT queries using Iceberg's snapshot history.

---

## 4. Embedding Stores

Embeddings are high-dimensional vectors (64-2048+ dimensions) that represent entities. In modern ML systems, embeddings are features — the output of a trained embedding model applied to an entity's raw attributes.

**Embeddings as features** in a feature store have unique characteristics:
- **Size**: A single embedding is 512 bytes (128-dim float32) to 8 KB (2048-dim float32). For 100M users × 256-dim embedding = 25.6 GB in the online store — significant but manageable in Redis.
- **Freshness**: Embeddings change slowly (re-embedding users weekly is often sufficient) vs behavioral counts (may need hourly updates).
- **Retrieval pattern**: Unlike scalar features (exact key lookup), embeddings are often used for approximate nearest neighbor (ANN) search. This requires a **vector store** (Pinecone, Weaviate, Qdrant, pgvector), which is a different infrastructure from the KV-based online store.

### 4.1 Hybrid Feature + Embedding Architecture

Production recommendation systems often combine:
1. **KV online store** (Redis): Scalar features, categorical features, recent behavioral counts. Sub-millisecond lookup.
2. **Vector store** (Pinecone/Weaviate): User and item embeddings for ANN retrieval. 5-20ms query latency.
3. **Offline store** (Parquet/Iceberg): Training data generation with PIT joins.

---

## 5. Freshness vs Cost Tradeoffs

Feature freshness — how recent the feature values are — directly trades off against compute and storage cost.

| Freshness | Computation approach | Infrastructure | Cost |
|-----------|---------------------|----------------|------|
| Daily | Batch Spark/BigQuery job | Hadoop/BigQuery | Low |
| Hourly | Mini-batch or streaming | Spark Streaming | Medium |
| 5-minute | Near-real-time streaming | Flink, Kafka Streams | High |
| Sub-second | Real-time computation | Inline feature computation | Very high |

**Decision framework**: How much does feature freshness affect model quality? For a fraud detection model, a user's transaction count in the last 5 minutes is extremely high-signal and worth streaming compute. For a content recommendation model, weekly user preference embeddings may be sufficient, making daily batch compute appropriate.

**Practical heuristic**: Start with daily batch features. Measure model performance on validation data with different feature lag assumptions (simulate stale features). If AUC drops significantly with 24-hour-old features, invest in streaming infrastructure for those features. Most features have negligible degradation with 24-hour lag; only a few require sub-hour freshness.

---

## 6. Feature Drift

**Feature drift** is the change over time in the statistical properties of a feature in production. Feature drift is distinct from model drift (model outputs changing) and is often the *cause* of model drift.

Types of feature drift:
- **Covariate shift**: The distribution of input features changes while the relationship between features and labels stays the same. Example: average user session count increases over time because the product grew.
- **Concept drift**: The relationship between features and labels changes. Example: "purchase intent" signals from pre-pandemic user behavior no longer predict purchases post-pandemic.

Monitoring feature drift in a feature store:
- Compute **PSI** or **KL divergence** between the current distribution of each feature and the training-time distribution.
- Track **feature null rates** and **out-of-range rates** (features outside the min/max seen in training).
- Alert when PSI > 0.2 for any high-importance feature.

Feature stores like **Tecton** and **Feast** include built-in feature monitoring that computes rolling statistics and alerts on drift.

---

## 7. Real Systems

### 7.1 Feast (Open Source)

Feast is the most widely used open-source feature store, originally developed by Gojek and now a CNCF project. Feast manages feature definitions (feature views), materializes features from offline sources (Parquet, BigQuery) to an online store (Redis, DynamoDB), and supports point-in-time correct training data generation.

Feast uses a two-stage architecture: a feature server (Python gRPC/HTTP service) backed by Redis for online serving, and a historical feature retrieval engine (Spark or Pandas) backed by Parquet/BigQuery for training.

### 7.2 Tecton

Tecton is a commercial feature platform (founded by the team that built Uber's Michelangelo). Tecton adds streaming features (Spark Streaming or Flink), automatic backfills, data quality monitoring, and a feature catalog with lineage tracking. Tecton's transformation layer allows features to be defined as Python transformations applied to raw data streams.

### 7.3 Hopsworks

Hopsworks is an open-source, self-hosted feature store with a broader ML platform footprint. It includes a Hive-based feature warehouse, an online feature store backed by RonDB (a MySQL cluster variant optimized for KV workloads), and native Python/Spark APIs. Hopsworks is commonly used in EU deployments where data residency requirements preclude cloud-only solutions.

### 7.4 Feature Lineage

**Feature lineage** is the provenance graph of a feature: the raw data sources it was computed from, the transformation code version, and the model versions that use it. Feature lineage is critical for:
- **Regulatory compliance**: Explain which data contributed to a decision.
- **Debugging**: When a model degrades, trace back which features changed and why.
- **Data deletion**: When a user requests data deletion (GDPR), identify all derived features computed from that user's data.

---

## Misconception: A feature store is just a database with features

A feature store is not just a database. The defining properties are: (1) **a unified definition layer** that ensures the same transformation logic is used offline and online, (2) **point-in-time correct retrieval** for training data generation, (3) **materialization pipelines** that keep online store values fresh, and (4) **feature versioning and lineage**. A Redis database storing features satisfies (1) and (3) at best; it does not provide PIT correctness or lineage. The distinction matters: teams that use ad-hoc databases as "feature stores" consistently produce models with training-serving skew.

## Misconception: Training-serving skew is always caused by feature computation differences

Training-serving skew has multiple sources beyond feature computation: different preprocessing normalization (different mean/std used in training vs serving), different tokenization versions, different input truncation behavior, or different default values for missing features. Feature stores address the feature computation consistency problem but not these other sources. Complete elimination of training-serving skew requires end-to-end test pipelines that validate that the serving path produces identical model inputs as the training pipeline for a set of reference examples.

## Misconception: Features with low importance scores can be ignored for freshness monitoring

Feature importance (from a feature importance analysis) measures a feature's impact on model performance on the training distribution. A feature can have low importance on the training distribution but become high-impact if it drifts significantly in production (because it shifts the model's predictions in unexpected ways). Always monitor drift for all features, not just high-importance ones. Conversely, a high-importance feature that is very stable (low drift historically) may need less frequent freshness updates than a low-importance but highly volatile feature.

## Misconception: Point-in-time correctness is only needed for event-based models

PIT correctness matters for any model trained on historical data where labels occur after the features are computed. Time-series forecasting, recommendation, fraud detection, churn prediction — all require PIT-correct features. Even "simple" models that are retrained monthly on batch data can have subtle PIT violations if the feature computation pipeline does not enforce temporal boundaries. PIT violations are insidious because the model trains successfully and evaluates well offline (since the leakage signals are strong) but degrades in production where the future information is not available.

## Misconception: Embeddings should be stored in the same online store as scalar features

Embeddings are 100-10,000× larger than scalar features per entity. Storing embeddings in the same Redis instance as scalar features will dominate memory usage and increase cache eviction pressure for frequently accessed scalar features. Moreover, embeddings are often used for ANN search, which requires specialized indexing data structures (HNSW, IVF-PQ) that Redis RediSearch provides in a limited way. Production systems separate scalar feature storage (Redis key-value) from embedding storage (dedicated vector database) and keep only the most critical embeddings in Redis when sub-millisecond retrieval is needed.

---

## 8. Practical Example: Feature Store for a Real-Time Recommendation System

A recommendation API serves 30,000 requests/second. Each request requires: user behavioral features (click rates, purchase counts, session statistics — updated hourly), user embedding (256-dim, updated daily), and item features (popularity stats, inventory status — updated every 5 minutes). The serving SLA is p99 < 50ms.

**Architecture**:
- **Redis** (online store): user behavioral features + item features. Single-key lookup per request: ~0.5ms.
- **Pinecone** (vector store): user embeddings for ANN retrieval (find similar users for collaborative filtering). ~5ms query.
- **Feature pipeline**: Flink for item features (5-min freshness), daily Spark for user embeddings.
- **Offline store**: Iceberg on S3, PIT-correct training data generation via Spark temporal join.
- **Monitoring**: Hourly PSI computation on 20 high-importance features; daily embedding drift check (mean cosine distance between today's and last week's embeddings).

Total feature retrieval time at serving: ~5.5ms (Redis 0.5ms + Pinecone 5ms). Well within 50ms SLA.

---

## 9. Exercise

**Exercise**: Design a feature store for a fraud detection model. The model uses: (1) user account age (static), (2) 30-day transaction count (updated daily), (3) 5-minute transaction velocity (updated in real time), (4) merchant category risk score (updated weekly), (5) user embedding (updated daily). For each feature: specify the required freshness, the computation approach (batch/streaming/real-time), the online store technology, and the estimated storage size per user (assume 100M users). Then design the PIT-correct training data generation pipeline for a dataset of 10M labeled transactions from the past year. What is the estimated Spark job runtime assuming 10M events × 5 features × 1 year of snapshots?

---

## References

- Feast documentation: https://docs.feast.dev
- Tecton feature platform: https://www.tecton.ai/blog/
- Hopsworks documentation: https://docs.hopsworks.ai
- "Feature Stores for ML" (Chip Huyen blog, 2020): https://huyenchip.com/2020/12/27/feature-stores.html
- Apache Iceberg time travel: https://iceberg.apache.org/docs/latest/spark-queries/#time-travel
- Uber Michelangelo paper: Hermann et al., "Meet Michelangelo: Uber's Machine Learning Platform" (2017)
- "Building Machine Learning Pipelines" (Hapke & Nelson, 2020) — Chapters on feature engineering and the training-serving skew
