# Serving Patterns: Canary, Shadow, Blue-Green, and Model Versioning

**Area C — Inference Infrastructure | Learning Memory OS Curriculum**

---

## 1. Why Deployment Patterns Matter

Deploying a new model version is one of the highest-risk operations in production ML. A model that performed well in offline evaluation may degrade real-user metrics due to distribution shift, calibration differences, latency regressions, or subtle behavioral changes that evals did not capture. The serving patterns in this topic — canary deployment, shadow traffic, blue-green deployment, and model versioning — are the engineering mechanisms that allow teams to release model updates with controlled, measurable risk.

The central principle is: **never route all traffic to an unvalidated model**. Every serving pattern described here enforces this principle in a different way.

---

## 2. Canary Deployment

A **canary deployment** routes a small fraction of production traffic (e.g., 1-5%) to the new model version while the remaining traffic continues to serve the current (stable) version. The name comes from the "canary in a coal mine" — a small group of users acts as an early warning system for problems.

### 2.1 Mechanics

The traffic split is controlled by the serving layer. Common implementations:

- **Load balancer weights**: Nginx or Envoy upstream weights direct X% of requests to the canary pool and (100-X)% to the stable pool.
- **Feature flags**: A request-level feature flag (e.g., based on user ID hash) assigns users to canary vs stable deterministically, allowing the same user to consistently get the same model during the rollout window.
- **Istio/service mesh**: Traffic policies in the service mesh allow percentage-based routing with header-based overrides for testing.

### 2.2 What to Monitor

During a canary rollout, the team monitors:

- **Latency (p50, p95, p99)**: A new model may be larger, use a different quantization level, or have different prefill characteristics that increase latency.
- **Error rates**: 5xx errors, timeout rates.
- **Business metrics**: Task completion rate, user satisfaction signals, downstream action rates (for recommendation models).
- **Model-specific metrics**: Output distribution similarity (e.g., KL divergence between canary and stable output logits), refusal rates (for safety models), perplexity on a held-out set.

If canary metrics are within acceptance bounds after a soak period (typically 24-72 hours), the rollout proceeds to 100%.

### 2.3 Rollback

The primary advantage of canary deployment is instant rollback: if the canary shows degraded metrics, return the canary weight to 0% in the load balancer. The stable model was never taken down, so the rollback is instantaneous with no user impact.

---

## 3. Shadow Traffic Testing

**Shadow traffic** (also called "dark launch" or "traffic mirroring") sends a copy of production requests to the new model in parallel with the current model, without using the new model's response. The production response always comes from the stable model; the new model processes requests as a side effect, and its outputs are logged but not served.

### 3.1 Use Cases

Shadow traffic is most useful when:

1. The new model is unvalidated and you need to observe its behavior on real production traffic patterns without any risk of serving degraded responses.
2. The new model has significantly different latency characteristics (e.g., a larger model that may time out on some request patterns) — you can measure tail latency under real load without impacting users.
3. You want to compare outputs of two models side-by-side on identical inputs for debugging or A/B analysis.

### 3.2 Infrastructure

Shadow traffic requires **request mirroring** at the proxy layer. Envoy's `mirror` filter creates an async copy of each request to the shadow cluster; the original request proceeds normally. The mirrored request's response is discarded at the proxy level (it is logged by the shadow service but never returned to the user).

The shadow cluster must be sized to handle the full production request rate asynchronously. Common cost optimization: use a lower-priority GPU pool for the shadow service, since its latency is not user-visible. On Kubernetes, this maps to running the shadow deployment on spot instances.

---

## 4. Blue-Green Deployment

**Blue-green deployment** maintains two complete, independent production environments: "blue" (currently serving) and "green" (new version, fully staged). The switch is instantaneous: a single DNS record or load balancer change redirects all traffic from blue to green.

### 4.1 Advantages

- **Zero downtime**: The transition is a single atomic routing change.
- **Full environment parity**: Green runs with full production traffic simulation during the staging window.
- **Instant rollback**: Revert the routing change to return to blue.

### 4.2 Cost

Blue-green requires double the infrastructure during the cutover period — both blue and green must be running simultaneously. For large LLM deployments (hundreds of H100s), this is expensive. Teams often run blue-green for short windows (< 24 hours) and use canary for extended rollouts to control cost.

### 4.3 Warm-Up Problem

Model serving has a **cold-start latency** problem: the first requests after starting a new deployment pod hit cache-cold KV caches and may trigger JIT compilation (for TorchScript or TorchInductor backends). Blue-green deployment solves this by warming up the green environment under real traffic (with shadow mirroring) before the switch, ensuring green is fully warm at cutover.

---

## 5. A/B Testing for Models

While canary deployment is used to validate that a new model is not *worse* than the current one, **A/B testing** is used to determine which of two model variants is *better* for a specific metric. The difference is statistical power: canary rollout exits when the new model is not significantly worse; A/B test runs until the confidence interval on the difference is small enough to declare a winner.

### 5.1 Assignment and Holdout

Users are randomly assigned to treatment (model B) or control (model A) based on a hash of their user ID and experiment ID. The hash ensures stability — the same user always gets the same model for the duration of the experiment. This is important because A/B tests measure user-level outcomes (e.g., 7-day retention), which are confounded if a user switches groups mid-experiment.

### 5.2 Statistical Significance

For small effect sizes (e.g., 1% improvement in task completion rate), A/B tests require large sample sizes. At typical production request rates, achieving 95% statistical power at 1% effect size may require 7-14 days. Teams use sequential testing or Bayesian A/B methods to allow early stopping when a clear winner emerges.

### 5.3 Segmented Analysis

Model behavior changes are rarely uniform across all users. A new model that improves average task completion may degrade performance for a specific user segment (e.g., non-English speakers, mobile users with short context windows). Production A/B tests always segment results by user attributes, request type, and query language.

---

## 6. Model Registry and Versioning

A **model registry** is the central catalog of model artifacts with their metadata, lineage, and deployment history. Every trained model checkpoint is registered with:

- **Version ID**: A unique identifier (often a content hash of the weights + config).
- **Training provenance**: Data version, training code commit, hyperparameters.
- **Evaluation results**: Offline benchmark scores, human eval results.
- **Deployment history**: Which environments (canary, production, shadow) the model has been deployed to, and when.
- **Tags**: `production`, `candidate`, `deprecated`, `rollback-target`.

### 6.1 Real Systems

**MLflow Model Registry** provides version tracking, stage transitions (Staging → Production → Archived), and lineage tracking. It integrates with MLflow's experiment tracking and artifact store.

**Seldon Deploy** adds serving-layer primitives: canary deployments, traffic splitting, and explainer sidecars, on top of Kubernetes.

**BentoML** provides a packaging format (Bento) that bundles model weights, inference code, dependencies, and metadata into a reproducible artifact. Bentos are stored in BentoML Cloud or a self-hosted Bento Store.

**KServe** (formerly KFServing) is a Kubernetes-native model serving platform with first-class support for canary rollouts, traffic splitting, and model versioning via `InferenceService` CRDs. KServe supports multiple serving runtimes (Triton, TorchServe, HuggingFace TGI) behind a unified API.

### 6.2 Rollback Strategy

A robust rollback strategy requires:

1. **Pinned previous version**: The registry must retain the artifact and deployment configuration of the model currently tagged `rollback-target`. Garbage collection policies must not delete it.
2. **Rollback runbook**: A documented, tested procedure for reverting the load balancer to the previous version. This should take < 5 minutes.
3. **Automated rollback triggers**: If canary metrics cross a predefined threshold (e.g., error rate > 2× baseline for > 5 minutes), the deployment system automatically pauses the rollout and pages the on-call engineer.

---

## 7. Traffic Shifting Patterns

Beyond simple percentage splits, production serving systems implement more nuanced traffic shifting strategies:

**Progressive rollout**: Increase canary percentage on a schedule — 1% → 5% → 25% → 50% → 100% — with automated checks between steps. Each step requires the canary to meet success criteria before proceeding.

**Request-type routing**: Route only a subset of request types to the new model. For example, a new model optimized for coding tasks might receive only requests classified as code-related while the stable model handles all other request types.

**Latency-aware routing**: Route only requests with loose latency SLAs to the new model if it is slower. Requests with strict latency requirements continue to hit the stable model.

**Geo-based rollout**: Roll out to one geographic region first, validate for 24 hours, then expand. This limits blast radius to a subset of users and makes rollback simpler (no global routing change needed).

---

## Misconception: Canary and A/B testing are the same thing

Canary deployment and A/B testing serve different purposes and require different infrastructure. Canary deployment is a **risk mitigation** mechanism: it routes a small percentage of traffic to a new model to detect regressions before full rollout. The goal is to exit quickly (< 72 hours) once confidence is established. A/B testing is an **experimentation** mechanism: it runs two variants simultaneously until statistical significance is achieved for a target metric. A/B tests typically run for days to weeks and require careful randomization to avoid confounding. The two can be combined (run an A/B test at low traffic while using canary deployment mechanics), but they are conceptually distinct.

## Misconception: Shadow traffic testing means your model is tested at 100% of production scale

Shadow traffic means your model processes 100% of production *requests*, but it is typically running on a smaller, lower-priority infrastructure footprint than the production model. The shadow service is not sized to serve 100% of production traffic with production latency SLAs — it is sized to keep up with request processing with a longer queue. If your shadow service falls behind (queue builds up), it is sampling from the request stream rather than processing every request. In practice, shadow testing validates request handling, output quality, and error rates, but not production-scale latency under full load.

## Misconception: Blue-green deployment eliminates downtime permanently

Blue-green deployment eliminates downtime *during the cutover*, but the green environment must still be started, warmed up, and validated before the switch. If a critical bug is discovered after the switch, rollback requires re-routing to blue — this is fast (seconds) but not zero-impact if session affinity is used (some in-flight requests may be interrupted). Additionally, blue-green does not help with data schema migration: if the new model requires a different request format, both blue and green must accept the old format (or the client must coordinate).

## Misconception: Model versioning is just storing model weights

Model versioning must capture *all* artifacts necessary to reproduce a deployment: weights, model config (architecture hyperparameters), tokenizer version, preprocessing code, serving code and dependencies, hardware config (quantization settings, tensor parallelism degree), and environment variables. A weight file without its tokenizer version, or a serving config without its quantization settings, is not reproducible. Teams that store only weights routinely discover they cannot recreate a deployment 6 months later.

## Misconception: Automated rollback is always safe

Automated rollback introduces its own risks. If an automated rollback triggers incorrectly (e.g., a monitoring metric spikes due to an unrelated infrastructure event), it creates unnecessary churn. If rollback is triggered mid-canary and the stable model was recently updated, the "rollback" may not actually revert to the last known-good state. Best practice: automated rollback pauses the rollout and pages on-call; full rollback to the previous model requires human approval in most production systems.

---

## 8. Practical Example: Canary Rollout with KServe

Deploying a new Llama-3-70B quantized model (INT4) replacing the current BF16 model, using KServe on a production K8s cluster:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-70b
spec:
  predictor:
    canaryTrafficPercent: 10
    model:
      modelFormat:
        name: huggingface
      storageUri: "s3://models/llama-70b-int4-v2"
    containers:
    - name: kserve-container
      resources:
        limits:
          nvidia.com/gpu: "4"
```

KServe automatically splits 10% of traffic to the new INT4 model and 90% to the existing stable model. Monitoring dashboard tracks p95 latency (expect INT4 to be ~2x faster), token quality (compare output distribution on a set of reference prompts), and refusal rate. After 48 hours with stable metrics, promote to 100% by setting `canaryTrafficPercent: 100`.

---

## 9. Exercise

**Exercise**: Design a deployment strategy for a safety classifier model that filters harmful content in a high-traffic API (10,000 requests/second). The new version improves recall on a held-out harmful content dataset by 15% but has 20% higher latency (18ms vs 15ms p99). Your SLA requires p99 < 20ms. Design a deployment plan that: (1) validates the latency claim under real traffic before full rollout, (2) measures the precision/recall tradeoff in production, (3) provides instant rollback, and (4) handles the cold-start latency spike at cutover. Specify which serving pattern(s) you use at each stage, the metrics you monitor, and the success criteria for each stage.

---

## References

- Seldon Deploy: https://docs.seldon.io/projects/seldon-deploy/
- BentoML documentation: https://docs.bentoml.com
- KServe: https://kserve.github.io/website/
- Envoy traffic mirroring: https://www.envoyproxy.io/docs/envoy/latest/api-v3/config/route/v3/route_components.proto#envoy-v3-api-msg-config-route-v3-routeaction-requestmirrorpolicy
- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
- Progressive delivery concepts: the "Progressive Delivery" chapter in the SRE Book (Google, 2022 edition)
