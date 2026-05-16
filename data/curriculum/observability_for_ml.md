# Observability for ML: Logging, Tracing, Metrics, and Drift Detection

**Area F — Production ML Systems | Learning Memory OS Curriculum**

---

## 1. The Observability Gap in ML Systems

Traditional software observability answers: "Is the service up? Is it slow? Which endpoint is failing?" ML systems have all of those questions *plus* a harder class: "Is the model correct? Has the input distribution shifted? Is the model's output quality degrading silently?" A model can serve 200 OK responses at 15ms p99 latency while producing increasingly wrong outputs — and standard infrastructure monitoring will not detect it.

Observability for ML has three layers:
1. **System observability**: CPU/GPU utilization, memory usage, latency, error rates — the same as any service.
2. **Model observability**: Input/output distributions, prediction confidence, output diversity, latency by request type.
3. **Data drift and quality monitoring**: Statistical comparison between production inputs and training distribution; model performance monitoring on labeled subsets.

All three layers are necessary. A failure in any one layer can cause production incidents that are invisible to the other two.

---

## 2. Structured Logging for ML Pipelines

Unstructured logs (`print("prediction:", output)`) are useless at scale. Production ML systems use **structured logging** — every log entry is a JSON document with typed fields that can be indexed, queried, and aggregated.

### 2.1 What to Log

For an LLM serving endpoint, every request should log:

```json
{
  "timestamp": "2025-01-15T10:23:47.123Z",
  "request_id": "req_abc123",
  "model_version": "llama3-70b-int4-v2",
  "input_token_count": 512,
  "output_token_count": 128,
  "latency_ms": 185,
  "ttft_ms": 45,
  "finish_reason": "stop",
  "routing_decision": "fast_path",
  "sampled_for_review": false
}
```

Logging the full input and output is expensive at scale (1M requests/day × 2KB average = 2 TB/day of logs) and raises privacy concerns. Instead, sample a fraction (e.g., 0.1-1%) of requests for full I/O logging; log only metadata for the rest.

### 2.2 Logging Infrastructure

At production ML scale, logs flow through:
- **Instrumented service** → structured log emission (Python `structlog`, Go `zap`)
- **Log collector** (Fluentd, Vector) → **Message queue** (Kafka, Pub/Sub) → **Storage** (BigQuery, Snowflake, S3)
- **Query layer** (BigQuery, Athena) for analysis; **dashboards** (Grafana, Looker) for visualization

For real-time anomaly detection, logs can be processed as a stream (Flink, Dataflow) with alerting when metrics cross thresholds.

---

## 3. OpenTelemetry Tracing for ML Pipelines

**OpenTelemetry (OTel)** is the CNCF-standard observability framework for distributed tracing, metrics, and logs. For ML pipelines, tracing is especially valuable because a single user request may touch many components: a routing layer, a preprocessing service, an LLM API call, a retrieval service (RAG), and a postprocessing layer.

### 3.1 Trace Structure

A trace is a tree of **spans**, each representing a unit of work. A span has:
- `trace_id`: Propagated across service boundaries to correlate all spans in a request.
- `span_id`: Unique to this span.
- `operation_name`: The name of the operation (e.g., `"llm.generate"`, `"retrieval.search"`).
- `start_time`, `end_time`: Wall clock timestamps.
- `attributes`: Key-value pairs for domain-specific metadata.
- `status`: Success or error.

For an LLM with RAG, the trace looks like:
```
request [200ms]
├── auth.verify [1ms]
├── retrieval.search [45ms]
│   ├── embedding.encode [10ms]
│   └── vector.search [35ms]
├── llm.generate [140ms]
│   ├── prefill [42ms]
│   └── decode [98ms]
└── response.format [2ms]
```

This trace immediately shows that retrieval is 22.5% of total latency — a candidate for optimization.

### 3.2 OTel Instrumentation

Python instrumentation:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer("llm-service")

with tracer.start_as_current_span("llm.generate") as span:
    span.set_attribute("model.version", "llama3-70b-v2")
    span.set_attribute("input.token_count", token_count)
    output = model.generate(input_ids)
    span.set_attribute("output.token_count", len(output[0]))
```

Traces are exported to backends like **Jaeger**, **Tempo** (Grafana), **Cloud Trace** (Google), or **Honeycomb**.

---

## 4. Metrics: Latency, Throughput, and Error Rate

ML serving metrics fall into three categories:

### 4.1 Latency Metrics

- **TTFT (Time to First Token)**: The latency from request arrival to the first generated token. Critical for interactive UX. Dominated by prefill time and queue wait time.
- **TPOT (Time Per Output Token)**: The average time per output token after the first. Dominated by decode speed and KV cache bandwidth.
- **E2E latency**: Total time from request to complete response. = TTFT + (output_length - 1) × TPOT.

Report latency as percentiles (p50, p95, p99), not averages. The p99 latency is the tail experience; averages hide outliers that dominate user-facing SLA violations.

### 4.2 Throughput Metrics

- **Requests per second (RPS)**: System-level throughput.
- **Tokens per second (TPS)**: More meaningful for LLMs because request cost varies with token count.
- **GPU utilization**: Fraction of time the GPU is actively executing kernels (ideally > 80%). Low GPU utilization indicates scheduling overhead or memory bandwidth bottlenecks.
- **MFU (Model FLOP Utilization)**: Actual FLOPs per second / theoretical peak FLOPs per second. For H100 with BF16, peak is ~1,979 TFLOPS. MFU > 50% is good; < 30% indicates efficiency problems.

### 4.3 Error Rate

- **HTTP 5xx rate**: Infrastructure failures.
- **Timeout rate**: Requests that exceeded the latency SLA.
- **Safety filter trigger rate**: For LLMs with safety filters, the rate of requests that were blocked. A sudden change in this rate may indicate an adversarial attack or a shift in input distribution.

---

## 5. Model Monitoring vs System Monitoring

**System monitoring** tracks whether the infrastructure is healthy: GPU utilization, pod restarts, memory OOM events. This is Prometheus + Grafana — standard SRE practice, not ML-specific.

**Model monitoring** tracks whether the model's outputs are still good: output quality, calibration, and behavior. This requires ML-specific signals:

- **Output confidence / entropy**: For classification models, monitor the average entropy of the output distribution. A spike in entropy may indicate distribution shift.
- **Output length distribution**: For generative models, monitor the distribution of output lengths. A shift (e.g., the model starts producing much longer or shorter outputs) may indicate a behavioral change.
- **Semantic similarity to reference outputs**: Sample requests with known good responses and compare cosine similarity between model outputs and reference responses.
- **Business outcomes**: If downstream business metrics (conversion rate, user retention) can be linked to model outputs, these are the most reliable model quality signals.

---

## 6. Drift Detection

**Drift** occurs when the production input distribution diverges from the training distribution, causing model performance to degrade silently.

### 6.1 PSI (Population Stability Index)

PSI measures the shift in a single feature's distribution between a baseline (training) and current (production) distribution:

```
PSI = Σ (p_i - q_i) * ln(p_i / q_i)
```

where `p_i` is the fraction of samples in bin `i` for the baseline, and `q_i` is the fraction for the current distribution. PSI thresholds: < 0.1 (no significant drift), 0.1-0.25 (moderate drift, investigate), > 0.25 (significant drift, take action).

PSI is commonly used for tabular features in recommendation and classification models. For LLMs, PSI can be applied to input token length distributions, input language distributions, or embedding statistics.

### 6.2 KL Divergence

**KL divergence** (Kullback-Leibler divergence) measures the information-theoretic distance between two distributions:

```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```

KL divergence is the theoretical basis for PSI. Unlike PSI, it is asymmetric: `KL(P || Q) ≠ KL(Q || P)`. For drift detection, Jensen-Shannon divergence (the symmetric average of KL in both directions) is often preferred.

For LLM output monitoring, compute the KL divergence between the distribution of output embeddings (projected to lower dimensions with PCA) over a rolling window vs the training period. A sustained increase in KL divergence indicates output distribution shift.

### 6.3 Shadow Evaluation

**Shadow evaluation** continuously evaluates a sample of production requests against a ground truth oracle (human raters or a stronger model). In production LLM systems, this often means:
- Sample 0.1-1% of production requests.
- Send them to a strong evaluation model (e.g., GPT-4 or Claude) with a rubric.
- Track the distribution of quality scores over time.

This is expensive (each evaluation costs ~0.01-0.10) but provides direct quality signal. Teams typically maintain a fixed budget for shadow evaluation and prioritize sampling requests from high-value user segments or request types where regressions would have the highest impact.

---

## 7. Real Systems

### 7.1 Arize AI

Arize is a commercial model observability platform. It ingests predictions, actual labels (when available), and features at inference time, then provides drift detection, performance monitoring, and embedding visualization. Arize's embedding visualizer (UMAP-projected) is particularly useful for understanding input distribution shift in LLM applications.

### 7.2 WhyLabs

WhyLabs (now Whylogen) is an open-source + commercial observability platform built around **WhyLogs**, a data logging library that computes approximate statistical summaries (sketches) of data distributions at low cost. WhyLogs uses CountMin sketches for frequency estimation and KLL sketches for quantile estimation, enabling drift detection with sub-MB summary sizes even for billion-row datasets.

### 7.3 Evidently AI

Evidently is an open-source Python library for model monitoring reports and tests. It generates HTML reports comparing two datasets (e.g., training vs production) across dozens of statistical tests and visualizations. Evidently integrates with standard ML pipelines (MLflow, Prefect, Airflow) and is commonly used for batch monitoring in production ML pipelines.

---

## Misconception: Logging every request input and output is necessary for good monitoring

Full request logging is expensive, slow (adds latency), and creates privacy risks. Most production ML systems log only metadata for the vast majority of requests (latency, token counts, model version, finish reason) and sample a small percentage (0.1-1%) for full I/O logging. This sampling approach provides sufficient statistical power for drift detection, output quality monitoring, and debugging while keeping storage costs manageable. Sampling should be stratified — oversample high-value requests, long-tail users, or requests that trigger unusual model behavior.

## Misconception: Drift detection is only necessary for tabular ML models

Distribution shift affects all ML systems, including LLMs. For LLMs, drift manifests as shifts in: input query distribution (users start asking different types of questions), input language distribution, output quality (if the model degrades due to distributional mismatch), and safety trigger rate. The detection methods differ — statistical tests on token distributions, embedding drift, or output quality proxies — but the underlying problem (production distribution differs from training distribution) is universal.

## Misconception: Monitoring p95 latency is sufficient for LLM SLAs

p95 latency captures 95% of requests but misses the most impactful tail. For interactive LLM applications, the p99.9 (1 in 1000) request determines whether users perceive the service as reliable. An LLM with p95 = 500ms and p99.9 = 10s will generate complaints and support tickets even though "95% of requests are fast." Always monitor and set SLAs at the p99 or p99.9 level for user-facing LLM services. Additionally, TTFT matters as much as total latency for interactive chat: a user waiting for the first token perceives the service as slow even if the full response arrives quickly.

## Misconception: KL divergence is symmetric

KL divergence is NOT symmetric: `KL(P || Q) ≠ KL(Q || P)`. This asymmetry matters for drift detection: `KL(production || training)` measures the cost of approximating the production distribution with the training distribution (relevant for understanding model failure), while `KL(training || production)` measures the opposite. For most drift monitoring purposes, use Jensen-Shannon divergence `JSD(P, Q) = (KL(P||M) + KL(Q||M))/2` where `M = (P+Q)/2`, which is symmetric and bounded [0, log(2)].

## Misconception: A model with stable system metrics (latency, throughput) needs no model monitoring

System metrics confirm the infrastructure is healthy, not the model. A model can produce incorrect, biased, or harmful outputs at perfect 99.9% uptime and 10ms p99 latency. This is especially dangerous for LLMs where "correctness" is hard to define and verify automatically. The 2023-2024 wave of LLM production incidents — model behavior changes after fine-tuning, instruction following degradation, increased hallucination rates — were all invisible to system monitoring. Model monitoring (output quality proxies, shadow evaluation, drift detection) is a separate and non-negotiable layer.

---

## 8. Practical Example: Monitoring a RAG-based LLM Service

A production customer support chatbot uses RAG (retrieval + LLM). The monitoring setup:

1. **System metrics**: Prometheus + Grafana. Dashboards for TTFT, TPOT, RPS, GPU utilization, retrieval latency.
2. **Structured logs**: Every request logs `request_id`, `user_segment`, `retrieved_doc_ids`, `input_token_count`, `output_token_count`, `finish_reason`, `routing_label`. Stored in BigQuery.
3. **Drift detection**: Daily batch job computes PSI on input query embeddings (256-dim PCA projection) vs baseline week. Alert if PSI > 0.2.
4. **Shadow evaluation**: 0.5% of requests sampled; their responses evaluated by a GPT-4 judge against a 5-point rubric (helpfulness, accuracy, safety). Daily dashboard tracks score distribution.
5. **Business metric correlation**: Weekly analysis correlates chatbot quality scores with CSAT (customer satisfaction) ratings. Establishes that a drop in quality score > 0.3 points predicts a statistically significant CSAT decrease within 7 days.

---

## 9. Exercise

**Exercise**: Instrument a FastAPI LLM serving endpoint with OpenTelemetry tracing and Prometheus metrics. Your instrumentation should: (1) create a span for each request that records `model_version`, `input_token_count`, and `output_token_count` as span attributes; (2) expose a `llm_request_latency_seconds` histogram metric with labels `model_version` and `finish_reason`; (3) implement a sampling-based input logger that writes 1% of full request/response pairs to a Postgres table; (4) implement a PSI-based drift detector that computes weekly PSI on input token length distribution against a baseline week and writes results to a monitoring table. The system should add no more than 5ms to p99 request latency.

---

## References

- OpenTelemetry documentation: https://opentelemetry.io/docs/
- Evidently AI: https://docs.evidentlyai.com
- WhyLogs / Whylogen: https://github.com/whylogs/whylogs
- Arize AI model observability: https://arize.com
- "Instrumentation of Production ML Systems" chapter in Chip Huyen's "Designing Machine Learning Systems" (2022)
- PSI reference: commonly attributed to industry practice, see e.g., "Population Stability Index (PSI) Explained" articles in the actuarial/credit-risk literature
- Prometheus documentation: https://prometheus.io/docs/
