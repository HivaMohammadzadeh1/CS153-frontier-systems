# Safe Rollout and Regression Testing for Serving Changes

## The risk
Serving changes — a new model version, a quantization scheme, a kernel/driver/engine upgrade, a prompt-template edit, a routing-policy change — can silently regress *quality* (worse answers) or *performance* (p99, throughput, cost) even when nothing errors. You need a safe path to ship them.

## An eval/regression harness
Before shipping, run the change against a fixed evaluation set and compare to the current production baseline on both axes:
- **Quality**: task accuracy / win-rate / LM-as-judge on a representative, contamination-free suite.
- **Performance & cost**: TTFT, TPOT, p95/p99, throughput, $/request.
Block the rollout if either regresses beyond a threshold.

## Progressive delivery
- **Canary**: route a small % of live traffic to the new version; watch quality and latency before ramping.
- **Shadow / mirror**: send a copy of real traffic to the candidate without serving its responses, to compare offline.
- **A/B**: measure downstream metrics with proper statistics.
- **Fast rollback**: keep the previous version warm so you can revert in seconds when a regression appears.

## Classic incidents
- Quality regression after a quantization or model swap → caught by the quality eval + rollback, not by error rates.
- p99 regression after a kernel/engine upgrade → caught by canary latency monitoring, not by unit tests.

## Interview-relevant reasoning
"How would you safely ship a quantization change?" Strong answer: offline quality+perf eval vs. baseline → canary with latency/quality guardrails → ramp → keep rollback ready. Never ship a serving change on vibes.
