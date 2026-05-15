# Learning Memory OS — System Architecture

```mermaid
flowchart TB
  subgraph Ingestion
    Lectures["CS336 / CS349D / CS153<br>lectures + papers + blogs"]
    Extractor["ArtifactExtractor<br>(Claude Opus → JSON)"]
    Lectures --> Extractor
  end

  subgraph Memory["Multi-tier Memory (Postgres + pgvector)"]
    direction LR
    Semantic[(Semantic<br>course facts)]
    Student[(Student<br>mastery + misconceptions)]
    Episodic[(Episodic<br>session events)]
    Intervention[(Intervention<br>strategy log)]
  end

  Extractor --> Semantic

  subgraph Selector["Context Routing Engine"]
    Score["Heuristic scorer<br>relevance + recency +<br>misconception + prereq + reuse"]
    Pack["Budgeted packer<br>greedy top-K under tokens"]
    Score --> Pack
  end

  Semantic --> Selector
  Student --> Selector
  Episodic --> Selector
  Intervention --> Selector

  subgraph Agents["Specialist Agents"]
    Tutor["Tutor"]
    Diagnostic["Diagnostic"]
    QuizGen["Quiz generator"]
    LabGen["Lab generator"]
  end

  Selector --> Agents
  Agents -->|"call"| LLM["Claude Opus<br>(generation)"]
  Agents -->|"log every<br>decision"| JSONL[("JSONL<br>interaction log")]
  JSONL -->|"trains"| Router["Phase 3:<br>LoRA fine-tuned<br>context router"]
  Router -.->|"replaces<br>heuristic"| Selector

  classDef phase3 fill:#e6f3ff,stroke:#1f6feb
  class Router phase3
```

## Layers

1. **Ingestion** — Heterogeneous source material (Stanford CS336/CS349D/CS153 lectures, canonical papers, blogs) is converted into structured Pydantic artifacts via a strong-LLM-driven extractor.
2. **Multi-tier memory** — Postgres + pgvector, four tiers:
   - *Semantic*: stable course/topic facts
   - *Student*: per-student mastery state + active misconceptions
   - *Episodic*: append-only events (questions, replies, quiz attempts)
   - *Intervention*: log of which tutoring strategy was used + its outcome
3. **Context routing engine** — Three phases:
   - *Phase 1 (MVP)*: heuristic ranker (relevance + recency + misconception priority + prerequisite + reuse) + budgeted packer
   - *Phase 2*: combinatorial selector (knapsack with redundancy penalty + dependency bonus)
   - *Phase 3 (resume bullet)*: LoRA fine-tuned small open model trained on synthetic trajectories
4. **Specialist agents** — Each agent calls the routing engine for context; none receive raw global state. Tutor, Diagnostic, Quiz/Lab generators, Critic.
5. **Interaction log** — JSONL of every routing decision + agent output. Evaluation-ready; also the data flywheel for Phase 3 training signal.

## The recursion

Area E of the curriculum (`agent_memory`, `context_selection`, `multi_agent_orchestration`) teaches exactly the techniques the system uses to build itself. The author is student-zero; the system trains the engineer in the engineering of itself.
