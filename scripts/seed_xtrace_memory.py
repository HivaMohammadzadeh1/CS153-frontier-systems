"""Seed synthetic learning history into XTrace for a demo student.

Sends a handful of multi-turn conversations covering different ML systems
topics. XTrace's server-side extractor turns each into an Episode (and ideally
some Facts). After seeding, the Profile tab in the Streamlit app should show
real "Past sessions" instead of meta-summaries of empty exchanges.

Run:
    uv run python scripts/seed_xtrace_memory.py            # seeds demo-user
    uv run python scripts/seed_xtrace_memory.py alice      # seeds 'alice'
"""

from __future__ import annotations

import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

import httpx

from learning_memory_os.config import get_settings


SESSIONS = [
    {
        "topic": "Tokenization (BPE, byte-level)",
        "conv_id": "seed_tokenization_01",
        "messages": [
            ("user", "Can you explain BPE merges intuitively? I always get confused about the merge order."),
            ("assistant", "BPE starts with characters and greedily merges the most frequent adjacent pair, iteratively. There's no semantic ranking — every choice is pure frequency. Each merge becomes a new token in the vocab and the algorithm repeats until you hit your vocab cap."),
            ("user", "So if 'the' appears most often, it'd be merged before 'th'?"),
            ("assistant", "Almost — 'th' would actually merge first because the bigram 'th' is much more frequent than the trigram 'the'. After 'th' becomes a symbol, the bigram 'th' + 'e' is then the most frequent, and that merge produces 'the'."),
            ("user", "Got it. So vocab size is the main knob — larger vocab means more merges and shorter sequences."),
            ("assistant", "Exactly. Larger vocab = longer 'token chunks' on average, shorter sequence length, but higher embedding-table cost. Modern LLMs sit around 30k–100k vocab — a sweet spot between sequence efficiency and embedding cost."),
        ],
    },
    {
        "topic": "KV cache eviction (paged attention)",
        "conv_id": "seed_kvcache_01",
        "messages": [
            ("user", "I'm implementing a paged KV cache for a 13B model. How big should the page size be?"),
            ("assistant", "16 tokens per page is the common default — that's what PagedAttention (vLLM) ships with. Smaller pages give more allocation flexibility but more page-table overhead per request."),
            ("user", "And how do I handle GQA — do attention heads share pages?"),
            ("assistant", "With GQA, query heads in a group share the same K/V cache, so each group has one set of pages per layer. Llama-2-70B uses 8x GQA, meaning 8 query heads share each K/V head and therefore each set of pages."),
            ("user", "What if my sequence length exceeds the budget? How should I evict?"),
            ("assistant", "Three eviction families: (1) H2O — keep tokens with high cumulative attention mass; (2) StreamingLLM — keep first N attention sinks plus last K recent tokens; (3) SnapKV — cluster recent attention patterns and keep representatives. H2O works well for chat, StreamingLLM for very long contexts."),
            ("user", "Right — so the choice depends on the access pattern of my workload."),
        ],
    },
    {
        "topic": "Distributed training (DDP)",
        "conv_id": "seed_ddp_01",
        "messages": [
            ("user", "I'm training a 1B model on 8 H100s with PyTorch DDP. My loss diverges after epoch 3. Any ideas?"),
            ("assistant", "Classic causes: (1) learning rate not scaled for global batch size, (2) gradient sync race condition with autocast mixed precision, (3) unsynced random state on workers, (4) gradient explosion that clipping isn't catching. What's your LR scaling rule?"),
            ("user", "Linear scaling — base_lr * world_size. I also have gradient clipping at 1.0."),
            ("assistant", "Linear scaling with grad clip 1.0 should be safe. Try logging gradient norms per layer — if some layers spike to 100+ before clipping, the clip is artificially flattening them and you may want per-parameter-group clipping instead. Also verify that DataLoader's worker seeds are deterministic across ranks."),
            ("user", "I'll add gradient-norm logging. Thanks."),
        ],
    },
    {
        "topic": "Quantization (INT8 calibration)",
        "conv_id": "seed_quant_01",
        "messages": [
            ("user", "I thought INT8 quantization significantly hurts model quality. Is that wrong?"),
            ("assistant", "Mostly wrong with caveats. With proper per-channel weight quantization and percentile-based activation clipping, modern INT8 PTQ achieves <0.5% perplexity degradation on most LLMs. The 'quantization hurts quality' belief comes from naive INT8 conversion that uses the same scale for an entire tensor."),
            ("user", "What does 'proper calibration' look like in practice?"),
            ("assistant", "Run inference on ~512 representative inputs to record activation distribution statistics. Pick per-tensor scale factors that minimize quantization error — often 99.9 percentile clipping rather than min/max. Frameworks like AWQ and SmoothQuant automate this and add weight equalization to make activation quantization easier."),
            ("user", "So basically a small calibration pass before deployment. That's much cheaper than I thought."),
            ("assistant", "Right — usually under 5 minutes on a single GPU for a 7B model. The myth that quantization is expensive comes from QAT (quantization-aware training), which is a different beast that retrains the model."),
        ],
    },
    {
        "topic": "Inference latency vs throughput",
        "conv_id": "seed_latency_01",
        "messages": [
            ("user", "I keep mixing up latency and throughput in inference. Help me get them straight."),
            ("assistant", "Latency = time per request. Throughput = requests per second across all in-flight work. They trade off: larger batches improve throughput (amortize fixed costs) but hurt per-request latency (you wait for the batch). Continuous batching (vLLM) tries to get the best of both by mixing requests of different lengths into the same forward pass."),
            ("user", "So if I'm serving a chat app where users wait, I should optimize for latency, not throughput?"),
            ("assistant", "Yes, with a caveat: TTFT (time to first token) matters more than total latency for chat UX. Streaming the first token while continuing to generate hides total latency. You'd still benefit from continuous batching because it boosts throughput without much TTFT cost."),
        ],
    },
]


def main() -> int:
    s = get_settings()
    if not s.xtrace_api_key or not s.xtrace_org_id:
        print("FAIL: XTRACE_API_KEY or XTRACE_ORG_ID missing from .env")
        return 1

    student_id = sys.argv[1] if len(sys.argv) > 1 else "demo-user"

    print(f"Seeding XTrace memory for '{student_id}' across {len(SESSIONS)} sessions:\n")

    client = httpx.Client(base_url=s.xtrace_base_url, timeout=30.0)
    headers = {"x-api-key": s.xtrace_api_key, "x-org-id": s.xtrace_org_id}

    sent_ok = 0
    for i, session in enumerate(SESSIONS, 1):
        msgs = [
            {
                "role": role,
                "content": content,
                "date": datetime.now(tz=timezone.utc).isoformat(),
                "dia_id": f"turn_{uuid.uuid4().hex[:10]}",
            }
            for role, content in session["messages"]
        ]
        body = {
            "messages": msgs,
            "user_id": student_id,
            "conv_id": session["conv_id"],
        }
        print(f"  [{i}/{len(SESSIONS)}] {session['topic']:<48s}  {len(msgs)} turns", end="")
        try:
            r = client.post("/v1/memories", headers=headers, json=body)
            if r.status_code >= 400:
                print(f"  FAIL {r.status_code}: {r.text[:160]}")
            else:
                print(f"  ok ({r.status_code})")
                sent_ok += 1
        except httpx.HTTPError as exc:
            print(f"  network error: {exc}")

    if sent_ok == 0:
        print("\nNothing was ingested. Check credentials.")
        return 1

    print(f"\nSent {sent_ok}/{len(SESSIONS)} sessions. Waiting 25s for the extractor...")
    time.sleep(25)

    print("\nVerifying — listing memories now:")
    try:
        r = client.post(
            "/v1/memories/search",
            headers=headers,
            json={
                "query": "memory",
                "filters": {"user_id": student_id},
                "limit": 100,
            },
        )
        data = r.json().get("data", [])
    except httpx.HTTPError as exc:
        print(f"  recall failed: {exc}")
        return 2

    print(f"  total stored for {student_id}: {len(data)}")
    print(f"  by kind: {dict(Counter(m['type'] for m in data))}")
    print()
    for m in data[:8]:
        text = m["text"].replace("\n", " ")[:160]
        print(f"  [{m['type']}] {text}{'...' if len(m['text']) > 160 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
