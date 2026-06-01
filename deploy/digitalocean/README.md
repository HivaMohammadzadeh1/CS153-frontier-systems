# Serve the fine-tuned 7B router on a DigitalOcean GPU Droplet (vLLM)

Stands up an OpenAI-compatible inference endpoint for the fine-tuned 7B context
router on a DigitalOcean GPU Droplet. `bootstrap.sh` merges the LoRA adapter into
the base model and serves it with vLLM as a systemd service on port 8000.

> The 7B (0.81 Jaccard) is the model worth serving. The smaller adapters work the
> same way — set `BASE`/`ADAPTER` accordingly.

## Prerequisites

- A DigitalOcean account with **GPU Droplets enabled** (request access if needed).
- [`doctl`](https://docs.digitalocean.com/reference/doctl/how-to/install/) authenticated: `doctl auth init`.
- An SSH key uploaded to DO: `doctl compute ssh-key list` (note its ID/fingerprint).
- A **HuggingFace token** with read access to the private adapter repo
  `hivamoh/lmos-router-qwen2_5_7b` (the base `Qwen/Qwen2.5-7B-Instruct` is public).

## 1. Create the GPU Droplet

GPU Droplets use the **AI/ML Ready** image (NVIDIA driver + CUDA preinstalled). Check
current GPU sizes/regions:

```bash
doctl compute size list | grep gpu          # e.g. gpu-h100x1-80gb
doctl compute image list-application | grep -i gpu   # AI/ML Ready image slug
```

A single H100 (80 GB) is plenty for the 7B. Create it (substitute the slugs/region
and your SSH key id):

```bash
doctl compute droplet create lmos-router \
  --size gpu-h100x1-80gb \
  --image gpu-h100x1-base \
  --region nyc2 \
  --ssh-keys <YOUR_SSH_KEY_ID> \
  --wait
IP=$(doctl compute droplet get lmos-router --format PublicIPv4 --no-header)
echo "droplet IP: $IP"
```

## 2. Run the bootstrap on the droplet

```bash
scp deploy/digitalocean/bootstrap.sh root@$IP:/root/
ssh root@$IP "HF_TOKEN=hf_xxx VLLM_API_KEY=choose-a-secret bash /root/bootstrap.sh"
```

It merges the adapter (~2 min) and starts vLLM (first load ~1–2 min). Watch it:

```bash
ssh root@$IP "journalctl -u lmos-router -f"
```

## 3. Lock down the port

vLLM has no built-in auth. Either set `VLLM_API_KEY` (above — requires a bearer token)
**and/or** restrict :8000 with a firewall to only your app's egress IPs:

```bash
doctl compute firewall create --name lmos-router-fw \
  --inbound-rules "protocol:tcp,ports:8000,address:<YOUR_APP_IP>/32 protocol:tcp,ports:22,address:0.0.0.0/0" \
  --droplet-ids <DROPLET_ID>
```

## 4. Test the endpoint

```bash
curl http://$IP:8000/v1/models -H "Authorization: Bearer choose-a-secret"

curl http://$IP:8000/v1/chat/completions \
  -H "Authorization: Bearer choose-a-secret" -H "Content-Type: application/json" \
  -d '{"model":"lmos-router-7b","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":8}'
```

## 5. Point the product at it

The product's router seam (`src/learning_memory_os/router/product_adapter.py`) currently
loads adapters in-process. To use this remote endpoint instead, add a small "remote API"
backend that POSTs the router prompt to `http://$IP:8000/v1/chat/completions` and parses
the comma-separated ids (same `format_router_input` / `parse_router_output` as
`router/prompt.py`). Set, e.g.:

```bash
LMOS_ROUTER_ENDPOINT=http://$IP:8000/v1   LMOS_ROUTER_API_KEY=choose-a-secret
```

(That remote backend is a follow-up — ask and I'll wire it in so the chat's
"Router" dropdown can target the hosted 7B.)

## Cost note

A GPU Droplet bills **hourly while it exists** (no scale-to-zero). For bursty/occasional
routing, destroy it when idle (`doctl compute droplet delete lmos-router`) or use the
Modal scale-to-zero variant instead.

## Alternative: skip the merge, serve base + LoRA

vLLM can serve the adapter without merging:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
  --lora-modules lmos-router=hivamoh/lmos-router-qwen2_5_7b \
  --host 0.0.0.0 --port 8000
```

Merging (what `bootstrap.sh` does) gives a single self-contained model and slightly
lower per-request overhead; the LoRA route is handy for hot-swapping multiple adapters.
