# Learning Memory OS

A context-routed tutor for ML systems engineers. CS 153 final project.

See `docs/superpowers/specs/2026-05-12-learning-memory-os-design.md` for the design spec.

## Dev setup
1. `cp .env.example .env` and fill in API keys
2. `docker compose up -d db`
3. `uv sync`
4. `uv run pytest`

## Demo app

Launch the Streamlit demo:
```bash
uv run streamlit run scripts/app.py
```

Then open `http://localhost:8501`. Features:
- Multi-turn chat with the tutor
- Observable routing: per-turn selected items, dropped items, score breakdowns
- Topic focus, token budget, student-mastery state visible in the sidebar

## Web app (production-style)

Launch:
```bash
uv run python -m scripts.serve --port 8000
```

Then open http://localhost:8000.

This is the polished frontend (FastAPI + HTML/JS/Tailwind). The Streamlit demo at `scripts/app.py` is still available for backend developers but the recommended UI is this one.
