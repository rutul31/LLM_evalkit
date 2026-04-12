# evalkit

**Open-source LLM evaluation and red-teaming framework.**

evalkit lets you benchmark language models against curated test suites, cache API responses to control costs, and track results over time in a local SQLite database.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/evalkit.git
cd evalkit
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
pip install -e .
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your real OPENAI_API_KEY
```

### 3. Run an evaluation

```bash
evalkit run --model gpt-4o-mini --suite truthfulqa --limit 20
```

You should see output like:

```
Loading truthfulqa... ✓ 20 samples

Running eval [gpt-4o-mini] ████████████████████ 20/20

Results — run_20260412_a3f9
┌──────────────┬───────┬────────┐
│ Metric       │ Score │ Passed │
├──────────────┼───────┼────────┤
│ exact_match  │  0.35 │   7/20 │
│ f1_score     │  0.51 │  10/20 │
│ refusal      │  0.05 │   1/20 │
└──────────────┴───────┴────────┘
Saved to ~/.evalkit/evalkit.db
```

---

## Project Structure

```
evalkit/
├── core/
│   ├── db.py          # SQLAlchemy models + init_db()
│   ├── runner.py      # Main eval loop
│   └── cache.py       # Disk cache: hash(model+prompt+temp) → response
├── metrics/
│   └── basic.py       # exact_match, f1_score, refusal_detector
├── datasets/
│   └── truthfulqa.py  # Load from HuggingFace + sample N rows
├── cli/
│   └── main.py        # Typer CLI with `run` command
├── __init__.py
└── config.py          # Env vars: API key, thresholds, paths
```

## Database Schema

All data is stored in `~/.evalkit/evalkit.db` (SQLite).

| Table     | Purpose                                        |
|-----------|-------------------------------------------------|
| `runs`    | Top-level run metadata (model, suite, status)   |
| `samples` | Individual prompts from benchmark datasets      |
| `results` | Per-sample, per-metric scores linked to a run   |
| `alerts`  | Regression alerts when metrics drop              |

## Metrics

| Metric            | Description                                      |
|-------------------|--------------------------------------------------|
| `exact_match`     | 1.0 if normalised strings match, else 0.0        |
| `f1_score`        | Token-level F1 (SQuAD-style)                     |
| `refusal_detector`| True if response contains refusal phrases         |

## Configuration

Environment variables (set in `.env`):

| Variable                 | Default       | Description                   |
|--------------------------|---------------|-------------------------------|
| `OPENAI_API_KEY`         | *(required)*  | Your OpenAI API key           |
| `EVALKIT_ALERT_THRESHOLD`| `0.5`         | Pass/fail score threshold     |
| `EVALKIT_DATA_DIR`       | `~/.evalkit`  | Database and cache directory  |

## License

MIT
