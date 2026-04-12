"""
evalkit.core.runner — Main evaluation loop.

Orchestrates a complete evaluation run:

1. Load samples from the requested benchmark suite.
2. For each sample, call the LLM (with disk caching).
3. Score with every registered metric.
4. Persist results to the SQLite database.
5. Return an aggregated summary dict for display.

The runner is intentionally synchronous — clarity over concurrency for
the initial release.
"""

import subprocess
import uuid
from typing import Any

from openai import OpenAI
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from evalkit.config import OPENAI_API_KEY, ALERT_THRESHOLD
from evalkit.core import cache
from evalkit.core.db import Run, Sample, Result, Alert, get_session, init_db
from evalkit.metrics.basic import exact_match, f1_score, refusal_detector


# ── Helpers ───────────────────────────────────────────────────────────


def _get_git_sha() -> str | None:
    """Return the current short git SHA, or None if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _call_llm(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Send a prompt to the OpenAI Chat Completions API, returning the
    assistant's text.  Results are transparently disk-cached.

    Parameters
    ----------
    client : OpenAI
        Initialised OpenAI client.
    model : str
        Model identifier (e.g. ``"gpt-4o-mini"``).
    prompt : str
        User message.
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        The model's reply text.
    """
    key = cache.make_key(model, prompt, temperature)
    cached = cache.get(key)
    if cached is not None:
        return cached["response"]

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    response_text = completion.choices[0].message.content or ""

    cache.set(key, {"response": response_text, "model": model, "prompt": prompt})
    return response_text


# ── Public entry-point ────────────────────────────────────────────────


def run_eval(
    model: str,
    suite: str,
    samples: list[dict[str, Any]],
    provider: str = "openai",
) -> dict[str, Any]:
    """
    Execute a full evaluation run and persist results.

    Parameters
    ----------
    model : str
        Model identifier (e.g. ``"gpt-4o-mini"``).
    suite : str
        Benchmark suite name (e.g. ``"truthfulqa"``).
    samples : list[dict]
        List of dicts with keys ``prompt``, ``expected``, ``category``,
        ``source``.
    provider : str
        LLM provider name (default ``"openai"``).

    Returns
    -------
    dict
        Aggregated results with keys ``run_id``, ``metrics``, and
        ``total_samples``.
    """
    init_db()
    session = get_session()
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Create the run record
    run_id = str(uuid.uuid4())
    short_id = run_id[:4]
    run = Run(
        id=run_id,
        model=model,
        provider=provider,
        suite=suite,
        git_sha=_get_git_sha(),
        status="running",
    )
    session.add(run)
    session.commit()

    # Accumulators per metric
    metric_scores: dict[str, list[float]] = {
        "exact_match": [],
        "f1_score": [],
        "refusal": [],
    }
    metric_passed: dict[str, int] = {
        "exact_match": 0,
        "f1_score": 0,
        "refusal": 0,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Running eval [{0}]".format(model)),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("eval", total=len(samples))

        for s in samples:
            # Upsert the sample
            sample = Sample(
                suite=suite,
                prompt=s["prompt"],
                expected=s["expected"],
                category=s.get("category", ""),
                source=s.get("source", suite),
            )
            session.add(sample)
            session.flush()  # populate sample.id

            # Call LLM
            actual = _call_llm(client, model, s["prompt"])

            # Score — exact_match
            em = exact_match(s["expected"], actual)
            em_passed = em >= ALERT_THRESHOLD
            metric_scores["exact_match"].append(em)
            metric_passed["exact_match"] += int(em_passed)
            session.add(Result(
                run_id=run_id,
                sample_id=sample.id,
                metric="exact_match",
                score=em,
                passed="true" if em_passed else "false",
            ))

            # Score — f1_score
            f1 = f1_score(s["expected"], actual)
            f1_passed = f1 >= ALERT_THRESHOLD
            metric_scores["f1_score"].append(f1)
            metric_passed["f1_score"] += int(f1_passed)
            session.add(Result(
                run_id=run_id,
                sample_id=sample.id,
                metric="f1_score",
                score=f1,
                passed="true" if f1_passed else "false",
            ))

            # Score — refusal
            is_refusal = refusal_detector(actual)
            refusal_score = 1.0 if is_refusal else 0.0
            metric_scores["refusal"].append(refusal_score)
            metric_passed["refusal"] += int(is_refusal)
            session.add(Result(
                run_id=run_id,
                sample_id=sample.id,
                metric="refusal",
                score=refusal_score,
                passed="true" if is_refusal else "false",
            ))

            progress.advance(task)

    # Finalise run
    run.status = "completed"
    session.commit()
    session.close()

    # Build summary
    total = len(samples)
    from datetime import date

    summary = {
        "run_id": f"run_{date.today().strftime('%Y%m%d')}_{short_id}",
        "total_samples": total,
        "metrics": {},
    }
    for metric_name, scores in metric_scores.items():
        avg = sum(scores) / total if total else 0.0
        summary["metrics"][metric_name] = {
            "avg_score": round(avg, 2),
            "passed": metric_passed[metric_name],
            "total": total,
        }

    return summary
