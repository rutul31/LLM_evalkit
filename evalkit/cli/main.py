"""
evalkit.cli.main — Typer CLI entry-point.

Provides the ``evalkit run`` command which orchestrates a complete
evaluation: load dataset → call LLM → score → persist → display.

Usage
-----
::

    evalkit run --model gpt-4o-mini --suite truthfulqa --limit 20

The command prints a Rich table summarising per-metric averages and
pass counts, then confirms the SQLite database path where full results
are stored.
"""

import typer
from rich.console import Console
from rich.table import Table

from evalkit.config import DB_PATH

app = typer.Typer(
    name="evalkit",
    help="Open-source LLM evaluation and red-teaming framework.",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()


# ── Dataset registry (extensible in future weeks) ────────────────────

_SUITE_LOADERS = {
    "truthfulqa": "evalkit.datasets.truthfulqa",
}


@app.command()
def run(
    model: str = typer.Option(..., help="Model identifier, e.g. gpt-4o-mini"),
    suite: str = typer.Option("truthfulqa", help="Benchmark suite to evaluate against"),
    limit: int = typer.Option(50, help="Number of samples to evaluate"),
) -> None:
    """
    Run an evaluation against a benchmark suite.

    Loads samples, queries the model for each, scores with all
    registered metrics, and writes results to the local SQLite
    database.
    """
    # Resolve dataset loader
    if suite not in _SUITE_LOADERS:
        console.print(f"[red]Unknown suite:[/red] {suite}")
        console.print(f"Available: {', '.join(_SUITE_LOADERS)}")
        raise typer.Exit(code=1)

    # Lazy import to keep CLI startup fast
    import importlib

    loader = importlib.import_module(_SUITE_LOADERS[suite])

    # Load samples
    console.print(f"Loading {suite}...", end=" ")
    samples = loader.load(limit=limit)
    console.print(f"[green]✓[/green] {len(samples)} samples\n")

    # Run evaluation
    from evalkit.core.runner import run_eval

    summary = run_eval(model=model, suite=suite, samples=samples)

    # ── Display results ───────────────────────────────────────────
    console.print()
    console.print(f"[bold]Results — {summary['run_id']}[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white", min_width=12)
    table.add_column("Score", justify="right", min_width=6)
    table.add_column("Passed", justify="right", min_width=7)

    for metric_name, data in summary["metrics"].items():
        table.add_row(
            metric_name,
            f"{data['avg_score']:.2f}",
            f"{data['passed']}/{data['total']}",
        )

    console.print(table)
    console.print(f"\nSaved to [bold]{DB_PATH}[/bold]")


if __name__ == "__main__":
    app()
