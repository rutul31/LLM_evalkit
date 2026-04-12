"""
evalkit.core.db — SQLAlchemy ORM models and database initialisation.

Defines four tables that capture every aspect of an evaluation run:

* **runs**    – top-level metadata (model, suite, status, cost).
* **samples** – individual prompts loaded from a benchmark dataset.
* **results** – per-sample, per-metric scores linked to a run.
* **alerts**  – automatic regression flags when a metric drops.

Call ``init_db()`` once at startup to create (or migrate) all tables
in the configured SQLite database.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from evalkit.config import DATABASE_URL, EVALKIT_DATA_DIR

Base = declarative_base()


def _new_id() -> str:
    """Generate a new UUID4 string for use as a primary key."""
    return str(uuid.uuid4())


# ── ORM Models ────────────────────────────────────────────────────────


class Run(Base):
    """A single evaluation run against a model + benchmark suite."""

    __tablename__ = "runs"

    id = Column(Text, primary_key=True, default=_new_id)
    model = Column(Text, nullable=False)
    provider = Column(Text, nullable=False, default="openai")
    suite = Column(Text, nullable=False)
    git_sha = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="pending")
    cost_usd = Column(Float, nullable=False, default=0.0)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    results = relationship("Result", back_populates="run", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Run {self.id[:8]} model={self.model} suite={self.suite}>"


class Sample(Base):
    """An individual evaluation prompt drawn from a benchmark dataset."""

    __tablename__ = "samples"

    id = Column(Text, primary_key=True, default=_new_id)
    suite = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)
    expected = Column(Text, nullable=False)
    category = Column(Text, nullable=True)
    source = Column(Text, nullable=False)

    results = relationship("Result", back_populates="sample")

    def __repr__(self) -> str:
        return f"<Sample {self.id[:8]} suite={self.suite}>"


class Result(Base):
    """Score produced by applying a single metric to one sample in a run."""

    __tablename__ = "results"

    id = Column(Text, primary_key=True, default=_new_id)
    run_id = Column(Text, ForeignKey("runs.id"), nullable=False)
    sample_id = Column(Text, ForeignKey("samples.id"), nullable=False)
    metric = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    passed = Column(Text, nullable=False)  # "true" / "false"
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    run = relationship("Run", back_populates="results")
    sample = relationship("Sample", back_populates="results")

    def __repr__(self) -> str:
        return f"<Result {self.metric}={self.score:.2f}>"


class Alert(Base):
    """Regression alert raised when a metric score drops between runs."""

    __tablename__ = "alerts"

    id = Column(Text, primary_key=True, default=_new_id)
    run_id = Column(Text, ForeignKey("runs.id"), nullable=False)
    metric = Column(Text, nullable=False)
    prev_score = Column(Float, nullable=False)
    curr_score = Column(Float, nullable=False)
    delta_pct = Column(Float, nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    run = relationship("Run", back_populates="alerts")

    def __repr__(self) -> str:
        return f"<Alert {self.metric} Δ={self.delta_pct:+.1f}%>"


# ── Engine / Session factory ──────────────────────────────────────────

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    EVALKIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session():
    """Return a new SQLAlchemy session (caller is responsible for closing)."""
    return SessionLocal()
