"""
evalkit.config — Centralised configuration loaded from environment variables.

Reads a .env file (via python-dotenv) and exposes typed settings used
throughout the framework: API keys, filesystem paths, and scoring
thresholds.  All paths are eagerly resolved to absolute locations so
downstream code never has to worry about tilde expansion.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (or wherever the process was started)
load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Data directory (cache + database) ────────────────────────────────
EVALKIT_DATA_DIR: Path = Path(
    os.getenv("EVALKIT_DATA_DIR", "~/.evalkit")
).expanduser().resolve()

# ── SQLite database path ─────────────────────────────────────────────
DB_PATH: Path = EVALKIT_DATA_DIR / "evalkit.db"
DATABASE_URL: str = f"sqlite:///{DB_PATH}"

# ── Disk cache location ──────────────────────────────────────────────
CACHE_DIR: Path = EVALKIT_DATA_DIR / "cache"

# ── Alert threshold ──────────────────────────────────────────────────
ALERT_THRESHOLD: float = float(os.getenv("EVALKIT_ALERT_THRESHOLD", "0.5"))
