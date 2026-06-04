"""Vercel serverless entrypoint.

Vercel's Python runtime serves the module-level ``app`` (an ASGI callable). The
package lives under ``src/``, so we add it to the path, then re-export the FastAPI
app unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from learning_memory_os.api import app  # noqa: E402,F401
