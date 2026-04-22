"""
Shared pytest bootstrap for the Multi-Modal Emotion Recognition test suite.

Given a local developer environment without deployed cloud services,
when tests import backend/frontend support modules,
then repository-root path wiring is applied so tests run deterministically.

Mocking strategy note:
- Hugging Face model downloads are avoided in unit tests through lightweight fakes
    and AST-based function extraction where appropriate.
- Supabase network interactions are isolated to service-level tests and should be
    mocked in those modules to preserve offline, local execution.
"""

# --- Imports ---
import sys
from pathlib import Path

# --- Test Bootstrap ---
# Make the repository root importable so tests can use package-style imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
