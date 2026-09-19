"""Regenerate openapi.json from the FastAPI app.

Usage: uv run python scripts/generate_openapi.py > openapi.json
"""

import json

from laya_api.api import app


print(json.dumps(app.openapi(), indent=2))
