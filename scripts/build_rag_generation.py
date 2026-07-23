from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import init_db
from app.services.rag.generation_service import build_shadow_generation


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and evaluate a shadow RAG index generation.")
    parser.add_argument("--generation-id")
    parser.add_argument("--source-root", action="append", default=[])
    args = parser.parse_args()
    init_db()
    result = build_shadow_generation(
        generation_id=args.generation_id,
        source_roots=args.source_root or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
