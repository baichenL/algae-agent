from __future__ import annotations

from pathlib import Path


ROOTS = (Path("app/services/rag"), Path("app/services/chat/rag_intent_handler.py"), Path("app/core/db/rag.py"))
SUSPICIOUS = ("�", "???", "瀹為", "璁烘", "閰嶆", "鍩瑰吇鍩", "鈧?")


def main() -> int:
    failures: list[str] = []
    files = []
    for root in ROOTS:
        files.extend(root.rglob("*.py") if root.is_dir() else [root])
    for path in files:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if path.name == "pdf_router.py" and ("bad = sum" in line or "mojibake =" in line):
                continue
            if any(marker in line for marker in SUSPICIOUS):
                failures.append(f"{path}:{line_number}: suspicious encoding constant")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"RAG encoding scan passed ({len(files)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
