from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def require_local_cache(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"source cache not found: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a locally cached AMANAH source file")
    parser.add_argument("path")
    args = parser.parse_args()
    path = require_local_cache(args.path)
    print(sha256_file(path))


if __name__ == "__main__":
    main()
