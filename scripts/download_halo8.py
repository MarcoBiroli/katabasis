#!/usr/bin/env python3
"""Download the Halo8 dataset from Zenodo (record 16737590).

Halo8 ships as an ASE database. This script fetches the archive into
``data/`` and points the loader at it. The full record is large (~20M
structures); pass ``--dry-run`` to print the URLs without downloading.

Zenodo record: https://zenodo.org/records/16737590
Paper:        https://www.nature.com/articles/s41597-025-05944-3
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records/16737590"


def list_files() -> list[dict]:
    with urllib.request.urlopen(ZENODO_API) as resp:  # noqa: S310 (trusted host)
        record = json.loads(resp.read().decode())
    return record.get("files", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true", help="list files, do not download")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = list_files()
    if not files:
        print("No files found in the Zenodo record (network or record issue).")
        return 1

    for f in files:
        key = f.get("key", "?")
        url = f.get("links", {}).get("self", "")
        size = f.get("size", 0)
        print(f"{key}\t{size/1e9:.2f} GB\t{url}")
        if args.dry_run:
            continue
        dest = args.out / key
        print(f"  downloading -> {dest}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310

    if not args.dry_run:
        print(f"\nDone. Point configs/data/*.yaml `db_path` at the file under {args.out}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
