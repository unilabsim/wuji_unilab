"""Local, provenance-checked Wuji assets. No network asset resolver is used."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ASSET_ROOT = Path(__file__).parent / "assets"


def verify_assets() -> dict[str, str]:
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = ASSET_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required Wuji asset is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Wuji asset checksum mismatch: {path}")
    return manifest["sha256"]


def main() -> None:
    print(f"Verified {len(verify_assets())} local Wuji assets in {ASSET_ROOT}")
