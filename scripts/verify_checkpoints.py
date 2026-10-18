"""Verify the hashes of locally available base or map-specific checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["base", "fine-tuned", "all"], default="base")
    args = parser.parse_args()
    manifests = []
    if args.group in {"base", "all"}:
        manifests.append(ROOT/"checkpoints/manifest.json")
    if args.group in {"fine-tuned", "all"}:
        manifests.append(ROOT/"checkpoints/fine_tuned/manifest.json")
    for manifest in manifests:
        if not manifest.exists():
            raise SystemExit(f"Missing manifest: {manifest}; import the historical weights first")
        for name, record in json.loads(manifest.read_text()).items():
            path = ROOT/record["file"]
            if not path.exists():
                raise SystemExit(f"Missing checkpoint: {path}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != record["sha256"]:
                raise SystemExit(f"Checksum mismatch: {path}")
            print(f"{name}: SHA-256 OK")


if __name__ == "__main__":
    main()
