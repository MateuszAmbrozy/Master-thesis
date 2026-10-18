"""Verify archive checksums and optionally compare CSV rows with source data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(source_root=None):
    records = json.loads((ROOT/"research/historical/file_manifest.json").read_text())
    measurements = 0
    for record in records:
        destination = ROOT/record["destination"]
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != record["destination_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {destination}")
        if destination.suffix != ".csv":
            continue
        with destination.open(newline="") as stream:
            imported = list(csv.DictReader(stream))
        measurements += len(imported)
        if source_root is not None:
            source = Path(source_root)/record["source"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
                raise ValueError(f"Original checksum mismatch: {source}")
            with source.open(newline="") as stream:
                original = list(csv.DictReader(stream))
            for row in original:
                row["Algorithm"] = row["Algorithm"].replace("PRIMAL_Klasyk", "PRIMAL_Classic").replace("PRIMAL_Zmod", "PRIMAL_Improved")
            if original != imported:
                raise ValueError(f"Historical measurement mismatch: {destination}")
    print(f"Verified {len(records)} archive files and {measurements} measurement rows.")
    return len(records), measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="Optional directory containing the original research folders")
    args = parser.parse_args()
    verify(args.source_root)
