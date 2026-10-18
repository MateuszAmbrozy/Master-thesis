"""Copy historical map-specific weights without replacing existing models."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from primal_mapf.runtime import ROOT, new_model
from primal_mapf.provenance import file_sha256


def import_models(source_root: Path) -> None:
    records, pending = {}, []
    for variant, folder in (("classic", "PRIMAL"), ("improved", "PRIMAL_IMPROVMENT")):
        base_path = ROOT / f"checkpoints/base/{variant}.pt"
        base = torch.load(base_path, map_location="cpu", weights_only=True)["model"] if base_path.exists() else None
        for map_name in ("maze", "room", "warehouse"):
            relative = Path(folder) / map_name / "checkpoint.pt"
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Missing source checkpoint: {source}")
            digest = file_sha256(source)
            target = ROOT / f"checkpoints/fine_tuned/{variant}/{map_name}.pt"
            if target.exists() and file_sha256(target) != digest:
                raise FileExistsError(f"Different weights already exist at {target}; nothing has been replaced")
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            new_model(variant).load_state_dict(checkpoint["model"])
            same_weights = (set(base) == set(checkpoint["model"]) and all(
                torch.equal(base[key], checkpoint["model"][key]) for key in base)) if base is not None else None
            episode = checkpoint.get("episode")
            records[f"{variant}/{map_name}"] = {
                "file": target.relative_to(ROOT).as_posix(), "source": relative.as_posix(),
                "sha256": digest, "size_bytes": source.stat().st_size,
                "episode": int(episode) if episode is not None else None,
                "weights_equal_base": same_weights,
                "historical_training_config": "unknown",
            }
            pending.append((source, target, digest))
    for source, target, digest in pending:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            shutil.copy2(source, temporary)
            if file_sha256(temporary) != digest:
                raise RuntimeError(f"Copy checksum mismatch: {temporary}")
            temporary.replace(target)
    manifest = ROOT / "checkpoints/fine_tuned/manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    for name, record in records.items():
        note = " (weights identical to base)" if record["weights_equal_base"] else ""
        print(f"{name}: SHA-256 OK{note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent/"primal_improvment/fine-tuning")
    args = parser.parse_args()
    try:
        import_models(args.source_root)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
