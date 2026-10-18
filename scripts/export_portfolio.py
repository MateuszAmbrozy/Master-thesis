"""Create an isolated source copy without parent Git history or local artifacts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("src", "native", "scripts", "configs", "docs", "tests", "third_party", "research", "maps", "checkpoints", ".github", "legacy")
FILES = ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", ".gitignore", "requirements-tested.txt", "requirements-lock-linux-cpu.txt", "MANIFEST.in")
ALLOWED_SUFFIXES = {".py", ".cpp", ".h", ".md", ".txt", ".json", ".csv", ".png", ".svg", ".map", ".yml", ".yaml"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".o", ".a", ".pt", ".pth", ".ckpt", ".tmp"}


def export(destination: Path) -> int:
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Export outside the project directory to avoid recursive copies")
    if destination.exists():
        raise FileExistsError("Export destination must not exist")
    candidates = [ROOT/name for name in FILES if (ROOT/name).is_file()]
    candidates.append(ROOT/"results/README.md")
    for folder in FOLDERS:
        for path in (ROOT/folder).rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Refusing to export symbolic link: {path}")
            public_animation = path.is_relative_to(ROOT/"docs/animations") and path.suffix in {".gif", ".mp4"}
            if path.is_file() and (path.suffix in ALLOWED_SUFFIXES or public_animation) and path.suffix not in IGNORED_SUFFIXES and not any(
                    part == "__pycache__" or part.endswith(".egg-info") or part in {"build", ".venv", ".pytest_cache"}
                    for part in path.relative_to(ROOT).parts):
                candidates.append(path)
    manifest = {}
    destination.mkdir(parents=True)
    for path in sorted(candidates):
        relative = path.relative_to(ROOT)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/"EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Exported {len(manifest)} source/documentation files to {destination}")
    print("No Git history, checkpoints, build products, environments, or run outputs were copied.")
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(Path(args.output))


if __name__ == "__main__":
    main()
