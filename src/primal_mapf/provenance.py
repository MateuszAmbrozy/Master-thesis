"""Record the runtime and exact supported source tree used for a run."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def runtime_metadata(root: Path) -> dict:
    versions = {}
    for package in ("torch", "numpy", "gymnasium", "primal-mapf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_files = {}
    for folder in ("src", "native", "scripts", "configs"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cpp", ".h", ".json"}:
                source_files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    git = {"commit": None, "scope": None, "dirty": None}
    try:
        def run(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        git["scope"] = "repository" if Path(run("rev-parse", "--show-toplevel")).resolve() == root.resolve() else "parent_repository"
        git["commit"] = run("rev-parse", "HEAD")
        git["dirty"] = bool(run("status", "--porcelain", "--", str(root)))
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "device": "cpu",
        "dependencies": versions,
        "git": git,
        "source_sha256": hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest(),
        "source_files": source_files,
    }


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
