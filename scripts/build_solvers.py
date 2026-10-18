"""Compile extensions for the Python interpreter running this script."""
import argparse
import shlex
import subprocess
import sysconfig
from pathlib import Path


if __name__ == "__main__":
    try:
        import pybind11
        pybind_include = pybind11.get_include()
    except ImportError:
        import torch
        pybind_include = str(Path(torch.__file__).parent/"include")
        if not (Path(pybind_include)/"pybind11/pybind11.h").exists():
            raise SystemExit('Install native dependencies: pip install -e ".[native]"')
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["all", "ca", "cbs"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root/"src/primal_mapf/solvers"
    for name in (["ca", "cbs"] if args.solver == "all" else [args.solver]):
        source = root/"native"/name
        files = (["ca_binder.cpp", "src/Agent.cpp", "src/CA_Environment.cpp", "src/CooperativeA.cpp",
                  "src/AStar/Astar.cpp", "src/AStar/Node.cpp"] if name == "ca" else
                 ["binder.cpp", "lib/HighLevel.cpp", "lib/LowLevel.cpp", "lib/ConstraintTree.cpp"])
        command = [*shlex.split(sysconfig.get_config_var("CXX") or "c++"), "-O3", "-std=c++17", "-shared", "-fPIC",
                   "-I"+pybind_include, "-I"+sysconfig.get_paths()["include"],
                   "-Ilib", "-Ilib/CA", "-Ilib/AStar", *files,
                   "-o", str(target/(name+"_core"+sysconfig.get_config_var("EXT_SUFFIX")))]
        subprocess.run(command, cwd=source, check=True)
        print(f"Built {name} for this Python interpreter")
