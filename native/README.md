# CA* and CBS

C++17 implementations of Cooperative A* and Conflict-Based Search, exposed to
Python through pybind11.

## Build

From the repository root, with your Python environment active:

```bash
python -m pip install -e ".[native]"
python scripts/build_solvers.py
```

To build one solver:

```bash
python scripts/build_solvers.py --solver ca
python scripts/build_solvers.py --solver cbs
```

The compiler needs C++17 support and the active interpreter's development
headers. On Ubuntu or Debian, these are provided by `build-essential` and
`python3-dev`. Builds have been tested on Linux x86_64.

Extensions are written to `src/primal_mapf/solvers/`. Build and run with the same
Python interpreter; rebuild after changing Python versions. Compiled files are
excluded from Git.

## Run

```bash
python scripts/evaluate.py --config configs/experiments/solvers_smoke.json --output results/solvers_check
```

Choose a new output directory for subsequent runs. The smoke configuration
compares PIBT, CA*, and CBS without model checkpoints.

## Python interface

Use [`primal_mapf.solvers.api`](../src/primal_mapf/solvers/api.py) rather than
calling the bindings directly. It accepts a binary obstacle array and lists
of `(row, column)` start and goal positions. Returned plans use
`path[timestep][agent]` coordinates.

The adapter preserves map dimensions with a blocked border, runs each native
call in a separate process, and applies a wall-clock timeout. It pads shorter
paths by waiting at goals, then checks obstacles, boundaries, vertex collisions,
edge swaps, goal completion, and the timestep limit.

CA* uses prioritized planning with finite goal reservations. CBS uses constrained
space-time A* and a sum-of-transition-cost priority. Low-level CBS searches stop
after 300,000 expanded states. A timeout or exhausted search returns no plan;
it does not establish that the scenario is unsolvable.

The CA* and CBS implementations were written by Mateusz Ambroży and are
distributed under the project's [MIT license](../LICENSE). Other MAPF
implementations were consulted during development. See
[attribution notes](../THIRD_PARTY_NOTICES.md) for details.
