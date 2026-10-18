# Validation record

Updated on 2026-10-16 for trainer `primal_joint_masked_v2` and package version
`0.2.0`. Checks below describe local evidence; CI configuration is not a claim
that GitHub-hosted jobs have already run.

## Environment

A new virtual environment was created under `/tmp` without access to packages
from the earlier research environment. Installation from the package indexes
completed successfully on Linux x86_64 with Python 3.12. Direct runtime versions:

| Dependency | Version |
| --- | --- |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.4 |
| Gymnasium | 1.4.0 |
| pybind11 | 3.1.0 |

The complete dependency snapshot is in
[requirements-lock-linux-cpu.txt](../requirements-lock-linux-cpu.txt).
The older research environment used PyTorch 2.10.0+cpu, NumPy 2.3.5, and
Gymnasium 1.2.3; those are historical validation versions, not the current
clean-install snapshot.

## Regression and workflow checks

All 29 unittest cases passed with both native extensions built and historical
base checkpoints available. Coverage includes:

- Legal-action normalization and policy gradients, with zero invalid-action mass.
- Joint rotations, following moves, vertex conflicts, swaps, cancellation chains,
  and randomized trajectory legality.
- Exact replay of solver plans and cloning updates for both architectures.
- Identical resumed and uninterrupted CPU RL model weights.
- Interruption between checkpoint saves and automatic removal of uncommitted
  CSV rows before resume.
- Common-profile observation and reward consistency.
- Configuration validation, malformed plans, and invalid endpoints.
- Gymnasium's environment checker, deterministic seeded resets, and truncation.
- Evaluation using newly trained weights without historical checkpoints.
- Fixed-map configurations, CLI selection, scratch training, initialization,
  resume, and evaluation for both networks on maze, room, and warehouse.
- Loading and short policy rollouts for all six historical map-specific models.
- GIF trajectory validation, endpoint-preserving frame sampling, frame changes,
  trace preservation, and protection against replacing existing animation outputs.
- Native compilation, rectangular maps, CBS vertex and edge conflicts, a goal
  occupant that must move away, and external worker timeout enforcement.

A complete smoke workflow trained both models, evaluated their new checkpoints
alongside PIBT, and exported an SVG without requiring historical weights.
Short smoke failures are expected and are not evidence of trained policy quality.

## Historical compatibility and archive integrity

The refactored networks were compared with the local pre-refactor network
implementations using the historical weights and identical random inputs.
Their forward outputs matched exactly in the checked CPU calls. Sampled
sequential transitions and observations, including the classic distance cap,
matched the earlier environment implementations under historical profiles.
These checks establish sampled compatibility, not full behavioral equivalence
for every possible input.

Archive verification confirmed 119 imported checksums and 9,360 measurement rows.
PIBT's three implementation files match the recorded upstream revision byte for
byte. The PRIMAL upstream license checksum and all three official map downloads
were recorded; nine archived map copies match the verified map texts.

## Experiment report

The [included report](portfolio/README.md) trains each architecture with three
seeds, using eight cloning episodes per model, and evaluates each on the same
eight scenarios. Configurations, exact scenarios, raw results, training logs,
versions, and source/model checksums are retained. The example establishes an
executable paired protocol, not convergence or statistical superiority.

The illustrated successful trajectory comes from PIBT. Training replay success
in cloning logs is not autonomous policy success.

## Packaging and isolated source copies

The package builds as a Python wheel. Wheel installation and the isolated source
export are checked separately from the original research workspace. Source export
excludes checkpoints, native libraries, virtual environments, caches, parent
Git history, and generated run directories; the curated report remains included.

The GitHub Actions workflow configures Python 3.11/3.12 jobs and a separate native
solver job. The native job requires both extensions rather than silently skipping
their checks. Local validation currently covers Python 3.12; hosted CI execution
and Python 3.11 results are not yet recorded.

## Remaining limits

Long training, GPU execution, Windows, macOS, and statistical replication of the
historical study have not been validated. CPU repeatability does not imply
bitwise reproducibility across dependency versions or hardware. Native search
bounds and timeouts are not proofs of unsolvability or unbounded guarantees.
The project author confirmed authorship of the native CA*/CBS implementations;
see [attribution notes](../THIRD_PARTY_NOTICES.md).

New local validation outputs are stored under `results/` and ignored by Git.
Historical measurements are retained separately and were not overwritten.
