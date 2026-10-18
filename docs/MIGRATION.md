# Source mapping and methodology

Initial source mapping prepared on 2026-10-14; supported workflow updated on
2026-10-16. Original source directories remain unchanged.

| Component | Original source |
|---|---|
| `classic/network.py` | `primal_improvment/classic_primal/ACNet.py` |
| `classic/environment.py` | `primal_improvment/classic_primal/mapf_env.py` |
| `improved/network.py` | `primal_improvment/ACNet.py` |
| `improved/environment.py` | `primal_improvment/mapf_env.py` |
| `solvers/pypibt` | `primal_improvment/pypibt/src/pypibt` |
| `native/ca`, `native/cbs` | C++ sources under `primal_improvment/CA` and `CBS` |
| `legacy/classic_train.py` | `kod2/main.py`, identical to `kod3/main.py` |
| `legacy/improved_train.py` | `primal_improvment/main.py` |
| Base models | `primal_improvment/fine-tuning/BAZA` |
| Maps | `.map` files from the original `mapy` directory |

Original external paths retain their spelling in provenance records.
English names, comments, messages, and documentation are used inside this
project. Proper names and upstream attribution are preserved.

## Current workflow changes

The current trainer is tagged `primal_joint_masked_v2`. Its source mapping above
records origin, not byte-for-byte identity after refactoring.

- Model parameter names and numerical forward operations are retained for weight
  compatibility. Network results now use named, tuple-compatible fields.
- Environment logic is shared. Thin model-specific classes preserve historical
  profiles; supported workflows use the common profile and joint execution by
  default. The classic historical profile retains its 75-unit magnitude cap
  before direction normalization.
- All joint policy decisions use a shared pre-transition state. Following moves
  and rotations are allowed; vertex and edge conflicts cancel moves consistently.
  Solver plans and model traces use the same trajectory validator.
- Legal-action masking and renormalization are applied to both executed RL
  actions and the policy-gradient/entropy terms.
- Behavioral cloning replays joint expert plans, including rotations. Sequential
  compatibility mode still rejects plans it cannot replay.
- Adam state and completed episode count are restored at resume. CSV rows beyond
  the checkpoint are removed, and duplicate committed episode numbers reconciled.
- Run metadata includes versions, hardware, source hashes, and Git context;
  evaluation also records model hashes and scenario-level confidence intervals.

Older sequential trainer checkpoints can initialize a new run with `--init`,
but cannot be resumed with the revised trainer. Historical base weights remain
loadable. Evaluating them with the common profile and joint execution changes
the protocol under which they are tested; no automatic equivalence is claimed.

## Historical methodology

The preserved trainers use asynchronous workers and shared optimization. The
supported trainer is single-process recurrent actor-critic with behavioral
cloning. Loss scaling, scheduling, expert selection, endpoint generation, and
execution semantics differ. Full historical training configurations and source
revisions are unknown. New runs must therefore be evaluated as new experiments.

Select `execution: "sequential"` and `environment_profile: "historical"` for
explicit historical-style evaluation. These options do not reconstruct the
historical asynchronous process schedule or original random sampling.

CA* and CBS receive a blocked border because the C++ bindings infer dimensions
from supplied coordinates. The parent process enforces a timeout and validates
returned plans. The native CA*/CBS implementations are by Mateusz Ambroży.

See [architecture notes](ARCHITECTURE.md) for the active protocol and
[validation notes](VALIDATION.md) for tested behavior.

## Imported research

The archive contains 22 CSV datasets and the corresponding 66 English plots,
20 historical Python scripts, nine accompanying maps, and two training figures.
CSV measurements and row ordering are preserved. Algorithm labels are
translated, and plots are regenerated from the imported CSV data. File mapping
and checksums are in `research/historical/file_manifest.json`.

Historical script imports and resource paths are adapted to the project layout.
These adapted scripts are reference implementations, not byte-identical copies
of the original files. The original files remain in the source directories.

The two base checkpoints and six historical map-specific checkpoints are now
available in the local checkpoint directories. Fine-tuned source paths and
checksums are recorded in `checkpoints/fine_tuned/manifest.json`. Base classic has `episode=300000`; improved
has `episode=253339`. Their metadata is in `checkpoints/manifest.json`.
Full historical training configurations, seeds, and commit IDs are unknown.
