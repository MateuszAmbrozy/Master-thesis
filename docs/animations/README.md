# Fine-tuned policy demonstrations

These animations show the improved PRIMAL model using its historical map-specific
checkpoint. Each displayed run has eight agents and seed 123. Actions are sampled
from the legal-action policy using sequential execution and the historical
environment profile, matching the execution style of these pretrained models.

| Map | Agents | Timesteps | Goals reached | GIF | MP4 |
| --- | --- | --- | --- | --- | --- |
| Maze | 8 | 67 | 8/8 | [View](maze_improved/animation.gif) | [Video](maze_improved/animation.mp4) |
| Room | 8 | 168 | 8/8 | [View](room_improved/animation.gif) | [Video](room_improved/animation.mp4) |
| Warehouse | 8 | 137 | 8/8 | [View](warehouse_improved/animation.gif) | [Video](warehouse_improved/animation.mp4) |

Circles represent agents and outlined squares their goals; matching colors identify
each pair. Short trails show recent movement. Playback is accelerated to 12
timesteps per second, with pauses at the beginning and end. All transitions
come from policy inference; no solver plans or corrections are applied.

## Trials and selection

The models were run once per map with four agents and once with eight, always
using seed 123 and a limit of 512 timesteps. All six attempts completed. The
eight-agent runs are shown here. [trials.csv](trials.csv) includes every attempt.
This small set illustrates behavior on these scenarios; it is not an estimate
of success over arbitrary maps, starts, or goals. The sequential protocol also
differs from the common-profile joint protocol used by the default new experiments.

## Reproduce

From the repository root:

```bash
python -m pip install -e ".[visualization]"
python scripts/animate_policy.py --map maze --variant improved --agents 8 --seed 123 --output results/animations/maze
python scripts/animate_policy.py --map room --variant improved --agents 8 --seed 123 --output results/animations/room
python scripts/animate_policy.py --map warehouse --variant improved --agents 8 --seed 123 --output results/animations/warehouse
```

The command uses `checkpoints/fine_tuned/improved/<map>.pt` by default.
Those files are local and excluded from Git; see
[model import instructions](../FIXED_MAPS.md#historical-weights) if they are missing.
Use `--variant classic` for the classic model or `--checkpoint PATH` for another
compatible checkpoint. Use a new output directory for each run.

Add `--mp4` to also export a video; this requires `ffmpeg` on PATH. GIF export
requires Pillow only. `--max-frames` controls the number of rendered states;
the first and final states are always retained, and the complete trace is saved.
These examples render every timestep.

Each animation directory contains the full `trajectory.json`, checkpoint hash,
render settings and output hashes in `render.json`, a final-state poster, and
runtime metadata. The maps, endpoints, and action-sampling seed are recorded
so a run can be checked independently. To render the saved trace without inference:

```bash
python scripts/animate_policy.py --from-trace docs/animations/maze_improved/trajectory.json --output results/animations/maze_render
```

Rendering a saved trace does not require the model weights. If the checkpoint
is present locally, its hash is checked against the recorded value.

A failed rollout is exported with its actual terminal state and a time-limit
label. The script does not search for a successful seed or replace the policy
with a solver.
