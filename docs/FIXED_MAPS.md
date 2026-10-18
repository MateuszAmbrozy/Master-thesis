# Training and evaluation on fixed maps

The `fixed_map.py` command selects the map, model, and checkpoint paths for you.
Run it from the repository root with your Python environment active.

| Map name | File |
| --- | --- |
| `maze` | `maps/maze-32-32-4.map` |
| `room` | `maps/room-64-64-16.map` |
| `warehouse` | `maps/warehouse-10-20-10-2-1.map` |

The map stays fixed; starts and goals are resampled each episode from its largest
connected free component. Training defaults to four agents and 10,000 episodes
for either architecture. Settings are in `configs/maps/<map>/`.

## Train from scratch

```bash
python scripts/fixed_map.py train --map maze --variant classic
python scripts/fixed_map.py train --map maze --variant improved
```

Replace `maze` with `room` or `warehouse` for the other maps.
Outputs go to `results/fixed_maps/maze/train_classic/` and `train_improved/`.

## Fine-tune a base model

```bash
python scripts/fixed_map.py train --map maze --variant classic --init base
python scripts/fixed_map.py train --map maze --variant improved --init base
```

These load `checkpoints/base/<variant>.pt` and write new runs to
`results/fixed_maps/maze/finetune_<variant>/`.

To initialize from a historical map-specific model instead:

```bash
python scripts/fixed_map.py train --map room --variant classic --init fine-tuned
python scripts/fixed_map.py train --map warehouse --variant improved --init fine-tuned
```

A custom checkpoint path also works with `--init`. Initialization loads weights
only, with a fresh optimizer and episode count. Historical checkpoints cannot
be resumed by the current trainer.

## Choose a training budget

```bash
python scripts/fixed_map.py train --map maze --variant improved --init base --episodes 1000 --agents 8 --output results/my_maze_run
```

Other options include `--max-steps`, `--seed`, and `--expert pibt|ca|cbs`.
CA* and CBS require compilation. Use `--dry-run` to print the selected
configuration without training.

Each new run requires an empty output directory. Source checkpoints are read
only. The run's own `checkpoint.pt` is periodically replaced with its latest
state; copy any model you want to retain before continuing.

## Resume a map-specific run

```bash
python scripts/fixed_map.py train --map maze --variant classic --resume results/fixed_maps/maze/finetune_classic/checkpoint.pt --episodes 20000
```

Resume reads the configuration saved in the checkpoint, including its original
output directory and training settings. `episodes` is the total target count,
not the number of additional episodes. The requested map and variant must match
the checkpoint. Changing training parameters other than those permitted by
resume is rejected.

## Evaluate

Historical map-specific models:

```bash
python scripts/fixed_map.py evaluate --map maze --weights fine-tuned
python scripts/fixed_map.py evaluate --map room --weights fine-tuned
python scripts/fixed_map.py evaluate --map warehouse --weights fine-tuned
```

By default, each command evaluates both model variants and PIBT with 4, 8, 16,
and 32 agents, using 20 scenarios per setting. For a short check of one model:

```bash
python scripts/fixed_map.py evaluate --map maze --variant classic --weights fine-tuned --agents 4 --episodes 2 --max-steps 64 --output results/maze_check
```

Select other checkpoint groups with `--weights`:

| Value | Checkpoints |
| --- | --- |
| `base` | `checkpoints/base/<variant>.pt` |
| `fine-tuned` | `checkpoints/fine_tuned/<variant>/<map>.pt` |
| `trained` | `results/fixed_maps/<map>/finetune_<variant>/checkpoint.pt` |
| `scratch` | `results/fixed_maps/<map>/train_<variant>/checkpoint.pt` |

For example, evaluate a model after fine-tuning:

```bash
python scripts/fixed_map.py evaluate --map maze --variant classic --weights trained
```

If training used a custom output directory, pass the checkpoint explicitly:

```bash
python scripts/fixed_map.py evaluate --map maze --variant improved --checkpoint results/my_maze_run/checkpoint.pt --output results/my_maze_evaluation
```

Use `--solvers pibt ca cbs` to include all solvers, or `--solvers` with no names
to evaluate models only. Repeated evaluations need a new `--output` directory.

Training and evaluation default to the common environment profile with joint
movement. For historical-style policy evaluation, add
`--environment-profile historical --execution sequential`. Those settings still
use the current endpoint sampler and do not exactly reproduce archived experiments.

The same configuration files can be used with `scripts/train.py` and
`scripts/evaluate.py` directly. For example:

```bash
python scripts/train.py --config configs/maps/maze/finetune_classic.json --init checkpoints/base/classic.pt
python scripts/evaluate.py --config configs/maps/maze/evaluate_trained.json
```

The last evaluation expects both fine-tuned variants to have been trained.

## Historical weights

Six historical map-specific models are available in the local research workspace.
They were copied into `checkpoints/fine_tuned/` and checked against their sources;
[the manifest](../checkpoints/fine_tuned/manifest.json) records hashes and episode counts.
All six contain weights different from their corresponding base model.
The full historical training configurations are unknown.

Check the local files:

```bash
python scripts/verify_checkpoints.py --group all
```

Weight files are ignored by Git and will not appear in a fresh clone. If the
original research files are available, import them with:

```bash
python scripts/import_fine_tuned.py --source-root /path/to/primal_improvment/fine-tuning
```

The source directory must contain `PRIMAL/<map>/checkpoint.pt` and
`PRIMAL_IMPROVMENT/<map>/checkpoint.pt` for all three maps. Import refuses to
replace an existing file with different weights. New training outputs remain
separate from these historical models.
