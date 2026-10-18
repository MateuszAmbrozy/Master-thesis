# Checkpoints

Model weights are stored locally and excluded from Git. Training from scratch
and reading the archived measurements do not require pretrained models.
Configurations that reference the base models require those files separately.

## Base models

| Path | Model | Saved episode |
| --- | --- | --- |
| `base/classic.pt` | Classic PRIMAL-style network | 300,000 |
| `base/improved.pt` | Residual and attention-based network | 253,339 |

[manifest.json](manifest.json) contains the original source locations, sizes,
and SHA-256 hashes. With both base files in place, check them from the repository
root:

```bash
python scripts/verify_checkpoints.py
```

The original full training configurations are unavailable; episode counts
alone are not comparable training budgets.

## Start from weights or resume a run

Use `--init` to load weights into a new training run:

```bash
python scripts/train.py --config configs/train_classic_finetune.json --init checkpoints/base/classic.pt
python scripts/train.py --config configs/train_improved_finetune.json --init checkpoints/base/improved.pt
```

The source files stay unchanged. New weights are written to the output directory
in the configuration, with a fresh optimizer and episode counter.

Use `--resume` to continue a checkpoint created by the current trainer:

```bash
python scripts/train.py --config configs/train_classic.json --resume results/train_classic/checkpoint.pt
```

Resume also restores optimizer state and the completed episode count. It requires
trainer version `primal_joint_masked_v2`; older checkpoints can be loaded through
`--init`.

A run periodically replaces its own `checkpoint.pt` with the latest state.
There is no automatic best-model selection. Save a separate copy of any weights
you want to retain before resuming.

## Fine-tuned models

Historical evaluation scripts expect these optional filenames:

```text
fine_tuned/classic/{maze,room,warehouse}.pt
fine_tuned/improved/{maze,room,warehouse}.pt
```

Six historical map-specific models were copied from the original research
workspace into these locations. Their sources, hashes, episode counts, and
comparison with base weights are recorded in
[fine_tuned/manifest.json](fine_tuned/manifest.json). They are local files and
are not included in Git.

```bash
python scripts/verify_checkpoints.py --group all
python scripts/fixed_map.py evaluate --map maze --weights fine-tuned
```

See [fixed-map training and evaluation](../docs/FIXED_MAPS.md) for importing the
original models, starting new training, and testing new checkpoints.
All `.pt`, `.pth`, and `.ckpt` files are ignored by Git.
