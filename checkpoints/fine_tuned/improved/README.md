# Fine-tuned improved weights

Optional weights for the residual and attention-based model. The historical evaluation
scripts use these filenames:

| File | Training map family |
| --- | --- |
| `maze.pt` | Maze |
| `room.pt` | Rooms |
| `warehouse.pt` | Warehouse |

These three historical files have been copied into the local workspace and
verified against their originals. They remain excluded from Git, so a fresh
clone requires a separate import. Hashes and source paths are in
[the fine-tuned manifest](../manifest.json).

```bash
python scripts/fixed_map.py evaluate --map maze --variant improved --weights fine-tuned
```

Replace `maze` with `room` or `warehouse` to select another model. New training
writes to `results/fixed_maps/`, keeping these historical files unchanged.
See [fixed-map usage](../../../docs/FIXED_MAPS.md).

To start another training run from these weights, use `--init`. To continue
a checkpoint from the current trainer with its optimizer state, use `--resume`.
See [checkpoint usage](../../README.md) for commands and save behavior.
