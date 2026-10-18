# Earlier training scripts

Reference copies of the asynchronous trainers used during development:

| File | Model |
| --- | --- |
| `classic_train.py` | Classic PRIMAL-style network |
| `improved_train.py` | Residual and attention-based network |

These scripts depend on the earlier module layout and are kept for reading,
not as runnable entry points. For training in this repository, use:

```bash
python scripts/train.py --config configs/train_classic.json
python scripts/train.py --config configs/train_improved.json
```

The classic trainer retains a mismatch between four environment agents and
three worker agents. The exact script revision that produced the base weights
is unknown. Source mappings and differences from the current trainer are listed
in [MIGRATION.md](../docs/MIGRATION.md).
