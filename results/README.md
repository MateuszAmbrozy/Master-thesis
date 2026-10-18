# Run outputs

Default location for training runs, evaluations, and trajectory exports.
Everything in this directory except this README is ignored by Git.

| Run | Files |
| --- | --- |
| Training | `checkpoint.pt`, `training.csv`, `config.json`, `metadata.json` |
| Evaluation | `raw.csv`, `scenarios.json`, `summary.json`, `summary_by_setting.csv`, `config.json`, `metadata.json` |
| Visualization | SVG trajectory |

Set `output` in the configuration to choose a run directory. Training and
evaluation require an empty directory for a new run.

The trainer replaces its own `checkpoint.pt` at each save. It stores the latest
saved state rather than the best model, so copy a checkpoint elsewhere before
continuing if you want to keep it. `--init` only reads its source weights;
`--resume` continues the selected run and reconciles its log with the saved
episode count.

The short runs created by `scripts/smoke.py` are installation checks.
Historical study results are stored separately in
[`research/historical/results/`](../research/historical/results/).
