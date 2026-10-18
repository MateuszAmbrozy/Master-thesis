# Research materials

Measurements, figures, and scripts from the earlier MAPF experiments.
These results belong to the earlier training setup; runs from the current
trainer are saved separately in [`results/`](../results/README.md).

## Contents

```text
historical/
  scripts/
    random/                        Random-map experiments
    fixed_maps/                    Base models on benchmark maps
    fixed_maps_fine_tuned/          Fine-tuned models on benchmark maps
    fixed_maps_base_vs_fine_tuned/  Base and fine-tuned model comparisons
  results/                         CSV measurements and figures
  file_manifest.json               Source paths and checksums
fine_tuning/
  improved/maze/                   Training curves
```

There are 22 CSV datasets, 66 result figures, 20 Python scripts, nine map copies,
and two training figures. Measurement values and row order are preserved.
Algorithm labels and result-figure text use English. The training figures are
kept in their original form because the per-episode data was unavailable.

`run_01`, `run_02`, and files at experiment roots are separate source collections.
Their overlap is unknown, so they should not be pooled as independent runs.
Full historical training configurations, seeds, and commit identifiers are
also unavailable.

## Check the data

From the repository root:

```bash
python scripts/verify_research.py
```

The script checks files against
[the manifest](historical/file_manifest.json). If the original research folders
are available, `--source-root PATH` also checks source hashes and measurement
rows after translating algorithm labels.

## Regenerate plots

```bash
python -m pip install -e ".[research]"
python scripts/plot_historical_results.py
```

This overwrites the result figures and updates their hashes in the manifest.
Commit or copy any versions you want to keep first. Existing CSV files and
figures can be read without model checkpoints.

## Experiment scripts

The scripts retain their earlier sampling and evaluation logic, with imports
and paths adapted to this repository. Some write to the working directory;
run them in a separate copy if you want to preserve the archived results.
They may require plotting dependencies, native solvers, and the appropriate
[checkpoint files](../checkpoints/README.md).

For new experiments, use `scripts/evaluate.py` and the configurations under
`configs/experiments/`. The [main README](../README.md#evaluation) describes the
current protocol; [MIGRATION.md](../docs/MIGRATION.md) explains the differences
from the earlier setup.
