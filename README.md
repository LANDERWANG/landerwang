# Silicon carbide polishing optimizer

This repository provides a lightweight optimisation tool for silicon
carbide (SiC) grinding/polishing recipes.  The model consumes the key
machine settings that typically vary during process tuning—spindle and
work table speeds, spindle load, initial surface roughness, wafer face,
and whether the step is a coarse or fine grind—and predicts:

* expected material removal (µm)
* grit consumption (expressed as the GR value)
* cycle time (s)

The GR value is prioritised during optimisation, aligning with the
requirement to keep abrasive wear under control.

## Quick start

The optimiser is implemented in [`polish_optimizer.py`](./polish_optimizer.py).
Use Python 3.9+ and run the script directly:

```bash
python polish_optimizer.py \
  --spindle 1800 --table 45 --load 60 \
  --roughness 0.30 --face C --condition coarse
```

The command above prints a single-point evaluation of the process
metrics.  Supply search ranges to let the tool explore a grid and select
the recipe with the highest GR value that satisfies optional
constraints:

```bash
python polish_optimizer.py \
  --spindle 1600 --table 35 --load 55 \
  --roughness 0.25 --face Si --condition fine \
  --spindle-range 1400,2200,100 --table-range 20,60,5 \
  --max-time 180
```

Add `--json` to emit machine-readable output for downstream analysis or
integration into notebooks and dashboards.

## Model overview

The underlying model is heuristic but captures the qualitative trends
observed in SiC processing:

* faster wheel/table speed and higher load increase removal rate but
  reduce GR due to accelerated abrasive wear
* C-face wafers polish slightly faster and exhibit better grit
  retention than Si-face wafers
* fine-grind steps aim for higher GR and better surface quality at the
  cost of longer cycle times

If needed, adjust the coefficients in `polish_optimizer.py` to calibrate
the predictions with production data.
