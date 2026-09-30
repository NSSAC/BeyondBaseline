# Scenario Sampling Runner

This repo provides code to reproduce the evaluation of different sampling algorithms across eight scenarios (Table 3 in the paper).  
It loads a synthetic linelist and population file, runs all 8 scenarios in one go, and outputs:

- Per-scenario CSVs of KL divergence values (one row per week, per algorithm).
- Three final plots (one per algorithm) comparing all 8 scenarios.
- Average running time (seconds) across the 8 scenarios for each algorithm.

---

## Requirements

- Python 3.8+
- Dependencies:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`

## Install

From the repository root:

```bash
pip install -e .
```

This installs the scenarios CLI commands defined in `pyproject.toml`.

## Validate CLI Installation

From the repository root:

```bash
python3 scripts/scenarios_simulation/verify_cli_entrypoints.py
```


# Run simulation with:
```bash
scenarios-runner \
  --linelist ../data/linelist.csv.xz \
  --population ../../va_persontrait_epihiper.csv \
  --infections ../data/linelist_allevents.csv.xz \
  --outdir ./result \
  --batch-size 100 \
  --no-replacement \
  --seed 42 \
  --algorithms "surs", "stratified", "LASSO-Greedy", "LASSO-Stratified" \
  --stratifiers "age", "race", "county", "sex" \
  --save-samples
```

# Run replicates test with:
```bash
scenarios-replicates \
  --replicates-dir ../data/replicate \
  --population ../../va_persontrait_epihiper.csv

# if you want uncertainty
# Note: plot_kl_uncertainty.py is referenced here but is not present in this repository snapshot.
```

# Run lasso test with:
```bash
scenarios-lasso-greedy \
  --linelist ../results/replicate_0/linelist.csv.xz \
  --population va_persontrait_epihiper.csv \
  --infections ../results/replicate_0/linelist_allevents.csv.xz \
  --stratifiers age race county sex ses \
  --outdir lasso_greedy_50-500 \
  --batch-size 100 \
  --min-group-size 50 \
  --max-group-size 500 \
  --step-group-size 50 \
  --seed 42

scenarios-lasso-stratified \
  --linelist ../results/replicate_0/linelist.csv.xz \
  --population va_persontrait_epihiper.csv \
  --infections ../results/replicate_0/linelist_allevents.csv.xz \
  --stratifiers age race county sex ses \
  --outdir lasso_stratified_50-500 \
  --batch-size 100 \
  --min-group-size 50 \
  --max-group-size 500 \
  --step-group-size 50 \
  --seed 42
```

# Aggregate metrics across replicate outputs:
```bash
scenarios-aggregate \
  --input-dir replicate_results \
  --outdir replicate_results
```

# Run lite version with:
```bash
# week 1:
scenarios-weekly \
  --linelist ../data/linelist.csv \
  --population ../../va_persontrait_epihiper.txt \
  --target "LL" \
  --current-date 2021-05-31 \
  --batch-size 100 \
  --algorithms surs \
  --no-replacement

# week 2:
scenarios-weekly \
  --linelist ../data/linelist.csv \
  --population ../../va_persontrait_epihiper.txt \
  --already-sequenced weekly_results/ \
  --target "LL"  \
  --current-date 2021-06-07 \
  --batch-size 100 \
  --algorithms surs \
  --no-replacement
```