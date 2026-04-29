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


# Run similation with:
```bash
python3 run_all_scenarios.py \
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
python3 run_replicates.py \
  --replicates-dir ../data/replicate \
  --population ../../va_persontrait_epihiper.csv

# if you want uncertainty
python3 plot_kl_uncertainty.py --root scenario_runs --outdir result_graphs
```

# Run lasso test with:
```bash
python3 lasso_test_greedy.py \
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
```

# Run lite version with:
```bash
# week 1:
python3 run_weekly_sampling.py \
  --linelist ../data/linelist.csv \
  --population ../../va_persontrait_epihiper.txt \
  --target "LL" \
  --current-date 2021-05-31 \
  --batch-size 100 \
  --algorithms surs \
  --no-replacement

# week 2:
python3 run_weekly_sampling.py \
  --linelist ../data/linelist.csv \
  --population ../../va_persontrait_epihiper.txt \
  --already-sequenced weekly_results/ \
  --target "LL"  \
  --current-date 2021-06-07 \
  --batch-size 100 \
  --algorithms surs \
  --no-replacement
```