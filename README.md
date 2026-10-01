# BeyondBaseline

**Decide which cases to sequence next, under a fixed laboratory budget.**

Genomic surveillance programs can sequence only a fraction of reported cases.
Which fraction you pick determines what you can see. BeyondBaseline takes a
line list, a population, and a weekly budget, and returns a ranked list of
cases to sequence — balancing demographic representativeness against
visibility of transmission structure.

It runs in two modes:

| mode | for | needs |
|---|---|---|
| **Operational** | a health department choosing this week's samples | a real line list + population table |
| **Research** | evaluating sampling strategies against known truth | simulated data with ground truth |

The operational mode needs no simulation, no digital twin, and no agent-based
model. It works on the data a surveillance program already has.

---

## Install

```bash
pip install -e .
```

Verify:

```bash
python scripts/scenarios_simulation/verify_cli_entrypoints.py
```

---

## Operational mode: `beyond-baseline`

One command, one week, one decision.

```bash
beyond-baseline \
    -l linelist.csv \
    -p population.txt \
    -b 400 \
    --current-date 2021-07-12 \
    --target "LL,P" \
    --time-budget 4 \
    --algorithm SURS \
    --outdir week28/
```

| flag | meaning |
|---|---|
| `-l, --linelist` | reported cases: one row per case, with a date and demographics |
| `-p, --population` | the population you are sampling from (census or synthetic) |
| `-b, --batch-size` | how many sequences you can run this week |
| `--current-date` | Monday of the week being decided |
| `--target` | what to match: `LL` line list, `P` population, `LL,P` blended |
| `--time-budget` | rolling pool window, in weeks |
| `--algorithm` | one of the strategies below |
| `-g, --already-sequenced` | last week's output, so cases are not re-picked |

### What you get

```
outdir/
├── samples_new_scen99_SURS.csv      the recommendation: which cases to sequence
├── history_updated_scen99_SURS.csv  cumulative record, feeds next week's -g
├── history_combined_all.csv         same, across all algorithms run
└── weekly_summary.txt               human-readable report
```

and on stdout:

```
Target: LL,P → 4S-4(LL,P)
  pool_window=4, decision_window=4, blend_alpha=0.5

  SURS   | pool=  1870 | batch=  400 | sampled=  400 | KL=0.120586 | 0.01s

Combined history: week28/history_combined_all.csv (400 rows)
  Next week: --already-sequenced week28/history_combined_all.csv
```

The KL figure is divergence between your chosen samples and the target
distribution — lower is more representative.

### Running week after week

Feed the previous week's history back in, so the sampler knows what has
already been sequenced and does not pick it again:

```bash
# week 28
beyond-baseline -l linelist.csv -p population.txt -b 400 \
    --current-date 2021-07-12 --target "LL,P" --algorithm SURS --outdir week28/

# week 29
beyond-baseline -l linelist.csv -p population.txt -b 400 \
    --current-date 2021-07-19 --target "LL,P" --algorithm SURS \
    -g week28/history_combined_all.csv --no-replacement --outdir week29/
```

### Choosing a target

| `--target` | matches | use when |
|---|---|---|
| `LL` | the recent line list | you want samples to mirror who is actually being reported |
| `P` | the underlying population | you want to correct for ascertainment bias |
| `LL,P` | a blend (`--blend-alpha`, default 0.5) | the usual compromise |

### Algorithms

| name | shorthand | behaviour |
|---|---|---|
| `SURS` | `surs` | simple uniform random sampling — the common baseline |
| `Stratified` | | rigid demographic quotas |
| `Uniform Random` | | uniform with a minimum per group |
| `Greedy` | | greedy KL minimisation against the target |
| `LASSO-Stratified` | `ls` | sparse risk modelling, then stratified |
| `LASSO-Greedy` | `lg` | sparse risk modelling, then greedy |
| `RL` | | Gittins/UCB bandit (experimental) |

`--algorithm` is repeatable, so several can be compared in one run.

### Why not just use SURS?

Enforcing strict demographic quotas corrects ascertainment bias, but it also
severs transmission links — you end up with a demographically tidy sample that
cannot see how the pathogen is actually moving between places. The LASSO
methods target this directly, using sparse risk modelling to find
high-leverage "boundary" cases that connect otherwise separate groups.

That trade-off between demographic equity and topological visibility is what
this software exists to navigate.

---

## Research mode: `beyond-baseline-sweep`

Evaluates many strategies across many scenarios against simulated data, where
the true transmission network is known.

```bash
beyond-baseline-sweep \
    --linelist   replicate_0/linelist.csv.xz \
    --population va_persontrait_epihiper.txt \
    --infections replicate_0/linelist_allevents.csv.xz \
    --outdir     runs/ \
    --batch-size 400 --no-replacement --seed 42 --save-samples \
    --algorithms surs stratified LASSO-Greedy LASSO-Stratified \
    --stratifiers age race county sex
```

Outputs `KL_series.csv`, `AUC_rankings.csv` and, with `--save-samples`, the
selected cases per scenario and algorithm.

Multiple replicates:

```bash
beyond-baseline-replicates --replicates-dir data/replicates --population pop.txt
beyond-baseline-aggregate  --input-dir replicate_results --outdir replicate_results
```

### Benchmarking lives in PhyloGAS

Metrics that require agent-based-model ground truth — mugration cosine
similarity, topological F1 — moved to
[PhyloGAS](https://github.com/NSSAC/PhyloGAS), so that BeyondBaseline stays
runnable on a real line list with no simulation attached.

```bash
beyond-baseline-sweep ... --save-samples --outdir runs/
phylogas benchmark mugration --truth <abm.json> \
    --samples 'runs/*_samples.csv.xz' --infections <allevents.csv.xz>
```

See [`cste_instructions.txt`](cste_instructions.txt) for the full before/after
mapping.

---

## Command reference

| command | purpose |
|---|---|
| `beyond-baseline` | **operational** — one week, one decision |
| `beyond-baseline-sweep` | research — many scenarios × algorithms |
| `beyond-baseline-replicates` | research — drive multiple replicates |
| `beyond-baseline-aggregate` | research — aggregate across replicates |
| `beyond-baseline-lasso-greedy` | LASSO-Greedy group-size sweep |
| `beyond-baseline-lasso-stratified` | LASSO-Stratified group-size sweep |

Older names remain available and behave identically, so existing sbatch
scripts keep working:

| deprecated | use instead |
|---|---|
| `scenarios-weekly` | `beyond-baseline` |
| `scenarios-runner` | `beyond-baseline-sweep` |
| `scenarios-replicates` | `beyond-baseline-replicates` |
| `scenarios-aggregate` | `beyond-baseline-aggregate` |
| `scenarios-lasso-greedy` | `beyond-baseline-lasso-greedy` |
| `scenarios-lasso-stratified` | `beyond-baseline-lasso-stratified` |

Running the scripts directly (`python3 run_all_scenarios.py ...`) also still
works.

---

## Input format

**Line list** — one row per reported case:

| column | notes |
|---|---|
| `date` | report or collection date (`--date-field` to rename) |
| `age_group`, `sex`, `smh_race`, `county_fips` | whichever you pass to `--stratifiers` |
| `alias_pid` | case identifier |
| `alias_contact` | infector, if known — enables coverage metrics |

**Population** — the sampling frame, in EpiHiper persontrait format: a JSON
schema line, then a header with `pid`, `gender`, `county_fips`, `smh_race`,
`age_group`.

`--stratifiers` accepts the short names `age`, `race`, `county`, `sex`, `ses`,
which map onto `age_group`, `smh_race`, `county_fips`, `sex`, `ses_category`.

---

## How evaluation works

Three independent axes:

1. **Demographic representativeness** — KL divergence of the sample against
   census, line list, or blended targets.
2. **Variant accuracy** — absolute error in estimated variant prevalence.
3. **Transmission topology** — Mean Reciprocal Distance for tree coverage;
   cosine similarity and topological F1 for geographic flow (these last two
   now computed by PhyloGAS, which holds the ground truth).

Supports sliding windows, multi-week strides (`4S-4`), no-replacement pooling,
and SLURM array submission for large sweeps.

---

## Components

| file | role |
|---|---|
| `run_weekly_sampling.py` | operational mode |
| `run_all_scenarios.py` | research sweeps |
| `sampling_algorithms.py` | the strategies, including proportional backfill and LASSO macro-bucketing |
| `scenarios_config.py` | scenario definitions and defaults |
| `aggregate_results.py` | cross-replicate aggregation; median Z-scores and AUC rankings |
| `submit_*.sbatch` | SLURM array templates |

---

## Related

* [TwinSampler](https://github.com/NSSAC/TwinSampler) — generates the
  ascertainment-biased line lists used in research mode
* [PhyloGAS](https://github.com/NSSAC/PhyloGAS) — the benchmarking framework
  that consumes BeyondBaseline's selections and scores them against ground truth
