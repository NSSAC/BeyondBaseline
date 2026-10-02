# Where the CSTE notebook's inputs come from now

`scripts/scenarios_simulation/analysis/cste_and_beyond.ipynb` is an analysis
notebook, not part of the generation pipeline. It reads files by path. Some of
those files are now written by PhyloGAS instead of BeyondBaseline, so this note
records what moved and why.

Nothing was deleted; every metric still exists.

---

## Why anything moved

BeyondBaseline must be runnable by a health department on a real line list,
with no agent-based model behind it. Metrics that compare against the ABM's
hidden truth cannot work in that setting, and keeping them here forced
`--infections` to be mandatory even for purely demographic measures.

The dividing line is now **ground truth**:

| needs | lives in |
|---|---|
| only the line list the sampler was handed | BeyondBaseline |
| the ABM's true infections / transmission graph | PhyloGAS |

---

## Input-by-input

| notebook reads | produced by, before | produced by, now |
|---|---|---|
| `replicate_*/KL_series.csv` | BeyondBaseline | **unchanged** |
| `replicate_*/AUC_rankings.csv` | BeyondBaseline | BeyondBaseline, **fewer `eval_type` rows** (see below) |
| `*_scenario{N}_{ALGO}_samples.csv.xz` | BeyondBaseline `--save-samples` | **unchanged** |
| `replicate_*/Mugration_Metrics.csv` | BeyondBaseline | `phylogas benchmark mugration` |
| `replicate_*/AUC_truth_rankings.csv` | *(did not exist)* | `phylogas benchmark truth` |

### `eval_type` rows

Still in `AUC_rankings.csv`:

| eval_type | metric |
|---|---|
| `A_targets` | KL vs census / line list / blended |
| `N_equity_*` | per-age-group coverage equity |

Moved to `AUC_truth_rankings.csv`:

| eval_type | why |
|---|---|
| `B_cumulative_infections` | KL against true infection counts |
| `C_stride_window_infections` | same, per stride |
| `E_stride_variant_prevalence_error` | prevalence MAE vs true variant counts |
| `F_stride_component_coverage` | needs the transmission graph |
| `M_8_week_rolling_tree_coverage` | same |
| `*_coverage_size_*` | same |

Figure D (weekly pool / infection / sample ratios) also moved: it divides by
true infection counts.

---

## Column rename

`variant_label` is now `variant_benchmark`, and it is written by PhyloGAS
rather than TwinSampler.

The name says what it is. These labels are matched against a real importation
schedule to give prevalence estimation something to be scored against; they
are deliberately independent of the lineage a sequence's genome implies. Where
both exist, their disagreement is itself a measurable quantity.

Readers accept the old name as a fallback, so existing files keep working.

---

## Before and after

**Before** — one command, but ABM data was mandatory:

```bash
python3 run_all_scenarios.py \
  --linelist      replicate_0/linelist.csv.xz \
  --population    va_persontrait_epihiper.txt \
  --infections    replicate_0/linelist_allevents.csv.xz \
  --abm_mugration replicate_0/linelist_mugration.json \
  --outdir        replicate_results/replicate_0 \
  --batch-size 100 --no-replacement --seed 42 --save-samples --no-plots \
  --algorithms surs stratified LASSO-Greedy LASSO-Stratified \
  --stratifiers age race county sex
```

**After** — selection, then scoring:

```bash
# 1. assign benchmark variants (PhyloGAS; was TwinSampler's --variant_mode)
phylogas assign-variants \
    --allevents replicate_0/linelist_allevents.csv.xz \
    --schedule  Virginia_importation_schedule.csv \
    --mode bipartite \
    --out replicate_0/linelist_allevents_variants.csv.xz

# 2. selection (BeyondBaseline; --infections no longer required)
beyond-baseline-sweep \
  --linelist   replicate_0/linelist.csv.xz \
  --population va_persontrait_epihiper.txt \
  --outdir     replicate_results/replicate_0 \
  --batch-size 100 --no-replacement --seed 42 --save-samples --no-plots \
  --algorithms surs stratified LASSO-Greedy LASSO-Stratified \
  --stratifiers age race county sex

# 3. scoring against truth (PhyloGAS)
phylogas benchmark truth \
    --samples    'replicate_results/replicate_0/*_samples.csv.xz' \
    --infections replicate_0/linelist_allevents_variants.csv.xz \
    --out        replicate_results/replicate_0/AUC_truth_rankings.csv

phylogas benchmark mugration \
    --truth      replicate_0/linelist_mugration.json \
    --samples    'replicate_results/replicate_0/*_samples.csv.xz' \
    --infections replicate_0/linelist_allevents_variants.csv.xz \
    --out        replicate_results/replicate_0/Mugration_Metrics.csv
```

Step 3 builds the transmission graph once and reuses it across every sample
set, so scoring N strategies costs about the same as scoring one.

---

## Notebook changes needed

Minimal. If step 3 writes into the same per-replicate directory the sweep
used, the existing globs keep working. Two additions:

```python
# alongside the existing AUC_rankings.csv glob
truth = sorted(ROOT.glob("replicate_*/AUC_truth_rankings.csv"))

# variant column, tolerating both names
vcol = "variant_benchmark" if "variant_benchmark" in df.columns else "variant_label"
```

---

## Verified

The ported code is verbatim, and parity was checked before and after:

* **Variant assignment** — both matchers (`temporal`, `bipartite`) produce an
  identical `component_id -> variant` mapping on a 4,000-event / 2,236-component
  fixture, via the library call and the `phylogas assign-variants` CLI.
* **Operational sampling** — unchanged: `KL=0.120586` on the same fixture
  before and after.

## Known issue, pre-existing

BeyondBaseline's built-in plotting path has `NameError`s (`scenario_ids`,
`n_algo`, `algo_list` referenced before assignment) that predate this work —
every recorded production invocation passes `--no-plots`, so it was never hit.
Rather than half-fix it while moving metrics out, it now prints a notice and
skips. Metric CSVs are unaffected; the notebook plots from those anyway.
