# Unapplied patches

## 0001-surs-seeding.patch

Makes SURS actually seedable. Two changes to `run_one_scenario`:

- `state["week_idx"]` is now set. `pure_uniform_sampler` reads
  `state.get("week_idx", 0)` but the driver wrote `state["week_id"]`, so the
  per-week rehash never happened.
- `state["base_seed"]` is drawn from the per-algorithm RNG instead of
  defaulting to `0`. That RNG already derives from `rng_master` by a stable
  split and SURS otherwise discards it, so no caller plumbing changes.

### Why this is not applied

It changes existing SURS samples, 4S more than 1S, so it must land at a
deliberate boundary rather than alongside a no-op refactor.

### Why it matters

`pure_uniform_sampler` makes zero `rng.` calls. With both seed inputs frozen
it is a pure function of the line list, so within-output sampling replicates
are impossible: several draws on one line list come back identical. SURS would
report a structural zero variance while every rng-using sampler (LASSO-Greedy
4 calls, Stratified 1, Greedy 2) showed real spread. Nothing errors, and the
artifact reads as "SURS is remarkably stable".

Across-output replicates work today only because `run_replicates.py` iterates
over separate EpiHiper input folders, changing the row indices and so the
hashes. That is the data varying, not the sampler being seeded.

### What it does not change

The current samples are not biased. The hash is pseudorandom with respect to
every covariate. Fixed priority plus `no_replacement` does tilt an overlapping
pool toward its newest week, but at ~400 drawn from a ~167k 4-week pool
(0.24%) the depletion is negligible, and for 1S the pools are disjoint so it
does not arise at all. This is a prerequisite for replicates, not a fix to
published numbers.

### To apply

    git apply patches/0001-surs-seeding.patch

Then expect `4S__surs_samples.csv.xz` to change, and wire
`sampling.replicates` (currently read by nothing) before relying on it.
