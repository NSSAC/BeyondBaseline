#!/usr/bin/env python3
# run_all_scenarios.py (history-pool + no-replacement + budgets + AUC + 1x3 figs)
from __future__ import annotations
import argparse
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from scipy.stats import pearsonr
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score

import sys
# Directory holding the sibling modules, for the flat-import fallback below.
_FLAT_IMPORT_DIR = Path(__file__).resolve().parent

try:
    from .recipes import algo_slug, recipe_id, sample_filename
    from .scenarios_config import (
        SCENARIOS,
        SCEN_LABELS,
        GROUP_FEATURES,
        DATE_FIELD_DEFAULT,
        START_DATE_DEFAULT,
        MINIMUM_POOL_SIZE_DEFAULT,
    )
    from .sampling_algorithms import make_group, kl_dist, ALGORITHMS as REGISTRY
except ImportError as exc:
    # Fall back to flat imports only when there is no parent package, which is
    # what running this file directly produces -- that ImportError carries no
    # module name. A dependency missing from inside one of the modules above
    # (scipy, say) does carry one, and must propagate: swallowing it here made
    # it resurface from the fallback as a misleading "No module named
    # <sibling>".
    if exc.name is not None:
        raise
    sys.path.insert(0, str(_FLAT_IMPORT_DIR))
    from recipes import algo_slug, recipe_id, sample_filename
    from scenarios_config import (
        SCENARIOS,
        SCEN_LABELS,
        GROUP_FEATURES,
        DATE_FIELD_DEFAULT,
        START_DATE_DEFAULT,
        MINIMUM_POOL_SIZE_DEFAULT,
    )
    from sampling_algorithms import make_group, kl_dist, ALGORITHMS as REGISTRY


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run scenarios 1-8; save 3 images (each 1x3). Also outputs AUC rankings. Infections required."
    )
    ap.add_argument("--linelist", required=True, help="Path to simulated_test_positive_linelist.csv")
    ap.add_argument("--population", required=True, help="Path to va_persontrait_epihiper.csv")
    ap.add_argument("--infections", required=False, default=None,
                    help="ABM all-events file. OPTIONAL since the ground-truth metrics "
                         "moved to PhyloGAS; only needed if the linelist lacks "
                         "alias_contact edges for the coverage figures.")
    ap.add_argument("--date-field", default=DATE_FIELD_DEFAULT, help=f"Linelist date column (default: {DATE_FIELD_DEFAULT})")
    ap.add_argument("--start-date", default=str(START_DATE_DEFAULT.date()), help=f"Week slicing anchor date (default: {START_DATE_DEFAULT.date()})")
    ap.add_argument("--min-pool", type=int, default=MINIMUM_POOL_SIZE_DEFAULT, help=f"Minimum weekly pool size (default: {MINIMUM_POOL_SIZE_DEFAULT})")
    ap.add_argument("--outdir", default="result", help="Output directory for CSVs/plots (default: result)")
    ap.add_argument("--outname", default=None, help="Optional basename prefix for all output files.")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    ap.add_argument("--roll-win-inf", type=int, default=4, help="Rolling window (weeks) for infections Plot 3 (default: 4)")
    ap.add_argument("--abm_mugration", required=False,
                    help=argparse.SUPPRESS)  # removed: see cste_instructions.txt

    # ---- Sampling budget overrides ----
    ap.add_argument("--batch-size", type=int,
                    help="Fixed weekly sampling budget N. If set, overrides fraction/cap for all scenarios.")
    ap.add_argument("--batch-frac", type=float,
                    help="Override scenario batch_frac (0.0–1.0) for all scenarios.")
    ap.add_argument("--batch-cap", type=int,
                    help="Override scenario batch_cap for all scenarios.")
    ap.add_argument("--min-per-group", type=int,
                    help="Override scenario min_per_group for all scenarios.")
    ap.add_argument("--min-coverage-frac", type=float,
                    help="Fractional min coverage per group for the 'Uniform Random' sampler (0<frac<=1). Default: 0.05")
    # Add a flag to disable plots
    ap.add_argument("--no-plots", action="store_true",
                    help="If set, disables the generation of all PNG plot files.")

    # Add a flag to enable saving the selected samples
    ap.add_argument("--save-samples", action="store_true",
                    help="If set, saves the full metadata for selected samples for each scenario and algorithm.")

    # ---- No-replacement across weeks (per algorithm) ----
    ap.add_argument("--no-replacement", action="store_true",
                    help="If set, do not re-sample the same row across weeks (per algorithm).")
    
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=["surs", "greedy", "stratified"],
        help=("Algorithms to run (space- or comma-separated). "
            "Accepts names or aliases, e.g.: surs, greedy, stratified, rl, "
            "'uniform random'. Default: surs greedy stratified"),
    )

    ap.add_argument(
        "--stratifiers",
        nargs="+",
        default=["age", "race", "county", "sex"],
        help=("Stratifier fields to build the group key. "
            "Allowed (case-insensitive): age, race, county, sex. "
            "Default: age race county sex"),
    )

    args = ap.parse_args()

    if getattr(args, "abm_mugration", None):
        ap.error(
            "--abm_mugration was removed; mugration benchmarking now lives in PhyloGAS.\n"
            "  Equivalent two-command workflow:\n"
            "    beyond-baseline-sweep ... --save-samples --outdir runs/\n"
            "    phylogas benchmark mugration --truth <abm.json> \\\n"
            "        --samples 'runs/*_samples.csv.xz' --infections <allevents.csv.xz>\n"
            "  See cste_instructions.txt."
        )
    return args

# --------- algorithm selection helpers ---------
# Map common aliases (case-insensitive) to registry keys
ALGO_ALIASES = {
    "surs": "SURS",
    "pure_uniform": "SURS",

    "greedy": "Greedy",

    "stratified": "Stratified",

    "rl": "RL",

    "uniform random": "Uniform Random",
    "uniform_random": "Uniform Random",
}

def _normalize_algo_name(name: str) -> str:
    key = name.strip().lower()
    if key in ALGO_ALIASES:
        return ALGO_ALIASES[key]
    # Accept every registry name by its recipe slug too ("lasso_greedy" ->
    # "LASSO-Greedy"). The slug is the cross-repo spelling: it is what
    # recipe ids and sample filenames carry and what PhyloGAS passes from
    # sampling.algorithms. Deriving it from the registry means a newly added
    # sampler is reachable by slug without another alias entry -- the LASSO
    # samplers had none, so --algorithms lasso_greedy failed outright.
    by_slug = {algo_slug(n): n for n in REGISTRY}
    return by_slug.get(algo_slug(name), name.strip())

def select_algorithms(registry: dict, requested: list[str]) -> dict:
    """
    Resolve aliases, validate against registry, preserve requested order, dedupe.
    Supports comma-separated items in the list (e.g., ['surs,greedy', 'stratified']).
    """
    # flatten possible comma-separated tokens
    tokens: list[str] = []
    for item in requested or []:
        if isinstance(item, str):
            tokens.extend([t for t in item.split(",") if t.strip()])
        else:
            tokens.append(item)

    # map aliases -> registry keys (or keep as-is if already exact)
    normalized = [_normalize_algo_name(t) for t in tokens]

    # validate
    missing = [n for n in normalized if n not in registry]
    if missing:
        avail = ", ".join(registry.keys())
        raise ValueError(f"Unknown algorithms: {missing}. Available: {avail}")

    # preserve order + dedupe
    selected: dict = {}
    for n in normalized:
        if n not in selected:
            selected[n] = registry[n]
    return selected

# --------- stratifier helpers ---------
STRAT_ALIAS = {
    "age": "age_group",
    "race": "smh_race",
    # The county NAME, not county_fips. Both the line list and the raw
    # population file carry `county` from the same persontrait column, so the
    # values match by construction. county_fips exists only in the line list:
    # TwinSampler composes it from the household file's admin1+admin2, which
    # the population file alone cannot supply.
    "county": "county",
    "sex": "sex",
    "ses": "ses_category",
}

def _normalize_stratifiers(tokens: list[str]) -> list[str]:
    if not tokens:
        raise ValueError("Empty --stratifiers list.")
    out = []
    for t in tokens:
        # allow comma-separated tokens in a single arg
        for tok in str(t).split(","):
            k = tok.strip().lower()
            if not k:
                continue
            if k not in STRAT_ALIAS:
                raise ValueError(f"Unknown stratifier '{tok}'. Allowed: {list(STRAT_ALIAS.keys())}")
            out.append(STRAT_ALIAS[k])
    # preserve order, dedupe
    seen, ordered = set(), []
    for c in out:
        if c not in seen:
            ordered.append(c); seen.add(c)
    return ordered

# --------- infection tree helpers ----------
def build_undirected_adj(df, pid_col="alias_pid", contact_col="alias_contact"):
    """
    Builds an adjacency list for the entire transmission network (undirected).
    Returns: dict {pid: [neighbor_pids]}
    """
    adj = {}
    
    # Ensure strings
    df[pid_col] = df[pid_col].astype(str)
    df[contact_col] = df[contact_col].astype(str)
    
    for _, row in df.iterrows():
        u = row[pid_col]
        v = row[contact_col]
        
        # Initialize
        if u not in adj: adj[u] = []
        
        # Valid edge check (ignore -1 or self-loops)
        if u and v and u != "-1" and v != "-1" and u != "nan" and v != "nan" and u != v:
            if v not in adj: adj[v] = []
            
            # Add undirected edge
            adj[u].append(v)
            adj[v].append(u)
            
    return adj

def calculate_coverage_score(target_population_set, sampled_set, adj_graph):
    """
    Computes Coverage Score = (1 / |Pt|) * Sum(1 / (d(u, S) + 1))
    using Multi-Source BFS.
    """
    if not target_population_set:
        return 0.0
    
    if not sampled_set:
        return 0.0 # d(u, S) is inf, 1/(inf+1) is 0

    # Multi-Source BFS Initialization
    queue = deque()
    distances = {} # Stores d(u, S)
    
    # Initialize with all sampled nodes that exist in the graph
    for s in sampled_set:
        if s in adj_graph: 
            distances[s] = 0
            # FIX: Append tuple (node, distance)
            queue.append((s, 0))
        # Note: If s is not in adj_graph (isolated), it doesn't help reach others, 
        # but it has distance 0 to itself. This is handled implicitly if s in target_population_set.
        # However, for the BFS to run, we only queue valid graph nodes.

    # BFS
    while queue:
        # FIX: Now this unpacks correctly
        current, dist = queue.popleft()
        
        # Explore neighbors
        if current in adj_graph:
            for neighbor in adj_graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    # FIX: Append tuple (neighbor, new_distance)
                    queue.append((neighbor, dist + 1))
    
    # Calculate Score
    total_score = 0.0
    
    for u in target_population_set:
        # If u was visited, we have a distance.
        if u in distances:
            d = distances[u]
            total_score += 1.0 / (d + 1.0)
        # If u corresponds to a sampled node that was isolated (not in adj_graph),
        # its distance to S is 0 (since it IS in S).
        elif u in sampled_set:
             total_score += 1.0 # 1 / (0 + 1)
        else:
            # d(u, S) = infinity -> term is 0
            total_score += 0.0
            
    return total_score / len(target_population_set)

AGE_GROUP_MAP = {
    "p": "Preschool (0-4)",
    "s": "Student (5-17)",
    "a": "Adult (18-49)",
    "o": "Older adult (50-64)",
    "g": "Senior (65+)",
}


def normalize_age_group_col(df, col="age_group"):
    """Map age_group codes (p/s/a/o/g) to long labels; leave long labels as-is."""
    if col in df.columns:
        raw = df[col]
        # case-insensitive match on single-letter codes
        mapped = (
            raw.astype(str).str.strip().str.lower()
            .map(AGE_GROUP_MAP)
        )
        # keep original values where no mapping applies (already long labels or NaN)
        df[col] = mapped.where(mapped.notna(), raw)
    return df

# ----------------- load & preprocess -----------------
def _check_stratifier_overlap(line_df, pop_df, features, min_overlap=0.5):
    """Refuse to compare a line list against a denominator it cannot match.

    Every KL metric here is the sampled distribution against the population
    distribution over the same group key. If a stratifier's categories differ
    between the two frames, the keys never line up and the divergence is
    computed against an effectively empty denominator -- which looks like a
    number rather than an error. That is what an unguarded .map() on
    smh_race produced: all-NaN in the population, "White" in the line list.
    """
    problems = []
    for col in features:
        a = set(line_df[col].dropna().unique()) - {"", "nan", "None"}
        b = set(pop_df[col].dropna().unique()) - {"", "nan", "None"}
        if not a or not b:
            problems.append(f"{col}: {'line list' if not a else 'population'} "
                            f"has no usable values")
            continue
        shared = a & b
        frac = len(shared) / min(len(a), len(b))
        if frac < min_overlap:
            problems.append(
                f"{col}: only {len(shared)} of {min(len(a), len(b))} categories "
                f"shared ({frac:.0%}). line list e.g. {sorted(a)[:3]}, "
                f"population e.g. {sorted(b)[:3]}")
    if problems:
        raise ValueError(
            "line list and population disagree on stratifier categories, so "
            "the KL metrics would be measured against a denominator that "
            "cannot match:\n  " + "\n  ".join(problems))


def load_linelist_and_population(linelist_path, population_path, date_field, start_date, min_pool, features: list[str]):
    line_df = pd.read_csv(linelist_path, parse_dates=[date_field], dtype={'alias_pid': str, 'alias_contact': str, 'sim_pid': str, 'pid': str, 'contact_pid': str})
    #read population_path using read csv. but look ahead if first line is JSON then skip it.
    with open(population_path, 'r') as f:
        first_line = f.readline()
        if first_line.strip().startswith("{"):
            # It's JSON, so skip it and read the rest as CSV
            pop_df = pd.read_csv(f, dtype={'sim_pid': str, 'pid': str})
        else:
            # Not JSON, so read from the beginning
            pop_df = pd.read_csv(population_path, dtype={'sim_pid': str, 'pid': str})
    # Normalize age_group in both population and linelist (handles codes or long labels)
    pop_df  = normalize_age_group_col(pop_df,  "age_group")
    line_df = normalize_age_group_col(line_df, "age_group")

    pop_df = pop_df.rename(columns={"gender": "sex"})
    pop_df["sex"]      = pop_df["sex"].astype(str).map({"1": "male", "2": "female"})
    # .fillna(original) because the persontrait file already holds expanded
    # names ("White"), which are not keys here -- an unguarded .map() turned
    # every population row's race into NaN while the line list kept "White",
    # so no group key matched and every KL figure was computed against an
    # empty denominator. Same shape of fallback DemographicsLoader uses.
    pop_df["smh_race"] = pop_df["smh_race"].astype(str).map({
        "W": "White", "B": "Black", "L": "Latino", "A": "Asian", "O": "Other"
    }).fillna(pop_df["smh_race"])

    line_df = make_group(line_df, features)
    pop_df  = make_group(pop_df,  features)
    _check_stratifier_overlap(line_df, pop_df, features)
    pop_dist_static = pop_df["group"].value_counts(normalize=True).sort_index()

    # weekly linelist history
    weekly_ll_hist = []
    cur = start_date
    while True:
        prev_mon = cur - timedelta(days=7)
        prev_sun = cur - timedelta(days=1)
        wk = line_df[(line_df[date_field] >= prev_mon) & (line_df[date_field] <= prev_sun)]
        if len(wk) < min_pool:
            break
        weekly_ll_hist.append(wk["group"].value_counts())
        cur += timedelta(weeks=1)

    return line_df, pop_df, pop_dist_static, weekly_ll_hist


def cum_kl_vs_linelist(weekly_sample_hist, weekly_ll_hist):
    cum_s, cum_l = pd.Series(dtype=float), pd.Series(dtype=float)
    out = []
    n = min(len(weekly_sample_hist), len(weekly_ll_hist))
    for i in range(n):
        cum_s = cum_s.add(weekly_sample_hist[i], fill_value=0)
        cum_l = cum_l.add(weekly_ll_hist[i],     fill_value=0)
        out.append(kl_dist(cum_s / cum_s.sum(), cum_l / cum_l.sum()))
    return out

def cum_kl_vs_population(weekly_sample_hist, pop_dist):
    cum_s = pd.Series(dtype=float); out = []
    for wk in weekly_sample_hist:
        cum_s = cum_s.add(wk, fill_value=0)
        out.append(kl_dist(cum_s / cum_s.sum(), pop_dist))
    return out

def roll_kl_vs_linelist(weekly_sample_hist, weekly_ll_hist, window_weeks=4):
    out = []
    n = min(len(weekly_sample_hist), len(weekly_ll_hist))
    for i in range(n):
        s = pd.Series(dtype=float); l = pd.Series(dtype=float)
        start = max(0, i - window_weeks + 1)
        for j in range(start, i + 1):
            s = s.add(weekly_sample_hist[j], fill_value=0)
            l = l.add(weekly_ll_hist[j],     fill_value=0)
        out.append(kl_dist(s / s.sum(), l / l.sum()))
    return out

def linelist_dist_at_week(weekly_ll_hist, week_idx, mode="cumulative", window_weeks=4):
    counts = pd.Series(dtype=float)
    if mode == "cumulative":
        rng = range(0, week_idx + 1)
    else:
        start = max(0, week_idx - window_weeks + 1)
        rng = range(start, week_idx + 1)
    for j in rng:
        if 0 <= j < len(weekly_ll_hist):
            counts = counts.add(weekly_ll_hist[j], fill_value=0)
    return counts / counts.sum() if counts.sum() > 0 else counts

def blended_target(linelist_dist, pop_dist, alpha=0.5):
    if linelist_dist is None or linelist_dist.empty:
        return pop_dist
    tgt = linelist_dist.mul(alpha).add(pop_dist.mul(1 - alpha), fill_value=0.0)
    s = tgt.sum()
    return tgt / s if s > 0 else tgt


def sampling_stride_weeks(scfg):
    if scfg.get("sampling_mode") == "stride":
        default_stride = scfg.get("decision_window_weeks", 1) or 1
        return max(1, int(scfg.get("sampling_stride_weeks", default_stride)))
    return 1


def target_dist_at_week(weekly_ll_hist, pop_dist, scfg, week_idx):
    if scfg.get("target_type") == "blend":
        ll_mode = scfg.get("target_linelist_mode", "cumulative")
        ll_window = scfg.get("target_linelist_window", 4)
        alpha = scfg.get("blend_alpha", 0.5)
        ll_dist = linelist_dist_at_week(weekly_ll_hist, week_idx, ll_mode, ll_window)
        target_dist = blended_target(ll_dist, pop_dist, alpha)
    elif scfg.get("target_mode") == "linelist_dynamic":
        ll_mode = scfg.get("target_linelist_mode", "cumulative")
        ll_window = scfg.get("target_linelist_window", 4)
        target_dist = linelist_dist_at_week(weekly_ll_hist, week_idx, ll_mode, ll_window)
    else:
        target_dist = pop_dist

    if target_dist is None or target_dist.empty:
        target_dist = pop_dist
    return target_dist


def split_samples_by_calendar_week(sample_df, date_field, start_date, num_weeks):
    if sample_df.empty or date_field not in sample_df.columns:
        return {}

    week0_start = start_date - pd.Timedelta(days=7)
    dated = sample_df.copy()
    dated[date_field] = pd.to_datetime(dated[date_field], errors="coerce")
    dated = dated.dropna(subset=[date_field])
    if dated.empty:
        return {}

    dated["_calendar_week_idx"] = ((dated[date_field] - week0_start).dt.days // 7).astype(int)
    dated = dated[(dated["_calendar_week_idx"] >= 0) & (dated["_calendar_week_idx"] < num_weeks)]
    if dated.empty:
        return {}

    out = {}
    for week_idx, week_df in dated.groupby("_calendar_week_idx", sort=True):
        out[int(week_idx)] = week_df.drop(columns=["_calendar_week_idx"]).copy()
    return out


def evaluation_week_numbers(scfg, n_points):
    stride_weeks = sampling_stride_weeks(scfg)
    if scfg.get("eval_metric") == "per_stride_kl" and stride_weeks > 1:
        return list(range(stride_weeks, stride_weeks * n_points + 1, stride_weeks))
    return list(range(1, n_points + 1))

def per_stride_kl_vs_target(weekly_sample_hist, weekly_ll_hist, pop_dist, scfg):
    """
    Per-stride KL:
    - default behavior: each week's sample distribution vs that week's target
    - stride behavior: aggregate a full stride block, then compare it against the
      target distribution for that block endpoint

    This reconstructs the target for each week using the scenario config,
    matching exactly what run_one_scenario computes at sampling time.
    """
    stride_weeks = sampling_stride_weeks(scfg)
    out = []
    if stride_weeks > 1:
        eval_weeks = range(stride_weeks - 1, len(weekly_sample_hist), stride_weeks)
    else:
        eval_weeks = range(len(weekly_sample_hist))

    for i in eval_weeks:
        start_idx = max(0, i - stride_weeks + 1)
        sample_counts = pd.Series(dtype=float)
        for j in range(start_idx, i + 1):
            if 0 <= j < len(weekly_sample_hist):
                sample_counts = sample_counts.add(weekly_sample_hist[j], fill_value=0)

        if sample_counts.sum() == 0:
            out.append(float("nan"))
            continue

        sample_dist = sample_counts / sample_counts.sum()
        target_dist = target_dist_at_week(weekly_ll_hist, pop_dist, scfg, i)

        out.append(kl_dist(sample_dist, target_dist))
    return out

def series_auc(ys, xs=None):
    """Trapezoidal AUC over actual week positions; ignores NaNs. Lower is better."""
    y = np.asarray(list(ys), dtype=float)
    if xs is None:
        x = np.arange(1, len(y) + 1, dtype=float)
    else:
        x = np.asarray(list(xs), dtype=float)
        if len(x) != len(y):
            raise ValueError("series_auc requires xs and ys to have the same length.")
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(y[m], x[m]))

# SCEN_LABELS = {
#     1: "CS-C(LL)", 2: "RS-R(LL)", 3: "RS-C(LL)",
#     4: "CS-C(LL,P)", 5: "RS-R(LL,P)", 6: "RS-C(LL,P)",
#     7: "CS-P", 8: "RS-P",
# }


# ----------------- scenario runner (seeded) -----------------
def run_one_scenario(line_df, date_field, pop_dist_static, weekly_ll_hist,
                     scfg, rng_master, start_date, min_pool, overrides=None,
                     algorithms: dict[str, callable] = None,
                     history_list: list = None):
    """
    overrides: dict with optional keys:
      - batch_size_fixed: int
      - batch_frac: float
      - batch_cap: int
      - min_per_group: int
      - no_replacement: bool
    """
    algorithms = algorithms or REGISTRY
    history_list = history_list or []
    overrides = overrides or {}
    ov_fixed = overrides.get("batch_size_fixed", None)
    ov_frac  = overrides.get("batch_frac", None)
    ov_cap   = overrides.get("batch_cap", None)
    ov_mpg   = overrides.get("min_per_group", None)
    ov_norep = bool(overrides.get("no_replacement", False))

    weekly_hist = {algo: [] for algo in algorithms.keys()}
    weekly_samples = {algo: [] for algo in algorithms.keys()}

    per_algo_eval, per_algo_time = {}, {}

    # per-algorithm child RNG (stable split)
    algo_rngs = {name: np.random.default_rng(rng_master.integers(0, 2**63 - 1)) for name in algorithms.keys()}

    for algo_name, sampler in algorithms.items():
        t0 = time.perf_counter()
        state = {}
        stride_weeks = sampling_stride_weeks(scfg)

        if algo_name == "SURS":
            # pure_uniform_sampler makes no rng calls: it selects on a hash of
            # (row index, base_seed, week_idx). Both seed inputs were frozen --
            # base_seed defaulted to 0 with nothing ever setting it, and
            # week_idx was never set because the driver wrote "week_id" -- so
            # SURS was a pure function of the line list. Across-replicate
            # variation worked only because run_replicates.py changes the input
            # folder, and so the row indices.
            #
            # That made within-output sampling replicates impossible: several
            # draws on one line list are identical, so SURS would show a
            # structural zero variance while every rng-using sampler showed
            # real spread -- an artifact that reads as a finding. Seeded from
            # the per-algorithm rng instead, which already derives from
            # rng_master by a stable split and which SURS otherwise ignores.
            # Drawn once, before the week loop, so it is fixed within a run.
            state["base_seed"] = overrides.get(
                "base_seed", int(algo_rngs[algo_name].integers(0, 2**31 - 1)))

        # For no-replacement: track used base indices (from line_df) per algorithm
        used_idx: set[int] = set()

        dec_win = scfg.get("decision_window_weeks", None)
        recent = deque(maxlen=max(0, (dec_win or 1) - 1))
        current_week = start_date + timedelta(weeks=stride_weeks - 1)
        week_idx_for_target = stride_weeks - 1
        rng = algo_rngs[algo_name]

        # starting bound for "history" pool (all past weeks up to current)
        first_window_start = start_date - pd.Timedelta(days=7)

        while True:
            prev_mon = current_week - timedelta(days=7)
            prev_sun = current_week - timedelta(days=1)
            week_df = line_df[(line_df[date_field] >= prev_mon) & (line_df[date_field] <= prev_sun)]

            # Progress the weekly clock only if the *weekly* pool is viable (unchanged behavior)
            if len(week_df) < min_pool:
                break

            # ----- choose the sampling pool -----
            if scfg.get("pool_mode") == "history":
                # all rows from the first window start through end of current week
                first_window_start = start_date - pd.Timedelta(days=7)
                pool_df = line_df[(line_df[date_field] >= first_window_start) & (line_df[date_field] <= prev_sun)]

            elif scfg.get("pool_mode") == "rolling":
                w = int(scfg.get("pool_window_weeks", 4))
                pool_start = current_week - pd.Timedelta(weeks=w)
                pool_df = line_df[(line_df[date_field] >= pool_start) & (line_df[date_field] <= prev_sun)]

            else:
                # default: current week's pool only
                pool_df = week_df

            # No-replacement: drop rows already used by this algorithm in previous weeks
            if ov_norep or scfg.get("no_replacement", False):
                if len(used_idx) > 0:
                    pool_df = pool_df.drop(index=list(used_idx), errors="ignore")

            # ----- Effective sampling knobs (apply overrides) -----
            eff_frac = ov_frac if ov_frac is not None else scfg["batch_frac"]
            eff_cap  = ov_cap  if ov_cap  is not None else scfg["batch_cap"]
            eff_mpg  = ov_mpg  if ov_mpg  is not None else scfg["min_per_group"]
            budget_multiplier = stride_weeks if scfg.get("sampling_mode") == "stride" else 1

            if ov_fixed is not None:
                batch_size = int(min(max(0, ov_fixed * budget_multiplier), len(pool_df)))
            else:
                batch_size = int(min(eff_frac * len(pool_df), eff_cap * budget_multiplier))

            min_per_group = int(max(0, eff_mpg))

            # If pool exhausted (e.g., due to no-replacement), stop this algorithm gracefully
            if batch_size <= 0 or len(pool_df) == 0:
                break

            # ----- target distribution for this week -----
            target_dist = target_dist_at_week(weekly_ll_hist, pop_dist_static, scfg, week_idx_for_target)

            # ----- restrict target to available groups this week -----
            avail_groups = pool_df["group"].value_counts().index
            if target_dist is None or target_dist.empty:
                target_dist = pop_dist_static

            # Keep only groups that exist in this week's pool
            target_dist = target_dist.reindex(avail_groups).dropna()

            # Renormalize to make it a valid probability distribution
            s = float(target_dist.sum() or 0.0)
            if s > 0:
                target_dist = target_dist / s
            else:
                # Fallback: if all groups were missing (shouldn't happen), assign uniform weights
                target_dist = pd.Series(1.0, index=avail_groups) / len(avail_groups)


            # ----- prior groups -----
            if dec_win is None:
                prior_groups = list(history_list)
                for s in weekly_hist[algo_name]:
                    for g, cnt in s.items():
                        prior_groups.extend([g] * int(cnt))
            else:
                prior_groups = list(history_list) + [g for lst in list(recent) for g in lst]

            # "week_idx" is the key pure_uniform_sampler reads
            # (state.get("week_idx", 0)); "week_id" was a typo that left it at
            # 0 every week, so SURS never rehashed per week. Both are set:
            # other readers use "week_id".
            state["week_idx"] = week_idx_for_target
            state["week_id"] = week_idx_for_target
            state["scenario_id"] = scfg.get("id")
            state["algo_name"] = algo_name

            # ----- sample (seeded) FROM CHOSEN POOL -----
            sample_df = sampler(pool_df, target_dist, batch_size, min_per_group, prior_groups, state, rng)

            # --- recover full linelist rows and preserve ORIGINAL base indices ---
            KEY_COLS = ["alias_pid", "sim_tick"]  
            usable_keys = [k for k in KEY_COLS if k in sample_df.columns and k in pool_df.columns]

            if not usable_keys:
                raise ValueError(f"Sampler '{algo_name}' is missing key columns {KEY_COLS}.")

            # We use a robust merge to perfectly identify the sampled rows regardless of index resets
            pool_df_with_idx = pool_df.copy()
            pool_df_with_idx["_base_idx"] = pool_df_with_idx.index

            keys_df = sample_df[usable_keys].dropna().drop_duplicates()
            sample_df = pool_df_with_idx.merge(keys_df, on=usable_keys, how="inner").copy()

            selected_base_idx = sample_df["_base_idx"].tolist()
            sample_df = sample_df.drop(columns=["_base_idx"], errors="ignore")

            # Update used indices for no-replacement using ORIGINAL indices
            if ov_norep or scfg.get("no_replacement", False):
                used_idx.update(selected_base_idx)

            sample_weeks = split_samples_by_calendar_week(
                sample_df, date_field, start_date, num_weeks=len(weekly_ll_hist)
            )
            block_start_idx = max(0, week_idx_for_target - stride_weeks + 1)

            for calendar_week_idx in range(block_start_idx, week_idx_for_target + 1):
                week_sample_df = sample_weeks.get(calendar_week_idx, sample_df.iloc[0:0].copy())
                weekly_hist[algo_name].append(week_sample_df["group"].value_counts())
                weekly_samples[algo_name].append(week_sample_df)

                if dec_win is not None:
                    recent.append(week_sample_df["group"].tolist())

            current_week += timedelta(weeks=stride_weeks)
            week_idx_for_target += stride_weeks

        per_algo_time[algo_name] = time.perf_counter() - t0

    # evaluation series (for final plots 1a–c)
    for algo, wh in weekly_hist.items():
        metric = scfg.get("eval_metric", "kl_vs_linelist_cum")
        if metric == "kl_vs_linelist_cum":
            ys = cum_kl_vs_linelist(wh, weekly_ll_hist)
        elif metric == "kl_vs_population_cum":
            ys = cum_kl_vs_population(wh, pop_dist_static)
        elif metric == "kl_vs_linelist_rolling":
            win = scfg.get("eval_window_weeks", 4)
            ys = roll_kl_vs_linelist(wh, weekly_ll_hist, window_weeks=win)
        elif metric == "mean_kl_cum":
            a = cum_kl_vs_linelist(wh, weekly_ll_hist)
            b = cum_kl_vs_population(wh, pop_dist_static)
            ys = [(ai + bi) / 2.0 for ai, bi in zip(a, b)]
        elif metric == "per_stride_kl":
            ys = per_stride_kl_vs_target(wh, weekly_ll_hist, pop_dist_static, scfg)
        else:
            raise ValueError(f"Unknown eval_metric: {metric}")
        per_algo_eval[algo] = ys

    return weekly_hist, per_algo_eval, per_algo_time, weekly_samples, state



# ----------------- main -----------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # identifiers so you can aggregate across many runs
    linelist_id = Path(args.linelist).stem
    run_id = f"{linelist_id}__seed{args.seed}"
    output_basename = args.outname.strip() if args.outname else None

    def out_path(filename: str) -> Path:
        return outdir / (f"{output_basename}_{filename}" if output_basename else filename)

    start_date = pd.to_datetime(args.start_date)
    rng_master = np.random.default_rng(args.seed)

    # Build overrides once
    overrides = {
        "batch_size_fixed": args.batch_size,
        "batch_frac": args.batch_frac,
        "batch_cap": args.batch_cap,
        "min_per_group": args.min_per_group,
        "no_replacement": args.no_replacement,
    }

    ALG = select_algorithms(REGISTRY, args.algorithms)
    print("Running algorithms:", ", ".join(ALG.keys()))

    selected_features = _normalize_stratifiers(args.stratifiers)
    print("Stratifiers (in order):", ", ".join(selected_features))

    # Load core inputs
    line_df, pop_df, POP_DIST_STATIC, weekly_ll_hist = load_linelist_and_population(
        args.linelist, args.population, args.date_field, start_date, args.min_pool, features=selected_features
    )

    # Run scenarios
    scenario_series   = {algo: {} for algo in ALG.keys()}
    total_algo_time   = {algo: 0.0 for algo in ALG.keys()}
    count_algo_runs   = {algo: 0   for algo in ALG.keys()}
    kl_rows = []  # accumulate per-week KL points across all KL metrics
    all_weekly_hist = {} # This will be populated to replace the replay loop
    all_weekly_samples = {}  # scenario_id -> {algo -> [DataFrame per week]}

    # Ground-truth series (true infection counts, true variant prevalence) are
    # no longer built here; those metrics moved to PhyloGAS. See the note below
    # where figures B/C/E/F/I-M used to be.

    # ---------------------------------------------------------------------
    # Run every scenario x algorithm, and save the selected samples.
    #
    # Ground-truth scoring of these samples moved to PhyloGAS; what remains
    # here is the selection itself plus --save-samples, which is how PhyloGAS
    # receives the choices:
    #   phylogas benchmark truth --samples 'runs/*_samples.csv.xz' ...
    # ---------------------------------------------------------------------
    scenario_ids = [scfg["id"] for scfg in SCENARIOS]
    scenario_cfg_map = {scfg["id"]: scfg for scfg in SCENARIOS}
    algo_list  = list(ALG.keys())
    n_algo    = len(algo_list)


    
    for scfg in SCENARIOS:
        label = SCEN_LABELS.get(scfg["id"], "")
        print(f"\n=== Running {scfg['name']} [{label}] ===")
        weekly_hist, per_algo_eval, per_algo_time, weekly_samples, algo_state = run_one_scenario(
            line_df, args.date_field, POP_DIST_STATIC, weekly_ll_hist,
            scfg, rng_master, start_date, args.min_pool, overrides, algorithms=ALG
        )

        # --- FINAL PRINT FOR THIS SCENARIO ---
        print(f"--- Results for {scfg['name']} [{label}] ---")
        for algo_name, sample_weeks_list in weekly_hist.items():
            # Sum up the total samples from all weeks
            total_samples = sum(s.sum() for s in sample_weeks_list)
            num_weeks = len(sample_weeks_list)

            avg_per_week = total_samples / num_weeks if num_weeks > 0 else 0
            
            print(f"  > Algorithm: {algo_name:<15} | Total Samples: {int(total_samples):<6} | "
                  f"Weeks Run: {num_weeks:<3} | Avg/Week: {avg_per_week:.1f}")
        # -------------------------------------

        all_weekly_hist[scfg["id"]] = weekly_hist
        all_weekly_samples[scfg["id"]] = weekly_samples

        # save per-scenario CSV + collect for final plots
        rows = []
        for algo, ys in per_algo_eval.items():
            if scfg["id"] in (4, 5, 6):
                label = f"{algo} Mean KL"
            elif scfg["eval_metric"] == "kl_vs_linelist_rolling":
                label = f"{algo} (Rolling {scfg.get('eval_window_weeks',4)}-Week KL)"
            elif scfg["eval_metric"] == "kl_vs_population_cum":
                label = f"{algo} vs. Population"
            else:
                label = f"{algo} vs. Line List"
            eval_weeks = evaluation_week_numbers(scfg, len(ys))
            for week_num, v in zip(eval_weeks, ys):
                rows.append({"scenario": scfg["id"], "label": label, "week": week_num, "kl": float(v)})
                # kl_targets: KL against that scenario's own target, per week
                kl_rows.append({
                    "run_id": run_id,
                    "linelist_id": linelist_id,
                    "algorithm": algo,
                    "scenario_id": scfg["id"],
                    "scenario_label": label,
                    "eval_type": "kl_targets",
                    "roll_window": None,
                    "week": week_num,
                    "kl": float(v),
                })
            scenario_series[algo][scfg["id"]] = (eval_weeks, ys)
            total_algo_time[algo] += per_algo_time.get(algo, 0.0)
            count_algo_runs[algo] += 1

        for algo, secs in per_algo_time.items():
            print(f"  {algo} time: {secs:.2f}s")

# --- POST-PROCESSING FOR THIS SCENARIO (SAMPLES & MUGRATION) ---
        label = SCEN_LABELS.get(scfg["id"], scfg["name"])
        if args.save_samples:
            print(f"  Saving selected samples for {scfg['name']} [{label}]...")
            
            for algo_name, sample_weeks_list in weekly_samples.items():
                if not sample_weeks_list:
                    continue

                full_sample_df = pd.concat(sample_weeks_list, ignore_index=True)
                sample_prefix = output_basename if output_basename else run_id
                
                # 1. Save Samples
                if args.save_samples:
                    full_sample_df_out = full_sample_df.assign(
                        run_id=run_id,
                        linelist_id=linelist_id,
                        scenario_id=scfg["id"],
                        scenario_name=scfg["name"],
                        algorithm=algo_name,
                    )
                    # The recipe id is the cross-repo contract (recipes.py):
                    # a consumer names it in config and finds this file by it.
                    # Previously the stem came from Path(linelist).stem, which
                    # for linelist.csv.xz is "linelist.csv", plus the seed --
                    # so the name was neither predictable nor referable.
                    rid = recipe_id(scfg["id"], algo_name)
                    sample_out_path = outdir / (
                        f"{output_basename}_{sample_filename(rid)}"
                        if output_basename else sample_filename(rid))
                    full_sample_df_out.to_csv(sample_out_path, index=False, compression="xz")
                    print(f"    - Saved {len(full_sample_df_out)} samples to {sample_out_path.name}")



    # NOTE: mugration benchmarking moved to PhyloGAS (clean break, 2026-09-30).
    # It required ABM ground truth (--infections + --abm_mugration), which a
    # health department running on a real linelist does not have. See
    # cste_instructions.txt for the equivalent two-command workflow.

    marker_map = {1:"o", 2:"s", 3:"D", 4:"^", 5:"v", 6:">", 7:"P", 8:"X"}

    if False: #this isn't needed now that all_weekly_hist is populated in the original run
        print("\nReplaying samples for plotting (seeded) …")
        marker_map = {1:"o", 2:"s", 3:"D", 4:"^", 5:"v", 6:">", 7:"P", 8:"X"}
        algo_list  = list(ALGORITHMS.keys())
        all_weekly_hist = {}
        rng_master2 = np.random.default_rng(args.seed)
        for scfg in SCENARIOS:
            wh, _, _ = run_one_scenario(
                line_df, args.date_field, POP_DIST_STATIC, weekly_ll_hist,
                scfg, rng_master2, start_date, args.min_pool, overrides
            )
            all_weekly_hist[scfg["id"]] = wh  # {algo -> [Series]}

    def _prepare_samples_df(weeks_list):
        if not weeks_list:
            return pd.DataFrame()
        all_samples_df = pd.concat(weeks_list, ignore_index=True)
        if args.date_field in all_samples_df.columns:
            all_samples_df[args.date_field] = pd.to_datetime(all_samples_df[args.date_field], errors="coerce")
        return all_samples_df
    
    # Small helpers for stride-aligned evaluations
    def _calendar_week_bounds(week_idx):
        anchor = start_date + timedelta(weeks=week_idx)
        return anchor - timedelta(days=7), anchor - timedelta(days=1)

    def _calendar_window_bounds(start_idx, end_idx):
        window_start, _ = _calendar_week_bounds(start_idx)
        _, window_end = _calendar_week_bounds(end_idx)
        return window_start, window_end

    def _stride_eval_indices(scfg, n_weeks):
        stride = sampling_stride_weeks(scfg)
        return list(range(stride - 1, n_weeks, stride))

    def _sum_hist_window(hist_list, start_idx, end_idx):
        out = pd.Series(dtype=float)
        if not hist_list:
            return out
        upper = min(end_idx, len(hist_list) - 1)
        for j in range(max(0, start_idx), upper + 1):
            out = out.add(hist_list[j], fill_value=0)
        return out

    def _filter_df_by_week_window(df, date_col, start_idx, end_idx):
        if df.empty or date_col not in df.columns:
            return df.iloc[0:0].copy()
        window_start, window_end = _calendar_window_bounds(start_idx, end_idx)
        mask = (df[date_col] >= window_start) & (df[date_col] <= window_end)
        return df.loc[mask]



    def _cum_kl_vs_stride(hist_list, ref_hist_list, scfg):
        n = min(len(hist_list), len(ref_hist_list))
        xs, ys = [], []
        for end_idx in _stride_eval_indices(scfg, n):
            sample_counts = _sum_hist_window(hist_list, 0, end_idx)
            ref_counts = _sum_hist_window(ref_hist_list, 0, end_idx)
            xs.append(end_idx + 1)
            if sample_counts.sum() == 0 or ref_counts.sum() == 0:
                ys.append(np.nan)
                continue
            ys.append(kl_dist(sample_counts / sample_counts.sum(), ref_counts / ref_counts.sum()))
        return xs, ys

    def _window_kl_vs_stride(hist_list, ref_hist_list, scfg, window_weeks=None):
        n = min(len(hist_list), len(ref_hist_list))
        stride = sampling_stride_weeks(scfg)
        win = int(window_weeks or stride)
        xs, ys = [], []
        for end_idx in _stride_eval_indices(scfg, n):
            start_idx = max(0, end_idx - win + 1)
            sample_counts = _sum_hist_window(hist_list, start_idx, end_idx)
            ref_counts = _sum_hist_window(ref_hist_list, start_idx, end_idx)
            xs.append(end_idx + 1)
            if sample_counts.sum() == 0 or ref_counts.sum() == 0:
                ys.append(np.nan)
                continue
            ys.append(kl_dist(sample_counts / sample_counts.sum(), ref_counts / ref_counts.sum()))
        return xs, ys

    def _axes_for_algos(n_algo: int, figsize_per_col=(7, 6)):
        """
        Create a 1 x n_algo row of axes, sharing Y.
        Returns: fig, [axes...]
        """
        fig, axes = plt.subplots(1, n_algo, figsize=(figsize_per_col[0]*n_algo, figsize_per_col[1]), sharey=True)
        if n_algo == 1:
            axes = [axes]
        return fig, list(axes)


    auc_rows = []  # dicts: eval_type, algorithm, scenario_id, scenario_label, weeks, auc

    def _record_series(eval_type, algo, scn, xs, ys, roll_window=None):
        label = SCEN_LABELS.get(scn, f"Scenario {scn}")
        for week_num, v in zip(xs, ys):
            kl_rows.append({
                "run_id": run_id,
                "linelist_id": linelist_id,
                "algorithm": algo,
                "scenario_id": scn,
                "scenario_label": label,
                "eval_type": eval_type,
                "roll_window": roll_window,
                "week": int(week_num),
                "kl": float(v),
            })
        auc_rows.append({
            "eval_type": eval_type,
            "algorithm": algo,
            "scenario_id": scn,
            "scenario_label": label,
            "weeks": len(ys),
            "auc": series_auc(ys, xs),
        })

    if not args.no_plots:
        # NOTE: this path was disabled on 2026-10-01, during the move of the
        # ground-truth metrics to PhyloGAS, on the claim that it had
        # pre-existing NameErrors and had never run. Both were wrong: the
        # move's own first commit (bb8f2fd) removed the definitions of
        # scenario_ids, n_algo and algo_list, which 7307403 later restored,
        # and before the move this block ran whenever plots were on -- it is
        # where every truth metric and the kl_targets AUC were computed.
        # The metrics now live elsewhere (kl_targets AUC below; the rest in
        # `phylogas benchmark truth`), so what remains here is figure
        # drawing only. It has not been re-tested since the move.
        print("\nNOTE: built-in plotting is disabled while it is re-tested after the")
        print("      2026-10-01 metrics move. The metric CSVs are still written; plot")
        print("      from those. Pass --no-plots to silence this.")
        args.no_plots = True

    if False:
        print("\nGenerating plots...")
        marker_map = {sid: m for sid, m in zip(scenario_ids, ["o","s","D","^","v",">","P","X"])}
        algo_list  = list(ALG.keys())
        rng_master2 = np.random.default_rng(args.seed)
        # =================== FIGURE A: targets (1×3) ===================
        figA, axesA = _axes_for_algos(n_algo)
        for ax, algo in zip(axesA, algo_list):
            ax.set_title(f"{algo}: Table-3 Scenarios")
            ax.set_xlabel("Week"); ax.set_ylabel("KL" if ax is axesA[0] else "")
            for scn in scenario_ids:
                x, y = scenario_series[algo][scn]
                label = SCEN_LABELS[scn]
                ax.plot(x, y, marker=marker_map.get(scn, "o"), linestyle="-", label=label)
                # kl_targets AUC is recorded after the plotting block.
            ax.grid(True, linestyle="--", alpha=0.6); ax.legend(ncol=4, fontsize=8); ax.set_xlim(left=0.9)
        figA.tight_layout()
        outA = out_path("kl_targets_table_1xN.png")
        plt.savefig(outA, dpi=150); print(f"Saved: {outA}")
        plt.close(figA)

        # NOTE: figures B, C, E, F, I-M moved to PhyloGAS (2026-10-01).
        # They scored samples against agent-based-model ground truth -- true
        # infection counts, true variant prevalence, and the hidden transmission
        # graph -- none of which a health department has for a real line list.
        # Keeping them here forced --infections to be mandatory.
        #
        #   phylogas assign-variants --allevents <allevents> --schedule <sched> --out <out>
        #   phylogas benchmark truth --samples 'runs/*_samples.csv.xz' \
        #       --infections <allevents> --out AUC_truth_rankings.csv
        #
        # See docs/cste_notebook_inputs.md for the full mapping.

        # NOTE: figure D (weekly pool/infection/sample ratios) also moved to
        # PhyloGAS -- it divides by true infection counts, which only the ABM
        # knows. See docs/cste_notebook_inputs.md.

        # NOTE: figure N (equity: tree coverage per age group) moved to PhyloGAS
        # (2026-10-09) as the equity_<age group> metrics of `phylogas benchmark
        # truth`. It is Mean Reciprocal Distance on the transmission graph,
        # built from alias_contact -- who infected whom -- which a real line
        # list does not carry. It was left here in the first move on the
        # reading that it needed only the line list, and so stopped being
        # computed when this plotting block was disabled.

    else:
        print("\n--no-plots flag detected. Skipping plot generation.")
    
    # =================== kl_targets AUC ===================
    # Computed from the series the scenario loop recorded. It used to be added
    # while drawing figure A, so when plotting was disabled AUC_rankings.csv
    # lost its only remaining metric and the sweep printed "No AUC data was
    # collected" on every run. kl_targets is the ranking this sweep owns -- it
    # needs only the line list -- so it is computed here, plots or not.
    for algo, by_scn in scenario_series.items():
        for scn, (x, y) in by_scn.items():
            auc_rows.append({
                "eval_type": "kl_targets",
                "algorithm": algo,
                "scenario_id": scn,
                "scenario_label": SCEN_LABELS.get(scn, f"Scenario {scn}"),
                "weeks": len(y),
                "auc": series_auc(y, x),
            })

    # ------------------- Save evaluation series for uncertainty bands -------------------
    kl_df = pd.DataFrame(kl_rows)
    kl_out = out_path("KL_series.csv")
    kl_df.to_csv(kl_out, index=False)
    print(f"Saved KL series: {kl_out}")

    # =================== AUC summary CSV (ranked) ===================
    if not auc_rows:
        print("\n[Warning] No AUC data was collected. AUC_rankings.csv will not be created.")
    else:
        auc_df = pd.DataFrame(auc_rows)
        
        # Check if the expected column exists to prevent the KeyError
        if "eval_type" in auc_df.columns:
            # lower AUC is better
            auc_df["rank_overall"] = auc_df.groupby("eval_type")["auc"].rank(method="dense", ascending=True)
            auc_df["rank_within_algo"] = auc_df.groupby(["eval_type", "algorithm"])["auc"].rank(method="dense", ascending=True)

            auc_out = out_path("AUC_rankings.csv")
            auc_df.sort_values(["eval_type", "rank_overall", "algorithm", "scenario_id"]).to_csv(auc_out, index=False)
            print(f"\nSaved AUC rankings: {auc_out}")

            # print top results per evaluation to console
            for et in auc_df["eval_type"].unique():
                top = (auc_df[auc_df["eval_type"] == et]
                    .sort_values(["rank_overall", "algorithm", "scenario_id"])
                    .head(8))
                print(f"\nTop AUCs for {et} (lower is better):")
                for _, r in top.iterrows():
                    print(f"  #{int(r['rank_overall'])}: {r['algorithm']} - {r['scenario_label']} "
                        f"(AUC={r['auc']:.4f}, weeks={int(r['weeks'])})")
        else:
            print("\n[Error] 'eval_type' missing from AUC results. Check metric calculations.")

    # print top results per evaluation to console
    # for et in auc_df["eval_type"].unique():
    #     top = (auc_df[auc_df["eval_type"] == et]
    #            .sort_values(["rank_overall", "algorithm", "scenario_id"])
    #            .head(8))
    #     print(f"\nTop AUCs for {et} (lower is better):")
    #     for _, r in top.iterrows():
    #         print(f"  #{int(r['rank_overall'])}: {r['algorithm']} - {r['scenario_label']} "
    #               f"(AUC={r['auc']:.4f}, weeks={int(r['weeks'])})")

    # print("\n=== Average running time across 8 scenarios (per algorithm) ===")
    # for algo in ALG.keys():
    #     n = max(1, count_algo_runs[algo])
    #     avg_secs = total_algo_time[algo] / n
    #     print(f"{algo}: {avg_secs:.2f}s on average over {n} scenarios")


if __name__ == "__main__":
    main()
