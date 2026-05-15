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

from scenarios_config import (
    SCENARIOS,
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
    ap.add_argument("--infections", required=True, help="Path to infections TSV")
    ap.add_argument("--date-field", default=DATE_FIELD_DEFAULT, help=f"Linelist date column (default: {DATE_FIELD_DEFAULT})")
    ap.add_argument("--start-date", default=str(START_DATE_DEFAULT.date()), help=f"Week slicing anchor date (default: {START_DATE_DEFAULT.date()})")
    ap.add_argument("--min-pool", type=int, default=MINIMUM_POOL_SIZE_DEFAULT, help=f"Minimum weekly pool size (default: {MINIMUM_POOL_SIZE_DEFAULT})")
    ap.add_argument("--outdir", default="result", help="Output directory for CSVs/plots (default: result)")
    ap.add_argument("--outname", default=None, help="Optional basename prefix for all output files.")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    ap.add_argument("--roll-win-inf", type=int, default=4, help="Rolling window (weeks) for infections Plot 3 (default: 4)")
    ap.add_argument("--abm_mugration",  required=False, help="Mugration file to be compared to. If set, enables generation of mugration file for selected scenario/algorithm")

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

    return ap.parse_args()

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
    return ALGO_ALIASES.get(key, name.strip())

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
    "county": "county_fips",
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
        if v and v != "-1" and v != "nan" and v != u:
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

def precompute_component_sizes(adj_graph, all_pids):
    """
    Returns a dict {pid: component_size} for every pid in the graph.
    Uses BFS/DFS to find connected components.
    """
    pid_to_size = {}
    visited = set()
    
    # Ensure all pids are in the map, defaulting to size 1 if isolated/missing from graph
    # (Though adj_graph usually contains everyone if built from linelist)
    for pid in all_pids:
        if pid not in pid_to_size:
            pid_to_size[pid] = 1

    for start_node in adj_graph:
        if start_node not in visited:
            # Found a new component, traverse it to count size
            component_nodes = []
            queue = deque([start_node])
            visited.add(start_node)
            component_nodes.append(start_node)
            
            while queue:
                curr = queue.popleft()
                for nbr in adj_graph.get(curr, []):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
                        component_nodes.append(nbr)
            
            # Assign size to all members
            size = len(component_nodes)
            for node in component_nodes:
                pid_to_size[node] = size
                
    return pid_to_size


# --------- shared helpers ---------
# Map short codes to long labels; keep existing long labels untouched.
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
def load_linelist_and_population(linelist_path, population_path, date_field, start_date, min_pool, features: list[str]):
    line_df = pd.read_csv(linelist_path, parse_dates=[date_field])
    #read population_path using read csv. but look ahead if first line is JSON then skip it.
    with open(population_path, 'r') as f:
        first_line = f.readline()
        if first_line.strip().startswith("{"):
            # It's JSON, so skip it and read the rest as CSV
            pop_df = pd.read_csv(f)
        else:
            # Not JSON, so read from the beginning
            pop_df = pd.read_csv(population_path)
    # Normalize age_group in both population and linelist (handles codes or long labels)
    pop_df  = normalize_age_group_col(pop_df,  "age_group")
    line_df = normalize_age_group_col(line_df, "age_group")

    pop_df = pop_df.rename(columns={"gender": "sex"})
    pop_df["sex"]      = pop_df["sex"].astype(str).map({"1": "male", "2": "female"})
    pop_df["smh_race"] = pop_df["smh_race"].astype(str).map({
        "W": "White", "B": "Black", "L": "Latino", "A": "Asian", "O": "Other"
    })

    line_df = make_group(line_df, features)
    pop_df  = make_group(pop_df,  features)
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


def build_weekly_infections(infections_path, pop_df, start_date, num_weeks_ref, date_col: str = "date"):
    """
    Build weekly infections history aligned to linelist slicing.
    Now requires a real date column in the infections file (default: 'date').
    """
    # Let pandas sniff the delimiter (comma, tab, etc.) and avoid skipping header rows.
    inf = pd.read_csv(infections_path, sep=None, engine="python")
    inf.columns = [c.strip() for c in inf.columns]

    inf = normalize_age_group_col(inf, "age_group")
    if date_col not in inf.columns:
        # try case-insensitive match (e.g., 'Date', 'DATE')
        ci_map = {c.lower(): c for c in inf.columns}
        if date_col.lower() in ci_map:
            date_col = ci_map[date_col.lower()]
        else:
            raise ValueError(
                f"Infections file must include a '{date_col}' column "
                f"(case-insensitive). Found columns: {list(inf.columns)}"
            )

    # Parse dates
    inf[date_col] = pd.to_datetime(inf[date_col], errors="coerce")
    if inf[date_col].isna().all():
        raise ValueError(f"Unable to parse any dates in infections column '{date_col}'.")

    # Map pid -> group using population file
    pid_col = "sim_pid" if "sim_pid" in inf.columns else ("pid" if "pid" in inf.columns else None)
    if pid_col is None:
        raise ValueError("Infections file must contain 'sim_pid' or 'pid' to map to demographic groups.")

    pop_pid_col = "sim_pid" if "sim_pid" in pop_df.columns else "pid"
    if pop_pid_col not in pop_df.columns:
        raise ValueError("Population file must have a 'sim_pid' or 'pid' column to map infections to 'group'.")

    pid_group_map = pop_df[[pop_pid_col, "group"]].dropna()
    inf = inf.merge(pid_group_map, left_on=pid_col, right_on=pop_pid_col, how="left")
    inf = inf.dropna(subset=["group"])

    # Weekly counts aligned to linelist weeks
    weekly_inf_hist = []
    cur = start_date
    for _ in range(num_weeks_ref):
        prev_mon = cur - timedelta(days=7)
        prev_sun = cur - timedelta(days=1)
        mask = (inf[date_col] >= prev_mon) & (inf[date_col] <= prev_sun)
        weekly_inf_hist.append(inf.loc[mask, "group"].value_counts())
        cur += timedelta(weeks=1)

    return weekly_inf_hist, inf

def build_weekly_variant_counts(
    infections_path,
    start_date,
    num_weeks_ref,
    date_col: str = "date",
    variant_col: str = "variant_label",
):
    """
    Build weekly *true* variant counts from the infections file.

    Returns
    -------
    weekly_variant_counts : list of pd.Series
        One entry per week. Each Series has index=variant_label, values=counts.
    """
    inf = pd.read_csv(infections_path, sep=None, engine="python")
    inf.columns = [c.strip() for c in inf.columns]

    # resolve date column (case-insensitive)
    if date_col not in inf.columns:
        ci_map = {c.lower(): c for c in inf.columns}
        if date_col.lower() in ci_map:
            date_col = ci_map[date_col.lower()]
        else:
            raise ValueError(
                f"Infections file must include a '{date_col}' column "
                f"(case-insensitive). Found columns: {list(inf.columns)}"
            )

    if variant_col not in inf.columns:
        raise ValueError(
            f"Infections file must include a '{variant_col}' column for variant labels. "
            f"Found columns: {list(inf.columns)}"
        )

    inf[date_col] = pd.to_datetime(inf[date_col], errors="coerce")
    if inf[date_col].isna().all():
        raise ValueError(f"Unable to parse any dates in infections column '{date_col}'.")

    weekly_variant_counts = []
    cur = start_date
    for _ in range(num_weeks_ref):
        prev_mon = cur - timedelta(days=7)
        prev_sun = cur - timedelta(days=1)
        mask = (inf[date_col] >= prev_mon) & (inf[date_col] <= prev_sun)
        wk = inf.loc[mask]

        if wk.empty:
            weekly_variant_counts.append(pd.Series(dtype=float))
        else:
            counts = wk[variant_col].value_counts()
            weekly_variant_counts.append(counts.astype(float))

        cur += timedelta(weeks=1)

    return weekly_variant_counts, inf



# ----------------- evaluation helpers -----------------
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
SCEN_LABELS = {
    1: "1S–1(LL)", 2: "4S–4(LL)", 3: "1S–1(LL,P)",
    4: "4S–4(LL,P)", 5: "1S–P", 6: "4S–P",
}


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
            state["base_seed"] = overrides.get("base_seed", 0)

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

            state["week_id"] = week_idx_for_target
            state["scenario_id"] = scfg.get("id")
            state["algo_name"] = algo_name

            # ----- sample (seeded) FROM CHOSEN POOL -----
            sample_df = sampler(pool_df, target_dist, batch_size, min_per_group, prior_groups, state, rng)

            # --- recover full linelist rows and preserve ORIGINAL base indices ---
            KEY_COLS = ["alias_pid", "sim_tick"]  # prefer both; fall back if needed

            # Case 1: sampler preserved original pool_df indices
            if sample_df.index.isin(pool_df.index).all():
                selected_base_idx = sample_df.index.tolist()
                sample_df = pool_df.loc[selected_base_idx].copy()

            else:
                usable_keys = [k for k in KEY_COLS if k in sample_df.columns and k in pool_df.columns]

                if not usable_keys:
                    raise ValueError(
                        f"Sampler '{algo_name}' did not preserve pool_df index and is missing key columns {KEY_COLS}. "
                        f"Sampler cols={list(sample_df.columns)}; pool cols={list(pool_df.columns)}"
                    )

                # keep original pool indices before merge
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
    kl_rows = []  # accumulate per-week KL points across all panels (A/B/C)
    all_weekly_hist = {} # This will be populated to replace the replay loop
    all_weekly_samples = {}  # scenario_id -> {algo -> [DataFrame per week]}

    # ---------- build infections weekly history ----------
    weekly_inf_hist, full_inf_df = build_weekly_infections(
        args.infections, pop_df, start_date, num_weeks_ref=len(weekly_ll_hist), date_col="date"
    )

    weekly_variant_counts_true, _ = build_weekly_variant_counts(
        args.infections,
        start_date,
        num_weeks_ref=len(weekly_ll_hist),
        date_col="date",
        variant_col="variant_label",
    )

    if args.abm_mugration != None:
        import mugration_station
        print("Mugration analysis")
        county_names, epihiper_matrix = mugration_station.read_traits_json(args.abm_mugration)
        normalized_epihiper_matrix = mugration_station.align_and_normalize_matrix(epihiper_matrix, county_names, county_names)

        print("\nBuilding directed transmission graph for Mugration analysis...")
        pid_col = "alias_pid" if "alias_pid" in full_inf_df.columns else "sim_pid"
        G_dir = mugration_station.build_directed_graph(full_inf_df, pid_col=pid_col, contact_col="alias_contact")
        
        # Establish mapping and alphabet using the full infection list instead of pop_df
        # This guarantees consistent matrix dimensions across all scenarios based on the true outbreak
        if "county" in full_inf_df.columns:
            unique_counties = sorted([str(c) for c in full_inf_df['county'].dropna().unique()])
        else:
            unique_counties = []
            
        alphabet = [""] + unique_counties
        sim_duration_years = len(weekly_ll_hist) / 52.1429
    
    scenario_ids = [scfg["id"] for scfg in SCENARIOS]
    scenario_cfg_map = {scfg["id"]: scfg for scfg in SCENARIOS}
    algo_list  = list(ALG.keys())
    n_algo    = len(algo_list)


    
    for scfg in SCENARIOS:
        print(f"\n=== Running {scfg['name']} ===")
        weekly_hist, per_algo_eval, per_algo_time, weekly_samples, algo_state = run_one_scenario(
            line_df, args.date_field, POP_DIST_STATIC, weekly_ll_hist,
            scfg, rng_master, start_date, args.min_pool, overrides, algorithms=ALG
        )

        # --- FINAL PRINT FOR THIS SCENARIO ---
        print(f"--- Results for {scfg['name']} ---")
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
                # Panel A ("targets"): save KL per week
                kl_rows.append({
                    "run_id": run_id,
                    "linelist_id": linelist_id,
                    "algorithm": algo,
                    "scenario_id": scfg["id"],
                    "scenario_label": label,
                    "eval_type": "A_targets",
                    "roll_window": None,
                    "week": week_num,
                    "kl": float(v),
                })
            scenario_series[algo][scfg["id"]] = (eval_weeks, ys)
            total_algo_time[algo] += per_algo_time.get(algo, 0.0)
            count_algo_runs[algo] += 1

        for algo, secs in per_algo_time.items():
            print(f"  {algo} time: {secs:.2f}s")

        if args.save_samples:
            print(f"  Saving selected samples for {scfg['name']}...")
        for algo_name, sample_weeks_list in weekly_samples.items():
            if not sample_weeks_list:
                continue

            # Use the data directly from the scenario run, which already has full rows
            full_sample_df = pd.concat(sample_weeks_list, ignore_index=True)
            sample_prefix = output_basename if output_basename else run_id
            if args.save_samples:
                # Add your metadata
                full_sample_df = full_sample_df.assign(
                    run_id=run_id,
                    linelist_id=linelist_id,
                    scenario_id=scfg["id"],
                    scenario_name=scfg["name"],
                    algorithm=algo_name,
                )

                sample_out_path = outdir / f"{sample_prefix}_scenario{scfg['id']}_{algo_name}_samples.csv.xz"
                full_sample_df.to_csv(sample_out_path, index=False, compression="xz")
                
                # This print should now match your "Results for Scenario X" log
                print(f"    - Saved {len(full_sample_df)} samples to {sample_out_path}")
            if args.abm_mugration is not None:
                if len(G_dir.nodes()) > 0 and len(alphabet) > 1:
                    print("Computing Mugration matrices for all scenarios/algorithms...")
                    for scfg in SCENARIOS:
                        sid = scfg["id"]
                        weekly_samples_scen = all_weekly_samples.get(sid, {})
                        if not weekly_samples_scen:
                            continue

                        for algo in algo_list:
                            weeks_list = weekly_samples_scen.get(algo, [])
                            if not weeks_list:
                                continue

                            # Combine the in-memory weeks to get the sampled nodes
                            all_samples_df = mugration_station._prepare_samples_df(weeks_list, date_field=args.date_field)
                            if all_samples_df.empty:
                                continue

                            # 1. Prepare Tip States
                            s_pid_col = "alias_pid"
                            
                            if "county" in all_samples_df.columns:
                                known_tips = all_samples_df.set_index(s_pid_col)["county"].to_dict()
                                known_tips = {str(k): str(v) for k, v in known_tips.items() if pd.notna(v)}
                            else:
                                print(f"Warning: 'county' column missing in samples for {algo}. Skipping inference.")
                                continue
                                
                            # Clean dictionary for processing
                            known_tips = {str(k): str(v) for k, v in known_tips.items() if pd.notna(v)}
                            
                            # 2. Run Inference
                            mug_res = mugration_station.simulate_inference_and_matrix(G_dir, known_tips, alphabet, sim_duration_years)
                            
                            # 3. Save Output
                            sample_prefix = output_basename if output_basename else run_id
                            mug_out_path = out_path(f"abmugration_{sample_prefix}_scenario{sid}_{algo}.json")

                            algo_normalized_matrix = mugration_station.align_and_normalize_matrix(mug_res['models']['county']["transition_matrix"], mug_res['models']['county']["alphabet"], county_names)

                            abm_flat = mugration_station.get_off_diagonals(normalized_epihiper_matrix, county_names)
                            run1_flat = mugration_station.get_off_diagonals(algo_normalized_matrix, county_names)

                            # 1. Pearson Correlation
                            r_1, _ = pearsonr(abm_flat, run1_flat)

                            print(f"Run 1 Pearson Correlation: {r_1:.4f}")

                            # 2. Mean Absolute Error (L1 Distance)
                            mae_1 = np.mean(np.abs(abm_flat - run1_flat))

                            print(f"Run 1 MAE: {mae_1:.4f}")
                            
                            with open(mug_out_path, "w") as f:
                                json.dump(mug_res, f, indent=2)
                            print(f"  - Saved Mugration JSON: {mug_out_path.name}")
                else:
                    print("Warning: Graph is empty or no county mapping found. Skipping Mugration JSONs.")



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
                auc_rows.append({
                    "eval_type": "A_targets",
                    "algorithm": algo,
                    "scenario_id": scn,
                    "scenario_label": label,
                    "weeks": len(y),
                    "auc": series_auc(y, x),
                    })
            ax.grid(True, linestyle="--", alpha=0.6); ax.legend(ncol=4, fontsize=8); ax.set_xlim(left=0.9)
        figA.tight_layout()
        outA = out_path("A_table3_targets_1xN.png")
        plt.savefig(outA, dpi=150); print(f"Saved: {outA}")
        plt.close(figA)

        # =================== FIGURE B: cumulative infections (1×3) ===================
        figB, axesB = _axes_for_algos(n_algo)
        for ax, algo in zip(axesB, algo_list):
            ax.set_title(f"{algo}: KL vs Cumulative Infections (Stride-Aligned)")
            ax.set_xlabel("Week"); ax.set_ylabel("KL" if ax is axesB[0] else "")
            for scn in scenario_ids:
                scfg = scenario_cfg_map[scn]
                x, ys = _cum_kl_vs_stride(all_weekly_hist[scn][algo], weekly_inf_hist, scfg)
                if not ys:
                    continue
                label = SCEN_LABELS[scn]
                ax.plot(x, ys, marker=marker_map.get(scn, "o"), linestyle="-", label=label)
                _record_series("B_cumulative_infections", algo, scn, x, ys)
            ax.grid(True, linestyle="--", alpha=0.6); ax.legend(ncol=4, fontsize=8); ax.set_xlim(left=0.9)
        figB.tight_layout()
        outB = out_path("B_vs_cumulative_infections_1xN.png")
        plt.savefig(outB, dpi=150); print(f"Saved: {outB}")
        plt.close(figB)

        # =================== FIGURE C: stride-window infections (1×N) ===================
        figC, axesC = _axes_for_algos(n_algo)
        for ax, algo in zip(axesC, algo_list):
            ax.set_title(f"{algo}: KL vs Stride-Window Infections")
            ax.set_xlabel("Week"); ax.set_ylabel("KL" if ax is axesC[0] else "")
            for scn in scenario_ids:
                scfg = scenario_cfg_map[scn]
                stride_weeks = sampling_stride_weeks(scfg)
                x, ys = _window_kl_vs_stride(
                    all_weekly_hist[scn][algo], weekly_inf_hist, scfg, window_weeks=stride_weeks
                )
                if not ys:
                    continue
                label = SCEN_LABELS[scn]
                ax.plot(x, ys, marker=marker_map.get(scn, "o"), linestyle="-", label=label)
                _record_series("C_stride_window_infections", algo, scn, x, ys, roll_window=stride_weeks)
            ax.grid(True, linestyle="--", alpha=0.6); ax.legend(ncol=4, fontsize=8); ax.set_xlim(left=0.9)
        figC.tight_layout()
        outC = out_path(f"C_vs_rolling{args.roll_win_inf}_infections_1xN.png")
        plt.savefig(outC, dpi=150); print(f"Saved: {outC}")
        plt.close(figC)
    
    
        # =================== FIGURE D: Weekly ratios (3 bars per week) ===================
        # Definitions:
        # - pool_size = LineList size in the current week
        # - infections_size = infections size in the current week
        # - sampled_per_week = samples actually drawn in the replay (pick one scenario+algorithm)
        #
        scenario_for_sampled = 1
        algo_for_sampled = list(ALG.keys())[0]
        
        # Build week-wise counts
        weeks_n = len(weekly_ll_hist)
        pool_weekly = [int(weekly_ll_hist[i].sum()) for i in range(weeks_n)]
        inf_weekly  = [int(weekly_inf_hist[i].sum()) for i in range(weeks_n)]
        
        # Use the replayed samples we already computed: all_weekly_hist[scenario_id][algo] -> list[Series]
        sampled_hist_list = all_weekly_hist.get(scenario_for_sampled, {}).get(algo_for_sampled, [])
        sampled_weekly = [int(s.sum()) if i < len(sampled_hist_list) else 0 for i, s in enumerate(sampled_hist_list + [pd.Series(dtype=float)]*max(0, weeks_n - len(sampled_hist_list)))]
        
        # Safe division helpers
        def _safe_div(num, den):
            return (num / den) if (den is not None and den != 0) else float("nan")
        
        ratio_pool_over_inf     = [_safe_div(pool_weekly[i], inf_weekly[i]) for i in range(weeks_n)]
        ratio_sampled_over_inf  = [_safe_div(sampled_weekly[i], inf_weekly[i]) for i in range(weeks_n)]
        ratio_sampled_over_pool = [_safe_div(sampled_weekly[i], pool_weekly[i]) for i in range(weeks_n)]
        
        # Plot grouped bars
        figD, axD = plt.subplots(figsize=(14, 6))
        x = np.arange(weeks_n) + 1  # week numbers starting at 1
        bar_w = 0.25
        axD.bar(x - bar_w, ratio_pool_over_inf,     width=bar_w, label="pool_size / infections_size")
        axD.bar(x,           ratio_sampled_over_inf, width=bar_w, label="sampled / infections_size")
        axD.bar(x + bar_w,   ratio_sampled_over_pool,width=bar_w, label="sampled / pool_size")
        
        axD.set_title(f"Weekly Ratios (Scenario {scenario_for_sampled}, Algo: {algo_for_sampled})")
        axD.set_xlabel("Week")
        axD.set_ylabel("Ratio")
        axD.set_xlim(0.5, weeks_n + 0.5)
        axD.grid(True, linestyle="--", alpha=0.6)
        axD.legend()
        
        figD.tight_layout()
        outD = out_path("D_weekly_sampling_ratios.png")
        plt.savefig(outD, dpi=150)
        print(f"Saved: {outD}")
        plt.close(figD)


        # =================== FIGURES E: Per-Stride Variant Prevalence Error ===================
        print("Computing per-stride variant prevalence errors...")

        stride_variant_err = {algo: {} for algo in algo_list}

        all_variants_true = {
            v
            for s in weekly_variant_counts_true
            for v in (s.index.tolist() if isinstance(s, pd.Series) else [])
            if v != "background"
        }

        for scfg in SCENARIOS:
            sid = scfg["id"]
            weekly_samples_scen = all_weekly_samples.get(sid, {})
            if not weekly_samples_scen:
                continue

            stride_weeks = sampling_stride_weeks(scfg)
            eval_idx = _stride_eval_indices(scfg, len(weekly_variant_counts_true))

            for algo in algo_list:
                weeks_list = weekly_samples_scen.get(algo, [])
                if not weeks_list:
                    continue

                all_samples_df = _prepare_samples_df(weeks_list)
                xs, ys = [], []

                for end_idx in eval_idx:
                    start_idx = max(0, end_idx - stride_weeks + 1)
                    df_block = _filter_df_by_week_window(all_samples_df, args.date_field, start_idx, end_idx)
                    true_counts = _sum_hist_window(weekly_variant_counts_true, start_idx, end_idx)

                    if len(df_block) > 0 and "variant_label" in df_block.columns:
                        counts_hat = df_block["variant_label"].value_counts()
                        if "background" in counts_hat.index:
                            counts_hat = counts_hat.drop("background")
                        total_hat = float(counts_hat.sum())
                        p_hat = (counts_hat / total_hat) if total_hat > 0 else counts_hat.astype(float)
                    else:
                        p_hat = pd.Series(dtype=float)

                    if "background" in true_counts.index:
                        true_counts = true_counts.drop("background")
                    total_true = float(true_counts.sum())
                    p_true = (true_counts / total_true) if total_true > 0 else true_counts.astype(float)

                    idx = sorted(set(all_variants_true) | set(p_hat.index) | set(p_true.index))
                    p_hat_al = p_hat.reindex(idx, fill_value=0.0)
                    p_true_al = p_true.reindex(idx, fill_value=0.0)

                    xs.append(end_idx + 1)
                    ys.append((p_hat_al - p_true_al).abs().sum())

                stride_variant_err[algo][sid] = (xs, ys)

        figE, axesE = _axes_for_algos(n_algo)
        for ax, algo in zip(axesE, algo_list):
            ax.set_title(f"{algo}: Per-Stride Variant Prevalence Error")
            ax.set_xlabel("Week")
            ax.set_ylabel("Total |Δ prevalence|" if ax is axesE[0] else "")

            for scn in scenario_ids:
                x, ys = stride_variant_err.get(algo, {}).get(scn, ([], []))
                if not ys:
                    continue
                label = SCEN_LABELS.get(scn, f"Scenario {scn}")
                ax.plot(x, ys, marker=marker_map.get(scn, "o"), linestyle="-", label=label)
                _record_series("E_stride_variant_prevalence_error", algo, scn, x, ys)

            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(ncol=4, fontsize=8)
            ax.set_xlim(left=0.9)

        figE.tight_layout()
        outE = out_path("E_variant_prevalence_error_stride.png")
        plt.savefig(outE, dpi=150)
        print(f"Saved: {outE}")
        plt.close(figE)


        # =================== FIGURES F: Per-Stride Component Coverage ===================
        print("Computing per-stride component coverage...")
        COMPONENT_COL = "component_id"

        if COMPONENT_COL in line_df.columns:
            weeks_n = len(weekly_ll_hist)

            def _safe_ratio(num, den):
                return (num / den) if (den is not None and den > 0) else np.nan

            stride_comp_cov = {algo: {} for algo in algo_list}

            for scfg in SCENARIOS:
                sid = scfg["id"]
                weekly_samples_scen = all_weekly_samples.get(sid, {})
                if not weekly_samples_scen:
                    continue

                stride_weeks = sampling_stride_weeks(scfg)
                eval_idx = _stride_eval_indices(scfg, weeks_n)

                for algo in algo_list:
                    weeks_list = weekly_samples_scen.get(algo, [])
                    if not weeks_list:
                        continue

                    all_samples_df = _prepare_samples_df(weeks_list)
                    xs, ys = [], []
                    for end_idx in eval_idx:
                        start_idx = max(0, end_idx - stride_weeks + 1)
                        df_samples = _filter_df_by_week_window(all_samples_df, args.date_field, start_idx, end_idx)
                        df_truth = _filter_df_by_week_window(line_df, args.date_field, start_idx, end_idx)

                        num = df_samples[COMPONENT_COL].dropna().nunique()
                        den = df_truth[COMPONENT_COL].dropna().nunique()
                        xs.append(end_idx + 1)
                        ys.append(_safe_ratio(num, den))

                    stride_comp_cov[algo][sid] = (xs, ys)

            figG, axesG = _axes_for_algos(n_algo)
            for ax, algo in zip(axesG, algo_list):
                ax.set_title(f"{algo}: Per-Stride Component Coverage")
                ax.set_xlabel("Week")
                ax.set_ylabel("% components covered" if ax is axesG[0] else "")

                for scn in scenario_ids:
                    x, ys = stride_comp_cov.get(algo, {}).get(scn, ([], []))
                    if not ys:
                        continue
                    ax.plot(
                        x,
                        np.array(ys) * 100.0,
                        marker=marker_map.get(scn, "o"),
                        linestyle="-",
                        label=SCEN_LABELS.get(scn, f"Scenario {scn}")
                    )
                    _record_series("F_stride_component_coverage", algo, scn, x, ys)

                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(ncol=4, fontsize=8)
                ax.set_xlim(left=0.9)

            figG.tight_layout()
            outG = out_path("F_component_coverage_stride.png")
            plt.savefig(outG, dpi=150)
            print(f"Saved: {outG}")
            plt.close(figG)

        # =================== FIGURES I, J, K, L: Coverage by Tree Size ===================
        if "alias_contact" in line_df.columns:
            print("Computing Tree Coverage Scores for various sizes (Figs I-L)...")
            
            # 1. Build Graph & Precompute Sizes
            adj_graph = build_undirected_adj(line_df, pid_col="alias_pid", contact_col="alias_contact")
            all_pids_set = set(line_df["alias_pid"].astype(str))
            pid_sizes = precompute_component_sizes(adj_graph, all_pids_set)
            
            # Define thresholds and Figure labels
            # (Threshold 0 = Figure I "All", 10 = Figure J, 100 = Figure K, 1000 = Figure L)
            THRESHOLDS = [
                (0,    "I", "All Sizes"),
                (10,   "J", "> 10"),
                (100,  "K", "> 100"),
                (1000, "L", "> 1000")
            ]
            
            # Structure: results[threshold][algo][scenario] = (week_numbers, scores)
            results_by_thresh = {t: {algo: {} for algo in algo_list} for t, _, _ in THRESHOLDS}

            # --- Compute Scores ---
            for scfg in SCENARIOS:
                sid = scfg["id"]
                weekly_samples_scen = all_weekly_samples.get(sid, {})
                if not weekly_samples_scen:
                    continue

                eval_idx = _stride_eval_indices(scfg, len(weekly_ll_hist))

                for algo in algo_list:
                    weeks_list = weekly_samples_scen.get(algo, [])
                    if not weeks_list:
                        continue

                    all_samples_df = _prepare_samples_df(weeks_list)

                    series_map = {t: ([], []) for t, _, _ in THRESHOLDS}
                    
                    for end_idx in eval_idx:
                        _, week_end_date = _calendar_week_bounds(end_idx)
                        
                        mask_sample = all_samples_df[args.date_field] <= week_end_date
                        s_col = "alias_pid" if "alias_pid" in all_samples_df.columns else "pid"
                        s_ids = set(all_samples_df.loc[mask_sample, s_col].astype(str))
                        
                        mask_pop = line_df[args.date_field] <= week_end_date
                        pt_ids_all = set(line_df.loc[mask_pop, "alias_pid"].astype(str))
                        
                        for thresh, _, _ in THRESHOLDS:
                            pt_ids_filtered = {
                                u for u in pt_ids_all 
                                if pid_sizes.get(u, 1) > thresh
                            }
                            
                            score = calculate_coverage_score(pt_ids_filtered, s_ids, adj_graph)
                            xs, ys = series_map[thresh]
                            xs.append(end_idx + 1)
                            ys.append(score)

                    for thresh, _, _ in THRESHOLDS:
                        results_by_thresh[thresh][algo][sid] = series_map[thresh]

            # --- Generate Plots I, J, K, L ---
            for thresh, letter, label_suffix in THRESHOLDS:
                print(f"Generating Figure {letter} (Tree Size {label_suffix})...")
                
                fig, axes = _axes_for_algos(n_algo)
                for ax, algo in zip(axes, algo_list):
                    ax.set_title(f"{algo}: Stride-Aligned Coverage ({label_suffix})")
                    ax.set_xlabel("Week")
                    ax.set_ylabel("Score" if ax is axes[0] else "")

                    for scn in scenario_ids:
                        x, ys = results_by_thresh[thresh].get(algo, {}).get(scn, ([], []))
                        if not ys:
                            continue
                        
                        ax.plot(
                            x, ys,
                            marker=marker_map.get(scn, "o"),
                            linestyle="-",
                            label=SCEN_LABELS.get(scn, f"Scenario {scn}")
                        )

                        # Store Metrics (Naming convention: Letter_Label)
                        eval_type = f"{letter}_coverage_size_{thresh}"
                        _record_series(eval_type, algo, scn, x, ys)

                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.legend(ncol=4, fontsize=8)
                    ax.set_xlim(left=0.9)
                    ax.set_ylim(0, 1.05)

                fig.tight_layout()
                fig_out_path = out_path(f"{letter}_coverage_size_gt_{thresh}_1xN.png")
                plt.savefig(fig_out_path, dpi=150)
                print(f"Saved: {fig_out_path}")
                plt.close(fig)

        # =================== FIGURE M: 8-Week Rolling Tree Coverage ===================
        if "alias_contact" in line_df.columns:
            print("Computing 8-Week Rolling Tree Coverage Score (Figure M)...")
            
            ROLL_WIN_TREE = 8
            rolling_tree_scores = {algo: {} for algo in algo_list}

            # If adj_graph isn't already built in your scope from the previous block, build it:
            if 'adj_graph' not in locals():
                adj_graph = build_undirected_adj(line_df, pid_col="alias_pid", contact_col="alias_contact")

            for scfg in SCENARIOS:
                sid = scfg["id"]
                weekly_samples_scen = all_weekly_samples.get(sid, {})
                if not weekly_samples_scen:
                    continue

                eval_idx = _stride_eval_indices(scfg, len(weekly_ll_hist))

                for algo in algo_list:
                    weeks_list = weekly_samples_scen.get(algo, [])
                    if not weeks_list:
                        continue

                    all_samples_df = _prepare_samples_df(weeks_list)

                    xs = []
                    score_series = []
                    
                    for end_idx in eval_idx:
                        start_idx = max(0, end_idx - ROLL_WIN_TREE + 1)
                        window_start_date, window_end_date = _calendar_window_bounds(start_idx, end_idx)
                        
                        mask_pop = (
                            (line_df[args.date_field] >= window_start_date) & 
                            (line_df[args.date_field] <= window_end_date)
                        )
                        pt_ids_window = set(line_df.loc[mask_pop, "alias_pid"].astype(str))
                        
                        mask_sample = (
                            (all_samples_df[args.date_field] >= window_start_date) & 
                            (all_samples_df[args.date_field] <= window_end_date)
                        )
                        s_col = "alias_pid" if "alias_pid" in all_samples_df.columns else "pid"
                        s_ids_window = set(all_samples_df.loc[mask_sample, s_col].astype(str))
                        
                        score = calculate_coverage_score(pt_ids_window, s_ids_window, adj_graph)
                        xs.append(end_idx + 1)
                        score_series.append(score)

                    rolling_tree_scores[algo][sid] = (xs, score_series)

            # --- Plot Figure M ---
            figM, axesM = _axes_for_algos(n_algo)
            for ax, algo in zip(axesM, algo_list):
                ax.set_title(f"{algo}: 8-Week Rolling Tree Coverage (Stride-Aligned)")
                ax.set_xlabel("Week")
                ax.set_ylabel("Coverage Score (0-1)" if ax is axesM[0] else "")

                for scn in scenario_ids:
                    x, ys = rolling_tree_scores.get(algo, {}).get(scn, ([], []))
                    if not ys:
                        continue
                    
                    ax.plot(
                        x, ys,
                        marker=marker_map.get(scn, "o"),
                        linestyle="-",
                        label=SCEN_LABELS.get(scn, f"Scenario {scn}")
                    )

                    _record_series("M_8_week_rolling_tree_coverage", algo, scn, x, ys, roll_window=8)

                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(ncol=4, fontsize=8)
                ax.set_xlim(left=0.9)
                ax.set_ylim(0, 1.05)

            figM.tight_layout()
            outM = out_path("M_tree_coverage_rolling8_1xN.png")
            plt.savefig(outM, dpi=150)
            print(f"Saved: {outM}")
            plt.close(figM)

        # =================== FIGURE N: Longitudinal Equity Heatmap (By Age Group) ===================
        if "alias_contact" in line_df.columns and "age_group" in line_df.columns:
            print("Computing Longitudinal Equity Heatmaps by Age Group (Figure N)...")

            # 1. Get unique, valid age groups from the linelist
            age_groups = sorted([ag for ag in line_df["age_group"].dropna().unique() if str(ag) != "nan"])
            
            # Data structure: age_cov[age_group][algo][scenario_id] = (week_numbers, scores)
            age_cov = {ag: {algo: {} for algo in algo_list} for ag in age_groups}

            # Ensure adj_graph is available
            if 'adj_graph' not in locals():
                adj_graph = build_undirected_adj(line_df, pid_col="alias_pid", contact_col="alias_contact")

            # --- Compute Scores ---
            for scfg in SCENARIOS:
                sid = scfg["id"]
                weekly_samples_scen = all_weekly_samples.get(sid, {})
                if not weekly_samples_scen:
                    continue

                eval_idx = _stride_eval_indices(scfg, len(weekly_ll_hist))

                for algo in algo_list:
                    weeks_list = weekly_samples_scen.get(algo, [])
                    if not weeks_list:
                        continue

                    all_samples_df = _prepare_samples_df(weeks_list)

                    series_map = {ag: ([], []) for ag in age_groups}

                    for end_idx in eval_idx:
                        _, week_end_date = _calendar_week_bounds(end_idx)

                        mask_sample = all_samples_df[args.date_field] <= week_end_date
                        s_col = "alias_pid" if "alias_pid" in all_samples_df.columns else "pid"
                        s_ids = set(all_samples_df.loc[mask_sample, s_col].astype(str))

                        mask_pop = line_df[args.date_field] <= week_end_date
                        pop_this_week = line_df.loc[mask_pop]

                        for ag in age_groups:
                            pt_ag_ids = set(pop_this_week.loc[pop_this_week["age_group"] == ag, "alias_pid"].astype(str))
                            
                            score = calculate_coverage_score(pt_ag_ids, s_ids, adj_graph)
                            xs, ys = series_map[ag]
                            xs.append(end_idx + 1)
                            ys.append(score)

                    for ag in age_groups:
                        age_cov[ag][algo][sid] = series_map[ag]
                        
                        ag_clean_csv = ag.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                        eval_type = f"N_equity_{ag_clean_csv}"
                        
                        xs, ys = series_map[ag]
                        _record_series(eval_type, algo, sid, xs, ys)

            # --- Plot Figure N (5 Separate Heatmaps) ---
            for ag in age_groups:
                matrix = []
                valid_row_labels = []
                all_eval_weeks = sorted({
                    week_num
                    for algo in algo_list
                    for sid in scenario_ids
                    for week_num in age_cov[ag][algo].get(sid, ([], []))[0]
                })
                
                # Build rows: grouped by Algorithm, then Scenario
                for algo in algo_list:
                    for sid in scenario_ids:
                        xs, ys = age_cov[ag][algo].get(sid, ([], []))
                        if ys:
                            row = pd.Series(ys, index=xs, dtype=float).reindex(all_eval_weeks)
                            matrix.append(row)
                            valid_row_labels.append(f"{algo} | {SCEN_LABELS.get(sid, f'Scen {sid}')}")
                
                if not matrix or not all_eval_weeks:
                    continue
                    
                # Convert to DataFrame for Seaborn
                matrix_df = pd.DataFrame(matrix, index=valid_row_labels, columns=all_eval_weeks)
                
                figN, axN = plt.subplots(figsize=(12, max(6, len(valid_row_labels) * 0.4)))
                
                # Plot Heatmap
                # RdYlGn places 0.0 (Poor Coverage) as Red and 1.0 (Perfect Coverage) as Green
                sns.heatmap(
                    matrix_df, 
                    cmap="RdYlGn", 
                    vmin=0.0, 
                    vmax=1.0, 
                    ax=axN,
                    cbar_kws={'label': 'Cumulative Tree Coverage Score (0.0 - 1.0)'}
                )
                
                axN.set_title(f"Longitudinal Equity: {ag}\nHow close is the average {ag} to a sampled individual at stride checkpoints?", fontsize=14)
                axN.set_xlabel("Week", fontsize=12)
                axN.set_ylabel("Algorithm | Scenario", fontsize=12)
                
                # Add horizontal lines to separate the algorithms visually
                for i in range(1, len(algo_list)):
                    axN.axhline(i * len(scenario_ids), color='white', linewidth=2)
                
                plt.tight_layout()
                
                # Sanitize the age group name for saving to the filesystem
                ag_clean_file = "".join([c if c.isalnum() else "_" for c in ag]).strip("_")
                outN = out_path(f"N_equity_heatmap_{ag_clean_file}.png")
                plt.savefig(outN, dpi=150)
                print(f"Saved: {outN}")
                plt.close(figN)
        else:
            print("Missing 'contact_pid' or 'age_group' column; skipping Figure N.")

    else:
        print("\n--no-plots flag detected. Skipping plot generation.")
    
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
