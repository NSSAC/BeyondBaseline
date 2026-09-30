#!/usr/bin/env python3
"""
run_weekly_sampling.py — Adaptive operational weekly sampling script.

New in this version:
  --target "LL" | "LL,P" | "P"
      Specifies the target distribution type directly instead of scenario IDs.
        LL   = linelist-only (rolling window of recent linelist data)
        LL,P = blended (alpha * linelist + (1-alpha) * population)
        P    = population-only (static census distribution)

  --current-date YYYY-MM-DD
      The Monday-aligned anchor for "this week".  The pool covers
      [current_date - 7d, current_date - 1d].  Replaces --week-number
      (which is still accepted for backward compatibility).

  --pool-window W      Rolling pool window in weeks (default: 4).
  --decision-window W  Prior-groups decision window in weeks (default: same
                       as --pool-window).
  --blend-alpha A      Blend ratio for LL,P target (default: 0.5).

Usage:
    python run_weekly_sampling.py \\
        --linelist linelist.csv \\
        --population pop.txt \\
        --target "LL,P" \\
        --current-date 2021-07-12 \\
        --pool-window 4 \\
        --blend-alpha 0.5 \\
        --batch-size 200 \\
        --no-replacement \\
        --algorithms surs greedy stratified \\
        --already-sequenced weekly_results/history_combined_all.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .scenarios_config import (
        SCENARIOS,
        DATE_FIELD_DEFAULT,
        START_DATE_DEFAULT,
        MINIMUM_POOL_SIZE_DEFAULT,
        BATCH_FRAC_DEFAULT,
        BATCH_CAP_DEFAULT,
        MIN_PER_GROUP_DEFAULT,
        BLEND_ALPHA,
    )
    from .sampling_algorithms import ALGORITHMS as REGISTRY, make_group, kl_dist
    from .run_all_scenarios import (
        normalize_age_group_col,
        linelist_dist_at_week,
        blended_target,
        select_algorithms,
        _normalize_stratifiers,
        SCEN_LABELS,
    )
except ImportError:
    from scenarios_config import (
        SCENARIOS,
        DATE_FIELD_DEFAULT,
        START_DATE_DEFAULT,
        MINIMUM_POOL_SIZE_DEFAULT,
        BATCH_FRAC_DEFAULT,
        BATCH_CAP_DEFAULT,
        MIN_PER_GROUP_DEFAULT,
        BLEND_ALPHA,
    )
    from sampling_algorithms import ALGORITHMS as REGISTRY, make_group, kl_dist
    from run_all_scenarios import (
        normalize_age_group_col,
        linelist_dist_at_week,
        blended_target,
        select_algorithms,
        _normalize_stratifiers,
        SCEN_LABELS,
    )


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Adaptive operational weekly sampler with direct target and date control."
        ),
    )

    # ---- required inputs ----
    ap.add_argument("--linelist", required=True,
                    help="Path to the linelist CSV.")
    ap.add_argument("--population", required=True,
                    help="Path to the population traits file (skiprows=1).")

    # ---- target specification (new) ----
    ap.add_argument(
        "--target",
        type=str, default=None,
        help=(
            "Target distribution type: 'LL' (linelist only), 'LL,P' (blended), "
            "or 'P' (population only).  If set, --scenarios is ignored and a "
            "single scenario config is built from --target, --pool-window, "
            "--decision-window, and --blend-alpha."
        ),
    )
    ap.add_argument("--blend-alpha", type=float, default=BLEND_ALPHA,
                    help=f"Blend ratio for LL,P target (default: {BLEND_ALPHA}).")
    ap.add_argument("--pool-window", type=int, default=4,
                    help="Rolling pool window in weeks (default: 4).")
    ap.add_argument(
        "--decision-window", type=int, default=None,
        help="Prior-groups decision window in weeks (default: same as --pool-window).",
    )

    # ---- date specification (new) ----
    ap.add_argument(
        "--current-date", type=str, default=None,
        help=(
            "The Monday-aligned date for 'this week' (YYYY-MM-DD).  "
            "Pool covers [current_date - 7d, current_date - 1d].  "
            "Mutually exclusive with --week-number."
        ),
    )

    # ---- backward-compatible args ----
    ap.add_argument("--week-number", type=int, default=None,
                    help="1-based global week index (backward compat; prefer --current-date).")
    ap.add_argument("--scenarios", nargs="+", type=int, default=None,
                    help="Run only these scenario IDs (ignored if --target is set).")

    # ---- history ----
    ap.add_argument(
        "--already-sequenced", dest="already_sequenced", default=None,
        help="Path to history CSV or directory of history files.",
    )

    # ---- anchor ----
    ap.add_argument("--start-date", default=str(START_DATE_DEFAULT.date()),
                    help=f"Anchor date for weekly slicing. Default: {START_DATE_DEFAULT.date()}")

    # ---- sampling budget ----
    ap.add_argument("--batch-size", type=int, help="Fixed weekly budget N.")
    ap.add_argument("--batch-frac", type=float, help="Override batch_frac (0-1).")
    ap.add_argument("--batch-cap", type=int, help="Override batch_cap.")
    ap.add_argument("--min-per-group", type=int, help="Override min_per_group.")

    # ---- behaviour ----
    ap.add_argument("--no-replacement", action="store_true",
                    help="Do not re-sample previously selected rows.")
    ap.add_argument("--min-pool", type=int, default=MINIMUM_POOL_SIZE_DEFAULT,
                    help=f"Minimum weekly pool size (default: {MINIMUM_POOL_SIZE_DEFAULT}).")
    ap.add_argument("--date-field", default=DATE_FIELD_DEFAULT,
                    help=f"Date column name (default: {DATE_FIELD_DEFAULT}).")

    # ---- algorithms / stratifiers ----
    ap.add_argument("--algorithms", nargs="+", default=["surs", "stratified", "LASSO-Stratified"],
                    help="Algorithms to run.")
    ap.add_argument("--stratifiers", nargs="+", default=["age", "race", "county", "sex"],
                    help="Stratifier fields.")

    # ---- output ----
    ap.add_argument("--outdir", default="weekly_results",
                    help="Output directory (default: weekly_results).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")

    return ap.parse_args()


# =====================================================================
# Build scenario config from --target
# =====================================================================
TARGET_ALIASES = {
    "ll":   "LL",
    "linelist": "LL",
    "ll,p": "LL,P",
    "ll+p": "LL,P",
    "blend": "LL,P",
    "p":    "P",
    "pop":  "P",
    "population": "P",
}

def build_scenario_from_target(target_str, pool_window, decision_window, blend_alpha):
    """
    Build a single scenario config dict from the --target string.

    Returns
    -------
    scfg : dict   (same schema as entries in SCENARIOS)
    label : str   (human-readable label)
    """
    key = target_str.strip().lower().replace(" ", "")
    target_type = TARGET_ALIASES.get(key)
    if target_type is None:
        raise ValueError(
            f"Unknown --target '{target_str}'.  "
            f"Valid: LL, LL,P, P (or aliases: {list(TARGET_ALIASES.keys())})"
        )

    dec_win = decision_window if decision_window is not None else pool_window
    label = f"{pool_window}S-{dec_win}({target_type})"

    scfg = {
        "id": 99,  # synthetic ID, won't collide with scenarios_config
        "name": f"Custom ({label})",
        "eval_metric": "per_stride_kl",
        "eval_window_weeks": None,
        "batch_frac": BATCH_FRAC_DEFAULT,
        "batch_cap": BATCH_CAP_DEFAULT,
        "min_per_group": MIN_PER_GROUP_DEFAULT,
        "pool_mode": "rolling",
        "pool_window_weeks": pool_window,
        "decision_window_weeks": dec_win,
        "no_replacement": True,
    }

    if target_type == "LL":
        scfg["target_mode"] = "linelist_dynamic"
        scfg["target_linelist_mode"] = "rolling"
        scfg["target_linelist_window"] = pool_window
    elif target_type == "LL,P":
        scfg["target_type"] = "blend"
        scfg["blend_alpha"] = blend_alpha
        scfg["target_linelist_mode"] = "rolling"
        scfg["target_linelist_window"] = pool_window
    elif target_type == "P":
        scfg["target_mode"] = "population_static"

    return scfg, label


# =====================================================================
# Date → week-number conversion
# =====================================================================
def date_to_week_number(current_date, start_date):
    """
    Convert a date to a 1-based week number relative to start_date.

    Week 1: current_week = start_date → pool [start_date-7d, start_date-1d]
    Week W: current_week = start_date + (W-1)*7d
    So W = floor((current_date - start_date).days / 7) + 1
    """
    delta_days = (current_date - start_date).days
    if delta_days < 0:
        raise ValueError(
            f"--current-date {current_date.date()} is before "
            f"--start-date {start_date.date()}"
        )
    week_number = (delta_days // 7) + 1
    return week_number


# =====================================================================
# Data loading
# =====================================================================
def load_data(linelist_path, population_path, date_field, features):
    line_df = pd.read_csv(linelist_path, parse_dates=[date_field])
    pop_df  = pd.read_csv(population_path, skiprows=1)

    pop_df  = normalize_age_group_col(pop_df,  "age_group")
    line_df = normalize_age_group_col(line_df, "age_group")

    pop_df = pop_df.rename(columns={"gender": "sex"})
    pop_df["sex"]      = pop_df["sex"].astype(str).map({"1": "male", "2": "female"})
    pop_df["smh_race"] = pop_df["smh_race"].astype(str).map({
        "W": "White", "B": "Black", "L": "Latino", "A": "Asian", "O": "Other",
    })

    line_df = make_group(line_df, features)
    pop_df  = make_group(pop_df,  features)
    pop_dist = pop_df["group"].value_counts(normalize=True).sort_index()

    return line_df, pop_df, pop_dist


def build_weekly_ll_hist(line_df, date_field, start_date, min_pool):
    weekly_ll_hist = []
    cur = start_date
    while True:
        prev_mon = cur - timedelta(days=7)
        prev_sun = cur - timedelta(days=1)
        wk = line_df[
            (line_df[date_field] >= prev_mon) & (line_df[date_field] <= prev_sun)
        ]
        if len(wk) < min_pool:
            break
        weekly_ll_hist.append(wk["group"].value_counts())
        cur += timedelta(weeks=1)
    return weekly_ll_hist


def infer_week_number(line_df, date_field, start_date, min_pool):
    weekly = build_weekly_ll_hist(line_df, date_field, start_date, min_pool)
    if not weekly:
        raise ValueError(
            f"No viable weeks in linelist (min_pool={min_pool}, "
            f"start_date={start_date.date()})."
        )
    return len(weekly)


# =====================================================================
# History loading
# =====================================================================
def load_history(path, scenario_id, algo_name, features, date_field):
    if path is None:
        return pd.DataFrame()

    p = Path(path)

    # Directory mode: scan for matching file
    if p.is_dir():
        target_file = None
        for pat in [
            f"history_updated_scen{scenario_id}_{algo_name}.csv.xz",
            f"history_updated_scen{scenario_id}_{algo_name}.csv",
        ]:
            candidate = p / pat
            if candidate.is_file():
                target_file = candidate
                break
        if target_file is None:
            matches = sorted(p.glob(f"history_updated_scen{scenario_id}_{algo_name}*"))
            if matches:
                target_file = matches[0]
        if target_file is None:
            return pd.DataFrame()
        past_df = pd.read_csv(target_file)

    elif p.is_file():
        past_df = pd.read_csv(p)
        if "scenario_id" in past_df.columns and "algorithm" in past_df.columns:
            mask = (
                (past_df["scenario_id"] == scenario_id)
                & (past_df["algorithm"] == algo_name)
            )
            past_df = past_df[mask].copy()
    else:
        return pd.DataFrame()

    past_df = normalize_age_group_col(past_df, "age_group")
    if "group" not in past_df.columns:
        past_df = make_group(past_df, features)
    if date_field in past_df.columns:
        past_df[date_field] = pd.to_datetime(past_df[date_field], errors="coerce")
    return past_df


def build_prior_groups_from_history(past_df, date_field, start_date,
                                    current_week_number, decision_window_weeks):
    if past_df.empty or "group" not in past_df.columns:
        return []
    if decision_window_weeks is None:
        return past_df["group"].dropna().tolist()
    if date_field not in past_df.columns:
        return past_df["group"].dropna().tolist()

    oldest_week_to_include = max(1, current_week_number - (decision_window_weeks - 1))
    groups = []
    for _, row in past_df.iterrows():
        row_date = row.get(date_field)
        if pd.isna(row_date):
            continue
        days_offset = (row_date - start_date).days
        row_week = (days_offset + 7) // 7
        if oldest_week_to_include <= row_week < current_week_number:
            g = row.get("group")
            if pd.notna(g):
                groups.append(g)
    return groups


def rebuild_used_indices(past_df, line_df, date_field):
    if past_df.empty:
        return set()
    KEY_COLS = ["sim_pid", "sim_tick"]
    usable_keys = [k for k in KEY_COLS if k in past_df.columns and k in line_df.columns]
    if not usable_keys:
        if "sim_pid" in past_df.columns and "sim_pid" in line_df.columns:
            past_pids = set(past_df["sim_pid"].astype(str))
            return set(line_df.index[line_df["sim_pid"].astype(str).isin(past_pids)].tolist())
        return set()
    keys_df = past_df[usable_keys].dropna().drop_duplicates()
    matched = line_df.merge(keys_df, on=usable_keys, how="inner")
    return set(matched.index.tolist())


# =====================================================================
# Adaptive single-week sampling core
# =====================================================================
def sample_one_week(
    line_df, date_field, pop_dist_static, weekly_ll_hist,
    scfg, rng, start_date, current_week_number,
    prior_groups, used_indices, overrides, algo_name,
):
    ov_fixed = overrides.get("batch_size_fixed", None)
    ov_frac  = overrides.get("batch_frac", None)
    ov_cap   = overrides.get("batch_cap", None)
    ov_mpg   = overrides.get("min_per_group", None)
    ov_norep = bool(overrides.get("no_replacement", False))

    current_week = start_date + timedelta(weeks=(current_week_number - 1))
    week_idx_for_target = current_week_number - 1

    prev_mon = current_week - timedelta(days=7)
    prev_sun = current_week - timedelta(days=1)

    week_df = line_df[
        (line_df[date_field] >= prev_mon) & (line_df[date_field] <= prev_sun)
    ]
    if len(week_df) < overrides.get("min_pool", MINIMUM_POOL_SIZE_DEFAULT):
        return pd.DataFrame(), {"error": "below min_pool", "pool_size": len(week_df)}

    # Pool
    pool_mode = scfg.get("pool_mode")
    if pool_mode == "history":
        first_window_start = start_date - pd.Timedelta(days=7)
        pool_df = line_df[
            (line_df[date_field] >= first_window_start)
            & (line_df[date_field] <= prev_sun)
        ]
    elif pool_mode == "rolling":
        w = int(scfg.get("pool_window_weeks", 4))
        pool_start = current_week - pd.Timedelta(weeks=w)
        pool_df = line_df[
            (line_df[date_field] >= pool_start) & (line_df[date_field] <= prev_sun)
        ]
    else:
        pool_df = week_df

    if ov_norep or scfg.get("no_replacement", False):
        if used_indices:
            pool_df = pool_df.drop(index=list(used_indices), errors="ignore")

    # Batch size
    eff_frac = ov_frac if ov_frac is not None else scfg["batch_frac"]
    eff_cap  = ov_cap  if ov_cap  is not None else scfg["batch_cap"]
    eff_mpg  = ov_mpg  if ov_mpg  is not None else scfg["min_per_group"]

    if ov_fixed is not None:
        batch_size = int(min(max(0, ov_fixed), len(pool_df)))
    else:
        batch_size = int(min(eff_frac * len(pool_df), eff_cap))

    min_per_group = int(max(0, eff_mpg))

    if batch_size <= 0 or len(pool_df) == 0:
        return pd.DataFrame(), {"error": "empty pool or zero batch", "pool_size": len(pool_df)}

    # Target distribution
    if scfg.get("target_type") == "blend":
        ll_mode   = scfg.get("target_linelist_mode", "cumulative")
        ll_window = scfg.get("target_linelist_window", 4)
        alpha     = scfg.get("blend_alpha", 0.5)
        ll_dist   = linelist_dist_at_week(weekly_ll_hist, week_idx_for_target, ll_mode, ll_window)
        target_dist = blended_target(ll_dist, pop_dist_static, alpha)
    elif scfg.get("target_mode") == "linelist_dynamic":
        ll_mode   = scfg.get("target_linelist_mode", "cumulative")
        ll_window = scfg.get("target_linelist_window", 4)
        target_dist = linelist_dist_at_week(weekly_ll_hist, week_idx_for_target, ll_mode, ll_window)
    else:
        target_dist = pop_dist_static

    avail_groups = pool_df["group"].value_counts().index
    if target_dist is None or target_dist.empty:
        target_dist = pop_dist_static
    target_dist = target_dist.reindex(avail_groups).dropna()
    s = float(target_dist.sum() or 0.0)
    if s > 0:
        target_dist = target_dist / s
    else:
        target_dist = pd.Series(1.0, index=avail_groups) / len(avail_groups)

    # State
    state = {}
    if algo_name == "SURS":
        state["base_seed"] = overrides.get("base_seed", 0)
    state["week_id"] = week_idx_for_target
    state["scenario_id"] = scfg.get("id")
    state["algo_name"] = algo_name

    # Sample
    sampler = REGISTRY[algo_name]
    sample_df = sampler(
        pool_df, target_dist, batch_size, min_per_group,
        prior_groups, state, rng,
    )

    # Recover full rows
    KEY_COLS = ["sim_pid", "sim_tick"]
    usable_keys = [k for k in KEY_COLS if k in sample_df.columns and k in pool_df.columns]
    if usable_keys and len(sample_df) > 0:
        keys_df = sample_df[usable_keys].dropna().drop_duplicates()
        sample_df = pool_df.merge(keys_df, on=usable_keys, how="inner").copy()

    # Diagnostics
    if scfg.get("target_mode") == "population_static" and scfg.get("target_type") != "blend":
        ll_weeks_used = 0
    else:
        ll_window = scfg.get("target_linelist_window", 1)
        ll_mode = scfg.get("target_linelist_mode", "cumulative")
        first = 0 if ll_mode == "cumulative" else max(0, week_idx_for_target - ll_window + 1)
        ll_weeks_used = sum(1 for j in range(first, week_idx_for_target + 1) if 0 <= j < len(weekly_ll_hist))

    diag = {
        "pool_size": len(pool_df),
        "batch_size": batch_size,
        "sampled": len(sample_df),
        "week_pool_start": str(prev_mon.date()),
        "week_pool_end": str(prev_sun.date()),
        "ll_hist_weeks_available": len(weekly_ll_hist),
        "ll_hist_weeks_used_for_target": ll_weeks_used,
    }
    if len(sample_df) > 0:
        p_week = sample_df["group"].value_counts(normalize=True)
        diag["kl_week_vs_target"] = round(float(kl_dist(
            p_week.reindex(target_dist.index, fill_value=0.0),
            target_dist.reindex(p_week.index, fill_value=0.0),
        )), 6)
    else:
        diag["kl_week_vs_target"] = float("nan")

    return sample_df, diag


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_date = pd.to_datetime(args.start_date)
    features   = _normalize_stratifiers(args.stratifiers)
    ALG        = select_algorithms(REGISTRY, args.algorithms)
    algo_list  = list(ALG.keys())

    print(f"Algorithms : {', '.join(algo_list)}")
    print(f"Stratifiers: {', '.join(features)}")

    # ---- load data ----
    line_df, pop_df, pop_dist = load_data(
        args.linelist, args.population, args.date_field, features,
    )
    print(f"Linelist: {len(line_df)} rows, "
          f"{line_df[args.date_field].min().date()} – "
          f"{line_df[args.date_field].max().date()}")

    # ---- build weekly_ll_hist ----
    weekly_ll_hist = build_weekly_ll_hist(
        line_df, args.date_field, start_date, args.min_pool,
    )
    print(f"LL history: {len(weekly_ll_hist)} viable weeks")

    # ---- determine week number ----
    if args.current_date is not None and args.week_number is not None:
        raise ValueError("Cannot specify both --current-date and --week-number.")

    if args.current_date is not None:
        current_date = pd.to_datetime(args.current_date)
        week_number = date_to_week_number(current_date, start_date)
        print(f"--current-date {current_date.date()} → week number {week_number}")
    elif args.week_number is not None:
        week_number = args.week_number
        if week_number < 1:
            raise ValueError("--week-number must be >= 1")
    else:
        week_number = infer_week_number(
            line_df, args.date_field, start_date, args.min_pool,
        )
        print(f"Auto-inferred week number: {week_number}")

    if week_number > len(weekly_ll_hist):
        print(f"  [WARN] week {week_number} > ll_hist ({len(weekly_ll_hist)}w). "
              f"Target uses available data only.")

    # Calendar dates
    current_week = start_date + timedelta(weeks=(week_number - 1))
    prev_mon = current_week - timedelta(days=7)
    prev_sun = current_week - timedelta(days=1)
    print(f"Week {week_number}: pool {prev_mon.date()} – {prev_sun.date()}")

    # ---- build scenario list ----
    if args.target is not None:
        # Build a single custom scenario from --target
        dec_win = args.decision_window if args.decision_window is not None else args.pool_window
        scfg, label = build_scenario_from_target(
            args.target, args.pool_window, dec_win, args.blend_alpha,
        )
        scenarios_to_run = [scfg]
        scen_labels = {scfg["id"]: label}
        print(f"Target: {args.target} → {label}")
        print(f"  pool_window={args.pool_window}, decision_window={dec_win}, "
              f"blend_alpha={args.blend_alpha}")
    elif args.scenarios:
        scenarios_to_run = [s for s in SCENARIOS if s["id"] in args.scenarios]
        scen_labels = SCEN_LABELS
        if not scenarios_to_run:
            raise ValueError(f"No matching scenarios for IDs {args.scenarios}")
    else:
        scenarios_to_run = SCENARIOS
        scen_labels = SCEN_LABELS

    # ---- overrides ----
    overrides = {
        "batch_size_fixed": args.batch_size,
        "batch_frac": args.batch_frac,
        "batch_cap": args.batch_cap,
        "min_per_group": args.min_per_group,
        "no_replacement": args.no_replacement,
        "min_pool": args.min_pool,
    }

    # ---- run ----
    all_history_frames = []
    summary_lines = [
        f"Weekly Sampling Report",
        f"Week number : {week_number}",
        f"Pool dates  : {prev_mon.date()} – {prev_sun.date()}",
        f"Target      : {args.target or 'scenarios'}",
        f"Seed        : {args.seed}",
        f"Algorithms  : {', '.join(algo_list)}",
        f"Stratifiers : {', '.join(features)}",
        f"LL hist     : {len(weekly_ll_hist)} weeks",
        "=" * 70,
    ]

    for scfg in scenarios_to_run:
        sid = scfg["id"]
        sname = scfg["name"]
        slabel = scen_labels.get(sid, sname)

        print(f"\n{'='*60}")
        print(f"=== {sname} (ID {sid}: {slabel}) ===")

        # Adaptive info
        pool_w = scfg.get("pool_window_weeks", 1)
        target_w = scfg.get("target_linelist_window", 1)
        dec_w = scfg.get("decision_window_weeks", None)
        print(f"  Pool: {pool_w}w | Target: {target_w}w | Decision: {dec_w}w")
        print(f"{'='*60}")

        # RNG: deterministic per (seed, scenario_id)
        rng_scn = np.random.default_rng(args.seed + sid * 1000003)
        algo_rngs = {
            name: np.random.default_rng(rng_scn.integers(0, 2**63 - 1))
            for name in algo_list
        }

        summary_lines.append(f"\n--- {sname} ({slabel}) ---")

        for algo_name in algo_list:
            t0 = time.perf_counter()

            past_df = load_history(
                args.already_sequenced, sid, algo_name, features, args.date_field,
            )

            dec_win = scfg.get("decision_window_weeks", None)
            prior_groups = build_prior_groups_from_history(
                past_df, args.date_field, start_date, week_number, dec_win,
            )

            if args.no_replacement or scfg.get("no_replacement", False):
                used_indices = rebuild_used_indices(past_df, line_df, args.date_field)
            else:
                used_indices = set()

            sample_df, diag = sample_one_week(
                line_df=line_df,
                date_field=args.date_field,
                pop_dist_static=pop_dist,
                weekly_ll_hist=weekly_ll_hist,
                scfg=scfg,
                rng=algo_rngs[algo_name],
                start_date=start_date,
                current_week_number=week_number,
                prior_groups=prior_groups,
                used_indices=used_indices,
                overrides=overrides,
                algo_name=algo_name,
            )

            elapsed = time.perf_counter() - t0

            if len(sample_df) > 0:
                sample_df = sample_df.assign(
                    scenario_id=sid,
                    scenario_name=sname,
                    algorithm=algo_name,
                    week_number=week_number,
                    sampling_week_start=prev_mon.date(),
                    sampling_week_end=prev_sun.date(),
                )

            # Save per-(scenario, algo) files
            new_path = outdir / f"samples_new_scen{sid}_{algo_name}.csv"
            sample_df.to_csv(new_path, index=False)

            combined_df = pd.concat([past_df, sample_df], ignore_index=True)
            hist_path = outdir / f"history_updated_scen{sid}_{algo_name}.csv"
            combined_df.to_csv(hist_path, index=False)

            all_history_frames.append(combined_df)

            # Console
            ll_used = diag.get("ll_hist_weeks_used_for_target", "?")
            line = (
                f"  {algo_name:<20s} | pool={diag.get('pool_size',0):>6d} "
                f"| batch={diag.get('batch_size',0):>5d} "
                f"| sampled={diag.get('sampled',0):>5d} "
                f"| KL={diag.get('kl_week_vs_target', float('nan')):>.6f} "
                f"| ll_used={ll_used} "
                f"| {elapsed:.2f}s"
            )
            print(line)
            summary_lines.append(line)
            summary_lines.append(
                f"    -> {new_path.name} | {hist_path.name} ({len(combined_df)} rows)"
            )

    # Combined history
    if all_history_frames:
        combined_all = pd.concat(all_history_frames, ignore_index=True)
        combined_path = outdir / "history_combined_all.csv"
        combined_all.to_csv(combined_path, index=False)
        print(f"\nCombined history: {combined_path} ({len(combined_all)} rows)")
        print(f"  Next week: --already-sequenced {combined_path}")
        print(f"  Or:        --already-sequenced {outdir}/")

    # Summary
    summary_path = outdir / "weekly_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()