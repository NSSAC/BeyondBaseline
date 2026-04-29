#!/usr/bin/env python3
"""
lasso_test.py

Sweep LASSO subgroup count (max_supergroups / cluster_bins) and evaluate:
    - mean weekly KL(sample || own target)
    - panel B
    - panel C
    - panel E
    - panel F
    - panel K
    - panel M

Outputs:
    - CSV:  lasso_all_scenarios_mean_kl_vs_own_target.csv
    - Plot: lasso_all_scenarios_mean_kl_vs_own_target.png
    - CSV/Plot pairs for B, C, E, F, K, M
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_all_scenarios import (
    _normalize_stratifiers,
    build_undirected_adj,
    build_weekly_infections,
    build_weekly_variant_counts,
    calculate_coverage_score,
    load_linelist_and_population,
    precompute_component_sizes,
    run_one_scenario,
    sampling_stride_weeks,
)
from scenarios_config import (
    SCENARIOS,
    DATE_FIELD_DEFAULT,
    START_DATE_DEFAULT,
    MINIMUM_POOL_SIZE_DEFAULT,
)
from sampling_algorithms import ALGORITHMS as REGISTRY
from sampling_algorithms import kl_dist


LASSO_KEY_DEFAULT = "LASSO-Greedy"

LASSO_ALIASES = {
    "LASSO-Greedy": LASSO_KEY_DEFAULT,
}

SCENARIO_LABELS = {
    1: "1S-1(LL)",
    2: "4S-4(LL)",
    3: "1S-1(LL,P)",
    4: "4S-4(LL,P)",
    5: "1S-P",
    6: "4S-P",
}

METRIC_CONFIGS = [
    {
        "key": "own_target",
        "value_col": "mean_kl_vs_own_target",
        "csv_name": "lasso_all_scenarios_mean_kl_vs_own_target.csv",
        "png_name": "lasso_all_scenarios_mean_kl_vs_own_target.png",
        "ylabel": "Mean KL(sample || own target)\nKL(sample || target); lower is better",
        "title": "Own-target KL: mean weekly KL(sample || target)",
    },
    {
        "key": "B_cumulative_infections",
        "value_col": "mean_B_cumulative_infections",
        "csv_name": "lasso_all_scenarios_B_cumulative_infections.csv",
        "png_name": "lasso_all_scenarios_B_cumulative_infections.png",
        "ylabel": "Mean KL vs cumulative infections\nKL(cum sample || cum infections); lower is better",
        "title": "Panel B: cumulative infections, KL(cumulative sample || cumulative infections)",
    },
    {
        "key": "C_stride_window_infections",
        "value_col": "mean_C_stride_window_infections",
        "csv_name": "lasso_all_scenarios_C_stride_window_infections.csv",
        "png_name": "lasso_all_scenarios_C_stride_window_infections.png",
        "ylabel": "Mean KL vs stride-window infections\nKL(windowed sample || windowed infections); lower is better",
        "title": "Panel C: stride-window infections, KL(stride window sample || stride window infections)",
    },
    {
        "key": "E_stride_variant_prevalence_error",
        "value_col": "mean_E_stride_variant_prevalence_error",
        "csv_name": "lasso_all_scenarios_E_stride_variant_prevalence_error.csv",
        "png_name": "lasso_all_scenarios_E_stride_variant_prevalence_error.png",
        "ylabel": "Mean variant prevalence error\nsum_v |p_hat(v) - p_true(v)|; lower is better",
        "title": "Panel E: variant prevalence error, sum_v |p_hat(v) - p_true(v)|",
    },
    {
        "key": "F_stride_component_coverage",
        "value_col": "mean_F_stride_component_coverage",
        "csv_name": "lasso_all_scenarios_F_stride_component_coverage.csv",
        "png_name": "lasso_all_scenarios_F_stride_component_coverage.png",
        "ylabel": "Mean component coverage\nunique sampled components / unique true components; higher is better",
        "title": "Panel F: component coverage, sampled distinct components / true distinct components",
    },
    {
        "key": "K_coverage_size_100",
        "value_col": "mean_K_coverage_size_100",
        "csv_name": "lasso_all_scenarios_K_coverage_size_100.csv",
        "png_name": "lasso_all_scenarios_K_coverage_size_100.png",
        "ylabel": "Mean tree coverage score (>100)\n(1/|Pt|) sum_u 1/(d(u,S)+1); higher is better",
        "title": "Panel K: cumulative tree coverage for size > 100, (1/|Pt|) sum_u 1/(d(u,S)+1)",
    },
    {
        "key": "M_8_week_rolling_tree_coverage",
        "value_col": "mean_M_8_week_rolling_tree_coverage",
        "csv_name": "lasso_all_scenarios_M_8_week_rolling_tree_coverage.csv",
        "png_name": "lasso_all_scenarios_M_8_week_rolling_tree_coverage.png",
        "ylabel": "Mean 8-week rolling tree coverage\n(1/|Pt|) sum_u 1/(d(u,S)+1); higher is better",
        "title": "Panel M: 8-week rolling tree coverage, (1/|Pt|) sum_u 1/(d(u,S)+1)",
    },
]


def resolve_lasso_key(name: str | None) -> str:
    if not name:
        key = LASSO_KEY_DEFAULT
    else:
        key = LASSO_ALIASES.get(name.strip().lower(), name.strip())

    if key not in REGISTRY:
        raise ValueError(
            f"LASSO algorithm '{key}' not found in REGISTRY.\n"
            f"Available keys: {list(REGISTRY.keys())}"
        )
    return key


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Sweep LASSO subgroup count and evaluate own-target plus panels "
            "B, C, E, F, K, and M for all 6 scenarios."
        )
    )

    ap.add_argument("--linelist", required=True,
                    help="Path to linelist CSV")
    ap.add_argument("--population", required=True,
                    help="Path to population file")
    ap.add_argument("--infections", default=None,
                    help="Path to infections file. Needed for panels B, C, and E.")

    ap.add_argument("--date-field", default=DATE_FIELD_DEFAULT,
                    help=f"Linelist date column (default: {DATE_FIELD_DEFAULT})")
    ap.add_argument("--start-date",
                    default=str(START_DATE_DEFAULT.date()),
                    help=f"Week slicing anchor date (default: {START_DATE_DEFAULT.date()})")
    ap.add_argument("--min-pool", type=int, default=MINIMUM_POOL_SIZE_DEFAULT,
                    help=f"Minimum weekly pool size (default: {MINIMUM_POOL_SIZE_DEFAULT})")

    ap.add_argument("--outdir", default="lasso_tuning_all_scenarios",
                    help="Output directory")

    ap.add_argument("--seed", type=int, default=42,
                    help="Global random seed")

    ap.add_argument(
        "--stratifiers",
        nargs="+",
        default=["age", "race", "county", "sex"],
        help="Stratifier fields"
    )

    ap.add_argument(
        "--lasso-algo-name",
        default=None,
        help="Optional registry name/alias for LASSO-Stratified"
    )

    ap.add_argument("--min-group-size", type=int, default=100)
    ap.add_argument("--max-group-size", type=int, default=2000)
    ap.add_argument("--step-group-size", type=int, default=100)

    ap.add_argument("--batch-size", type=int, default=None,
                    help="Optional fixed batch size override")

    return ap.parse_args()


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype=float).fillna(0.0)
    total = float(s.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return s / total


def mean_finite(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmean(arr))

def rolling_sum_hist(hist_list: list[pd.Series], end_idx: int, win: int = 4) -> pd.Series:
    out = pd.Series(dtype=float)
    start = max(0, end_idx - win + 1)
    for j in range(start, end_idx + 1):
        out = out.add(hist_list[j], fill_value=0)
    return out


def build_target_for_scenario(
    scen_id: int,
    week_idx: int,
    weekly_ll_hist: list[pd.Series],
    pop_dist_static: pd.Series,
) -> pd.Series:
    pop = normalize_series(pop_dist_static)

    if scen_id == 1:
        return normalize_series(weekly_ll_hist[week_idx])

    if scen_id == 2:
        return normalize_series(rolling_sum_hist(weekly_ll_hist, week_idx, win=4))

    if scen_id == 3:
        ll = normalize_series(weekly_ll_hist[week_idx])
        return normalize_series(0.5 * ll + 0.5 * pop)

    if scen_id == 4:
        ll_roll = normalize_series(rolling_sum_hist(weekly_ll_hist, week_idx, win=4))
        return normalize_series(0.5 * ll_roll + 0.5 * pop)

    if scen_id == 5:
        return pop

    if scen_id == 6:
        return pop

    raise ValueError(f"Unsupported scenario id: {scen_id}")


def mean_kl_vs_own_target(
    sample_hist: list[pd.Series],
    scen_id: int,
    weekly_ll_hist: list[pd.Series],
    pop_dist_static: pd.Series,
) -> float:
    vals = []
    n = min(len(sample_hist), len(weekly_ll_hist))

    for w in range(n):
        s = normalize_series(sample_hist[w])
        t = build_target_for_scenario(
            scen_id=scen_id,
            week_idx=w,
            weekly_ll_hist=weekly_ll_hist,
            pop_dist_static=pop_dist_static,
        )

        if s.sum() <= 0 or t.sum() <= 0:
            continue

        vals.append(float(kl_dist(s, t)))

    return mean_finite(vals)


def calendar_week_bounds(start_date: pd.Timestamp, week_idx: int):
    anchor = start_date + pd.Timedelta(weeks=week_idx)
    return anchor - pd.Timedelta(days=7), anchor - pd.Timedelta(days=1)


def calendar_window_bounds(start_date: pd.Timestamp, start_idx: int, end_idx: int):
    window_start, _ = calendar_week_bounds(start_date, start_idx)
    _, window_end = calendar_week_bounds(start_date, end_idx)
    return window_start, window_end


def stride_eval_indices(scfg, n_weeks: int):
    stride = sampling_stride_weeks(scfg)
    return list(range(stride - 1, n_weeks, stride))


def sum_hist_window(hist_list: list[pd.Series], start_idx: int, end_idx: int) -> pd.Series:
    out = pd.Series(dtype=float)
    if not hist_list:
        return out
    upper = min(end_idx, len(hist_list) - 1)
    for j in range(max(0, start_idx), upper + 1):
        out = out.add(hist_list[j], fill_value=0)
    return out


def prepare_samples_df(weeks_list: list[pd.DataFrame], date_field: str) -> pd.DataFrame:
    if not weeks_list:
        return pd.DataFrame()
    all_samples_df = pd.concat(weeks_list, ignore_index=True)
    if date_field in all_samples_df.columns:
        all_samples_df[date_field] = pd.to_datetime(all_samples_df[date_field], errors="coerce")
    return all_samples_df


def filter_df_by_week_window(
    df: pd.DataFrame,
    date_col: str,
    start_idx: int,
    end_idx: int,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df.iloc[0:0].copy()
    window_start, window_end = calendar_window_bounds(start_date, start_idx, end_idx)
    mask = (df[date_col] >= window_start) & (df[date_col] <= window_end)
    return df.loc[mask]


def cum_kl_vs_stride(
    hist_list: list[pd.Series],
    ref_hist_list: list[pd.Series],
    scfg,
):
    n = min(len(hist_list), len(ref_hist_list))
    xs, ys = [], []
    for end_idx in stride_eval_indices(scfg, n):
        sample_counts = sum_hist_window(hist_list, 0, end_idx)
        ref_counts = sum_hist_window(ref_hist_list, 0, end_idx)
        xs.append(end_idx + 1)
        if sample_counts.sum() == 0 or ref_counts.sum() == 0:
            ys.append(np.nan)
            continue
        ys.append(kl_dist(sample_counts / sample_counts.sum(), ref_counts / ref_counts.sum()))
    return xs, ys


def window_kl_vs_stride(
    hist_list: list[pd.Series],
    ref_hist_list: list[pd.Series],
    scfg,
    window_weeks: int | None = None,
):
    n = min(len(hist_list), len(ref_hist_list))
    stride = sampling_stride_weeks(scfg)
    win = int(window_weeks or stride)
    xs, ys = [], []
    for end_idx in stride_eval_indices(scfg, n):
        start_idx = max(0, end_idx - win + 1)
        sample_counts = sum_hist_window(hist_list, start_idx, end_idx)
        ref_counts = sum_hist_window(ref_hist_list, start_idx, end_idx)
        xs.append(end_idx + 1)
        if sample_counts.sum() == 0 or ref_counts.sum() == 0:
            ys.append(np.nan)
            continue
        ys.append(kl_dist(sample_counts / sample_counts.sum(), ref_counts / ref_counts.sum()))
    return xs, ys


def variant_prevalence_error_series(
    all_samples_df: pd.DataFrame,
    weekly_variant_counts_true: list[pd.Series],
    scfg,
    date_field: str,
    start_date: pd.Timestamp,
):
    if not weekly_variant_counts_true:
        return [], []

    all_variants_true = {
        v
        for s in weekly_variant_counts_true
        for v in (s.index.tolist() if isinstance(s, pd.Series) else [])
        if v != "background"
    }

    stride_weeks = sampling_stride_weeks(scfg)
    eval_idx = stride_eval_indices(scfg, len(weekly_variant_counts_true))
    xs, ys = [], []

    for end_idx in eval_idx:
        start_idx = max(0, end_idx - stride_weeks + 1)
        df_block = filter_df_by_week_window(all_samples_df, date_field, start_idx, end_idx, start_date)
        true_counts = sum_hist_window(weekly_variant_counts_true, start_idx, end_idx)

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

    return xs, ys


def component_coverage_series(
    all_samples_df: pd.DataFrame,
    line_df: pd.DataFrame,
    scfg,
    date_field: str,
    start_date: pd.Timestamp,
    n_weeks: int,
):
    if "component_id" not in line_df.columns:
        return [], []

    eval_idx = stride_eval_indices(scfg, n_weeks)
    stride_weeks = sampling_stride_weeks(scfg)
    xs, ys = [], []

    for end_idx in eval_idx:
        start_idx = max(0, end_idx - stride_weeks + 1)
        df_samples = filter_df_by_week_window(all_samples_df, date_field, start_idx, end_idx, start_date)
        df_truth = filter_df_by_week_window(line_df, date_field, start_idx, end_idx, start_date)

        num = df_samples["component_id"].dropna().nunique() if "component_id" in df_samples.columns else 0
        den = df_truth["component_id"].dropna().nunique()
        xs.append(end_idx + 1)
        ys.append((num / den) if den > 0 else np.nan)

    return xs, ys


def cumulative_tree_coverage_series(
    all_samples_df: pd.DataFrame,
    line_df: pd.DataFrame,
    scfg,
    date_field: str,
    start_date: pd.Timestamp,
    n_weeks: int,
    adj_graph: dict,
    pid_sizes: dict,
    threshold: int = 100,
):
    pid_col = "sim_pid" if "sim_pid" in line_df.columns else ("pid" if "pid" in line_df.columns else None)
    if pid_col is None:
        return [], []

    eval_idx = stride_eval_indices(scfg, n_weeks)
    sample_pid_col = "sim_pid" if "sim_pid" in all_samples_df.columns else ("pid" if "pid" in all_samples_df.columns else None)
    xs, ys = [], []

    for end_idx in eval_idx:
        _, week_end_date = calendar_week_bounds(start_date, end_idx)

        if sample_pid_col is None:
            s_ids = set()
        else:
            mask_sample = all_samples_df[date_field] <= week_end_date
            s_ids = set(all_samples_df.loc[mask_sample, sample_pid_col].astype(str))

        mask_pop = line_df[date_field] <= week_end_date
        pt_ids_all = set(line_df.loc[mask_pop, pid_col].astype(str))
        pt_ids_filtered = {u for u in pt_ids_all if pid_sizes.get(u, 1) > threshold}

        xs.append(end_idx + 1)
        ys.append(calculate_coverage_score(pt_ids_filtered, s_ids, adj_graph))

    return xs, ys


def rolling_tree_coverage_series(
    all_samples_df: pd.DataFrame,
    line_df: pd.DataFrame,
    scfg,
    date_field: str,
    start_date: pd.Timestamp,
    n_weeks: int,
    adj_graph: dict,
    roll_win: int = 8,
):
    pid_col = "sim_pid" if "sim_pid" in line_df.columns else ("pid" if "pid" in line_df.columns else None)
    if pid_col is None:
        return [], []

    sample_pid_col = "sim_pid" if "sim_pid" in all_samples_df.columns else ("pid" if "pid" in all_samples_df.columns else None)
    eval_idx = stride_eval_indices(scfg, n_weeks)
    xs, ys = [], []

    for end_idx in eval_idx:
        start_idx = max(0, end_idx - roll_win + 1)
        window_start, window_end = calendar_window_bounds(start_date, start_idx, end_idx)

        mask_pop = (
            (line_df[date_field] >= window_start) &
            (line_df[date_field] <= window_end)
        )
        pt_ids_window = set(line_df.loc[mask_pop, pid_col].astype(str))

        if sample_pid_col is None:
            s_ids_window = set()
        else:
            mask_sample = (
                (all_samples_df[date_field] >= window_start) &
                (all_samples_df[date_field] <= window_end)
            )
            s_ids_window = set(all_samples_df.loc[mask_sample, sample_pid_col].astype(str))

        xs.append(end_idx + 1)
        ys.append(calculate_coverage_score(pt_ids_window, s_ids_window, adj_graph))

    return xs, ys


def save_metric_outputs(
    df: pd.DataFrame,
    value_col: str,
    out_csv: Path,
    out_png: Path,
    ylabel: str,
    title: str,
):
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV to {out_csv}")

    plt.figure(figsize=(10, 6))
    scenario_order = [1, 2, 3, 4, 5, 6]

    for sid in scenario_order:
        sub = df[df["scenario_id"] == sid].sort_values("group_size")
        if sub.empty:
            continue
        label = str(sub["scenario_name"].iloc[0])
        plt.plot(
            sub["group_size"],
            sub[value_col],
            marker="o",
            label=label,
            alpha=0.9,
        )

    mean_sub = df[df["scenario_id"] == 0].sort_values("group_size")
    plt.plot(
        mean_sub["group_size"],
        mean_sub[value_col],
        marker="o",
        linewidth=3,
        label="Mean of 6 scenarios",
    )

    plt.xlabel("LASSO subgroup count (max_supergroups / cluster_bins)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saved plot to {out_png}")
    return mean_sub


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lasso_key = resolve_lasso_key(args.lasso_algo_name)
    base_lasso_sampler = REGISTRY[lasso_key]

    start_date = pd.to_datetime(args.start_date)
    selected_features = _normalize_stratifiers(args.stratifiers)

    print(f"Using LASSO algorithm: {lasso_key}")
    print("Stratifiers:", ", ".join(selected_features))

    line_df, pop_df, pop_dist_static, weekly_ll_hist = load_linelist_and_population(
        args.linelist,
        args.population,
        args.date_field,
        start_date,
        args.min_pool,
        features=selected_features,
    )

    weekly_inf_hist = None
    weekly_variant_counts_true = None
    if args.infections:
        weekly_inf_hist = build_weekly_infections(
            args.infections,
            pop_df,
            start_date,
            num_weeks_ref=len(weekly_ll_hist),
            date_col="date",
        )
        try:
            weekly_variant_counts_true = build_weekly_variant_counts(
                args.infections,
                start_date,
                num_weeks_ref=len(weekly_ll_hist),
                date_col="date",
                variant_col="variant_label",
            )
        except Exception as exc:
            print(f"[WARN] Skipping panel E variant prevalence metric: {exc}")
    else:
        print("[WARN] --infections not provided; panels B, C, and E will be skipped.")

    pid_col = "sim_pid" if "sim_pid" in line_df.columns else ("pid" if "pid" in line_df.columns else None)
    adj_graph = None
    pid_sizes = None
    if "contact_pid" in line_df.columns and pid_col is not None:
        adj_graph = build_undirected_adj(line_df, pid_col=pid_col, contact_col="contact_pid")
        pid_sizes = precompute_component_sizes(adj_graph, set(line_df[pid_col].astype(str)))
    else:
        print("[WARN] Missing contact graph columns; panels K and M will be skipped.")

    scenarios = sorted(
        [scfg for scfg in SCENARIOS if scfg.get("id") in {1, 2, 3, 4, 5, 6}],
        key=lambda x: x["id"]
    )

    group_sizes = list(range(args.min_group_size, args.max_group_size + 1, args.step_group_size))

    metric_results = {cfg["key"]: [] for cfg in METRIC_CONFIGS}

    for gsize in group_sizes:
        print(f"\n=== subgroup count = {gsize} ===")

        def lasso_with_groups(pool_df, target_dist, batch_size, min_per_group, prior_groups, state, rng):
            st = dict(state or {})
            st["max_supergroups"] = int(gsize)
            st["cluster_bins"] = int(gsize)
            return base_lasso_sampler(
                pool_df,
                target_dist,
                batch_size,
                min_per_group,
                prior_groups,
                st,
                rng,
            )

        ALG = {lasso_key: lasso_with_groups}

        metric_vals = {cfg["key"]: [] for cfg in METRIC_CONFIGS}

        for scfg in scenarios:
            scen_id = int(scfg["id"])
            scen_name = SCENARIO_LABELS.get(scen_id, scfg.get("name", f"Scenario {scen_id}"))

            rng_master = np.random.default_rng(args.seed)

            overrides = {}
            if args.batch_size is not None:
                overrides["batch_size_fixed"] = args.batch_size

            weekly_hist, _, _, weekly_samples, _ = run_one_scenario(
                line_df=line_df,
                date_field=args.date_field,
                pop_dist_static=pop_dist_static,
                weekly_ll_hist=weekly_ll_hist,
                scfg=scfg,
                rng_master=rng_master,
                start_date=start_date,
                min_pool=args.min_pool,
                overrides=overrides,
                algorithms=ALG,
            )

            weekly_sample_hist = weekly_hist[lasso_key]
            all_samples_df = prepare_samples_df(weekly_samples[lasso_key], args.date_field)

            mean_kl = mean_kl_vs_own_target(
                sample_hist=weekly_sample_hist,
                scen_id=scen_id,
                weekly_ll_hist=weekly_ll_hist,
                pop_dist_static=pop_dist_static,
            )
            metric_values = {
                "own_target": mean_kl,
                "B_cumulative_infections": np.nan,
                "C_stride_window_infections": np.nan,
                "E_stride_variant_prevalence_error": np.nan,
                "F_stride_component_coverage": np.nan,
                "K_coverage_size_100": np.nan,
                "M_8_week_rolling_tree_coverage": np.nan,
            }

            if weekly_inf_hist is not None:
                _, ys_b = cum_kl_vs_stride(weekly_sample_hist, weekly_inf_hist, scfg)
                _, ys_c = window_kl_vs_stride(
                    weekly_sample_hist,
                    weekly_inf_hist,
                    scfg,
                    window_weeks=sampling_stride_weeks(scfg),
                )
                metric_values["B_cumulative_infections"] = mean_finite(ys_b)
                metric_values["C_stride_window_infections"] = mean_finite(ys_c)

            if weekly_variant_counts_true is not None:
                _, ys_e = variant_prevalence_error_series(
                    all_samples_df,
                    weekly_variant_counts_true,
                    scfg,
                    args.date_field,
                    start_date,
                )
                _, ys_f = component_coverage_series(
                    all_samples_df,
                    line_df,
                    scfg,
                    args.date_field,
                    start_date,
                    len(weekly_ll_hist),
                )
                metric_values["E_stride_variant_prevalence_error"] = mean_finite(ys_e)
                metric_values["F_stride_component_coverage"] = mean_finite(ys_f)

            if adj_graph is not None and pid_sizes is not None:
                _, ys_k = cumulative_tree_coverage_series(
                    all_samples_df,
                    line_df,
                    scfg,
                    args.date_field,
                    start_date,
                    len(weekly_ll_hist),
                    adj_graph,
                    pid_sizes,
                    threshold=100,
                )
                _, ys_m = rolling_tree_coverage_series(
                    all_samples_df,
                    line_df,
                    scfg,
                    args.date_field,
                    start_date,
                    len(weekly_ll_hist),
                    adj_graph,
                    roll_win=8,
                )
                metric_values["K_coverage_size_100"] = mean_finite(ys_k)
                metric_values["M_8_week_rolling_tree_coverage"] = mean_finite(ys_m)

            for cfg in METRIC_CONFIGS:
                value = metric_values[cfg["key"]]
                metric_results[cfg["key"]].append({
                    "group_size": gsize,
                    "scenario_id": scen_id,
                    "scenario_name": scen_name,
                    cfg["value_col"]: value,
                })
                metric_vals[cfg["key"]].append(value)

            print(
                f"  Scenario {scen_id}: "
                f"own_target={metric_values['own_target']:.6f}, "
                f"B={metric_values['B_cumulative_infections']:.6f}, "
                f"C={metric_values['C_stride_window_infections']:.6f}, "
                f"E={metric_values['E_stride_variant_prevalence_error']:.6f}, "
                f"F={metric_values['F_stride_component_coverage']:.6f}, "
                f"K={metric_values['K_coverage_size_100']:.6f}, "
                f"M={metric_values['M_8_week_rolling_tree_coverage']:.6f}"
            )

        for cfg in METRIC_CONFIGS:
            overall_metric = mean_finite(metric_vals[cfg["key"]])
            metric_results[cfg["key"]].append({
                "group_size": gsize,
                "scenario_id": 0,
                "scenario_name": "Mean of 6 scenarios",
                cfg["value_col"]: overall_metric,
            })
            print(f"  Overall {cfg['key']}: {overall_metric:.6f}")

    mean_sub = None
    for cfg in METRIC_CONFIGS:
        metric_df = pd.DataFrame(metric_results[cfg["key"]])
        if metric_df.empty or not np.isfinite(metric_df[cfg["value_col"]]).any():
            print(f"[WARN] No valid values for {cfg['key']}; skipping CSV/plot output.")
            continue
        metric_mean_sub = save_metric_outputs(
            metric_df,
            cfg["value_col"],
            outdir / cfg["csv_name"],
            outdir / cfg["png_name"],
            cfg["ylabel"],
            f"{lasso_key}: {cfg['title']}",
        )
        if cfg["key"] == "own_target":
            mean_sub = metric_mean_sub

    if mean_sub is None:
        raise SystemExit("Own-target metric output was not generated.")

    own_target_value_col = next(cfg["value_col"] for cfg in METRIC_CONFIGS if cfg["key"] == "own_target")
    best_row = mean_sub.loc[mean_sub[own_target_value_col].idxmin()]
    print(
        f"\nBest overall group size = {int(best_row['group_size'])}, "
        f"mean KL = {best_row[own_target_value_col]:.6f}"
    )


if __name__ == "__main__":
    main()
