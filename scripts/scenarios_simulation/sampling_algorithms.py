#!/usr/bin/env python3
# sampling_algorithms.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import entropy
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

# ----------------- helpers -----------------
def make_group(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        df[col] = df[col].astype(str).str.strip()
    df["group"] = df[features].agg("_".join, axis=1)
    return df

def kl_dist(p: pd.Series, q: pd.Series, eps: float = 1e-12) -> float:
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0.0).astype(float)
    q = q.reindex(idx, fill_value=0.0).astype(float)

    # clip away zeros *then* renormalize
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()

    return float(entropy(p, q))


def _rint(rng: np.random.Generator) -> int:
    """Draw a fresh 32-bit random integer for pandas' random_state."""
    # pandas accepts an int seed; use full 32-bit range for variety
    return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

# ----------------- samplers -----------------
def reward_function(group, all_groups, target_dist, eps=1e-9):
    current = pd.Series(all_groups).value_counts(normalize=True)
    curr = max(current.get(group, eps), eps)
    targ = max(target_dist.get(group, eps), eps)
    return -np.log(curr / targ)

def gittins_ucb_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,             # kept for compatibility, ignored
    pulls: dict[str, int],
    rewards: dict[str, float],
    prior_groups: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    UCB1-style bandit:
        G_g = mu_g + sqrt( 2 * log(sum_h pulls[h] + 1) / (pulls[g] + 1) )
    where mu_g = rewards[g] / max(1, pulls[g]).
    - No min-coverage bootstrap.
    - Samples without replacement.
    - Respects `batch_size`.
    """
    if batch_size <= 0 or line_df.empty:
        return line_df.iloc[0:0].copy()

    df = line_df.copy()
    group_counts = df["group"].value_counts().to_dict()
    sampled_groups: list[str] = []
    used_idx = pd.Index([])

    for _ in range(batch_size):
        # Compute UCB index per available group
        total_pulls = int(sum(pulls.values()))
        best_score = -np.inf
        best_groups: list[str] = []

        for g, c in group_counts.items():
            if c <= 0:
                continue
            p_g = int(pulls.get(g, 0))
            r_g = float(rewards.get(g, 0.0))

            mu = r_g / (p_g if p_g > 0 else 1.0)  # mean reward
            explore = np.sqrt(2.0 * np.log(total_pulls + 1.0) / (p_g + 1.0))  # your formula
            score = mu + explore

            if score > best_score + 1e-9:
                best_score = score
                best_groups = [g]
            elif abs(score - best_score) <= 1e-9:
                best_groups.append(g)

        if not best_groups:
            break

        # Tie-break uniformly at random among best groups
        best_g = rng.choice(np.array(best_groups, dtype=object))

        # Draw ONE row from the chosen group without replacement
        pool_g = df[df["group"] == best_g].drop(index=used_idx, errors="ignore")
        if pool_g.empty:
            group_counts[best_g] = 0
            continue

        take = pool_g.sample(1, replace=False, random_state=_rint(rng))
        used_idx = used_idx.union(take.index)

        # Update bandit stats with shaped reward
        sampled_groups.append(best_g)
        pulls[best_g]   = pulls.get(best_g, 0) + 1
        rewards[best_g] = rewards.get(best_g, 0.0) + reward_function(
            best_g, prior_groups + sampled_groups, target_dist
        )

        group_counts[best_g] -= 1
        if group_counts[best_g] < 0:
            group_counts[best_g] = 0

    return df.loc[used_idx].copy()


def greedy_kl_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,      # kept for compatibility, ignored
    prior_groups: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Greedy KL-divergence sampler.

    Iteratively chooses the group that minimizes the KL divergence
    between the empirical group distribution (prior + current batch)
    and the target distribution.
    """
    df = line_df.copy()
    group_counts = df["group"].value_counts().to_dict()
    picked = []

    for _ in range(batch_size):
        best_g, best_kl = None, float("inf")

        for g, c in group_counts.items():
            if c <= 0:
                continue

            trial = prior_groups + picked + [g]
            p = (
                pd.Series(trial)
                .value_counts(normalize=True)
                .reindex(target_dist.index, fill_value=0.0)
            )
            q = target_dist.reindex(p.index).fillna(0.0)  # <-- ensure no NaNs
            kl = kl_dist(p + 1e-9, q + 1e-9)


            if kl < best_kl:
                best_kl, best_g = kl, g

        if best_g is None:
            break  # no more available groups
        picked.append(best_g)
        group_counts[best_g] -= 1

    # Materialize actual rows for each chosen group
    counts = pd.Series(picked).value_counts()
    parts = []
    for g, cnt in counts.items():
        pool = df[df["group"] == g]
        parts.append(pool.sample(min(cnt, len(pool)), random_state=_rint(rng)))

    return pd.concat(parts) if parts else pd.DataFrame()


def greedy_kl_sampler_vectorized(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,      # unused (kept for compatibility)
    prior_groups: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Vectorized Greedy KL sampler.
    At each of `batch_size` steps, pick the group whose +1 update minimizes KL(p||q),
    computed in a single vectorized pass across all admissible groups.

    Notes
    -----
    - Restricts/renormalizes the target to groups *present in the pool*.
    - Honors per-group availability (no replacement within the week).
    - Uses a constant denominator trick: for each candidate, the denominator of p' is c.sum()+1,
      so KL differences collapse to replacing just the i-th term (no k×n matrices).
    """
    # quick exits
    if line_df.empty or batch_size <= 0:
        return line_df.iloc[0:0].copy()

    df = line_df

    # ---- availability & groups in this pool ----
    avail_counts = df["group"].value_counts()
    groups = avail_counts.index
    nG = len(groups)
    if nG == 0:
        return df.iloc[0:0].copy()

    # map group -> column index
    g2i = {g: i for i, g in enumerate(groups)}

    # ---- align/renormalize target to available groups ----
    q = target_dist.reindex(groups).fillna(0.0).to_numpy(dtype=float)
    qs = float(q.sum())
    if qs <= 0:
        q = np.full(nG, 1.0 / nG, dtype=float)
    else:
        q = q / qs

    # ---- initialize counts from prior history (restricted to available groups) ----
    c = np.zeros(nG, dtype=float)
    if prior_groups:
        for g, cnt in pd.Series(prior_groups).value_counts().items():
            idx = g2i.get(g)
            if idx is not None:
                c[idx] += float(cnt)

    # ---- capacities (how many we can still draw this week) ----
    cap = avail_counts.to_numpy(dtype=int)

    # ---- pre-shuffle per-group row indices for O(1) draws ----
    # build group -> shuffled index array and a position pointer
    group_indices = {}
    group_pos = np.zeros(nG, dtype=int)
    for i, g in enumerate(groups):
        idx_arr = df.index[df["group"] == g].to_numpy()
        if len(idx_arr) > 1:
            idx_arr = idx_arr[rng.permutation(len(idx_arr))]
        group_indices[i] = idx_arr

    picked_row_ids = []

    # ---- constants for KL computation ----
    eps = 1e-12
    log_q = np.log(np.clip(q, eps, None))

    # ---- main loop: pick 1 at a time, but evaluate all candidates vectorized ----
    for _ in range(batch_size):
        # candidates are groups with remaining capacity
        cand_mask = (cap > 0)
        if not np.any(cand_mask):
            break

        cand_idx = np.flatnonzero(cand_mask)

        # denominator is the same for all candidates: total_new = sum(c) + 1
        total_new = float(c.sum() + 1.0)

        # base p' for all j if we DIDN'T add a unit anywhere (used to form the shared part)
        p_base = c / total_new
        p_base = np.clip(p_base, eps, None)
        base_contrib = p_base * (np.log(p_base) - log_q)  # elementwise
        S_base = float(base_contrib.sum())

        # candidate-specific replacement of the i-th term:
        # new p_i' = (c_i + 1) / total_new
        p_new_i = (c[cand_idx] + 1.0) / total_new
        p_new_i = np.clip(p_new_i, eps, None)
        new_contrib_i = p_new_i * (np.log(p_new_i) - log_q[cand_idx])

        # KL for each candidate i: replace i-th term in the shared sum
        KLs = S_base - base_contrib[cand_idx] + new_contrib_i

        # choose argmin with RNG tie-break
        best_val = KLs.min()
        ties = cand_idx[np.flatnonzero(np.abs(KLs - best_val) <= 1e-12)]
        i_choice = rng.choice(ties) if len(ties) > 1 else ties[0]

        # commit selection
        c[i_choice] += 1.0
        cap[i_choice] -= 1

        # take one pre-shuffled row index from that group
        pos = group_pos[i_choice]
        if pos >= len(group_indices[i_choice]):
            # should not happen due to cap, but guard anyway
            continue
        picked_row_ids.append(group_indices[i_choice][pos])
        group_pos[i_choice] = pos + 1

    out = df.loc[picked_row_ids].copy()
    
    # --- NATIVE PROPORTIONAL BACKFILL ---
    shortfall = batch_size - len(out)
    if shortfall > 0:
        remaining_pool = df.drop(index=out.index, errors="ignore")
        if not remaining_pool.empty:
            # Re-normalize target weights for the remaining groups
            avail_groups = remaining_pool["group"].unique()
            rem_target = target_dist.reindex(avail_groups).fillna(0.0)
            total_weight = rem_target.sum()
            
            if total_weight > 0:
                rem_target = rem_target / total_weight
                sample_weights = remaining_pool["group"].map(rem_target)
            else:
                sample_weights = None # Fallback to uniform if targets are zero
                
            filler = remaining_pool.sample(
                n=min(shortfall, len(remaining_pool)), 
                replace=False, 
                weights=sample_weights,
                random_state=_rint(rng)
            )
            out = pd.concat([out, filler])

    return out.reset_index(drop=True)


DEFAULT_MIN_COVERAGE_FRAC = 0.05  # 5% default

def uniform_sampler_with_min_coverage(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group,                 # kept for backward compat but ignored unless used as a fraction (0<val<=1)
    prior_groups: list[str],       # unused
    state: dict,                   # read min_coverage_frac here if provided by driver
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Uniform sampling with *fractional* minimum coverage per group.

    Behavior:
      - Determine min_coverage as a FRACTION:
          1) If state["min_coverage_frac"] is set (0<frac<=1), use that.
          2) Else if min_per_group is a float in (0,1], treat as fraction (forward compat).
          3) Else fall back to DEFAULT_MIN_COVERAGE_FRAC (0.05).
      - For each target group g, sample ceil(frac * count(g)) (capped by available).
      - Fill any remaining budget with uniform random from the remaining pool.

    Notes:
      - If the required per-group minimum exceeds the batch_size in aggregate, this
        function will return more than batch_size rows (same as previous behavior).
    """
    df = line_df.copy()
    if len(df) == 0 or batch_size <= 0:
        return df.iloc[0:0].copy()

    # Resolve fraction
    frac = None
    if isinstance(state, dict):
        frac = state.get("min_coverage_frac", None)
    if frac is None and isinstance(min_per_group, float) and (0.0 < min_per_group <= 1.0):
        frac = float(min_per_group)
    if frac is None:
        frac = DEFAULT_MIN_COVERAGE_FRAC

    # Work off target groups (as before)
    group_counts = df["group"].value_counts()
    targets = target_dist.index.tolist()

    parts = []
    for g in targets:
        if g in group_counts and group_counts[g] > 0:
            need = int(np.ceil(frac * group_counts[g]))
            if need > 0:
                take = min(need, group_counts[g])
                parts.append(df[df["group"] == g].sample(take, random_state=_rint(rng)))

    initial = pd.concat(parts) if parts else pd.DataFrame()

    # Fill remainder uniformly
    remaining = batch_size - len(initial)
    if remaining <= 0:
        return initial

    rest_pool = df.drop(index=initial.index, errors="ignore")
    take = min(remaining, len(rest_pool))
    rand = rest_pool.sample(take, replace=False, random_state=_rint(rng)) if take > 0 else pd.DataFrame()
    return pd.concat([initial, rand])


def pure_uniform_sampler(
    line_df, target_dist, batch_size, min_per_group, prior_groups, state, rng
):
    if len(line_df) == 0 or batch_size <= 0:
        return line_df.iloc[0:0].copy()

    import zlib
    # Expect base_seed in state or fall back to 0
    base_seed = int((state or {}).get("base_seed", 0))
    # Optionally pass a week index in state to keep per-week variation stable
    week_idx = int((state or {}).get("week_idx", 0))

    idx = line_df.index.to_numpy()
    # 32-bit stable hash per row
    h = np.array([zlib.crc32(f"{int(i)}|{base_seed}|{week_idx}".encode()) for i in idx], dtype=np.uint32)
    take = min(batch_size, len(idx))
    picked = idx[np.argpartition(h, take-1)[:take]]
    # Return rows in deterministic order by hash (optional: keep “selection order” by sorting h[picked])
    order = np.argsort(h[np.isin(idx, picked)])
    return line_df.loc[picked[order]].copy().reset_index(drop=True)



def stratified_proportional_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,            # unused (kept for compatibility)
    prior_groups: list[str],       # unused
    state: dict,                   # optional flags (see below)
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Stratified sampling proportional to target distribution.

    Steps
    -----
    1) Keep only groups present in `line_df` (availability > 0).
    2) Normalize target probabilities across present groups.
    3) Compute ideal allocations = p * batch_size.
    4) Use largest-remainder (Hamilton) rounding to make integer allocations.
    5) Enforce availability caps; redistribute any leftover to groups that still
       have availability, favoring larger remainders.
    6) Sample without replacement per group.
    7) If still short of `batch_size`:
         - If state.get("fill_from_nontarget", True) is True, fill uniformly
           from remaining non-target groups, else return the stratified part.

    Options in `state`
    ------------------
    - fill_from_nontarget: bool (default True)
         Whether to fill any remaining budget from groups not in target_dist.
    - renormalize_to_available: bool (default True)
         If some target groups are absent, renormalize target probs over the
         present ones. If False, their mass is discarded and total may be < batch_size.
    """
    df = line_df.copy()
    if df.empty or batch_size <= 0:
        return df.iloc[0:0].copy()

    state = state or {}
    fill_from_nontarget = bool(state.get("fill_from_nontarget", True))
    renorm = bool(state.get("renormalize_to_available", True))

    # Availability by group in the pool
    avail = df["group"].value_counts().astype(int)

    # Focus on target groups; intersect with available groups
    target_dist = target_dist.copy()
    target_dist = target_dist[target_dist > 0]  # drop zero-mass groups
    target_groups_present = target_dist.index.intersection(avail.index)

    if len(target_groups_present) == 0:
        # No target groups available; fall back to uniform fill
        take = min(batch_size, len(df))
        return df.sample(take, replace=False, random_state=_rint(rng)).copy()

    td = target_dist.loc[target_groups_present].astype(float)

    # Normalize over available target groups (recommended)
    total_mass = td.sum()
    if total_mass <= 0:
        # degenerate; fallback to uniform over available target groups
        p = pd.Series(1.0, index=td.index) / len(td)
    else:
        p = td / total_mass if renorm else (td / td.sum())

    # Ideal fractional allocations
    ideal = p * batch_size

    # Integer floors and remainders
    base_alloc = np.floor(ideal).astype(int)
    remainders = (ideal - base_alloc).astype(float)

    # Enforce availability caps immediately
    caps = avail.loc[base_alloc.index]
    base_alloc = base_alloc.clip(upper=caps)

    # Total we can request overall is limited by total availability of target groups
    target_total_cap = int(caps.sum())
    desired_total = min(int(batch_size), target_total_cap)

    # How many more we need to assign among target groups
    remaining = desired_total - int(base_alloc.sum())
    if remaining > 0:
        # Largest-remainder with caps:
        # Sort groups by remainder descending; break ties randomly with rng
        # We'll cycle through until remaining is 0 or no capacity left.
        order = remainders.sort_values(ascending=False)
        if len(order) > 1:
            # stable random tie-break: shuffle indices with same remainder
            # create a tiny random jitter to remainders using rng
            jitter = pd.Series(rng.random(len(order)), index=order.index) * 1e-12
            order = (order + jitter).sort_values(ascending=False)

        alloc = base_alloc.copy()
        idx_cycle = list(order.index)

        # Assign one by one (budget sizes are typically manageable)
        i = 0
        while remaining > 0 and len(idx_cycle) > 0:
            g = idx_cycle[i % len(idx_cycle)]
            if alloc[g] < caps[g]:
                alloc[g] += 1
                remaining -= 1
            # Move to next
            i += 1
            # Stop if no groups have capacity left
            if all(alloc[h] >= caps[h] for h in idx_cycle):
                break
    else:
        alloc = base_alloc

    # Materialize per-group samples for target groups
    parts = []
    for g, k in alloc.items():
        if k <= 0:
            continue
        pool_g = df[df["group"] == g]
        if len(pool_g) == 0:
            continue
        take = min(k, len(pool_g))
        parts.append(pool_g.sample(take, replace=False, random_state=_rint(rng)))

    taken_df = pd.concat(parts) if parts else pd.DataFrame()

    # If we’re still short and allowed, fill from non-target groups uniformly
    shortfall = batch_size - len(taken_df)

    # week_id = (state or {}).get("week_id", "?")
    # print(
    #     f"[Stratified] week={week_id} leftovers_for_nontarget={max(shortfall, 0)} "
    #     f"(batch={batch_size}, stratified_target={len(taken_df)})"
    # )

    if shortfall > 0 and fill_from_nontarget:
        remaining_pool = df.drop(index=taken_df.index, errors="ignore")
        if not remaining_pool.empty:
            # Prefer groups not in target first; if not enough, take from any
            non_target_pool = remaining_pool[~remaining_pool["group"].isin(p.index)]
            if len(non_target_pool) >= shortfall:
                filler = non_target_pool.sample(shortfall, replace=False, random_state=_rint(rng))
            else:
                # take all non-target, then top-up from the remainder
                parts_fill = []
                if len(non_target_pool) > 0:
                    parts_fill.append(non_target_pool.sample(len(non_target_pool), replace=False, random_state=_rint(rng)))
                still = shortfall - sum(len(x) for x in parts_fill) if parts_fill else shortfall
                if still > 0:
                    rest = remaining_pool.drop(index=(pd.concat(parts_fill).index if parts_fill else []), errors="ignore")
                    if len(rest) > 0:
                        parts_fill.append(rest.sample(min(still, len(rest)), replace=False, random_state=_rint(rng)))
                filler = pd.concat(parts_fill) if parts_fill else pd.DataFrame()
            taken_df = pd.concat([taken_df, filler]) if not filler.empty else taken_df

    # Return whatever we could get (may be < batch_size if availability is limited and no fill)
    return taken_df.reset_index(drop=True)


def _compute_marginal_kl_vector(c: np.ndarray, q: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    For each group i, compute KL(p' || q) where p' is p after adding 1 to group i.
    Returns the absolute KL (not the delta vs current), sufficient for ranking i.
    """
    log_q = np.log(np.clip(q, eps, None))
    total_new = float(c.sum() + 1.0)
    p_base = np.clip(c / total_new, eps, None)
    base_contrib = p_base * (np.log(p_base) - log_q)
    S_base = float(base_contrib.sum())
    # candidate-specific replacement of term i
    p_new_i = np.clip((c + 1.0) / total_new, eps, None)
    new_contrib_i = p_new_i * (np.log(p_new_i) - log_q)
    KLs = S_base - base_contrib + new_contrib_i
    return KLs

def _group_features_default(groups: pd.Index,
                            q: pd.Series,
                            cap: pd.Series,
                            prior_counts: pd.Series) -> pd.DataFrame:
    """
    Lightweight numeric features per group that don't depend on raw categorical columns.
    Works everywhere.
    """
    # Normalize prior to a distribution on the same support
    prior_mass = prior_counts.reindex(groups).fillna(0.0)
    if prior_mass.sum() > 0:
        p_prior = prior_mass / prior_mass.sum()
    else:
        p_prior = pd.Series(0.0, index=groups)

    qg = q.reindex(groups).fillna(0.0)
    cg = cap.reindex(groups).fillna(0).astype(float)
    dev = p_prior - qg

    X = pd.DataFrame({
        "q": qg.values,
        "cap": cg.values,
        "log_cap1": np.log1p(cg.values),
        "prior": p_prior.values,
        "dev": dev.values,
        "abs_dev": np.abs(dev.values),
        "q_x_logcap": qg.values * np.log1p(cg.values),
    }, index=groups)
    return X

def lasso_clustered_vecgreedy_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,      # unused (compat)
    prior_groups: list[str],
    state: dict,             # see options below
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    LASSO-clustered Vectorized Greedy

    Pipeline per call:
      1) Build per-group availability (cap), target probs q, and prior counts c (restricted to available groups).
      2) Compute true marginal KL for each group (label y) from current c & q.
      3) Build numeric features X_g and fit LASSO (CV by default) to predict y.
      4) Predict y_hat and quantile-bin into <= max_supergroups clusters (supergroups).
      5) Run vectorized Greedy over supergroups (fast).
      6) For each supergroup selection, pick a member group by weights ~ cap_g * q_g (and pop one row).

    `state` options:
      - max_supergroups: int (default 50)  -> upper bound on number of clusters.
      - lasso_alpha: float or None (default None => use LassoCV)
      - lasso_cv_folds: int (default 5)    -> only used when lasso_alpha is None
      - cluster_bins: int or None (default None => same as max_supergroups)
      - member_pick: {"weighted","local_greedy"} (default "weighted")
      - debug: bool (default False)
    """
    if line_df.empty or batch_size <= 0:
        return line_df.iloc[0:0].copy()

    state = state or {}
    max_super = int(state.get("max_supergroups", 1500))
    cluster_bins = int(state.get("cluster_bins", max_super))
    lasso_alpha = state.get("lasso_alpha", None)
    lasso_cv_folds = int(state.get("lasso_cv_folds", 5))
    member_mode = (state.get("member_pick", "weighted")).lower()
    debug = bool(state.get("debug", False))

    df = line_df.copy()

    # ---- availability & groups ----
    avail_counts = df["group"].value_counts().astype(int)
    groups = avail_counts.index
    nG = len(groups)
    if nG == 0:
        return df.iloc[0:0].copy()

    # ---- target probs restricted to available groups ----
    q = target_dist.reindex(groups).fillna(0.0).astype(float)
    qs = float(q.sum())
    if qs <= 0:
        q = pd.Series(1.0 / nG, index=groups)
    else:
        q = q / qs

    # ---- prior counts on available groups ----
    prior_series = pd.Series(prior_groups).value_counts() if prior_groups else pd.Series(dtype=float)
    prior_series = prior_series.reindex(groups).fillna(0.0).astype(float)

    # ---- capacities and initial counts vectors ----
    cap = avail_counts.copy()
    c = prior_series.copy()   # we *do not* normalize; this is absolute count context for marginal calc

    # ---- labels: true marginal KL per group ----
    y_true = pd.Series(
        _compute_marginal_kl_vector(c.values.astype(float), q.values.astype(float)),
        index=groups, dtype=float
    )

    # Only train on candidates with capacity > 0
    mask_cand = cap > 0
    Gc = groups[mask_cand.values]
    if len(Gc) == 0:
        return df.iloc[0:0].copy()

    # ---- features for LASSO ----
    X_all = _group_features_default(groups, q, cap, prior_series)
    X = X_all.loc[Gc]
    y = y_true.loc[Gc]

    # scale
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.values)

    # ---- fit LASSO (sparse) ----
    if lasso_alpha is None:
        model = LassoCV(
            cv=lasso_cv_folds,
            random_state=int(rng.integers(0, 2**31 - 1)),
            max_iter=200_000,
            tol=1e-4,
            n_jobs=None
        )
    else:
        model = Lasso(alpha=float(lasso_alpha), max_iter=200_000, tol=1e-4)


    model.fit(Xs, y.values)
    y_hat = pd.Series(model.predict(Xs), index=Gc)

    # ---- cluster by predicted marginal into ≤ cluster_bins supergroups ----
    B = max(2, min(cluster_bins, len(Gc)))
    # Quantile binning; add small jitter to break ties robustly
    jitter = pd.Series(rng.normal(0, 1e-12, size=len(y_hat)), index=y_hat.index)
    ranks = (y_hat + jitter).rank(method="first", pct=True)
    bins = np.minimum((ranks * B).astype(int), B-1)  # 0..B-1
    cluster_ids = pd.Series(bins.values, index=Gc, name="cluster_id")

    # Build supergroup aggregates
    members_by_cluster = {}
    for cid, g in cluster_ids.items():
        pass  # placeholder; we build below using groupby

    cluster_df = pd.DataFrame({
        "group": Gc,
        "cluster": cluster_ids.values,
        "q": q.loc[Gc].values,
        "cap": cap.loc[Gc].values,
        "c": c.loc[Gc].values
    })
    agg = cluster_df.groupby("cluster", sort=True).agg({
        "q": "sum",
        "cap": "sum",
        "c": "sum"
    }).rename_axis("cluster").reset_index()

    # Filter out empty-cap clusters
    agg = agg[agg["cap"] > 0].reset_index(drop=True)
    if agg.empty:
        return df.iloc[0:0].copy()

    # supergroup arrays
    qS = agg["q"].to_numpy(dtype=float)
    capS = agg["cap"].to_numpy(dtype=int)
    cS = agg["c"].to_numpy(dtype=float)
    nS = len(agg)

    # Map cluster -> member groups frame (for fan-out later)
    members = (cluster_df[["cluster", "group", "q", "cap", "c"]]
               .sort_values(["cluster", "group"])
               .reset_index(drop=True))

    # Pre-shuffle row indices per atomic group for O(1) draws
    grp_col = df["group"]
    group_indices = {}
    group_pos = {}
    for g in groups:
        idx_arr = df.index[grp_col == g].to_numpy()
        if len(idx_arr) > 1:
            idx_arr = idx_arr[rng.permutation(len(idx_arr))]
        group_indices[g] = idx_arr
        group_pos[g] = 0

    # ---- run vectorized Greedy at supergroup level ----
    remaining = min(batch_size, int(cap.sum()))
    picked_row_ids = []

    eps = 1e-12
    log_qS = np.log(np.clip(qS, eps, None))

    # helper: one supergroup draw, then fan to member
    def pick_from_super(cid: int):
        # choose member according to weights ~ cap_g * q_g (or local greedy if asked)
        sub = members[members["cluster"] == cid]
        if sub.empty:
            return None
        if member_mode == "local_greedy":
            # compute local marginal KLs for members using current c & q (1D compute)
            # Note: use proportional version inside cluster
            q_sub = sub["q"].to_numpy(dtype=float)
            qs = float(q_sub.sum())
            q_sub = (q_sub / qs) if qs > 0 else np.full(len(sub), 1.0 / len(sub))
            c_sub = sub["c"].to_numpy(dtype=float)
            cap_sub = sub["cap"].to_numpy(dtype=int)
            if cap_sub.sum() <= 0:
                return None
            # Only consider those with cap>0
            mask = cap_sub > 0
            if not np.any(mask):
                return None
            cands = sub.index[mask]
            cvec = c_sub.copy()
            qvec = q_sub.copy()
            # compute true marginal KLs for these members
            KLm = _compute_marginal_kl_vector(cvec, qvec)[mask]
            j_rel = int(np.argmin(KLm))
            sel_idx = cands[j_rel]
            g = sub.loc[sel_idx, "group"]
        else:
            # weighted by cap * q
            w = (sub["cap"].clip(lower=0).to_numpy(dtype=float) *
                 np.maximum(sub["q"].to_numpy(dtype=float), eps))
            if w.sum() <= 0:
                return None
            probs = w / w.sum()
            sel_idx = rng.choice(sub.index.to_numpy(), p=probs)
            g = sub.loc[sel_idx, "group"]

        # pop one row from that member group
        pos = group_pos[g]
        arr = group_indices[g]
        if pos >= len(arr):
            # no more real rows (shouldn’t happen due to cap), skip
            return None
        row_id = arr[pos]
        group_pos[g] = pos + 1

        # update member bookkeeping
        members.loc[members["group"] == g, "cap"] -= 1
        members.loc[members["group"] == g, "c"] += 1.0
        return int(row_id)

    while remaining > 0 and np.any(capS > 0):
        # candidates with capacity
        cand = np.flatnonzero(capS > 0)
        if len(cand) == 0:
            break

        total_new = float(cS.sum() + 1.0)
        p_base = np.clip(cS / total_new, eps, None)
        base_contrib = p_base * (np.log(p_base) - log_qS)
        S_base = float(base_contrib.sum())

        p_new_i = np.clip((cS[cand] + 1.0) / total_new, eps, None)
        new_contrib_i = p_new_i * (np.log(p_new_i) - log_qS[cand])
        KLs = S_base - base_contrib[cand] + new_contrib_i

        j = int(np.argmin(KLs))
        cid = int(cand[j])

        # commit at supergroup level
        cS[cid] += 1.0
        capS[cid] -= 1

        # fan to a member group and take a row
        rid = pick_from_super(cid)
        if rid is not None:
            picked_row_ids.append(rid)
            remaining -= 1
        else:
            # if member selection failed (no capacity), keep looping; the capS will hit zero
            continue

    out = df.loc[picked_row_ids].copy() if picked_row_ids else df.iloc[0:0].copy()

    # --- NATIVE PROPORTIONAL BACKFILL ---
    shortfall = batch_size - len(out)
    if shortfall > 0:
        remaining_pool = df.drop(index=out.index, errors="ignore")
        if not remaining_pool.empty:
            avail_groups = remaining_pool["group"].unique()
            rem_target = target_dist.reindex(avail_groups).fillna(0.0)
            total_weight = rem_target.sum()
            
            if total_weight > 0:
                rem_target = rem_target / total_weight
                sample_weights = remaining_pool["group"].map(rem_target)
            else:
                sample_weights = None
                
            filler = remaining_pool.sample(
                n=min(shortfall, len(remaining_pool)), 
                replace=False, 
                weights=sample_weights,
                random_state=_rint(rng)
            )
            out = pd.concat([out, filler])

    # --- compute KL at group level (prior + this batch) ---
    try:
        if prior_groups or not out.empty:
            combined = (prior_groups or []) + out["group"].tolist()
            p_groups = pd.Series(combined).value_counts(normalize=True)
            q_groups = target_dist.reindex(p_groups.index).fillna(0.0).astype(float)
            kl_groups = kl_dist(p_groups, q_groups)
        else:
            kl_groups = float("nan")
    except Exception:
        kl_groups = float("nan")

    # --- compute KL at supergroup (cluster) level ---
    try:
        total_c = float(cS.sum())
        sum_qS = float(qS.sum())
        if total_c > 0 and sum_qS > 0:
            p_clusters = pd.Series(cS / total_c, index=agg["cluster"])
            q_clusters = pd.Series(qS / sum_qS, index=agg["cluster"])
            kl_clusters = kl_dist(p_clusters, q_clusters)
        else:
            kl_clusters = float("nan")
    except Exception:
        kl_clusters = float("nan")

    # --- log alpha and KLs ---
    try:
        alpha_used = float(getattr(model, "alpha_", getattr(model, "alpha", float("nan"))))
    except Exception:
        alpha_used = float("nan")

    # week_id = (state or {}).get("week_id", "?")
    # print(
    #     f"[LASSO-VecGreedy] week={week_id} done | "
    #     f"groups={nG} trainable={len(Gc)} clusters={B} "
    #     f"picked={len(out)} alpha={alpha_used:.6g} "
    #     f"KL_groups={kl_groups:.6g} KL_clusters={kl_clusters:.6g}"
    # )

    return out.reset_index(drop=True)


def _rint(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))

def _largest_remainder_alloc(p: pd.Series, total: int, caps: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Hamilton rounding with capacity caps."""
    if total <= 0 or p.empty:
        return pd.Series(0, index=p.index, dtype=int)
    p = p.clip(lower=0)
    s = float(p.sum())
    if s <= 0:
        p = pd.Series(1.0, index=p.index) / len(p)
    else:
        p = p / s
    ideal = p * total
    base = np.floor(ideal).astype(int)
    base = base.clip(upper=caps.reindex(base.index).fillna(0).astype(int))
    remaining = int(total - base.sum())
    if remaining <= 0:
        return base

    rem = (ideal - base).astype(float)
    # jitter to break ties reproducibly
    if len(rem) > 1:
        jitter = pd.Series(rng.random(len(rem)), index=rem.index) * 1e-12
        rem = rem + jitter
    order = rem.sort_values(ascending=False).index.tolist()

    alloc = base.copy()
    i = 0
    while remaining > 0 and len(order) > 0:
        g = order[i % len(order)]
        cap_g = int(caps.get(g, 0))
        if alloc[g] < cap_g:
            alloc[g] += 1
            remaining -= 1
        i += 1
        if all(alloc[h] >= int(caps.get(h, 0)) for h in order):
            break
    return alloc

def lasso_clustered_stratified_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,      # unused (compat)
    prior_groups: list[str],
    state: dict,             # options below
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    LASSO-clustered Stratified Sampler

    Same LASSO fit + quantile clustering as the greedy variant, but allocation is:
      1) Across supergroups: proportional to aggregated target mass q_S (Hamilton rounding, with caps).
      2) Within each supergroup: proportional to member q_g (again Hamilton, with caps).
      3) Sample without replacement per group.

    State options:
      - max_supergroups: int (default 50)
      - cluster_bins: int or None (default None => max_supergroups)
      - lasso_alpha: float or None (default None => LassoCV)
      - lasso_cv_folds: int (default 5)  # when lasso_alpha is None
      - intra_weights: {"q", "cap_q"} (default "cap_q")
            "q": allocate within supergroup ∝ q_g
            "cap_q": allocate within supergroup ∝ cap_g * q_g
      - renormalize_to_available: bool (default True)
      - fill_shortfall_uniform: bool (default True)  # if stratified cannot fill due to caps
      - debug: bool (default False)
    """
    df = line_df.copy()
    if df.empty or batch_size <= 0:
        return df.iloc[0:0].copy()

    state = state or {}
    max_super = int(state.get("max_supergroups", 1500))
    cluster_bins = int(state.get("cluster_bins", max_super))
    lasso_alpha = state.get("lasso_alpha", None)
    lasso_cv_folds = int(state.get("lasso_cv_folds", 5))
    intra_mode = str(state.get("intra_weights", "cap_q")).lower()
    renorm = bool(state.get("renormalize_to_available", True))
    fill_shortfall_uniform = bool(state.get("fill_shortfall_uniform", True))
    debug = bool(state.get("debug", False))

    # ---- availability & groups (pool-level) ----
    avail_counts = df["group"].value_counts().astype(int)
    groups = avail_counts.index
    if len(groups) == 0:
        return df.iloc[0:0].copy()

    cap = avail_counts.copy()

    # ---- target probs restricted to available groups ----
    q = target_dist.reindex(groups).fillna(0.0).astype(float)
    qs = float(q.sum())
    q = (q / qs) if (qs > 0) else pd.Series(1.0 / len(groups), index=groups)

    # ---- prior counts on available groups ----
    prior_series = pd.Series(prior_groups).value_counts() if prior_groups else pd.Series(dtype=float)
    prior_series = prior_series.reindex(groups).fillna(0.0).astype(float)

    # ---- labels: true marginal KL per group ----
    y_true = pd.Series(
        _compute_marginal_kl_vector(prior_series.values.astype(float), q.values.astype(float)),
        index=groups, dtype=float
    )

    # trainable set (cap > 0)
    mask_cand = cap > 0
    Gc = groups[mask_cand.values]
    if len(Gc) == 0:
        return df.iloc[0:0].copy()

    # ---- features + scaling ----
    X_all = _group_features_default(groups, q, cap, prior_series)
    X = X_all.loc[Gc]
    y = y_true.loc[Gc]

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.values)

    # ---- fit LASSO ----
    if lasso_alpha is None:
        model = LassoCV(
            cv=lasso_cv_folds,
            random_state=_rint(rng),
            max_iter=200_000,
            tol=1e-4,
            n_jobs=None
        )
    else:
        model = Lasso(alpha=float(lasso_alpha), max_iter=200_000, tol=1e-4)

    model.fit(Xs, y.values)
    y_hat = pd.Series(model.predict(Xs), index=Gc)

    # ---- quantile-bin predictions => supergroups ----
    B = max(2, min(cluster_bins, len(Gc)))
    jitter = pd.Series(rng.normal(0, 1e-12, size=len(y_hat)), index=y_hat.index)
    ranks = (y_hat + jitter).rank(method="first", pct=True)
    bins = np.minimum((ranks * B).astype(int), B - 1)  # 0..B-1
    cluster_ids = pd.Series(bins.values, index=Gc, name="cluster")

    # member rows (for later fan-out and sampling)
    members = (pd.DataFrame({
        "group": Gc,
        "cluster": cluster_ids.values,
        "q": q.loc[Gc].values,
        "cap": cap.loc[Gc].values
    }).sort_values(["cluster", "group"]).reset_index(drop=True))

    if isinstance(state, dict):
        week = state.get("week_id")
        scen = state.get("scenario_id")
        algo = state.get("algo_name")

        log_df = members.copy()
        log_df.insert(0, "week", week)
        log_df.insert(1, "scenario", scen)
        log_df.insert(2, "algorithm", algo)

        history = state.setdefault("cluster_history", [])
        history.append(log_df)

    # aggregated supergroup stats
    agg = members.groupby("cluster", sort=True).agg({
        "q": "sum",
        "cap": "sum",
    }).rename_axis("cluster").reset_index()

    # filter clusters with zero cap
    agg = agg[agg["cap"] > 0].reset_index(drop=True)
    if agg.empty:
        return df.iloc[0:0].copy()

    # ---- 1) allocate budget across supergroups (Hamilton) ----
    qS = pd.Series(agg["q"].to_numpy(float), index=agg["cluster"])
    capS = pd.Series(agg["cap"].to_numpy(int), index=agg["cluster"])

    allocS = _largest_remainder_alloc(qS, int(min(batch_size, int(capS.sum()))), capS, rng)

    # ---- 2) within each supergroup, allocate to member groups ----
    parts = []
    for cid, kS in allocS.items():
        if kS <= 0:
            continue
        sub = members[members["cluster"] == cid].copy()
        if sub.empty:
            continue

        if intra_mode == "q":
            weights = pd.Series(sub["q"].to_numpy(float), index=sub["group"])
        else:  # "cap_q" (default)
            w = sub["cap"].clip(lower=0).to_numpy(float) * np.maximum(sub["q"].to_numpy(float), 0.0)
            weights = pd.Series(w, index=sub["group"])

        caps_g = pd.Series(sub["cap"].to_numpy(int), index=sub["group"])
        allocG = _largest_remainder_alloc(weights, int(kS), caps_g, rng)

        # materialize samples per member group
        for g, k in allocG.items():
            if k <= 0:
                continue
            pool_g = df[df["group"] == g]
            if len(pool_g) == 0:
                continue
            take = min(int(k), len(pool_g))
            parts.append(pool_g.sample(take, replace=False, random_state=_rint(rng)))

    out = pd.concat(parts) if parts else df.iloc[0:0].copy()

    # ---- 3) if shortfall, optionally fill uniformly from remaining pool ----
    shortfall = int(batch_size - len(out))
    # week_id = (state or {}).get("week_id", "?")
    # print(
    #     f"[LASSO-Stratified] week={week_id} leftovers_for_uniform={max(shortfall, 0)} "
    #     f"(batch={batch_size}, stratified={len(out)})"
    # )
    if shortfall > 0 and fill_shortfall_uniform:
        remaining_pool = df.drop(index=out.index, errors="ignore")
        if not remaining_pool.empty:
            more = remaining_pool.sample(min(shortfall, len(remaining_pool)),
                                         replace=False, random_state=_rint(rng))
            out = pd.concat([out, more], ignore_index=False)

    # log alpha used (optional parity with greedy)
    try:
        alpha_used = float(getattr(model, "alpha_", getattr(model, "alpha", float("nan"))))
    except Exception:
        alpha_used = float("nan")


    return out.reset_index(drop=True)

# mapping used by the driver
ALGORITHMS = {
    "SURS": pure_uniform_sampler,
    "Uniform Random": uniform_sampler_with_min_coverage,
    "Greedy": lambda df, td, bs, mpg, prior, state, rng: greedy_kl_sampler_vectorized(
        df, td, bs, mpg, prior, rng
    ),
    "Stratified": stratified_proportional_sampler,
    "RL":   lambda df, td, bs, mpg, prior, state, rng: gittins_ucb_sampler(
        df, td, bs, mpg, state.setdefault("pulls", {}), state.setdefault("rewards", {}), prior, rng
    ),
    "LASSO-Greedy" : lasso_clustered_vecgreedy_sampler,
    "LASSO-Stratified": lasso_clustered_stratified_sampler,
}
