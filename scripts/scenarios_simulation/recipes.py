#!/usr/bin/env python3
"""Canonical recipe ids: one per (scenario x algorithm).

A *recipe* is the unit a downstream consumer selects. PhyloGAS carries one (or
several) through to a Nextstrain tree, so a recipe id is a cross-repo
contract: it appears in PhyloGAS's config and in the sample filename this
package writes. Both sides therefore have to agree on the exact string, which
is why it is derived here from SCENARIOS and the algorithm registry rather
than spelled out in either repo.

Not every algorithm responds to every scenario axis. An algorithm that
ignores the target distribution is named by the axes it actually reads, so
its id is shorter and does not imply a dependence it does not have --
see TARGET_BLIND_ALGORITHMS.

    >>> recipe_id(4, "SURS")
    '4S__surs'
    >>> sample_filename(recipe_id(4, "LASSO-Greedy"))
    '4S-4_LL-P__lasso_greedy_samples.csv.xz'

`scenarios-recipes` prints the index.
"""
from __future__ import annotations

import sys

try:
    from .scenarios_config import SCENARIOS, SCEN_LABELS, ALGORITHM_NAMES
except ImportError as exc:                      # running the file directly
    if exc.name is not None:
        raise
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scenarios_config import SCENARIOS, SCEN_LABELS, ALGORITHM_NAMES

SEP = "__"


def scenario_slug(scenario_id: int) -> str:
    """Filesystem-safe slug for a scenario, from its label.

    "4S-4(LL,P)" -> "4S-4_LL-P". The parenthesised part names the target
    distribution, so it is kept rather than reduced to the bare id: a reader
    of a filename or a config entry can tell 4S-4_LL-P from 4S-P without
    consulting the index.
    """
    label = SCEN_LABELS.get(scenario_id)
    if label is None:
        raise KeyError(f"no label for scenario {scenario_id}; "
                       f"known: {sorted(SCEN_LABELS)}")
    out = (label.replace("–", "-").replace("—", "-")
                .replace("(", "_").replace(")", "").replace(",", "-")
                .replace(" ", ""))
    return out


def algo_slug(algorithm: str) -> str:
    """"LASSO-Greedy" -> "lasso_greedy", matching how algorithms are named in
    PhyloGAS's `sampling.algorithms`."""
    return algorithm.strip().lower().replace("-", "_").replace(" ", "_")


# Algorithms that never read the target distribution. `pure_uniform_sampler`
# takes target_dist, min_per_group and prior_groups and reads none of them: it
# selects on a hash of the row index against the pool. So every scenario that
# agrees on the stride and pool produces the *identical* SURS sample, and
# naming one of them "4S-4_LL-P__surs" implies a dependence on the (LL,P)
# target that never touched the file. Collapsed to the stride instead.
#
# The duplicate grid cells still exist for scoring -- a target-blind sample is
# scored against each scenario's own target, so its KL series legitimately
# differs per scenario. Only the sample identity collapses.
TARGET_BLIND_ALGORITHMS = frozenset({"SURS"})

# The scenario fields a target-blind algorithm can actually observe, used to
# check that collapsing is safe rather than assuming it.
_SAMPLE_AXES = ("decision_window_weeks", "pool_mode", "pool_window_weeks",
                "batch_frac", "batch_cap", "no_replacement")


def stride_slug(scenario_id: int) -> str:
    """"4S-4(LL,P)" -> "4S". The stride token, all a target-blind algorithm
    responds to."""
    return scenario_slug(scenario_id).split("-")[0]


def _sample_axes(scfg: dict) -> tuple:
    return tuple(scfg.get(k) for k in _SAMPLE_AXES)


def canonical_scenario_slug(scenario_id: int, algorithm: str) -> str:
    """The scenario half of a recipe id, for this algorithm.

    Full slug for a target-aware algorithm; the stride alone for a
    target-blind one, but only after checking that every scenario sharing
    that stride agrees on the axes the algorithm can see. If they ever
    diverge, collapsing would hide a real difference, so this raises rather
    than silently merging two distinct samples onto one filename.
    """
    if algorithm not in TARGET_BLIND_ALGORITHMS:
        return scenario_slug(scenario_id)
    stride = stride_slug(scenario_id)
    peers = [sc for sc in SCENARIOS if stride_slug(sc["id"]) == stride]
    distinct = {_sample_axes(sc) for sc in peers}
    if len(distinct) > 1:
        raise ValueError(
            f"{algorithm} is declared target-blind, but scenarios "
            f"{[sc['id'] for sc in peers]} share stride {stride!r} while "
            f"differing on {_SAMPLE_AXES}. Collapsing them onto one id would "
            f"merge two different samples. Either drop {algorithm} from "
            f"TARGET_BLIND_ALGORITHMS or split the stride token.")
    return stride


def recipe_id(scenario_id: int, algorithm: str) -> str:
    return (f"{canonical_scenario_slug(scenario_id, algorithm)}"
            f"{SEP}{algo_slug(algorithm)}")


def grid_recipe_id(scenario_id: int, algorithm: str) -> str:
    """The uncollapsed scenario x algorithm id, for every grid cell.

    Equal to recipe_id() except for a target-blind algorithm, where several
    grid ids map onto one canonical recipe. Kept so an older config entry
    still resolves, and so the scoring grid can be enumerated.
    """
    return f"{scenario_slug(scenario_id)}{SEP}{algo_slug(algorithm)}"


def sample_filename(rid: str) -> str:
    """The sample file a recipe writes, relative to the run's outdir."""
    return f"{rid}_samples.csv.xz"


def all_recipes() -> dict:
    """Every canonical recipe id -> what it denotes.

    One entry per distinct *sample*, not per grid cell: a target-blind
    algorithm contributes one recipe per stride, and `scenario_ids` lists
    every scenario that recipe's sample serves. Ordered by first appearance
    so the printed index is stable.
    """
    out = {}
    for scfg in SCENARIOS:
        sid = scfg["id"]
        for algorithm in ALGORITHM_NAMES:
            rid = recipe_id(sid, algorithm)
            if rid in out:
                out[rid]["scenario_ids"].append(sid)
                out[rid]["scenario_labels"].append(
                    SCEN_LABELS.get(sid, scfg.get("name", "")))
                continue
            out[rid] = {
                "scenario_id": sid,
                "scenario_ids": [sid],
                "scenario_label": SCEN_LABELS.get(sid, scfg.get("name", "")),
                "scenario_labels": [SCEN_LABELS.get(sid, scfg.get("name", ""))],
                "algorithm": algorithm,
                "collapsed": algorithm in TARGET_BLIND_ALGORITHMS,
                "sample_file": sample_filename(rid),
            }
    return out


def aliases() -> dict:
    """Uncollapsed grid id -> canonical recipe id, for the ids that moved.

    Only non-identity entries, so an empty dict means nothing collapsed.
    """
    out = {}
    for scfg in SCENARIOS:
        for algorithm in ALGORITHM_NAMES:
            grid = grid_recipe_id(scfg["id"], algorithm)
            canon = recipe_id(scfg["id"], algorithm)
            if grid != canon:
                out[grid] = canon
    return out


def split_recipe(rid: str) -> tuple[str, str]:
    """A recipe id back into (scenario_slug, algo_slug)."""
    if SEP not in rid:
        raise ValueError(f"{rid!r} is not a recipe id (expected "
                         f"'<scenario>{SEP}<algorithm>')")
    scen, _, algo = rid.partition(SEP)
    return scen, algo


def resolve(rid: str) -> dict:
    """Look up a recipe id, failing with the valid set rather than a KeyError."""
    recipes = all_recipes()
    if rid not in recipes:
        alias = aliases().get(rid)
        if alias is not None:
            # An older config naming a grid cell still works, but say so: the
            # sample it wanted is shared with the other scenarios of that
            # stride, and the id it should use is the canonical one.
            print(f"note: {rid!r} is an alias for {alias!r} "
                  f"(that algorithm ignores the target distribution)",
                  file=sys.stderr)
            return recipes[alias]
        scen, algo = (split_recipe(rid) if SEP in rid else (rid, ""))
        near = [r for r in recipes if r.startswith(scen + SEP)] or \
               [r for r in recipes if r.endswith(SEP + algo)]
        raise KeyError(
            f"unknown recipe {rid!r}. "
            + (f"Did you mean one of {near[:4]}? " if near else "")
            + f"`scenarios-recipes` lists all {len(recipes)}.")
    return recipes[rid]


def main() -> int:
    """Print the recipe index."""
    import argparse

    ap = argparse.ArgumentParser(
        description="List canonical recipe ids (scenario x algorithm). These "
                    "are the strings a downstream config refers to.")
    ap.add_argument("--scenario", type=int, default=None,
                    help="only this scenario id")
    ap.add_argument("--algorithm", default=None,
                    help="only this algorithm (name or slug)")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="ids only, one per line")
    args = ap.parse_args()

    rows = all_recipes()
    if args.scenario is not None:
        rows = {k: v for k, v in rows.items()
                if args.scenario in v["scenario_ids"]}
    if args.algorithm:
        want = algo_slug(args.algorithm)
        rows = {k: v for k, v in rows.items() if algo_slug(v["algorithm"]) == want}
    if not rows:
        print("no recipes match that filter", file=sys.stderr)
        return 1

    if args.quiet:
        print("\n".join(rows))
        return 0

    w = max(len(k) for k in rows)
    sc = max(len(", ".join(v["scenario_labels"])) for v in rows.values())
    print(f"{'recipe id':{w}}  {'scenario(s)':{sc}}  algorithm")
    print(f"{'-' * w}  {'-' * sc}  {'-' * 18}")
    for rid, meta in rows.items():
        labels = ", ".join(meta["scenario_labels"])
        print(f"{rid:{w}}  {labels:{sc}}  {meta['algorithm']}")
    collapsed = [r for r, m in rows.items() if m.get("collapsed")]
    print(f"\n{len(rows)} recipes. Sample file: <recipe id>_samples.csv.xz")
    if collapsed:
        print(f"{len(collapsed)} of them serve several scenarios with one "
              f"sample, because the algorithm ignores the target "
              f"distribution: {', '.join(collapsed)}.")
        print("Those scenarios are still scored separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
