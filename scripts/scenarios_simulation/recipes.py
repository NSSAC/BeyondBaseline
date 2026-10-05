#!/usr/bin/env python3
"""Canonical recipe ids: one per (scenario x algorithm).

A *recipe* is the unit a downstream consumer selects. PhyloGAS carries one (or
several) through to a Nextstrain tree, so a recipe id is a cross-repo
contract: it appears in PhyloGAS's config and in the sample filename this
package writes. Both sides therefore have to agree on the exact string, which
is why it is derived here from SCENARIOS and the algorithm registry rather
than spelled out in either repo.

    >>> recipe_id(4, "SURS")
    '4S-4_LL-P__surs'
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


def recipe_id(scenario_id: int, algorithm: str) -> str:
    return f"{scenario_slug(scenario_id)}{SEP}{algo_slug(algorithm)}"


def sample_filename(rid: str) -> str:
    """The sample file a recipe writes, relative to the run's outdir."""
    return f"{rid}_samples.csv.xz"


def all_recipes() -> dict:
    """Every valid recipe id -> what it denotes. Ordered by scenario, then
    algorithm, so the printed index is stable."""
    out = {}
    for scfg in SCENARIOS:
        sid = scfg["id"]
        for algorithm in ALGORITHM_NAMES:
            rid = recipe_id(sid, algorithm)
            out[rid] = {
                "scenario_id": sid,
                "scenario_label": SCEN_LABELS.get(sid, scfg.get("name", "")),
                "algorithm": algorithm,
                "sample_file": sample_filename(rid),
            }
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
        rows = {k: v for k, v in rows.items() if v["scenario_id"] == args.scenario}
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
    print(f"{'recipe id':{w}}  {'scenario':10}  algorithm")
    print(f"{'-' * w}  {'-' * 10}  {'-' * 18}")
    for rid, meta in rows.items():
        print(f"{rid:{w}}  {meta['scenario_label']:10}  {meta['algorithm']}")
    print(f"\n{len(rows)} recipes. Sample file: <recipe id>_samples.csv.xz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
