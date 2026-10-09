#!/usr/bin/env python3
"""Canonical names for the sweep's evaluation metrics.

These replace the single-letter "panel" codes (``A_``, ``B_`` ... ``N_``) the
original scenario notebooks used. A letter told you nothing about what was
measured, and it leaked into output filenames, so a results directory was a
pile of ``lasso_all_scenarios_K_coverage_size_100.csv`` with no way to tell
what K was.

The new name is the old one with the letter dropped. The one exception is
``A_targets``, which became ``kl_targets`` so it says what it measures.

Two things used to be inferred from the letter prefix -- whether a higher
value is better, and which metric family a panel belongs to -- and each
notebook spelled the prefix tuples slightly differently. They are declared
here once instead.

Where each metric is computed:

* ``needs_truth = False`` -- BeyondBaseline computes it from the line list it
  was handed, so a health department can run it on real data.
* ``needs_truth = True``  -- needs the ABM's true infection counts or its
  transmission graph, so PhyloGAS computes it
  (``phylogas.benchmark.truth_metrics``).
"""

from __future__ import annotations

import re
from typing import NamedTuple


class EvalSpec(NamedTuple):
    """What one evaluation metric is and how to read it."""

    higher_is_better: bool
    family: str
    needs_truth: bool
    summary: str


# Fixed-name metrics. Families are the axis labels the notebooks group by.
EVALS: dict[str, EvalSpec] = {
    "kl_targets": EvalSpec(
        False, "KL divergence", False,
        "KL(sample || that scenario's own target distribution)",
    ),
    "cumulative_infections": EvalSpec(
        False, "KL divergence", True,
        "KL(cumulative sample || cumulative true infections)",
    ),
    "stride_window_infections": EvalSpec(
        False, "KL divergence", True,
        "KL(stride-window sample || stride-window true infections)",
    ),
    "stride_variant_prevalence_error": EvalSpec(
        False, "Prevalence error", True,
        "sum_v |p_hat(v) - p_true(v)| over the stride window",
    ),
    "stride_component_coverage": EvalSpec(
        True, "Component coverage", True,
        "distinct sampled components / distinct true components",
    ),
    "coverage_size_0": EvalSpec(
        True, "Tree coverage", True,
        "cumulative tree coverage, components of any size",
    ),
    "coverage_size_10": EvalSpec(
        True, "Tree coverage", True,
        "cumulative tree coverage, components larger than 10",
    ),
    "coverage_size_100": EvalSpec(
        True, "Tree coverage", True,
        "cumulative tree coverage, components larger than 100",
    ),
    "coverage_size_1000": EvalSpec(
        True, "Tree coverage", True,
        "cumulative tree coverage, components larger than 1000",
    ),
    "8_week_rolling_tree_coverage": EvalSpec(
        True, "Tree coverage", True,
        "(1/|Pt|) sum_u 1/(d(u,S)+1) over an 8-week rolling window",
    ),
}

# Per-stratifier metrics, named ``<prefix><stratifier>`` at run time.
EVAL_PREFIXES: dict[str, EvalSpec] = {
    # needs_truth: it is tree coverage (Mean Reciprocal Distance) on the
    # transmission graph built from alias_contact, which a real line list
    # does not carry. Recorded as False until 2026-10-09, which is why it was
    # left behind when the truth metrics moved and then stopped running.
    "equity_": EvalSpec(
        True, "Equity", True,
        "per-age-group tree coverage (1/|Pt|) sum_u 1/(d(u,S)+1), cumulative",
    ),
}

# The representative set the ranking summaries score on. Was the prefix
# tuple ("A_", "C_", "E_", "F_", "K_", "M_").
SELECTED_EVALS: tuple[str, ...] = (
    "kl_targets",
    "stride_window_infections",
    "stride_variant_prevalence_error",
    "stride_component_coverage",
    "coverage_size_100",
    "8_week_rolling_tree_coverage",
)

# Retired letter codes, kept only so old result CSVs still load. ``D_``,
# ``G_`` and ``H_`` appeared in prefix tuples but were never emitted by any
# code in this repository.
_LETTER_RE = re.compile(r"^[A-N]_")


def canonicalize(eval_type: str) -> str:
    """Return the current name for an ``eval_type``, old or new.

    Strips a retired letter prefix and maps the one renamed metric.

    >>> canonicalize("F_stride_component_coverage")
    'stride_component_coverage'
    >>> canonicalize("A_targets")
    'kl_targets'
    >>> canonicalize("coverage_size_100")
    'coverage_size_100'
    """
    name = _LETTER_RE.sub("", str(eval_type))
    return "kl_targets" if name == "targets" else name


def spec(eval_type: str) -> EvalSpec | None:
    """Look up an ``eval_type``, tolerating old letter-coded names."""
    name = canonicalize(eval_type)
    if name in EVALS:
        return EVALS[name]
    for prefix, sp in EVAL_PREFIXES.items():
        if name.startswith(prefix):
            return sp
    return None


def higher_is_better(eval_type: str) -> bool:
    """True when a larger value means better performance.

    Unknown metrics are treated as lower-is-better, which is what the
    notebooks assumed for anything outside their prefix tuples.
    """
    sp = spec(eval_type)
    return bool(sp.higher_is_better) if sp else False


def metric_family(eval_type: str) -> str:
    """Axis-label family for an ``eval_type``."""
    sp = spec(eval_type)
    return sp.family if sp else "Other"


def is_equity(eval_type: str) -> bool:
    """True for the per-stratifier equity series."""
    return canonicalize(eval_type).startswith("equity_")


def needs_truth(eval_type: str) -> bool:
    """True when this metric requires ABM ground truth (so: PhyloGAS)."""
    sp = spec(eval_type)
    return bool(sp.needs_truth) if sp else False
