#!/usr/bin/env python3
"""Quick check that scenarios CLI entry points are installed and invokable."""

from __future__ import annotations

import subprocess
import sys

COMMANDS = [
    "scenarios-runner",
    "scenarios-weekly",
    "scenarios-replicates",
    "scenarios-lasso-greedy",
    "scenarios-lasso-stratified",
    "scenarios-aggregate",
]


def check_command(cmd: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [cmd, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "not found on PATH"

    if result.returncode == 0:
        return True, "ok"

    first_line = (result.stdout or "").strip().splitlines()
    detail = first_line[0] if first_line else f"exit code {result.returncode}"
    return False, detail


def main() -> int:
    failures = 0
    print("Checking scenarios CLI entry points...\n")

    for cmd in COMMANDS:
        ok, detail = check_command(cmd)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {cmd}: {detail}")
        if not ok:
            failures += 1

    if failures:
        print(f"\n{failures} command(s) failed. Did you run 'pip install -e .' from repo root?")
        return 1

    print("\nAll scenarios CLI entry points are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
