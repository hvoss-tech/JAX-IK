"""Run a benchmark script in a fresh subprocess and append a row to
results.tsv.

Usage:
    python run_experiment.py --status keep --description "some change"
    python run_experiment.py --bench drag --status keep --description "..."

Runs the chosen benchmark in a subprocess (so peak-memory measurement
starts fresh and a crash in one experiment can't corrupt the next), parses
its RESULT line, and appends a tab-separated row (commit, time_in_ms,
success_rate, memory_mb, avg_spread_from_target, std_spread_from_target,
status, description) to results.tsv in the repo root. Different
benchmarks report different subsets of these fields (e.g. the drag
benchmark has no success_rate, since it doesn't gate on a threshold) --
any field a benchmark's RESULT line doesn't include is logged as "-". On
a crash (non-zero exit / unparseable output) logs every numeric field as
0 (or "-" for avg_spread_from_target / std_spread_from_target) and
status=crash regardless of --status.
"""

import argparse
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RESULTS_TSV = os.path.join(_ROOT, "results.tsv")

BENCH_SCRIPTS = {
    "arm_reach": os.path.join(_HERE, "ik_jax_lib_bench.py"),
    "drag": os.path.join(_HERE, "ik_jax_lib_bench_drag.py"),
}

# Each RESULT field is optional in the parsed line -- a benchmark just
# omits whatever it doesn't measure (see BENCH_SCRIPTS above).
FIELD_RE = {
    "time_in_ms": re.compile(r"\btime_ms=([\d.]+)"),
    "success_rate": re.compile(r"\bsuccess_rate=([\d.]+)"),
    "memory_mb": re.compile(r"\bmemory_mb=([\d.]+)"),
    # "average spread from target": the drag benchmark's dist_mean (mean
    # end-effector distance from the target across the whole dragged
    # path); other benchmarks that don't measure this leave it as "-".
    "avg_spread_from_target": re.compile(r"\bdist_mean=([\d.]+)"),
    # "standard deviation of spread from target": the drag benchmark's
    # dist_std (how much the end-effector distance from the target varies
    # across the dragged path, not just its mean) -- other benchmarks that
    # don't measure this leave it as "-".
    "std_spread_from_target": re.compile(r"\bdist_std=([\d.]+)"),
}

HEADER = [
    "commit",
    "time_in_ms",
    "success_rate",
    "memory_mb",
    "avg_spread_from_target",
    "std_spread_from_target",
    "status",
    "description",
]


def _git_commit_short():
    out = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _ensure_header():
    if os.path.exists(RESULTS_TSV) and os.path.getsize(RESULTS_TSV) > 0:
        with open(RESULTS_TSV, "r") as f:
            first_line = f.readline().rstrip("\n")
        if first_line.split("\t") == HEADER:
            return
    with open(RESULTS_TSV, "w") as f:
        f.write("\t".join(HEADER) + "\n")


def _parse_result_line(line):
    fields = {}
    for name, pattern in FIELD_RE.items():
        m = pattern.search(line)
        if m:
            fields[name] = float(m.group(1))
    return fields


def run_experiment(
    status: str, description: str, bench: str = "arm_reach", timeout: float = 1800.0
):
    _ensure_header()
    commit = _git_commit_short()
    bench_script = BENCH_SCRIPTS[bench]

    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, bench_script],
        cwd=os.path.dirname(bench_script),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    sys.stderr.write(stderr)
    sys.stdout.write(stdout)

    fields = {}
    for line in stdout.splitlines():
        if line.startswith("RESULT"):
            fields = _parse_result_line(line)

    if result.returncode != 0 or not fields:
        row_status = "crash"
        time_ms = 0.0
        success_rate = "-"
        memory_mb = 0.0
        avg_spread = "-"
        std_spread = "-"
    else:
        row_status = status
        time_ms = fields.get("time_in_ms", 0.0)
        success_rate = (
            f"{fields['success_rate']:.2f}" if "success_rate" in fields else "-"
        )
        memory_mb = fields.get("memory_mb", 0.0)
        avg_spread = (
            f"{fields['avg_spread_from_target']:.6f}"
            if "avg_spread_from_target" in fields
            else "-"
        )
        std_spread = (
            f"{fields['std_spread_from_target']:.6f}"
            if "std_spread_from_target" in fields
            else "-"
        )

    if "\t" in description:
        raise ValueError("description must not contain tabs")

    row = (
        f"{commit}\t{time_ms:.6f}\t{success_rate}\t{memory_mb:.1f}\t"
        f"{avg_spread}\t{std_spread}\t{row_status}\t{description}\n"
    )
    with open(RESULTS_TSV, "a") as f:
        f.write(row)

    print(f"Logged: {row.strip()}")
    return row_status, time_ms, success_rate, memory_mb, avg_spread, std_spread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=["keep", "discard", "crash"], required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--bench", choices=list(BENCH_SCRIPTS), default="arm_reach")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    run_experiment(args.status, args.description, bench=args.bench, timeout=args.timeout)


if __name__ == "__main__":
    main()
