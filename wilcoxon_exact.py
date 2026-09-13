"""Regenerate the exact paired Wilcoxon signed-rank tables for the circuit-level cases.

WHY THIS SCRIPT IS IN THE ARCHIVE. The Data availability statement promises "a signed-rank table
computed with the exact test alongside the normal-approximation table that the plotting pipeline
emits by default". Shipping the two CSVs alone makes them readable but not checkable: a table with no
generator can only be trusted, and trust is the thing the statement is trying to avoid asking for.
This script rebuilds every exact table from the per-run records that ship beside it, so any reader can
re-derive the numbers quoted in Section 5 without the cluster, the simulator or the PDK.

    python wilcoxon_exact.py            # rebuild the tables
    python wilcoxon_exact.py --check    # rebuild in memory and diff against the shipped tables

WHAT THE APPROXIMATION FLOOR IS, AND WHY IT IS DETECTED BY W RATHER THAN BY VALUE. The signed-rank
statistic W is bounded below by zero, reached when all thirty paired differences fall on the same
side. The normal approximation maps that single extremal statistic to one p-value regardless of how
decisive the win is -- about 1.7344e-6 at n=30 -- so every all-thirty win reports the same number and
the approximation has stopped measuring anything.

An earlier version of this table detected that condition by comparing the approximate p-value against
the literal constant 1.7343976282946372e-06 with an absolute tolerance of 1e-18. That constant came
from the cluster's hand-rolled implementation; SciPy's normal CDF returns 1.7343976283205784e-06 for
the same data, which differs by 2.6e-17 -- above the tolerance. The consequence was that the
`approx_at_floor` column read 0 for every comparison in every file, including the five where W is
exactly 0. The test here is `W == 0`, which is the condition itself rather than a fingerprint of one
implementation of it, and is therefore exact and portable.

THE SCOPE COLUMN carries the manuscript's own case name -- SKY130-CM-Pub, SKY130-CM-Cal,
SKY130-Base, SKY130-Hard, SKY130-Highdim -- so that a row in these tables can be matched to the row
of Table 8 it supports without a translation step. Earlier tables used internal directory names
(`SKY130-base`, `SKY130-BGR-banba`); those are relabelled here and nothing else about them changes.

The exact test has no such floor. At n=30 an all-thirty win gives two-sided p = 2/2**30 =
1.8626451492309570e-09, three orders of magnitude below the approximation's floor, and that value is
a measurement rather than a saturation point.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ALGOS = ["ABC", "GWO", "FA", "PSO", "GA", "ACO"]
N_PAIRS = 30
# two-sided exact p for W=0 at n=30; the floor of what the exact test can report here
EXACT_FLOOR = 2.0 / 2 ** 30

# case directory -> (analysis subdirectory, scope label written into the table)
CASES = {
    "banba_150k":     ("banba",     "SKY130-CM-Pub"),
    "banba_cal_150k": ("banba_cal", "SKY130-CM-Cal"),
    "base_150k":      ("base",      "SKY130-Base"),
    "hard_150k":      ("hard",      "SKY130-Hard"),
    "highdim_220k":   ("highdim",   "SKY130-Highdim"),
}

# The Holm-adjusted exact p-values quoted in Section 5 for the two current-mode cases, as printed in
# the manuscript. Asserted here so that a reader running this script is told whether the paper's
# numbers still follow from the shipped data, instead of having to compare by eye.
PAPER = {
    ("banba_150k", "ABC vs GWO"): 7.6e-03,
    ("banba_150k", "ABC vs GA"): 1.6e-04,
    ("banba_150k", "ABC vs PSO"): 4.2e-05,
    ("banba_150k", "ABC vs ACO"): 1.9e-08,
    ("banba_150k", "ABC vs FA"): 3.7e-08,
    ("banba_cal_150k", "ABC vs GWO"): 0.700,
}
COLUMNS = ["scope", "comparison", "n_pairs", "wilcoxon_statistic", "p_value_exact",
           "holm_adjusted_p_exact", "holm_reject_0_05", "favours", "p_value_normal_approx",
           "holm_adjusted_p_normal_approx", "approx_at_floor"]


def holm(ps):
    """Holm step-down adjustment. Ordered ascending, each raw p scaled by the number of hypotheses
    not yet rejected, then made monotone by a running maximum and clipped at 1."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, adj, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def build(case):
    """Compute the table for one case from its shipped per_run_records.csv."""
    src = RESULTS / case / "per_run_records.csv"
    if not src.exists():
        return None, f"missing {src.relative_to(HERE).as_posix()}"
    d = pd.read_csv(src)
    piv = d.pivot(index="run", columns="algo", values="best_fitness")[ALGOS]
    if piv.shape != (N_PAIRS, len(ALGOS)):
        return None, f"{case}: expected {N_PAIRS}x{len(ALGOS)} block, got {piv.shape}"
    ranks = piv.rank(axis=1, method="average").mean()

    others = [a for a in ALGOS if a != "ABC"]
    ex = [wilcoxon(piv["ABC"].values, piv[a].values, method="exact") for a in others]
    ap = [wilcoxon(piv["ABC"].values, piv[a].values, method="approx").pvalue for a in others]
    h_ex, h_ap = holm([r.pvalue for r in ex]), holm(ap)

    scope = CASES[case][1]
    rows = []
    for a, r, phe, pa, pha in zip(others, ex, h_ex, ap, h_ap):
        rows.append({
            "scope": scope,
            "comparison": f"ABC vs {a}",
            "n_pairs": N_PAIRS,
            "wilcoxon_statistic": float(r.statistic),
            "p_value_exact": r.pvalue,
            "holm_adjusted_p_exact": phe,
            "holm_reject_0_05": int(phe < 0.05),
            "favours": "ABC" if ranks["ABC"] < ranks[a] else a,
            "p_value_normal_approx": pa,
            "holm_adjusted_p_normal_approx": pha,
            # the statistic at its lower bound: all thirty pairs on one side. This is the condition
            # the normal approximation saturates on, tested directly.
            "approx_at_floor": int(float(r.statistic) == 0.0),
        })
    return pd.DataFrame(rows)[COLUMNS], None


def main():
    check = "--check" in sys.argv
    problems, floors = [], 0
    print("=" * 96)
    print("EXACT PAIRED WILCOXON SIGNED-RANK TABLES" + ("  (--check: nothing is written)" if check else ""))
    print("=" * 96)

    for case, (sub, scope) in CASES.items():
        t, err = build(case)
        if err:
            print(f"\n{case}: {err}")
            problems.append(err)
            continue
        out = RESULTS / case / sub / "wilcoxon_holm_exact.csv"
        print(f"\n{scope}  ({out.relative_to(HERE).as_posix()})")
        print(t[["comparison", "wilcoxon_statistic", "p_value_exact", "holm_adjusted_p_exact",
                 "holm_reject_0_05", "favours", "p_value_normal_approx",
                 "approx_at_floor"]].to_string(index=False))

        nf = int(t["approx_at_floor"].sum())
        floors += nf
        if nf:
            print(f"  {nf} of {len(t)} comparisons have W=0: all {N_PAIRS} pairs on one side, so the "
                  f"normal approximation is saturated there and its p-value is not a measurement. "
                  f"The exact test reports {EXACT_FLOOR:.6e}.")

        # Reproduction check against whatever is already shipped, on the columns it has.
        if out.exists():
            old = pd.read_csv(out)
            shared = [c for c in COLUMNS if c in old.columns
                      and c not in ("approx_at_floor", "comparison", "scope")]
            was = str(old["scope"].iloc[0])
            if was != scope:
                print(f"  scope relabelled {was!r} -> {scope!r} to match the manuscript's case name")
            merged = old[["comparison"] + shared].merge(
                t[["comparison"] + shared], on="comparison", suffixes=("_old", "_new"))
            if len(merged) != len(t):
                problems.append(f"{case}: shipped table has different comparisons")
            for c in shared:
                if c in ("scope", "comparison", "favours"):
                    bad = (merged[f"{c}_old"] != merged[f"{c}_new"]).sum()
                else:
                    a, b = merged[f"{c}_old"].astype(float), merged[f"{c}_new"].astype(float)
                    bad = int((~((a - b).abs() <= 1e-12 * b.abs().clip(lower=1e-30))).sum())
                if bad:
                    problems.append(f"{case}: {bad} value(s) of {c} do not reproduce")
            miss = [c for c in COLUMNS if c not in old.columns]
            if miss:
                print(f"  shipped table predates this script and lacks {miss}; "
                      f"{'run without --check to add them' if check else 'added'}")

        for (c, comp), want in PAPER.items():
            if c != case:
                continue
            got = float(t.loc[t["comparison"] == comp, "holm_adjusted_p_exact"].iloc[0])
            # the manuscript prints two significant figures; agreement to that precision is the claim
            ok = f"{got:.1e}" == f"{want:.1e}"
            print(f"  manuscript quotes {want:.3g} for {comp}: computed {got:.4e}  "
                  f"{'agrees' if ok else 'DISAGREES'}")
            if not ok:
                problems.append(f"{case}/{comp}: manuscript {want:.3g} vs computed {got:.4e}")

        if not check:
            t.to_csv(out, index=False)

    print("\n" + "=" * 96)
    print(f"{floors} comparison(s) across all cases sit on the normal-approximation floor; "
          f"the exact test separates them.")
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print("   " + p)
    else:
        print("Every shipped value reproduces, and every manuscript number agrees with the data.")
    print("=" * 96)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
