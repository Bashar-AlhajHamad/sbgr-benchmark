"""
penalty_sensitivity.py
======================
Penalty-scaling sensitivity study for the SBGR benchmark (Reviewer #3, Weakness 4).

The manuscript uses the penalized fitness  f(x) = -PSRR(x) + lambda * penalty(x)
with lambda = 1000 (reproduced from the motivating BGR case study). A reviewer noted
that, because PSRR is ~O(100 dB) while one violation adds `lambda` units, the ranking
is feasibility-dominated, and asked whether the ABC advantage is an artifact of this
specific scaling.

This script re-runs the SAME comparison protocol as run.py (same cases, budgets, paired
seeding, population) for several values of lambda, and reports — per lambda —
    (i)  each algorithm's success rate (fraction of runs whose final best is feasible),
    (ii) each algorithm's average rank on final penalized fitness (lower is better),
    (iii) whether ABC remains the top-ranked / highest-success method.

Nothing about the benchmark changes except lambda; problems.py is NOT modified (we use a
thin subclass that overrides only the fitness scaling), so the lambda=1000 column
reproduces the main-paper setting exactly.

USAGE (full scale, matching the paper — heavy; run on your machine):
    python penalty_sensitivity.py --runs 30 --pop 40 \
        --max-evals 150000 --highdim-max-evals 220000 \
        --lambdas 100,1000,10000 --outdir results_penalty

QUICK PILOT (fast, to verify behavior / preview the trend):
    python penalty_sensitivity.py --runs 5 --pop 40 \
        --max-evals 12000 --highdim-max-evals 15000 \
        --lambdas 10,100,1000,10000 --outdir results_penalty_pilot

Outputs (in --outdir):
    per_run_records.csv      one row per (lambda, case, run, algorithm)
    summary_by_lambda.csv    success rate + average rank per (lambda, case, algorithm)
    verdict.txt              plain-text robustness verdict per lambda
"""

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from problems import SBGRProblem
from algorithms import ObjectiveWrapper, ALGORITHMS

# case -> seed offset, identical to run.py so instances match the main campaign
CASE_OFFSET = {"base": 0, "hard": 100000, "highdim": 200000}


class SBGRProblemLambda(SBGRProblem):
    """SBGR with a configurable penalty weight; everything else is inherited unchanged."""

    def __init__(self, dim: int, seed: int, case: str, penalty_weight: float = 1000.0):
        super().__init__(dim=dim, seed=seed, case=case)
        # SBGRProblem is not effectively frozen for new attributes (verified), so this is safe.
        self.penalty_weight = float(penalty_weight)

    def fitness_from_metrics(self, m) -> float:
        return float(-m["PSRR_DB"] + self.penalty_weight * m["penalty"])


def dim_for_case(case: str, base_dim: int, highdim_dim: int) -> int:
    return max(base_dim, highdim_dim) if case == "highdim" else base_dim


def budget_for_case(case: str, base_evals: int, highdim_evals: int) -> int:
    return highdim_evals if case == "highdim" else base_evals


def run_campaign(args) -> pd.DataFrame:
    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows = []
    total = len(lambdas) * len(args.cases) * args.runs * len(args.algos)
    job = 0
    t_start = time.perf_counter()

    for lam in lambdas:
        for case in args.cases:
            cdim = dim_for_case(case, args.dim, args.highdim_dim)
            budget = budget_for_case(case, args.max_evals, args.highdim_max_evals)
            for run in range(args.runs):
                run_seed = args.seed + CASE_OFFSET[case] + run * 1000
                # one problem instance per (case, run), shared by all algorithms (paired design)
                problem = SBGRProblemLambda(dim=cdim, seed=run_seed, case=case, penalty_weight=lam)
                for algo in args.algos:
                    job += 1
                    rng = np.random.default_rng(run_seed)  # same seed per run for every algorithm
                    obj = ObjectiveWrapper(func=None,
                                           eval_with_metrics=problem.evaluate_with_metrics,
                                           max_evals=budget)
                    t0 = time.perf_counter()
                    res = ALGORITHMS[algo](obj=obj, rng=rng, lb=problem.lb, ub=problem.ub,
                                           pop=args.pop, dim=cdim)
                    dt = time.perf_counter() - t0
                    m = problem.metrics(res.best_x)  # metrics of the final best solution
                    rows.append({
                        "lambda": lam,
                        "case": case,
                        "run": run,
                        "algo": algo,
                        "best_fitness": float(res.best_f),
                        "PSRR_DB": float(m["PSRR_DB"]),
                        "penalty": int(m["penalty"]),
                        "is_feasible": int(m["is_feasible"]),
                        "runtime_sec": dt,
                    })
                    if job % 25 == 0 or job == total:
                        el = time.perf_counter() - t_start
                        print(f"  [{job}/{total}] lambda={lam:g} {case} run{run+1} {algo} "
                              f"| best={res.best_f:.4g} feas={m['is_feasible']} | elapsed={el:.1f}s",
                              flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # rank within each (lambda, case, run) block by final penalized fitness (lower is better)
    df = df.copy()
    df["rank"] = (df.groupby(["lambda", "case", "run"])["best_fitness"]
                    .rank(method="average", ascending=True))
    summ = (df.groupby(["lambda", "case", "algo"])
              .agg(success_rate=("is_feasible", "mean"),
                   mean_fitness=("best_fitness", "mean"),
                   median_fitness=("best_fitness", "median"),
                   avg_rank=("rank", "mean"),
                   n=("run", "count"))
              .reset_index())
    return df, summ


def build_verdict(df: pd.DataFrame, summ: pd.DataFrame, lambdas: List[float]) -> str:
    lines = []
    lines.append("PENALTY-SCALING SENSITIVITY — ROBUSTNESS VERDICT")
    lines.append("=" * 62)
    lines.append("Per lambda: overall average rank and success rate across ALL case/run")
    lines.append("blocks (lower rank is better). ABC is the proposed method.\n")
    # overall (pooled across cases) ranks per (lambda, algo)
    dd = df.copy()
    dd["rank_overall"] = (dd.groupby(["lambda", "case", "run"])["best_fitness"]
                            .rank(method="average", ascending=True))
    for lam in sorted(set(df["lambda"])):
        sub = dd[dd["lambda"] == lam]
        by_algo = (sub.groupby("algo")
                      .agg(avg_rank=("rank_overall", "mean"),
                           success=("is_feasible", "mean"))
                      .sort_values("avg_rank"))
        top_rank_algo = by_algo.index[0]
        top_succ_algo = by_algo["success"].idxmax()
        lines.append(f"lambda = {lam:g}")
        for algo, r in by_algo.iterrows():
            tag = "  <-- ABC" if algo == "ABC" else ""
            lines.append(f"    {algo:5s}  avg_rank={r['avg_rank']:.3f}  success={r['success']*100:5.1f}%{tag}")
        holds = (top_rank_algo == "ABC") and (top_succ_algo == "ABC")
        lines.append(f"    => ABC best by rank: {top_rank_algo=='ABC'} | "
                     f"ABC best by success: {top_succ_algo=='ABC'} | "
                     f"conclusion holds: {holds}\n")
    # headline
    abc_rank1_all = all(
        dd[dd['lambda']==lam].groupby('algo')['rank_overall'].mean().idxmin() == 'ABC'
        for lam in set(df['lambda'])
    )
    lines.append("-" * 62)
    lines.append(f"HEADLINE: ABC is the top-ranked method at EVERY tested lambda: {abc_rank1_all}")
    lines.append("(If True, the ABC advantage is robust to the penalty scaling and is not")
    lines.append(" an artifact of the specific value lambda = 1000.)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="SBGR penalty-scaling sensitivity study")
    ap.add_argument("--cases", nargs="*", default=["base", "hard", "highdim"],
                    choices=["base", "hard", "highdim"])
    ap.add_argument("--algos", nargs="*", default=["ABC", "GWO", "FA", "PSO", "GA", "ACO"])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--dim", type=int, default=12)
    ap.add_argument("--highdim-dim", type=int, default=30)
    ap.add_argument("--max-evals", type=int, default=150000)
    ap.add_argument("--highdim-max-evals", type=int, default=220000)
    ap.add_argument("--lambdas", type=str, default="100,1000,10000",
                    help="comma-separated penalty weights, e.g. 10,100,1000,10000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results_penalty")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Penalty-sensitivity campaign: lambdas={args.lambdas} cases={args.cases} "
          f"algos={args.algos} runs={args.runs} pop={args.pop}")
    df = run_campaign(args)
    df_ranked, summ = summarize(df)

    df.to_csv(out / "per_run_records.csv", index=False)
    summ.to_csv(out / "summary_by_lambda.csv", index=False)
    verdict = build_verdict(df_ranked, summ, [float(x) for x in args.lambdas.split(",")])
    (out / "verdict.txt").write_text(verdict)

    print("\n" + verdict)
    print(f"\nSaved: {out/'per_run_records.csv'}, {out/'summary_by_lambda.csv'}, {out/'verdict.txt'}")


if __name__ == "__main__":
    main()