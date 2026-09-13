"""
protocol_sweep.py -- is the SPICE ranking difference caused by the PROTOCOL or by the EVALUATOR?

THE QUESTION. The published surrogate campaign ranks ABC first and GWO last. The SPICE-Base
campaign ranked GWO first and ABC second, and the rank correlation between the two orderings is
essentially zero. Read naively that says the surrogate's optimizer ranking does not transfer to
a real circuit. But the two campaigns differ in three ways at once, so the naive reading is not
yet earned:

    published surrogate : D = 12,  SN = 40,  150,000 evaluations per run
    SPICE campaign      : D =  7,  SN = 20,    2,500 evaluations per run

Any of the three could produce the difference on its own, and 150,000 evaluations per run is
unreachable with a real simulator (180 runs x 150,000 x 0.416 s = 3,120 CPU-hours). So the
attribution cannot be settled by making SPICE match; it has to be settled by making the
SURROGATE match SPICE. That is what this file does, and it costs minutes instead of months:

    run the surrogate AT THE SPICE PROTOCOL (D = 7, SN = 20/40, 2,500 evaluations).
      * if the ranking still puts ABC near the top and GWO last, the protocol is exonerated and
        the transfer failure is a property of the EVALUATOR -- the real circuit's landscape.
      * if the ranking instead moves toward the SPICE ordering, the transfer failure was an
        artefact of comparing a 2,500-evaluation result against a 150,000-evaluation one.

A SECOND, INDEPENDENT FINDING THIS MEASURES. ABC abandons a food source after
`limit = floor(0.6 * SN * D)` consecutive failed trials. At D = 7 that is 168 for SN = 40, while
the reachable high-water mark of any trial counter within 2,500 evaluations is about 129 -- so
the scout phase fires ZERO times and ABC degenerates to a greedy local search with no restart
mechanism. Note the direction: `limit` grows with SN while the number of generations shrinks
with SN, so raising SN from 20 to 40 in the name of protocol compliance makes the mechanism
LESS active, not more. The scout counter is therefore measured here rather than argued about,
because "ABC's distinguishing mechanism was inactive" is either a confound to disclose or a
finding to report, and only a number can decide which.

STAGES (each writes its own CSV; --stage selects, default all):
    validate  D=12, SN=40, 150k -- must reproduce the published ranks, or this harness is wrong
              and nothing else in this file means anything. This is a GATE, not a result.
    control   D=7, 2,500, SN in {20, 40}, several seed families -- the decisive comparison
    sn        D=7, 2,500, SN swept -- isolates population, and where the scout phase dies
    budget    D=7, SN=40, budget swept -- says whether buying more SPICE compute would help,
              i.e. whether the 17 days needed to revive the scout phase would buy anything

    python protocol_sweep.py                       # everything, ~25 min
    python protocol_sweep.py --stage control        # the decisive one, ~2 min
    python protocol_sweep.py --quick               # smoke test, ~30 s
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import algorithms as A                                   # noqa: E402
from algorithms import ALGORITHMS, ObjectiveWrapper      # noqa: E402
from problems import SBGRProblem                         # noqa: E402
import run as R                                          # noqa: E402

ALGOS = ["ABC", "GWO", "FA", "PSO", "GA", "ACO"]

# The two orderings being attributed between.
#   PUBLISHED: manuscript Table tab:core_quant_results, SBGR-Base row block.
#   SPICE    : the first SPICE-Base campaign (D=7, SN=20, 2,500 evaluations, 30 runs).
PUBLISHED = {"ABC": 2.300, "PSO": 3.000, "GA": 3.033, "ACO": 3.100, "FA": 4.633, "GWO": 4.933}
SPICE = {"GWO": 1.833, "ABC": 2.867, "ACO": 3.267, "PSO": 3.900, "FA": 4.400, "GA": 4.733}

# The published campaign's own settings, reproduced exactly: run.py:598 forms the per-run seed
# as `args.seed + case_offset + run * 1000` with `case_offset["base"] == 0` and --seed 42.
PUB_DIM, PUB_POP, PUB_BUDGET, PUB_SEED = 12, 40, 150_000, 42
SPICE_DIM, SPICE_BUDGET = 7, 2_500


# ------------------------------------------------------------------ scout instrumentation
def _make_instrumented_abc():
    """A verbatim copy of abc_optimize with a scout counter added.

    Copied rather than imported-and-patched because algorithms.py is read-only for this project.
    Only counter statements are inserted -- no rng call is added, removed or reordered -- so the
    search trajectory is bit-identical to the real ABC. The anchors are asserted unique so a
    future edit to algorithms.py makes this fail loudly instead of silently measuring nothing.
    """
    src = inspect.getsource(A.abc_optimize).replace("def abc_optimize(", "def _abc_i(", 1)
    a1 = ("            if trials[i] >= limit:\n"
          "                X[i] = rng.uniform(lb, ub, size=(dim,))")
    a2 = "        # Scout bees\n"
    if src.count(a1) != 1 or src.count(a2) != 1:
        raise RuntimeError("abc_optimize no longer matches the instrumentation anchors; "
                           "re-check the scout block before trusting any scout count")
    src = src.replace(a1, "            if trials[i] >= limit:\n"
                          "                _P['scouts'] += 1\n"
                          "                X[i] = rng.uniform(lb, ub, size=(dim,))")
    src = src.replace(a2, "        _P['tmax'] = max(_P['tmax'], int(trials.max()))\n" + a2)
    probe = {"scouts": 0, "tmax": 0}
    ns = dict(vars(A))
    ns["_P"] = probe
    exec(compile(src, "<abc_instrumented>", "exec"), ns)
    return ns["_abc_i"], probe


_ABC_I, _PROBE = _make_instrumented_abc()


def scout_activity(dim: int, pop: int, budget: int, reps: int, seed_base: int) -> dict:
    """Mean scouts per run and the high-water mark of any trial counter."""
    prob = SBGRProblem(dim=dim, seed=seed_base, case="base")
    n, tmax = [], 0
    for r in range(reps):
        _PROBE["scouts"], _PROBE["tmax"] = 0, 0
        obj = ObjectiveWrapper(func=None, eval_with_metrics=prob.evaluate_with_metrics,
                               max_evals=budget, progress_cb=None)
        _ABC_I(obj=obj, rng=np.random.default_rng(seed_base + r * 1000),
               lb=prob.lb, ub=prob.ub, pop=pop, dim=dim)
        n.append(_PROBE["scouts"])
        tmax = max(tmax, _PROBE["tmax"])
    return {"scouts_per_run": float(np.mean(n)), "trial_high_water": tmax,
            "limit": int(0.6 * pop * dim), "generations": budget // (2 * pop)}


# ------------------------------------------------------------------ campaign + ranking
def campaign(dim: int, pop: int, budget: int, runs: int, seed_base: int) -> pd.DataFrame:
    """One paired-block campaign, mirroring run.py's inner loop exactly: one problem instance
    per run shared by all algorithms, and the same seed for every algorithm within a run."""
    rows = []
    for k in range(runs):
        s = seed_base + k * 1000
        prob = SBGRProblem(dim=dim, seed=s, case="base")
        for algo in ALGOS:
            obj = ObjectiveWrapper(func=None, eval_with_metrics=prob.evaluate_with_metrics,
                                   max_evals=budget, progress_cb=None)
            res = ALGORITHMS[algo](obj=obj, rng=np.random.default_rng(s),
                                   lb=prob.lb, ub=prob.ub, pop=pop, dim=dim)
            m = prob.metrics(res.best_x)
            rows.append({"dim": dim, "pop": pop, "budget": budget, "seed_base": seed_base,
                         "run": k, "algo": algo, "best_fitness": float(res.best_f),
                         "is_feasible": int(m.get("is_feasible", 0)),
                         "penalty": int(m.get("penalty", 0)),
                         "PSRR_DB": float(m.get("PSRR_DB", np.nan))})
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    pivot = df.pivot(index="run", columns="algo", values="best_fitness").reindex(columns=ALGOS)
    fried, ranks = R.friedman_from_pivot(pivot, ALGOS, "sweep")
    avg = dict(zip(ranks["algo"], ranks["avg_rank"])) if len(ranks) else {}
    v = [avg.get(a, np.nan) for a in ALGOS]
    succ = df.groupby("algo")["is_feasible"].mean().reindex(ALGOS)
    return {
        "ranks": avg,
        "success": {a: float(succ[a]) for a in ALGOS},
        "chi2": float(fried["friedman_chi2"].iloc[0]) if len(fried) else float("nan"),
        "p": float(fried["friedman_p"].iloc[0]) if len(fried) else float("nan"),
        "rho_published": float(spearmanr([PUBLISHED[a] for a in ALGOS], v).statistic),
        "rho_spice": float(spearmanr([SPICE[a] for a in ALGOS], v).statistic),
    }


def order(d: dict) -> str:
    return "  <  ".join(a for a, _ in sorted(d.items(), key=lambda kv: kv[1]))


def report(label: str, res: dict, scouts: dict | None = None, dt: float | None = None) -> None:
    head = f"=== {label}"
    if dt is not None:
        head += f"   ({dt:.0f}s)"
    print(head)
    print(f"  chi2 = {res['chi2']:.2f}   p = {res['p']:.2e}")
    print(f"  ranking : {order(res['ranks'])}")
    print("  ranks   : " + "  ".join(f"{a}={res['ranks'].get(a, float('nan')):.3f}"
                                     for a in ALGOS))
    print("  success : " + "  ".join(f"{a}={100*res['success'][a]:.0f}%" for a in ALGOS))
    print(f"  Spearman vs published {res['rho_published']:+.3f}   "
          f"vs SPICE campaign {res['rho_spice']:+.3f}")
    if scouts:
        print(f"  ABC scouts: {scouts['scouts_per_run']:.2f}/run   "
              f"limit={scouts['limit']}   generations={scouts['generations']}   "
              f"trial high-water={scouts['trial_high_water']}"
              + ("   <-- SCOUT PHASE NEVER FIRES"
                 if scouts["scouts_per_run"] == 0 else ""))
    print()


def rowify(label: str, dim: int, pop: int, budget: int, seed_base: int,
           res: dict, scouts: dict | None) -> dict:
    r = {"label": label, "dim": dim, "pop": pop, "budget": budget, "seed_base": seed_base,
         "chi2": res["chi2"], "p": res["p"],
         "rho_published": res["rho_published"], "rho_spice": res["rho_spice"],
         "ordering": order(res["ranks"])}
    r.update({f"rank_{a}": res["ranks"].get(a, np.nan) for a in ALGOS})
    r.update({f"success_{a}": res["success"][a] for a in ALGOS})
    r.update(scouts or {})
    return r


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["validate", "control", "sn", "budget"],
                    choices=["validate", "control", "sn", "budget"])
    ap.add_argument("--runs", type=int, default=30, help="30 for parity with both campaigns")
    ap.add_argument("--reps", type=int, default=8, help="repetitions for the scout counter")
    ap.add_argument("--out", default="results_protocol_sweep")
    ap.add_argument("--quick", action="store_true",
                    help="tiny version for verifying the script itself, not for reporting")
    args = ap.parse_args()

    runs, reps = (4, 2) if args.quick else (args.runs, args.reps)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"surrogate protocol sweep | runs={runs} | "
          f"{'QUICK -- not for reporting' if args.quick else 'full'}\n")
    print("the two orderings being attributed between (lower average rank is better)")
    print(f"  published surrogate  D=12, SN=40, 150k : {order(PUBLISHED)}")
    print(f"  SPICE campaign       D= 7, SN=20, 2.5k : {order(SPICE)}\n")

    summary, raw, gate_ok = [], [], None

    # ---- GATE: reproduce the published campaign, or stop ----
    if "validate" in args.stage:
        dim, pop = PUB_DIM, PUB_POP
        budget = 6_000 if args.quick else PUB_BUDGET
        t0 = time.perf_counter()
        df = campaign(dim, pop, budget, runs, PUB_SEED)
        res = analyse(df)
        raw.append(df)
        report(f"GATE  published protocol  D={dim} SN={pop} budget={budget}", res,
               dt=time.perf_counter() - t0)
        summary.append(rowify("validate", dim, pop, budget, PUB_SEED, res, None))
        # None, not False, in quick mode: the budget is reduced there, so the gate is not
        # applicable and must not be reported as a failure.
        gate_ok = None if args.quick else res["rho_published"] >= 0.90
        if args.quick:
            print("  (quick mode: budget reduced, so the gate is not evaluated)\n")
        elif gate_ok:
            print(f"  GATE PASSED: rho = {res['rho_published']:+.3f} against the published "
                  f"ranks, so this harness reproduces the published campaign and the cells "
                  f"below can be trusted.\n")
        else:
            print(f"  GATE FAILED: rho = {res['rho_published']:+.3f} against the published "
                  f"ranks. This harness does not reproduce the published campaign, so NO "
                  f"conclusion below is usable. Fix the harness first.\n")

    # ---- THE DECISIVE CELL: surrogate at the SPICE protocol ----
    if "control" in args.stage:
        seed_families = [PUB_SEED] if args.quick else [PUB_SEED, 777, 20260806, 999983]
        for pop in (40, 20):
            for sb in seed_families:
                t0 = time.perf_counter()
                df = campaign(SPICE_DIM, pop, SPICE_BUDGET, runs, sb)
                res = analyse(df)
                raw.append(df)
                sc = scout_activity(SPICE_DIM, pop, SPICE_BUDGET, reps, sb)
                report(f"CONTROL  surrogate at the SPICE protocol  D={SPICE_DIM} SN={pop} "
                       f"budget={SPICE_BUDGET}  seed family {sb}", res, sc,
                       time.perf_counter() - t0)
                summary.append(rowify(f"control_SN{pop}", SPICE_DIM, pop, SPICE_BUDGET,
                                      sb, res, sc))

    # ---- population axis ----
    if "sn" in args.stage:
        pops = (10, 40) if args.quick else (6, 10, 12, 16, 20, 30, 40)
        for pop in pops:
            t0 = time.perf_counter()
            df = campaign(SPICE_DIM, pop, SPICE_BUDGET, runs, PUB_SEED)
            res = analyse(df)
            raw.append(df)
            sc = scout_activity(SPICE_DIM, pop, SPICE_BUDGET, reps, PUB_SEED)
            report(f"SN sweep  D={SPICE_DIM} SN={pop} budget={SPICE_BUDGET}", res, sc,
                   time.perf_counter() - t0)
            summary.append(rowify("sn_sweep", SPICE_DIM, pop, SPICE_BUDGET, PUB_SEED, res, sc))

    # ---- budget axis: would buying more SPICE compute change anything? ----
    if "budget" in args.stage:
        budgets = (2_500, 5_000) if args.quick else (2_500, 5_000, 10_000, 20_000, 40_000,
                                                    80_000)
        for b in budgets:
            t0 = time.perf_counter()
            df = campaign(SPICE_DIM, PUB_POP, b, runs, PUB_SEED)
            res = analyse(df)
            raw.append(df)
            sc = scout_activity(SPICE_DIM, PUB_POP, b, reps, PUB_SEED)
            cpu_h = 180 * b * 0.416 / 3600
            report(f"budget sweep  D={SPICE_DIM} SN={PUB_POP} budget={b}   "
                   f"[this budget would cost {cpu_h:.0f} CPU-h in SPICE, "
                   f"{cpu_h/4:.0f} h on 4 workers]", res, sc, time.perf_counter() - t0)
            summary.append(rowify("budget_sweep", SPICE_DIM, PUB_POP, b, PUB_SEED, res, sc))

    sm = pd.DataFrame(summary)
    sm.to_csv(outdir / "summary.csv", index=False)
    if raw:
        pd.concat(raw, ignore_index=True).to_csv(outdir / "per_run.csv", index=False)
    print(f"wrote {(outdir / 'summary.csv').resolve()}")
    print(f"wrote {(outdir / 'per_run.csv').resolve()}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    ctl = sm[sm.label.str.startswith("control")] if len(sm) else sm
    if len(ctl):
        pub = ctl["rho_published"].mean()
        spi = ctl["rho_spice"].mean()
        print(f"  surrogate at the SPICE protocol, averaged over {len(ctl)} cells:")
        print(f"    Spearman vs the PUBLISHED ordering : {pub:+.3f}")
        print(f"    Spearman vs the SPICE   ordering   : {spi:+.3f}")
        if pub > 0.5 and pub > spi + 0.4:
            print("\n  -> THE PROTOCOL IS EXONERATED. Run at the SPICE dimension, population")
            print("     and budget, the surrogate still reproduces its own published ordering.")
            print("     The SPICE ranking difference is therefore a property of the EVALUATOR")
            print("     -- the real circuit's landscape -- and not of the reduced budget. The")
            print("     existing SPICE campaign is a valid comparison and does not need to be")
            print("     re-run at a different population.")
        elif spi > pub + 0.4:
            print("\n  -> THE PROTOCOL EXPLAINS IT. At matched protocol the surrogate moves to")
            print("     the SPICE ordering, so the earlier reading compared a 2,500-evaluation")
            print("     result against a 150,000-evaluation one. Report it as a budget effect.")
        else:
            print("\n  -> INCONCLUSIVE: neither ordering is clearly reproduced. Report the")
            print("     correlations themselves and claim no attribution.")
    dead = sm[(sm.get("scouts_per_run") == 0)] if "scouts_per_run" in sm else sm.iloc[0:0]
    if len(dead):
        print(f"\n  ABC's scout phase fires ZERO times in {len(dead)} of the measured cells, "
              f"including\n  SN=40 at 2,500 evaluations. `limit` grows with SN while the "
              f"generation count shrinks,\n  so raising SN toward the published 40 makes the "
              f"mechanism LESS active. Disclose the\n  measured scout count rather than "
              f"implying the published configuration was reproduced.")
    if gate_ok is False:
        print("\n  WARNING: the validate gate FAILED, so treat everything above as untrusted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
