# -*- coding: utf-8 -*-
"""Two controls that explain WHY the surrogate ranking and the circuit ranking disagree.

Section 5 runs SBGR at D = 12 and D = 30 with its frozen thresholds. Section 6 runs a SKY130
bandgap at D = 7 and D = 16 with thresholds re-anchored to that circuit. So the disagreement
between the two rankings has three candidate explanations, and only one of them is the one the
paper wants to claim:

  (a) DIMENSION. The circuit simply has fewer sizable devices. Rankings move with dimension, and
      nobody has checked whether SBGR's ordering survives being run at 7 and 16.
  (b) CONSTRAINT PRESSURE. Measured here: 0 of 20,000 uniform samples are feasible on SBGR at
      D = 12 (base, hard) and D = 30 (highdim), against 11 of 512 -- 2.15 % -- on the circuit.
      That is roughly four orders of magnitude. SBGR is a find-any-feasible-point problem; the
      circuit is a polish-an-easy-optimum problem. Those reward opposite behaviours.
  (c) THE EVALUATOR. What the paper would like to conclude.

Neither control touches Section 5. Both add NEW conditions on the same surrogate, with the same
code, the same protocol, and the same seeds; the published D = 12 run is included as the control
on the control and must reproduce ABC 2.300 / PSO 3.000 / GA 3.033 / ACO 3.100 / FA 4.633 /
GWO 4.933.

problems.py and algorithms.py are read-only and are not modified: the relaxed thresholds are set
on the problem INSTANCE, which is what _build_constraints would have returned had the case been
defined that way.

    python sbgr_controls.py --experiment dim      --job-index $SLURM_ARRAY_TASK_ID --outdir DIR
    python sbgr_controls.py --experiment pressure --job-index $SLURM_ARRAY_TASK_ID --outdir DIR
    python sbgr_controls.py --experiment dim      --merge --outdir DIR
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from algorithms import ALGORITHMS, ObjectiveWrapper                    # noqa: E402
from problems import SBGRProblem                                       # noqa: E402

ALG = ["ABC", "GWO", "FA", "PSO", "GA", "ACO"]
POP, BUDGET, RUNS, SEED0 = 40, 150_000, 30, 42        # run.py:536 default; base case_offset = 0
PUB = {"ABC": 2.300, "PSO": 3.000, "GA": 3.033, "ACO": 3.100, "FA": 4.633, "GWO": 4.933}
DIMS = [12, 7, 16]
TARGET_FEAS = 0.0215        # the circuit's measured random feasibility, 11 of 512

FIELDS = ["experiment", "condition", "dim", "run", "seed", "algo",
          "best_f", "PSRR_DB", "penalty", "is_feasible", "runtime_sec"]


class WithFailures:
    """SBGR with contiguous unmeasurable regions, mirroring the one property of the circuit that
    the surrogate cannot have by construction.

    29.2 % of highdim's circuit evaluations returned no measurement at all: the amplifier has no
    unity-gain crossing there, so the loop-gain measurement has nothing to find. SBGR returns a
    value at every point in its box, always. That is a structural difference, and it is the third
    candidate mechanism -- after dimension and constraint pressure, both already excluded.

    The dead region is a union of three random half-spaces rather than scattered points, because
    the circuit's failures are contiguous: they are where the topology stops working, not where a
    solver happens to hiccup. A failure returns exactly what spice_problem.py returns -- the PSRR
    floor and a full penalty, i.e. fitness 6200 -- so a failed point can never win an argmin.
    The region is a deterministic function of x, so the evaluator stays reproducible.
    """

    def __init__(self, inner, frac: float, seed: int, planes: int = 3):
        self._p = inner
        self.name, self.dim, self.lb, self.ub = inner.name, inner.dim, inner.lb, inner.ub
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(planes, inner.dim))
        self._A = A / np.linalg.norm(A, axis=1, keepdims=True)
        S = rng.uniform(0.0, 1.0, size=(20000, inner.dim)) @ self._A.T
        lo, hi = -6.0, 6.0                       # calibrate so the union covers `frac`
        for _ in range(40):
            t = 0.5 * (lo + hi)
            if (S > t).any(axis=1).mean() > frac:
                lo = t
            else:
                hi = t
        self._t = 0.5 * (lo + hi)
        self.dead_frac = float((S > self._t).any(axis=1).mean())

    def _dead(self, x) -> bool:
        u = (np.asarray(x, dtype=float) - self._p.lb) / (self._p.ub - self._p.lb)
        return bool((self._A @ u > self._t).any())

    def _failed(self) -> dict:
        m = {k: float("nan") for k in
             ("VREF", "TC", "LOOP_GAIN_DB", "PHASE_MARGIN_DEG", "GAIN_MARGIN_DB", "POWER_UW")}
        m.update({"PSRR_DB": -200.0, "penalty": 6, "is_feasible": 0})
        m.update({k: 1 for k in ("viol_vref", "viol_tc", "viol_loop_gain",
                                 "viol_phase_margin", "viol_gain_margin", "viol_power")})
        return m

    def evaluate_with_metrics(self, x):
        if self._dead(x):
            m = self._failed()
            return 6200.0, m           # -(-200) + 1000*6, identical to the SPICE failure floor
        return self._p.evaluate_with_metrics(x)

    def metrics(self, x) -> dict:
        return self._failed() if self._dead(x) else self._p.metrics(x)


def relaxed(base: dict, s: float) -> dict:
    """Loosen every specification by the same relative amount s, in its own direction."""
    mid = 0.5 * (base["VREF_min"] + base["VREF_max"])
    half = 0.5 * (base["VREF_max"] - base["VREF_min"])
    return {
        "VREF_min": mid - half * (1 + s), "VREF_max": mid + half * (1 + s),
        "TC_max": base["TC_max"] * (1 + s),
        "LoopGain_min": base["LoopGain_min"] * max(0.0, 1 - s),
        "PhaseMargin_min": base["PhaseMargin_min"] * max(0.0, 1 - s),
        "GainMargin_min": base["GainMargin_min"] * max(0.0, 1 - s),
        "Power_max": base["Power_max"] * (1 + s),
    }


def feas_rate(dim: int, cons: dict, n: int = 4000, seed: int = 7) -> float:
    p = SBGRProblem(dim=dim, seed=SEED0, case="base")
    p._constraints = dict(cons)
    rng = np.random.default_rng(seed)
    X = rng.uniform(p.lb, p.ub, size=(n, dim))
    return sum(int(p.metrics(x)["is_feasible"]) for x in X) / n


def calibrate(dim: int = 12) -> tuple[dict, float, float]:
    """Find the relaxation that puts SBGR's random feasibility at the circuit's 2.15 %."""
    base = SBGRProblem(dim=dim, seed=SEED0, case="base").constraints
    lo, hi = 0.0, 8.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if feas_rate(dim, relaxed(base, mid)) < TARGET_FEAS:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    c = relaxed(base, s)
    return c, s, feas_rate(dim, c, n=20000)


def conditions(exp: str):
    if exp == "dim":
        return [(f"D{d}", d, None) for d in DIMS]
    if exp == "failure":
        return [("clean", 12, None), ("failures", 12, "FAIL")]
    cons, s, fr = calibrate()
    print(f"[calibrate] relaxation s={s:.4f} -> random feasibility {100*fr:.3f} % "
          f"(target {100*TARGET_FEAS:.2f} %)", flush=True)
    print(f"[calibrate] {cons}", flush=True)
    return [("frozen", 12, None), ("relaxed", 12, cons)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["dim", "pressure", "failure"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--job-index", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    conds = conditions(args.experiment)

    if args.merge:
        files = sorted((out / "rows").glob("*.csv"))
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        expect = len(conds) * RUNS * len(ALG)
        print(f"{len(files)} row files, {len(d)} rows (expected {expect})")
        if len(d) != expect:
            print("REFUSING to summarise an incomplete set")
            return 1
        try:
            from scipy.stats import spearmanr
            rr = lambda a, b: float(spearmanr([a[k] for k in sorted(a)],
                                              [b[k] for k in sorted(b)]).statistic)
        except Exception:
            rr = lambda a, b: float(np.corrcoef(
                pd.Series([a[k] for k in sorted(a)]).rank(),
                pd.Series([b[k] for k in sorted(b)]).rank())[0, 1])
        print(f"\n  {'condition':10s} " + "".join(f"{a:>9s}" for a in ALG)
              + f"{'ABC':>5s}{'feas%':>8s}{'rho vs published':>18s}")
        for name, dim, _ in conds:
            s = d[d.condition == name]
            r = s.pivot_table(index="run", columns="algo",
                              values="best_f").rank(axis=1).mean().to_dict()
            o = sorted(r, key=lambda z: r[z])
            print(f"  {name:10s} " + "".join(f"{r[a]:9.3f}" for a in ALG)
                  + f"{o.index('ABC')+1:5d}{100*s.is_feasible.mean():7.1f}%{rr(PUB, r):+18.4f}"
                  + f"   {' < '.join(o)}")
        print(f"  {'published':10s} " + "".join(f"{PUB[a]:9.3f}" for a in ALG) + f"{1:5d}")
        d.to_csv(out / "all_rows.csv", index=False)
        return 0

    total = len(conds) * RUNS
    if args.job_index is None or not 0 <= args.job_index < total:
        print(f"--job-index must be in [0, {total}) for {len(conds)} conditions x {RUNS} runs")
        return 2
    ci, run = divmod(args.job_index, RUNS)
    name, dim, cons = conds[ci]
    dst = out / "rows" / f"{args.experiment}_{name}_run{run:03d}.csv"
    if dst.exists():
        print(f"{dst.name} exists; nothing to do")
        return 0

    seed = SEED0 + run * 1000
    rows = []
    for a in ALG:
        p = SBGRProblem(dim=dim, seed=seed, case="base")
        if cons == "FAIL":
            # 0.292 is highdim's measured share of evaluations that returned no measurement
            p = WithFailures(p, frac=0.292, seed=90000 + run)
        elif cons is not None:
            p._constraints = dict(cons)
        obj = ObjectiveWrapper(func=None, eval_with_metrics=p.evaluate_with_metrics,
                               max_evals=args.budget)
        t0 = time.perf_counter()
        res = ALGORITHMS[a](obj=obj, rng=np.random.default_rng(seed),
                            lb=p.lb, ub=p.ub, pop=POP, dim=dim)
        dt = time.perf_counter() - t0
        m = p.metrics(res.best_x)
        rows.append({"experiment": args.experiment, "condition": name, "dim": dim, "run": run,
                     "seed": seed, "algo": a, "best_f": float(res.best_f),
                     "PSRR_DB": float(m["PSRR_DB"]), "penalty": int(m["penalty"]),
                     "is_feasible": int(m["is_feasible"]), "runtime_sec": dt})
        print(f"  {name} run {run} {a}: {res.best_f:.4f} ({dt:.0f}s)", flush=True)
    tmp = dst.with_suffix(".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
