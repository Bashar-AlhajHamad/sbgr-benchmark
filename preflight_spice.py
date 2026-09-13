# -*- coding: utf-8 -*-
"""Pre-flight verification of the SPICE campaign that does NOT need ngspice or the PDK.

Everything here is a property of the code and the frozen thresholds, so it can be checked in
seconds. NgspiceServer is stubbed out, which is the point: it isolates the plumbing (does
--case actually reach the constraint dict? does --pop reach the optimizer? does the resume
guard fire?) from the simulator.
"""
import io
import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "spice"))

import ngspice_bridge as nb

fails, checks = [], 0


def ck(name, cond, detail=""):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- stub the simulator
class FakeServer:
    load_seconds = 0.0
    restarts = 0

    def __init__(self, deck, **kw):
        self.deck = deck

    def load(self):
        pass

    def close(self):
        pass

    def send(self, *a, **k):
        return None


nb.NgspiceServer = FakeServer
nb.find_ngspice = lambda *a, **k: "fake-ngspice"

import spice_problem as sp
import problems as P

# ---------------------------------------------------------------- 1. case plumbing
print("\n1. --case reaches the constraint set (a silent fallback to base would put a "
      "fabricated\n   Hard result in the paper)")
# Only cases with FROZEN thresholds are campaign-facing. A registered but uncalibrated case is
# probe-only; that it cannot be built for a campaign is checked separately below, as a feature.
CAMPAIGN_CASES = [c for c in sorted(sp.CASES) if sp.CALIBRATED.get(c, False)]
PROBE_ONLY = [c for c in sorted(sp.CASES) if not sp.CALIBRATED.get(c, False)]
probs = {c: sp.SpiceBGRProblem(sky130_lib="/fake/sky130.lib.spice", case=c)
         for c in CAMPAIGN_CASES}
ck("base  -> CONSTRAINTS", probs["base"].constraints == sp.CONSTRAINTS)
ck("hard  -> CONSTRAINTS_HARD", probs["hard"].constraints == sp.CONSTRAINTS_HARD)
ck("hard differs from base", probs["hard"].constraints != probs["base"].constraints)
ck("highdim keeps base thresholds", probs["highdim"].constraints == sp.CONSTRAINTS,
   "dimensionality is the only difference, as the surrogate also does")
ck("dims 7 / 7 / 16", [probs[c].dim for c in ("base", "hard", "highdim")] == [7, 7, 16])

# The two Banba cases exist to separate THRESHOLD sensitivity from topology and dimensionality, so
# they must share everything except the thresholds. If they ever diverge in deck or variables, the
# comparison silently stops measuring what it claims to.
if "banba" in probs and "banba_cal" in probs:
    ck("banba -> CONSTRAINTS_BANBA", probs["banba"].constraints == sp.CONSTRAINTS_BANBA)
    ck("banba_cal -> CONSTRAINTS_BANBA_CAL",
       probs["banba_cal"].constraints == sp.CONSTRAINTS_BANBA_CAL)
    ck("the two Banba threshold sets differ",
       probs["banba"].constraints != probs["banba_cal"].constraints)
    ck("the two Banba cases share one deck",
       probs["banba"]._srv.deck == probs["banba_cal"]._srv.deck,
       "same netlist text, so only the thresholds vary")
    ck("the two Banba cases share one variable set",
       probs["banba"]._vars == probs["banba_cal"]._vars)
    ck("Banba dim is 7, as the Kuijk Base case", probs["banba"].dim == 7)
    ck("the Banba V_REF window is the PUBLISHED one",
       (sp.CONSTRAINTS_BANBA["VREF_min"], sp.CONSTRAINTS_BANBA["VREF_max"]) == (0.798, 0.802),
       "798-802 mV, unreachable in the voltage-mode Kuijk deck")
    ck("the Banba anchor is inside its own V_REF window",
       sp.CONSTRAINTS_BANBA["VREF_min"] <= 0.8011 <= sp.CONSTRAINTS_BANBA["VREF_max"],
       "measured 801.1 mV")
# was hardcoded to 3 when there were three cases; the INTENT is "distinct", not "three", and
# a count that has to be edited every time a case is added is a check that will one day be
# edited to pass rather than to be true.
ck("names distinct", len({probs[c].name for c in probs}) == len(probs),
   str(sorted(probs[c].name for c in probs)))
ck("highdim deck is the highdim template",
   "high-dimensional" in probs["highdim"]._srv.deck)
ck("base and hard share ONE deck",
   probs["base"]._srv.deck == probs["hard"]._srv.deck,
   "identical circuit, thresholds are the only change")

# ---------------------------------------------------------------- 2. frozen thresholds
print("\n2. the frozen thresholds do what their comments claim, evaluated on the MEASURED "
      "nominal\n   point (VREF 1219.90 mV, TC 24.49 ppm/C, P 47.27 uW, LG 53.21 dB, "
      "PM 62.99 deg, GM 14.86 dB)")
# Both PSRR definitions at the nominal point, measured from the rail transfer:
# 56.29 dB at 100 Hz, 23.85 dB worst case over 10 Hz - 10 MHz.
NOMINAL_METRICS = dict(psrr_db=56.29, psrrwc_db=23.85, vref_v=1.219903, tc_ppm=24.49,
                       power_uw=47.27, lg_db=53.21, pm_deg=62.99, gm_db=14.86)
for case, want in (("base", 0), ("hard", 3)):
    m = probs[case]._metrics_from(dict(NOMINAL_METRICS), probs[case]._nominal)
    got = m["penalty"]
    viol = [k for k in m if k.startswith("viol_") and m[k]]
    ck(f"{case}: nominal penalty == {want}", got == want,
       f"got {got}, violated {viol}")
ck("base: nominal is feasible",
   probs["base"]._metrics_from(dict(NOMINAL_METRICS), {})["is_feasible"] == 1,
   "the feasible set is non-empty by construction")
ck("hard: nominal is NOT feasible",
   probs["hard"]._metrics_from(dict(NOMINAL_METRICS), {})["is_feasible"] == 0,
   "Hard must require optimization beyond the reference design")
_MINS = ("VREF_min", "LoopGain_min", "PhaseMargin_min", "GainMargin_min")
_MAXS = ("VREF_max", "TC_max", "Power_max")
ck("hard is a tightening of base in every spec",
   all(sp.CONSTRAINTS_HARD[k] >= sp.CONSTRAINTS[k] for k in _MINS)
   and all(sp.CONSTRAINTS_HARD[k] <= sp.CONSTRAINTS[k] for k in _MAXS),
   "no spec was accidentally LOOSENED")
ck("hard is strictly tighter in at least four specs",
   sum(sp.CONSTRAINTS_HARD[k] != sp.CONSTRAINTS[k] for k in _MINS + _MAXS) >= 4,
   f"{sum(sp.CONSTRAINTS_HARD[k] != sp.CONSTRAINTS[k] for k in _MINS + _MAXS)}/7 moved")
ck("VREF window brackets the nominal in both cases",
   sp.CONSTRAINTS["VREF_min"] < 1.219903 < sp.CONSTRAINTS["VREF_max"]
   and sp.CONSTRAINTS_HARD["VREF_min"] < 1.219903 < sp.CONSTRAINTS_HARD["VREF_max"])

# ---------------------------------------------------------------- 3. objective identity
print("\n3. the objective is bit-identical to the surrogate's, so section 6 optimizes the same "
      "thing\n   as section 5")
# ---------------------------------------------------------------------------------------------
# An uncalibrated case must REFUSE to build for a campaign. With no thresholds every candidate is
# feasible, so the campaign completes and returns a full six-way ranking that means nothing -- and
# looks exactly like a result. This is the single worst failure available in this harness, so it is
# checked rather than trusted.
for _c in PROBE_ONLY:
    try:
        sp.SpiceBGRProblem(sky130_lib="/fake/l.spice", case=_c)
        ck(f"uncalibrated case {_c!r} refuses to build for a campaign", False,
           "it built without thresholds")
    except ValueError as _e:
        ck(f"uncalibrated case {_c!r} refuses to build for a campaign",
           "no frozen thresholds" in str(_e), str(_e)[:60])
if not PROBE_ONLY:
    print("   (no probe-only cases registered)")

sur = P.SBGRProblem(dim=12, seed=1, case="base")
ck("lambda == 1000 in both", sp.LAMBDA == 1000.0)
ck("six specifications", sp.N_SPECS == 6)
_nm = probs["base"]._metrics_from(dict(NOMINAL_METRICS), {})
ck("the objective is the BAND WORST CASE, not the 100 Hz value",
   _nm["PSRR_DB"] == _nm["PSRR_WC_DB"] and _nm["PSRR_DB"] != _nm["PSRR_100HZ_DB"],
   f"objective={_nm['PSRR_DB']}  100Hz={_nm['PSRR_100HZ_DB']}")
ck("every row self-labels its objective definition",
   _nm["psrr_def"] == sp.OBJECTIVE and "worst_case" in sp.OBJECTIVE, sp.OBJECTIVE)
ck("the single-frequency value is still recorded as a diagnostic",
   _nm["PSRR_100HZ_DB"] == 56.29)
ck("the random-search baseline was re-derived under the CURRENT objective",
   sp.PROBE_BEST_FEASIBLE_PSRR_DB is not None
   and sp.PROBE_BEST_FEASIBLE_PSRR_DB != sp.PROBE_BEST_FEASIBLE_PSRR_DB_SINGLE_FREQ_LEGACY,
   f"{sp.PROBE_BEST_FEASIBLE_PSRR_DB} dB, not the superseded "
   f"{sp.PROBE_BEST_FEASIBLE_PSRR_DB_SINGLE_FREQ_LEGACY} dB")
ck("the nominal design beats random search on this objective",
   sp.NOMINAL_OBJECTIVE_DB > sp.PROBE_BEST_FEASIBLE_PSRR_DB,
   f"nominal {sp.NOMINAL_OBJECTIVE_DB} dB vs random {sp.PROBE_BEST_FEASIBLE_PSRR_DB} dB -- "
   f"so the reference design, not random sampling, is the bar that matters")
ck("the nominal objective matches the measured nominal point",
   abs(sp.NOMINAL_OBJECTIVE_DB - NOMINAL_METRICS["psrrwc_db"]) < 0.01)
ck("the objective change did not touch the thresholds",
   probs["base"].constraints == sp.CONSTRAINTS
   and probs["hard"].constraints == sp.CONSTRAINTS_HARD,
   "the pre-registered freeze survives, because the objective is not a threshold")
ck("cache is sized for a 150k-evaluation budget",
   probs["base"]._cache_size <= 50_000,
   f"{probs['base']._cache_size} entries x 4072 B = "
   f"{probs['base']._cache_size*4072/2**20:.0f} MB per worker")
for psrr, pen in ((56.29, 0), (10.0, 3), (-200.0, 6)):
    m = {"PSRR_DB": psrr, "penalty": pen}
    f_spice = probs["base"].fitness_from_metrics(m)
    ck(f"fitness(-PSRR={psrr}, pen={pen}) == {-psrr + 1000 * pen}",
       abs(f_spice - (-psrr + 1000 * pen)) < 1e-9, f"{f_spice}")
xs = sur.lb + 0.37 * (sur.ub - sur.lb)
f_sur, m_sur = sur.evaluate_with_metrics(xs)
ck("surrogate uses the same closed form",
   abs(f_sur - (-m_sur["PSRR_DB"] + 1000 * m_sur["penalty"])) < 1e-9)
ck("surrogate penalty is a 0..6 count", 0 <= m_sur["penalty"] <= 6)
ck("metric key sets match",
   {"PSRR_DB", "penalty", "is_feasible", "VREF", "TC", "LOOP_GAIN_DB",
    "PHASE_MARGIN_DEG", "GAIN_MARGIN_DB", "POWER_UW", "viol_vref", "viol_tc",
    "viol_loop_gain", "viol_phase_margin", "viol_gain_margin", "viol_power"}
   <= set(m_sur) & set(probs["base"]._metrics_from(dict(NOMINAL_METRICS), {})),
   "run.py's row schema and its statistics helpers both key on these")

# ---------------------------------------------------------------- 4. fail-soft
print("\n4. fail-soft: a simulation failure must never win, and must never be non-finite\n"
      "   (np.argmin([1.0, nan, 2.0]) == 1, so one nan destroys a run in silence)")
fm = probs["base"]._failed_metrics(probs["base"]._nominal, nb.DECK_ERROR, "x" * 500)
ff = probs["base"].fitness_from_metrics(fm)
ck("failed fitness is finite", np.isfinite(ff), f"{ff}")
ck("failed fitness == 6200", abs(ff - 6200.0) < 1e-9)
ck("failed loses to the worst plausible real candidate", ff > 1000 * 6 - 200)
ck("failed PSRR is the floor, not nan", fm["PSRR_DB"] == sp.PSRR_FLOOR_DB)
ck("failed sets all six violation flags",
   all(fm[k] == 1 for k in ("viol_vref", "viol_tc", "viol_loop_gain",
                            "viol_phase_margin", "viol_gain_margin", "viol_power")))
ck("failure detail is truncated", len(fm["sim_detail"]) <= 120)
ck("np.argmin cannot prefer a failure",
   int(np.argmin([-56.29, ff, 3000.0])) == 0)

# ---------------------------------------------------------------- 5. design space
print("\n5. design space: clipping, quantisation, and the bounds the optimizer will actually "
      "see")
for case in ("base", "highdim"):
    p = probs[case]
    ck(f"{case}: lb < ub everywhere", bool(np.all(p.lb < p.ub)))
    xnom = np.array([p._nominal[v[0]] for v in p._vars], dtype=float)
    ck(f"{case}: nominal is inside the box",
       bool(np.all(p.lb <= xnom) and np.all(xnom <= p.ub)))
    # Clipping is the published bound-handling protocol and applies to every optimized variable.
    # In highdim it is followed by a second, narrower correction: the deck sizes matched partners
    # as primary*ratio, and clipping alone does not keep those products inside the model's
    # validity range -- l=0.15 with r=0.5 gives 0.075 um against a 0.15 um minimum. The ratio is
    # therefore lifted just enough to make the product legal, so a ratio need NOT sit at its own
    # bound after clipping. The primaries still must.
    ratios = {v[0] for v in p._vars if v[0].startswith("r_")}
    for tag, x_out, want_idx in (("ub", p.ub + 1e6, 2), ("lb", p.lb - 1e6, 1)):
        q = p.quantise(x_out)
        prim_ok = all(abs(q[v[0]] - v[want_idx]) < 1e-9
                      for v in p._vars if v[0] not in ratios)
        ck(f"{case}: out-of-box primaries are clipped to {tag}", prim_ok,
           "clipping is the published bound-handling protocol")
        ck(f"{case}: every derived geometry legal after clipping to {tag}",
           all(lo <= q[a] * q[r] <= hi for a, r, (lo, hi) in p.DERIVED_HIGHDIM
               if a in q and r in q),
           "an out-of-range product is a deck error and trips the circuit breaker")
        ck(f"{case}: ratios stay within their own declared bounds",
           all(v[1] - 1e-9 <= q[v[0]] <= v[2] + 1e-9 for v in p._vars if v[0] in ratios),
           "the clamp may lift a ratio but must never push it outside [0.5, 2]")
    ck(f"{case}: lb and ub are exact grid points",
       all(v[4] == 0 or abs(((v[2] - v[1]) / v[4]) - round((v[2] - v[1]) / v[4])) < 1e-6
           for v in p._vars),
       "otherwise ub is unreachable and the box is silently smaller than stated")
    ck(f"{case}: quantise is idempotent",
       p.quantise([q[v[0]] for v in p._vars]) == q)
    ck(f"{case}: w_pass ceiling is the model limit",
       dict((v[0], v[2]) for v in p._vars).get("w_pass") == 100.0,
       "one out-of-range w_pass poisoned every later candidate in a session")

# highdim reduces to base at every ratio 1
hd = probs["highdim"]
qh = hd.quantise([hd._nominal[v[0]] for v in hd._vars])
ck("highdim nominal has every ratio == 1",
   all(abs(qh[k] - 1.0) < 1e-12 for k in qh if k.startswith("r_")),
   "that is what makes it provably the same circuit as base")
ck("highdim nominal shares base's shared values",
   all(abs(qh[k] - sp.NOMINAL[k]) < 1e-12 for k in ("w_pdiff", "w_ncasc", "w_pmir",
                                                    "w_pass", "iss", "vbnv")))

# ---------------------------------------------------------------- 6. cache key
print("\n6. the result cache cannot alias two distinct designs (the defect that returned "
      "another\n   candidate's metrics with ok=True)")
p = probs["base"]
keys = set()
for mul in (1.0, 1.0 + 1e-4, 1.0 + 1e-3, 2.0, 3.0, 7.0):
    x = [p._nominal[v[0]] * mul for v in p._vars]
    q = p.quantise(x)
    keys.add(tuple(sorted((k, float(f"{v:.9g}")) for k, v in q.items())))
ck("6 distinct scalings -> 6 distinct keys", len(keys) == 6, f"got {len(keys)}")

# ---------------------------------------------------------------- 7. driver plumbing
print("\n7. driver: --pop and --case reach the row, and the resume guard refuses to mix "
      "configurations")
import run_spice as RS
for _case in CAMPAIGN_CASES:
    _f = RS.row_fields(_case)
    _p = probs[_case]
    _m = _p._metrics_from(dict(NOMINAL_METRICS), _p._nominal)
    _xk = {f"x_{k[2:]}" for k in _m if k.startswith("x_")}
    _ck = {k for k in _m if k.startswith("c_")}
    ck(f"{_case}: schema carries case/pop/objective label",
       all(c in _f for c in ("case", "pop", "psrr_def")))
    ck(f"{_case}: schema has no duplicates", len(_f) == len(set(_f)))
    # The schema is built PER CASE because highdim has 16 differently-named variables. A fixed
    # list of base's 7 names would have made csv.DictWriter(extrasaction="ignore") drop every
    # highdim design vector silently -- the campaign would finish and the designs would be gone.
    ck(f"{_case}: schema carries all {_p.dim} design variables",
       _xk <= set(_f) and len(_xk) == _p.dim,
       f"{len(_xk)} of {_p.dim}; without these the best circuit cannot be reported")
    ck(f"{_case}: schema carries the thresholds in force", _ck <= set(_f), f"{len(_ck)}")
    ck(f"{_case}: cumulative counters are named as cumulative",
       all(c in _f for c in ("sim_evals_worker_cum", "server_restarts_worker_cum"))
       and "sim_ok" not in _f,
       "they are per-worker histories, not per-run values")
ck("base and highdim schemas actually differ",
   set(RS.row_fields("base")) != set(RS.row_fields("highdim")),
   f"{len(RS.row_fields('base'))} vs {len(RS.row_fields('highdim'))} fields")

print("\n7b. PVT: an off-nominal corner or supply can never leak into an optimization run")
for _kw in (dict(corner="ss"), dict(vdd=1.62), dict(corner="ff", vdd=1.98)):
    try:
        sp.SpiceBGRProblem(sky130_lib="/fake/l.spice", case="base", **_kw)
        ck(f"refuses {_kw}", False, "IT WAS ACCEPTED")
    except ValueError:
        ck(f"refuses {_kw} without allow_off_nominal", True)
_q = sp.SpiceBGRProblem(sky130_lib="/fake/l.spice", case="base", corner="ss", vdd=1.62,
                        allow_off_nominal=True)
ck("the corner reaches the deck when explicitly allowed",
   ".lib /fake/l.spice ss" in _q._srv.deck and "DC 1.62 AC 1" in _q._srv.deck)
ck("the optimization deck is tt at 1.8 V",
   ".lib /fake/sky130.lib.spice tt" in probs["base"]._srv.deck
   and "DC 1.8 AC 1" in probs["base"]._srv.deck,
   "the published problem's single-corner nominal setting")
ck("the PVT grid is 5 corners x 3 supplies",
   len(sp.PVT_CORNERS) * len(sp.PVT_SUPPLIES) == 15,
   f"{sp.PVT_CORNERS} x {sp.PVT_SUPPLIES}")
try:
    sp.SpiceBGRProblem(sky130_lib="/fake/l.spice", case="base", corner="zz",
                       allow_off_nominal=True)
    ck("an unknown corner is refused", False, "IT WAS ACCEPTED")
except ValueError:
    ck("an unknown corner is refused", True)

tmp = Path(tempfile.mkdtemp())
fake = tmp / "per_run_records.csv"
pd.DataFrame([{"case": "base", "pop": 20, "algo": "ABC", "run": 0,
               "best_fitness": -1.0}]).to_csv(fake, index=False)


def try_main(argv):
    old = sys.argv
    sys.argv = ["run_spice.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            RS.main()
        return 0, buf.getvalue()
    except SystemExit as e:
        return int(e.code or 0), buf.getvalue()
    except Exception as e:  # noqa: BLE001
        return -1, buf.getvalue() + f"\n{type(e).__name__}: {e}"
    finally:
        sys.argv = old


rc, out = try_main(["--lib", "/fake/sky130.lib.spice", "--outdir", str(tmp),
                    "--case", "hard", "--pop", "40"])
ck("case mismatch is refused", rc == 2 and "already contains case" in out,
   f"rc={rc}")
rc, out = try_main(["--lib", "/fake/sky130.lib.spice", "--outdir", str(tmp),
                    "--case", "base", "--pop", "40"])
ck("pop mismatch is refused", rc == 2 and "already contains pop" in out, f"rc={rc}")
rc, out = try_main(["--lib", "/fake/sky130.lib.spice", "--outdir", str(tmp),
                   "--case", "base", "--pop", "20", "--algos", "ABC", "--runs", "1"])
ck("the matching configuration is accepted and resumes",
   "1 (algo, run) pairs already complete" in out, f"rc={rc}")

print(f"\n{'=' * 70}\n{checks - len(fails)}/{checks} checks passed")
if fails:
    print("FAILED: " + "; ".join(fails))
sys.exit(1 if fails else 0)

