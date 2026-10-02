"""
SpiceOpampProblem -- Circuit B of the case study: a SKY130 two-stage Miller-compensated operational
amplifier, evaluated at transistor level by ngspice, duck-typing the same interface the algorithms
already consume so that algorithms.py needs no change at all.

RELATIONSHIP TO spice_problem.py. This SUBCLASSES SpiceBGRProblem and overrides only what is
circuit-specific: the design space, the specifications, the ngspice command sequence and the mapping
from raw measurements to metrics. Everything else -- the result cache, the deck-error circuit
breaker, the box clipping and grid snapping, and above all the timeout recovery that resynchronises
the ngspice pipe -- is INHERITED, unmodified.

That choice is deliberate on two counts. First, spice_problem.py produced the published Circuit A
results and is referenced by the Zenodo record, so it is not edited here; the subclass adds a circuit
without touching the file that ran the first campaign. Second, the timeout-recovery path is the part
of this harness that was hardest to get right -- a timeout does not kill ngspice, so the abandoned
command's output arrives late and is read as the answer to the next one, and from then on every
result in that worker is silently wrong. Re-implementing that for a second circuit would be the
single most likely place to reintroduce it. Inheriting it means there is one copy to be right.

    p = SpiceOpampProblem(sky130_lib=<path to sky130.lib.spice>)
    f, m = p.evaluate_with_metrics(x)          # x is a length-7 vector in the box
    p.close()

Not thread-safe: one instance, and therefore one ngspice process, per worker.

OBJECTIVE. DC open-loop differential gain in dB, MAXIMIZED. Reported as a fitness to be minimised,

    f(x) = -Gain_dB + 1000 * penalty

with penalty the COUNT of violated specifications -- identical in form to Circuit A, so the two
circuits are scored the same way and nothing about the comparison depends on the scoring.

SIX SPECIFICATIONS, the same count as Circuit A's, one of them a two-sided window:

    UGB   >= .        unity-gain bandwidth
    PM    >= .        phase margin
    Power <= .        supply power at the operating point
    Area  <= .        sum of W*L over the eight transistors
    PSRR  >= .        low-frequency supply rejection
    Vout in [., .]    quiescent output level: a two-sided window, the counterpart of Circuit A's
                      V_REF window, and a functional requirement rather than a performance target --
                      an amplifier parked at a rail has no swing whatever its gain

PSRR appears here as a CONSTRAINT where it was Circuit A's OBJECTIVE. That is the point of the pair:
the same physical quantity carrying a different role tests whether an optimizer ranking transfers,
rather than repeating one problem twice.

WHY AN NMOS INPUT PAIR. Measured on this PDK at L = 1 um, tt, 27 C: |Vth| = 1.033 V for pfet_01v8
against 0.615 V for nfet_01v8. A PMOS pair therefore needs an input common mode below
Vdd - |Vth| - Vdsat = 0.57 V, so at the mid-rail 0.9 V a PMOS pair sits below threshold, the tail
device collapses into its linear region and the amplifier does not amplify -- measured, on the first
version of this deck: 0.157 uA in the input pair and 2.2 mV across the tail. Recorded because it is
a property of this process, not a preference.
"""
from __future__ import annotations

import numpy as np

import ngspice_bridge as nb
from spice_problem import LAMBDA, Counters, SpiceBGRProblem, _grab

TEMPLATE = "opamp_sky130.cir.tmpl"
N_SPECS = 6

# ---------------------------------------------------------------- design space
# name, lower, upper, unit, quantisation step (0 = continuous).
# SEVEN variables, matching the dimension of Circuit A's base case, so the two rankings are compared
# at equal D and dimensionality is not a confound. Bounds follow Circuit A's wherever the quantity is
# the same kind: 1-40 um for the moderate widths, 100 um for the wide devices because that is the
# model's W ceiling, and 0.15 um as the process minimum length.
# WHY THE STAGE-2 WIDTHS ARE DERIVED RATIOS AND NOT FREE ABSOLUTE SIZES. Measured, on a 512-point
# probe of the free-width box: only 3.3 % of points had a quiescent output anywhere off a rail. The
# cause is structural rather than incidental -- the output centres only when the stage-2 driver and
# sink currents match, i.e. when (w6/w3)*(w5/2*w7) ~ 1, and a product of four independent uniforms
# lands near 1 rarely. The consequence was a benchmark in which the output window alone decided
# feasibility (3.3 % marginal) while the other five specifications were free (>90 % each): a
# two-constraint problem wearing six labels.
#
# Circuit A met exactly this on its highdim deck and recorded the same fix: with fully independent
# per-device sizes, "84 of 96 random points had no unity-gain crossing at all -- 94 % of the space
# unmeasurable, 0 % feasible", so the MISMATCH was bounded as a ratio instead. Its stated reason
# applies verbatim here: matched pairs and mirror ratios have tolerances, not arbitrary sizes.
#
#     m2 = w6/w3   the stage-2 mirror ratio            ->  w6 = m2 * w3
#     k            deviation from zero-offset balance   ->  w7 = k * w5 * m2 / 2
#
# so the balance ratio is exactly 1/k and is bounded by construction. Measured effect on a 256-point
# probe: 100 % of points measurable (from 95 %), output window 11.7 % (from 3.3 %), and no derived
# width clipped at the model's limits in any of the 256.
#
# Bounds are set so the DERIVED widths stay inside the model range without clipping doing the work:
# w6 = m2*w3 <= 5*20 = 100 um, exactly the model ceiling.
VARS = (
    ("w1",    1.0,  40.0, "um", 0.005),   # input pair M1/M2
    ("w3",    1.0,  20.0, "um", 0.005),   # PMOS mirror load M3/M4
    ("w5",    1.0,  20.0, "um", 0.005),   # NMOS tail sink and bias mirror Mb/M5
    ("m2",    1.0,   5.0, "-",  0.0),     # stage-2 mirror ratio; w6 = m2 * w3
    ("k",     0.5,   2.0, "-",  0.0),     # balance deviation; w7 = k * w5 * m2 / 2
    ("l_all", 0.15,  2.0, "um", 0.005),   # common channel length
    ("cc",    0.1,  10.0, "pF", 0.0),     # Miller compensation capacitance
)
VAR_NAMES = tuple(v[0] for v in VARS)

# l_all is capped at 2 um and not Circuit A's 8 um. Measured: area is dominated by l_all -- median
# 68 um^2 for L < 0.5 against 1297 um^2 for L > 4 -- so with 8 um in the box no area threshold that
# admits the reference design can also exclude anything. Channel lengths beyond ~2 um are not an
# analog design choice at this node in any case; they only add area and cost bandwidth. Circuit A
# could afford the wider range because area was not one of its specifications.

# The reference design, chosen by a criterion declared BEFORE the search that found it: minimise
# supply power subject to gain >= 60 dB, UGB >= 5 MHz, phase margin >= 60 deg and a comfortably
# centred output, Vout in [0.6, 1.2] V. One of 720 grid candidates met it.
#
# It replaces a hand-derived point that burned 66.9 uW -- roughly the 75th percentile of the box.
# Because gate 1 forbids a threshold that excludes the reference design, that anchor made Power_max
# untightenable below 66.9 uW, so 81 % of the box passed it and Power stopped being a specification
# at all. A power-hungry anchor is a defect in the setup, not a cosmetic matter. Re-anchoring fixes
# only where the anchor sits: every algorithm faces identical thresholds whatever the anchor, so this
# changes difficulty and cannot change fairness.
NOMINAL = dict(w1=10.0, w3=10.0, w5=4.0, m2=3.0, k=0.85, l_all=1.0, cc=3.0)

# The same point expressed in the DECK's parameters. Needed because the deck declares w6 and w7,
# not m2 and k, so the initial render must use the derived widths; every candidate thereafter is
# applied with alterparam. Kept as an explicit constant rather than computed at import, so that the
# rendered default is a stated number that can be checked against the measured reference metrics.
NOMINAL_DECK = dict(w1=10.0, w3=10.0, w5=4.0, w6=30.0, w7=5.1, l_all=1.0, cc=3.0)

# Measured at tt, 27 C, 1.8 V, from the derived widths w6 = 30.0 and w7 = 5.1 um.
REFERENCE_METRICS = dict(Gain=83.4923, UGB=5.5498, PhaseMargin=62.4100, Power=58.5027,
                         Vout=1.1066, Area=83.1000, PSRR=89.9284)

# ---------------------------------------------------------------- specifications
# ===================== FROZEN 2026-08-26 -- DO NOT RE-TUNE AFTER ANY CAMPAIGN =====================
# Calibrated against a 512-point Latin-hypercube probe of the design box (scipy.stats.qmc, seed
# 20260826 -- the pre-registration date, distinct from Circuit A's 20260804 so the two probes are
# independent samples rather than the same points re-scored). All 512 simulated successfully. The
# probe was run THROUGH this class, so the quantiser, the derived-geometry clamp, the result cache,
# the deck-error breaker and the ngspice timeout recovery were all exercised, not assumed.
#
# THE CALIBRATION RULE, and why it is not the wording in the pre-registration.
#
# The pre-registration says "uniform relative slack, adjusted only so that random joint feasibility
# falls between 0.5 % and 5 %". Applied to this circuit that rule is DEGENERATE: the scalar must
# reach t = 1.04 before joint feasibility enters the band, and at that value UGB_min = -0.31 MHz,
# PhaseMargin_min = -2.44 deg and PSRR_min = -3.66 dB. Four of the six specifications become
# thresholds no design can fail, and the benchmark is really two constraints wearing six labels.
#
# It is also not what Circuit A did. Its frozen slacks are 0.12 %, 6.2 %, 16.4 %, 6.0 %, 4.8 % and
# 5.8 % -- per specification, not uniform. The pre-registration mis-describes its own precedent, and
# that wording is precisely what produced the degenerate result.
#
# The rule used instead has exactly ONE degree of freedom:
#
#   declare one target marginal pass rate T; for each scaled specification set the threshold to the
#   empirical quantile of the probe that lets T % of the box pass, clamped so the reference design
#   still passes; then require joint feasibility in [0.5 %, 5 %]. T is the only knob and it moves all
#   six thresholds together.
#
# This is MORE constraining than uniform slack, not less: balance becomes a property of the
# construction rather than something hoped for and checked afterwards. And it cannot favour any
# optimizer, because all six algorithms face identical thresholds -- threshold choice sets the
# problem's difficulty and never its fairness. Selection among candidate T values was declared before
# the numbers were seen: prefer a T whose marginal rates are all inside [10 %, 60 %], and among those
# take the one whose joint feasibility is nearest the geometric centre of the band. T = 60 was the
# only value both in band and balanced.
#
# ALL FOUR ACCEPTANCE GATES PASS, and all were checked BEFORE any optimization run existed:
#   1. the reference design is feasible (penalty 0), so the feasible set is non-empty by construction
#      and the benchmark does not exclude its own anchor
#   2. joint feasibility of random in-box candidates = 0.586 % (3 of 512), inside [0.5 %, 5 %]:
#      not trivially reachable, not unreachable
#   3. every specification's marginal pass rate is inside [10 %, 60 %] -- five at 60.0 % by
#      construction and the output window at 12.7 % -- so no specification is either free or the sole
#      binding constraint. This is the gate the first two attempts at this problem failed, and the
#      gate that forced both the bounded-ratio parameterisation and the re-anchored reference design.
#   4. the best objective random sampling reaches while feasible is 81.18 dB, against the reference
#      design's 83.49 dB and a box-wide best of 84.98 dB. So beating random search is a weak claim
#      here and beating the reference design is the meaningful one -- the same structure as Circuit A,
#      where the two bars were 15.50 dB and 23.85 dB.
#
# HONEST NOTE ON GATE 4. Only 3 of 512 probe points are feasible, so 81.18 dB is an estimate from
# three samples and should be read as such. Circuit A had 11 of 512 for the same purpose.
CONSTRAINTS = {
    "UGB_min": 1.8602,          # MHz    reference 5.5498    slack 66.48 %
    "PhaseMargin_min": 56.7470, # deg    reference 62.4100   slack  9.07 %
    "PSRR_min": 82.1639,        # dB     reference 89.9284   slack  8.63 %
    "Power_max": 60.0640,       # uW     reference 58.5027   slack  2.67 %
    "Area_max": 148.1022,       # um^2   reference 83.1000   slack 78.22 %
    # The output window is NOT scaled by T. An amplifier parked at a rail has no swing whatever its
    # gain, so this is a functional requirement set from the rail rather than a performance target --
    # the counterpart of Circuit A's V_REF window. It lands at 12.7 % and is reported at that.
    "Vout_min": 0.35,           # V
    "Vout_max": 1.45,           # V
}
CALIBRATED = True

# The bars a campaign has to clear, both reported, and the second is the headline -- as in Circuit A.
PROBE_BEST_FEASIBLE_GAIN_DB = 81.18     # best of 3 feasible points in a 512-point random probe
NOMINAL_OBJECTIVE_DB = 83.4923          # the reference design itself

# The band the objective is clamped to, mirroring Circuit A's PSRR floor and ceiling. A design that
# does not simulate is scored at the floor rather than dropped, because the unmeasurable corners of
# the box are the violent ones and discarding them would flatter every threshold.
GAIN_FLOOR_DB = -200.0
GAIN_CEIL_DB = 200.0

CASES = {"base": (TEMPLATE, VARS, NOMINAL, CONSTRAINTS)}

# NOMINAL and NOMINAL_DECK must describe the same design. Checked at import, because a drift between
# them would change the deck's rendered defaults without changing anything the optimizer sees.
assert abs(NOMINAL_DECK["w6"] - NOMINAL["m2"] * NOMINAL["w3"]) < 1e-9, "NOMINAL_DECK w6 drifted"
assert abs(NOMINAL_DECK["w7"] - NOMINAL["k"] * NOMINAL["w5"] * NOMINAL["m2"] / 2.0) < 1e-9,     "NOMINAL_DECK w7 drifted"
CORNERS = ("tt", "ss", "ff", "sf", "fs")

# ---------------------------------------------------------------- ngspice command sequence
# The ordering here is not stylistic. Every line of it was forced by measurement, on this deck or on
# Circuit A's:
#   * `destroy all` FIRST, and it is NOT optional -- this was learned the expensive way. It was tried,
#     then REMOVED, on the strength of a 150-candidate test that showed resident memory flat at 50 MB.
#     A campaign run is 2,500 evaluations: the test was sixteen times too short to see the
#     accumulation. On TRUBA, three of six op-amp shards were OOM-killed
#     ("Detected 8 oom_kill events"), while zero LDO shards were -- and the LDO evaluator had kept
#     its `destroy all`. Two AC plots per candidate over 2,500 candidates across 30 concurrent
#     ngspice processes is enough to exhaust 32 GB.
#
#     The rows survived only because each run is appended and fsync'd as it finishes; under the
#     earlier write-at-the-end design all 150 would have been lost. Two independent mistakes, one of
#     which happened to cover for the other.
#   * the differential and supply stimuli are NEVER armed together -- each is zeroed before the other
#     is swept, or the two gains superpose and both are wrong.
#   * `meas ac ... FIND v(x)` returns the REAL PART of a complex vector, so magnitudes are built with
#     `let vo = mag(...)` first.
#   * `vdb()` inside `meas` aborts the whole AC analysis in ngspice-46. Not used.
#   * one `print` per value: an unsupported probe then aborts only its own line.
_ARM_DIFF = ("alter @Vdd[acmag] = 0", "alter @Vip[acmag] = 0.5", "alter @Vin[acmag] = -0.5")
_ARM_SUPPLY = ("alter @Vip[acmag] = 0", "alter @Vin[acmag] = 0", "alter @Vdd[acmag] = 1")
_SWEEP_AC = "ac dec 20 1 1e9"


class SpiceOpampProblem(SpiceBGRProblem):
    """Circuit B. See the module docstring for what is overridden and what is inherited."""

    # Circuit A's highdim deck derives matched partners as primary*ratio and clamps those products
    # via this table. This deck's derivation is a different formula, so the table is emptied and
    # quantise() is overridden below instead.
    DERIVED_HIGHDIM = ()

    # SKY130 01v8 FET validity range, micrometres. The DERIVED widths must respect it exactly as the
    # optimized variables do.
    MODEL_W = (0.42, 100.0)
    MODEL_L = (0.15, 16.0)

    # The .param names the deck actually declares. quantise() must produce exactly these and nothing
    # else, because `alterparam` on a name the deck does not declare is accepted SILENTLY and does
    # nothing -- so a typo or a leftover ratio name would leave that geometry pinned at its rendered
    # default for every candidate, with no error anywhere and every result quietly wrong.
    DECK_PARAMS = frozenset(("w1", "w3", "w5", "w6", "w7", "l_all", "cc"))

    def __init__(self, sky130_lib: str, exe: str | None = None, case: str = "base",
                 constraints: dict | None = None, timeout: float = 30.0,
                 cache_size: int = 20_000, deck_error_budget: float = 0.02,
                 breaker_after: int = 200,
                 corner: str = "tt", vdd: float = 1.8, allow_off_nominal: bool = False,
                 require_calibrated: bool = True):
        if case not in CASES:
            raise ValueError(f"unknown case {case!r}; available: {sorted(CASES)}")
        template, self._vars, self._nominal, default_constraints = CASES[case]
        self._constraints = dict(constraints if constraints is not None else default_constraints)
        if require_calibrated and not self._constraints:
            raise ValueError(
                "Circuit B's thresholds are not calibrated yet. Run the 512-point Latin-hypercube "
                "probe and freeze CONSTRAINTS per PREREGISTRATION_2026-08-26 section 4, or pass "
                "require_calibrated=False for a diagnostic run that is not a campaign.")
        # An OPTIMIZATION run must be at the nominal corner and supply, because that is the problem
        # the thresholds were calibrated at. The PVT pass sets allow_off_nominal=True deliberately.
        if (corner != "tt" or vdd != 1.8) and not allow_off_nominal:
            raise ValueError(
                f"corner={corner!r}, vdd={vdd!r} is off-nominal. Optimization runs at tt/1.8 V "
                f"only; pass allow_off_nominal=True if this is the PVT verification pass.")
        if corner not in CORNERS:
            raise ValueError(f"unknown corner {corner!r}; known: {sorted(CORNERS)}")

        self.corner, self.vdd = corner, float(vdd)
        self.case = case
        self.name = "SKY130-OPAMP" if case == "base" else f"SKY130-OPAMP-{case}"
        self.dim = len(self._vars)
        self.lb = np.array([v[1] for v in self._vars], dtype=float)
        self.ub = np.array([v[2] for v in self._vars], dtype=float)
        self._step = np.array([v[4] for v in self._vars], dtype=float)
        self.counters = Counters()
        self._cache: dict = {}
        self._cache_size = cache_size
        self._deck_error_budget = deck_error_budget
        self._breaker_after = breaker_after
        self._consecutive_launch_errors = 0

        self._exe = exe or nb.find_ngspice()
        lib = str(sky130_lib).replace("\\", "/")
        if " " in lib:
            raise ValueError(f"the .lib path must not contain spaces (SPICE tokenises on "
                             f"whitespace): {lib!r}")
        # UPPERCASE placeholders, lowercase .param names. Getting that mapping wrong is SILENT:
        # ngspice accepts alterparam on an unknown name and does nothing, so every candidate would
        # return identical metrics with no error anywhere. Verified for this deck by sweeping w7 and
        # confirming the gain moves: 28.93 dB at 5 um, 82.92 at 17, 36.67 at 40.
        subs = {"SKY130_LIB": lib, "CORNER": corner, "VDD_DC": repr(float(vdd))}
        # The DECK's parameters, not the optimizer's: the deck declares w6 and w7 while the design
        # vector carries m2 and k. Rendering from the optimizer's nominal leaves ${W6} and ${W7}
        # unsubstituted -- caught by the check below, which is why it is there.
        subs.update({k.upper(): v for k, v in NOMINAL_DECK.items()})
        deck = nb.render_netlist(template, subs)
        if "${" in deck:
            raise ValueError(f"template placeholder left unsubstituted in {template}")
        self._srv = nb.NgspiceServer(deck, exe=self._exe, eval_timeout=timeout,
                                     load_timeout=400.0)
        self._srv.load()

    # ------------------------------------------------------------ design vector -> deck parameters
    def quantise(self, x) -> dict:
        """Clip into the box, snap to the device grid, then DERIVE the two stage-2 widths.

        Returns the deck's parameters, not the optimizer's variables: m2 and k are replaced by w6 and
        w7. The optimizer's own vector is never written back -- that would be repair, which changes
        the search dynamics -- so the returned dict is a view of what ngspice was given, and the
        recorded design vector stays self-consistent with it.

        The derived widths are clamped into the model's range, as Circuit A clamps its own derived
        geometry, and for the same reason: a legal m2 and a legal w3 can still multiply to an illegal
        width, and ngspice reports that as a deck error, which trips the circuit breaker and kills a
        run that was otherwise healthy. The bounds are chosen so the clamp is a guard rather than a
        participant -- it did not engage on any of 256 probe points.
        """
        v = np.minimum(np.maximum(np.asarray(x, dtype=float), self.lb), self.ub)
        opt = {}
        for i, (name, lo, hi, _u, step) in enumerate(self._vars):
            val = float(v[i])
            if step > 0:
                # step from lb so that lb and ub are themselves exact grid points
                val = lo + round((val - lo) / step) * step
                val = min(max(val, lo), hi)
            opt[name] = val

        w6 = min(max(opt["m2"] * opt["w3"], self.MODEL_W[0]), self.MODEL_W[1])
        w7 = min(max(opt["k"] * opt["w5"] * opt["m2"] / 2.0, self.MODEL_W[0]), self.MODEL_W[1])
        out = {"w1": opt["w1"], "w3": opt["w3"], "w5": opt["w5"],
               "w6": round(w6, 6), "w7": round(w7, 6),
               "l_all": min(max(opt["l_all"], self.MODEL_L[0]), self.MODEL_L[1]),
               "cc": opt["cc"]}
        # The silent-failure guard. Cheap, and the failure it catches is invisible by construction.
        assert set(out) == self.DECK_PARAMS, f"deck parameter mismatch: {sorted(set(out))}"
        return out

    # ------------------------------------------------------------ objective
    def fitness_from_metrics(self, m: dict) -> float:
        return float(-m["GAIN_DB"] + LAMBDA * m["penalty"])

    # ------------------------------------------------------------ simulation
    def _simulate(self, params: dict):
        """One candidate in a single exchange: differential gain, bandwidth and phase, then supply
        rejection, then the operating point. Returns (values, failure_kind, detail)."""
        # `:.9g` and NOT `{v!r}`. Under NumPy 2.x, repr() of a numpy scalar is `np.float64(5.72916)`,
        # which is not a number SPICE can parse: the `.param` silently fails, the expression that uses
        # it reports `Undefined parameter`, and ngspice exits(1). In server mode that means the process
        # DIES mid-exchange, so the end marker never arrives and the harness records a TIMEOUT --
        # a fatal netlist error disguised as a slow simulation. Cost of learning that: several hours
        # chasing pipe desynchronisation, plot accumulation and machine memory pressure.
        #
        # The inherited quantise() already returns plain Python floats, so this is belt-and-braces
        # rather than a live fix; it is here because the failure is silent, expensive and one stray
        # numpy scalar away at all times.
        out = self._srv.send(
            ["destroy all"]
            + [f"alterparam {k}={float(v):.9g}" for k, v in params.items()]
            + ["reset", *_ARM_DIFF, _SWEEP_AC,
               "let vo = mag(v(out))",
               "let ph = 180/PI*ph(v(out))",
               "meas ac gain_lf FIND vo AT=100",
               "meas ac ph_lf   FIND ph AT=100",
               "meas ac ugb     WHEN vo=1 FALL=1",
               "meas ac ph_ugb  FIND ph WHEN vo=1 FALL=1",
               "let gain_db = 20*log10(gain_lf + 1e-30)",
               "print gain_db", "print ugb", "print ph_lf", "print ph_ugb",
               *_ARM_SUPPLY, _SWEEP_AC,
               "let vs = mag(v(out))",
               "meas ac supp_lf FIND vs AT=100",
               "let supp_db = 20*log10(supp_lf + 1e-30)",
               "print supp_db",
               "op", "print v(out)", "print i(vdd)"])
        if out is None:
            return None, nb.TIMEOUT, "no response during the ac/op phase"
        kind, detail = nb.classify_output(out, 0)
        if kind is not None:
            return None, kind, detail
        v = _grab(out, "gain_db", "ugb", "ph_lf", "ph_ugb", "supp_db", "v(out)", "i(vdd)")
        hard = [k for k in ("gain_db", "v(out)", "i(vdd)") if v[k] is None]
        if hard:
            return None, nb.MEASURE_FAILED, "no value for " + ", ".join(hard)
        # A missing crossing is a REAL and informative outcome, not a broken measurement: the design
        # never reaches unity gain, so it has no bandwidth and no phase margin to report. It is scored
        # as infeasible at the objective floor, like a design that violates every specification, but
        # it carries its own detail string so the campaign can report how much of the box behaves
        # this way -- as Circuit A reports its simulation failures rather than hiding them.
        soft = [k for k in ("ugb", "ph_lf", "ph_ugb") if v[k] is None]
        if soft:
            return None, nb.MEASURE_FAILED, "no unity-gain crossing (" + ", ".join(soft) + ")"
        return v, None, ""

    # ------------------------------------------------------------ metrics
    def _metrics_from(self, v: dict, params: dict) -> dict:
        c = self._constraints
        gain = float(min(max(float(v["gain_db"]), GAIN_FLOOR_DB), GAIN_CEIL_DB))
        ugb_mhz = float(v["ugb"]) / 1e6
        # Phase margin as 180 deg MINUS the phase accumulated between 100 Hz and the crossing. An
        # NMOS pair with a PMOS mirror load inverts, so the absolute phase at the crossing sits near
        # -180 and the usual "180 + phase" form reports nonsense: on the first draft of this deck it
        # returned 276-325 deg. Taking the accumulation makes the measure independent of that sign.
        pm = 180.0 - ((float(v["ph_lf"]) - float(v["ph_ugb"])) % 360.0)
        power_uw = abs(float(v["i(vdd)"])) * self.vdd * 1e6
        vout = float(v["v(out)"])
        psrr = gain - float(v["supp_db"]) if v.get("supp_db") is not None else GAIN_FLOOR_DB
        # Mb/M5 share w5, M1/M2 share w1, M3/M4 share w3; M6 and M7 are single devices.
        area = (2*params["w1"] + 2*params["w3"] + 2*params["w5"]
                + params["w6"] + params["w7"]) * params["l_all"]

        viol_ugb = int(ugb_mhz < c["UGB_min"])
        viol_pm = int(pm < c["PhaseMargin_min"])
        viol_pw = int(power_uw > c["Power_max"])
        viol_area = int(area > c["Area_max"])
        viol_psrr = int(psrr < c["PSRR_min"])
        viol_vout = int(vout < c["Vout_min"] or vout > c["Vout_max"])
        penalty = viol_ugb + viol_pm + viol_pw + viol_area + viol_psrr + viol_vout

        m = {
            "GAIN_DB": gain, "penalty": penalty, "is_feasible": int(penalty == 0),
            "UGB_MHZ": ugb_mhz, "PHASE_MARGIN_DEG": pm, "POWER_UW": power_uw,
            "VOUT_V": vout, "AREA_UM2": area, "PSRR_DB": psrr,
            "viol_ugb": viol_ugb, "viol_phase_margin": viol_pm, "viol_power": viol_pw,
            "viol_area": viol_area, "viol_psrr": viol_psrr, "viol_vout": viol_vout,
            "sim_ok": 1, "sim_failure": None,
        }
        m.update({"c_" + k: val for k, val in c.items()})
        m.update({"x_" + k: val for k, val in params.items()})
        return m

    def _failed_metrics(self, params: dict, failure: str, detail: str) -> dict:
        c = self._constraints
        area = (2*params["w1"] + 2*params["w3"] + 2*params["w5"]
                + params["w6"] + params["w7"]) * params["l_all"]
        m = {
            "GAIN_DB": GAIN_FLOOR_DB, "penalty": N_SPECS, "is_feasible": 0,
            "UGB_MHZ": float("nan"), "PHASE_MARGIN_DEG": float("nan"),
            "POWER_UW": float("nan"), "VOUT_V": float("nan"),
            "AREA_UM2": area,            # geometry is known even when the simulation is not
            "PSRR_DB": float("nan"),
            "viol_ugb": 1, "viol_phase_margin": 1, "viol_power": 1,
            "viol_area": 1, "viol_psrr": 1, "viol_vout": 1,
            "sim_ok": 0, "sim_failure": failure, "sim_detail": detail[:120],
        }
        m.update({"c_" + k: val for k, val in c.items()})
        m.update({"x_" + k: val for k, val in params.items()})
        return m
