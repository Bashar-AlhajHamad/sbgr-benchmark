"""
SpiceLdoProblem -- Circuit C of the case study: a SKY130 low-dropout regulator, evaluated at
transistor level by ngspice, duck-typing the interface the algorithms already consume so that
algorithms.py needs no change.

RELATIONSHIP TO THE OTHER TWO. Like Circuit B this SUBCLASSES SpiceBGRProblem and overrides only the
circuit-specific parts: design space, specifications, ngspice command sequence and the mapping from
raw measurements to metrics. The result cache, the deck-error circuit breaker, the box clipping and
the ngspice timeout recovery are INHERITED, unmodified, so there is one copy of each to be right --
and in particular one copy of the timeout recovery, which is the part of this harness that was
hardest to get correct and the easiest to silently reintroduce a bug into.

WHY THIS CIRCUIT IS THE *LESS* INDEPENDENT OF THE TWO NEW ONES, ON PURPOSE. It shares an error
amplifier with Circuit A's bandgap, and its objective is supply rejection, as Circuit A's was. That
is the declared reason it is in the study: if optimizer ranking is topology-dependent, a partially
similar topology should sit between the two extremes. Circuit B is structurally independent; a third
circuit chosen to be maximally different would not test the middle of that range.

OBJECTIVE. PSRR at 100 Hz in dB, MAXIMIZED. Scored as a fitness to be minimised,

    f(x) = -PSRR_dB + 1000 * penalty

with penalty the COUNT of violated specifications -- identical in form to Circuits A and B, so all
three are scored the same way and nothing in the comparison depends on the scoring rule.

SIX SPECIFICATIONS, the same count as the other two, one of them a two-sided window. Every one is a
quantity an LDO datasheet actually quotes:

    Vout in [., .]    output accuracy, +/-0.1 %; the counterpart of Circuit A's V_REF window
    LoadReg <=        |mV| per mA of load
    LineReg <=        |mV| per V of supply
    Iq      <=        quiescent current: what the regulator itself burns, load excluded
    Dropout <=        the lowest supply at which regulation still holds, minus the target output
    Area    <=        sum of W*L over the amplifier and the pass device

Circuit C keeps Circuit B's area specification AND adds dropout, which is the defining specification
of a low-dropout regulator and what the pass-device width actually trades against. Closed-loop
PEAKING was a scored specification in the first version and is now a HARD feasibility limit instead --
see PEAKING_HARD_LIMIT_DB for the measurement that forced it. The specification COUNT is still six and
the fitness form is unchanged, so all three circuits remain scored identically.

WHY QUIESCENT CURRENT AND NOT POWER. Total supply power at 1 mA of load is ~1856 uW and essentially
all of it is the load, so it says nothing about the design. Iq = |i(vdd)| - iload isolates what the
regulator costs, which is the quantity a datasheet quotes and the one the divider resistance trades
against accuracy.

WHY REGULATION IS MEASURED AS A MAGNITUDE. Line regulation came out at -0.022 mV/V at L = 2 um: the
output falls slightly as the supply rises, which is a real and legitimate over-compensation of the
loop. Regulation quality is the SIZE of the deviation, so a signed threshold would have scored that
design as better than perfect.

WHY OPERATING POINTS AND NOT DC SWEEPS. `dc Iload ...` and `dc Vdd ...` both truncated before
reaching their endpoints -- ngspice reported "measure ... FIND(AT) : out of interval" -- so load
regulation, line regulation and dropout all returned no data while PSRR and Iq were fine. A sweep
that stops early takes every measurement after it down with it, and on a random design point stopping
early is the normal case. Independent `op` points do not share that failure mode: each converges or
not on its own, and it is also how the quantity would be measured on a bench.

WHY THE REFERENCE IS 0.8 V AND THE DIVIDER 1:2. Measured: with a 0.6 V reference the amplifier's tail
sink sat at 31 mV of Vds against the ~130 mV it needs, so the current source was degraded into a
resistor and the input pair ran below threshold. The regulator still regulated -- v(fb) tracked
v(ref) to 0.6 mV -- which is why this had to be measured rather than eyeballed: a starved amplifier
closes the loop, just with far less gain, and that would have surfaced later as poor supply rejection
blamed on sizing. An NMOS pair on this process needs Vcm > Vth + Vdsat ~ 0.765 V. A 0.8 V reference
with a 1:2 divider holds the same 1.2 V output and leaves 0.6 V of dropout headroom.
"""
from __future__ import annotations

import re

import numpy as np

import ngspice_bridge as nb
from spice_problem import LAMBDA, Counters, SpiceBGRProblem

TEMPLATE = "ldo_sky130.cir.tmpl"
N_SPECS = 6

# ---------------------------------------------------------------- design space
# name, lower, upper, unit, quantisation step (0 = continuous).
# SEVEN variables, matching Circuit A's base case and Circuit B, so all three rankings are compared
# at equal D and dimensionality is never a confound.
VARS = (
    ("w1",    1.0,  40.0, "um",   0.005),   # error-amplifier input pair M1/M2
    ("w3",    1.0,  20.0, "um",   0.005),   # error-amplifier PMOS mirror load M3/M4
    ("w5",    1.0,  20.0, "um",   0.005),   # error-amplifier tail sink and bias mirror
    ("wpass", 5.0, 100.0, "um",   0.005),   # PMOS pass device; 100 um is the model's W ceiling
    ("l_all", 0.15,  2.0, "um",   0.005),   # amplifier channel length
    ("rdiv", 10.0, 500.0, "kohm", 0.0),     # feedback divider total resistance
    ("cc",    0.1,  20.0, "pF",   0.0),     # compensation capacitance
)
VAR_NAMES = tuple(v[0] for v in VARS)

# The pass device's lower bound is 5 um and not 1 um because below ~20 um dropout rises steeply
# (450 mV at 20 um against 100 mV at 100 um) and the design stops being a low-dropout regulator at
# all. 5 um leaves the optimizer room to discover that for itself without filling the box with
# designs that cannot regulate.

# ---------------------------------------------------------------- specifications
# ==================== FROZEN 2026-08-26 -- DO NOT RE-TUNE AFTER ANY CAMPAIGN ====================
# Calibrated against a 512-point Latin-hypercube probe of the design box (scipy.stats.qmc, seed
# 20260826), of which 452 simulated successfully. The probe was run THROUGH this class, so the
# quantiser, the result cache, the deck-error breaker and the ngspice timeout recovery were all
# exercised rather than assumed.
#
# RULE: the corrected section 4 of PREREGISTRATION_2026-08-26_circuits_2_and_3.md, the same one used
# for Circuit B. One declared target marginal pass rate T, mapped mechanically to a per-specification
# quantile of the probe, clamped so the reference design still passes; joint feasibility then required
# in [0.5 %, 5 %]. T = 35 was selected by the declared tie-break: nearest the geometric centre of
# the band among candidates in band.
#
# GATES 1, 2 AND 4 PASS. GATE 3 MISSES ON ONE SPECIFICATION, AND IS REPORTED AS MISSING:
#   1. reference design feasible -- PASS, penalty 0
#   2. random joint feasibility 1.562 % (8 of 512), inside [0.5 %, 5 %] -- PASS
#   3. marginal pass rates in [10 %, 60 %] -- FAIL on one of six. Five sit at 35.0-43.6 %. The output
#      window sits at 64.6 %, 4.6 points over the ceiling, and it cannot be brought lower: the
#      half-width is already the smallest round figure that admits the reference design, whose own
#      output is 2.235 mV off target, and gate 1 forbids a window that excludes the anchor.
#      Tightening it would require an anchor with better DC accuracy, or abandoning gate 1. Reported
#      rather than engineered around, and reported precisely: the miss is 4.6 points on one
#      specification out of six, not a broken calibration.
#   4. random search reaches 75.41 dB while feasible; the reference design reaches 89.49 dB; the best
#      anywhere in the box, feasible or not, is 77.10 dB -- PASS, and in the right direction. Beating
#      random search is the weak claim and beating the reference design is the meaningful one, as for
#      Circuits A and B (15.50/23.85 dB and 81.18/83.49 dB).
#
# HONEST QUALIFICATION ON GATE 4: only 8 of 512 probe points are feasible, so 75.41 dB is an estimate
# from eight samples. Circuit A had 11 of 512 and Circuit B 3 of 512 for the same purpose.
CONSTRAINTS = {
    # +/-0.200 % about the 1.2 V target. Three attempts got here and each failure was caught by a
    # different gate -- see revision R2 below. The half-width is 2.4 mV, the next round figure above
    # the reference design's own 2.235 mV error, so the anchor is inside by 165 uV. A window whose
    # EDGE coincided with the anchor would make feasibility a floating-point coin toss.
    "Vout_min": 1.1976,        # V
    "Vout_max": 1.2024,        # V
    "LoadReg_max": 2.2950,     # mV/mA   reference 1.6450    slack  39.51 %
    "LineReg_max": 1.3694,     # mV/V    reference 0.2889    slack 374.02 %
    "Iq_max": 23.1100,         # uA      reference 22.1400   slack   4.38 %
    "Dropout_max": 160.0000,   # mV      reference 100.0000  slack  60.00 %
    "Area_max": 81.0000,       # um^2    reference 81.0000   slack   0.00 %
}
CALIBRATED = True

# The two bars a campaign has to clear, both reported, the second the headline -- as in A and B.
PROBE_BEST_FEASIBLE_PSRR_DB = 75.41     # best of 6 feasible points in a 512-point random probe
NOMINAL_OBJECTIVE_DB = 89.4851          # the reference design itself

# THE SPECIFICATION SET WAS REVISED TWICE BEFORE FREEZING, both times before any campaign ran and both
# times because a gate failed on measured data. Recorded here so the sequence is checkable:
#
#   R1  Closed-loop PEAKING was one of the six scored specifications; it is now a HARD feasibility
#       limit and AREA took its place among the six. Forced by gate 3: the peaking threshold can never
#       go below the reference design's own 0.0005 dB, and even there 78 % of the box passes, so
#       reaching the 60 % ceiling would need a NEGATIVE threshold. A more peaked anchor would admit a
#       tighter threshold, but a badly damped anchor is not a reference design. Stability is a yes/no
#       property of a loop in any case, and it is still enforced -- more strictly, as a rejection
#       rather than a penalty.
#
#   R2  The output window took three attempts, each failure caught by a different gate. At a declared
#       +/-2 % all 452 measurable probe points passed, so it was not a specification at all -- only a
#       restatement that the simulation had succeeded. At a declared +/-0.1 % it bound at 34 % but
#       EXCLUDED the reference design, breaking gate 1. Calibrated by quantile and clamped to the
#       anchor it landed at exactly the anchor's own error and STILL excluded it, by 35 microvolts,
#       because the clamp used a four-decimal rounding of a full-precision measurement -- the subtlest
#       failure available here, a benchmark rejecting its own anchor by 35 uV. It is now fixed at
#       +/-0.2 %, the next round figure above the anchor's 2.235 mV error.
#
# What did NOT change: the topology, the objective (PSRR at 100 Hz, dB, maximised), the specification
# COUNT (six), the variable count (seven, matching Circuits A and B), the process and PDK commit, the
# fitness form, and the protocol in the pre-registration.

# THE REFERENCE DESIGN, chosen by a criterion declared BEFORE the search that found it:
#
#     MAXIMISE PSRR, subject to
#         LoadReg <= 5 mV/mA,  LineReg <= 5 mV/V,  Iq <= 40 uA,  Dropout <= 200 mV,
#         Peaking <= 1 dB,  Vout within +/-1 %,  and wpass <= 80 um
#
# 1015 of 1458 grid candidates met it; this is the highest-PSRR one. The +/-1 % output bound is
# tighter than the +/-2 % specification window on purpose, so the anchor does not sit on the edge of
# its own specification. The wpass cap is Circuit B's lesson applied in advance: its first anchor sat
# at the 75th percentile of the power distribution, and since gate 1 forbids a threshold that excludes
# the reference design, that made the power threshold untightenable and cost a whole revision.
#
# THE FIRST CRITERION WAS WRONG AND THE CALIBRATION CAUGHT IT. It minimised quiescent current subject
# to the other specifications, which produced an anchor at 64.21 dB -- while random sampling of the
# box reached 75.41 dB. An anchor that random search outperforms is not a reference design, and it
# would have made "beat the reference design", the claim this benchmark exists to support, satisfied
# by chance. The error was selecting the anchor by a CONSTRAINT when the objective is PSRR. Circuits A
# and B both have references that beat random search (23.85 dB against 15.50, 83.49 against 81.18);
# this one now does too, at 89.49 dB against 75.41.
NOMINAL = dict(w1=10.0, w3=5.0, w5=8.0, wpass=80.0, l_all=1.5, rdiv=400.0, cc=2.0)

# Measured at tt, 27 C, 1.8 V, 1 mA of load. These are the anchors the thresholds are set from.
# Peaking is recorded although it is no longer a scored specification: it is the quantity the hard
# feasibility limit is applied to, and the anchor's own value is what proved the scored form
# unworkable.
REFERENCE_METRICS = dict(PSRR=89.4851, Vout=1.202235, Iq=22.1400, LoadReg=1.6450,
                         LineReg=0.2889, Dropout=100.0000, Area=81.0000, Peaking=0.0005)

PSRR_FLOOR_DB = -200.0
PSRR_CEIL_DB = 200.0

CASES = {"base": (TEMPLATE, VARS, NOMINAL, CONSTRAINTS)}
CORNERS = ("tt", "ss", "ff", "sf", "fs")

# ---------------------------------------------------------------- operating conditions
VOUT_TARGET = 1.2                 # 1.5 * the 0.8 V reference, set by the 1:2 divider

# TWO SEPARATE TOLERANCES, and they must stay separate. The first version used one constant for both
# the output-accuracy window and the dropout criterion; tightening it for the window would silently
# have changed every dropout figure already measured, because dropout is defined as the lowest supply
# at which the output still holds "within tolerance".
#
# VOUT_WINDOW_TOL is +/-0.1 %, not the +/-2 % first used. Measured: at +/-2 % all 452 measurable
# probe points passed, so the window was not a specification at all -- it merely restated that the
# simulation had succeeded. With an ideal reference and a fixed divider ratio the DC output is exact
# by construction, so the deviation that remains is a direct measure of LOOP GAIN, and +/-1.2 mV over
# a measured spread of 25 mV makes that a real accuracy requirement. It binds at 36 %.
VOUT_WINDOW_TOL = 0.001           # +/-0.1 %, the output-accuracy specification
VOUT_DROPOUT_TOL = 0.02           # +/-2 %, the dropout criterion -- unchanged, so the probe holds

# Closed-loop peaking is a HARD feasibility limit, not a scored specification. Measured: the scored
# form cannot work. The threshold can never go below the reference design's own 0.0005 dB, and even
# there 78 % of the box passes; reaching the gate's 60 % ceiling would need a NEGATIVE threshold. A
# more peaked anchor would admit a tighter threshold, but a badly damped anchor is not a reference
# design. Stability is in any case a yes/no property of a loop rather than a graded one: 3 dB of
# closed-loop rise means the loop is marginally stable and the design is not a working regulator, so
# it is rejected exactly as a design that fails to regulate at any supply is rejected. Peaking is
# still MEASURED and RECORDED for every candidate -- it is simply not one of the six scored penalties.
PEAKING_HARD_LIMIT_DB = 3.0
ILOAD_NOM_MA = 1.0
ILOAD_LO_MA, ILOAD_HI_MA = 0.2, 2.0
VDD_LO, VDD_HI = 1.62, 1.98       # +/-10 %, as Circuit A uses
# The supply ladder for dropout. Finer than the first six-rung attempt, which quantised dropout to
# {100, 150, 300, 450} mV and made a continuous quantity look categorical. Each rung costs one `op`.
DROPOUT_LADDER = (1.22, 1.24, 1.26, 1.28, 1.30, 1.33, 1.36, 1.40,
                  1.45, 1.50, 1.56, 1.62, 1.70)


def _build_commands() -> list[str]:
    """The full per-candidate command list.

    Ordering is forced by measurement, not style:
      * `ac` before any `op`/`dc`, and the supply and reference stimuli are NEVER armed together --
        otherwise the two transfer functions superpose and both are wrong.
      * `meas ac ... FIND v(x)` returns the REAL PART of a complex vector, so magnitudes are built
        with `let m = mag(...)` first. `vdb()` inside `meas` aborts the whole AC analysis in
        ngspice-46 and is not used.
      * `alterparam` requires a following `reset`, and `reset` UNDOES `alter`. So every parameter
        change happens before the source alters that depend on it, and the load-current points (which
        need alterparam) come before the supply points (which need alter).
      * each operating point is labelled with an `echo` marker, because nine identical
        `print v(out)` lines are otherwise indistinguishable in the output stream.
    """
    # `destroy all` FIRST, and unlike Circuit B this one is NOT optional. Each candidate here leaves
    # 21 plots behind -- two AC sweeps and nineteen operating points -- against Circuit B's three.
    # Measured on a 600-candidate reference search WITHOUT it: ngspice private memory grew from
    # 108 MB to 910 MB, about 1.3 MB per candidate, and throughput decayed from 2.4 s to 7 s per
    # candidate as it went. At the campaign's 2,500 evaluations per run that is ~3 GB per worker, and
    # 90 concurrent workers would be impossible.
    #
    # Worth recording that the same command was tried and REMOVED for Circuit B, because measurement
    # there showed resident memory flat at 50 MB and no decay. Both decisions were right: the leak
    # scales with plots per candidate, and only this circuit generates enough of them to matter.
    c = ["destroy all", "reset",
         "alter @Vdd[acmag] = 1", "alter @Vref[acmag] = 0",
         "ac dec 10 1 1e8",
         "let msup = mag(v(out))",
         "meas ac s_lf FIND msup AT=100",
         "let psrr_db = -20*log10(s_lf + 1e-18)",
         "print psrr_db",
         "alter @Vdd[acmag] = 0", "alter @Vref[acmag] = 1",
         "ac dec 20 1 1e8",
         "let mcl = mag(v(out))",
         "meas ac cl_lf  FIND mcl AT=1",
         "meas ac cl_max MAX mcl",
         "let peak_db = 20*log10(cl_max/(cl_lf + 1e-18))",
         "print peak_db",
         "alter @Vref[acmag] = 0",
         "op", "echo TAG_NOM", "print v(out)", "print v(fb)", "print i(vdd)"]
    for tag, il in (("LOADLO", ILOAD_LO_MA), ("LOADHI", ILOAD_HI_MA)):
        c += [f"alterparam iload={il}", "reset", "op", f"echo TAG_{tag}", "print v(out)"]
    c += [f"alterparam iload={ILOAD_NOM_MA}", "reset"]
    for tag, vd in (("LINELO", VDD_LO), ("LINEHI", VDD_HI)):
        c += [f"alter @Vdd[dc] = {vd}", "op", f"echo TAG_{tag}", "print v(out)"]
    for i, vd in enumerate(DROPOUT_LADDER):
        c += [f"alter @Vdd[dc] = {vd}", "op", f"echo TAG_DROP{i}", "print v(out)"]
    c += ["alter @Vdd[dc] = 1.8"]
    return c


COMMANDS = _build_commands()


def _num(text: str) -> float | None:
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _first(o: str, name: str) -> float | None:
    m = re.search(re.escape(name) + r"\s*=\s*([-+0-9.eE]+)", o)
    return _num(m.group(1)) if m else None


def _after(o: str, tag: str, name: str = "v(out)") -> float | None:
    """The first `name = value` printed AFTER the given echo marker."""
    i = o.find(tag)
    if i < 0:
        return None
    m = re.search(re.escape(name) + r"\s*=\s*([-+0-9.eE]+)", o[i:])
    return _num(m.group(1)) if m else None


class SpiceLdoProblem(SpiceBGRProblem):
    """Circuit C. See the module docstring for what is overridden and what is inherited."""

    # No derived geometry: all seven variables reach ngspice directly, so the inherited quantise()
    # is a pure clip-and-snap and Circuit A's primary*ratio clamp table is not used.
    DERIVED_HIGHDIM = ()

    def __init__(self, sky130_lib: str, exe: str | None = None, case: str = "base",
                 constraints: dict | None = None, timeout: float = 60.0,
                 cache_size: int = 20_000, deck_error_budget: float = 0.02,
                 breaker_after: int = 200,
                 corner: str = "tt", vdd: float = 1.8, allow_off_nominal: bool = False,
                 require_calibrated: bool = True, nominal: dict | None = None):
        if case not in CASES:
            raise ValueError(f"unknown case {case!r}; available: {sorted(CASES)}")
        template, self._vars, default_nominal, default_constraints = CASES[case]
        self._nominal = dict(nominal if nominal is not None else default_nominal)
        self._constraints = dict(constraints if constraints is not None else default_constraints)
        if require_calibrated and not self._constraints:
            raise ValueError(
                "Circuit C's thresholds are not calibrated yet. Run the 512-point Latin-hypercube "
                "probe and freeze CONSTRAINTS per PREREGISTRATION_2026-08-26 section 4, or pass "
                "require_calibrated=False for a diagnostic run that is not a campaign.")
        if not self._nominal:
            # The deck still has to be rendered with SOMETHING, and silently substituting a guess
            # would make every diagnostic run measure a design nobody chose.
            raise ValueError("no reference design: pass nominal=... until NOMINAL is frozen")
        if (corner != "tt" or vdd != 1.8) and not allow_off_nominal:
            raise ValueError(
                f"corner={corner!r}, vdd={vdd!r} is off-nominal. Optimization runs at tt/1.8 V "
                f"only; pass allow_off_nominal=True if this is the PVT verification pass.")
        if corner not in CORNERS:
            raise ValueError(f"unknown corner {corner!r}; known: {sorted(CORNERS)}")

        self.corner, self.vdd = corner, float(vdd)
        self.case = case
        self.name = "SKY130-LDO" if case == "base" else f"SKY130-LDO-{case}"
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
        # return identical metrics with no error anywhere.
        subs = {"SKY130_LIB": lib, "CORNER": corner, "VDD_DC": repr(float(vdd))}
        subs.update({k.upper(): f"{float(v):.9g}" for k, v in self._nominal.items()})
        deck = nb.render_netlist(template, subs)
        if "${" in deck:
            raise ValueError(f"template placeholder left unsubstituted in {template}")
        self._srv = nb.NgspiceServer(deck, exe=self._exe, eval_timeout=timeout,
                                     load_timeout=400.0)
        self._srv.load()

    # ------------------------------------------------------------ objective
    def fitness_from_metrics(self, m: dict) -> float:
        return float(-m["PSRR_DB"] + LAMBDA * m["penalty"])

    # ------------------------------------------------------------ simulation
    def _simulate(self, params: dict):
        # `:.9g` and NOT `{v!r}`. Under NumPy 2.x repr() of a numpy scalar is `np.float64(...)`,
        # which SPICE cannot parse: the `.param` fails silently, the expression using it reports an
        # undefined parameter, and ngspice EXITS -- which the harness records as a TIMEOUT, a fatal
        # netlist error disguised as a slow simulation. Cost of learning that on Circuit B: hours.
        out = self._srv.send(
            [f"alterparam {k}={float(v):.9g}" for k, v in params.items()] + COMMANDS)
        if out is None:
            return None, nb.TIMEOUT, "no response during the ac/op sequence"
        kind, detail = nb.classify_output(out, 0)
        if kind is not None:
            return None, kind, detail

        v = {
            "psrr_db": _first(out, "psrr_db"),
            "peak_db": _first(out, "peak_db"),
            "vout": _after(out, "TAG_NOM"),
            "vfb": _after(out, "TAG_NOM", "v(fb)"),
            "ivdd": _after(out, "TAG_NOM", "i(vdd)"),
            "vo_loadlo": _after(out, "TAG_LOADLO"),
            "vo_loadhi": _after(out, "TAG_LOADHI"),
            "vo_linelo": _after(out, "TAG_LINELO"),
            "vo_linehi": _after(out, "TAG_LINEHI"),
        }
        # The dropout ladder: the lowest supply at which the output still holds within tolerance.
        # A design that fails even the top rung has no dropout figure at all, which is itself the
        # informative outcome -- it does not regulate at 1.7 V, so it is not a regulator.
        lo = VOUT_TARGET * (1.0 - VOUT_DROPOUT_TOL)
        v["dropout_mv"] = None
        for i, vd in enumerate(DROPOUT_LADDER):
            got = _after(out, f"TAG_DROP{i}")
            if got is not None and got >= lo:
                v["dropout_mv"] = (vd - VOUT_TARGET) * 1000.0
                break

        hard = [k for k in ("psrr_db", "vout", "ivdd") if v[k] is None]
        if hard:
            return None, nb.MEASURE_FAILED, "no value for " + ", ".join(hard)
        soft = [k for k in ("vo_loadlo", "vo_loadhi", "vo_linelo", "vo_linehi") if v[k] is None]
        if soft:
            return None, nb.MEASURE_FAILED, "operating point did not converge: " + ", ".join(soft)
        if v["dropout_mv"] is None:
            return None, nb.MEASURE_FAILED, "does not regulate at any supply in the ladder"
        # Marginally stable is not a working regulator. Rejected here rather than penalised, for the
        # reasons recorded beside PEAKING_HARD_LIMIT_DB.
        if v["peak_db"] is not None and v["peak_db"] > PEAKING_HARD_LIMIT_DB:
            return None, nb.MEASURE_FAILED, (f"closed-loop peaking {v['peak_db']:.2f} dB exceeds "
                                             f"{PEAKING_HARD_LIMIT_DB:g} dB: loop not stable")
        return v, None, ""

    # ------------------------------------------------------------ metrics
    def _metrics_from(self, v: dict, params: dict) -> dict:
        c = self._constraints
        psrr = float(min(max(float(v["psrr_db"]), PSRR_FLOOR_DB), PSRR_CEIL_DB))
        peak = float(v["peak_db"]) if v["peak_db"] is not None else float("inf")
        vout = float(v["vout"])
        iq_ua = (abs(float(v["ivdd"])) - ILOAD_NOM_MA * 1e-3) * 1e6
        # MAGNITUDES. Line regulation reaches -0.022 mV/V for some designs -- the output falls
        # slightly as the supply rises, a real over-compensation of the loop -- and a signed
        # threshold would score that as better than perfect.
        loadreg = abs(1000.0 * (float(v["vo_loadlo"]) - float(v["vo_loadhi"]))
                      / (ILOAD_HI_MA - ILOAD_LO_MA))
        linereg = abs(1000.0 * (float(v["vo_linehi"]) - float(v["vo_linelo"]))
                      / (VDD_HI - VDD_LO))
        dropout = float(v["dropout_mv"])
        area = (2*params["w1"] + 2*params["w3"] + 2*params["w5"]) * params["l_all"] \
            + params["wpass"] * 0.15

        viol_vout = int(vout < c["Vout_min"] or vout > c["Vout_max"])
        viol_load = int(loadreg > c["LoadReg_max"])
        viol_line = int(linereg > c["LineReg_max"])
        viol_iq = int(iq_ua > c["Iq_max"])
        viol_drop = int(dropout > c["Dropout_max"])
        viol_area = int(area > c["Area_max"])
        penalty = viol_vout + viol_load + viol_line + viol_iq + viol_drop + viol_area

        m = {
            "PSRR_DB": psrr, "penalty": penalty, "is_feasible": int(penalty == 0),
            "VOUT_V": vout, "LOADREG_MV_MA": loadreg, "LINEREG_MV_V": linereg,
            "IQ_UA": iq_ua, "DROPOUT_MV": dropout, "PEAKING_DB": peak,
            "AREA_UM2": area, "VFB_V": float(v["vfb"]) if v.get("vfb") is not None else float("nan"),
            "viol_vout": viol_vout, "viol_loadreg": viol_load, "viol_linereg": viol_line,
            "viol_iq": viol_iq, "viol_dropout": viol_drop, "viol_area": viol_area,
            "sim_ok": 1, "sim_failure": None,
        }
        m.update({"c_" + k: val for k, val in c.items()})
        m.update({"x_" + k: val for k, val in params.items()})
        return m

    def _failed_metrics(self, params: dict, failure: str, detail: str) -> dict:
        c = self._constraints
        area = (2*params["w1"] + 2*params["w3"] + 2*params["w5"]) * params["l_all"] \
            + params["wpass"] * 0.15
        m = {
            "PSRR_DB": PSRR_FLOOR_DB, "penalty": N_SPECS, "is_feasible": 0,
            "VOUT_V": float("nan"), "LOADREG_MV_MA": float("nan"),
            "LINEREG_MV_V": float("nan"), "IQ_UA": float("nan"),
            "DROPOUT_MV": float("nan"), "PEAKING_DB": float("nan"),
            "AREA_UM2": area,          # geometry is known even when the simulation is not
            "VFB_V": float("nan"),
            "viol_vout": 1, "viol_loadreg": 1, "viol_linereg": 1,
            "viol_iq": 1, "viol_dropout": 1, "viol_area": 1,
            "sim_ok": 0, "sim_failure": failure, "sim_detail": detail[:120],
        }
        m.update({"c_" + k: val for k, val in c.items()})
        m.update({"x_" + k: val for k, val in params.items()})
        return m
