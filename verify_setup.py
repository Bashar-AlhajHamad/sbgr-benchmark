"""
verify_setup.py -- run this FIRST on a new machine, before committing hours to a campaign.

Checks, in the order that matters:
  1. Python packages
  2. ngspice binary is found
  3. the bridge's own three suites pass
  4. the SKY130 library resolves and simulates
  5. the nominal design reproduces its three PUBLISHED numbers
     (VREF 1220 mV, TC 24 ppm/degC, power 47 uW) -- if this fails, nothing downstream means
     anything, and it is the one check that proves the whole toolchain moved correctly
  6. the loop-gain probe is intact (LG/PM/GM at the nominal point)
  7. the frozen thresholds still make the nominal design feasible
  8. results can actually be WRITTEN and re-read from the output directory
     -- added because a campaign once ran for hours and its checkpoint CSV turned out never
        to have reached the disk. Never walk away from a long run without proving this.

    python verify_setup.py --lib <path to sky130.lib.spice> --outdir results_spice
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPICE = HERE / "spice"
sys.path.insert(0, str(SPICE))

FAILS: list[str] = []


def check(label: str, ok: bool, extra: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  -- ' + extra) if extra else ''}")
    if not ok:
        FAILS.append(label)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="path to sky130.lib.spice (forward slashes)")
    ap.add_argument("--outdir", default="results_spice")
    args = ap.parse_args()

    print("=== 1. python packages ===")
    for mod in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            __import__(mod)
            check(mod, True)
        except Exception as e:  # noqa: BLE001
            check(mod, False, str(e))
    if FAILS:
        print("\n  install them with:  py -m pip install numpy pandas scipy matplotlib Jinja2")
        return 1

    print("\n=== 2. ngspice binary ===")
    import ngspice_bridge as nb
    try:
        exe = nb.find_ngspice()
        check("found", True, exe)
    except RuntimeError as e:
        check("found", False, str(e))
        return 1

    print("\n=== 3. the bridge's own suites ===")
    for flag in ("--selftest", "--phase2", "--server"):
        r = subprocess.run([sys.executable, str(SPICE / "ngspice_bridge.py"), flag],
                           capture_output=True, text=True, timeout=900)
        ok = r.returncode == 0
        tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-1:] or [""]
        check(f"ngspice_bridge.py {flag}", ok, tail[0][:70])

    print("\n=== 4-7. the library, the anchor, the loop probe, the thresholds ===")
    lib = args.lib.replace("\\", "/")
    if " " in lib:
        check("the .lib path has no spaces", False,
              "SPICE tokenises on whitespace -- move the PDK somewhere without spaces")
        return 1
    if not Path(lib).is_file():
        check("the .lib file exists", False, lib)
        return 1
    check("the .lib file exists", True)

    import spice_problem as sp
    import numpy as np
    t0 = time.perf_counter()
    p = sp.SpiceBGRProblem(sky130_lib=lib, timeout=60.0)
    print(f"       (library parsed in {p._srv.load_seconds:.0f}s)")
    try:
        x = np.array([sp.NOMINAL[n] for n in sp.VAR_NAMES], dtype=float)
        f, m = p.evaluate_with_metrics(x)
        dt = time.perf_counter() - t0
        check("the nominal point simulates", m["sim_ok"] == 1, str(m.get("sim_failure")))
        print(f"       VREF={m['VREF']*1000:.2f} mV   TC={m['TC']:.2f} ppm/C   "
              f"P={m['POWER_UW']:.2f} uW")
        print(f"       PSRR={m['PSRR_DB']:.2f} dB   LG={m['LOOP_GAIN_DB']:.2f} dB   "
              f"PM={m['PHASE_MARGIN_DEG']:.2f} deg   GM={m['GAIN_MARGIN_DB']:.2f} dB")
        check("VREF reproduces the published 1220 mV", abs(m["VREF"] * 1000 - 1220) < 5,
              f"{m['VREF']*1000:.2f} mV")
        check("TC reproduces the published 24 ppm/C", abs(m["TC"] - 24) < 3,
              f"{m['TC']:.2f}")
        check("power reproduces the published 47 uW", abs(m["POWER_UW"] - 47) < 3,
              f"{m['POWER_UW']:.2f}")
        check("the loop-gain probe returns finite LG/PM/GM",
              all(np.isfinite([m["LOOP_GAIN_DB"], m["PHASE_MARGIN_DEG"],
                               m["GAIN_MARGIN_DB"]])))
        check("the nominal design is FEASIBLE under the frozen thresholds",
              m["penalty"] == 0, f"penalty={m['penalty']}")
        check("fitness is finite", np.isfinite(f), f"{f:.3f}")
        print(f"       one full evaluation cost {dt - p._srv.load_seconds:.2f}s")
    finally:
        p.close()

    print("\n=== 8. the output directory is really writable ===")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    probe = out / "_write_probe.csv"
    try:
        with open(probe, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["probe", 1])
        text = probe.read_text(encoding="utf-8")
        check("a file written there can be re-read", "probe" in text, str(probe.resolve()))
        probe.unlink()
        check("and deleted again", not probe.exists())
    except OSError as e:
        check("the output directory is writable", False, str(e))

    print("\n" + "=" * 66)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED -- do not start the campaign:")
        for f_ in FAILS:
            print(f"  - {f_}")
        return 1
    print("ALL CHECKS PASSED.")
    print("Next:  py bench_workers.py --lib <lib>      (pick --workers by measurement)")
    print("Then:  py run_spice.py --lib <lib> --outdir results_spice --workers <N>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
