# Circuit B (SKY130 two-stage op-amp) — calibration artifacts

Everything the frozen thresholds in `spice/spice_problem_opamp.py` rest on. Kept so the calibration
is reproducible from the data rather than quoted from a log.

| file | what it is |
|---|---|
| `lhs_probe.csv` | **The probe the thresholds were frozen from.** 512 Latin-hypercube points, seed 20260826, all 512 simulated successfully. Run *through* `SpiceOpampProblem`, so the quantiser, derived-geometry clamp, cache, deck-error breaker and ngspice timeout recovery were all exercised. |
| `lhs_probe.py` | the script that produced it |
| `calibrate.py` | the calibration. Reads `lhs_probe.csv`, scans the single declared knob *T*, prints the thresholds and all four acceptance gates. Re-running it reproduces the frozen numbers exactly. |
| `pick_reference.py` | the reference-design search, with its criterion declared in the docstring above the code that uses it |
| `lhs_probe_SUPERSEDED_free_widths.csv` | the earlier probe of the **free-width** box. Superseded, and kept deliberately: revision R2 (bounding the stage-2 widths as ratios) rests on the 3.3 % output-window rate measured here, so deleting it would remove the evidence for a design decision. |

## Reproducing

```bash
python lhs_probe.py        # ~4 min, writes probe_final.csv
python calibrate.py        # prints thresholds + gates
```

## The numbers, for quick reference

Reference design `w1=10 w3=10 w5=4 m2=3 k=0.85 l_all=1 cc=3` → derived `w6=30.0 w7=5.1`:
gain 83.4923 dB, UGB 5.5498 MHz, PM 62.4100°, power 58.5027 µW, Vout 1.1066 V,
area 83.1000 µm², PSRR 89.9284 dB.

All four gates pass: reference feasible; joint feasibility 0.586 % (3/512); every marginal pass rate
in [10 %, 60 %] (five at 60.0 %, output window 12.7 %); random-feasible best 81.18 dB against the
reference design's 83.49 dB.

See `CIRCUIT_B_DESIGN_FROZEN_2026-08-26.md` §7 for the record and §8 for the revision log.
