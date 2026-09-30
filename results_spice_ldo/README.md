# Circuit C (SKY130 low-dropout regulator) — calibration artifacts

Everything the frozen thresholds in `spice/spice_problem_ldo.py` rest on, so the calibration is
reproducible from the data rather than quoted from a log.

| file | what it is |
|---|---|
| `lhs_probe.csv` | **The probe the thresholds were frozen from.** 512 Latin-hypercube points, seed 20260826; 452 simulated successfully, and all 60 failures are the same informative mode — the design does not regulate at any supply in the ladder. Run *through* `SpiceLdoProblem`, so the quantiser, cache, deck-error breaker and ngspice timeout recovery were exercised rather than assumed. |
| `lhs_probe.py` | the script that produced it |
| `calibrate.py` | the calibration. Scans the single declared knob *T*, prints the thresholds and all four gates. Re-running reproduces the frozen numbers exactly. |
| `pick_reference.py` | the reference-design search, with its criterion declared in the docstring above the code that uses it |

## Reproducing

```bash
python lhs_probe.py     # ~5 min
python calibrate.py     # prints thresholds + gates
```

## The reference design

`w1=10 w3=5 w5=8 wpass=80 l_all=1.5 rdiv=400 cc=2` — chosen by maximising PSRR subject to
LoadReg ≤ 5 mV/mA, LineReg ≤ 5 mV/V, Iq ≤ 40 µA, Dropout ≤ 200 mV, Peaking ≤ 1 dB, Vout within ±1 %,
and `wpass ≤ 80 µm`. 1015 of 1458 grid candidates met it.

| | value |
|---|---|
| PSRR | 89.4851 dB |
| Vout | 1.202235 V |
| Iq | 22.14 µA |
| Load regulation | 1.645 mV/mA |
| Line regulation | 0.2889 mV/V |
| Dropout | 100 mV |
| Area | 81.0 µm² |
| Peaking | 0.0005 dB |

## Gates

**1, 2 and 4 pass. 3 misses on one specification of six, and that is reported, not worked around.**

- **1** reference design feasible — PASS, penalty 0, verified through the full evaluator
- **2** random joint feasibility **1.562 %** (8 of 512), inside [0.5 %, 5 %] — PASS
- **3** marginal pass rates in [10 %, 60 %] — **FAIL on one of six.** Five sit at 35.0–43.6 %; the
  output window sits at **64.6 %**. It cannot go lower: the half-width is already the smallest round
  figure that admits the reference design, whose own output is 2.235 mV off target, and gate 1
  forbids a window excluding the anchor.
- **4** random search reaches **75.41 dB** feasible; the reference design reaches **89.49 dB** — PASS,
  and in the right direction. Only 8 probe points are feasible, so 75.41 dB is an eight-sample
  estimate.

## The first criterion was wrong, and the calibration caught it

The reference design was originally chosen by **minimising quiescent current**, which produced an
anchor at 64.21 dB — while random sampling of the box reached 75.41 dB. An anchor that random search
outperforms is not a reference design, and it would have made *"beat the reference design"* — the claim
this benchmark exists to support — satisfied by chance. The error was selecting the anchor by a
**constraint** when the objective is PSRR.

See `spice_problem_ldo.py` for the full revision log, including the two specification-set revisions
(peaking moved from a scored penalty to a hard feasibility limit; the output window took three
attempts, the third failing by 35 microvolts).
