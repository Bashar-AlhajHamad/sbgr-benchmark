# SPICE bridge (ngspice)

An optional transistor-level cross-check for the SBGR surrogate benchmark. It lets the
same six optimizers run against a **real ngspice** evaluation of a bandgap-reference (BGR)
circuit, so the *relative ranking* found on the surrogate can be spot-checked on an actual
netlist. This is an **illustrative cross-check, not a validation** — the full published
campaign (~10^8 evaluations) is infeasible in SPICE by design, which is itself the reason
the surrogate benchmark exists.

The bridge is built in phases; each phase ends in something demonstrably working.

## Status

- **Phase 1 — minimal chain (in place):** `ngspice_bridge.py` renders a netlist from a
  template, runs ngspice in batch, parses one `.measure` result, and returns a float.
  The placeholder circuit is a resistive divider with a known closed-form answer, so the
  chain can be proven with zero analog work.
- Phase 2 (hardening), Phase 3 (problems.py interface parity), Phase 4 (real BGR netlist),
  Phase 5 (micro-campaign): not started.

## Requirements

- Python 3 (standard library only for Phase 1 — no numpy needed yet).
- **ngspice** installed. The bridge locates it in this order:
  1. `NGSPICE_EXE` — a full path to the binary (explicit override).
  2. `ngspice` / `ngspice_con` on `PATH`.
  3. Common Windows install dirs, and a repo-local portable copy under `spice/tools/`.

## Quick start

From the `spice/` directory:

```sh
python ngspice_bridge.py --find       # confirm ngspice is located
python ngspice_bridge.py --selftest   # render + run + parse the divider
```

`--selftest` renders the divider with RTOP=2 kΩ, RBOT=1 kΩ, so it expects
`v(out) = 1000 / (2000 + 1000) = 0.3333 V` and reports PASS/MISMATCH.

If `--find` reports NOT FOUND, either add ngspice to `PATH`, set `NGSPICE_EXE`, or unzip a
portable Windows build under `spice/tools/`.

## Files

| File | Purpose |
|---|---|
| `ngspice_bridge.py` | Phase 1 chain: `find_ngspice`, `render_netlist`, `run_ngspice`, `parse_measure`, `evaluate_divider`, plus a `--find`/`--selftest` CLI. |
| `templates/divider.cir.tmpl` | Placeholder resistive-divider deck; `${RTOP}`/`${RBOT}` substituted at render time. |
