#!/bin/bash
# Native setup -- no container. Run ON THE LOGIN NODE (needs the internet).
#
#   bash 00_setup_native.sh
#
# WHY NOT THE CONTAINER. The Apptainer build failed on TRUBA for two compounding reasons, both
# visible in its own output:
#   * the build tree was on /arf/scratch, a parallel filesystem, which reported
#     "destination filesystem does not support xattrs" and then failed every rpm that unpacks a
#     symlink: "cpio: symlink failed - Inappropriate ioctl for device"
#   * real fakeroot is unavailable -- "User not listed in /etc/subuid, trying root-mapped
#     namespace" -- so the fallback namespace cannot create those files either
# Moving the build to a local disk might fix the first but not reliably the second, and neither
# is worth fighting: ngspice is a plain autotools program that needs no root at all.
#
# WHAT IS LOST, AND WHAT REPLACES IT. A container image pins the binary by digest. Building from
# source pins it by (source tarball version + compiler version + configure flags), all of which
# this script records into env.sh and the gate prints. That is the standard level of specification
# for an HPC paper, and it preserves the property that actually matters: ONE binary, built once,
# used by all 540 tasks. Do not rebuild it between cases.
set -euo pipefail

USER_ID="${TRUBA_USER:-$USER}"
ROOT=/arf/scratch/${USER_ID}/sky130-bgr
CODE=${ROOT}/code
PDK=${ROOT}/pdk
PREFIX=${ROOT}/opt
BUILD=${ROOT}/build
VENV=${ROOT}/venv

NGSPICE_VERSION=46
# The PDK commit every validated measurement was made against. Pinned, not "latest": a different
# commit is a different set of transistor models, and the frozen thresholds were calibrated here.
PDK_COMMIT=c6d73a35f524070e85faff4a6a9eef49553ebc2b

mkdir -p "${ROOT}" "${PDK}" "${PREFIX}" "${BUILD}" "${ROOT}/logs" "${ROOT}/results"

echo "=== 0. clean up the failed container attempt ==="
rm -rf /arf/scratch/${USER_ID}/tmp/build-temp-* 2>/dev/null || true
rm -f "${ROOT}/sky130-ngspice.sif" 2>/dev/null || true
echo "  done"

echo
echo "=== 1. toolchain on this node ==="
for t in gcc make bison flex wget tar; do
    if command -v "$t" >/dev/null 2>&1; then
        printf "  %-6s %s\n" "$t" "$(command -v $t)"
    else
        printf "  %-6s MISSING\n" "$t"
    fi
done
echo "  gcc version: $(gcc --version 2>/dev/null | head -1 || echo 'none')"

echo
echo "=== 2. python ==="
# Prefer a newer interpreter if the module system offers one; the manuscript states 3.13 for the
# surrogate campaign, and any difference has to be recorded rather than assumed away. numpy's
# default_rng (PCG64) stream is version-stable, so results reproduce across interpreter versions.
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$cand" >/dev/null 2>&1 && { PY=$(command -v "$cand"); break; }
done
[ -n "${PY}" ] || { echo "  FATAL: no python3 found"; exit 1; }
echo "  using ${PY}  ($(${PY} --version 2>&1))"
echo "  (if a newer python module exists, 'module avail 2>&1 | grep -i python' will show it;"
echo "   re-run this script after loading it and the venv will be rebuilt on that interpreter)"

if [ ! -x "${VENV}/bin/python" ]; then
    "${PY}" -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --quiet --upgrade pip
"${VENV}/bin/python" -m pip install --quiet \
    "numpy>=2.0,<3" "pandas>=2.2,<4" "scipy>=1.13,<2" "matplotlib>=3.8,<4" volare
"${VENV}/bin/python" - <<'EOF'
import sys, numpy, pandas, scipy
print(f"  python {sys.version.split()[0]}  numpy {numpy.__version__}  "
      f"pandas {pandas.__version__}  scipy {scipy.__version__}")
EOF

echo
echo "=== 3. ngspice ==="
NGSPICE="${PREFIX}/bin/ngspice"
if [ -x "${NGSPICE}" ]; then
    echo "  already built: ${NGSPICE}"
else
    # A pre-existing system/module ngspice is NOT used even if present: its version and configure
    # flags are unknown, and every measurement in this project was validated against 46.
    cd "${BUILD}"
    TARBALL="ngspice-${NGSPICE_VERSION}.tar.gz"
    if [ ! -f "${TARBALL}" ]; then
        echo "  downloading ngspice-${NGSPICE_VERSION}"
        wget -q --show-progress \
            "https://sourceforge.net/projects/ngspice/files/ng-spice-rework/${NGSPICE_VERSION}/${TARBALL}/download" \
            -O "${TARBALL}" \
        || wget -q --show-progress \
            "https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/${NGSPICE_VERSION}/${TARBALL}" \
            -O "${TARBALL}"
    fi
    rm -rf "ngspice-${NGSPICE_VERSION}"
    tar xf "${TARBALL}"
    cd "ngspice-${NGSPICE_VERSION}"
    # --without-x, --with-readline=no : batch/pipe workload, no display and no terminal, and both
    #   libraries only add ways for a headless job to fail.
    # NOT -march=native : the login node and the compute nodes may differ, and a binary tuned for
    #   one microarchitecture running on another is how a campaign acquires results it cannot
    #   reproduce.
    # NOT --enable-openmp : parallelism is one SLURM task per (algo, run); a threaded solver would
    #   oversubscribe its allocated core and slow every neighbouring task.
    echo "  configuring"
    ./configure --prefix="${PREFIX}" \
        --without-x --with-readline=no \
        --enable-xspice --disable-debug \
        CFLAGS="-O2 -fcommon" > "${ROOT}/logs/ngspice_configure.log" 2>&1 \
        || { echo "  FATAL: configure failed; see ${ROOT}/logs/ngspice_configure.log";
             tail -25 "${ROOT}/logs/ngspice_configure.log"; exit 1; }
    echo "  compiling (a few minutes)"
    make -j"$(nproc)" > "${ROOT}/logs/ngspice_make.log" 2>&1 \
        || { echo "  FATAL: make failed; see ${ROOT}/logs/ngspice_make.log";
             tail -25 "${ROOT}/logs/ngspice_make.log"; exit 1; }
    make install >> "${ROOT}/logs/ngspice_make.log" 2>&1
    cd "${ROOT}" && rm -rf "${BUILD}/ngspice-${NGSPICE_VERSION}"
fi
"${NGSPICE}" --version | head -2 | sed 's/^/  /'

echo
echo "=== 4. PDK (${PDK_COMMIT:0:12}) ==="
LIB=""
for sub in combined ngspice; do
    C="${PDK}/volare/sky130/versions/${PDK_COMMIT}/sky130A/libs.tech/${sub}/sky130.lib.spice"
    [ -f "${C}" ] && { LIB="${C}"; break; }
done
if [ -z "${LIB}" ]; then
    echo "  fetching with volare"
    PDK_ROOT="${PDK}" "${VENV}/bin/volare" enable --pdk sky130 "${PDK_COMMIT}" \
        || echo "  (volare can exit non-zero after a successful extraction; checking below)"
    for sub in combined ngspice; do
        C="${PDK}/volare/sky130/versions/${PDK_COMMIT}/sky130A/libs.tech/${sub}/sky130.lib.spice"
        [ -f "${C}" ] && { LIB="${C}"; break; }
    done
fi
if [ -z "${LIB}" ]; then
    echo "  FAILED: sky130.lib.spice not found under ${PDK}"
    echo "  Fallback: download the 'common' and 'sky130_fd_pr' .tar.zst assets from the volare"
    echo "  GitHub release for commit ${PDK_COMMIT} and extract them into ${PDK}."
    exit 1
fi
echo "  OK  ${LIB}"
echo "  corners available: $(grep -cE '^[[:space:]]*\.lib[[:space:]]+[A-Za-z]' "${LIB}")"

echo
echo "=== 5. env.sh ==="
cat > "${ROOT}/env.sh" <<EOF
# sourced by every job script -- written by 00_setup_native.sh on $(date -Is)
export SKY130_ROOT=${ROOT}
export SKY130_CODE=${CODE}
export SKY130_LIB=${LIB}
export SKY130_PDK_COMMIT=${PDK_COMMIT}
export SKY130_PY=${VENV}/bin/python
export NGSPICE_EXE=${NGSPICE}
export PATH=${PREFIX}/bin:\$PATH
# one core per task: a threaded BLAS would oversubscribe the single allocated CPU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LC_ALL=C
# provenance for section 6
export SKY130_NGSPICE_VERSION=${NGSPICE_VERSION}
export SKY130_BUILD_CC="$(gcc --version 2>/dev/null | head -1)"
EOF
sed 's/^/  /' "${ROOT}/env.sh"

echo
echo "=== 6. self-test ==="
source "${ROOT}/env.sh"
cd "${SKY130_CODE}"
"${SKY130_PY}" preflight_spice.py 2>&1 | tail -4 | sed 's/^/  /'

echo
echo "=== done ==="
echo
echo "NEXT, and do not skip it -- the anchor gate on the debug queue:"
echo "    sbatch ${CODE}/truba/01_gate_native.slurm"
echo
echo "It decides whether THIS ngspice build reproduces the validated circuit measurements"
echo "(1219.90 mV / 24.49 ppm/degC / 47.27 uW, loop gain 53.21 dB, PM 62.99 deg, GM 14.86 dB)."
echo "Submitting the campaign before it passes risks ~17,460 core-hours on numbers we cannot trust."
