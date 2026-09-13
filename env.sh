# Provenance record for the Section 6 campaign, written by 00_setup_native.sh on the cluster
# and sourced by every job. The cluster account name has been replaced by ${USER}; nothing else
# is altered, and the file still works when sourced. The lines that matter for reproducing the
# reported numbers are SKY130_LIB, SKY130_PDK_COMMIT and SKY130_NGSPICE_VERSION -- verify with
#     source env.sh && md5sum "$SKY130_LIB"      # expect 365ab743568de364c2214767735a89c6
# The pinned PDK ships two model libraries; this one is the standard binned set, not continuous.
# sourced by every job script -- written by 00_setup_native.sh on 2026-08-06T14:14:55+03:00
export SKY130_ROOT=/arf/scratch/${USER}/sky130-bgr
export SKY130_CODE=/arf/scratch/${USER}/sky130-bgr/code
export SKY130_LIB=/arf/scratch/${USER}/sky130-bgr/pdk/volare/sky130/versions/c6d73a35f524070e85faff4a6a9eef49553ebc2b/sky130A/libs.tech/combined/sky130.lib.spice
export SKY130_PDK_COMMIT=c6d73a35f524070e85faff4a6a9eef49553ebc2b
export SKY130_PY=/arf/scratch/${USER}/sky130-bgr/venv/bin/python
export NGSPICE_EXE=/arf/scratch/${USER}/sky130-bgr/opt/bin/ngspice
export PATH=/arf/scratch/${USER}/sky130-bgr/opt/bin:$PATH
# one core per task: a threaded BLAS would oversubscribe the single allocated CPU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LC_ALL=C
# provenance for section 6
export SKY130_NGSPICE_VERSION=46
export SKY130_BUILD_CC="gcc (GCC) 11.3.1 20221121 (Red Hat 11.3.1-4)"
