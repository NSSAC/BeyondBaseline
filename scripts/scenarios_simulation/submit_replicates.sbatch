#!/bin/bash
#SBATCH --job-name=sim_replicates
#SBATCH -p bii
#SBATCH -A bii_nssac
#SBATCH -c 4                       # 4 cores is optimal
#SBATCH --mem=64G                  # 64GB is sufficient for the graph operations
#SBATCH -t 04:00:00
#SBATCH --array=0-19               # Runs replicates 0 through 19
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# ------------------ EDIT THESE PATHS ------------------
REPLICATES_DIR="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/linelist_generation/results"
POP_FILE="/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/va_persontrait_epihiper.txt"
OUTDIR_BASE="replicate_results"
# ------------------------------------------------------

echo "Activating Conda environment..."
set +u
source /project/biocomplexity/asw3xp/miniconda3/etc/profile.d/conda.sh
conda activate 
set -u

# Setup paths for THIS specific array task
RID="${SLURM_ARRAY_TASK_ID}"
REPL="replicate_${RID}"
REPDIR="${REPLICATES_DIR}/${REPL}"
OUTDIR="${OUTDIR_BASE}/${REPL}"

LINELIST_FILE="${REPDIR}/linelist.csv.xz"
INFECTIONS_FILE="${REPDIR}/linelist_allevents.csv.xz"
MUGRATION_FILE="${REPDIR}/linelist_mugration.json"  # <-- Ground truth file

# Sanity checks
if [[ ! -d "${REPDIR}" ]]; then
  echo "Directory ${REPDIR} does not exist. Exiting gracefully."
  exit 0
fi
if [[ ! -f "${LINELIST_FILE}" ]] || [[ ! -f "${INFECTIONS_FILE}" ]] || [[ ! -f "${MUGRATION_FILE}" ]]; then
  echo "ERROR: Missing linelist, infections, or mugration JSON file in ${REPDIR}" >&2
  exit 2
fi

mkdir -p "${OUTDIR}"

echo "--- Processing ${REPL} ---"

cmd=(
    "python3" "run_all_scenarios.py"
    "--linelist" "${LINELIST_FILE}"
    "--population" "${POP_FILE}"
    "--infections" "${INFECTIONS_FILE}"
    "--abm_mugration" "${MUGRATION_FILE}"  # <-- Added mugration parameter
    "--outdir" "${OUTDIR}"
    "--batch-size" "100"
    "--no-replacement"
    "--seed" "42"
    "--algorithms" "surs" "stratified" "LASSO-Greedy" "LASSO-Stratified"
    "--stratifiers" "age" "race" "county" "sex"
    "--save-samples"
    "--no-plots"
)

echo "Executing command:"
printf "  %q " "${cmd[@]}"
echo ""

"${cmd[@]}"

echo "--- Finished ${REPL} ---"
