#!/bin/bash -l
#SBATCH --job-name=gtcn_sweeps
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=a30-4.6gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gpus=1

set -euo pipefail

# -------- Paths --------
WORKDIR="/work/$USER/WiFi_Sensing_Projects/WiFi_RSS_RTT_Dataset_for_Model_Selection"
ENV_DIR="/work/$USER/conda_envs/WiMANS"
PY="$ENV_DIR/bin/python"

mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

# -------- Determinism (redundant with mainn.py, harmless) --------
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="42"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[INFO] Host      : $(hostname)"
echo "[INFO] SLURM JOB : ${SLURM_JOB_ID:-NA}"
echo "[INFO] GPU       : $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# set -x
# "$PY" -u main_causal_residual_ap.py
# set +x

# set -x
# "$PY" -u main_causal_residual_cpu_time.py
# set +x

set -x
"$PY" -u main_causal_residual_cpu_time_training_time.py
set +x


echo "[DONE] Finished main_causal_residual_cpu_time_training_time.py"
