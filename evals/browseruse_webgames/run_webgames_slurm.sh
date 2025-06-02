#!/bin/bash
#SBATCH --job-name=webvoyager_browseruse
#SBATCH --output=webvoyager_browseruse_%j.out
#SBATCH --error=webvoyager_browseruse_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=1500G
#SBATCH --cpus-per-task=200
#SBATCH --nodelist=slurmus-a3nodeset-11

set -euxo pipefail

export PATH="/home/george_convergence_ai/.local/bin:$PATH"
export PATH="/opt/conda/bin:$PATH"

echo "Batch job started on $(hostname)"
srun echo "Running on $(hostname)"

echo "Running in $(pwd)"

rm -rf ./.venv

# Create scratch directory if it doesn't exist
SCRATCH_DIR="/scratch/george_convergence_ai/webvoyager_venv"
srun mkdir -p "$SCRATCH_DIR"

# Copy project files to scratch
srun cp -vr . "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# Create venv if it doesn't exist, otherwise use existing one
if [ ! -d "$SCRATCH_DIR/.venv" ]; then
    srun uv venv --python 3.11
else
    echo "Using existing venv in $SCRATCH_DIR"
fi


srun uv sync
# srun npx playwright install --with-deps chromium
srun uv run browseruse_webgames.py

echo "Job completed"
