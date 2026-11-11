# Shared settings for every SLURM job. Edit the three paths once, then everything else uses them.
#
# DCC storage: /work/$USER is fast scratch but purged after ~75 days of no access, so keep the repo,
# conda envs, and anything you need for the poster in your group space (/hpc/group/<lab>/$USER),
# and use /work only for caches and big intermediate files.

export PROJECT_DIR="${PROJECT_DIR:-$HOME/xai_proj2}"          # where you cloned the repo
export CONDA_ENVS="${CONDA_ENVS:-/work/$USER/conda_envs}"     # conda env location
export HF_HOME="${HF_HOME:-/work/$USER/hf_cache}"             # model weights (7B is ~15 GB)

export WANDB_PROJECT="who-does-the-tutor-learn-from"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$PROJECT_DIR/logs"

# DCC provides conda through the module system; check the exact name with `module avail -i conda`.
module load Miniconda3 2>/dev/null || module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)"

activate_env() { conda activate "$CONDA_ENVS/$1"; }
cd "$PROJECT_DIR"
