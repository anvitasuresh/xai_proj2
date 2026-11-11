# Shared settings for every SLURM job on Duke DCC.
#
# Storage: home directories are small, and the GPU env + model weights need ~30 GB. Clone the repo into
# your group space (/hpc/group/<lab>/$USER) or /work/$USER (fast, but purged after ~75 days without
# access, so copy results you need for the poster somewhere permanent).
export HF_HOME="${HF_HOME:-/work/$USER/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/$USER/uv_cache}"
export WANDB_PROJECT="who-does-the-tutor-learn-from"
mkdir -p "$HF_HOME" logs
