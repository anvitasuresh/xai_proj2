# Running the GPU stages on Kaggle (free)

Kaggle gives about 30 free GPU-hours a week on **2x T4** (16 GB each), with sessions up to 12 hours. That is enough for stages 1-2 (roughly 4-6 T4-hours). The `t4` profile (`configs/t4/`) runs the same models and rubric as the main configs, in float16 and split across both GPUs.

## One-time setup
1. Make a Kaggle account and **verify your phone number**. Without it, notebooks have no internet access.
2. On your laptop, run `bash scripts/run_stage.sh items` and `variants`, then upload `data/processed/` as a private **Kaggle Dataset** (named e.g. `tutorbias-data`). The GPU steps read it from there, so variants aren't rebuilt on GPU time.

## Notebook
New notebook, Settings: **Accelerator = GPU T4 x2**, **Internet = on**. Add your dataset as input. Cells:

```bash
%%bash
cd /kaggle/working
git clone https://github.com/anvitasuresh/xai_proj2 && cd xai_proj2
mkdir -p data && cp -r /kaggle/input/tutorbias-data/processed data/      # adjust to your dataset name
export VENV_ROOT=/tmp HF_HOME=/tmp/hf                                     # big files outside the 20 GB working dir
bash scripts/setup_local.sh gpu
```

```bash
%%bash
cd /kaggle/working/xai_proj2 && export HF_HOME=/tmp/hf
PROFILE=t4 bash scripts/run_stage.sh replies    # ~1-1.5 h
PROFILE=t4 bash scripts/run_stage.sh judge      # ~10 min
```

Later, after labeling and the E1 check (upload `data/labels/` to the dataset first):

```bash
%%bash
cd /kaggle/working/xai_proj2 && export HF_HOME=/tmp/hf
PROFILE=t4 bash scripts/run_stage.sh audit      # ~3-4 h
```

## Keeping outputs
Files in `/kaggle/working` disappear when an interactive session ends. Either download `data/` and `results/` before you stop, or run long steps with **Save Version -> Save & Run All**, which runs in the background and keeps `/kaggle/working` as the notebook's output. Every GPU step is resumable: if a session stops early, copy its partial `results/` back in and rerun the same command.

## Notes
- T4s don't support bfloat16, so the t4 profile uses float16. The judge model and rubric are unchanged, but report "fp16 on T4" in the README if the final numbers come from Kaggle.
- Reward models are not quantized: a 4-bit RM would be a different model from the one we say we audit.
