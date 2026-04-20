# Running NNEF on FASRC / Cannon

Account: `qzha` — Home: `/n/home03/qzha` (100 GB, backed up).

Everything (conda env, repo, packaged `.h5` data, checkpoints) lives under
`$HOME` since the `atchan_lab` `holylabs` directory isn't writable to
users. Total usage for this project is <15 GB, well under quota.

**Data prep runs locally**, the cluster only does GPU training + ESM
precompute. You package raw bead/chimeric CSVs into `.h5` on your laptop
(via `local_extractor_v2.py`), ship the h5 up, and train.

## Files

| File                       | What it does                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| `env_setup.sh`             | One-time: create `$HOME/envs/nnef` conda env + install torch, h5py, esm/... |
| `train.slurm`              | **v2_cart_offset**: `--use_cart_coords --use_seq_offset` (no ESM).          |
| `train_v2_dihedral.slurm`  | **v2 + dihedral**: same as `train.slurm` + `--use_dihedral` (no ESM).       |
| `train_v1_pure.slurm`      | **v1_pure** baseline: no `--use_` flags, bit-identical to 2022 NNEF.        |
| `train_v1_esm.slurm`       | **v1 + ESM only**: same as v1_pure + `--use_esm` (no cart/offset/dihedral); needs `hhsuite_esm_v2.h5`. |
| `train_yang_style.slurm`   | Retrain **Yang-style** (`--mixture_seq 1 --legacy_local_frame`, no cart/ESM) on v2 HDF5 → `runs/yang_retrain_<JOBID>/`. |
| `train_v3_full.slurm`      | **v3_full** full stack: cart + offset + ESM-C 600M + phi/psi dihedral.      |
| `precompute_esm.slurm`     | One-off: ESM-C 600M per-residue cache -> `hhsuite_esm_v2.h5`.               |
| `sync_code_and_rama_v2_to_cluster.sh` | Laptop: rsync code + v2/rama h5 to FASRC (see script header).        |
| `sync_3drobot_decoys_to_cluster.sh`   | Laptop: rsync `nnef/data/decoys/3DRobot_set/` (beads + lists) to Cannon. |
| `sync_casp14_decoys_to_cluster.sh`    | Laptop: rsync `nnef/data/decoys/casp14/` (beads + lists; optional `CASP14_SRC=…`). |
| `sync_runs_to_cluster.sh`             | Laptop: push selected `runs/<exp>/` checkpoints when cluster lacks them. |
| `eval_3drobot_batch.slurm`            | Cannon: GPU batch job → `python nnef/scripts/batch_eval_3drobot.py --device cuda`. |
| `pull_models_from_cluster.sh`       | Laptop: pull `runs/<exp>/` or `--all-final` (every `models/model.pt`). |
| `pull_slurm_logs_from_cluster.sh`     | Laptop: pull `runs/slurm-<JOBID>.{out,err}` (training stdout; not inside `<exp>/`). |
| `pull_eval_from_cluster.sh`           | Laptop: pull `eval/` (summary.csv, plots); optional `--with-decoys`, `--with-slurm-logs`. |
| `pull_decoy_loss_from_cluster.sh`    | Laptop: pull only selected `nnef/data/decoys/<set>/decoy_loss_*` dirs (or `--all`). |
| `eval_casp14_all_local.sh`           | Laptop/GPU: **CASP14-only** re-eval; Yang (`legacy`/`mixture_rama 0`) vs v1–v3 flags (see script header). |
| `eval_v1v2v3_rama_casp14_only_local.sh` | Laptop: **six** CASP14-only evals (v1/v2/v3 + v1r/v2r/v3r) → `eval/<run>_casp14_only/`. |
| `submit_v3_casp14_only.sh`           | Cannon: **two** GPU jobs, v3 pair, **CASP14 only** (no 3DRobot) → `eval/<run>_casp14_only/`. |

## Ablation matrix

Three models over the same v2 dataset, optimizer, batch size, and epoch
budget. **Only the feature flags change**, so any delta attributes
cleanly to that feature.

| Model           | `--use_cart_coords` | `--use_seq_offset` | `--use_esm` | `--use_dihedral` | sbatch            |
|-----------------|:-------------------:|:------------------:|:-----------:|:----------------:|-------------------|
| `v1_pure`       |          -          |          -         |      -      |         -        | `train_v1_pure`   |
| `v2_cart_offset`|          x          |          x         |      -      |         -        | `train`           |
| `v2_dihedral`   |          x          |          x         |      -      |         x        | `train_v2_dihedral` |
| `v3_full`       |          x          |          x         |      x      |         x        | `train_v3_full`   |

## Workflow

### A. On your laptop — build the h5s and sync up

```bash
cd /Library/Camille/FYP/nnef

# 1. Build the v2 structure h5 locally (only when raw data / extractor changes).
python -u -m nnef.data_prep_scripts.local_extractor_v2 build-h5 \
    --bead_dir data_hh/bead_csvs --chim_dir data_hh/chimeric \
    --out_h5 nnef/data/hhsuite_CB_v2.h5 \
    --pdb_list nnef/data/hhsuite_CB_v2_pdb_list.csv \
    --k 10 --dist_cutoff 20.0 \
    --compression gzip --compression_level 4 \
    --num_workers 8 --min_identity 0.9 --min_coverage 0.5

# 1b. (Optional) Rebuild the chimeric seq h5 when the pdb_list changes.
python -m nnef.data_prep_scripts.build_seq_h5 \
    --chim_dir data_hh/chimeric \
    --out_h5  nnef/data/hhsuite_pdb_seq_v2.h5 \
    --pdb_list nnef/data/hhsuite_CB_v2_pdb_list.csv \
    --num_workers 8

# 1c. (Optional) Rebuild CASP decoy bead CSVs from raw TS/PDB files under
#     data_hh/<TARGET>/ (v2 N,CA,C,CB columns; overwrites nnef/data/decoys/...).
PYTHONPATH=nnef python nnef/data_prep_scripts/regenerate_decoy_beads.py \
 --decoy_set casp14 --pdb_root data_hh --targets T1053 --overwrite --num_workers 8

# 2. Sync source code (diff-only, skips binaries / caches / raw CSVs / pdfs).
rsync -avh --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='runs/' \
    --exclude='params/' \
    --exclude='nnef/data/*.h5' \
    --exclude='data_hh/' \
    --exclude='*.pdf' \
    ./ qzha@login.rc.fas.harvard.edu:/n/home03/qzha/nnef/

# 3. Sync the packaged h5s + small sidecar CSVs into $HOME/nnef_data.
#    Note: do NOT ship hhsuite_esm_v2.h5 from your laptop -- it's 5-6 GB
#    and we generate it ON the cluster via precompute_esm.slurm.
#    hhsuite_rama_v2.h5 must align 1:1 with hhsuite_CB_v2.h5; build with
#    `python -m nnef.data_prep_scripts.build_rama_h5_v2 build --cb_h5 ... --bead_dir ... --out_h5 ...`
#    then `verify` before rsync.
rsync -avh --progress \
    nnef/data/hhsuite_CB_v2.h5 \
    nnef/data/hhsuite_CB_v2_pdb_list.csv \
    nnef/data/hhsuite_pdb_seq_v2.h5 \
    nnef/data/hhsuite_rama_v2.h5 \
    qzha@login.rc.fas.harvard.edu:/n/home03/qzha/nnef_data/
```

Re-run step 2 whenever code changes. Step 1 + step 3 only when you want a
fresh dataset.

### A2. Laptop — pull checkpoints + training logs for TensorBoard / plots

Checkpoints and TensorBoard events live under `runs/<exp_id>/` on the cluster.
Training **Slurm** text logs are separate files at `runs/slurm-<JOBID>.{out,err}`
(use the training job’s numeric ID).

```bash
cd /path/to/nnef
chmod +x fasrc/pull_slurm_logs_from_cluster.sh

# One or more experiment folders (models/, events.out.tfevents.*, …)
# Original v1 / v2 (ablation baselines) + four “stacked” runs (rama v2 / v3 / …):
bash fasrc/pull_models_from_cluster.sh \
  v1_pure_6171704 v2_run_6160264 \
  v1_pure_rama_v2_6228201 \
  v2_dihedral_rama_v2_6228517 v3_full_6223467 v3_full_rama_v2_6229240

# Match the training JOBIDs for those runs (see cluster: ls ~/nnef/runs/slurm-*.out)
bash fasrc/pull_slurm_logs_from_cluster.sh \
  6171704 6160264 6228201 6228517 6223467 6229240

# Browse runs locally
tensorboard --logdir runs --port 6006

# Static PDFs from event files (after pulls land under runs/<exp>/)
# Default x-axis is epoch (global_step / steps_per_epoch); auto-infers steps/epoch or use --steps_per_epoch 1000
python nnef/scripts/plot_training_tensorboard.py \
  --runs runs/v1_pure_6171704 runs/v2_run_6160264 \
        runs/v1_pure_rama_v2_6228201 \
        runs/v2_dihedral_rama_v2_6228517 runs/v3_full_6223467 runs/v3_full_rama_v2_6229240 \
  --out_dir figures/training_compare
```

Slurm `.out` files are plain text: `grep 'coords_rama_loss:' runs/slurm-6229240.out`, etc.

### A3. Laptop — pull evaluation results (after GPU eval on Cannon)

```bash
cd /path/to/nnef
chmod +x fasrc/pull_eval_from_cluster.sh
# summary.csv + plots under eval/<run>_casp14_3drobot/
bash fasrc/pull_eval_from_cluster.sh
# Optional: GPU eval Slurm logs
bash fasrc/pull_eval_from_cluster.sh --with-slurm-logs

# Per-target decoy_loss CSVs (pick folders; smaller than full --with-decoys)
chmod +x fasrc/pull_decoy_loss_from_cluster.sh
bash fasrc/pull_decoy_loss_from_cluster.sh \
  casp14/decoy_loss_yang_exp1 \
  casp14/decoy_loss_v1_pure_rama_v2_6228201 \
  3DRobot_set/decoy_loss_v1_pure_rama_v2_6228201
# Entire tree: bash fasrc/pull_decoy_loss_from_cluster.sh --all
```

`v2_run_6160264` is **v2_cart_offset** (`train.slurm`). If you want **v2 + dihedral** without Rama instead, swap in `runs/v2_dihedral_6172503` and pull `v2_dihedral_6172503` / Slurm `6172503`.

### B. On FASRC — one-time environment setup

```bash
ssh qzha@login.rc.fas.harvard.edu

# Build the conda env inside a small compute allocation (NOT on login).
cd /n/home03/qzha/nnef
salloc -p test --time=1:00:00 --mem=8G --cpus-per-task=4
bash fasrc/env_setup.sh            # ~5-10 min
exit
```

`env_setup.sh` installs both `esm` (EvolutionaryScale; ESM-C 600M) and
`fair-esm` (Meta; ESM2 fallback). Only `esm` is needed for the v3 run.

### C2. Queue everything else while v2 is already running

If `train.slurm` (v2_cart_offset) is already **R** on one GPU, you can still
submit the other ablations; extra `gpu` jobs may **PD** until a second GPU is
free. ESM precompute uses **`gpu_h200`** and may start in parallel.

From `~/nnef` after syncing code:

```bash
bash fasrc/submit_remaining_ablations.sh
```

This submits **v1_pure**, **precompute_esm**, and **v3_full** (the last chained
`afterok` on the ESM job).

### C. On FASRC — train the three models

Each run writes `runs/<exp_id>/models/model.pt` **every epoch** (always the
latest weights) and, every **50** epochs (`--save_interval 50`), an additional
snapshot `model_epoch_0050.pt`, `model_epoch_0100.pt`, … (1-based epoch in the
filename). For decoy scoring, pass
`--load_checkpoint runs/<exp>/models/model_epoch_0100.pt` (overrides
`load_exp/models/model.pt`), or copy/symlink that file to `model.pt`.

```bash
cd /n/home03/qzha/nnef

# --- v1 pure baseline ----------------------------------------------------
sbatch fasrc/train_v1_pure.slurm
# -> runs/<v1_pure_JOBID>/models/model.pt  (+ model_epoch_*.pt every 50 epochs)

# --- v2_cart_offset (currently running, no resubmit needed) --------------
# sbatch fasrc/train.slurm
# -> runs/<v2_run_JOBID>/models/model.pt

# --- v3 ESM-C full stack -------------------------------------------------
# 1) Build the ESM cache on the cluster (one-off, ~1.5-3.5 h on H200).
ESM_JOB=$(sbatch --parsable fasrc/precompute_esm.slurm)
echo "ESM precompute job = $ESM_JOB"

# 2) Chain the training so it only starts once the cache is ready.
sbatch --dependency=afterok:$ESM_JOB fasrc/train_v3_full.slurm
# -> runs/<v3_full_JOBID>/models/model.pt
```

Monitor:
```bash
squeue -u qzha
tail -f runs/slurm-<JOBID>.out
```

### D. On FASRC — evaluate (three-way correlation on decoy sets)

Once all three training runs have a checkpoint, run the unified evaluator
from the login node (read-only; CPU or tiny GPU is enough):

```bash
cd /n/home03/qzha/nnef
source $HOME/envs/nnef/bin/activate

# Score one checkpoint across CASP14 + 3DRobot:
python nnef/scripts/evaluate_decoys.py \
    --load_exp runs/v1_pure_<JOBID> \
    --decoy_sets casp14,3DRobot_set \
    --exp_tag v1_pure \
    --out_dir runs/eval/v1_pure \
    --plot

# Repeat for v2 and v3_full (the v3 invocation also needs --use_esm +
# --esm_h5_path $DATA_DIR/hhsuite_esm_v2.h5 --use_dihedral).

# Cross-experiment summary table:
python nnef/scripts/evaluate_decoys.py \
    --compare_exps runs/eval/v1_pure,runs/eval/v2_cart_offset,runs/eval/v3_full \
    --out_dir runs/eval/compare
```

> **Note on `--use_dihedral` at inference.** The dihedral head reads phi/psi
> from `xn,yn,zn,xc,yc,zc` columns in the decoy bead CSV. Legacy decoy sets
> (CASP/3DRobot extracted with `utils.extract_beads` before this change)
> only carry CA + CB, so `load_protein_decoy` returns `dihedral_full=None`
> and the dihedral branch contributes zero — harmless but also no gain on
> that eval. To exercise the v3_full dihedral uplift, re-extract the decoy
> beads with the updated `extract_beads` (it now emits N and C columns).

### E. Laptop → Cannon: 3DRobot data + checkpoints, then batch eval on GPU

Apple MPS can crash (SIGBUS) on long 3DRobot jobs. Easiest path: **keep preparing
decoys locally**, **rsync the tree + code + weights**, and **score on an A100**
with CUDA.

**1) On your laptop** (repo root):

```bash
cd /Library/Camille/FYP/nnef

# Code + training h5s (if not already current on Cannon)
bash fasrc/sync_code_and_rama_v2_to_cluster.sh

# Unpacked 3DRobot targets under nnef/data/decoys/3DRobot_set/ (excludes decoy_loss_*)
bash fasrc/sync_3drobot_decoys_to_cluster.sh

# If those checkpoints live only under ./runs on the laptop, push them:
bash fasrc/sync_runs_to_cluster.sh
# or: bash fasrc/sync_runs_to_cluster.sh exp1 v1_pure_6171704
```

**2) On Cannon**:

```bash
ssh qzha@login.rc.fas.harvard.edu
cd /n/home03/qzha/nnef
sbatch fasrc/eval_3drobot_batch.slurm
squeue -u qzha
tail -f runs/slurm-3dr-eval-<JOBID>.out
```

Summaries appear under `~/nnef/eval/3dr_<run_name>/summary.csv`. Optional one-target
smoke before the full 200:

```bash
cd /n/home03/qzha/nnef
sbatch --export=ALL,EXTRA_ARGS='--targets 1C5EA' fasrc/eval_3drobot_batch.slurm
```

(`EXTRA_ARGS` is read inside the Slurm script and appended to
`batch_eval_3drobot.py`.)

### Yang + v1 + v2 (+ optional v3) on CASP14 + 3DRobot

`fasrc/eval_yang_v1_v2_casp14_3drobot.slurm` runs three checkpoints by default.
After **`train_v3_full.slurm`** completes, re-submit with **`V3_RUN`** set to the
new directory (e.g. `runs/v3_full_rama_v2_6229240`) so step (4) runs and the
merged `comparison.csv` includes v3. Requires **`$DATA_DIR/hhsuite_esm_v2.h5`**.

```bash
# Still training while squeue shows nnef-v3? Wait until model.pt exists.
ls runs/v3_full_rama_v2_<JOBID>/models/model.pt   # default train_v3_full.slurm exp_id
ls runs/v3_full_<JOBID>/models/model.pt            # e.g. no-Rama v3: v3_full_6223467

sbatch --export=ALL,V3_RUN=runs/v3_full_6223467,DATA_DIR=$HOME/nnef_data \
  fasrc/eval_yang_v1_v2_casp14_3drobot.slurm
```

### v1_pure + Rama v2 only (extra checkpoint, no clash with Yang+v1+v2 eval)

Outputs go to `eval/<run_basename>_casp14_3drobot/` and `decoy_loss_<run_basename>/`, not`v1_pure_ckpt`. To **avoid two GPU eval jobs at once**, chain after your main eval:

```bash
cd ~/nnef
sbatch --dependency=afterok:<MAIN_EVAL_JOBID> fasrc/eval_v1_pure_rama_v2_only.slurm
# e.g. sbatch --dependency=afterok:6259187 fasrc/eval_v1_pure_rama_v2_only.slurm
```

Or if the main eval has finished: `sbatch fasrc/eval_v1_pure_rama_v2_only.slurm`

### Parallel eval: v2_dihedral+Rama, v3 (no-rama name), v3_full_rama_v2

Three **independent** GPU jobs (different logs / `eval/<run>_casp14_3drobot/` — safe to submit together):

```bash
cd ~/nnef
# v2 dihedral + Rama v2 (default LOAD_EXP matches train_v2_dihedral.slurm naming)
sbatch fasrc/eval_v2_dihedral_rama_casp14_3drobot.slurm

# v3 without rama in folder name (edit JOBID if needed)
sbatch --export=ALL,LOAD_EXP=runs/v3_full_6223467,DATA_DIR=$HOME/nnef_data fasrc/eval_v3_full_casp14_3drobot.slurm

# v3 + Rama (default LOAD_EXP=runs/v3_full_rama_v2_6229240; override JOBID if needed)
# If the run folder was saved as v3_full_<JOBID>, rename once: mv runs/v3_full_6229240 runs/v3_full_rama_v2_6229240
sbatch --export=ALL,DATA_DIR=$HOME/nnef_data fasrc/eval_v3_full_rama_v2_casp14_3drobot.slurm
```

v3 jobs need ``$DATA_DIR/hhsuite_esm_v2.h5``. Shared driver: ``fasrc/eval_one_run_casp14_3drobot.sh``.

## Partition cheat-sheet

| Partition        | Use                                                 |
|------------------|-----------------------------------------------------|
| `test`           | < 1 h, up to 4 CPUs — env setup, smoke tests.       |
| `gpu_test`       | Short GPU debugging runs (MIG slice only).          |
| `gpu`            | Main GPU training (A100 40GB).                      |
| `gpu_h200`       | H200, faster for fp16 ESM forward (precompute).     |
| `gpu_requeue`    | Backfill GPU; cheaper but preemptible.              |

Run `sinfo -s` on the cluster to see which partitions your account can
actually submit to.

## Common pitfalls

- **Never run heavy compute on the login node.** Use `salloc` or `sbatch`.
- **CUDA mismatch.** If `torch.cuda.is_available()` is False in a GPU
  job, swap `cu121` for `cu118` in `env_setup.sh` and reinstall torch.
- **ESM package confusion.** `pip install esm` (EvolutionaryScale) and
  `pip install fair-esm` (Meta) share the `esm` namespace. `env_setup.sh`
  installs both; `precompute_esm.py` probes `esm.models.esmc.ESMC` to
  pick ESM-C, and falls back to `fair_esm.pretrained.*` for ESM2.
- **Home quota**. `du -sh /n/home03/qzha` occasionally; cached
  `~/.cache/huggingface/` from ESM-C weights is ~3 GB.
- **Module name drift.** FASRC renames modules periodically. If
  `module load python/3.10.12-fasrc01` fails, run `module avail python`
  and pick the newest version; update both `env_setup.sh` and the slurms.
- **v3 ESM cache is BIG** (~6 GB). Build it ON the cluster with
  `precompute_esm.slurm`, don't try to rsync it from your laptop.
