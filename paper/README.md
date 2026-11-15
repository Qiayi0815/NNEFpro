# Paper release: RIBBON, an energy function for peptides

This folder holds the weights, lists, preregistration, run commands, structures and tables behind the paper.
The code is in the rest of the repository (`nnef/`, `eval/`, `fasrc/`). Paths inside command files are relative
to the repository root on the cluster where the runs were made; absolute cluster paths (`/n/home03/...`,
`/n/holylabs/...`) are left as they were run.

Final model: RIBBON, run `cbrib_k10_s42_56982226` (seed 42, trained on ideal Cβ).
"Original NNEF" is the published checkpoint `params/exp1` (Yang et al., 2022), already in this repository.

## Quick start

    source paper/weights/ribbon_seed42/flags.sh      # sets $FLAGS and NNEF_CB=ideal
    # flags.sh points --load_exp at runs/cbrib_k10_s42_56982226; to use the copy in this folder:
    FLAGS="${FLAGS/runs\/cbrib_k10_s42_56982226/paper/weights/ribbon_seed42}"

`--load_exp DIR` reads `DIR/models/model.pt`. Peptides shorter than 15 residues cannot be scored.
The command files read structures from `nnef/data/decoys/<set>/<ID>/<ID>_native_bead.csv`; the same files
are under `paper/structures/<set>/<ID>/`.

## Contents

| item | what it is | supports |
|---|---|---|
| `weights/ribbon_seed42/models/model.pt` | RIBBON weights, last epoch (200) of run `cbrib_k10_s42_56982226` | all figures |
| `weights/ribbon_seed42/flags.sh` | model and inference flags needed to load the checkpoint; sets `NNEF_CB=ideal` | all figures |
| `weights/ribbon_seed42/train_command.txt` | training command lines as run (line 1 = the paper model; lines 2–3 = other runs submitted together); calls `fasrc/train_ribbon.slurm` | Methods, training |
| `lists/training/hhsuite_CB_v2_k10_pdb_list_holdout.csv` | the 28,264 training chains (chain, sampling weight, length) | Methods, training |
| `lists/training/hhsuite_CB_v2_k10_pdb_list.csv` | the 28,440 chains before the holdout was removed | Methods, training |
| `lists/training/holdout_exclude_ids.txt` | 236 PDB codes whose chains were removed from training (3DRobot targets and other benchmark proteins) | Methods, training |
| `lists/pep_nmr/targets_summary.csv` | the 537 curated NMR peptides: length, conformer used (`target_model`), class, sequence, homolog flag | Figs 2, 3 |
| `lists/pep_nmr/curation_all588.csv` | all 588 entries of the source table with the reason each was dropped | Methods |
| `lists/pep_nmr/seq_clusters_id40.tsv` | MMseqs2 clusters at 40% identity (representative, member) | Methods |
| `lists/pep_nmr/dev206_targets.txt` | the 206 development peptides | Figs 2, 5 (rule derivation) |
| `lists/pep_new/pep_new_reps.txt` | the 136 held-out peptides | Figs 2, 5 |
| `lists/pep_new/pep_new_reps_ge15.txt` | the 113 of them with at least 15 residues (derived from `targets_summary.csv`, length ≥ 15) | Figs 2, 5 |
| `lists/pep_new/targets_summary.csv`, `curation_all.csv`, `seq_clusters_id40.tsv`, `pep_nmr_similarity.csv` | curation of the held-out set: 344 → 311 → 213 → 136 | Methods |
| `lists/si_18_peptides.csv` | the 18 peptides with finite-temperature runs (three in Fig 4, the rest in the SI) | Fig 4, SI |
| `lists/3drobot/targets_3drobot.txt` | 3DRobot target list as used on the cluster (200 chains; the first line `1A0A` is not a 3DRobot chain) | Figs 2, 3e |
| `lists/3drobot/excluded_from_t0_chain_break.txt` | the 4 proteins left out of the T = 0 test (chain break) | Fig 2 |
| `preregistration/PREREG_pep_new.md` | preregistration of 27 October 2026 with its amendment, unmodified | Fig 5 |
| `commands/singleT/` | Monte Carlo runs with RIBBON: one line per run (`cmds.txt`), wrapper (`one.sh`), array job (`run.slurm`) | Fig 4, SI |
| `commands/singleT_null/` | the same for the E = 0 and Ramachandran-only controls | Fig 4, SI |
| `commands/t0_peptides/` | T = 0 minimization of peptides. `cmds.txt` holds three checkpoints; the paper model is the lines with `--load_exp runs/cbrib_k10_s42_56982226` (`cb42_dev`, 206 peptides; `cb42_new`, held-out) | Fig 2 |
| `commands/t0_peptides_original_nnef/` | T = 0 minimization of the 206 peptides with the original NNEF | Fig 2 |
| `commands/t0_proteins/` | T = 0 minimization of the 196 3DRobot proteins, both models | Fig 2 |
| `commands/shuffle/one.sh`, `analyze1.slurm` | sequence-shuffle scoring wrapper and the summary step (the lines that called `one.sh` were not saved) | Fig 3 |
| `commands/scoring/submit.sh`, `remc_pack.slurm` | submission script for decoy scoring of pep_nmr, the AlphaFold2 models and 3DRobot (`cb42_w10`, `cb42_w10_c1000`; uses `fasrc/pep_score_cpu.sh`) and the job script it calls for packed runs | Figs 3, 5 |
| `commands/af2_select/` | selection analysis scripts as run (`eval/decoys/af2_select.py`, `prereg_pep_new.py`). The command line that wrote `af2_select_pep_new_cb42` was not saved; its log shows `--set pep_new_af2` with models `cb42_w10_c1000` and `rib_k10_smooth_w10_c1000` | Fig 5 |
| `commands/amber/` | Amber command lists and the submission chain | Fig 4 |
| `amber/amber_pep_md.py`, `amber_pack.slurm` | Amber ff14SB + GBn2 simulation script (OpenMM) and its job script | Fig 4 |
| `analysis/t0_counts.py`, `t0_clash.py`, `nmr_bundle.py` | T = 0 counts and per-position deviation; interpenetration test; NMR ensemble spread. `nmr_bundle.py` imports `kabsch` from a plotting helper that is not part of this release | Fig 2 |
| `structures/pep_nmr/<ID>/<ID>_native_bead.csv` | NMR conformer (N, Cα, C, Cβ per residue) of the 522 peptides with at least 15 residues | Figs 2–4 |
| `structures/pep_new/<ID>/<ID>_native_bead.csv` | NMR conformer of the 136 held-out peptides | Figs 2, 5 |
| `structures/3drobot_native/<ID>/native_bead.csv` | native structures of the 200 3DRobot proteins | Fig 2 |
| `structures/t0_minimized/ribbon/`, `original_nnef/` | minimized structures of the 206 development peptides: `final_bb.npy` (N, Cα, C, Cβ), `meta.json`, energy and RMSD trace | Fig 2 |
| `structures/t0_minimized/ribbon_heldout/` | minimized structures of the 112 held-out peptides (RIBBON) | Fig 2 |
| `structures/fig4_trajectories/` | Cα trajectories of 2N6M, 1E0N and 2RT4 (see below) | Fig 4 |
| `tables/fig2_t0/` | `counts.csv`, `per_target.csv`, `per_position.csv` (model `final` = RIBBON, `exp1` = original NNEF; sets `peptide_dev`, `peptide_heldout`, `protein`) | Fig 2 |
| `tables/fig2_nmr_bundle/` | NMR ensemble spread per peptide and per position; `t0_clash.csv` = closest non-neighbour distances | Fig 2 |
| `tables/fig3/fig3d_helix_set.csv`, `analysis/fig3d_check.py` | per-peptide check of the ideal-helix test: whether the ideal helix entered RIBBON's comparison (`helix_in`, `ncommon`) and whether it was the lowest-energy candidate; the 194 peptides of Fig 3d are the non-helical ones with L ≥ 15, `helix_in` true and `ncommon` > 0 | Fig 3d |
| `tables/fig3/shuffle_per_target.csv` | sequence-shuffle test per peptide (model `cb42` = RIBBON) | Fig 3 |
| `tables/fig3/pep_compare_per_target.csv` | ranking per peptide for RIBBON (`cb42_w10`), RW+ and Ramachandran-only | Fig 3 |
| `tables/fig3/fig3d_check.csv` | helix test per peptide | Fig 3d |
| `tables/fig3/prot_per_target.csv` | energy–RMSD correlation per 3DRobot protein (`cb42_w10`, `yang2022`) | Fig 3e |
| `tables/fig5/af2_select_pep_new_cb42/` | AlphaFold2 model selection on the held-out set; `prereg_per_target.csv` is the preregistered test | Fig 5 |
| `tables/fig5/af2_select_pep_new_cb42_medoid/` | the same with GDT-TS against the medoid conformer | SI |

The tables also contain rows for other checkpoints and scorers that were evaluated alongside; the names above
mark the rows used in the paper.

## Fig 4 trajectories

`structures/fig4_trajectories/<method>/<peptide>/<run>/traj_ca.npz`, coordinates in Å.

- `ribbon_mc`: RIBBON Monte Carlo at kT = 1.739, seeds `native` (0) and `native_s1` (1). `ca` has shape
  (1, 2501, L, 3); `frame_sweep`; `native_ca`. Extracted from the full run files (5 chains at 5 kT, four beads
  per residue), keeping the chain and atom used in the figure.
- `zero_mc`, `rama_aa_mc`: the E = 0 and Ramachandran-only controls. `ca` has shape (2, 2501, L, 3) for
  kT = 1.0 and 1.739; the figure uses kT = 1.0.
- `amber`: Amber runs `s0`, `s1`, `s2`. `ca` has shape (frames, L, 3), one frame every 10 ps; frame 0 is the
  NMR conformer. The `s0` run of 1E0N ends at 45 ns and has no `meta.json`.

Each run folder keeps the `meta.json` written by the run.

## Not included

- Full Monte Carlo and Amber trajectories (all beads, all temperatures, the other 15 peptides): too large
  for the repository.
- The minimized structures of the 3DRobot proteins (their RMSD values are in `tables/fig2_t0/per_target.csv`).
- The AlphaFold2 models, the decoy sets and the training HDF5 files.
- 3DRobot decoys are available from the Zhang lab; the NMR entries from the PDB.
