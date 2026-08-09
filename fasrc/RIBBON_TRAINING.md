# Training the `ribbon` model

`ribbon` = the modified NNEF (intra-residue N–CA–C coordinate frame + a
Ramachandran head) trained on a rebuilt dataset. It shares all core code with
the Yang baseline (`nnef/train_chimeric.py`, `nnef/model/local_ss.py`,
`nnef/protein_os.py`); only the data it consumes and three flags differ:

```
--mixture_rama 10 --coords_rama_loss_lamda 1   # Ramachandran head ON
(NO --legacy_local_frame)                       # v2 intra-residue frame
--grad_clip_norm 1.0                            # (Yang uses 0; both supported)
```

Just want to **use** a trained checkpoint (score structures / decoys)? You do not
need any of this — see [../TESTING.md](../TESTING.md) and
`checkpoints/ribbon/models/model.pt`.

## Pipeline (run in order, on a Slurm/GPU cluster)

Each step writes an `.h5` under `$NS` (a scratch dir; override the `NS`/`OUT_*`
env vars at the top of each slurm). All are FASRC-flavoured but the `python`
commands inside are portable.

| # | Step | Script | Produces |
|---|------|--------|----------|
| 1 | Download PDBs + extract Cβ beads | `nnef/data_prep_scripts/fetch_and_beads.py` | per-chain `*_bead.csv` |
| 2 | Structure h5 (v2 frame + 15-res blocks) | `fasrc/build_v2_h5.slurm` → `local_extractor_v2.py build-h5` | `hhsuite_CB_v2_new.h5` |
| 3 | Ramachandran (φ/ψ) h5, aligned 1:1 with step 2 | `fasrc/build_rama_h5.slurm` → `build_rama_h5_v2.py` | `hhsuite_rama_v2_new.h5` |
| 4 | Sequence h5 + Yang sampling weights | `fasrc/build_seq_and_weights.slurm` → `build_seq_h5.py`, `compute_weights.py` | `hhsuite_pdb_seq_yang.h5`, weighted `pdb_list.csv` |
| 5 | **Train** | `fasrc/train_ribbon.slurm` → `nnef/train_chimeric.py` | `runs/ribbon_<jobid>/models/model.pt` |

```bash
# 2–4 build the three h5 inputs (structure / rama / sequence+weights)
sbatch fasrc/build_v2_h5.slurm
sbatch fasrc/build_rama_h5.slurm         # after step 2
sbatch fasrc/build_seq_and_weights.slurm

# 5 train (EPOCHS defaults 200; set 1000 to match the paper budget)
sbatch --export=ALL,EPOCHS=1000 fasrc/train_ribbon.slurm
```

Overridable env: `EPOCHS`, `PDB_H5`, `RAMA_H5`, `SEQ_H5`, `PDB_LIST`,
`MIXTURE_R`, `MIXTURE_ANGLE`, `EXP_ID`, `NS`, `DATA_DIR`.

### Variants
- **Yang reproduction** (no rama, legacy frame): `fasrc/train_yang_repro.slurm`.
- **esm** (ribbon + ESM-C embedding): `fasrc/train_esm.slurm` (needs an ESM cache).
- **α/β-only** subset (Yang's radical partition): filter the pdb_list with
  `nnef/data_prep_scripts/filter_alpha_beta.py`, then pass it as `PDB_LIST`.

## Training data
The rebuilt h5 files are large and are **not** in the repo. To reproduce from
scratch, run steps 1–4 against a PISCES/CullPDB list; or contact the authors for
the prebuilt `.h5`. The released `checkpoints/` are ready to use without any data.
