# Testing this energy function on your own structures

This is a fork of Yang, Xiong & Zonta's NNEF (2022) — see [README.md](README.md)
for the original paper. This fork adds:

- a retrain on a larger (28,446-chain) dataset,
- a Ramachandran head + v2 coordinate frame (**ribbon**),
- a frozen ESM-C sequence embedding adapter (**esm**),
- an ablation replacing the Gaussian angle/radial output distributions with
  circular (von Mises) and directional (von Mises–Fisher) ones.

This guide is the fast path to scoring **your own PDB structure(s)** with any
of these checkpoints — no cluster access, no training data, no evolutionary
profiles needed (the model reads raw sequence + coordinates).

## 1. Setup (~5 min)

```bash
conda create -n nnef-score python=3.10 -y
conda activate nnef-score
pip install -r requirements-scoring.txt
```

Checkpoints are already included in this repo under `checkpoints/<name>/models/model.pt`
(each ~8 MB) — nothing extra to download for the six checkpoints below.

## 2. Score a structure

```bash
python nnef/scripts/score_for_collaborator.py \
    --checkpoint yang \
    --input your_protein.pdb \
    --out_csv scores.csv
```

Prints the energy and writes `scores.csv` (columns: `structure,energy`). **Lower
energy = more native-like**, by construction (the model is trained as
`P ∝ exp(-E)`, so it assigns low energy to real/near-native structures and
higher energy to decoys).

Score a whole folder of structures (e.g. several decoys of one target) in one
call:

```bash
python nnef/scripts/score_for_collaborator.py \
    --checkpoint ribbon \
    --input path/to/decoys/ \
    --out_csv scores.csv
```

If you have known quality labels per structure (e.g. GDT_TS, TM-score — from
your own decoy set), pass them and the script will also report the
correlation, which is the standard way this project validates a checkpoint:

```bash
# labels.csv:  structure,label
python nnef/scripts/score_for_collaborator.py \
    --checkpoint yang --input path/to/decoys/ --out_csv scores.csv \
    --labels_csv labels.csv
```

A working checkpoint should show **energy and quality label negatively
correlated** (Pearson r around −0.8 to −0.9 is what this project's checkpoints
get on the CASP14 T1026 benchmark, decoy structures against GDT_TS).

## 3. Which checkpoint?

| `--checkpoint` | What it is | Notes |
|---|---|---|
| `yang` | Yang 2022 architecture, retrained on 28k chains, Gaussian output | closest to the original paper; use this as your baseline |
| `yang_vonmises` | same backbone, circular von Mises distribution for angles | ablation — see caveat below |
| `yang_vmf` | same backbone, von Mises–Fisher (spherical) direction distribution | ablation — see caveat below |
| `ribbon` | `yang` + v2 coordinate frame + Ramachandran head | our main modified model |
| `ribbon_vonmises` | `ribbon` + von Mises angles | ablation |
| `esm` | `ribbon` + frozen ESM-C 600M embedding | **extra setup needed, see §5** |

All six were trained on the same 28,446-chain dataset, 1000 epochs, and are
directly comparable to each other. (One checkpoint, `yang` with a von
Mises–Fisher + log-normal radial ablation, is deliberately **not** included
here — it was trained before a units bug fix in the radial-distance loss and
its numbers aren't trustworthy yet.)

**Caveat on the `*_vonmises`/`*_vmf` checkpoints:** in our own testing, they
fit the training data marginally better but were **less stable in long MD**
than the Gaussian baseline at the same sampler settings — plausibly because
each output distribution implies a different "effective temperature" for the
energy landscape (see `nnef/scripts/calibrate_teff.py` if you want to dig into
that), and we ran everything at one fixed sampler setting rather than
recalibrating it per checkpoint. Decoy-scoring correlation, which is what this
script does, was **not** noticeably different across the ablation — so this
caveat mainly matters if you go on to run MD/sampling with these checkpoints,
not for plain structure scoring.

## 4. Input format details

- Standard PDB format, one file per structure.
- Every residue needs backbone **N, CA, C** (+ **CB**, except glycine which
  uses CA). Residues missing any of these are silently dropped — the script
  prints a warning with the count, check it matches your expectation.
- Multi-chain files: all chains are read and scored together as one sequence
  (matches how the model was trained on single continuous chains — if you
  want to score chains independently, split the PDB first).
- No MSA / evolutionary profile needed — every checkpoint here was trained
  with `--seq_type residue` (sequence read directly, not via profile).

## 5. Using the `esm` checkpoint (optional, more setup)

The `esm` checkpoint needs a precomputed per-residue ESM-C 600M embedding for
your sequence, which `score_for_collaborator.py` does not compute on the fly.
If you want to test it:

1. `pip install fair-esm`
2. Precompute embeddings for your sequence(s) — see
   `nnef/data_prep_scripts/precompute_esm.py` for the reference implementation.
3. Pass the resulting `.h5` path via `ESM_H5_OVERRIDE`/`--esm_h5_path` — this
   isn't wired into `score_for_collaborator.py` yet; ping us if you want to
   use it and we'll extend the script rather than have you hand-build the CLI
   call (the ESM flags are error-prone to get right by hand — see the
   docstring at the top of `nnef/scripts/score_for_collaborator.py`).

For a first pass, `yang` and `ribbon` (no ESM) are the ones to start with.

## Questions / issues

This is active research code from an ongoing FYP, not a polished package —
if something breaks or a result looks off, that's useful signal, not
necessarily your mistake. Ping [contact info] rather than debugging silently;
several checkpoint/architecture mismatches in this project have been the kind
of bug that fails silently (wrong energy, no crash), so if a number looks
surprising, say so before assuming it's expected.
