#!/usr/bin/env python3
"""Precompute ESM-C embeddings for CASP14 decoy TARGETS, keyed by target id.

The v3 (ESM) decoy scorer looks up ``esm_h5[pdb_id]['esm'] -> (L, d_esm)`` where
``pdb_id`` is the CASP14 target (e.g. ``T1053``). protein_os asserts
``esm_full.shape[0] == coords.shape[0]``, so the embedding must align 1:1 with
each decoy's residues. All decoys of a target share ONE native sequence and
length (verified: every T1053 decoy = 580 residues), so we rebuild the target's
sequence from a representative decoy's bead csv -- the ``group_name`` column,
ordered by ``group_num`` -- and embed it once. This is exactly the sequence the
scorer pairs with every decoy of that target.

Reuses ``ESMEmbedder`` from precompute_esm.py (same model/dtype as training's
hhsuite_esm_v2_new.h5: esmc_600m, d_esm=1152).

    python data_prep_scripts/precompute_esm_casp14.py \
        --targets T1053 T1026 --decoy_set casp14 \
        --out /n/netscratch/atchan_lab/qzha/nnef_data/casp14_esm.h5
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_esm import ESMEmbedder  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECOY_ROOT = os.path.join(ROOT, 'nnef', 'data', 'decoys')


def target_native_seq(decoy_set, target):
    """Canonical 1-letter sequence for a target, from a MODAL-length decoy.

    Decoys of one target do NOT all share a length: most model the full target
    (e.g. T1026 -> 172 residues, 508/517 decoys) but a few are truncated
    (147/148/152...). The scorer asserts ``esm_full.shape[0] == coords.shape[0]``
    per decoy, so a single shared ESM array can only align to decoys of ONE
    length. We pick the MOST COMMON decoy length (the canonical target) so the
    ESM covers the large majority; the rare off-length decoys fall back to
    no-ESM (tolerated by score_target). Ordered by ``group_num`` so it aligns
    with the coords the scorer builds from the same rows.

    Returns ``(seq, src_filename, n_modal, n_total)``.
    """
    d = os.path.join(DECOY_ROOT, decoy_set, target)
    beads = sorted(f for f in os.listdir(d) if f.endswith('_bead.csv'))
    if not beads:
        raise FileNotFoundError(f'no *_bead.csv under {d}')
    # Cheap length count (lines - header) per decoy; find the modal length.
    lengths = {}
    for b in beads:
        with open(os.path.join(d, b)) as fh:
            n = sum(1 for _ in fh) - 1
        lengths.setdefault(n, []).append(b)
    modal_len = max(lengths, key=lambda k: len(lengths[k]))
    src = sorted(lengths[modal_len])[0]
    df = pd.read_csv(os.path.join(d, src)).sort_values('group_num')
    seq = ''.join(str(x).strip() for x in df['group_name'])
    return seq, src, len(lengths[modal_len]), len(beads)


def native_seq_from_bead(bead_path):
    """1-letter sequence from a single native bead csv (group_name by group_num).

    Used for MD targets (yang_small), where each target IS one native structure
    at ``<root>/<target>/<target>_bead.csv`` -- there is no decoy ensemble, so
    the modal-length logic does not apply; the native's own length is canonical.
    """
    df = pd.read_csv(bead_path).sort_values('group_num')
    return ''.join(str(x).strip() for x in df['group_name'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', required=True)
    ap.add_argument('--decoy_set', default='casp14')
    ap.add_argument('--native_bead_root', default=None,
                    help='if set, read <root>/<target>/<target>_bead.csv as the '
                         'native (MD mode) instead of the decoy ensemble')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='esmc_600m')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    a = ap.parse_args()

    dtype = torch.float16 if a.dtype == 'float16' else torch.float32
    emb = ESMEmbedder(a.model, a.device, dtype)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with h5py.File(a.out, 'a') as out:
        for t in a.targets:
            if a.native_bead_root:
                bead = os.path.join(a.native_bead_root, t, f'{t}_bead.csv')
                seq = native_seq_from_bead(bead)
                note = f'native L={len(seq)} (src={bead})'
            else:
                seq, src, n_modal, n_total = target_native_seq(a.decoy_set, t)
                note = f'L={len(seq)} covers {n_modal}/{n_total} decoys (src={src})'
            e = emb.embed(seq)['esm']  # (L, d_esm) float16
            assert e.shape[0] == len(seq), \
                f'{t}: emb L={e.shape[0]} != seq L={len(seq)}'
            if t in out:
                del out[t]
            g = out.create_group(t)
            g.create_dataset('esm', data=np.asarray(e), compression='gzip',
                             chunks=(min(e.shape[0], 64), e.shape[1]))
            print(f'[casp14_esm] {t}: d={e.shape[1]} {note}', flush=True)
    print(f'[casp14_esm] wrote {a.out}')


if __name__ == '__main__':
    main()
