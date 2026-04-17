"""CPU smoke tests for nnef/data_prep_scripts/local_extractor_v2.py.

Covers:
  * ``compute_local_frame``: correct axes and rigid-body invariance.
  * ``extract_blocks_v2``: both terminal and interior blocks, with the right
    shape, residue count, distance semantics.
  * End-to-end ``build_h5_from_bead_csvs``: result h5 has the schema the
    training-side ``DatasetLocalGenCM`` reader expects.
"""
from __future__ import annotations

import os
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

# Import path gymnastics: run from anywhere.
_THIS = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_THIS)
_REPO = os.path.dirname(_PKG)
for p in (_REPO, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_prep_scripts.local_extractor_v2 import (  # noqa: E402
    _BEAD_V2_COLS,
    build_h5_from_bead_csvs,
    compute_local_frame,
    extract_blocks_v2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_chain(n_res: int = 30, seed: int = 0,
                         fold: str = 'helix') -> pd.DataFrame:
    """Build a toy chain where backbone atoms follow a deterministic but
    non-degenerate pattern. The ``helix`` layout keeps nearest neighbors
    within ~6 A, so the default 20 A block cutoff accepts all blocks.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Parametric helix around the z-axis with 3.6 residues / turn, pitch
    # ~5.4 A -> comfortably < 20 A for the 15 nearest residues.
    for i in range(n_res):
        if fold == 'helix':
            theta = 2 * np.pi * i / 3.6
            ca = np.array([2.3 * np.cos(theta),
                           2.3 * np.sin(theta),
                           i * 1.5]) + rng.normal(scale=0.05, size=3)
        else:
            ca = np.array([i * 3.8, 0.0, 0.0]) + rng.normal(scale=0.2, size=3)
        n = ca + np.array([-1.2, 0.5, 0.0]) + rng.normal(scale=0.05, size=3)
        c = ca + np.array([1.1, -0.3, 0.4]) + rng.normal(scale=0.05, size=3)
        cb = ca + np.array([0.0, 0.8, 0.8]) + rng.normal(scale=0.05, size=3)
        rows.append((
            'A', i + 1, 'A',
            n[0], n[1], n[2],
            ca[0], ca[1], ca[2],
            c[0], c[1], c[2],
            cb[0], cb[1], cb[2],
        ))
    return pd.DataFrame(rows, columns=_BEAD_V2_COLS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_frame_axes_and_invariance():
    n  = np.array([0.0, 1.0, 0.0])
    ca = np.array([1.0, 1.0, 0.0])
    c  = np.array([2.0, 2.0, 0.0])

    R = compute_local_frame(n, ca, c)
    assert R is not None, 'frame should be defined'

    # x-axis points from CA to N, y lies in the CA-N-C plane, z is orthogonal
    # to both and unit-length. Verify the canonical relations.
    x_hat, y_hat, z_hat = R[0], R[1], R[2]
    assert np.allclose(np.linalg.norm(x_hat), 1.0)
    assert np.allclose(np.linalg.norm(y_hat), 1.0)
    assert np.allclose(np.linalg.norm(z_hat), 1.0)
    assert abs(np.dot(x_hat, y_hat)) < 1e-6
    assert abs(np.dot(x_hat, z_hat)) < 1e-6
    assert abs(np.dot(y_hat, z_hat)) < 1e-6

    # CA->N in the local frame should be +x.
    assert np.allclose(R @ (n - ca), np.array([1.0, 0.0, 0.0]), atol=1e-6)

    # Rigid-body invariance: rotate + translate the whole residue, the local
    # frame-expressed position of C must be unchanged.
    rot = np.array([[0.0, -1.0, 0.0],
                    [1.0,  0.0, 0.0],
                    [0.0,  0.0, 1.0]])
    t = np.array([7.0, -3.0, 2.0])
    n2, ca2, c2 = rot @ n + t, rot @ ca + t, rot @ c + t
    R2 = compute_local_frame(n2, ca2, c2)
    assert np.allclose(R @ (c - ca), R2 @ (c2 - ca2), atol=1e-6)

    print('[ok] compute_local_frame axes + rigid-body invariance')


def test_frame_degenerate():
    # Collinear N, CA, C -> y axis construction fails.
    n  = np.array([0.0, 0.0, 0.0])
    ca = np.array([1.0, 0.0, 0.0])
    c  = np.array([2.0, 0.0, 0.0])
    assert compute_local_frame(n, ca, c) is None
    print('[ok] degenerate frame returns None')


def test_extract_blocks_shapes_and_segments():
    df_beads = make_synthetic_chain(n_res=30)
    df_blocks = extract_blocks_v2(df_beads, k=10)
    assert df_blocks is not None
    assert len(df_blocks) % 15 == 0
    num_blocks = len(df_blocks) // 15

    centers = df_blocks['center_num'].values.reshape(num_blocks, 15)[:, 0]
    assert len(np.unique(centers)) == num_blocks

    # At least one block should be terminal (gc in {1,2,29,30}) and at least
    # one should be interior (gc in {3..28}).
    terminal_centers = set(centers) & {1, 2, 29, 30}
    interior_centers = set(centers) - {1, 2, 29, 30}
    assert terminal_centers, f'expected terminal centers, got {sorted(set(centers))}'
    assert interior_centers, f'expected interior centers, got {sorted(set(centers))}'

    # Interior blocks: positions 0..4 must form a strict peptide window
    # {gc-2..gc+2}. Pick the first interior block we see.
    for bi in range(num_blocks):
        block = df_blocks.iloc[bi * 15: (bi + 1) * 15]
        if block['center_num'].iloc[0] in terminal_centers:
            continue
        gc = int(block['center_num'].iloc[0])
        window = set(block.iloc[:5]['group_num'].tolist())
        assert window == {gc - 2, gc - 1, gc, gc + 1, gc + 2}, window
        break

    # Every block has the 'seg' column populated, and segments for chain-
    # contiguous residues (within positions 5..14) share an id.
    assert 'seg' in df_blocks.columns
    assert df_blocks['seg'].min() >= 1

    # All 15 local_{x,y,z} must be finite.
    assert np.isfinite(df_blocks[['local_x', 'local_y', 'local_z']].values).all()

    print(f'[ok] extract_blocks_v2: {num_blocks} blocks, '
          f'{len(terminal_centers)} terminal / {len(interior_centers)} interior')


def test_build_h5_and_schema():
    """Round-trip: build a tiny h5 and verify the dataset keys + shapes the
    training-side loader expects are present and consistent."""
    with tempfile.TemporaryDirectory() as tmp:
        bead_dir = os.path.join(tmp, 'beads')
        chim_dir = os.path.join(tmp, 'chim')
        os.makedirs(bead_dir)
        os.makedirs(chim_dir)

        pdb_id = 'TEST_A'
        df_beads = make_synthetic_chain(n_res=30)
        df_beads.to_csv(os.path.join(bead_dir, 'TEST_bead.csv'), index=False)
        # Chimeric file the pipeline uses only for a length-sanity check.
        pd.DataFrame({'seq': ['A' * 30]}).to_csv(
            os.path.join(chim_dir, f'{pdb_id}.chimeric'), index=False,
        )

        out_h5 = os.path.join(tmp, 'blocks.h5')
        pdb_list = os.path.join(tmp, 'pdbs.csv')
        build_h5_from_bead_csvs(
            bead_dir=bead_dir,
            chim_dir=chim_dir,
            out_h5=out_h5,
            pdb_list_csv=pdb_list,
            amino_acids_csv=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'amino_acids.csv',
            ),
        )

        assert os.path.exists(out_h5)
        with h5py.File(out_h5, 'r') as f:
            assert pdb_id in f, f'pdb id missing in h5: {list(f.keys())}'
            grp = f[pdb_id]
            for key in ('seq', 'group_num', 'coords', 'seg', 'start_id', 'res_counts'):
                assert key in grp, f'missing dataset: {key}'
            nb = grp['coords'].shape[0]
            assert grp['coords'].shape == (nb, 15, 3)
            assert grp['seq'].shape == (nb, 15)
            assert grp['group_num'].shape == (nb, 15)
            assert grp['seg'].shape == (nb, 15)
            assert grp['start_id'].shape == (nb, 15)
            assert grp['res_counts'].shape == (nb, 3)

        df_pdbs = pd.read_csv(pdb_list)
        assert pdb_id in df_pdbs['pdb'].values

    print('[ok] build_h5_from_bead_csvs schema + pdb list')


if __name__ == '__main__':
    test_frame_axes_and_invariance()
    test_frame_degenerate()
    test_extract_blocks_shapes_and_segments()
    test_build_h5_and_schema()
    print('\nall local_extractor_v2 smoke tests passed.')
