"""Score protein decoys with a trained NNEF checkpoint.

Two entry points:

* ``score_target(pdb_id, decoy_set, decoy_loss_dir, args, device, energy_fn, ...)``
  -- importable helper that scores every decoy for one target and writes
  ``<pdb_id>_decoy_loss.csv`` next to the target's ``list.csv`` layout. Used
  both by this script's CLI and by ``nnef/scripts/evaluate_decoys.py``.

* CLI (``python nnef/decoy_score.py --decoy_set ...``) -- iterates over every
  target listed in the decoy set's ``pdb_no_missing_residue.csv`` (or
  ``no_missing_residue.txt`` for CASP11) and invokes ``score_target`` per target.
  Preserves the original single-file layout and behaviour. Existing CSVs are
  skipped so the script is restart-safe.

The ``--static_decoy`` branch is kept for historical compatibility but
references a ``DatasetLocalGenOS`` class that is no longer shipped in this
repo; use the default (dynamic) scoring path.
"""

import os
from torch.utils.data import DataLoader, SequentialSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional dev dependency
    class SummaryWriter:  # type: ignore
        def __init__(self, *a, **kw):
            pass

        def add_scalar(self, *a, **kw):
            pass

        def close(self):
            pass

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py

import options
from dataset.data_chimeric import DatasetLocalGenCM
from trainer.local_trainer import LocalGenTrainer
from protein_os import Protein
from model import LocalTransformer
from utils import load_protein_decoy, resolve_model_checkpoint_path
from physics.grad_minimizer import GradMinimizerCartesian
from utils import test_setup
from paths import data_path, ensure_dir


# --------------------------------------------------------------------------- #
# Helpers shared with nnef/scripts/evaluate_decoys.py                         #
# --------------------------------------------------------------------------- #

_SUPPORTED_DECOY_SETS = ('3DRobot_set', 'casp11', 'casp13', 'casp14')


def load_target_list(decoy_set):
    """Return the list of target pdb ids for ``decoy_set``.

    ``3DRobot_set`` / ``casp13`` / ``casp14`` ship a single-column CSV
    ``pdb_no_missing_residue.csv`` with header ``pdb``. ``casp11`` ships
    ``no_missing_residue.txt`` in the same format.
    """
    if decoy_set == '3DRobot_set':
        return pd.read_csv(data_path('decoys', decoy_set, 'pdb_no_missing_residue.csv'))['pdb'].values
    if decoy_set == 'casp11':
        return pd.read_csv(data_path('decoys', decoy_set, 'no_missing_residue.txt'))['pdb'].values
    if decoy_set in ('casp13', 'casp14'):
        return pd.read_csv(data_path('decoys', decoy_set, 'pdb_no_missing_residue.csv'))['pdb'].values
    raise ValueError(f'decoy_set must be one of {_SUPPORTED_DECOY_SETS}, got {decoy_set!r}')


def _load_decoy_list_df(pdb_id, decoy_set):
    """Load the per-target decoy manifest (``NAME`` column + optional
    quality metric column such as ``GDT_TS`` or ``RMSD``)."""
    if decoy_set == '3DRobot_set':
        return pd.read_csv(data_path('decoys', decoy_set, pdb_id, 'list.csv'))
    if decoy_set == 'casp11':
        return pd.read_csv(
            data_path('decoys', decoy_set, pdb_id, 'list.txt'),
            header=None, names=['NAME'],
        )
    if decoy_set in ('casp13', 'casp14'):
        return pd.read_csv(data_path('decoys', decoy_set, pdb_id, 'list.csv'))
    raise ValueError(f'decoy_set must be one of {_SUPPORTED_DECOY_SETS}, got {decoy_set!r}')


def _decoy_id_for(decoy_name, decoy_set):
    """Decoy filenames differ between sets: CASP{13,14} use bare model ids
    (``T1053TS004_1``) while 3DRobot / CASP11 append ``.pdb`` which we strip
    to get the ``{decoy_id}_bead.csv`` lookup key."""
    if decoy_set in ('casp13', 'casp14'):
        return decoy_name
    return decoy_name[:-4]


def _lookup_esm(esm_h5, pdb_id):
    """Return the per-residue ESM embedding for ``pdb_id`` as a numpy array of
    shape ``(L_chain, d_esm)``, or ``None`` if the key is not present.

    Supports both HDF5 layouts used by ``nnef/data_prep_scripts``:

    * ``esm_h5[pdb_id]`` is a **group** containing an ``esm`` dataset
      (preferred; matches ``precompute_esm.py``).
    * ``esm_h5[pdb_id]`` is the dataset itself (flat layout).
    """
    if pdb_id not in esm_h5:
        return None
    entry = esm_h5[pdb_id]
    if isinstance(entry, h5py.Group):
        if 'esm' not in entry:
            return None
        return entry['esm'][...]
    return entry[...]


def score_target(pdb_id, decoy_set, decoy_loss_dir, args, device, energy_fn,
                 trainer=None, skip_if_exists=True, esm_h5=None):
    """Score every decoy for ``pdb_id`` and write ``<pdb>_decoy_loss.csv``.

    Parameters
    ----------
    pdb_id : str
        Target identifier as it appears in the decoy-set directory.
    decoy_set : str
        One of ``3DRobot_set / casp11 / casp13 / casp14``.
    decoy_loss_dir : str
        Subdirectory under ``nnef/data/decoys/<decoy_set>/`` where the result
        csv is written (matches the CLI flag ``--decoy_loss_dir``).
    args : argparse.Namespace
        Full args namespace as produced by ``options.get_decoy_parser()``.
        Only the following fields are consulted directly here:
        ``mode, relax, relax_steps, static_decoy, batch_size``.
    device, energy_fn
        The scoring model + device, as returned by ``utils.test_setup``.
    trainer : LocalGenTrainer | None, optional
        Only used by the legacy ``--static_decoy`` path.
    skip_if_exists : bool, default True
        If the output csv already exists, load and return it without
        rescoring. This keeps the script restart-safe and lets
        ``evaluate_decoys.py`` reuse previously scored CSVs.
    esm_h5 : h5py.File | None, optional
        Pre-opened ESM embedding cache for v3 (``--use_esm``). When given,
        each decoy's ``Protein`` is constructed with
        ``esm_full=esm_h5[pdb_id]['esm'][...]`` (or the dataset directly if
        the entry is not a group). Per-chain ESM is shared across all decoys
        of the same target because they share the WT sequence. Missing
        entries are tolerated: Protein is built with ``esm_full=None``,
        which falls back to the baseline path (still bit-identical).

    Returns
    -------
    pd.DataFrame | None
        The per-decoy dataframe with an appended ``loss`` column, or ``None``
        if the load failed (e.g. manifest missing).
    """
    out_dir = data_path('decoys', decoy_set, decoy_loss_dir)
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, f'{pdb_id}_decoy_loss.csv')

    if skip_if_exists and os.path.exists(out_csv):
        try:
            cached = pd.read_csv(out_csv)
        except Exception as exc:  # noqa: BLE001
            print(
                f'[score_target] {pdb_id}: unreadable {out_csv} ({exc!r}), rescoring...',
                flush=True,
            )
        else:
            try:
                manifest = _load_decoy_list_df(pdb_id, decoy_set)
            except FileNotFoundError:
                manifest = None
            n_exp = len(manifest) if manifest is not None else None
            if (
                manifest is not None
                and 'loss' in cached.columns
                and len(cached) == n_exp
                and cached['loss'].notna().all()
            ):
                return cached
            if (
                manifest is not None
                and 'loss' in cached.columns
                and len(cached) == n_exp
                and cached['loss'].isna().any()
            ):
                n_bad = int(cached['loss'].isna().sum())
                print(
                    f'[score_target] {pdb_id}: reusing {out_csv} ({n_bad}/{len(cached)} '
                    f'NaN losses — often non-finite energy for some decoys; rescoring rarely '
                    f'fixes this. Use --no_skip_if_exists to force recomputation.',
                    flush=True,
                )
                return cached
            if (
                manifest is not None
                and len(cached) < n_exp
            ):
                print(
                    f'[score_target] {pdb_id}: {out_csv} has {len(cached)}/{n_exp} rows, '
                    f'rescoring...',
                    flush=True,
                )
            elif manifest is None or 'loss' not in cached.columns:
                print(
                    f'[score_target] {pdb_id}: bad cache shape/columns in {out_csv}, rescoring...',
                    flush=True,
                )

    try:
        df = _load_decoy_list_df(pdb_id, decoy_set)
    except FileNotFoundError as exc:
        print(f'[score_target] skip {pdb_id}: manifest missing ({exc})')
        return None
    decoy_list = df['NAME'].values

    loss_all = []

    if args.static_decoy:
        # Preserved for historical reasons; requires a DatasetLocalGenOS class
        # that is no longer shipped. Kept as-is so anyone reviving it sees the
        # original code path.
        if trainer is None:
            raise RuntimeError('static_decoy=True requires a LocalGenTrainer')
        for decoy in decoy_list:
            decoy_h5 = data_path(
                'decoys', decoy_set, pdb_id, f'{decoy[:-4]}_local_rot_CA.h5'
            )
            if not os.path.exists(decoy_h5):
                loss_all.append(999)
                continue
            test_data = h5py.File(decoy_h5, 'r')
            test_dataset = DatasetLocalGenOS(test_data, args)  # noqa: F821
            sampler = SequentialSampler(test_dataset)
            loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=sampler)
            for data in loader:
                loss_terms = trainer.step(data)
                loss = sum(t.item() for t in loss_terms)
                loss_all.append(loss)
    else:
        # ESM cache lookup is done once per target because every decoy of the
        # same target shares the WT sequence.
        n_decoys = len(decoy_list)
        print(
            f'[score_target] {pdb_id} ({decoy_set}): {n_decoys} decoys '
            f'(progress every 100)...',
            flush=True,
        )
        esm_full_np = _lookup_esm(esm_h5, pdb_id) if esm_h5 is not None else None
        for j, decoy in enumerate(decoy_list):
            decoy_id = _decoy_id_for(decoy, decoy_set)
            (seq, coords_native, profile, dihedral_full,
             n_xyz, ca_xyz, c_xyz, chain_group_num) = load_protein_decoy(
                pdb_id, decoy_id, args.mode, device, args
            )
            esm_full = None
            if esm_full_np is not None:
                if esm_full_np.shape[0] != coords_native.shape[0]:
                    # Defensive: a sequence/structure length mismatch would
                    # crash Protein.__init__. Log once per decoy and skip the
                    # ESM branch for this chain so scoring still runs.
                    print(
                        f'[score_target] {pdb_id}/{decoy_id}: esm length '
                        f'{esm_full_np.shape[0]} != coords length '
                        f'{coords_native.shape[0]}, falling back to no-ESM'
                    )
                else:
                    esm_full = torch.from_numpy(np.asarray(esm_full_np)).to(
                        device=device, dtype=torch.float32
                    )
            # Dihedral is only attached when the decoy bead CSV actually has
            # N/C atom columns (load_protein_decoy returns None otherwise).
            # ProteinBase.use_dihedral is still stamped by test_setup, so a
            # chain without dihedrals falls back to zero contribution; this
            # keeps v1/v2 checkpoints bit-identical and only ablates v3
            # dihedrals on decoy sets that have the required backbone data.
            protein = Protein(
                seq, coords_native, profile,
                esm_full=esm_full, dihedral_full=dihedral_full,
                n_coords=n_xyz, ca_coords=ca_xyz, c_coords=c_xyz,
                chain_group_num=chain_group_num,
            )
            energy = protein.get_energy(energy_fn).item()

            if args.relax:
                minimizer = GradMinimizerCartesian(
                    energy_fn, protein, num_steps=args.relax_steps,
                )
                minimizer.run()
                energy = minimizer.energy_best
            loss_all.append(energy)
            # Progress: one forward per decoy; full targets can take many minutes.
            _step = j + 1
            if _step % 100 == 0 or _step == n_decoys:
                print(
                    f'[score_target] {pdb_id}: {_step}/{n_decoys} decoys',
                    flush=True,
                )

    if len(loss_all) == 0:
        print(f'[score_target] {pdb_id}: empty decoy list')
        return None
    print(f'[score_target] {pdb_id}: {len(loss_all)} decoys, first loss = {loss_all[0]:.3f}')

    df['loss'] = np.array(loss_all)
    df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------- #
# CLI driver (preserves the original single-script behaviour)                 #
# --------------------------------------------------------------------------- #

def main():
    parser = options.get_decoy_parser()
    args = options.parse_args_and_arch(parser)

    if args.static_decoy:
        writer = SummaryWriter('./runs/test/')
        device = torch.device(args.device)
        model = LocalTransformer(args)
        _ckpt = resolve_model_checkpoint_path(args)
        model.load_state_dict(
            torch.load(_ckpt, map_location=torch.device('cpu'))
        )
        model.to(device)
        model.eval()
        trainer = LocalGenTrainer(writer, model, device, args)
        energy_fn = None
    else:
        device, _model, energy_fn, _proteinbase = test_setup(args)
        trainer = None

    decoy_set = args.decoy_set
    decoy_loss_dir = args.decoy_loss_dir

    pdb_list = load_target_list(decoy_set)

    # Open the ESM cache once for the whole CLI run. When --use_esm is off we
    # leave esm_h5=None so the baseline scoring path is untouched.
    esm_h5 = None
    if getattr(args, 'use_esm', False) and getattr(args, 'esm_h5_path', None):
        if os.path.exists(args.esm_h5_path):
            esm_h5 = h5py.File(args.esm_h5_path, 'r')
            print(f'[decoy_score] opened esm h5 {args.esm_h5_path} '
                  f'({len(esm_h5.keys())} entries)')
        else:
            print(f'[decoy_score] --use_esm set but {args.esm_h5_path} '
                  f'missing; falling back to no-ESM scoring')

    try:
        for pdb_id in tqdm(pdb_list):
            score_target(
                pdb_id, decoy_set, decoy_loss_dir, args, device, energy_fn,
                trainer=trainer, skip_if_exists=True, esm_h5=esm_h5,
            )
    finally:
        if esm_h5 is not None:
            esm_h5.close()


if __name__ == '__main__':
    main()
