"""Precompute per-residue ESM-C embeddings for every v2 chain.

Motivation
----------
NNEF-v3 (``--use_esm``) looks up a frozen per-residue language-model
embedding for every block during training and scoring. Running ESM inside
the training loop would be prohibitively slow (ESM-C 600M forward >>
NNEF forward), so we precompute once and cache to disk keyed by PDB4_C.

Model choice
------------
We use **ESM-C 600M** (EvolutionaryScale, 2024), not ESM2:

* ESM-C 600M matches or beats ESM2-3B on structure / function benchmarks
  at ~6x fewer params (better per-param -> better ablation story).
* Hidden dim ``1152`` (vs ESM2-650M's 1280) trims the cache by ~10%
  (~5.7 GB for 12.4k chains @ avg 200 res, float16).
* The ``esm`` package from EvolutionaryScale ships open weights for 300M
  and 600M; install with ``pip install esm``.

Output layout (HDF5)
--------------------
``out_h5[pdb_key]`` is a **group** with the following datasets:

* ``seq`` : 1-D ``S1`` byte array, length ``L``, the WT one-letter sequence.
* ``esm`` : ``(L, d_esm)`` ``float16``, per-residue embedding with BOS/EOS
  stripped so shape aligns with ``hhsuite_CB_v2.h5[pdb_key]['coords']``.
* Attributes: ``model`` (e.g. ``esmc_600m``), ``d_esm`` (1152), ``seq_source``
  (always ``hhsuite_pdb_seq_v2:row0`` today).

Any pre-existing keys in ``--out_h5`` are left alone (resume-safe) unless
``--overwrite`` is set. This script is designed to be ``sbatch``-able for
multi-hour runs on a single GPU.

Usage
-----
Local smoke test on CPU (5 chains, ~a couple minutes)::

    python nnef/data_prep_scripts/precompute_esm.py --limit 5 --device cpu \\
        --out_h5 nnef/data/hhsuite_esm_v2_smoke.h5

FASRC full run on one H200::

    sbatch fasrc/precompute_esm.slurm

(See ``fasrc/precompute_esm.slurm`` for the full CLI.)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

try:
    from nnef.paths import data_path
except ImportError:  # allow "python file.py" from the repo root
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
    from nnef.paths import data_path


# --------------------------------------------------------------------------- #
# AA decoding (inverse of build_seq_h5._ENC)                                  #
# --------------------------------------------------------------------------- #
def _load_idx2aa() -> List[str]:
    df = pd.read_csv(data_path('amino_acids.csv'))
    idx2aa = [''] * 20
    for _, r in df.iterrows():
        # CSV stores 1-based idx; h5 seq stores 0-based.
        idx2aa[int(r.idx) - 1] = str(r.AA)
    if any(a == '' for a in idx2aa):
        raise RuntimeError(f'amino_acids.csv is missing rows: {idx2aa!r}')
    return idx2aa


def _decode_wt(seq_row: np.ndarray, idx2aa: List[str]) -> str:
    """``(L,) int8`` -> one-letter string. Rejects anything out of [0, 19]."""
    arr = np.asarray(seq_row, dtype=np.int64)
    if arr.min() < 0 or arr.max() > 19:
        raise ValueError(f'seq index out of range: min={arr.min()} max={arr.max()}')
    return ''.join(idx2aa[i] for i in arr)


# --------------------------------------------------------------------------- #
# Model loader: keep ESM-C as the primary path, with a best-effort fallback   #
# to fair-esm ESM2 so this script is still useful if ``esm`` isn't available. #
# --------------------------------------------------------------------------- #
def _infer_esmc_width(client: torch.nn.Module) -> int:
    """Embedding width for ESM-C, compatible across `esm` versions."""
    ed = getattr(client, "embed_dim", None)
    if ed is not None:
        return int(ed)
    emb = getattr(client, "embed", None)
    if emb is not None:
        w = getattr(emb, "embedding_dim", None)
        if w is not None:
            return int(w)
        w = getattr(emb, "weight", None)
        if w is not None and w.ndim >= 2:
            return int(w.shape[1])
    raise AttributeError(
        "Cannot infer ESM-C embedding width: ESMC has no `embed_dim` and no "
        "`embed` table (upgrade/downgrade the `esm` package or report this)."
    )


class ESMEmbedder:
    """Thin wrapper over the model + tokenizer so the main loop is
    family-agnostic. ``embed(seq)`` returns ``(L, d_esm)`` float16 numpy.
    """

    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.family, self.client, self.d_esm = self._build(model_name)
        print(f'[precompute_esm] loaded {model_name} '
              f'(family={self.family}, d_esm={self.d_esm}, device={self.device}, '
              f'dtype={self.dtype})')

    def _build(self, model_name: str):
        if model_name.startswith('esmc_'):
            try:
                from esm.models.esmc import ESMC
            except ImportError as exc:
                raise ImportError(
                    f"ESM-C model '{model_name}' requested but the 'esm' package "
                    f"from EvolutionaryScale is not importable. Install with "
                    f"`pip install esm` (not `fair-esm`, which is a different "
                    f"package that only ships ESM1/2)."
                ) from exc
            client = ESMC.from_pretrained(model_name).to(self.device).to(self.dtype)
            client.eval()
            # Older `esm` exposed `embed_dim` on ESMC; current OSS ESMC only stores
            # width on the token embedding table (`nn.Embedding.embedding_dim`).
            d_esm = _infer_esmc_width(client)
            return 'esmc', client, d_esm

        if model_name.startswith('esm2_'):
            try:
                import esm as fair_esm  # fair-esm's namespace
            except ImportError as exc:
                raise ImportError(
                    f"ESM2 model '{model_name}' requested but 'fair-esm' is not "
                    f"importable. Install with `pip install fair-esm`."
                ) from exc
            loader = getattr(fair_esm.pretrained, model_name)
            model, alphabet = loader()
            model = model.to(self.device).to(self.dtype).eval()
            self._fair_alphabet = alphabet
            self._fair_batch_converter = alphabet.get_batch_converter()
            self._fair_repr_layer = model.num_layers
            d_esm = int(model.embed_dim)
            return 'esm2', model, d_esm

        raise ValueError(f'unknown model family: {model_name}')

    @torch.no_grad()
    def embed(self, seq: str) -> np.ndarray:
        """Return per-residue embedding ``(L, d_esm)`` float16, BOS/EOS stripped."""
        if self.family == 'esmc':
            from esm.sdk.api import ESMProtein, LogitsConfig
            protein = ESMProtein(sequence=seq)
            tok = self.client.encode(protein)
            # encode -> Tensor of shape (L+2,) with BOS/EOS already added.
            if tok.dim() == 1:
                tok = tok.unsqueeze(0)
            tok = tok.to(self.device)
            out = self.client.logits(
                tok, LogitsConfig(sequence=True, return_embeddings=True),
            )
            # out.embeddings: (1, L+2, d_esm)
            emb = out.embeddings[0, 1:-1]           # strip BOS / EOS
        else:  # esm2
            data = [('query', seq)]
            _, _, batch_tokens = self._fair_batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            out = self.client(
                batch_tokens,
                repr_layers=[self._fair_repr_layer],
                return_contacts=False,
            )
            # fair-esm prepends <cls> and appends <eos>
            emb = out['representations'][self._fair_repr_layer][0, 1:-1]

        if emb.shape[0] != len(seq):
            raise RuntimeError(
                f'embedding length {emb.shape[0]} != seq length {len(seq)} '
                f'for {self.model_name}'
            )
        return emb.to(torch.float16).cpu().numpy()


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def _iter_chains(seq_h5_path: str,
                 pdb_list_csv: Optional[str],
                 limit: Optional[int],
                 ) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield ``(pdb_key, seq_row_int8)`` in CSV order (if provided) or h5 order.

    ``seq_row_int8`` is ``seq_h5[pdb_key][0]`` -- the WT row of the chimeric MSA,
    guaranteed canonical 0..19 by ``build_seq_h5.py``.
    """
    with h5py.File(seq_h5_path, 'r') as seq_h5:
        if pdb_list_csv and os.path.isfile(pdb_list_csv):
            df = pd.read_csv(pdb_list_csv)
            col = 'pdb' if 'pdb' in df.columns else df.columns[0]
            keys: List[str] = list(df[col].astype(str))
        else:
            keys = list(seq_h5.keys())

        yielded = 0
        for key in keys:
            if key not in seq_h5:
                continue
            arr = seq_h5[key][0][...]   # (L,) int8
            yield key, arr
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _write_chain(out_h5: h5py.File, key: str, seq: str, emb: np.ndarray,
                 model_name: str) -> None:
    if key in out_h5:
        del out_h5[key]
    grp = out_h5.create_group(key)
    grp.create_dataset(
        'seq',
        data=np.frombuffer(seq.encode('ascii'), dtype='S1'),
    )
    grp.create_dataset(
        'esm',
        data=emb,
        dtype='float16',
        compression='gzip',
        compression_opts=4,
        shuffle=True,
        # Chunk = one residue-row; keeps per-block gather fast on training.
        chunks=(min(emb.shape[0], 64), emb.shape[1]),
    )
    grp.attrs['model'] = model_name
    grp.attrs['d_esm'] = int(emb.shape[1])
    grp.attrs['seq_source'] = 'hhsuite_pdb_seq_v2:row0'


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seq_h5', default=data_path('hhsuite_pdb_seq_v2.h5'),
                   help='Input: seq h5 with WT on row 0 per key.')
    p.add_argument('--pdb_list', default=data_path('hhsuite_CB_v2_pdb_list.csv'),
                   help='Optional CSV with a `pdb` column to restrict/order '
                        'the chain list. Defaults to the v2 pdb_list.')
    p.add_argument('--out_h5', default=data_path('hhsuite_esm_v2.h5'),
                   help='Output HDF5 cache path.')
    p.add_argument('--model', default='esmc_600m',
                   help='Model name. Primary: esmc_300m / esmc_600m. '
                        'Fallback: esm2_t33_650M_UR50D / esm2_t30_150M_UR50D.')
    p.add_argument('--device', default='cuda:0',
                   help='PyTorch device. Use cpu for smoke tests.')
    p.add_argument('--dtype', default='float16', choices=['float16', 'float32'],
                   help='Model compute dtype; embedding is always stored as '
                        'float16 on disk.')
    p.add_argument('--max_len', type=int, default=1024,
                   help='Skip chains longer than this to avoid PLM OOM.')
    p.add_argument('--limit', type=int, default=None,
                   help='Only process the first N chains (smoke tests).')
    p.add_argument('--flush_every', type=int, default=50,
                   help='HDF5 flush cadence (chains). Lower = more restart-safe, '
                        'higher = slightly faster.')
    p.add_argument('--overwrite', action='store_true', default=False,
                   help='Recompute keys even if they already exist in --out_h5.')
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    if args.device.startswith('cpu') and args.dtype == 'float16':
        print('[precompute_esm] CPU + float16 is unstable; forcing float32.')
        dtype = torch.float32

    idx2aa = _load_idx2aa()

    # Determine which keys we still need to compute.
    existing: set = set()
    mode = 'w' if (args.overwrite or not os.path.exists(args.out_h5)) else 'a'
    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5)), exist_ok=True)

    if mode == 'a' and os.path.exists(args.out_h5):
        with h5py.File(args.out_h5, 'r') as h:
            existing = set(h.keys())
        print(f'[precompute_esm] {len(existing)} chains already present in '
              f'{args.out_h5}; resuming.')

    embedder = ESMEmbedder(args.model, args.device, dtype)

    skip_stats = {'too_long': 0, 'already': 0, 'decode_error': 0,
                  'forward_error': 0, 'done': 0}
    t0 = time.time()

    with h5py.File(args.out_h5, mode) as out_h5:
        out_h5.attrs['model'] = args.model
        out_h5.attrs['d_esm'] = int(embedder.d_esm)

        it = _iter_chains(args.seq_h5, args.pdb_list, args.limit)
        for i, (key, seq_row) in enumerate(it, start=1):
            if key in existing and not args.overwrite:
                skip_stats['already'] += 1
                continue
            try:
                seq = _decode_wt(seq_row, idx2aa)
            except ValueError as exc:
                skip_stats['decode_error'] += 1
                print(f'  [{i}] skip {key}: decode error ({exc})')
                continue
            if len(seq) > args.max_len:
                skip_stats['too_long'] += 1
                continue

            try:
                emb = embedder.embed(seq)
            except Exception as exc:  # OOM or CUDA error - keep going on others
                skip_stats['forward_error'] += 1
                print(f'  [{i}] skip {key} (L={len(seq)}): forward error '
                      f'({type(exc).__name__}: {exc})')
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()
                continue

            _write_chain(out_h5, key, seq, emb, args.model)
            skip_stats['done'] += 1

            if skip_stats['done'] % args.flush_every == 0:
                out_h5.flush()
                rate = skip_stats['done'] / max(time.time() - t0, 1e-6)
                print(f'  [{i}] {key} L={len(seq)} done={skip_stats["done"]} '
                      f'rate={rate:.2f} chain/s '
                      f'(skip: {",".join(f"{k}={v}" for k, v in skip_stats.items() if k != "done" and v > 0) or "-"})')

        out_h5.flush()

    elapsed = time.time() - t0
    print(f'\n[precompute_esm] finished in {elapsed / 60:.1f} min')
    print(f'  written this run : {skip_stats["done"]}')
    print(f'  already cached   : {skip_stats["already"]}')
    print(f'  skipped          : too_long={skip_stats["too_long"]}, '
          f'decode_error={skip_stats["decode_error"]}, '
          f'forward_error={skip_stats["forward_error"]}')
    print(f'  cache            : {args.out_h5}')


if __name__ == '__main__':
    main()
