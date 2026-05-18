"""Centralized data path resolver for the NNEF package.

Why this file exists
--------------------
Historically the code mixed two relative-path styles:

  * ``'data/amino_acids.csv'``       (works only when CWD == repo-root/nnef/)
  * ``'nnef/data/amino_acids.csv'``  (works only when CWD == repo-root/)

Both break the moment the user runs a script from a different directory.
Routing every read/write of bundled data through :func:`data_path` fixes
that: the returned path is absolute and pinned to this package's own
location on disk, so the CLI behaves identically regardless of CWD.

Typical use
-----------
    from paths import data_path

    amino_acids = pd.read_csv(data_path('amino_acids.csv'))
    h5 = h5py.File(data_path('decoys', decoy_set, f'{pdb_id}.h5'), 'r')

For output directories (training runs, fold trajectories, decoy scores)
you can either pass :func:`data_path` explicitly, or use :func:`ensure_dir`
which also creates intermediate parents.
"""
from __future__ import annotations

import os
from typing import Union

_StrPath = Union[str, os.PathLike]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../nnef/nnef
PACKAGE_DIR: str = _THIS_DIR
REPO_ROOT: str = os.path.dirname(_THIS_DIR)              # .../nnef
DATA_DIR: str = os.path.join(_THIS_DIR, 'data')          # .../nnef/nnef/data


def data_path(*parts: _StrPath) -> str:
    """Return an absolute path under ``nnef/data/<parts...>``.

    ``parts`` may be any number of path segments; empty calls return
    :data:`DATA_DIR` itself.
    """
    return os.path.join(DATA_DIR, *map(os.fspath, parts))


def ensure_dir(path: _StrPath) -> str:
    """``os.makedirs(path, exist_ok=True)`` and return ``path`` as ``str``."""
    p = os.fspath(path)
    os.makedirs(p, exist_ok=True)
    return p
