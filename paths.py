"""Re-export shim so legacy ``from paths import data_path`` keeps working
regardless of CWD or whether the code is invoked as a module (``python -m
nnef.foo``) or a script (``cd nnef && python foo.py``). All actual logic
lives in :mod:`nnef.paths`.
"""
from nnef.paths import DATA_DIR, PACKAGE_DIR, REPO_ROOT, data_path, ensure_dir

__all__ = ['DATA_DIR', 'PACKAGE_DIR', 'REPO_ROOT', 'data_path', 'ensure_dir']
