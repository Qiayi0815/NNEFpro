"""Verification-only wrapper: run nnef/scripts/evaluate_decoys.py on decoy set pep_nmr
without editing decoy_score.py (registers the set in memory and returns NaN energy
for decoys where the v2 scorer drops every block). Same CLI as evaluate_decoys.py."""
import os, sys, runpy
sys.path.insert(0, os.path.expanduser("~/nnef/nnef"))
import decoy_score as ds
ds._SUPPORTED_DECOY_SETS = ds._SUPPORTED_DECOY_SETS + ("pep_nmr",)
import pandas as pd
from paths import data_path
_lt, _ld, _di = ds.load_target_list, ds._load_decoy_list_df, ds._decoy_id_for
ds.load_target_list = lambda s: pd.read_csv(data_path("decoys", s, "pdb_no_missing_residue.csv"))["pdb"].values if s == "pep_nmr" else _lt(s)
ds._load_decoy_list_df = lambda p, s: pd.read_csv(data_path("decoys", s, p, "list.csv")) if s == "pep_nmr" else _ld(p, s)
ds._decoy_id_for = lambda n, s: n if s == "pep_nmr" else _di(n, s)
import torch
_ge = ds.Protein.get_energy
def _safe(self, *a, **k):
    try:
        return _ge(self, *a, **k)
    except RuntimeError as e:
        if "every block was dropped" not in str(e):
            raise
        return (torch.tensor(float("nan")), None) if k.get("return_terms") else torch.tensor(float("nan"))
ds.Protein.get_energy = _safe
sys.argv = [os.path.expanduser("~/nnef/nnef/scripts/evaluate_decoys.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
