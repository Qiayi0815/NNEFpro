#!/usr/bin/env python3
"""AF2Rank baseline (Roney & Ovchinnikov, PRL 2022) for pep_af2 model selection.

Each candidate structure is fed to AlphaFold2 (model_1_ptm, 1 recycle, no MSA) as a template with
its sequence and side chains masked; composite = pTM * pLDDT * TM(input, output). Lower score =
better, so score = -composite. Output: eval/baselines_pep/af2rank/pep_af2/scores.csv (target,NAME,score)
plus the raw terms in af2rank_terms.csv. Runs in the localcolabfold env with colabdesign on PYTHONPATH:

    PYTHONPATH=<af2rank_pylib> <colabfold-conda>/bin/python eval/baselines_pep/af2rank_score.py \
        --data_dir <localcolabfold>/colabfold [--targets 1ERD,2LE7] [--chunk i/n]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from colabdesign import mk_af_model
from tmtools import tm_align

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = '/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work'
DEC = os.path.join(REPO, 'nnef', 'data', 'decoys', 'pep_af2')
OUT = os.path.join(REPO, 'eval', 'baselines_pep', 'af2rank', 'pep_af2')
PAT = re.compile(r'^(?P<pdb>\w+?)_unrelaxed_rank_\d+_alphafold2_ptm_model_(?P<m>\d)_seed_(?P<s>\d+)\.pdb$')
SETS = {'msa': '', 'drop': '_drop'}


def candidates(pdb):
    for sname, suf in SETS.items():
        for f in sorted(glob.glob(f'{WORK}/af2_{sname}/{pdb}_unrelaxed_rank_*.pdb')):
            m = PAT.match(os.path.basename(f))
            if m and m['pdb'] == pdb:
                yield f'{pdb}_af2_m{m["m"]}_s{int(m["s"])}{suf}', f


def score_one(af, f):
    af.prep_inputs(f, chain='A', rm_template_seq=True, rm_template_sc=True)
    af.set_seq(mode='wildtype')
    seq = ''.join('ARNDCQEGHILKMFPSTWYVX'[k] for k in np.asarray(af._wt_aatype))  # AF2 restype order
    af.predict(models=['model_1_ptm'], num_recycles=1, verbose=False)
    log = af.aux['log']
    i_ca = np.asarray(af._inputs['batch']['all_atom_positions'][:, 1], float)
    o_ca = np.asarray(af.aux['atom_positions'][:, 1], float)
    tm_io = tm_align(i_ca, o_ca, seq, seq).tm_norm_chain1
    plddt, ptm = float(log['plddt']), float(log['ptm'])
    return dict(plddt=plddt, ptm=ptm, tm_io=tm_io, composite=ptm * plddt * tm_io)


def main():
    global WORK, DEC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='dir containing params/params_model_1_ptm.npz')
    ap.add_argument('--targets', default='')
    ap.add_argument('--chunk', default='0/1', help='i/n: score every n-th target starting at i')
    ap.add_argument('--set', default='pep_af2')
    ap.add_argument('--work', default=WORK, help='ColabFold work dir with af2_msa/ and af2_drop/')
    a = ap.parse_args()
    WORK = a.work
    DEC = os.path.join(REPO, 'nnef', 'data', 'decoys', a.set)
    OUT = os.path.join(REPO, 'eval', 'baselines_pep', 'af2rank', a.set)
    targets = a.targets.split(',') if a.targets else list(pd.read_csv(f'{DEC}/pdb_no_missing_residue.csv').pdb)
    i, n = map(int, a.chunk.split('/'))
    targets = targets[i::n]
    os.makedirs(f'{OUT}/parts', exist_ok=True)
    af = mk_af_model(protocol='fixbb', use_templates=True, data_dir=a.data_dir, model_names=['model_1_ptm'])
    for t in targets:
        part = f'{OUT}/parts/{t}.csv'
        if os.path.exists(part):
            continue
        lst = pd.read_csv(f'{DEC}/{t}/list.csv')
        out = []
        for name, f in candidates(t):
            try:
                out.append(dict(target=t, NAME=name, **score_one(af, f)))
            except Exception as exc:  # noqa: BLE001
                print(f'[af2rank] {name} failed: {exc!r}', flush=True)
        d = pd.DataFrame(out)
        if len(d):
            d = d[d.NAME.isin(set(lst.NAME))]
            d.to_csv(part, index=False)
            print(f'[af2rank] {t}: {len(d)} models, composite {d.composite.min():.3f}-{d.composite.max():.3f}', flush=True)
    parts = sorted(glob.glob(f'{OUT}/parts/*.csv'))
    if parts:
        allp = pd.concat([pd.read_csv(p) for p in parts])
        allp.to_csv(f'{OUT}/af2rank_terms.csv', index=False)
        allp.assign(score=-allp.composite)[['target', 'NAME', 'score']].to_csv(f'{OUT}/scores.csv', index=False)


if __name__ == '__main__':
    main()
