#!/usr/bin/env python3
"""Rosetta REF2015 baseline for pep_af2 model selection (PyRosetta).

Per AF2 candidate: score_raw = REF2015 of the structure as predicted (hydrogens added by PyRosetta);
score_min = REF2015 after repacking side chains and minimising side chains only (backbone fixed,
so the backbone being judged is unchanged while side-chain clashes of unrelaxed models are removed).
Writes eval/baselines_pep/rosetta_{raw,min}/pep_af2/scores.csv (target,NAME,score; lower = better).

    <pyrosetta venv>/bin/python eval/baselines_pep/rosetta_score.py [--targets 1ERD] [--chunk i/n]
"""
import argparse
import glob
import os
import re

import pandas as pd
import pyrosetta
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import InitializeFromCommandline, RestrictToRepacking
from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = '/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_af2_work'
DEC = os.path.join(REPO, 'nnef', 'data', 'decoys', 'pep_af2')
OUTR = os.path.join(REPO, 'eval', 'baselines_pep')
PAT = re.compile(r'^(?P<pdb>\w+?)_unrelaxed_rank_\d+_alphafold2_ptm_model_(?P<m>\d)_seed_(?P<s>\d+)\.pdb$')
SETS = {'msa': '', 'drop': '_drop'}


def candidates(pdb):
    for sname, suf in SETS.items():
        for f in sorted(glob.glob(f'{WORK}/af2_{sname}/{pdb}_unrelaxed_rank_*.pdb')):
            m = PAT.match(os.path.basename(f))
            if m and m['pdb'] == pdb:
                yield f'{pdb}_af2_m{m["m"]}_s{int(m["s"])}{suf}', f


def main():
    global WORK, DEC
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='')
    ap.add_argument('--chunk', default='0/1')
    ap.add_argument('--set', default='pep_af2')
    ap.add_argument('--work', default=WORK, help='ColabFold work dir with af2_msa/ and af2_drop/')
    a = ap.parse_args()
    WORK = a.work
    DEC = os.path.join(REPO, 'nnef', 'data', 'decoys', a.set)
    pyrosetta.init('-mute all -ex1 -ex2aro -use_input_sc')
    sf = pyrosetta.create_score_function('ref2015')
    tf = TaskFactory()
    tf.push_back(InitializeFromCommandline())
    tf.push_back(RestrictToRepacking())
    mm = MoveMap()
    mm.set_bb(False)
    mm.set_chi(True)
    targets = a.targets.split('+') if a.targets else list(pd.read_csv(f'{DEC}/pdb_no_missing_residue.csv').pdb)
    i, n = map(int, a.chunk.split('/'))
    pdir = f'{OUTR}/rosetta_parts' + ('' if a.set == 'pep_af2' else f'_{a.set}')
    os.makedirs(pdir, exist_ok=True)
    for t in targets[i::n]:
        part = f'{pdir}/{t}.csv'
        if os.path.exists(part):
            continue
        rows = []
        for name, f in candidates(t):
            try:
                pose = pyrosetta.pose_from_pdb(f)
                raw = sf(pose)
                pack = PackRotamersMover(sf)
                pack.task_factory(tf)
                pack.apply(pose)
                MinMover(mm, sf, 'lbfgs_armijo_nonmonotone', 1e-3, True).apply(pose)
                rows.append(dict(target=t, NAME=name, raw=raw, min=sf(pose)))
            except Exception as exc:  # noqa: BLE001
                print(f'[rosetta] {name} failed: {exc!r}', flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f'[rosetta] {t}: {len(rows)} models', flush=True)
    parts = sorted(glob.glob(f'{pdir}/*.csv'))
    if parts:
        d = pd.concat([pd.read_csv(p) for p in parts])
        for k in ('raw', 'min'):
            os.makedirs(f'{OUTR}/rosetta_{k}/{a.set}', exist_ok=True)
            d.rename(columns={k: 'score'})[['target', 'NAME', 'score']].to_csv(f'{OUTR}/rosetta_{k}/{a.set}/scores.csv', index=False)


if __name__ == '__main__':
    main()
