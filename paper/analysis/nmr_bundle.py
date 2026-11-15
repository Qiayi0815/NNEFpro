"""Spread of the deposited NMR ensemble of each peptide, as the yardstick for the T = 0 test, and the sample condition from the PDB header.
Run from paper_plan/figures: python3 scripts/nmr_bundle.py  ->  nmr_bundle/per_target.csv, nmr_bundle/per_position.csv
Inputs: pep_native/<ID>/<ID>_native_bead.csv (the conformer used as the experimental structure) and pdb_nmr/<ID>.pdb (all deposited conformers).
RMSD definitions follow the T = 0 test: Cβ RMSD after superposition on all Cβ (Gly excluded here, the PDB has no Cβ for it); per-position deviation = Cα after superposition on all Cα."""
import os, re, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from render_vmd import kabsch

KEY = [('micelle', r'MICELL|\bSDS\b|\bDPC\b|DODECYL|BICELL|\bDHPC\b|\bLPPG\b|\bLMPG\b'), ('lipid', r'LIPOPOLYSACCHARIDE|\bLPS\b|VESICLE|LIPOSOME|BILAYER|NANODISC|MEMBRANE'),
       ('cosolvent', r'\bTFE\b|TRIFLUOROETHANOL|\bHFIP\b|HEXAFLUORO|\bDMSO\b|METHANOL|CHLOROFORM|ACETONITRILE')]


def models(pdb):
    out, cur, chains = [], {}, set()
    for l in open(pdb):
        if l.startswith('ENDMDL'):
            if cur: out.append(cur); cur = {}
        elif l.startswith('ATOM') and l[16] in ' A':
            chains.add(l[21]); nm = l[12:16].strip()
            if nm in ('CA', 'CB'): cur.setdefault((l[21], nm), {})[int(l[22:26])] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
    if cur: out.append(cur)
    return out, chains


def dev(P, Q):
    R, Pc, Qc = kabsch(P, Q); return np.linalg.norm((P - Pc) @ R + Qc - Q, axis=1)


def main():
    os.makedirs('nmr_bundle', exist_ok=True); rows, pos = [], []
    for d in sorted(glob.glob('pep_native/*')):
        T = os.path.basename(d); pdb = f'pdb_nmr/{T}.pdb'
        if not os.path.exists(pdb) or os.path.getsize(pdb) < 2000: rows.append(dict(target=T, status='no pdb')); continue
        nb = pd.read_csv(f'{d}/{T}_native_bead.csv'); n_ca = nb[['xca', 'yca', 'zca']].values.astype(float); n_cb = nb[['xcb', 'ycb', 'zcb']].values.astype(float); L = len(nb)
        head = ' '.join(l[10:80] for l in open(pdb) if l.startswith(('TITLE', 'KEYWDS', 'REMARK 210', 'COMPND'))).upper()
        cond = [k for k, pat in KEY if re.search(pat, head)]
        M, chains = models(pdb); best = None
        for ch in sorted(chains):   # the chain and residue numbering that reproduce our experimental conformer
            ids = sorted(M[0].get((ch, 'CA'), {}))
            if len(ids) < L: continue
            for off in range(len(ids) - L + 1):
                win = ids[off:off + L]
                for k, m in enumerate(M):
                    ca = m.get((ch, 'CA'), {})
                    if all(i in ca for i in win):
                        r = np.sqrt((dev(np.array([ca[i] for i in win]), n_ca) ** 2).mean())
                        if best is None or r < best[0]: best = (r, ch, win, k)
        if best is None or best[0] > 0.5: rows.append(dict(target=T, status='not matched', n_models=len(M), condition='+'.join(cond) or 'none found', n_chains=len(chains))); continue
        r0, ch, win, k0 = best; cbs, cas = [], []
        nong = np.array([g != 'G' for g in nb.group_name])
        for k, m in enumerate(M):
            if k == k0: continue
            ca, cb = m.get((ch, 'CA'), {}), m.get((ch, 'CB'), {})
            if not all(i in ca for i in win): continue
            cas.append(dev(np.array([ca[i] for i in win]), n_ca))
            w = [i for i, ok in zip(win, nong) if ok]
            if all(i in cb for i in w) and len(w) > 3: cbs.append(np.sqrt((dev(np.array([cb[i] for i in w]), n_cb[nong]) ** 2).mean()))
        rows.append(dict(target=T, status='ok', n_models=len(M), model_used=k0 + 1, match_rmsd=r0, n_chains=len(chains), condition='+'.join(cond) or 'none found', L=L,
                         bundle_cb_rmsd=float(np.median(cbs)) if cbs else np.nan, bundle_ca_rmsd=float(np.median([np.sqrt((c ** 2).mean()) for c in cas])) if cas else np.nan))
        if cas:
            med = np.median(np.array(cas), 0); pos += [dict(target=T, L=L, pos=i, dev=float(v)) for i, v in enumerate(med)]
    r = pd.DataFrame(rows); r.to_csv('nmr_bundle/per_target.csv', index=False); pd.DataFrame(pos).to_csv('nmr_bundle/per_position.csv', index=False)
    print(r.status.value_counts().to_dict()); ok = r[r.status == 'ok']; print('models', ok.n_models.describe()[['min', '50%', 'max']].to_dict())
    print('bundle Cβ RMSD to the used conformer: median %.2f, quartiles %.2f-%.2f' % (ok.bundle_cb_rmsd.median(), ok.bundle_cb_rmsd.quantile(.25), ok.bundle_cb_rmsd.quantile(.75)))
    print(ok.condition.value_counts().to_dict()); print('model used = 1:', (ok.model_used == 1).mean().round(2), 'multi-chain entries:', (ok.n_chains > 1).sum())


if __name__ == '__main__': main()
