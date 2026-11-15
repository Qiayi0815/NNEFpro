import numpy as np, pandas as pd, glob, os, sys
D=sys.argv[1]; meta=pd.read_csv(sys.argv[2]).set_index('pdb')
z=np.load(glob.glob(D+'/pep_nmr/*.npz')[0]); print(z.files, z['E_cen'].shape, list(z['names'][:6].astype(str)), list(z['names'][-6:].astype(str)))
rows=[]
for f in sorted(glob.glob(D+'/pep_nmr/*.npz')):
    t=os.path.basename(f)[:-4]; z=np.load(f); E=z['E_cen']; nm=z['names'].astype(str)
    scored=np.isfinite(E).any(1); common=np.isfinite(E[scored]).mean(0)>=0.9
    full=scored&np.isfinite(E[:,common]).all(1)
    e=np.full(len(E),np.nan); e[full]=E[full][:,common].mean(1)
    def idx(k): 
        w=[i for i,n in enumerate(nm) if k in n]; return w[0] if w else None
    h=idx('ctrl_helix'); x=idx('ctrl_ext'); p=idx('ctrl_ppii') if idx('ctrl_ppii') is not None else idx('ctrl_PPII'); hp=idx('ctrl_hairpin'); n=idx('native')
    r=dict(t=t,L=E.shape[1],ndec=len(E),nscored=int(scored.sum()),nfull=int(full.sum()),ncommon=int(common.sum()))
    for k,i in (('helix',h),('ext',x),('ppii',p),('hairpin',hp),('native',n)):
        r[k+'_blocks']=int(np.isfinite(E[i]).sum()) if i is not None else -1
        r[k+'_in']=bool(i is not None and full[i])
    r['helix_wins']=bool(h is not None and full[h] and np.nanargmin(e)==h) if np.isfinite(e).any() else False
    # pairwise on residues both scored
    if h is not None and n is not None:
        m=np.isfinite(E[h])&np.isfinite(E[n]); r['pair_n']=int(m.sum())
        r['pair_helix_lower']=bool(m.sum()>0 and E[h][m].mean()<E[n][m].mean())
    # helix lowest among ALL scored decoys, on residues the helix has, each decoy on its shared residues w/ helix
    if h is not None and np.isfinite(E[h]).any():
        hm=np.isfinite(E[h]); ok=scored&np.isfinite(E[:,hm]).all(1); ee=E[ok][:,hm].mean(1); names=nm[ok]
        r['hset_n']=int(ok.sum()); r['helix_wins_hset']=bool(names[np.argmin(ee)]==nm[h])
    rows.append(r)
df=pd.DataFrame(rows).set_index('t').join(meta[['ss_class']] if 'ss_class' in meta else meta)
df.to_csv(D+'/fig3d_check.csv')
nh=df[(df.ss_class!='helix')&(df.L>=15)]
print('non-helical L>=15:',len(nh))
print('have any full decoy:',(nh.nfull>0).sum())
for k in ('helix','ext','ppii','hairpin','native'):
    print(k,'blocks==0:',(nh[k+'_blocks']==0).sum(),' missing:',(nh[k+'_blocks']<0).sum(),' in comparison:',nh[k+'_in'].sum())
print('helix_wins (as coded):',nh.helix_wins.sum(),'of',len(nh),'; among helix-in:',nh[nh.helix_in].helix_wins.sum(),'of',nh.helix_in.sum())
print('pairwise helix<native (shared residues):',nh.pair_helix_lower.sum(),'of',nh.pair_n.gt(0).sum())
print('helix lowest on helix-residue set:',nh.helix_wins_hset.sum(),'of',nh.helix_wins_hset.notna().sum(),' median #decoys compared',nh.hset_n.median())
print(nh.loc[['1U6V']] .T if '1U6V' in nh.index else 'no 1U6V')
print(nh.groupby('ss_class')[['helix_in','helix_wins','pair_helix_lower','helix_wins_hset']].sum())
