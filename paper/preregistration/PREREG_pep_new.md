# Pre-registered analysis: post-AF2-cutoff peptide validation (pep_new)

Written 2026-10-27, BEFORE any pep_new structure, prediction or score was generated or inspected.
The rule below was derived from pep_af2 (206 McDonald 2023 cluster representatives); pep_new is
used once, as an independent test.

## Data (pep_new)
- RCSB: SOLUTION NMR, initial release > 2018-04-30 (after the AlphaFold2 training cutoff), one
  protein entity, one chain, 10-40 residues, L-polypeptide, no non-standard monomers, no ligands,
  canonical 20 amino acids; then the pep_nmr curation filters (build_pep_nmr.py curate).
- Excluded: any entry in the McDonald 2023 table / pep_nmr, and any sequence with >= 40% identity
  (MMseqs2) to a pep_nmr sequence. One representative per 40%-identity cluster within pep_new.
- Candidates: ColabFold exactly as pep_af2 (alphafold2_ptm, 5 models x 5 seeds, MSA set + dropout set,
  no templates). Quality = GDT_TS to the closest NMR model (gdt_ts_max_ens).

## Fixed scorer and rule
- Energy: ribbon smooth k10, w = 10, --struct_dist_cutoff 1000 (tag rib_k10_smooth_w10_c1000,
  run smooth_rib_k10_54636530). No retraining, no parameter change.
- Gated rule: conf = max pLDDT over a peptide's candidates. If conf >= 80 pick the max-pLDDT model;
  otherwise pick the lowest-energy model.

## Hypotheses and tests (one-sided, alpha = 0.05)
- H1 (primary): mean over peptides of d_gate = GDT(gated pick) - GDT(pLDDT pick) > 0.
  Test: one-sided Wilcoxon signed-rank on non-zero d_gate; also report mean with bootstrap 95% CI.
- H2: Spearman(d_ribbon, conf) < 0, with d_ribbon = GDT(ribbon pick) - GDT(pLDDT pick);
  permutation p (af2_conf_trend.py, unchanged).
- Secondary/descriptive only: bins <70 / 70-80 / 80-90 / >=90; AF2Rank, Rosetta, RW+, rama_aa,
  random as references; medoid-GDT as an alternative quality measure.

Whatever the outcome, results are reported as obtained.

## Amendment 1 (2026-10-27, after data curation, before any pep_new prediction/score was inspected)
- Data as built: 344 new entries -> 311 after pep_nmr curation -> 213 after removing >= 40% identity
  to pep_nmr (union of query-normalised identity and the clustering rule) -> 136 cluster
  representatives (MMseqs2, same command as pep_nmr).
- Length: the k10 model cannot form a block for chains < 15 residues (23 representatives), and
  pep_af2 (where the rule was derived) contained only peptides >= 15 residues. H1 and H2 are
  therefore tested on representatives with length >= 15 (expected 113). Secondary: all 136, with
  the gated rule falling back to the pLDDT pick where the energy cannot score (d = 0 there).
- Disulfide-rich peptides are kept (as in pep_af2); training-set homologs are kept and reported
  as a descriptive split (as in pep_nmr).
