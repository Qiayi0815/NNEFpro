#!/usr/bin/env bash
# Score the pep_nmr peptide decoys with one model on CPU partitions (peptides are
# short; no GPU needed). One pass of per_residue_decoy_energy.py gives per-block
# energies -> whole-decoy energy (mean over blocks) AND per-residue energies.
#
#   [SET=pep_nmr CHUNK=27 OUT=eval/per_residue_pep] bash fasrc/pep_score_cpu.sh TAG RUN FAMILY K "MODEL_FLAGS" [PLAMBDA]
#     FAMILY ribbon | yang ; PLAMBDA = profile λ the model was trained with (default 1)
#     SET may be any decoy set with a pdb_no_missing_residue.csv (3DRobot_set, casp14_full, ...)
# Output: OUT/TAG/SET/<target>.npz
set -euo pipefail
TAG=$1; RUN=$2; FAMILY=$3; K=$4; MF=$5; PL=${6:-1}
case "$FAMILY" in
  ribbon) FAM="--mixture_rama 10" ;;
  yang)   FAM="--mixture_rama 0 --legacy_local_frame" ;;
  *) echo "FAMILY must be ribbon|yang"; exit 1 ;;
esac
SET=${SET:-pep_nmr}; CHUNK=${CHUNK:-27}; OUT=${OUT:-eval/per_residue_pep}
TL=nnef/data/decoys/$SET/pdb_no_missing_residue.csv
NT=$(( ($(wc -l < $TL) - 1 + CHUNK - 1) / CHUNK ))
mkdir -p runs/pack/pep_logs
sbatch --parsable --array=0-$((NT - 1)) -p ${PART:-shared,sapphire} --cpus-per-task=4 --mem=16G --time=${TIME:-4:00:00} \
  --job-name=sc_${SET:0:4}_$TAG -o runs/pack/pep_logs/${SET}_${TAG}_%a.out --wrap "
module load python/3.10.12-fasrc01 >/dev/null 2>&1; source \$(dirname \$(dirname \$(command -v python)))/etc/profile.d/conda.sh; conda activate ~/envs/nnef
T=\$(tail -n +2 $TL | sed -n \"\$((SLURM_ARRAY_TASK_ID*$CHUNK+1)),\$((SLURM_ARRAY_TASK_ID*$CHUNK+$CHUNK))p\" | paste -sd,)
[ -z \"\$T\" ] && exit 0
export OMP_NUM_THREADS=4
python -s nnef/scripts/per_residue_decoy_energy.py --load_exp $RUN --device cpu --mode CB --v2_fast \
  --mixture_seq 1 --seq_len $((K + 4)) --seq_type residue --residue_type_num 20 --embed_size 32 --dim 128 \
  --n_layers 4 --attn_heads 4 --mixture_r 2 --mixture_angle 3 --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 \
  --coords_angle_loss_lamda 1 --profile_loss_lamda $PL --coords_rama_loss_lamda 1 --use_position_weights \
  --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 $FAM $MF \
  --decoy_set $SET --targets \$T --tag $TAG --out_dir $OUT 2>&1 | grep -v -i warn
"
