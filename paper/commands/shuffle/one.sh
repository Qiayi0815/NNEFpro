#!/usr/bin/env bash
# one.sh TAG RUN cb(ideal|orig) TARGETS OUT
cd ~/nnef; [ "$3" = ideal ] && export NNEF_CB=ideal; export OMP_NUM_THREADS=2
~/envs/nnef/bin/python -s nnef/scripts/seq_shuffle_test.py --load_exp $2 --device cpu --mode CB --v2_fast --mixture_seq 1 --seq_len 14 --seq_type residue --residue_type_num 20 --embed_size 32 --dim 128 --n_layers 4 --attn_heads 4 --mixture_r 2 --mixture_angle 3 --smooth_gaussian --smooth_r 0.3 --smooth_angle 45 --coords_angle_loss_lamda 1 --profile_loss_lamda 1 --coords_rama_loss_lamda 1 --use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3 --mixture_rama 10 --seq_head masked_center --smooth_nbr --smooth_tau 1.0 --energy_profile_weight 10 --decoy_set pep_nmr --targets $4 --tag $1 --out_dir $5 2>&1 | grep -v -i warn
