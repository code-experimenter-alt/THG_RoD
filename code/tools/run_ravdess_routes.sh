#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/fu/miniconda3/bin/python
CODE=/home/fu/tmmrr_code
DATA=/mnt/hdd4t/tmmrr_data/ravdess/extracted
OUT=/mnt/hdd4t/tmmrr_data/ravdess/full_routes
CACHE=/mnt/hdd4t/tmmrr_cache/ravdess_full_routes
mkdir -p "$OUT" "$CACHE"

common=(--data_root "$DATA" --out_dir "$OUT" --cache_dir "$CACHE"
  --exp_id CV_MEL_BS_WEAK_RELEASE --sigmas 1 --split_mode official
  --group_col actor --label_col emotion --train_csv train.csv --dev_csv dev.csv
  --test_csv test.csv --train_folder . --dev_folder . --test_folder .
  --epochs_teacher 4 --epochs_student 5 --batch_size 32)

cd "$CODE"
for seed in 0 1 2; do
  hard_prefix="seed${seed}_sigma1_splitofficial_groupactor_ratios0.6_0.2_0.1_0.1_tbackmel_tmodelcnn_mel_featmel_sr16000_m40_L100_dsaf1_eta1e-05_dp1_C8_lossbalanced_softmax_tau1_ept4_lrt0.001_wd0.0001_bs32"
  hard_ckpt="$OUT/${hard_prefix}_tagravdess_full_hard_student_hard.pt"
  "$PYTHON" -m src.main "${common[@]}" --seeds "$seed" --student_mode hard --train_teacher 1 --tag ravdess_full_hard
  "$PYTHON" -m src.main "${common[@]}" --seeds "$seed" --student_mode kd --train_teacher 0 --tag ravdess_full_kd
  "$PYTHON" -m src.main "${common[@]}" --seeds "$seed" --student_mode hakd --train_teacher 0 --tag ravdess_full_hakd
  "$PYTHON" -m src.main "${common[@]}" --seeds "$seed" --student_mode tcrd --train_teacher 0 --paired_hard_ckpt "$hard_ckpt" --tag ravdess_full_tcrd
  "$PYTHON" -m src.main "${common[@]}" --seeds "$seed" --student_mode rc_tcrd --train_teacher 0 --paired_hard_ckpt "$hard_ckpt" --tag ravdess_full_rc_tcrd
done
