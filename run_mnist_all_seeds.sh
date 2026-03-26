#!/bin/bash
# Run MNIST feature generation + classification for all 3 conserve seeds sequentially.
# Usage: nohup bash run_mnist_all_seeds.sh > results/mnist_w1000/all_seeds.log 2>&1 &

set -e
cd ~/autodl-tmp/critical-nca-reservoir
mkdir -p results/mnist_w1000

# ── Seed 42 ────────────────────────────────────────────────────────────────
echo "===== [$(date)] Generating features: seed42 ====="
python reservoir_mnist_make_dataset.py --logdir logs/train_nca_conserve/20260306-184946
mv mnist_x_train_v2.csv results/mnist_w1000/train_seed42.csv
mv mnist_x_test_v2.csv  results/mnist_w1000/test_seed42.csv

echo "===== [$(date)] Classifying: seed42 ====="
python reservoir_mnist_classify.py \
    --train_csv results/mnist_w1000/train_seed42.csv \
    --test_csv  results/mnist_w1000/test_seed42.csv \
    --label "conserve_w1000_seed42" --raw_baseline --runs 10

# ── Seed 43 (CSVs already exist) ───────────────────────────────────────────
echo "===== [$(date)] Classifying: seed43 ====="
python reservoir_mnist_classify.py \
    --train_csv results/mnist_w1000/train_seed43.csv \
    --test_csv  results/mnist_w1000/test_seed43.csv \
    --label "conserve_w1000_seed43" --raw_baseline --runs 10

# ── Seed 44 ────────────────────────────────────────────────────────────────
echo "===== [$(date)] Generating features: seed44 ====="
python reservoir_mnist_make_dataset.py --logdir logs/train_nca_conserve/20260309-234527
mv mnist_x_train_v2.csv results/mnist_w1000/train_seed44.csv
mv mnist_x_test_v2.csv  results/mnist_w1000/test_seed44.csv

echo "===== [$(date)] Classifying: seed44 ====="
python reservoir_mnist_classify.py \
    --train_csv results/mnist_w1000/train_seed44.csv \
    --test_csv  results/mnist_w1000/test_seed44.csv \
    --label "conserve_w1000_seed44" --raw_baseline --runs 10

echo "===== [$(date)] ALL DONE ====="
