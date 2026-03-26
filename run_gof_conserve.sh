#!/bin/bash
# =============================================================
# Conserve machine: batch GOF evaluation
# =============================================================
# Upload this + batch_gof_eval.py to the conserve machine,
# then run: bash run_gof_conserve.sh
#
# Expected directory structure (already in place):
#   critical-nca-reservoir-conserve/
#     ├── batch_gof_eval.py         (NEW — upload this)
#     ├── summarize_gof.py          (NEW — upload this)
#     ├── run_gof_conserve.sh       (NEW — upload this)
#     ├── critical_nca.py
#     ├── evaluate_criticality.py
#     ├── utils.py
#     ├── test_nca.py
#     └── logs/
#         └── train_nca_conserve/
#             ├── 20260306-184946/   (seed 42, 25 ckpts)
#             ├── 20260308-103239/   (seed 43, 28 ckpts)
#             └── 20260309-234527/   (seed 44, 19 ckpts)
# =============================================================

set -e

# Work from the script's directory (conserve machine root)
cd "$(dirname "$0")"

mkdir -p gof_results
mkdir -p gof_logs

echo "Starting CONSERVE batch GOF evaluation at $(date)"
echo "Working directory: $(pwd)"
echo "============================================"

# 3 parallel jobs — one per seed (72 checkpoints total)

echo "Launching conserve seed 42 (25 ckpts)..."
nohup python batch_gof_eval.py --variant conserve --seed 42 --width 1000 \
  > gof_logs/conserve_seed42.log 2>&1 &
PID42=$!

echo "Launching conserve seed 43 (28 ckpts)..."
nohup python batch_gof_eval.py --variant conserve --seed 43 --width 1000 \
  > gof_logs/conserve_seed43.log 2>&1 &
PID43=$!

echo "Launching conserve seed 44 (19 ckpts)..."
nohup python batch_gof_eval.py --variant conserve --seed 44 --width 1000 \
  > gof_logs/conserve_seed44.log 2>&1 &
PID44=$!

echo ""
echo "3 conserve jobs launched: PIDs $PID42, $PID43, $PID44"
echo ""
echo "Monitor:  tail -f gof_logs/conserve_seed*.log"
echo "Progress: wc -l gof_results/gof_conserve_seed*.csv"
echo "Check:    ps aux | grep batch_gof_eval"
echo ""
echo "Waiting for all 3 jobs to finish..."
wait $PID42 $PID43 $PID44
echo ""
echo "All conserve GOF jobs completed at $(date)"
echo "============================================"

# Step 2: Summarize results
echo ""
echo "Running summarize_gof.py..."
python summarize_gof.py --indir gof_results 2>&1 | tee gof_results/conserve_summary.txt

echo ""
echo "============================================"
echo "Summary saved to gof_results/conserve_summary.txt"
echo "Review it, then proceed to re-run downstream tasks"
echo "if the best-GOF checkpoint differs from gen 499."
echo ""
echo "To re-run downstream tasks with a different checkpoint,"
echo "edit the TF checkpoint pointer file in the logdir:"
echo "  logs/train_nca_conserve/<timestamp>/checkpoint"
echo "to point to the best-GOF ckpt, then re-run:"
echo "  python test_nca.py            (GOF verification)"
echo "  python reservoir_mnist.py ...  (MNIST dataset)"
echo "  python reservoir_mnist_classify.py ...  (MNIST classify)"
echo "  python reservoir_cartpole_train_qlearning.py ...  (CartPole)"
echo "  python reservoir_cartpole_evaluate_rl.py ...  (CartPole eval)"
echo "============================================"
