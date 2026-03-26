#!/bin/bash
# Run 5-bit memory task for all Phase 2 seeds
#
# For baseline machine (critical-nca-reservoir-baseline):
#   cd /root/critical-nca-reservoir-baseline
#   bash run_5bit_all_seeds.sh baseline
#
# For conserve machine (critical-nca-reservoir-conserve):
#   cd /root/critical-nca-reservoir-conserve
#   bash run_5bit_all_seeds.sh conserve

set -e

VARIANT=${1:-""}

if [ "$VARIANT" = "baseline" ]; then
    LOGDIRS=(
        "logs/train_nca/20260306-184855"
        "logs/train_nca/20260308-172413"
        "logs/train_nca/20260310-115208"
    )
    SEEDS=(42 43 44)
elif [ "$VARIANT" = "conserve" ]; then
    LOGDIRS=(
        "logs/train_nca_conserve/20260306-184946"
        "logs/train_nca_conserve/20260308-103239"
        "logs/train_nca_conserve/20260309-234527"
    )
    SEEDS=(42 43 44)
else
    echo "Usage: bash run_5bit_all_seeds.sh <baseline|conserve>"
    exit 1
fi

echo "=== Running 5-bit memory task for $VARIANT variant ==="
echo "Date: $(date)"
echo ""

for i in 0 1 2; do
    LOGDIR="${LOGDIRS[$i]}"
    SEED="${SEEDS[$i]}"
    echo "--- Seed $SEED: $LOGDIR ---"
    python reservoir_X-bit_make_dataset.py --logdir "$LOGDIR" 2>&1 | tee "5bit_${VARIANT}_seed${SEED}.log"
    echo ""
done

echo "=== All seeds complete for $VARIANT ==="
echo "Results in: 5bit_${VARIANT}_seed*.log"
