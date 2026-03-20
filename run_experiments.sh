#!/bin/bash
set -e

# ============================================================
# RunPod experiment runner for Parameter Golf
# ============================================================
# Usage:
#   1xH100:  bash run_experiments.sh 1
#   8xH100:  bash run_experiments.sh 8
# ============================================================

NGPU=${1:-1}
echo "Running with ${NGPU} GPU(s)"

# -- Setup --
cd /workspace
if [ ! -d parameter-golf ]; then
    git clone https://github.com/NewstellerBot/parameter-golf.git
fi
cd parameter-golf

# Download full dataset (all 80 training shards)
if [ ! -f data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin ]; then
    echo "Downloading dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

# -- Experiment 1: Current SOTA baseline --
echo ""
echo "============================================================"
echo "EXP 1: SOTA baseline (int6 + MLP3x + Muon WD + overtone init)"
echo "============================================================"
RUN_ID=exp1_sota_baseline \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py

# -- Experiment 2: SOTA + linearize last MLP layer --
echo ""
echo "============================================================"
echo "EXP 2: SOTA + linearize last MLP (layer 8)"
echo "============================================================"
RUN_ID=exp2_linearize_last \
LINEARIZE_MLP_LAYERS=8 \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py

# -- Experiment 3: SOTA + linearize last 2 MLP layers --
echo ""
echo "============================================================"
echo "EXP 3: SOTA + linearize last 2 MLPs (layers 7,8)"
echo "============================================================"
RUN_ID=exp3_linearize_2 \
LINEARIZE_MLP_LAYERS=7,8 \
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py

echo ""
echo "============================================================"
echo "All experiments complete. Check logs/ for results."
echo "============================================================"
echo ""
echo "Quick results:"
for f in logs/exp*.txt; do
    echo "--- $(basename $f) ---"
    grep -E "final_int8_zlib_roundtrip_exact|final_sliding_window_exact|Total submission size int8" "$f" 2>/dev/null || true
done
