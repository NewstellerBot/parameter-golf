#!/bin/bash
set -e

# ============================================================
# RunPod experiment runner for Parameter Golf
# ============================================================
# Usage:
#   1xH100:  bash run_experiments.sh 1
#   8xH100:  bash run_experiments.sh 8
#
# Run a single experiment:
#   bash run_experiments.sh 1 3    (run only exp 3 on 1 GPU)
# ============================================================

NGPU=${1:-1}
ONLY_EXP=${2:-0}  # 0 = run all
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

run_exp() {
    local exp_num=$1
    local desc=$2
    shift 2
    if [ "$ONLY_EXP" != "0" ] && [ "$ONLY_EXP" != "$exp_num" ]; then
        return
    fi
    echo ""
    echo "============================================================"
    echo "EXP ${exp_num}: ${desc}"
    echo "============================================================"
    env "$@" torchrun --standalone --nproc_per_node=$NGPU train_gpt.py
}

# ============================================================
# EXPERIMENTS
# ============================================================
# Defaults baked into train_gpt.py (combined SOTA):
#   NUM_LAYERS=9  MODEL_DIM=512  MLP_HIDDEN=1536  (MLP 3x)
#   TRAIN_SEQ_LEN=2048  TRAIN_BATCH_TOKENS=786432
#   WARMDOWN_ITERS=20000  GRAD_CLIP_NORM=1.0
#   MATRIX_LR=0.06  SCALAR_LR=0.06  TIED_EMBED_LR=0.07
#   MUON_WD=0.02  EVAL_STRIDE=256
#   Int6 quant + FP16 embed + overtone init + phase-transition resid_mix
# ============================================================

# -- Exp 1: Our combined SOTA baseline --
run_exp 1 "Combined SOTA baseline" \
    RUN_ID=exp1_baseline

# -- Exp 2: Linearize last MLP --
run_exp 2 "Linearize last MLP (layer 8)" \
    RUN_ID=exp2_linearize_last \
    LINEARIZE_MLP_LAYERS=8

# -- Exp 3: Linearize last 2 MLPs --
run_exp 3 "Linearize last 2 MLPs (layers 7,8)" \
    RUN_ID=exp3_linearize_2 \
    LINEARIZE_MLP_LAYERS=7,8

# -- Exp 4: Stronger Muon weight decay (top open PRs use 0.04) --
run_exp 4 "Muon WD=0.04 (from best open PRs)" \
    RUN_ID=exp4_muon_wd04 \
    MUON_WD=0.04

# -- Exp 5: 10 layers (fits with int6, proven in OvertoneInit submission) --
run_exp 5 "10 layers + MLP 2x (int6 fits in 16MB)" \
    RUN_ID=exp5_10layers \
    NUM_LAYERS=10 \
    MLP_HIDDEN=0 \
    MLP_MULT=2

# -- Exp 6: 10 layers + linearize last to reclaim space for MLP 3x --
run_exp 6 "10 layers + MLP 3x + linearize last" \
    RUN_ID=exp6_10L_mlp3x_linearize \
    NUM_LAYERS=10 \
    LINEARIZE_MLP_LAYERS=9

# -- Exp 7: Higher Muon momentum (proven at long context) --
run_exp 7 "Muon momentum=0.99 + longer warmup" \
    RUN_ID=exp7_high_momentum \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500

# -- Exp 8: Different seed for variance check --
run_exp 8 "Seed=42 (variance check on exp1)" \
    RUN_ID=exp8_seed42 \
    SEED=42

# ============================================================
echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
for f in logs/exp*.txt; do
    echo ""
    echo "--- $(basename $f .txt) ---"
    grep -E "final_int8_zlib_roundtrip_exact|final_sliding_window_exact|Total submission size int8|model_params|linearize_mlp" "$f" 2>/dev/null || true
done
