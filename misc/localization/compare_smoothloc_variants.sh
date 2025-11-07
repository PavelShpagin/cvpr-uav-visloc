#!/bin/bash
# Compare SmoothLoc Variants
# Run all 4 variants and compare results

set -e

REPO_ROOT="/home/pavel/dev/drone-embeddings"
VENV="$REPO_ROOT/venv/bin/python"
DATASET="stream2"
VPR="modernloc"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         SMOOTHLOC ABLATION STUDY - ALL VARIANTS                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "VPR Method: $VPR"
echo ""

# 1. Original SmoothLoc (with centering, weighted average)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] SmoothLoc (Original - Weighted + Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc/eval.py" \
    --method $VPR \
    --dataset $DATASET \
    --window 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] SmoothLoc-Simplified (Weighted + No Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc_simplified/eval.py" \
    --dataset $DATASET \
    --vpr $VPR \
    --window 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] SmoothLoc-Top1 (Top-1 Refs + No Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc_top1/eval.py" \
    --dataset $DATASET \
    --vpr $VPR \
    --window 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] SmoothLoc-Top1-Unique (Top-1 Unique Refs + No Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc_top1/eval.py" \
    --dataset $DATASET \
    --vpr $VPR \
    --window 10 \
    --unique

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                     COMPARISON SUMMARY                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Extract and compare results
echo "Variant                          | ATE      "
echo "─────────────────────────────────┼──────────"

# Original
ATE1=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc (Original)             | $ATE1"

# Simplified
ATE2=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc_simplified/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc-Simplified             | $ATE2"

# Top1
ATE3=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc_top1/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc-Top1                   | $ATE3"

# Top1-Unique
ATE4=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc_top1_unique/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc-Top1-Unique            | $ATE4"

echo ""
echo "For comparison:"
echo "BayesianLoc                      | 33.71m"
echo ""
echo "✅ Ablation study complete!"
echo ""



# Run all 4 variants and compare results

set -e

REPO_ROOT="/home/pavel/dev/drone-embeddings"
VENV="$REPO_ROOT/venv/bin/python"
DATASET="stream2"
VPR="modernloc"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         SMOOTHLOC ABLATION STUDY - ALL VARIANTS                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "VPR Method: $VPR"
echo ""

# 1. Original SmoothLoc (with centering, weighted average)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/4] SmoothLoc (Original - Weighted + Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc/eval.py" \
    --method $VPR \
    --dataset $DATASET \
    --window 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/4] SmoothLoc-Simplified (Weighted + No Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc_simplified/eval.py" \
    --dataset $DATASET \
    --vpr $VPR \
    --window 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/4] SmoothLoc-Top1 (Top-1 Refs + No Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc_top1/eval.py" \
    --dataset $DATASET \
    --vpr $VPR \
    --window 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/4] SmoothLoc-Top1-Unique (Top-1 Unique Refs + No Centering)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$VENV "$REPO_ROOT/research/localization/smoothloc_top1/eval.py" \
    --dataset $DATASET \
    --vpr $VPR \
    --window 10 \
    --unique

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                     COMPARISON SUMMARY                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Extract and compare results
echo "Variant                          | ATE      "
echo "─────────────────────────────────┼──────────"

# Original
ATE1=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc (Original)             | $ATE1"

# Simplified
ATE2=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc_simplified/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc-Simplified             | $ATE2"

# Top1
ATE3=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc_top1/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc-Top1                   | $ATE3"

# Top1-Unique
ATE4=$(grep "ATE:" "$REPO_ROOT/research/results/smoothloc_top1_unique/$DATASET/${VPR}_results.txt" | awk '{print $2}')
echo "SmoothLoc-Top1-Unique            | $ATE4"

echo ""
echo "For comparison:"
echo "BayesianLoc                      | 33.71m"
echo ""
echo "✅ Ablation study complete!"
echo ""
