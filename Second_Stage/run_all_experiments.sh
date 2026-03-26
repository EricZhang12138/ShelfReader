#!/bin/bash
# Train all 5 models (image-only baseline + 4 fusion strategies) and evaluate each.
# Backbone: MobileNetV3-Large, Text: TinyBERT (all layers frozen).
# Stage 1 encoders are trained once and reused across fusion strategies.

set -e

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate mlp

DATA_ROOT="/mnt/external/home/YuouZhang/Documents_external/GroceryStoreDataset/dataset"
OCR_CACHE="ocr_cache_paddle.json"
CKPT_DIR="checkpoints"
FREEZE=4

cd "$(dirname "$0")"

echo "============================================================"
echo "STEP 0: Train Image-Only Baseline"
echo "============================================================"
python train_image_only.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE"

echo ""
echo "============================================================"
echo "STEP 1: Stage 1 - Pretrain encoders (shared across fusions)"
echo "============================================================"
python train.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --freeze_text_layers $FREEZE \
    --stage1_only

echo ""
echo "============================================================"
echo "STEP 2a: Train Concat Fusion"
echo "============================================================"
python train.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --fusion concat \
    --freeze_text_layers $FREEZE \
    --load_image_encoder "$CKPT_DIR/image_encoder_stage1.pth" \
    --load_text_encoder "$CKPT_DIR/text_encoder_stage1.pth"

echo ""
echo "============================================================"
echo "STEP 2b: Train Gated Fusion"
echo "============================================================"
python train.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --fusion gated \
    --freeze_text_layers $FREEZE \
    --load_image_encoder "$CKPT_DIR/image_encoder_stage1.pth" \
    --load_text_encoder "$CKPT_DIR/text_encoder_stage1.pth"

echo ""
echo "============================================================"
echo "STEP 2c: Train Cross Attention Fusion"
echo "============================================================"
python train.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --fusion cross_attention \
    --freeze_text_layers $FREEZE \
    --load_image_encoder "$CKPT_DIR/image_encoder_stage1.pth" \
    --load_text_encoder "$CKPT_DIR/text_encoder_stage1.pth"

echo ""
echo "============================================================"
echo "STEP 2d: Train Score Fusion (Fuzzy OCR)"
echo "============================================================"
python train_score_fusion.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE"

echo ""
echo "============================================================"
echo "STEP 3: Evaluate all models"
echo "============================================================"

echo ""
echo "--- Image-Only Baseline ---"
python evaluate_image_only.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_image_only.pth"

echo ""
echo "--- Concat Fusion ---"
python evaluate.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_concat_freeze${FREEZE}.pth"

echo ""
echo "--- Gated Fusion ---"
python evaluate.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_gated_freeze${FREEZE}.pth"

echo ""
echo "--- Cross Attention Fusion ---"
python evaluate.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_cross_attention_freeze${FREEZE}.pth"

echo ""
echo "--- Score Fusion (masked) ---"
python evaluate_score_fusion.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_score_fusion_masked.pth"

echo ""
echo "============================================================"
echo "STEP 4: Generate training plots"
echo "============================================================"
python plot_training.py \
    --log_dir logs \
    --out_dir figures \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --ckpt_dir "$CKPT_DIR" \
    --per_category

echo ""
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
