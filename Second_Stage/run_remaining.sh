#!/bin/bash
# Resume from gated (freeze=2). Stage1, concat already done.
set -e
source activate mlp

DATA_ROOT="/mnt/external/home/YuouZhang/Documents_external/GroceryStoreDataset/dataset"
OCR_CACHE="ocr_cache_paddle.json"
CKPT_DIR="checkpoints"
FREEZE=2

cd "$(dirname "$0")"

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
echo "STEP 3: Evaluate all models"
echo "============================================================"

echo ""
echo "--- Image-Only Baseline ---"
python evaluate_image_only.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_image_only.pth"

echo ""
echo "--- Concat Fusion (freeze=2) ---"
python evaluate.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_concat_freeze${FREEZE}.pth"

echo ""
echo "--- Gated Fusion (freeze=2) ---"
python evaluate.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_gated_freeze${FREEZE}.pth"

echo ""
echo "--- Cross Attention Fusion (freeze=2) ---"
python evaluate.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_cross_attention_freeze${FREEZE}.pth"

echo ""
echo "--- Score Fusion (unchanged) ---"
python evaluate_score_fusion.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_score_fusion_masked.pth"

echo ""
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
