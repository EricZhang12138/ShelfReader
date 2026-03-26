#!/bin/bash
# Train all models with:
#   - MobileNetV3-Small image encoder
#   - DistilBERT Multilingual text encoder (freeze 2/6 layers)
#   - Combined OCR cache (PaddleOCR + EasyOCR Swedish)

set -e
source activate mlp

DATA_ROOT="/mnt/external/home/YuouZhang/Documents_external/GroceryStoreDataset/dataset"
OCR_CACHE="ocr_cache_combined.json"
CKPT_DIR="checkpoints"
FREEZE=2

cd "$(dirname "$0")"

echo "============================================================"
echo "STEP 0: Train Image-Only Baseline"
echo "============================================================"
python train_image_only.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE"

echo ""
echo "============================================================"
echo "STEP 1: Stage 1 - Pretrain encoders"
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
echo "--- Score Fusion ---"
python evaluate_score_fusion.py \
    --data_root "$DATA_ROOT" \
    --ocr_cache "$OCR_CACHE" \
    --checkpoint "$CKPT_DIR/best_model_score_fusion_masked.pth"

echo ""
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
