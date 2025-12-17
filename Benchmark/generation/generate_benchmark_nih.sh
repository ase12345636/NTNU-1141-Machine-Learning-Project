#!/bin/bash
# Script to generate CXReasonBench for NIH CXR-14 dataset
# For use with WSL (Windows Subsystem for Linux)

# Set paths (WSL format: /mnt/d instead of d:)
NIH_BASE_DIR="/mnt/d/CXReasonBench"
CHEXSTRUCT_DIR="${NIH_BASE_DIR}/nih_cxr14"
IMAGE_DIR="${NIH_BASE_DIR}/dataset"
CXAS_DIR="${NIH_BASE_DIR}/CXAS/cxas"
OUTPUT_DIR="${NIH_BASE_DIR}/output_nih"

echo "============================================================="
echo "CXReasonBench Generation for NIH CXR-14 Dataset"
echo "Pipeline: dx_by_dicoms → viz CSVs → segmasks → Path2/Path1"
echo "============================================================="
echo ""

# Check if base directory exists
if [ ! -d "${NIH_BASE_DIR}" ]; then
    echo "Error: Base directory ${NIH_BASE_DIR} does not exist!"
    exit 1
fi

# Create output directory
echo "Creating output directory..."
mkdir -p "${OUTPUT_DIR}"

# Step 1: Generate dx_by_dicoms.json for NIH dataset
echo ""
echo "Step 1/5: Generating dx_by_dicoms.json for NIH dataset..."
echo "---------------------------------------------------"
python generate_dx_by_dicoms_nih.py \
    --chexstruct_base_dir "${CHEXSTRUCT_DIR}" \
    --output_file "${OUTPUT_DIR}/dx_by_dicoms.json"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate dx_by_dicoms.json"
    exit 1
fi

# Step 2: Create visualization CSV files
echo ""
echo "Step 2/5: Creating visualization CSV files..."
echo "---------------------------------------------------"
python create_viz_csvs_nih.py \
    --dx_by_dicoms_file "${OUTPUT_DIR}/dx_by_dicoms.json" \
    --output_dir "${NIH_BASE_DIR}/nih-cxr14_viz" \
    --nih_csv_dir "${CHEXSTRUCT_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create visualization CSVs"
    exit 1
fi

# Step 3: Generate segmentation masks visualization
echo ""
echo "Step 3/5: Generating segmentation masks..."
echo "---------------------------------------------------"
python bodypart_segmask.py \
    --dataset_name nih-cxr14 \
    --saved_base_dir "${NIH_BASE_DIR}" \
    --save_base_dir "${OUTPUT_DIR}" \
    --nih_image_base_dir "${IMAGE_DIR}" \
    --cxas_base_dir "${CXAS_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate segmentation masks"
    echo "This is usually due to:"
    echo "  1. detectron2 not installed - Run: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    echo "  2. Pillow version incompatibility - Run: pip install 'pillow==9.5.0'"
    echo "  3. Missing CXAS files for some images"
    exit 1
fi

# Step 4: Generate QA pairs - Path 2 (Guided Reasoning)
echo ""
echo "Step 4/5: Generating QA pairs (Path 2 - Guided Reasoning)..."
echo "---------------------------------------------------"
python generate_benchmark.py \
    --dataset_name nih-cxr14 \
    --inference_path path2 \
    --chexstruct_base_dir "${CHEXSTRUCT_DIR}" \
    --cxreasonbench_base_dir "${OUTPUT_DIR}" \
    --nih_image_base_dir "${IMAGE_DIR}" \
    --save_base_dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate Path 2 QA pairs"
    exit 1
fi

# Step 5: Generate QA pairs - Path 1 (Direct Reasoning)
echo ""
echo "Step 5/5: Generating QA pairs (Path 1 - Direct Reasoning)..."
echo "---------------------------------------------------"
python generate_benchmark.py \
    --dataset_name nih-cxr14 \
    --inference_path path1 \
    --chexstruct_base_dir "${CHEXSTRUCT_DIR}" \
    --cxreasonbench_base_dir "${OUTPUT_DIR}" \
    --nih_image_base_dir "${IMAGE_DIR}" \
    --save_base_dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate QA pairs"
    exit 1
fi

echo ""
echo "==================================================="
echo "✓ Done! Output saved to ${OUTPUT_DIR}"
echo "==================================================="
