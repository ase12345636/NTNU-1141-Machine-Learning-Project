#!/bin/bash
# =============================================================================
# CXReasonBench 一鍵生成腳本 - NIH CXR-14 數據集
# =============================================================================
# 作者: 自動生成
# 日期: 2025-12-17
# 用途: 完整執行從CheXStruct輸出到QA對生成的整個Pipeline
# =============================================================================

set -e  # 遇到錯誤立即停止

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 配置區域 - 請根據您的實際路徑修改
# =============================================================================

# 基本路徑 (WSL格式: /mnt/d 而不是 d:)
BASE_DIR="/mnt/d/CXReasonBench"
CHEXSTRUCT_DIR="${BASE_DIR}/nih_cxr14"
IMAGE_DIR="${BASE_DIR}/dataset"
CXAS_DIR="${BASE_DIR}/CXAS/cxas"
OUTPUT_DIR="${BASE_DIR}/output_nih"

# Python環境 (如果使用conda環境，請取消註釋並修改)
# CONDA_ENV="cxrb_gen"
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ${CONDA_ENV}

# =============================================================================
# 函數定義
# =============================================================================

print_header() {
    echo -e "${BLUE}=============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}>>> $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        print_error "檔案不存在: $1"
        return 1
    fi
    return 0
}

check_dir_exists() {
    if [ ! -d "$1" ]; then
        print_error "目錄不存在: $1"
        return 1
    fi
    return 0
}

# =============================================================================
# 環境檢查
# =============================================================================

print_header "環境檢查"

print_step "檢查基本目錄..."
check_dir_exists "${BASE_DIR}" || exit 1
check_dir_exists "${CHEXSTRUCT_DIR}" || exit 1
check_dir_exists "${IMAGE_DIR}" || exit 1
check_dir_exists "${CXAS_DIR}" || exit 1

print_step "檢查Python依賴..."
python -c "import pandas, numpy, tqdm, PIL, cv2" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "基本依賴檢查通過"
else
    print_error "缺少必要的Python套件"
    echo "請執行: pip install pandas numpy tqdm pillow opencv-python"
    exit 1
fi

# 檢查detectron2 (非必須，但建議安裝)
python -c "import detectron2" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "detectron2 未安裝，步驟3可能會失敗"
    print_warning "安裝方法: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
fi

print_success "環境檢查完成"
echo ""

# =============================================================================
# 創建輸出目錄
# =============================================================================

print_header "創建輸出目錄"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/qa"
mkdir -p "${OUTPUT_DIR}/segmask_bodypart"
mkdir -p "${OUTPUT_DIR}/pnt_on_cxr"
print_success "輸出目錄創建完成"
echo ""

# =============================================================================
# 步驟 1: 生成 dx_by_dicoms.json
# =============================================================================

print_header "步驟 1/6: 生成 dx_by_dicoms.json"
print_step "這個檔案將診斷任務映射到對應的DICOM ID..."

DX_BY_DICOMS_FILE="${OUTPUT_DIR}/dx_by_dicoms.json"

if [ -f "${DX_BY_DICOMS_FILE}" ]; then
    print_warning "dx_by_dicoms.json 已存在，跳過此步驟"
    print_warning "如需重新生成，請刪除檔案: ${DX_BY_DICOMS_FILE}"
else
    python generate_dx_by_dicoms_nih.py \
        --chexstruct_base_dir "${CHEXSTRUCT_DIR}" \
        --output_file "${DX_BY_DICOMS_FILE}"
    
    if [ $? -eq 0 ]; then
        print_success "dx_by_dicoms.json 生成成功"
    else
        print_error "dx_by_dicoms.json 生成失敗"
        exit 1
    fi
fi

echo ""

# =============================================================================
# 步驟 2: 創建可視化 CSV 檔案
# =============================================================================

print_header "步驟 2/6: 創建可視化 CSV 檔案 (從CXAS計算viz欄位)"
print_step "生成每個診斷任務需要可視化的影像列表，並從CXAS masks計算viz欄位..."

VIZ_DIR="${BASE_DIR}/nih-cxr14_viz"

if [ -d "${VIZ_DIR}" ] && [ "$(ls -A ${VIZ_DIR}/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
    print_warning "可視化CSV檔案已存在，跳過此步驟"
    print_warning "如需重新生成，請刪除目錄: ${VIZ_DIR}"
else
    python create_viz_csvs_nih.py \
        --dx_by_dicoms_file "${DX_BY_DICOMS_FILE}" \
        --output_dir "${VIZ_DIR}" \
        --nih_csv_dir "${CHEXSTRUCT_DIR}" \
        --cxas_dir "${CXAS_DIR}"
    
    if [ $? -eq 0 ]; then
        print_success "可視化CSV檔案創建成功（包含從CXAS計算的viz欄位）"
        echo "位置: ${VIZ_DIR}"
        
        # Create separate dx_by_dicoms for Path 2 (only images with valid viz data)
        print_step "創建 dx_by_dicoms_path2.json (僅包含viz數據完整的圖像)..."
        DX_BY_DICOMS_PATH2="${OUTPUT_DIR}/dx_by_dicoms_path2.json"
        python update_dx_by_dicoms_from_viz.py \
            --viz_dir "${VIZ_DIR}" \
            --dx_by_dicoms_file "${DX_BY_DICOMS_FILE}" \
            --output_path2_file "${DX_BY_DICOMS_PATH2}"
        
        if [ $? -eq 0 ]; then
            print_success "dx_by_dicoms_path2.json 已創建"
            echo "  Path 1 (reasoning): 使用所有圖像 (${DX_BY_DICOMS_FILE})"
            echo "  Path 2 (guidance):  僅使用viz完整圖像 (${DX_BY_DICOMS_PATH2})"
        else
            print_error "創建 dx_by_dicoms_path2.json 失敗"
            exit 1
        fi
    else
        print_error "可視化CSV檔案創建失敗"
        exit 1
    fi
fi

echo ""

# # # =============================================================================
# # # 步驟 3: 生成分割遮罩可視化
# # # =============================================================================

# # print_header "步驟 3/6: 生成分割遮罩可視化"
# # print_step "為每個診斷任務創建解剖結構分割遮罩的可視化..."
# # print_warning "這一步需要 detectron2，可能需要較長時間..."

# # SEGMASK_DIR="${OUTPUT_DIR}/segmask_bodypart"

# # # 設定並行處理的worker數量 (根據CPU核心數調整)
NUM_WORKERS=8
# # print_step "使用 ${NUM_WORKERS} 個並行worker加速處理"

# # # 檢查是否已有部分輸出
# # EXISTING_SEGMASKS=$(find "${SEGMASK_DIR}" -type f -name "*.png" 2>/dev/null | wc -l)
# # if [ ${EXISTING_SEGMASKS} -gt 100 ]; then
# #     print_warning "已存在 ${EXISTING_SEGMASKS} 個分割遮罩檔案"
# #     print_step "bodypart_segmask.py 會自動跳過完成率>80%的診斷任務"
# # fi

# # python bodypart_segmask.py \
# #     --dataset_name nih-cxr14 \
# #     --saved_base_dir "${BASE_DIR}" \
# #     --save_base_dir "${OUTPUT_DIR}" \
# #     --nih_image_base_dir "${IMAGE_DIR}" \
# #     --cxas_base_dir "${CXAS_DIR}" \
# #     --chexmask_base_dir "${BASE_DIR}/CXAS/chexmask" \
# #     --num_workers ${NUM_WORKERS}

# # if [ $? -eq 0 ]; then
# #     print_success "分割遮罩可視化生成成功"
# # else
# #     print_error "分割遮罩可視化生成失敗"
# #     print_warning "可能原因: detectron2未安裝或CXAS檔案缺失"
# #     exit 1
# # fi

# # echo ""

# # =============================================================================
# # 步驟 4: 生成地標點可視化 (Path2需要)
# # =============================================================================

# print_header "步驟 4/6: 生成地標點可視化 (Path2需要)"
# print_step "為Path2創建帶有解剖地標和座標的可視化..."
# print_warning "這一步也需要 detectron2..."

# PNT_DIR="${OUTPUT_DIR}/pnt_on_cxr"

# print_step "使用 ${NUM_WORKERS} 個並行worker加速處理"

# # 檢查是否已有部分輸出
# EXISTING_PNTS=$(find "${PNT_DIR}" -type f -name "*.png" 2>/dev/null | wc -l)
# if [ ${EXISTING_PNTS} -gt 100 ]; then
#     print_warning "已存在 ${EXISTING_PNTS} 個地標點檔案"
#     print_step "point_on_cxr.py 會自動跳過完成率>80%的診斷任務"
# fi

# python point_on_cxr.py \
#     --dataset_name nih-cxr14 \
#     --saved_base_dir "${BASE_DIR}" \
#     --save_base_dir "${OUTPUT_DIR}" \
#     --nih_image_base_dir "${IMAGE_DIR}" \
#     --cxas_base_dir "${CXAS_DIR}" \
#     --num_workers ${NUM_WORKERS}

# if [ $? -eq 0 ]; then
#     print_success "地標點可視化生成成功"
# else
#     print_error "地標點可視化生成失敗"
#     print_warning "Path2將無法執行，但Path1仍可繼續"
# fi

# echo ""

# =============================================================================
# 步驟 6: 生成 Path1 QA對 (直接推理)
# =============================================================================

print_header "步驟 6/6: 生成 Path1 QA對 (直接推理)"
print_step "生成無視覺引導的問答對..."

python generate_benchmark.py \
    --dataset_name nih-cxr14 \
    --inference_path path1 \
    --chexstruct_base_dir "${CHEXSTRUCT_DIR}" \
    --cxreasonbench_base_dir "${OUTPUT_DIR}" \
    --nih_image_base_dir "${IMAGE_DIR}" \
    --save_base_dir "${OUTPUT_DIR}" \
    --workers "${NUM_WORKERS}"

if [ $? -eq 0 ]; then
    print_success "Path1 QA對生成成功"
    
    # 統計生成的QA對數量
    PATH1_QA_COUNT=$(find "${PATH2_DIR}" -path "*/path1/*.json" -type f 2>/dev/null | wc -l)
    echo "生成的Path1 QA對數量: ${PATH1_QA_COUNT}"
else
    print_error "Path1 QA對生成失敗"
    exit 1
fi

echo ""

# =============================================================================
# 步驟 5: 生成 Path2 QA對 (引導式推理)
# =============================================================================

print_header "步驟 5/6: 生成 Path2 QA對 (引導式推理)"
print_step "生成帶有視覺引導的問答對..."

PATH2_DIR="${OUTPUT_DIR}/qa"

python generate_benchmark.py \
    --dataset_name nih-cxr14 \
    --inference_path path2 \
    --chexstruct_base_dir "${CHEXSTRUCT_DIR}" \
    --cxreasonbench_base_dir "${OUTPUT_DIR}" \
    --nih_image_base_dir "${IMAGE_DIR}" \
    --save_base_dir "${OUTPUT_DIR}" \
    --workers "${NUM_WORKERS}"

if [ $? -eq 0 ]; then
    print_success "Path2 QA對生成成功"
    
    # 統計生成的QA對數量
    PATH2_QA_COUNT=$(find "${PATH2_DIR}" -path "*/path2/*.json" -type f 2>/dev/null | wc -l)
    echo "生成的Path2 QA對數量: ${PATH2_QA_COUNT}"
else
    print_error "Path2 QA對生成失敗"
    exit 1
fi

echo ""


# # =============================================================================
# # 完成總結
# # =============================================================================

# print_header "✨ 全部完成！"

# echo -e "${GREEN}所有步驟執行成功！${NC}"
# echo ""
# echo "輸出檔案位置:"
# echo "  📁 主輸出目錄: ${OUTPUT_DIR}"
# echo "  📄 診斷映射: ${OUTPUT_DIR}/dx_by_dicoms.json"
# echo "  🖼️  分割遮罩: ${OUTPUT_DIR}/segmask_bodypart/"
# echo "  📍 地標點: ${OUTPUT_DIR}/pnt_on_cxr/"
# echo "  💬 QA對: ${OUTPUT_DIR}/qa/"
# echo ""

# # 顯示統計信息
# if [ -f "${DX_BY_DICOMS_FILE}" ]; then
#     echo "統計信息:"
#     echo "  診斷任務數: $(python -c "import json; print(len(json.load(open('${DX_BY_DICOMS_FILE}'))))" 2>/dev/null || echo "N/A")"
    
#     TOTAL_QA=$(find "${OUTPUT_DIR}/qa" -name "*.json" -type f 2>/dev/null | wc -l)
#     echo "  總QA對數: ${TOTAL_QA}"
    
#     SEGMASK_COUNT=$(find "${OUTPUT_DIR}/segmask_bodypart" -name "*.png" -type f 2>/dev/null | wc -l)
#     echo "  分割遮罩數: ${SEGMASK_COUNT}"
    
#     PNT_COUNT=$(find "${OUTPUT_DIR}/pnt_on_cxr" -name "*.png" -type f 2>/dev/null | wc -l)
#     echo "  地標點圖數: ${PNT_COUNT}"
# fi

# echo ""
# echo -e "${BLUE}下一步:${NC}"
# echo "  1. 檢查輸出檔案: cd ${OUTPUT_DIR}"
# echo "  2. 執行評估: 參考 Benchmark/Evaluation/README.md"
# echo ""

# print_success "Pipeline執行完成！🎉"
