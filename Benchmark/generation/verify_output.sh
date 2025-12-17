#!/bin/bash
# =============================================================================
# CXReasonBench 輸出驗證腳本
# =============================================================================
# 用途: 檢查生成的benchmark檔案是否完整
# =============================================================================

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

OUTPUT_DIR="/mnt/d/CXReasonBench/output_nih"

print_header() {
    echo -e "${BLUE}=============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================${NC}"
}

print_check() {
    echo -e "${YELLOW}[檢查]${NC} $1"
}

print_ok() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}  ✗ $1${NC}"
}

print_header "CXReasonBench 輸出驗證"
echo ""

# =============================================================================
# 檢查基本檔案
# =============================================================================

print_check "檢查基本檔案..."

if [ -f "${OUTPUT_DIR}/dx_by_dicoms.json" ]; then
    TASK_COUNT=$(python3 -c "import json; print(len(json.load(open('${OUTPUT_DIR}/dx_by_dicoms.json'))))" 2>/dev/null)
    print_ok "dx_by_dicoms.json 存在 (包含 ${TASK_COUNT} 個診斷任務)"
else
    print_error "dx_by_dicoms.json 不存在"
fi

echo ""

# =============================================================================
# 檢查診斷任務
# =============================================================================

print_check "檢查診斷任務..."

EXPECTED_TASKS=(
    "inclusion"
    "inspiration"
    "rotation"
    "projection"
    "cardiomegaly"
    "mediastinal_widening"
    "carina_angle"
    "trachea_deviation"
    "aortic_knob_enlargement"
    "ascending_aorta_enlargement"
    "descending_aorta_enlargement"
    "descending_aorta_tortuous"
)

for task in "${EXPECTED_TASKS[@]}"; do
    if [ -d "${OUTPUT_DIR}/qa/${task}" ]; then
        print_ok "${task}"
    else
        print_warn "${task} 目錄不存在"
    fi
done

echo ""

# =============================================================================
# 檢查QA對
# =============================================================================

print_check "檢查QA對..."

TOTAL_PATH1=0
TOTAL_PATH2=0

for task in "${EXPECTED_TASKS[@]}"; do
    if [ -d "${OUTPUT_DIR}/qa/${task}" ]; then
        PATH1_COUNT=$(find "${OUTPUT_DIR}/qa/${task}/path1" -name "*.json" -type f 2>/dev/null | wc -l)
        PATH2_COUNT=$(find "${OUTPUT_DIR}/qa/${task}/path2" -name "*.json" -type f 2>/dev/null | wc -l)
        
        TOTAL_PATH1=$((TOTAL_PATH1 + PATH1_COUNT))
        TOTAL_PATH2=$((TOTAL_PATH2 + PATH2_COUNT))
        
        if [ $PATH1_COUNT -gt 0 ] && [ $PATH2_COUNT -gt 0 ]; then
            print_ok "${task}: Path1=${PATH1_COUNT}, Path2=${PATH2_COUNT}"
        elif [ $PATH1_COUNT -gt 0 ]; then
            print_warn "${task}: Path1=${PATH1_COUNT}, Path2=0 (缺少Path2)"
        elif [ $PATH2_COUNT -gt 0 ]; then
            print_warn "${task}: Path1=0 (缺少Path1), Path2=${PATH2_COUNT}"
        else
            print_error "${task}: 無QA對"
        fi
    fi
done

echo ""
echo "  總計:"
echo "    Path1 QA對: ${TOTAL_PATH1}"
echo "    Path2 QA對: ${TOTAL_PATH2}"
echo "    總QA對數: $((TOTAL_PATH1 + TOTAL_PATH2))"

echo ""

# =============================================================================
# 檢查可視化檔案
# =============================================================================

print_check "檢查可視化檔案..."

SEGMASK_COUNT=$(find "${OUTPUT_DIR}/segmask_bodypart" -name "*.png" -type f 2>/dev/null | wc -l)
if [ $SEGMASK_COUNT -gt 0 ]; then
    print_ok "分割遮罩: ${SEGMASK_COUNT} 個"
else
    print_warn "分割遮罩: 無檔案"
fi

PNT_COUNT=$(find "${OUTPUT_DIR}/pnt_on_cxr" -name "*.png" -type f 2>/dev/null | wc -l)
if [ $PNT_COUNT -gt 0 ]; then
    print_ok "地標點圖: ${PNT_COUNT} 個"
else
    print_warn "地標點圖: 無檔案"
fi

echo ""

# =============================================================================
# 檢查Path1階段完整性
# =============================================================================

print_check "檢查Path1階段完整性..."

PATH1_STAGES=("init" "stage1" "stage2" "stage3" "stage4")
INCOMPLETE_TASKS=0

for task in "${EXPECTED_TASKS[@]}"; do
    MISSING_STAGES=()
    for stage in "${PATH1_STAGES[@]}"; do
        if [ ! -d "${OUTPUT_DIR}/qa/${task}/path1/${stage}" ]; then
            MISSING_STAGES+=("${stage}")
        fi
    done
    
    if [ ${#MISSING_STAGES[@]} -eq 0 ]; then
        # print_ok "${task}: 所有階段完整"
        :
    else
        print_warn "${task}: 缺少階段 ${MISSING_STAGES[*]}"
        INCOMPLETE_TASKS=$((INCOMPLETE_TASKS + 1))
    fi
done

if [ $INCOMPLETE_TASKS -eq 0 ]; then
    print_ok "所有任務的Path1階段完整"
else
    print_warn "${INCOMPLETE_TASKS} 個任務的Path1不完整"
fi

echo ""

# =============================================================================
# 檢查Path2階段完整性
# =============================================================================

print_check "檢查Path2階段完整性..."

PATH2_STAGES=("stage1" "stage2" "stage3")
INCOMPLETE_TASKS_P2=0

for task in "${EXPECTED_TASKS[@]}"; do
    MISSING_STAGES=()
    for stage in "${PATH2_STAGES[@]}"; do
        if [ ! -d "${OUTPUT_DIR}/qa/${task}/path2/${stage}" ]; then
            MISSING_STAGES+=("${stage}")
        fi
    done
    
    if [ ${#MISSING_STAGES[@]} -eq 0 ]; then
        # print_ok "${task}: 所有階段完整"
        :
    else
        print_warn "${task}: 缺少階段 ${MISSING_STAGES[*]}"
        INCOMPLETE_TASKS_P2=$((INCOMPLETE_TASKS_P2 + 1))
    fi
done

if [ $INCOMPLETE_TASKS_P2 -eq 0 ]; then
    print_ok "所有任務的Path2階段完整"
else
    print_warn "${INCOMPLETE_TASKS_P2} 個任務的Path2不完整"
fi

echo ""

# =============================================================================
# 範例QA對預覽
# =============================================================================

print_check "範例QA對預覽..."

SAMPLE_FILE=$(find "${OUTPUT_DIR}/qa/cardiomegaly/path1/init" -name "*.json" -type f 2>/dev/null | head -1)

if [ -n "$SAMPLE_FILE" ]; then
    echo ""
    echo "  檔案: ${SAMPLE_FILE##*/}"
    echo "  內容:"
    python3 -m json.tool "$SAMPLE_FILE" 2>/dev/null | head -20
    echo "  ..."
else
    print_warn "找不到範例檔案"
fi

echo ""

# =============================================================================
# 總結
# =============================================================================

print_header "驗證總結"

echo ""
echo "狀態摘要:"
echo "  ✓ 完成的診斷任務: $(ls -d ${OUTPUT_DIR}/qa/*/ 2>/dev/null | wc -l) / 12"
echo "  ✓ Path1 完整任務: $((12 - INCOMPLETE_TASKS)) / 12"
echo "  ✓ Path2 完整任務: $((12 - INCOMPLETE_TASKS_P2)) / 12"
echo "  ✓ 總QA對數: $((TOTAL_PATH1 + TOTAL_PATH2))"
echo "  ✓ 可視化檔案: $((SEGMASK_COUNT + PNT_COUNT))"

echo ""

if [ $INCOMPLETE_TASKS -eq 0 ] && [ $INCOMPLETE_TASKS_P2 -eq 0 ] && [ $TOTAL_PATH1 -gt 0 ] && [ $TOTAL_PATH2 -gt 0 ]; then
    echo -e "${GREEN}✅ 所有檢查通過！Benchmark生成完整。${NC}"
    exit 0
elif [ $TOTAL_PATH1 -gt 0 ] || [ $TOTAL_PATH2 -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Benchmark部分完成，但有一些缺失。${NC}"
    exit 1
else
    echo -e "${RED}❌ Benchmark生成不完整或失敗。${NC}"
    exit 2
fi
