#!/bin/bash
# 執行前檢查清單腳本

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CXReasonBench 執行前檢查清單${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

check_item() {
    local description=$1
    local command=$2
    
    echo -n "[ ] $description ... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗${NC}"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
        return 1
    fi
}

check_python_package() {
    python3 -c "import $1" 2>/dev/null
}

# Python 檢查
echo -e "${YELLOW}Python 環境:${NC}"
check_item "Python 3.9+" "python3 --version | grep -E '3\.(9|1[0-9])'"
check_item "pandas" "check_python_package pandas"
check_item "numpy" "check_python_package numpy"
check_item "tqdm" "check_python_package tqdm"
check_item "PIL (Pillow)" "check_python_package PIL"
check_item "cv2 (opencv-python)" "check_python_package cv2"
check_item "matplotlib" "check_python_package matplotlib"
check_item "scipy" "check_python_package scipy"

echo -n "[ ] detectron2 (建議) ... "
if check_python_package detectron2; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠ (非必須，但建議安裝)${NC}"
fi

echo ""

# 目錄檢查
echo -e "${YELLOW}數據目錄:${NC}"
check_item "基本目錄" "test -d /mnt/d/CXReasonBench"
check_item "CheXStruct輸出" "test -d /mnt/d/CXReasonBench/nih_cxr14"
check_item "NIH影像目錄" "test -d /mnt/d/CXReasonBench/dataset"
check_item "CXAS分割結果" "test -d /mnt/d/CXReasonBench/CXAS/cxas"

echo ""

# 檔案檢查
echo -e "${YELLOW}關鍵檔案:${NC}"
check_item "cardiomegaly.csv" "test -f /mnt/d/CXReasonBench/nih_cxr14/cardiomegaly.csv"
check_item "inclusion.csv" "test -f /mnt/d/CXReasonBench/nih_cxr14/inclusion.csv"
check_item "至少一個影像" "find /mnt/d/CXReasonBench/dataset -name '*.png' | head -1"
check_item "至少一個CXAS目錄" "ls -d /mnt/d/CXReasonBench/CXAS/cxas/*/ 2>/dev/null | head -1"

echo ""

# 腳本檢查
echo -e "${YELLOW}執行腳本:${NC}"
check_item "主腳本存在" "test -f /mnt/d/CXReasonBench/Benchmark/generation/run_nih_benchmark_generation.sh"
check_item "腳本可執行" "test -x /mnt/d/CXReasonBench/Benchmark/generation/run_nih_benchmark_generation.sh"

echo ""

# 磁碟空間
echo -e "${YELLOW}系統資源:${NC}"
AVAILABLE_GB=$(df -BG /mnt/d | tail -1 | awk '{print $4}' | sed 's/G//')
echo -n "[ ] 可用空間 (需要 >50GB) ... "
if [ "$AVAILABLE_GB" -gt 50 ]; then
    echo -e "${GREEN}✓ (${AVAILABLE_GB}GB 可用)${NC}"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "${RED}✗ (僅 ${AVAILABLE_GB}GB 可用)${NC}"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "檢查結果: ${GREEN}${CHECKS_PASSED} 通過${NC} / ${RED}${CHECKS_FAILED} 失敗${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ 所有檢查通過！可以開始執行。${NC}"
    echo ""
    echo "執行命令:"
    echo "  bash run_nih_benchmark_generation.sh"
    echo ""
    exit 0
elif [ $CHECKS_FAILED -le 2 ]; then
    echo -e "${YELLOW}⚠️  有少數檢查失敗，但可能不影響執行。${NC}"
    echo ""
    echo "建議修復後再執行，或嘗試執行看看:"
    echo "  bash run_nih_benchmark_generation.sh"
    echo ""
    exit 1
else
    echo -e "${RED}❌ 多個檢查失敗，請先修復這些問題。${NC}"
    echo ""
    echo "常見解決方案:"
    echo "  1. 安裝缺少的Python套件: pip install -r requirements.txt"
    echo "  2. 檢查數據目錄路徑是否正確"
    echo "  3. 確保已執行CheXStruct pipeline"
    echo ""
    exit 2
fi
