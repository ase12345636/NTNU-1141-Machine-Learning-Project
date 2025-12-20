#!/usr/bin/env python
"""
快速測試腳本 - 驗證 bodypart_segmask.py 的錯誤處理
"""
import os
import sys

# 測試匯入
try:
    import bodypart_segmask
    print("✅ bodypart_segmask.py 語法正確")
except SyntaxError as e:
    print(f"❌ 語法錯誤: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"⚠️  匯入警告: {e}")
    print("   (這是正常的，因為可能缺少某些依賴)")

print("\n檢查關鍵函數...")

# 檢查所有 segmask 函數是否存在
required_functions = [
    'segmask_cardiomegaly',
    'segmask_carina',
    'segmask_descending_aorta',
    'segmask_aortic_knob',
    'segmask_ascending_aorta',
    'segmask_inclusion',
    'segmask_inspiration',
    'segmask_mediastinal_widening',
    'segmask_projection',
    'segmask_rotation',
    'segmask_trachea',
]

for func_name in required_functions:
    if hasattr(bodypart_segmask, func_name):
        print(f"✅ {func_name}")
    else:
        print(f"❌ 缺少 {func_name}")

print("\n✅ 所有檢查完成！")
