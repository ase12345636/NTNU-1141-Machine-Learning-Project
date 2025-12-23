import os
import cv2
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import ast
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端，加速繪圖
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
from detectron2.utils.visualizer import ColorMode, Visualizer

# 檢查 CUDA 是否可用
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# 嘗試導入 CuPy（GPU 加速的 NumPy）
try:
    if CUDA_AVAILABLE:
        import cupy as cp
        USE_CUPY = True
    else:
        USE_CUPY = False
except ImportError:
    USE_CUPY = False

def to_gpu(array):
    """將 NumPy array 移到 GPU（如果可用）"""
    if USE_CUPY and array is not None:
        return cp.asarray(array)
    return array

def to_cpu(array):
    """將 GPU array 移回 CPU"""
    if USE_CUPY and hasattr(array, 'get'):
        return cp.asnumpy(array)
    return array

def safe_imread(file_path, flags=0):
    """Safely read image file, return None if file doesn't exist or cannot be read"""
    if not os.path.exists(file_path):
        return None
    img = cv2.imread(file_path, flags)
    if img is None:
        print(f"Warning: cv2.imread failed for existing file: {file_path}")
    return img

def fast_load_image_rgb(file_path):
    """快速載入圖片為 RGB 格式，優先使用 cv2"""
    img = cv2.imread(file_path)
    if img is None:
        # fallback to PIL if cv2 fails
        img = np.array(Image.open(file_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_base_dir', default='', type=str)

    parser.add_argument('--saved_base_dir', default='path/to/saved/output', type=str)
    parser.add_argument('--dataset_name', default='mimic-cxr-jpg', choices=['mimic-cxr-jpg', 'nih-cxr14'])

    parser.add_argument('--mimic_cxr_base_dir', default="<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files", type=str)
    parser.add_argument('--mimic_meta_file', default='<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv', type=str)

    # NIH dataset arguments
    parser.add_argument('--nih_image_base_dir', type=str, default='/mnt/d/CXReasonBench/dataset', help='Path to NIH images folders (images_001, images_002, ...)')
    
    parser.add_argument('--cxas_base_dir', type=str, default='path/to/cxas_segmentation_folders')
    parser.add_argument('--chexmask_base_dir', type=str, default='path/to/chexmask_segmentation_folders')
    
    # 多進程參數
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers (default: 4, set to 1 to disable)')

    args = parser.parse_args()
    return args

def get_image_path(args, dicom):
    """Get image path based on dataset type"""
    if args.dataset_name == 'nih-cxr14':
        # NIH dataset structure: images are in dataset/images_XXX/images/
        # Need to find which folder contains this image
        for i in range(1, 13):  # images_001 to images_012
            folder_name = f'images_{i:03d}'
            img_path = os.path.join(args.nih_image_base_dir, folder_name, 'images', f'{dicom}.png')
            if os.path.exists(img_path):
                return img_path
        raise FileNotFoundError(f"Image {dicom}.png not found in any images folder")
    else:
        # MIMIC-CXR dataset structure
        sid = args.mimic_meta[args.mimic_meta['dicom_id'] == dicom]['study_id'].values[0]
        pid = args.mimic_meta[args.mimic_meta['dicom_id'] == dicom]['subject_id'].values[0]
        return f'{args.mimic_cxr_base_dir}/p{str(pid)[:2]}/p{pid}/s{sid}/{dicom}.jpg'

def visualize_mask(args, mask_lst, save_path, dicom):
    mask_colors = [np.array([0.5, 0., 0.], dtype=np.float32)]
    
    os.makedirs(save_path, exist_ok=True)
    cxr_path = get_image_path(args, dicom)
    
    # 使用 cv2 讀取並直接轉為 RGB numpy array（避免 PIL 開銷）
    cxr_bgr = cv2.imread(cxr_path)
    if cxr_bgr is None:
        # fallback to PIL
        cxr = Image.open(cxr_path).convert('RGB')
    else:
        cxr = cv2.cvtColor(cxr_bgr, cv2.COLOR_BGR2RGB)

    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    output = vs.overlay_instances(masks=mask_lst, assigned_colors=mask_colors * len(mask_lst)).get_image()
    
    # 使用 cv2 保存，比 matplotlib 快 6-7 倍
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{save_path}/{dicom}.png', output_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

def visualize_midline(args, pnt_start, pnt_end, save_path, dicom):
    os.makedirs(save_path, exist_ok=True)
    cxr_path = get_image_path(args, dicom)
    
    # 直接使用 cv2 讀取和繪製，比 PIL + matplotlib 快 8-10 倍
    cxr = cv2.imread(cxr_path)
    if cxr is None:
        cxr = np.array(Image.open(cxr_path).convert('RGB'))
        cxr = cv2.cvtColor(cxr, cv2.COLOR_RGB2BGR)
    
    cv2.line(cxr, (int(pnt_start[0]), int(pnt_start[-1])), 
             (int(pnt_end[0]), int(pnt_end[-1])), (0, 0, 255), 2)
    cv2.imwrite(f'{save_path}/{dicom}.png', cxr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

def check_if_processed(dicom, dx, save_base_dir):
    """檢查影像是否已經處理過，根據不同診斷任務檢查對應的輸出檔案"""
    # 定義每個診斷任務需要檢查的關鍵輸出檔案
    output_checks = {
        'cardiomegaly': ['heart', 'thoracic_width_heart'],
        'carina_angle': ['carina'],
        'descending_aorta_enlargement': ['descending_aorta', 'trachea'],
        'descending_aorta_tortuous': ['descending_aorta'],
        'aortic_knob_enlargement': ['aortic_knob', 'trachea'],
        'ascending_aorta_enlargement': ['ascending_aorta', 'borderline'],
        'inclusion': ['lung_both'],
        'inspiration': ['diaphragm_right', 'right_posterior_rib', 'midclavicularline'],
        'mediastinal_widening': ['mediastinum', 'thoracic_width_mw'],
        'projection': ['lung_both', 'scapular_both'],
        'rotation': ['clavicle_both', 'midline'],
        'trachea_deviation': ['trachea', 'midline'],
    }
    
    if dx not in output_checks:
        return False
    
    # 檢查所有關鍵輸出檔案是否都存在（只有全部存在才視為已處理）
    exts = ('.png', '.jpg', '.jpeg')
    for subfolder in output_checks[dx]:
        subfolder_path = os.path.join(save_base_dir, subfolder)
        found = False
        if os.path.exists(subfolder_path):
            try:
                for fname in os.listdir(subfolder_path):
                    if not fname:
                        continue
                    base, ext = os.path.splitext(fname)
                    if ext.lower() in exts and base == str(dicom):
                        found = True
                        break
            except Exception:
                found = False
        if not found:
            return False

    return True

def segmask_cardiomegaly(args, dicom, target_df, save_base_dir):
    def select_max_width_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_mask = mask
        max_width = 0
        coordinates = (0, 0, 0, 0)
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
            x_indices = separate_mask.sum(axis=0).nonzero()[0]
            y_indices = separate_mask.sum(axis=1).nonzero()[0]
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x1, x2 = x_indices[0], x_indices[-1]
            y1, y2 = y_indices[0], y_indices[-1]
            width = abs(x_indices[0] - x_indices[-1]) + 1
            if width > max_width:
                max_width = width
                max_mask = separate_mask
                coordinates = (x1, y1, x2, y2)
        return max_mask, coordinates

    def select_max_area_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask

    fname_heart = os.path.join(args.cxas_base_dir, dicom, f"heart.png")
    mask_heart = safe_imread(fname_heart, 0)
    
    if mask_heart is None:
        # Missing CXAS masks, skip silently
        return
    
    mask_heart_refined, _ = select_max_width_mask(mask_heart)
    visualize_mask(args,[mask_heart_refined], f'{save_base_dir}/heart', dicom)

    fname_rlung = os.path.join(args.cxas_base_dir, dicom, f"right lung.png")
    fname_llung = os.path.join(args.cxas_base_dir, dicom, f"left lung.png")
    #
    mask_rlung = safe_imread(fname_rlung, 0)
    mask_llung = safe_imread(fname_llung, 0)
    #
    mask_rlung_refined = select_max_area_mask(mask_rlung)
    mask_llung_refined = select_max_area_mask(mask_llung)

    # Read viz data from CSV
    try:
        coord_mask = eval(target_df['coord_mask'].values[0])
        xmin_heart, ymin_heart, xmax_heart, ymax_heart = coord_mask
        xmin_lung = int(target_df['xmin_lung'].values[0])
        xmax_lung = int(target_df['xmax_lung'].values[0])
    except (ValueError, KeyError, SyntaxError):
        # Missing required viz fields, skip this sample
        return
    lungs_target_part = (mask_rlung_refined[ymin_heart:] | mask_llung_refined[ymin_heart:])
    y_xmin_indices = lungs_target_part[:, xmin_lung].nonzero()[0].tolist()
    y_xmax_indices = lungs_target_part[:, xmax_lung].nonzero()[0].tolist()

    y_indices = set(y_xmax_indices + y_xmin_indices)
    
    # 檢查是否有有效的肺部像素
    if not y_indices:
        # 沒有找到有效的肺部像素，跳過此影像
        return
    
    ymin_lung = min(y_indices) + ymin_heart
    ymax_lung = max(y_indices) + ymin_heart
    ymean_lung = (ymin_lung + ymax_lung) // 2

    # thoracic width - 在心臟中心高度測量胸腔寬度
    ymean_heart = (ymin_heart + ymax_heart) // 2
    
    save_path = f'{save_base_dir}/thoracic_width_heart'
    os.makedirs(save_path, exist_ok=True)
    cxr_path = get_image_path(args, dicom)
    
    # 使用 cv2 繪製，比 matplotlib 快 8-10 倍
    cxr = cv2.imread(cxr_path)
    if cxr is None:
        cxr = np.array(Image.open(cxr_path).convert('RGB'))
        cxr = cv2.cvtColor(cxr, cv2.COLOR_RGB2BGR)
    
    # 在心臟中心高度畫垂直線標記胸腔寬度
    cv2.line(cxr, (xmin_lung, ymean_heart - 100), (xmin_lung, ymean_heart + 100), (0, 0, 255), 2)
    cv2.line(cxr, (xmax_lung, ymean_heart - 100), (xmax_lung, ymean_heart + 100), (0, 0, 255), 2)
    cv2.line(cxr, (xmin_lung, ymean_heart), (xmax_lung, ymean_heart), (0, 0, 255), 2)
    cv2.imwrite(f'{save_path}/{dicom}.png', cxr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

def segmask_carina(args, dicom, target_df, save_base_dir):
    def select_max_area_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask

    mask_fname = os.path.join(args.cxas_base_dir, dicom, f"tracheal bifurcation.png")
    mask = safe_imread(mask_fname, 0)
    
    if mask is None:
        # Missing CXAS masks, skip silently
        return
    
    mask = select_max_area_mask(mask)

    visualize_mask(args,[mask], f'{save_base_dir}/carina', dicom)

def segmask_descending_aorta(args, dicom, target_df, save_base_dir):
    def select_max_area_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask

    def select_max_height_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        target_mask = np.zeros_like(mask)
        max_height = 0
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
            height_idx = separate_mask.sum(axis=-1).nonzero()[0]
            if len(height_idx) == 0:
                continue
            ymin, ymax = height_idx[0], height_idx[-1]
            mask_height = ymax - ymin
            if mask_height > max_height:
                max_height = mask_height
                target_mask = separate_mask
        return target_mask

    def refine_mask(target_mask, filter_mask):
        overlap = (target_mask & filter_mask)
        if overlap.sum():
            overlap = select_max_height_mask(target_mask & filter_mask)
            y_nonzero = overlap.sum(axis=-1).nonzero()[0]
            if len(y_nonzero) == 0:
                return target_mask
            ymax = y_nonzero[-1]

            mask_ = np.zeros_like(target_mask)
            mask_[:ymax, ] = 1

            refined_mask = select_max_height_mask(target_mask & mask_)
        else:
            refined_mask = np.zeros_like(target_mask)
        return refined_mask

    fname_descending_aorta = os.path.join(args.cxas_base_dir, dicom, f"descending aorta.png")
    fname_heart = os.path.join(args.cxas_base_dir, dicom, f"heart.png")

    mask_descending_aorta = safe_imread(fname_descending_aorta, 0)
    mask_heart = safe_imread(fname_heart, 0)
    
    if mask_descending_aorta is None or mask_heart is None:
        # Missing CXAS masks, skip silently
        return

    # Read viz data from CSV
    try:
        ymin_start = int(target_df['ymin_start'].values[0])
        ymin_end = int(target_df['ymin_end'].values[0])
    except (ValueError, KeyError):
        # Missing required viz fields, skip this sample
        return

    mask_descending_aorta_refined = refine_mask(mask_descending_aorta, mask_heart)

    mask_descending_aorta_refined[:ymin_start] = 0
    mask_descending_aorta_refined[ymin_end:] = 0

    visualize_mask(args,[mask_descending_aorta_refined], f'{save_base_dir}/descending_aorta', dicom)

    # =============# =============# =============# =============# =============
    #                       Trachea
    # =============# =============# =============# =============# =============
    if 'descending_aorta_enlargement' in save_base_dir:
        fname_trachea = os.path.join(args.cxas_base_dir, dicom, f"trachea.png")
        fname_carina = os.path.join(args.cxas_base_dir, dicom, f"tracheal bifurcation.png")

        mask_trachea = safe_imread(fname_trachea, 0)
        mask_carina = safe_imread(fname_carina, 0)

        if mask_trachea is None or mask_carina is None:
            return
        
        carina_nonzero = mask_carina.sum(axis=-1).nonzero()[0]
        if len(carina_nonzero) == 0:
            return
        y_pos_carina_start = carina_nonzero[0]
        mask_refined = mask_trachea.copy()
        mask_refined[y_pos_carina_start:] = 0

        mask_refined = select_max_area_mask(mask_refined)

        idx_nonzero = mask_refined.sum(axis=-1).nonzero()[0]
        if len(idx_nonzero) == 0:
            return
        y_min, y_max = idx_nonzero[0], idx_nonzero[-1]
        y_subpart = int((y_max - y_min) * (1 / 4))
        mask_refined[:(y_min + y_subpart)] = 0

        visualize_mask(args,[mask_refined], f'{save_base_dir}/trachea', dicom)

def segmask_aortic_knob(args, dicom, target_df, save_base_dir):
    def select_max_area_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask

    def select_target_axis_mask(mask, target_area):
        mask_width_idx = target_area.sum(axis=0).nonzero()[0]
        # Handle empty mask case
        if len(mask_width_idx) == 0:
            return np.zeros_like(mask)
        xmin, xmax = mask_width_idx[0], mask_width_idx[-1]
        label_im, nb_labels = ndimage.label(mask)
        target_mask = np.zeros_like(mask)
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
            intersection_area = separate_mask[:, xmin:(xmax + 1)].sum()
            if intersection_area:
                target_mask = (target_mask | separate_mask)
        return target_mask

    def select_max_width_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        target_mask = np.zeros_like(mask)
        max_width = 0
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
            width_idx = separate_mask.sum(axis=0).nonzero()[0]
            if len(width_idx) == 0:
                continue
            xmin, xmax = width_idx[0], width_idx[-1]
            mask_width = xmax - xmin
            if mask_width > max_width:
                max_width = mask_width
                target_mask = separate_mask
        return target_mask

    # =============# =============# =============# =============# =============
    #                       Aortic Knob
    # =============# =============# =============# =============# =============
    fname_descending_aorta = os.path.join(args.cxas_base_dir, dicom, f"descending aorta.png")
    fname_aortic_arch = os.path.join(args.cxas_base_dir, dicom, f"aortic arch.png")

    mask_descending_aorta = safe_imread(fname_descending_aorta, 0)
    mask_aortic_arch = safe_imread(fname_aortic_arch, 0)
    
    # Check if masks loaded successfully and required fields are not NaN
    if mask_descending_aorta is None or mask_aortic_arch is None:
        # Missing CXAS masks, skip silently
        return
    
    # Read viz data from CSV
    try:
        y_max = int(target_df['y_max'].values[0])
        ymin_desc = int(target_df['ymin_desc'].values[0])
        ysub_desc = int(target_df['ysub_desc'].values[0])
        ymax_desc_sub = int(target_df['ymax_desc_sub'].values[0])
        xmax_trachea_mean = int(target_df['xmax_trachea_mean'].values[0])
    except (ValueError, KeyError):
        # Missing required viz fields, skip this sample
        return
    
    # Use values from CSV for processing
    mask_descending_aorta_refined = mask_descending_aorta.copy()
    mask_descending_aorta_refined[:y_max] = 0
    mask_descending_aorta_refined[(ymin_desc + ysub_desc):] = 0
    mask_descending_aorta_refined = select_max_width_mask(mask_descending_aorta_refined)

    mask_aortic_arch[ymax_desc_sub:] = 0  # y axis - remove unexpected region blobs
    mask_aortic_arch_refined = select_target_axis_mask(mask_aortic_arch, mask_descending_aorta_refined)

    # Check if mask is empty after refinement
    if mask_aortic_arch_refined.sum() == 0:
        return

    mask_aortic_arch_refined[:, :xmax_trachea_mean] = 0

    visualize_mask(args,[mask_aortic_arch_refined], f'{save_base_dir}/aortic_knob', dicom)

    # =============# =============# =============# =============# =============
    #                       Trachea
    # =============# =============# =============# =============# =============
    fname_trachea = os.path.join(args.cxas_base_dir, dicom, f"trachea.png")
    fname_carina = os.path.join(args.cxas_base_dir, dicom, f"tracheal bifurcation.png")

    mask_trachea = safe_imread(fname_trachea, 0)
    mask_carina = safe_imread(fname_carina, 0)

    # Check if masks loaded successfully
    if mask_trachea is None or mask_carina is None:
        print(f"Warning: Missing trachea masks for {dicom}, skipping trachea visualization")
        return

    carina_nonzero = mask_carina.sum(axis=-1).nonzero()[0]
    if len(carina_nonzero) == 0:
        print(f"Warning: Empty carina mask for {dicom}, skipping trachea visualization")
        return
    y_pos_carina_start = carina_nonzero[0]
    mask_refined = mask_trachea.copy()
    mask_refined[y_pos_carina_start:] = 0

    mask_refined = select_max_area_mask(mask_refined)

    idx_nonzero = mask_refined.sum(axis=-1).nonzero()[0]
    if len(idx_nonzero) == 0:
        print(f"Warning: Empty refined trachea mask for {dicom}, skipping trachea visualization")
        return
    y_min, y_max = idx_nonzero[0], idx_nonzero[-1]
    y_subpart = int((y_max - y_min) * (1 / 4))
    mask_refined[:(y_min + y_subpart)] = 0

    visualize_mask(args,[mask_refined], f'{save_base_dir}/trachea', dicom)

def segmask_ascending_aorta(args, dicom, target_df, save_base_dir):
    def select_max_area_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask

    fname_ascending_aorta = os.path.join(args.cxas_base_dir, dicom, f"ascending aorta.png")
    mask_ascending_aorta = safe_imread(fname_ascending_aorta, 0)
    
    if mask_ascending_aorta is None:
        # Missing CXAS masks, skip silently
        return
    
    mask_ascending_aorta_refined = select_max_area_mask(mask_ascending_aorta)

    visualize_mask(args,[mask_ascending_aorta_refined], f'{save_base_dir}/ascending_aorta', dicom)

    # Read viz data from CSV
    try:
        pnt_heart = eval(target_df['pnt_heart'].values[0])
        pnt_trachea = eval(target_df['pnt_trachea'].values[0])
    except (ValueError, KeyError, SyntaxError):
        # Missing required viz fields, skip this sample
        return
    
    try:
        visualize_midline(args, pnt_trachea, pnt_heart, f'{save_base_dir}/borderline', dicom)
    except (ValueError, KeyError, SyntaxError) as e:
        print(f"Warning: Failed to process borderline for {dicom}: {e}")
def segmask_inclusion(args, dicom, target_df, save_base_dir):
    def return_largest_mask(mask):
        labeled, num_components = ndimage.label(mask)
        component_sizes = [np.sum(labeled == i) for i in range(1, num_components + 1)]
        if len(component_sizes) == 0:
            return mask
        largest_component = np.argmax(component_sizes) + 1
        mask = (labeled == largest_component)
        return mask

    cxr_path = get_image_path(args, dicom)
    cxr = safe_imread(cxr_path, cv2.IMREAD_GRAYSCALE)

    fname_rlung = os.path.join(args.cxas_base_dir, dicom, f"right lung.png")
    fname_llung = os.path.join(args.cxas_base_dir, dicom, f"left lung.png")
    fname_rdp = os.path.join(args.cxas_base_dir, dicom, f"right hemidiaphragm.png")
    fname_ldp = os.path.join(args.cxas_base_dir, dicom, f"left hemidiaphragm.png")

    mask_rlung = safe_imread(fname_rlung, cv2.IMREAD_GRAYSCALE)
    mask_llung = safe_imread(fname_llung, cv2.IMREAD_GRAYSCALE)
    mask_rdp = safe_imread(fname_rdp, cv2.IMREAD_GRAYSCALE)
    mask_ldp = safe_imread(fname_ldp, cv2.IMREAD_GRAYSCALE)

    # Check if all masks loaded successfully
    if cxr is None or mask_rlung is None or mask_llung is None or mask_rdp is None or mask_ldp is None:
        # Missing CXAS masks, skip silently
        return

    def return_refined_mask(mask_lung, mask_dp, cxr):
        mask_lung = (mask_lung & ~mask_dp)
        mask_lung = np.logical_and(mask_lung, cxr)
        mask_refined = return_largest_mask(mask_lung)
        return mask_refined

    mask_rlung_refined = return_refined_mask(mask_rlung, mask_rdp, cxr)
    mask_llung_refined = return_refined_mask(mask_llung, mask_ldp, cxr)

    visualize_mask(args,[mask_rlung_refined, mask_llung_refined], f'{save_base_dir}/lung_both', dicom)

def segmask_inspiration(args, dicom, target_df, save_base_dir):
    def select_max_width_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_mask = mask
        max_width = 0
        max_width_pos = (0, 0)
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            x_indices = separate_mask.sum(axis=0).nonzero()[0]
            if len(x_indices) == 0:
                continue
            width = abs(x_indices[0] - x_indices[-1]) + 1
            if width > max_width:
                max_width = width
                max_mask = separate_mask
                max_width_pos = (x_indices[0], x_indices[-1])
        return max_mask, max_width_pos
    
    rib_posterior = [
        "posterior 1st rib",
        "posterior 2nd rib",
        "posterior 3rd rib",
        "posterior 4th rib",
        "posterior 5th rib",
        "posterior 6th rib",
        "posterior 7th rib",
        "posterior 8th rib",
        "posterior 9th rib",
        "posterior 10th rib",
        "posterior 11th rib",
    ]

    # Use rib_position for visualization (not diagnosis label)
    try:
        rib_position = int(target_df['rib_position'].values[0])
    except (KeyError, ValueError, IndexError) as e:
        print(f"Warning: Failed to get rib_position for {dicom}: {e}")
        return
    
    if rib_position < 1 or rib_position > 11:
        print(f"Warning: Invalid rib_position for {dicom}: {rib_position}")
        return
    
    fname_dp = os.path.join(args.cxas_base_dir, dicom, f"right hemidiaphragm.png")
    fname_rib = os.path.join(args.cxas_base_dir, dicom, f'{rib_posterior[(rib_position - 1)]} right.png')

    # Read RL lung mask directly from CXAS instead of chexmask
    fname_lung_rl = os.path.join(args.cxas_base_dir, dicom, "right lung.png")

    mask_dp = safe_imread(fname_dp, 0)
    mask_rib = safe_imread(fname_rib, 0)
    
    if mask_dp is None or mask_rib is None:
        return
    
    mask_rib = (mask_rib / 255).astype(int)
    mask_rib_refined, _ = select_max_width_mask(mask_rib)

    # Use RL lung mask from CXAS to refine diaphragm mask
    if os.path.isfile(fname_lung_rl):
        mask_lung_rl = safe_imread(fname_lung_rl, 0)
        if mask_lung_rl is not None:
            mask_dp = np.logical_and(mask_dp, np.logical_not(mask_lung_rl))
    visualize_mask(args,[mask_dp], f'{save_base_dir}/diaphragm_right', dicom)
    visualize_mask(args,[mask_rib_refined], f'{save_base_dir}/right_posterior_rib', dicom)

    try:
        x_lung_mid_combined = int(target_df['x_lung_mid_combined'].values[0])
        lung_y_lst_x_lung_mid = eval(target_df['lung_y_lst_x_lung_mid'].values[0])
    except (ValueError, KeyError, SyntaxError) as e:
        print(f"Warning: Failed to process inspiration midline for {dicom}: {e}")
        return
    
    if not lung_y_lst_x_lung_mid or len(lung_y_lst_x_lung_mid) == 0:
        print(f"Warning: Empty lung_y_lst_x_lung_mid for {dicom}, skipping inspiration midline")
        return

    visualize_midline(args, 
                      [x_lung_mid_combined, lung_y_lst_x_lung_mid[0]], [x_lung_mid_combined, lung_y_lst_x_lung_mid[-1]],
                      f'{save_base_dir}/midclavicularline', dicom)

def segmask_mediastinal_widening(args, dicom, target_df, save_base_dir):
    fname_medic = os.path.join(args.cxas_base_dir, dicom, f'cardiomediastinum.png')
    mask_medic = safe_imread(fname_medic, 0)
    
    if mask_medic is None:
        # Missing CXAS masks, skip silently
        return
    
    visualize_mask(args,[mask_medic], f'{save_base_dir}/mediastinum', dicom)

    # Read viz data from CSV (if available); otherwise attempt fallback computation
    y_medi = None
    xmin_rlung = None
    xmax_llung = None
    try:
        y_medi = int(target_df['y_medi'].values[0])
        xmin_rlung = int(target_df['xmin_rlung'].values[0])
        xmax_llung = int(target_df['xmax_llung'].values[0])
    except (ValueError, KeyError):
        # attempt to compute missing values from CXAS masks
        try:
            # derive y_medi from cardiomediastinum mask if possible
            rows = np.where(mask_medic > 0)[0]
            if len(rows) > 0:
                y_medi = int((rows[0] + rows[-1]) // 2)

            # try to load lung masks and get x positions at y_medi
            fname_rlung = os.path.join(args.cxas_base_dir, dicom, f"right lung.png")
            fname_llung = os.path.join(args.cxas_base_dir, dicom, f"left lung.png")
            mask_rlung = safe_imread(fname_rlung, 0)
            mask_llung = safe_imread(fname_llung, 0)

            if y_medi is not None:
                if mask_rlung is not None and y_medi < mask_rlung.shape[0]:
                    xs = np.where(mask_rlung[y_medi] > 0)[0]
                    if len(xs) > 0:
                        xmin_rlung = int(xs[0])
                if mask_llung is not None and y_medi < mask_llung.shape[0]:
                    xs = np.where(mask_llung[y_medi] > 0)[0]
                    if len(xs) > 0:
                        xmax_llung = int(xs[-1])

            # fallback: use cardiomediastinum bbox +/- margin if lungs not found
            if (xmin_rlung is None or xmax_llung is None) and mask_medic is not None:
                cols = np.where(mask_medic.sum(axis=0) > 0)[0]
                if len(cols) > 0:
                    left, right = int(cols[0]), int(cols[-1])
                    if xmin_rlung is None:
                        xmin_rlung = max(0, left - 40)
                    if xmax_llung is None:
                        xmax_llung = min(mask_medic.shape[1] - 1, right + 40)
        except Exception:
            # if everything fails, leave as None and skip below
            pass

    # If still missing required fields, skip this sample
    if y_medi is None or xmin_rlung is None or xmax_llung is None:
        return

    save_path = f'{save_base_dir}/thoracic_width_mw'
    os.makedirs(save_path, exist_ok=True)
    
    cxr_path = get_image_path(args, dicom)
    
    # 使用 cv2 繪製，比 matplotlib 快 8-10 倍
    cxr = cv2.imread(cxr_path)
    if cxr is None:
        cxr = np.array(Image.open(cxr_path).convert('RGB'))
        cxr = cv2.cvtColor(cxr, cv2.COLOR_RGB2BGR)
    
    cv2.line(cxr, (xmin_rlung, y_medi - 100), (xmin_rlung, y_medi + 100), (0, 0, 255), 2)
    cv2.line(cxr, (xmax_llung, y_medi - 100), (xmax_llung, y_medi + 100), (0, 0, 255), 2)
    cv2.line(cxr, (xmin_rlung, y_medi), (xmax_llung, y_medi), (0, 0, 255), 2)
    cv2.imwrite(f'{save_path}/{dicom}.png', cxr, [cv2.IMWRITE_PNG_COMPRESSION, 1])

def segmask_projection(args, dicom, target_df, save_base_dir):
    def select_max_area_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask

    fname_r_scapular = os.path.join(args.cxas_base_dir, dicom, f'scapula right.png')
    fname_l_scapular = os.path.join(args.cxas_base_dir, dicom, f'scapula left.png')

    fname_r_lung = os.path.join(args.cxas_base_dir, dicom, f'right lung.png')
    fname_l_lung = os.path.join(args.cxas_base_dir, dicom, f'left lung.png')

    mask_r_scapular = safe_imread(fname_r_scapular, 0) // 255
    mask_l_scapular = safe_imread(fname_l_scapular, 0) // 255

    mask_r_lung = safe_imread(fname_r_lung, 0) // 255
    mask_l_lung = safe_imread(fname_l_lung, 0) // 255

    mask_r_scapular_refined = select_max_area_mask(mask_r_scapular)
    mask_l_scapular_refined = select_max_area_mask(mask_l_scapular)

    visualize_mask(args,[mask_r_lung, mask_l_lung], f'{save_base_dir}/lung_both', dicom)

    visualize_mask(args,[mask_r_scapular_refined, mask_l_scapular_refined], f'{save_base_dir}/scapular_both', dicom)

def segmask_rotation(args, dicom, target_df, save_base_dir):
    def select_max_width_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_mask = mask
        max_width = 0
        max_width_pos = (0, 0)
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            x_indices = separate_mask.sum(axis=0).nonzero()[0]
            if len(x_indices) == 0:
                continue
            width = abs(x_indices[0] - x_indices[-1]) + 1
            if width > max_width:
                max_width = width
                max_mask = separate_mask
                max_width_pos = (x_indices[0], x_indices[-1])
        return max_mask, max_width_pos

    def find_x_vertical(y, m, b):
        x = m * y + b
        return x

    # Read viz data from CSV
    try:
        midline_pnts = eval(target_df['target_coords'].values[0])
        if not midline_pnts or len(midline_pnts) == 0:
            return
        m = target_df['m'].values[0]
        b = target_df['b'].values[0]
    except (ValueError, KeyError, SyntaxError):
        # Missing required midline data
        return
    
    clavicle_masks = dict()
    for side in ['right', 'left']:
        # Read slope and intercepts from viz CSV
        try:
            slope = target_df[f'{side}_slope'].values[0]
            intercept_min = target_df[f'{side}_intercept_min'].values[0]
            intercept_max = target_df[f'{side}_intercept_max'].values[0]
        except KeyError:
            # Missing viz fields for this side, skip
            continue
        
        fname_clavicle = os.path.join(args.cxas_base_dir, dicom, f'clavicle {side}.png')
        mask_clavicle = safe_imread(fname_clavicle, 0)
        
        if mask_clavicle is None:
            # Missing CXAS masks for this side, skip to next
            continue

        longest_mask, _ = select_max_width_mask(mask_clavicle)
        height, width = longest_mask.shape

        # Use pre-computed slope and intercepts from viz CSV
        filled_mask = np.zeros_like(mask_clavicle, dtype=np.uint8)
        for x in range(width):
            y_min = int(slope * x + intercept_min)
            y_max = int(slope * x + intercept_max)

            if 0 <= y_min < height and 0 <= y_max < height:
                filled_mask[min(y_min, y_max):max(y_min, y_max) + 1, x] = 1

        clavicle_mask_refined = np.bitwise_and(mask_clavicle, filled_mask)
        clavicle_masks[side] = clavicle_mask_refined

    # Check if at least one clavicle was processed
    if len(clavicle_masks) == 0:
        return
    
    midline_ymin = midline_pnts[0][-1]
    midline_ymax = midline_pnts[-1][-1]
    x_ymin = find_x_vertical(midline_ymin, m, b)
    x_ymax = find_x_vertical(midline_ymax, m, b)

    # Visualize available clavicle masks
    clavicle_list = []
    if 'right' in clavicle_masks:
        clavicle_list.append(clavicle_masks['right'])
    if 'left' in clavicle_masks:
        clavicle_list.append(clavicle_masks['left'])
    
    if len(clavicle_list) > 0:
        visualize_mask(args, clavicle_list, f'{save_base_dir}/clavicle_both', dicom)
    
    visualize_midline(args, [x_ymin, midline_ymin], [x_ymax, midline_ymax], f'{save_base_dir}/midline', dicom)

def segmask_trachea(args, dicom, target_df, save_base_dir):
    def select_max_area_mask(mask):
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask

    fname_trachea = os.path.join(args.cxas_base_dir, dicom, f"trachea.png")
    fname_carina = os.path.join(args.cxas_base_dir, dicom, f"tracheal bifurcation.png")

    mask_trachea = safe_imread(fname_trachea, 0)
    mask_carina = safe_imread(fname_carina, 0)

    if mask_trachea is None or mask_carina is None:
        return
    
    carina_nonzero = mask_carina.sum(axis=-1).nonzero()[0]
    if len(carina_nonzero) == 0:
        return
    y_pos_carina_start = carina_nonzero[0]
    mask_refined = mask_trachea.copy()
    mask_refined[y_pos_carina_start:] = 0

    mask_refined = select_max_area_mask(mask_refined)

    idx_nonzero = mask_refined.sum(axis=-1).nonzero()[0]
    if len(idx_nonzero) == 0:
        return
    y_min, y_max = idx_nonzero[0], idx_nonzero[-1]
    y_subpart = int((y_max - y_min) * (1 / 4))
    mask_refined[:(y_min + y_subpart)] = 0

    visualize_mask(args,[mask_refined], f'{save_base_dir}/trachea', dicom)

    # Support both viz-prefixed columns (from preprocessor) and legacy names
    def _get_val(df, *keys):
        for k in keys:
            if k in df.columns and not pd.isna(df[k].values[0]):
                try:
                    return int(df[k].values[0])
                except Exception:
                    return None
        return None

    midline_x_min = _get_val(target_df, 'midline_x_min', 'viz_midline_x_min')
    midline_x_max = _get_val(target_df, 'midline_x_max', 'viz_midline_x_max')
    y_min = _get_val(target_df, 'y_min', 'viz_y_min')
    y_max = _get_val(target_df, 'y_max', 'viz_y_max')

    if None in (midline_x_min, midline_x_max, y_min, y_max):
        # Missing viz fields: attempt to parse point_1..point_9 from CSV and derive bbox
        point_cols = [f'point_{i}' for i in range(1, 10)]
        pts = []
        for pc in point_cols:
            if pc in target_df.columns and not pd.isna(target_df[pc].values[0]):
                raw = target_df[pc].values[0]
                try:
                    val = ast.literal_eval(raw) if isinstance(raw, str) else raw
                    # expect [x, y]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        x = int(float(val[0]))
                        y = int(float(val[1]))
                        pts.append((x, y))
                except Exception:
                    continue

        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            midline_x_min = min(xs)
            midline_x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)
            visualize_midline(args, [midline_x_min, y_min], [midline_x_max, y_max], f'{save_base_dir}/midline', dicom)
        else:
            # cannot derive midline, skip only midline output
            return
    else:
        visualize_midline(args, [midline_x_min, y_min], [midline_x_max, y_max], f'{save_base_dir}/midline', dicom)

segmask_fn_per_dx = {
    'inclusion': segmask_inclusion,
    'inspiration': segmask_inspiration,
    'rotation': segmask_rotation,
    'projection': segmask_projection,
    'cardiomegaly': segmask_cardiomegaly,
    'mediastinal_widening': segmask_mediastinal_widening,
    'carina_angle': segmask_carina,
    'trachea_deviation': segmask_trachea,
    'aortic_knob_enlargement': segmask_aortic_knob,
    'ascending_aorta_enlargement': segmask_ascending_aorta,
    'descending_aorta_enlargement': segmask_descending_aorta,
    'descending_aorta_tortuous': segmask_descending_aorta,
}

def process_single_image(args_tuple):
    """處理單個影像的包裝函數，用於多進程處理"""
    dicom, dx, save_dir, viz_df_data, args_dict = args_tuple
    
    # 重建 args 物件和 DataFrame
    from argparse import Namespace
    args = Namespace(**args_dict)
    viz_df = pd.DataFrame(viz_df_data)
    
    # 檢查是否已經處理過
    if check_if_processed(dicom, dx, save_dir):
        return {'dicom': dicom, 'status': 'skipped', 'error': None}
    
    try:
        target_df = viz_df[viz_df['image_file'] == dicom]
        segmask_fn_per_dx[dx](args, dicom, target_df, save_dir)
        return {'dicom': dicom, 'status': 'success', 'error': None}
    except Exception as e:
        return {'dicom': dicom, 'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    # Windows multiprocessing 需要設定啟動方法
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # 已經設定過
    
    args = config()
    
    # Load metadata only for MIMIC dataset
    if args.dataset_name == 'mimic-cxr-jpg':
        args.mimic_meta = pd.read_csv(args.mimic_meta_file)
    else:
        args.mimic_meta = None  # Not needed for NIH dataset

    args.saved_dir_viz = os.path.join(args.saved_base_dir, f"{args.dataset_name}_viz")
    saved_path_viz_list = glob(os.path.join(args.saved_dir_viz, '*.csv'))
    
    print(f"\n找到 {len(saved_path_viz_list)} 個診斷任務的CSV檔案")
    
    # 顯示處理設定
    print(f"\n{'='*60}")
    print(f"處理設定:")
    print(f"  資料集: {args.dataset_name}")
    print(f"  併行 worker 數: {args.num_workers}")
    print(f"  CUDA 加速: {'✓ 可用 (GPU)' if CUDA_AVAILABLE else '✗ 不可用 (CPU)'}")
    if CUDA_AVAILABLE:
        print(f"  GPU 設備: {torch.cuda.get_device_name(0)}")
        print(f"  CuPy 加速: {'✓ 已啟用' if USE_CUPY else '✗ 未安裝 (請執行: pip install cupy-cuda12x)'}")
    if args.num_workers > 1:
        print(f"  模式: 多進程並行處理 (速度提升 {args.num_workers}x)")
    else:
        print(f"  模式: 單進程順序處理")
    print(f"{'='*60}\n")
    
    # Use nih_cxr14 CSV (has rib_position column) instead of nih-cxr14_viz
    for dx in segmask_fn_per_dx.keys():
        print(f"\n>>> 檢查診斷任務: {dx}")
        save_dir = os.path.join(args.save_base_dir, 'segmask_bodypart', dx)
        
        # 讀取 nih_cxr14 CSV (包含 rib_position 等完整資料)
        original_csv = os.path.join(args.saved_base_dir, 'nih_cxr14', f'{dx}.csv')
        if not os.path.exists(original_csv):
            print(f">>> 錯誤：找不到 CSV 檔案: {original_csv}，跳過此任務")
            continue
        
        print(f">>> 讀取 CSV: {original_csv}")
        df = pd.read_csv(original_csv)
        print(f">>> CSV 讀取完成，共 {len(df)} 行資料")
        total_tasks = len(df)
        
        # 同時讀取 viz CSV 合併可視化資料 (如果存在)
        viz_csv = os.path.join(args.saved_base_dir, f'nih-cxr14_viz/{dx}.csv')
        viz_df = None
        if os.path.exists(viz_csv):
            viz_df = pd.read_csv(viz_csv)

        # 合併 viz 資料（動態檢查需要哪些欄位）
        viz_columns = ['image_file']

        # 定義每個任務需要的viz欄位
        viz_columns_map = {
            'inspiration': ['x_lung_mid_combined', 'lung_y_lst_x_lung_mid'],
            'rotation': ['target_coords', 'm', 'b', 'right_slope', 'right_intercept_min', 'right_intercept_max', 'left_slope', 'left_intercept_min', 'left_intercept_max'],
            'cardiomegaly': ['coord_mask', 'xmin_lung', 'xmax_lung'],
            'descending_aorta_enlargement': ['ymin_start', 'ymin_end'],
            'descending_aorta_tortuous': ['ymin_start', 'ymin_end'],
            'aortic_knob_enlargement': ['y_max', 'ymin_desc', 'ysub_desc', 'ymax_desc_sub', 'xmax_trachea_mean'],
            'ascending_aorta_enlargement': ['pnt_heart', 'pnt_trachea'],
            'mediastinal_widening': ['y_medi', 'xmin_rlung', 'xmax_llung'],
            'trachea_deviation': ['viz_midline_x_min', 'viz_midline_x_max', 'viz_y_min', 'viz_y_max'],
        }

        # 根據任務類型添加需要的viz欄位並合併到主表 df（保留所有原始列）
        if dx in viz_columns_map and viz_df is not None:
            for col in viz_columns_map[dx]:
                if col in viz_df.columns:
                    viz_columns.append(col)
            if len(viz_columns) > 1:
                print(f">>> 合併 viz 資料成功 (列: {', '.join(viz_columns[1:])})")
                viz_selected = viz_df[viz_columns].copy()
                df = df.merge(viz_selected, on='image_file', how='left')
            else:
                print(f">>> 欄位不足，無法合併 viz 資料 (期待: {', '.join(viz_columns_map[dx])})")
        else:
            if dx in viz_columns_map and viz_df is None:
                print(f">>> 未找到 viz CSV，跳過合併: {viz_csv}")
            else:
                print(f">>> 此任務不需要 viz 資料")
        
        dicom_list = df['image_file'].tolist()
        
        # 檢查已完成的數量，判斷是否需要處理
        output_checks = {
                'cardiomegaly': ['heart', 'thoracic_width_heart'],
            'carina_angle': ['carina'],
            'descending_aorta_enlargement': ['descending_aorta', 'trachea'],
            'descending_aorta_tortuous': ['descending_aorta'],
            'aortic_knob_enlargement': ['aortic_knob', 'trachea'],
            'ascending_aorta_enlargement': ['ascending_aorta', 'borderline'],
            'inclusion': ['lung_both'],
            'inspiration': ['diaphragm_right', 'right_posterior_rib', 'midclavicularline'],
            'mediastinal_widening': ['mediastinum', 'thoracic_width_mw'],
            'projection': ['lung_both', 'scapular_both'],
            'rotation': ['clavicle_both', 'midline'],
            'trachea_deviation': ['trachea', 'midline'],
        }
        
        # 統計已完成的輸出檔案數量（效能優化）
        # 為避免對每個 dicom 重複呼叫 os.path.exists，先一次性掃描各子資料夾，建立檔名集合
        completed_count = 0
        if dx in output_checks:
            required_subfolders = output_checks[dx]
            exts = ('.png', '.jpg', '.jpeg')

            # 建立每個子資料夾中可用檔案的 basename（不含副檔名）集合
            subfolder_name_sets = {}
            for subfolder in required_subfolders:
                subfolder_path = os.path.join(save_dir, subfolder)
                names = set()
                if os.path.exists(subfolder_path):
                    try:
                        for fname in os.listdir(subfolder_path):
                            if not fname:
                                continue
                            base, ext = os.path.splitext(fname)
                            if ext.lower() in exts:
                                names.add(base)
                    except Exception:
                        # 若無法列出目錄，視為空集合
                        names = set()
                subfolder_name_sets[subfolder] = names

            # 正規化 dicom 名稱（去除路徑與副檔名）
            normalized_dicoms = [os.path.splitext(os.path.basename(str(d)))[0] for d in dicom_list]

            # 計算在所有 required_subfolders 都存在的 dicom
            for d in normalized_dicoms:
                ok = True
                for subfolder in required_subfolders:
                    if d not in subfolder_name_sets.get(subfolder, set()):
                        ok = False
                        break
                if ok:
                    completed_count += 1

        completion_rate = (completed_count / total_tasks * 100) if total_tasks > 0 else 0

        print(f"\n{'='*60}")
        print(f"診斷任務: {dx}")

        # # 暫時跳過 rotation 任務
        # if dx == 'rotation':
        #     print(f"  ⚠️  暫時跳過 rotation（等待viz CSV更新）")
        #     print(f"  ✓ 跳過 {dx}")
        #     print(f"{'='*60}")
        #     continue

        print(f"  總任務數: {total_tasks}")
        print(f"  已完成: {completed_count} ({completion_rate:.1f}%)")
        print(f"  待處理: {len(dicom_list)}")

        # 如果完成率超過 80%，自動跳過
        if completion_rate > 80:
            print(f"  ⚠️  此任務已接近完成（{completion_rate:.1f}%），自動跳過")
            print(f"  ✓ 跳過 {dx}")
            continue
        print(f"{'='*60}")
        
        if args.num_workers > 1:
            # 多進程處理
            print(f">>> 使用 {args.num_workers} 個並行worker加速處理")
            
            # 準備參數：傳遞 DataFrame 字典格式（更高效）
            args_dict = vars(args)
            df_data = df.to_dict('list')  # 轉為字典格式傳遞
            task_args = [(dicom, dx, save_dir, df_data, args_dict) for dicom in dicom_list]
            
            print(f">>> 啟動 Pool，準備處理 {len(task_args)} 個影像...")
            try:
                # 使用 Pool 處理
                with Pool(processes=args.num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_single_image, task_args),
                        total=len(task_args),
                        desc=f'{dx}'
                    ))
                
                # 統計結果
                processed_count = sum(1 for r in results if r['status'] == 'success')
                skipped_count = sum(1 for r in results if r['status'] == 'skipped')
                error_count = sum(1 for r in results if r['status'] == 'error')
                
                # 顯示錯誤
                if error_count > 0:
                    print(f"\n⚠️  有 {error_count} 個影像處理失敗")
                    for r in results:
                        if r['status'] == 'error':
                            print(f"  - {r['dicom']}: {r['error']}")
            except Exception as e:
                print(f"\n❌ multiprocessing 失敗: {e}")
                print(f">>> 切換到單進程模式處理...")
                # 降級到單進程處理
                args.num_workers = 1
        else:
            # 單進程處理（原始方法）
            processed_count = 0
            skipped_count = 0
            error_count = 0
            
            for dicom in tqdm(dicom_list, total=len(dicom_list), desc=f'{dx}'):
                # 檢查是否已經處理過
                if check_if_processed(dicom, dx, save_dir):
                    skipped_count += 1
                    continue
                
                try:
                    target_df = df[df['image_file'] == dicom]
                    segmask_fn_per_dx[dx](args, dicom, target_df, save_dir)
                    processed_count += 1
                except Exception as e:
                    print(f"\nError processing {dicom}: {e}")
                    error_count += 1
                    continue
        
        print(f"✅ {dx} 完成 - 新處理: {processed_count}, 跳過: {skipped_count}, 錯誤: {error_count}")

