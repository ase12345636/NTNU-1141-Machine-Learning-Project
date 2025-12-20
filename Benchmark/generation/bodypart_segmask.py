import os
import cv2
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端，加速繪圖
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode, Visualizer
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
from detectron2.config import get_cfg

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
    print("Warning: CuPy not installed. Install with: pip install cupy-cuda12x")

# 配置 detectron2 使用 GPU
def setup_detectron2_device():
    """配置 detectron2 使用 GPU 或 CPU"""
    cfg = get_cfg()
    cfg.MODEL.DEVICE = DEVICE
    return cfg

# 全域配置
DETECTRON2_CFG = setup_detectron2_device()

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
    """Safely read image file, return None if file doesn't exist"""
    if not os.path.exists(file_path):
        return None
    return cv2.imread(file_path, flags)

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
    cxr = Image.open(cxr_path).convert('RGB')

    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    plt.imshow(vs.overlay_instances(masks=mask_lst, assigned_colors=mask_colors * len(mask_lst)).get_image())
    plt.axis('off')
    plt.savefig(f'{save_path}/{dicom}.png', bbox_inches='tight')
    plt.close()

def visualize_midline(args, pnt_start, pnt_end, save_path, dicom):
    os.makedirs(save_path, exist_ok=True)
    cxr_path = get_image_path(args, dicom)
    cxr = Image.open(cxr_path).convert('RGB')

    plt.imshow(cxr)
    plt.plot([pnt_start[0], pnt_end[0]], [pnt_start[-1], pnt_end[-1]], color="red", linewidth=1.5)
    plt.axis('off')
    plt.savefig(f'{save_path}/{dicom}.png', bbox_inches='tight')
    plt.close()

def check_if_processed(dicom, dx, save_base_dir):
    """檢查影像是否已經處理過，根據不同診斷任務檢查對應的輸出檔案"""
    # 定義每個診斷任務需要檢查的關鍵輸出檔案
    output_checks = {
        'cardiomegaly': ['heart', 'lung_both', 'thoracic_width'],
        'carina_angle': ['carina'],
        'descending_aorta_enlargement': ['descending_aorta', 'trachea'],
        'descending_aorta_tortuous': ['descending_aorta', 'trachea'],
        'aortic_knob_enlargement': ['aortic_knob', 'trachea'],
        'ascending_aorta_enlargement': ['ascending_aorta', 'lung_both', 'diaphragm'],
        'inclusion': ['lung_both'],
        'inspiration': ['diaphragm_right', 'right_posterior_rib', 'midclavicularline'],
        'mediastinal_widening': ['mediastinum', 'thoracic_width_mw'],
        'projection': ['lung_both', 'scapular_both'],
        'rotation': ['clavicle_both', 'midline'],
        'trachea_deviation': ['trachea', 'midline'],
    }
    
    if dx not in output_checks:
        return False
    
    # 檢查至少一個關鍵輸出檔案是否存在
    for subfolder in output_checks[dx]:
        output_path = os.path.join(save_base_dir, subfolder, f'{dicom}.png')
        if os.path.exists(output_path):
            return True
    
    return False

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

    try:
        coord_mask = eval(target_df['coord_mask'].values[0])
        xmin_heart, ymin_heart, xmax_heart, ymax_heart = coord_mask

        xmin_lung = int(target_df['xmin_lung'].values[0])
        xmax_lung = int(target_df['xmax_lung'].values[0])
    except (ValueError, KeyError, SyntaxError) as e:
        print(f"Warning: Failed to process cardiomegaly for {dicom}: {e}")
        return
    lungs_target_part = (mask_rlung_refined[ymin_heart:] | mask_llung_refined[ymin_heart:])
    y_xmin_indices = lungs_target_part[:, xmin_lung].nonzero()[0].tolist()
    y_xmax_indices = lungs_target_part[:, xmax_lung].nonzero()[0].tolist()

    y_indices = set(y_xmax_indices + y_xmin_indices)
    ymin_lung = min(y_indices) + ymin_heart
    ymax_lung = max(y_indices) + ymin_heart
    ymean_lung = (ymin_lung + ymax_lung) // 2

    # thoracic width - 在心臟中心高度測量胸腔寬度
    ymean_heart = (ymin_heart + ymax_heart) // 2
    
    save_path = f'{save_base_dir}/thoracic_width_heart'
    os.makedirs(save_path, exist_ok=True)
    cxr_path = get_image_path(args, dicom)
    cxr = Image.open(cxr_path).convert('RGB')

    plt.imshow(cxr)
    # 在心臟中心高度畫垂直線標記胸腔寬度
    plt.plot([xmin_lung, xmin_lung], [ymean_heart - 100, ymean_heart + 100], color="red", linewidth=1.5)
    plt.plot([xmax_lung, xmax_lung], [ymean_heart - 100, ymean_heart + 100], color="red", linewidth=1.5)
    plt.plot([xmin_lung, xmax_lung], [ymean_heart, ymean_heart], color="red", linewidth=1.5)
    plt.axis('off')
    plt.savefig(f'{save_path}/{dicom}.png', bbox_inches='tight')
    plt.close()

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
            ymax = overlap.sum(axis=-1).nonzero()[0][-1]

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

    try:
        ymin_start = int(target_df['ymin_start'].values[0])
        ymin_end = int(target_df['ymin_end'].values[0])
    except (ValueError, KeyError) as e:
        print(f"Warning: Failed to process descending_aorta for {dicom}: {e}")
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

        y_pos_carina_start = mask_carina.sum(axis=-1).nonzero()[0][0]
        mask_refined = mask_trachea.copy()
        mask_refined[y_pos_carina_start:] = 0

        mask_refined = select_max_area_mask(mask_refined)

        idx_nonzero = mask_refined.sum(axis=-1).nonzero()[0]
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
    
    try:
        y_max = int(target_df['y_max'].values[0])
        ymin_desc = int(target_df['ymin_desc'].values[0])
        ysub_desc = int(target_df['ysub_desc'].values[0])

        mask_descending_aorta_refined = mask_descending_aorta.copy()
        mask_descending_aorta_refined[:y_max] = 0
        mask_descending_aorta_refined[(ymin_desc + ysub_desc):] = 0
        mask_descending_aorta_refined = select_max_width_mask(mask_descending_aorta_refined)

        ymax_desc_sub = int(target_df['ymax_desc_sub'].values[0])
        mask_aortic_arch[ymax_desc_sub:] = 0  # y axis - remove unexpected region blobs
        mask_aortic_arch_refined = select_target_axis_mask(mask_aortic_arch, mask_descending_aorta_refined)

        # Check if mask is empty after refinement
        if mask_aortic_arch_refined.sum() == 0:
            print(f"Warning: Empty aortic arch mask after refinement for {dicom}, skipping visualization")
            return

        xmax_trachea_mean = int(target_df['xmax_trachea_mean'].values[0])
        mask_aortic_arch_refined[:, :xmax_trachea_mean] = 0

        visualize_mask(args,[mask_aortic_arch_refined], f'{save_base_dir}/aortic_knob', dicom)
    except (ValueError, KeyError) as e:
        print(f"Warning: Failed to process aortic_knob for {dicom}: {e}")
        return

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

    y_pos_carina_start = mask_carina.sum(axis=-1).nonzero()[0][0]
    mask_refined = mask_trachea.copy()
    mask_refined[y_pos_carina_start:] = 0

    mask_refined = select_max_area_mask(mask_refined)

    idx_nonzero = mask_refined.sum(axis=-1).nonzero()[0]
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

    try:
        pnt_heart = eval(target_df['pnt_heart'].values[0])
        pnt_trachea = eval(target_df['pnt_trachea'].values[0])
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

    label = target_df['label'].values[0]
    fname_dp = os.path.join(args.cxas_base_dir, dicom, f"right hemidiaphragm.png")
    fname_rib = os.path.join(args.cxas_base_dir, dicom, f'{rib_posterior[(label - 1)]} right.png')

    fname_lung_chexmask = os.path.join(args.chexmask_base_dir, f'RL', f'{dicom}.png')

    mask_dp = safe_imread(fname_dp, 0)
    mask_rib = safe_imread(fname_rib, 0)
    
    if mask_dp is None or mask_rib is None:
        # Missing CXAS masks, skip silently
        return
    
    mask_rib = (mask_rib / 255).astype(int)
    mask_rib_refined, _ = select_max_width_mask(mask_rib)

    if os.path.isfile(fname_lung_chexmask):
        mask_lung_chexmask = safe_imread(fname_lung_chexmask, 0)
        mask_dp = np.logical_and(mask_dp, np.logical_not(mask_lung_chexmask))
    visualize_mask(args,[mask_dp], f'{save_base_dir}/diaphragm_right', dicom)
    visualize_mask(args,[mask_rib_refined], f'{save_base_dir}/right_posterior_rib', dicom)

    try:
        x_lung_mid_combined = int(target_df['x_lung_mid_combined'].values[0])
        lung_y_lst_x_lung_mid = eval(target_df['lung_y_lst_x_lung_mid'].values[0])
    except (ValueError, KeyError, SyntaxError) as e:
        print(f"Warning: Failed to process inspiration midline for {dicom}: {e}")
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

    try:
        y_medi = int(target_df['y_medi'].values[0])
        xmin_rlung = int(target_df['xmin_rlung'].values[0])
        xmax_llung = int(target_df['xmax_llung'].values[0])
    except (ValueError, KeyError) as e:
        print(f"Warning: Failed to process mediastinal_widening thoracic width for {dicom}: {e}")
        # Still visualize the mediastinum mask even if thoracic width fails
        return

    save_path = f'{save_base_dir}/thoracic_width_mw'
    os.makedirs(save_path, exist_ok=True)
    
    cxr_path = get_image_path(args, dicom)
    cxr = Image.open(cxr_path).convert('RGB')

    plt.imshow(cxr)
    plt.plot([xmin_rlung, xmin_rlung], [y_medi - 100, y_medi + 100], color="red", linewidth=1.5)
    plt.plot([xmax_llung, xmax_llung], [y_medi - 100, y_medi + 100], color="red", linewidth=1.5)
    plt.plot([xmin_rlung, xmax_llung], [y_medi, y_medi], color="red", linewidth=1.5)
    plt.axis('off')
    plt.savefig(f'{save_path}/{dicom}.png', bbox_inches='tight')
    plt.close()

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
            width = abs(x_indices[0] - x_indices[-1]) + 1
            if width > max_width:
                max_width = width
                max_mask = separate_mask
                max_width_pos = (x_indices[0], x_indices[-1])
        return max_mask, max_width_pos

    def find_x_vertical(y, m, b):
        x = m * y + b
        return x

    clavicle_masks = dict()
    for side in ['right', 'left']:
        fname_clavicle = os.path.join(args.cxas_base_dir, dicom, f'clavicle {side}.png')
        mask_clavicle = safe_imread(fname_clavicle, 0)
        
        if mask_clavicle is None:
            # Missing CXAS masks, skip silently
            return

        longest_mask, _ = select_max_width_mask(mask_clavicle)
        height, width = longest_mask.shape

        slope = target_df[f'{side}_slope']
        intercept_min = target_df[f'{side}_intercept_min']
        intercept_max = target_df[f'{side}_intercept_max']

        filled_mask = np.zeros_like(mask_clavicle, dtype=np.uint8)
        for x in range(width):
            y_min = int(slope * x + intercept_min)
            y_max = int(slope * x + intercept_max)

            if 0 <= y_min < height and 0 <= y_max < height:
                filled_mask[min(y_min, y_max):max(y_min, y_max) + 1, x] = 1

        clavicle_mask_refined = np.bitwise_and(mask_clavicle, filled_mask)
        clavicle_masks[side] = clavicle_mask_refined

    midline_pnts = eval(target_df['target_coords'].values[0])
    m = target_df['m'].values[0]
    b = target_df['b'].values[0]
    midline_ymin = midline_pnts[0][-1]
    midline_ymax = midline_pnts[-1][-1]
    x_ymin = find_x_vertical(midline_ymin, m, b)
    x_ymax = find_x_vertical(midline_ymax, m, b)

    visualize_mask(args,[clavicle_masks['right'], clavicle_masks['left']], f'{save_base_dir}/clavicle_both', dicom)
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

    y_pos_carina_start = mask_carina.sum(axis=-1).nonzero()[0][0]
    mask_refined = mask_trachea.copy()
    mask_refined[y_pos_carina_start:] = 0

    mask_refined = select_max_area_mask(mask_refined)

    idx_nonzero = mask_refined.sum(axis=-1).nonzero()[0]
    y_min, y_max = idx_nonzero[0], idx_nonzero[-1]
    y_subpart = int((y_max - y_min) * (1 / 4))
    mask_refined[:(y_min + y_subpart)] = 0

    visualize_mask(args,[mask_refined], f'{save_base_dir}/trachea', dicom)

    midline_x_min = int(target_df['midline_x_min'].values[0])
    midline_x_max = int(target_df['midline_x_max'].values[0])
    y_min = int(target_df['y_min'].values[0])
    y_max = int(target_df['y_max'].values[0])

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
    dicom, dx, save_dir, viz_df, args = args_tuple
    
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
    args = config()
    
    # Load metadata only for MIMIC dataset
    if args.dataset_name == 'mimic-cxr-jpg':
        args.mimic_meta = pd.read_csv(args.mimic_meta_file)
    else:
        args.mimic_meta = None  # Not needed for NIH dataset

    args.saved_dir_viz = os.path.join(args.saved_base_dir, f"{args.dataset_name}_viz")
    saved_path_viz_list = glob(os.path.join(args.saved_dir_viz, '*.csv'))
    
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
    
    for saved_path_viz in saved_path_viz_list:
        dx = Path(saved_path_viz).stem
        if dx in segmask_fn_per_dx:
            save_dir = os.path.join(args.save_base_dir, 'segmask_bodypart', dx)

            viz_df = pd.read_csv(saved_path_viz)
            dicom_list = viz_df['image_file'].tolist()
            
            # 檢查已完成的數量，判斷是否需要處理
            output_checks = {
                'cardiomegaly': ['heart', 'lung_both', 'thoracic_width_heart'],
                'carina_angle': ['carina'],
                'descending_aorta_enlargement': ['descending_aorta', 'trachea'],
                'descending_aorta_tortuous': ['descending_aorta', 'trachea'],
                'aortic_knob_enlargement': ['aortic_knob', 'trachea'],
                'ascending_aorta_enlargement': ['ascending_aorta', 'lung_both', 'diaphragm'],
                'inclusion': ['lung_both'],
                'inspiration': ['diaphragm_right', 'right_posterior_rib', 'midclavicularline'],
                'mediastinal_widening': ['mediastinum', 'thoracic_width_mw'],
                'projection': ['lung_both', 'scapular_both'],
                'rotation': ['clavicle_both', 'midline'],
                'trachea_deviation': ['trachea', 'midline'],
            }
            
            # 統計已完成的輸出檔案數量
            completed_count = 0
            if dx in output_checks:
                for subfolder in output_checks[dx]:
                    subfolder_path = os.path.join(save_dir, subfolder)
                    if os.path.exists(subfolder_path):
                        completed_count = max(completed_count, len([f for f in os.listdir(subfolder_path) if f.endswith('.png')]))
            
            completion_rate = (completed_count / len(dicom_list) * 100) if len(dicom_list) > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"診斷任務: {dx}")
            print(f"  總影像數: {len(dicom_list)}")
            print(f"  已完成: {completed_count} ({completion_rate:.1f}%)")
            
            # 如果完成率超過 80%，詢問是否跳過
            if completion_rate > 80:
                print(f"  ⚠️  此任務已接近完成，是否跳過？")
                user_input = input(f"  跳過 {dx}? (y/n，直接Enter視為n): ").strip().lower()
                if user_input == 'y':
                    print(f"  ✓ 跳過 {dx}")
                    continue
            print(f"{'='*60}")
            
            if args.num_workers > 1:
                # 多進程處理
                # 準備參數元組
                task_args = [(dicom, dx, save_dir, viz_df, args) for dicom in dicom_list]
                
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
                        target_df = viz_df[viz_df['image_file'] == dicom]
                        segmask_fn_per_dx[dx](args, dicom, target_df, save_dir)
                        processed_count += 1
                    except Exception as e:
                        print(f"\nError processing {dicom}: {e}")
                        error_count += 1
                        continue
            
            print(f"✅ {dx} 完成 - 新處理: {processed_count}, 跳過: {skipped_count}, 錯誤: {error_count}")

