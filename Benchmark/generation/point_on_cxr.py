import os
try:
    import cv2
except Exception:
    cv2 = None
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from adjustText import adjust_text
from detectron2.utils.visualizer import ColorMode, Visualizer
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
import ast

# CUDA support
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_base_dir', default='path/to/save/output', type=str)

    parser.add_argument('--saved_base_dir', default='path/to/saved/output', type=str)
    parser.add_argument('--dataset_name', default='mimic-cxr-jpg', choices=['mimic-cxr-jpg', 'nih-cxr14'])

    parser.add_argument('--mimic_cxr_base_dir', default="<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files", type=str)
    parser.add_argument('--mimic_meta_file', default='<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv', type=str)

    # NIH dataset arguments
    parser.add_argument('--nih_image_base_dir', type=str, default='/mnt/d/CXReasonBench/dataset', help='Path to NIH images folders (images_001, images_002, ...)')

    parser.add_argument('--cxas_base_dir', type=str, default='path/to/cxas_segmentation_folders')
    
    # 多進程參數
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers (default: 4, set to 1 to disable)')
    parser.add_argument('--test_one_per_dx', action='store_true', help='If set, process only one image per diagnosis for quick validation')
    parser.add_argument('--ignore_has_valid_viz', action='store_true', help='If set, ignore the has_valid_viz column and process all rows in the viz CSV')

    args = parser.parse_args()
    return args


mask_color = [np.array([0.5, 0., 0.], dtype=np.float32)]


def imread_gray(path):
    """Read image as grayscale. Uses cv2 if available, otherwise falls back to PIL."""
    if cv2 is not None:
        try:
            return cv2.imread(path, 0)
        except Exception:
            return None
    else:
        try:
            img = Image.open(path).convert('L')
            return np.array(img)
        except Exception:
            return None

def safe_eval(val):
    """Safely parse Python literals from CSV fields. Returns None on failure/NaN/empty.

    Accepts already-parsed lists/tuples/dicts/numbers and returns them unchanged.
    """
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    if val is None:
        return None
    # If already a Python object, return as-is
    if isinstance(val, (list, tuple, dict, int, float)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s == '':
            return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return None
    # fallback
    return val

def safe_int(x):
    """Convert x to an int safely. Return None if conversion not possible."""
    try:
        if x is None:
            return None
        if isinstance(x, (list, tuple, dict)):
            return None
        # handle numpy types and strings
        if isinstance(x, str):
            s = x.strip()
            if s == '':
                return None
        fx = float(x)
        if np.isnan(fx):
            return None
        return int(round(fx))
    except Exception:
        return None

def safe_point(p):
    """Normalize a parsed point-like object to (x, y) ints. Return None on failure."""
    if not p:
        return None
    # if nested list/tuple (like [x,y] or (x,y))
    try:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x = safe_int(p[0])
            y = safe_int(p[-1])
            if x is None or y is None:
                return None
            return (x, y)
        # if string, try parsing
        parsed = safe_eval(p) if isinstance(p, str) else p
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            x = safe_int(parsed[0]); y = safe_int(parsed[-1])
            if x is None or y is None:
                return None
            return (x, y)
    except Exception:
        return None
    return None

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

def return_cxr(args, dicom):
    """Load and return chest X-ray image"""
    cxr_path = get_image_path(args, dicom)
    cxr = Image.open(cxr_path).convert('RGB')
    return cxr

def pnt_cardiomegaly(args, dicom, target_df, save_dir):
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

    coord_mask = safe_eval(target_df.get('coord_mask', pd.Series([None])).values[0])
    if not coord_mask or len(coord_mask) != 4:
        print(f"Warning: Invalid coord_mask for {dicom}, skipping cardiomegaly")
        return
    xmin_heart, ymin_heart, xmax_heart, ymax_heart = coord_mask
    # ensure integer coordinates
    xmin_heart = safe_int(xmin_heart)
    ymin_heart = safe_int(ymin_heart)
    xmax_heart = safe_int(xmax_heart)
    ymax_heart = safe_int(ymax_heart)
    if None in (xmin_heart, ymin_heart, xmax_heart, ymax_heart):
        print(f"Warning: coord_mask contains non-integer values for {dicom}, skipping cardiomegaly")
        return
    ymean_heart = (ymin_heart + ymax_heart) // 2

    xmin_lung = safe_int(target_df.get('xmin_lung', pd.Series([None])).values[0])
    xmax_lung = safe_int(target_df.get('xmax_lung', pd.Series([None])).values[0])
    if xmin_lung is None or xmax_lung is None:
        print(f"Warning: Missing lung x-bounds for {dicom}, skipping cardiomegaly")
        return

    fname_rlung = os.path.join(args.cxas_base_dir, dicom, f"right lung.png")
    fname_llung = os.path.join(args.cxas_base_dir, dicom, f"left lung.png")

    mask_rlung = cv2.imread(fname_rlung, 0)
    mask_llung = cv2.imread(fname_llung, 0)
    
    if mask_rlung is None or mask_llung is None:
        print(f"Warning: Failed to read lung masks for {dicom}, skipping cardiomegaly")
        return

    mask_rlung_refined = select_max_area_mask(mask_rlung)[ymin_heart:]
    mask_llung_refined = select_max_area_mask(mask_llung)[ymin_heart:]
    lungs = (mask_rlung_refined | mask_llung_refined)

    # validate column indices against lungs width
    try:
        cols = lungs.shape[1]
    except Exception:
        print(f"Warning: Unexpected lung mask shape for {dicom}, skipping cardiomegaly")
        return
    if not (0 <= xmin_lung < cols) or not (0 <= xmax_lung < cols):
        print(f"Warning: Lung x-bounds out of range for {dicom} (cols={cols}, xmin={xmin_lung}, xmax={xmax_lung}), skipping")
        return

    y_xmin_indices = lungs[:, xmin_lung].nonzero()[0].tolist()
    y_xmax_indices = lungs[:, xmax_lung].nonzero()[0].tolist()

    y_indices = set(y_xmax_indices + y_xmin_indices)
    if not y_indices:
        return  # No valid lung pixels found
    ymin_lung = min(y_indices) + ymin_heart
    ymax_lung = max(y_indices) + ymin_heart
    ymean_lung = (ymin_lung + ymax_lung) // 2


    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize=10
    fontsize=25
    plt.imshow(cxr)

    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")

    texts = []
    plt.plot([xmin_lung, xmin_lung], [ymin_lung, ymax_lung], color="blue", linewidth=markersize)
    texts.append(plt.text(xmin_lung, ymax_lung + 40, f'{xmin_lung}',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='center'))

    plt.plot([xmax_lung, xmax_lung], [ymin_lung, ymax_lung], color="blue", linewidth=markersize)
    texts.append(plt.text(xmax_lung, ymax_lung + 40, f'{xmax_lung}',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='center'))

    plt.plot([xmin_lung, xmax_lung], [ymean_lung, ymean_lung], color="blue", linewidth=markersize)
    texts.append(plt.text(xmin_lung + 10, ymean_lung, 'Thoracic Width',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    # =============# =============# =============# =============# =============# =============# =============
    plt.plot([xmin_heart, xmin_heart], [ymin_heart, ymax_heart], color="red", linewidth=markersize)
    texts.append(plt.text(xmin_heart + 10, ymin_heart - 70, f'{xmin_heart}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    plt.plot([xmax_heart, xmax_heart], [ymin_heart, ymax_heart], color="red", linewidth=markersize)
    texts.append(plt.text(xmax_heart + 50, ymin_heart - 70, f'{xmax_heart}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='center'))

    plt.plot([xmin_heart, xmax_heart], [ymean_heart, ymean_heart], color="red", linewidth=markersize)
    texts.append(plt.text((xmin_heart + xmax_heart) // 2, ymean_heart, f'Heart Width',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='bottom', horizontalalignment='left'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_carina(args, dicom, target_df, save_dir):
    # refined_tgpnt may be missing; try to construct from point_1/2/3 if available
    refined_tgpnt = safe_eval(target_df.get('refined_tgpnt', pd.Series([None])).values[0])
    if not refined_tgpnt or len(refined_tgpnt) != 3:
        # attempt to read point_1/2/3 fields
        if all(c in target_df.columns for c in ('point_1', 'point_2', 'point_3')):
            p1 = safe_eval(target_df.get('point_1', pd.Series([None])).values[0])
            p2 = safe_eval(target_df.get('point_2', pd.Series([None])).values[0])
            p3 = safe_eval(target_df.get('point_3', pd.Series([None])).values[0])
            if p1 and p2 and p3:
                refined_tgpnt = [p1, p2, p3]
    if not refined_tgpnt or len(refined_tgpnt) != 3:
        print(f"Warning: Invalid refined_tgpnt for {dicom}, skipping carina")
        return
    # normalize points to integer (x,y)
    raw_pnt_rl, raw_pnt_c, raw_pnt_ll = refined_tgpnt
    pnt_rl = safe_point(raw_pnt_rl)
    pnt_c = safe_point(raw_pnt_c)
    pnt_ll = safe_point(raw_pnt_ll)
    if not pnt_rl or not pnt_c or not pnt_ll:
        print(f"Warning: Invalid carina points for {dicom}, skipping carina")
        return

    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 10
    fontsize = 25
    plt.imshow(cxr)

    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")

    texts = []
    plt.plot([pnt_rl[0]], [pnt_rl[1]], color='red', marker='o', markersize=markersize)
    texts.append(plt.text(pnt_rl[0] - 20, pnt_rl[1] + 15, f'A: {pnt_rl}',
                          fontsize=fontsize, color='red', bbox=bbox_props,
                          verticalalignment='top', horizontalalignment='right'))

    plt.plot([pnt_c[0]], [pnt_c[1]], color='green', marker='o', markersize=markersize)
    texts.append(plt.text(pnt_c[0], pnt_c[1] - 25, f'B: {pnt_c}',
                          fontsize=fontsize, color='green', bbox=bbox_props,
                          verticalalignment='bottom', horizontalalignment='center'))

    plt.plot([pnt_ll[0]], [pnt_ll[1]], color='blue', marker='o', markersize=markersize)
    texts.append(plt.text(pnt_ll[0] + 20, pnt_ll[1] + 15, f'C: {pnt_ll}',
                          fontsize=fontsize, color='blue', bbox=bbox_props,
                          verticalalignment='top', horizontalalignment='left'))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_desc_aorta_enlarged(args, dicom, target_df, save_dir):
    y_width = target_df['trachea_y_width'].values[0]
    xmin_width = target_df['trachea_xmin_width'].values[0]
    xmax_width = target_df['trachea_xmax_width'].values[0]

    pnts_desc_aorta = []
    pnt_width_lst = []
    for column in target_df.columns:
        if 'pnt_' in column:
            pnts_desc_aorta.append(column)
    for i in range(len(pnts_desc_aorta) // 2):
        pnt_r = safe_eval(target_df.get(f'pnt_r_{i + 1}', pd.Series([None])).values[0])
        pnt_l = safe_eval(target_df.get(f'pnt_{i + 1}', pd.Series([None])).values[0])
        
        if not pnt_r or not pnt_l:
            print(f"Warning: Empty point for {dicom} at index {i}, skipping")
            continue

        p_r = safe_point(pnt_r)
        p_l = safe_point(pnt_l)
        if not p_r or not p_l:
            print(f"Warning: Non-integer points for {dicom} at index {i}, skipping")
            continue
        pnt_width = (p_r[0] - p_l[0])
        pnt_width_lst.append(pnt_width)
    
    if not pnt_width_lst:
        print(f"Warning: No valid point widths for {dicom}, skipping descending aorta")
        return

    pnt_indices = np.array(pnt_width_lst).argmax()
    tg_pnt_r = safe_eval(target_df.get(f'pnt_r_{pnt_indices + 1}', pd.Series([None])).values[0])
    tg_pnt_l = safe_eval(target_df.get(f'pnt_{pnt_indices + 1}', pd.Series([None])).values[0])
    p_tg_r = safe_point(tg_pnt_r)
    p_tg_l = safe_point(tg_pnt_l)
    if not p_tg_r or not p_tg_l:
        print(f"Warning: Empty or invalid target points for {dicom}, skipping descending aorta visualization")
        return

    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 10
    fontsize = 25

    img_height, img_width = cxr.size
    plt.imshow(cxr)

    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")

    texts = []
    plt.plot([xmin_width, xmin_width], [y_width - 50, y_width + 50], color="blue", linewidth=markersize)
    texts.append(plt.text(xmin_width - 30, y_width - (y_width * 0.1), f'{xmin_width}',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='right'))

    plt.plot([xmax_width, xmax_width], [y_width - 50, y_width + 50], color="blue", linewidth=markersize)
    texts.append(plt.text(xmax_width + 30, y_width - (y_width * 0.1), f'{xmax_width}',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    plt.plot([xmin_width, xmax_width], [y_width, y_width], color="blue", linewidth=markersize)
    texts.append(plt.text(xmax_width + 10, y_width - (y_width * 0.3), 'Trachea Width',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='center'))

    # ================# ================# ================# ================# ================# ================
    plt.plot([p_tg_r[0], p_tg_r[0]], [p_tg_r[1] - 50, p_tg_r[1] + 50], color="red", linewidth=markersize)
    texts.append(plt.text(p_tg_r[0] + 20, p_tg_r[1] + ((img_height - p_tg_r[1]) * 0.05), f'{p_tg_r[0]}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    plt.plot([p_tg_l[0], p_tg_l[0]], [p_tg_l[1] - 50, p_tg_l[1] + 50], color="red", linewidth=markersize)
    texts.append(plt.text(p_tg_l[0] - 20, p_tg_l[1] + ((img_height - p_tg_l[1]) * 0.05), f'{p_tg_l[0]}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='right'))

    plt.plot([p_tg_l[0], p_tg_r[0]], [p_tg_l[1], p_tg_l[1]], color="red", linewidth=markersize)
    texts.append(plt.text(p_tg_r[0] + 10, p_tg_l[1] + ((img_height - p_tg_l[1]) * 0.15), 'Descending Aorta Width',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='center'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_desc_aorta_tortuous(args, dicom, target_df, save_dir):
    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 10
    fontsize = 25

    plt.imshow(cxr)

    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")
    for column in target_df.columns:
        if 'pnt_r_' in column:
            pnt_r = safe_eval(target_df.get(column, pd.Series([None])).values[0])
            if not pnt_r or len(pnt_r) == 0:
                continue
            plt.plot([pnt_r[0]], [pnt_r[-1]], color="red", marker='o', markersize=markersize)
            plt.text(pnt_r[0] + 10, pnt_r[-1] - 5, f'{pnt_r}',
                     fontsize=fontsize, color='red', bbox=bbox_props,
                     verticalalignment='top', horizontalalignment='left')

    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_aortic_knob_enlarged(args, dicom, target_df, save_dir):
    y_width = safe_int(target_df.get('trachea_y_width', pd.Series([None])).values[0])
    xmin_width = safe_int(target_df.get('trachea_xmin_width', pd.Series([None])).values[0])
    xmax_width = safe_int(target_df.get('trachea_xmax_width', pd.Series([None])).values[0])

    aortic_arch_start = safe_int(target_df.get('xmax_trachea_mean', pd.Series([None])).values[0])

    xmax_aortic_arch = safe_int(target_df.get('xmax_aortic_arch', pd.Series([None])).values[0])
    xmax_desc_sub = safe_int(target_df.get('xmax_desc_sub', pd.Series([None])).values[0])

    ymin_aortic_arch = safe_int(target_df.get('ymin_aortic_arch', pd.Series([None])).values[0])
    ymax_aortic_arch = safe_int(target_df.get('ymax_aortic_arch', pd.Series([None])).values[0])

    # We'll compute sensible fallbacks later once we know image width/height

    cxr = return_cxr(args, dicom)

    img_height, img_width = cxr.size
    # fallback for missing horizontal bounds: use aortic_arch_start + 50 or 10% of remaining width
    if xmax_aortic_arch is None and xmax_desc_sub is None:
        if aortic_arch_start is not None:
            aortic_arch_end = aortic_arch_start + max(50, int((img_width - aortic_arch_start) * 0.1))
        else:
            aortic_arch_end = int(img_width * 0.6)
    else:
        vals = [v for v in (xmax_aortic_arch, xmax_desc_sub) if v is not None]
        aortic_arch_end = max(vals) if vals else int(img_width * 0.6)

    # fallback for missing vertical bounds: try y_max or ysub_desc, else use upper-third of image
    if ymin_aortic_arch is None or ymax_aortic_arch is None:
        possible_y = safe_int(target_df.get('y_max', pd.Series([None])).values[0])
        possible_ysub = safe_int(target_df.get('ysub_desc', pd.Series([None])).values[0])
        if possible_y is not None:
            y_aortic_arch = possible_y
        elif possible_ysub is not None:
            y_aortic_arch = possible_ysub
        else:
            y_aortic_arch = int(img_height * 0.33)
    else:
        y_aortic_arch = (ymin_aortic_arch + ymax_aortic_arch) // 2
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 10
    fontsize = 25

    plt.imshow(cxr)
    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")

    texts = []
    # draw trachea width if values exist
    if y_width is not None and xmin_width is not None and xmax_width is not None:
        try:
            plt.plot([xmin_width, xmin_width], [y_width - 50, y_width + 50], color="blue", linewidth=markersize)
            texts.append(plt.text(xmin_width - 30, y_width - (y_width * 0.1), f'{xmin_width}',
                     fontsize=fontsize, color='blue', bbox=bbox_props,
                     verticalalignment='top', horizontalalignment='right'))

            plt.plot([xmax_width, xmax_width], [y_width - 50, y_width + 50], color="blue", linewidth=markersize)
            texts.append(plt.text(xmax_width + 30, y_width - (y_width * 0.1), f'{xmax_width}',
                     fontsize=fontsize, color='blue', bbox=bbox_props,
                     verticalalignment='top', horizontalalignment='left'))

            plt.plot([xmin_width, xmax_width], [y_width, y_width], color="blue", linewidth=markersize)
            texts.append(plt.text(xmax_width + 10, y_width - (y_width * 0.2), 'Trachea Width',
                     fontsize=fontsize, color='blue', bbox=bbox_props,
                     verticalalignment='top', horizontalalignment='center'))
        except Exception:
            pass


    # ================
    # plot aortic arch markers (use aortic_arch_start fallback if missing)
    if aortic_arch_start is None:
        aortic_arch_start = max(10, int(aortic_arch_end - max(50, int(img_width * 0.05))))

    plt.plot([aortic_arch_start, aortic_arch_start], [y_aortic_arch - 50, y_aortic_arch + 50], color="red", linewidth=markersize)
    texts.append(plt.text(aortic_arch_start - 30, y_aortic_arch + ((img_height - y_aortic_arch) * 0.05), f'{aortic_arch_start}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='right'))

    plt.plot([aortic_arch_end, aortic_arch_end], [y_aortic_arch - 50, y_aortic_arch + 50], color="red", linewidth=markersize)
    texts.append(plt.text(aortic_arch_end + 30, y_aortic_arch + ((img_height - y_aortic_arch) * 0.05), f'{aortic_arch_end}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    plt.plot([aortic_arch_start, aortic_arch_end], [y_aortic_arch, y_aortic_arch], color="red", linewidth=markersize)
    texts.append(plt.text(aortic_arch_end + 10, y_aortic_arch + ((img_height - y_aortic_arch) * 0.1), 'Aortic Knob Width',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='center'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_asc_aorta_enlarged(args, dicom, target_df, save_dir):
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

    # load points first
    pnt_heart = safe_eval(target_df.get('pnt_heart', pd.Series([None])).values[0])
    pnt_trachea = safe_eval(target_df.get('pnt_trachea', pd.Series([None])).values[0])
    pnt_heart = safe_point(pnt_heart)
    pnt_trachea = safe_point(pnt_trachea)
    if not pnt_heart or not pnt_trachea:
        print(f"Warning: Empty pnt_heart or pnt_trachea for {dicom}, skipping aortic knob")
        return

    # load CXR to determine image size; will also be used for visualization
    cxr = return_cxr(args, dicom)
    h, w = cxr.size

    # attempt to read mask; if missing, create a zero placeholder matching image size
    fname_ascending_aorta = os.path.join(args.cxas_base_dir, dicom, f"ascending aorta.png")
    mask_ascending_aorta = imread_gray(fname_ascending_aorta)
    if mask_ascending_aorta is None:
        print(f"Warning: Failed to read ascending aorta mask for {dicom}, creating placeholder mask and continuing")
        try:
            mask_ascending_aorta = np.zeros((h, w), dtype=np.uint8)
        except Exception:
            # fallback: try reversed shape
            mask_ascending_aorta = np.zeros((w, h), dtype=np.uint8)

    mask_refined = select_max_area_mask(mask_ascending_aorta)
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 1.5
    fontsize = 25

    plt.imshow(cxr)
    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    plt.imshow(vs.overlay_instances(masks=[mask_refined], assigned_colors=mask_color).get_image())
    plt.plot([pnt_heart[0], pnt_trachea[0]], [pnt_heart[1], pnt_trachea[1]], color="blue", linewidth=markersize)

    plt.axis('off')
    save_dir_all = f'{save_dir}/mask_n_line'
    os.makedirs(save_dir_all, exist_ok=True)
    plt.savefig(f'{save_dir_all}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_inclusion(args, dicom, target_df, save_dir):
    apex_right = safe_eval(target_df.get('apex_right', pd.Series([None])).values[0])
    apex_left = safe_eval(target_df.get('apex_left', pd.Series([None])).values[0])
    side_right = safe_eval(target_df.get('side_right', pd.Series([None])).values[0])
    side_left = safe_eval(target_df.get('side_left', pd.Series([None])).values[0])
    cpa_right = safe_eval(target_df.get('cpa_right', pd.Series([None])).values[0])
    cpa_left = safe_eval(target_df.get('cpa_left', pd.Series([None])).values[0])

    pnt_lst = [apex_right, apex_left, side_right, side_left, cpa_right, cpa_left]
    # normalize points
    norm_pnts = []
    for p in pnt_lst:
        norm = safe_point(p)
        norm_pnts.append(norm)
    if any(not p for p in norm_pnts):
        print(f"Warning: Empty lung points for {dicom}, skipping inclusion")
        return

    cxr = return_cxr(args, dicom)

    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    plt.imshow(cxr)
    markersize = 10
    color_lst = ['red', 'red', 'green', 'green', 'blue', 'blue']
    for idx, pnt_xy in enumerate(norm_pnts):
        plt.plot([pnt_xy[0]], [pnt_xy[1]], color=color_lst[idx], marker='o', markersize=markersize)

    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def pnt_inspiration(args, dicom, target_df, save_dir):
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

    label = safe_int(target_df.get('label', pd.Series([None])).values[0])
    if label is None:
        print(f"Warning: Missing label for {dicom}, skipping inspiration")
        return
    # allow labels outside range by clamping to valid ribs (some CSVs use 0-based or 0 to indicate first)
    if not (1 <= label <= len(rib_posterior)):
        old_label = label
        label = max(1, min(label, len(rib_posterior)))
        print(f"Info: Adjusted label {old_label} -> {label} for {dicom}")
    fname_rib = os.path.join(args.cxas_base_dir, dicom, f'{rib_posterior[(label - 1)]} right.png')

    mask_rib = cv2.imread(fname_rib, 0)
    
    if mask_rib is None:
        print(f"Warning: Failed to read rib mask for {dicom}, skipping inspiration")
        return
    
    mask_rib = (mask_rib / 255).astype(int)
    mask_rib_refined, _ = select_max_width_mask(mask_rib)

    cxr = return_cxr(args, dicom)

    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    x_lung_mid_combined = safe_int(target_df.get('x_lung_mid_combined', pd.Series([None])).values[0])
    lung_y_lst_x_lung_mid = safe_eval(target_df.get('lung_y_lst_x_lung_mid', pd.Series([None])).values[0])
    
    if not lung_y_lst_x_lung_mid or len(lung_y_lst_x_lung_mid) == 0 or x_lung_mid_combined is None:
        print(f"Warning: Empty lung_y_lst_x_lung_mid or x_lung_mid_combined for {dicom}, skipping inspiration point")
        return
    # ensure integer y values
    y0 = safe_int(lung_y_lst_x_lung_mid[0])
    y1 = safe_int(lung_y_lst_x_lung_mid[-1])
    if y0 is None or y1 is None:
        print(f"Warning: Invalid lung y-values for {dicom}, skipping inspiration point")
        return

    plt.plot([x_lung_mid_combined, x_lung_mid_combined],
             [y0, y1], color="red", linewidth=10)

    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    plt.imshow(vs.overlay_instances(masks=[mask_rib_refined], assigned_colors=mask_color).get_image())
    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_mw(args, dicom, target_df, save_dir):
    y_medi = safe_int(target_df.get('y_medi', pd.Series([None])).values[0])
    xmin_medi = safe_int(target_df.get('xmin_medi', pd.Series([None])).values[0])
    xmax_medi = safe_int(target_df.get('xmax_medi', pd.Series([None])).values[0])

    xmin_rlung = safe_int(target_df.get('xmin_rlung', pd.Series([None])).values[0])
    xmax_llung = safe_int(target_df.get('xmax_llung', pd.Series([None])).values[0])
    if None in (y_medi, xmin_medi, xmax_medi, xmin_rlung, xmax_llung):
        print(f"Warning: Missing mediastinal/ lung bounds for {dicom}, skipping mediastinal width")
        return

    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    plt.imshow(cxr)
    markersize=10
    fontsize=25
    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")

    texts = []
    plt.plot([xmin_rlung, xmin_rlung], [y_medi - 100, y_medi + 100], color="blue", linewidth=markersize)
    texts.append(plt.text(xmin_rlung, y_medi + 120, f'{xmin_rlung}',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='right'))

    plt.plot([xmax_llung, xmax_llung], [y_medi - 100, y_medi + 100], color="blue", linewidth=markersize)
    texts.append(plt.text(xmax_llung, y_medi + 120, f'{xmax_llung}',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    plt.plot([xmin_rlung, xmax_llung], [y_medi, y_medi], color="blue", linewidth=markersize)
    texts.append(plt.text(xmin_rlung + 10, (y_medi + 10), 'Thoracic Width',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    # =============# =============# =============# =============# =============# =============# =============
    plt.plot([xmin_medi, xmin_medi], [y_medi - 100, y_medi + 100], color="red", linewidth=markersize)
    texts.append(plt.text(xmin_medi, y_medi + 120, f'{xmin_medi}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='right'))

    plt.plot([xmax_medi, xmax_medi], [y_medi - 100, y_medi + 100], color="red", linewidth=markersize)
    texts.append(plt.text(xmax_medi, y_medi + 120, f'{xmax_medi}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    plt.plot([xmin_medi, xmax_medi], [y_medi, y_medi], color="red", linewidth=markersize)
    texts.append(plt.text(((xmin_medi + xmax_medi) // 2), (y_medi - 150), f'Mediastinal Width',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='bottom', horizontalalignment='center'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_projection(args, dicom, target_df, save_dir):
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

    mask_r_scapular = cv2.imread(fname_r_scapular, 0)
    mask_l_scapular = cv2.imread(fname_l_scapular, 0)

    mask_r_lung = cv2.imread(fname_r_lung, 0)
    mask_l_lung = cv2.imread(fname_l_lung, 0)
    
    if mask_r_scapular is None or mask_l_scapular is None or mask_r_lung is None or mask_l_lung is None:
        print(f"Warning: Failed to read scapular/lung masks for {dicom}, skipping rotation")
        return
    
    mask_r_scapular = mask_r_scapular // 255
    mask_l_scapular = mask_l_scapular // 255
    mask_r_lung = mask_r_lung // 255
    mask_l_lung = mask_l_lung // 255

    mask_r_scapular_refined = select_max_area_mask(mask_r_scapular)
    mask_l_scapular_refined = select_max_area_mask(mask_l_scapular)

    overlap_region_r = np.logical_and(mask_r_lung, mask_r_scapular_refined)
    overlap_region_l = np.logical_and(mask_l_lung, mask_l_scapular_refined)

    nonoverlap_region_r = np.logical_and(mask_r_scapular_refined, ~overlap_region_r)
    nonoverlap_region_l = np.logical_and(mask_l_scapular_refined, ~overlap_region_l)


    mask_per_type = {
        'overlap_r': overlap_region_r,
        'overlap_l': overlap_region_l,
        'nononverlap_r': nonoverlap_region_r,
        'nononverlap_l': nonoverlap_region_l
    }
    center_pos_per_type = {}
    for name, mask in mask_per_type.items():
        y_coords, x_coords = np.nonzero(mask)  # y = row indices, x = col indices

        if len(x_coords) > 0 and len(y_coords) > 0:
            # Compute center position as the mean of nonzero coordinates
            x_center = int(np.mean(x_coords))
            y_center = int(np.mean(y_coords))
            center_pos = [x_center, y_center]
            center_pos_per_type[name] = center_pos

    overlap_region_right = target_df['overlap_region_right'].values[0]
    overlap_region_left = target_df['overlap_region_left'].values[0]
    scapular_region_right = target_df['scapular_region_right'].values[0]
    scapular_region_left = target_df['scapular_region_left'].values[0]

    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 10
    fontsize = 25

    plt.imshow(cxr)
    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="square,pad=0")
    if 'nononverlap_r' in center_pos_per_type:
        plt.text(center_pos_per_type['nononverlap_r'][0], center_pos_per_type['nononverlap_r'][-1],
                 f'{scapular_region_right}',
                 fontsize=fontsize, color='red', bbox=bbox_props,
                 verticalalignment='top', horizontalalignment='right')
    if 'nononverlap_l' in center_pos_per_type:
        plt.text(center_pos_per_type['nononverlap_l'][0], center_pos_per_type['nononverlap_l'][-1],
                 f'{scapular_region_left}',
                 fontsize=fontsize, color='red', bbox=bbox_props,
                 verticalalignment='top', horizontalalignment='left')
    plt.imshow(vs.overlay_instances(masks=[mask_r_scapular_refined, mask_l_scapular_refined],
                                    assigned_colors=mask_color * 2).get_image())  # labels=[str(scapular_region_right), str(scapular_region_left)],

    if 'overlap_r' in center_pos_per_type:
        plt.text(center_pos_per_type['overlap_r'][0], center_pos_per_type['overlap_r'][-1],
                 f'{overlap_region_right}',
                 fontsize=fontsize, color='blue', bbox=bbox_props,
                 verticalalignment='top', horizontalalignment='left')
    if 'overlap_l' in center_pos_per_type:
        plt.text(center_pos_per_type['overlap_l'][0], center_pos_per_type['overlap_l'][-1],
                 f'{overlap_region_left}',
                 fontsize=fontsize, color='blue', bbox=bbox_props,
                 verticalalignment='top', horizontalalignment='right')
    mask_color_overlap = [np.array([0, 0, 0.5], dtype=np.float32)]
    plt.imshow(vs.overlay_instances(masks=[overlap_region_r, overlap_region_l],
                                    assigned_colors=mask_color_overlap * 2).get_image())


    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_rotation(args, dicom, target_df, save_dir):
    def find_x_vertical(y, m, b):
        x = m * y + b
        return x

    right_end_pnt = safe_eval(target_df.get('right_end_pnt', pd.Series([None])).values[0])
    left_end_pnt = safe_eval(target_df.get('left_end_pnt', pd.Series([None])).values[0])
    
    if not right_end_pnt or not left_end_pnt:
        print(f"Warning: Empty clavicle end points for {dicom}, skipping rotation visualization")
        return

    midline_pnts = safe_eval(target_df.get('target_coords', pd.Series([None])).values[0])
    if not midline_pnts or len(midline_pnts) == 0:
        print(f"Warning: Empty midline_pnts for {dicom}, skipping rotation point")
        return
    m = round(target_df['m'].values[0], 3)
    b = round(target_df['b'].values[0], 3)

    midline_ymin = midline_pnts[0][-1]
    midline_ymax = midline_pnts[-1][-1]

    x_ymin = find_x_vertical(midline_ymin, m, b)
    x_ymax = find_x_vertical(midline_ymax, m, b)

    cxr = return_cxr(args, dicom)

    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    plt.imshow(cxr)
    markersize = 10

    texts = []
    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")
    plt.plot([right_end_pnt[0]], [right_end_pnt[-1]], color='red', marker='o', markersize=markersize)
    texts.append(plt.text((right_end_pnt[0] - 30), (right_end_pnt[-1] + 20), f'{right_end_pnt}',
             fontsize=25, color='red', bbox=bbox_props, verticalalignment='top', horizontalalignment='right'))

    plt.plot([left_end_pnt[0]], [left_end_pnt[-1]], color='blue', marker='o', markersize=markersize)
    texts.append(plt.text((left_end_pnt[0] + 30), (left_end_pnt[-1] + 20), f'{left_end_pnt}',
             fontsize=25, color='blue', bbox=bbox_props, verticalalignment='top', horizontalalignment='left'))

    plt.plot([x_ymin, x_ymax], [midline_ymin, midline_ymax], color="green", linewidth=markersize)
    texts.append(plt.text(x_ymin, (midline_ymin - 20), f'slope: {m} \n intercept: {b}',
             fontsize=25, color='green', bbox=bbox_props, verticalalignment='bottom', horizontalalignment='center'))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_trachea(args, dicom, target_df, save_dir):
    def find_x_vertical(y, m, b):
        x = m * y + b
        return x

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

    mask_trachea = cv2.imread(fname_trachea, 0)
    mask_carina = cv2.imread(fname_carina, 0)
    
    if mask_trachea is None or mask_carina is None:
        print(f"Warning: Failed to read trachea/carina masks for {dicom}, skipping trachea")
        return

    carina_nonzero = mask_carina.sum(axis=-1).nonzero()[0]
    if len(carina_nonzero) == 0:
        return  # Empty carina mask
    y_pos_carina_start = carina_nonzero[0]
    mask_refined = mask_trachea.copy()
    mask_refined[y_pos_carina_start:] = 0

    mask_refined = select_max_area_mask(mask_refined)

    x_min, x_max = target_df['midline_x_min'].values[0], target_df['midline_x_max'].values[0]
    y_min, y_max = target_df['y_min'].values[0], target_df['y_max'].values[0]
    target_points_y = np.linspace(y_min, y_max, 9, dtype=int)

    target_points = []
    for idx, y in enumerate(range(y_min, y_max + 1)):
        if y in target_points_y:
            if x_min == x_max:
                target_x = x_max
            else:
                m = (x_max - x_min) / (y_max - y_min)
                b = x_min - (m * y_min)
                target_x = find_x_vertical(y, m, b)
            target_points.append([target_x, y])

    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 1.5
    fontsize = 25

    plt.imshow(cxr)
    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    plt.imshow(vs.overlay_instances(masks=[mask_refined], assigned_colors=mask_color).get_image())

    for coord in target_points:
        plt.plot([coord[0]], [coord[-1]], color='blue', marker='o', markersize=markersize)

    plt.axis('off')
    save_dir_all = f'{save_dir}/mask_n_pnts'
    os.makedirs(save_dir_all, exist_ok=True)
    plt.savefig(f'{save_dir_all}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()


pnt_fn_per_dx = {
    'inclusion': pnt_inclusion,
    'inspiration': pnt_inspiration,
    'rotation': pnt_rotation,
    'projection': pnt_projection,
    'cardiomegaly': pnt_cardiomegaly,
    'mediastinal_widening': pnt_mw,
    'carina_angle': pnt_carina,
    'trachea_deviation': pnt_trachea,
    'aortic_knob_enlargement': pnt_aortic_knob_enlarged,
    'ascending_aorta_enlargement': pnt_asc_aorta_enlarged,
    'descending_aorta_enlargement': pnt_desc_aorta_enlarged,
    'descending_aorta_tortuous': pnt_desc_aorta_tortuous,
}

def check_if_processed(dicom, save_dir):
    """檢查影像是否已經處理過"""
    # check top-level and common subfolders where functions save outputs
    candidates = [
        os.path.join(save_dir, f'{dicom}.png'),
        os.path.join(save_dir, 'mask_n_line', f'{dicom}.png'),
        os.path.join(save_dir, 'mask_n_pnts', f'{dicom}.png'),
        os.path.join(save_dir, 'mask_n_pnts', f'{dicom}.jpg'),
        os.path.join(save_dir, 'mask_n_line', f'{dicom}.jpg'),
    ]
    for c in candidates:
        try:
            if os.path.exists(c):
                return True
        except Exception:
            continue
    return False

def process_single_pnt_image(args_tuple):
    """處理單個影像的包裝函數，用於多進程處理"""
    dicom, dx, save_dir, viz_df_data, args_dict = args_tuple
    
    # 重建 args 物件和 DataFrame
    from argparse import Namespace
    args = Namespace(**args_dict)
    viz_df = pd.DataFrame(viz_df_data)
    
    # 檢查是否已經處理過
    if check_if_processed(dicom, save_dir):
        return {'dicom': dicom, 'status': 'skipped', 'error': None}
    
    try:
        target_df = viz_df[viz_df['image_file'] == dicom]
        if target_df.empty:
            return {'dicom': dicom, 'status': 'skipped', 'error': 'no target row'}
        pnt_fn_per_dx[dx](args, dicom, target_df, save_dir)
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
    
    # Load metadata based on dataset type
    if args.dataset_name == 'mimic-cxr-jpg':
        args.mimic_meta = pd.read_csv(args.mimic_meta_file)
    else:
        args.mimic_meta = None  # Not needed for NIH dataset
    
    args.saved_dir_viz = os.path.join(args.saved_base_dir, f"{args.dataset_name}_viz")
    saved_path_viz_list = glob(os.path.join(args.saved_dir_viz, '*.csv'))
    
    # 顯示處理設定
    print(f"\n{'='*60}")
    print(f"Point-on-CXR 處理設定:")
    print(f"  資料集: {args.dataset_name}")
    print(f"  併行 worker 數: {args.num_workers}")
    if args.num_workers > 1:
        print(f"  模式: 多進程並行處理 (速度提升 {args.num_workers}x)")
    else:
        print(f"  模式: 單進程順序處理")
    print(f"  CUDA 加速: {'✓ 可用 (GPU)' if CUDA_AVAILABLE else '✗ 不可用 (CPU)'}")
    if CUDA_AVAILABLE:
        print(f"  GPU 設備: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    for saved_path_viz in saved_path_viz_list:
        dx = Path(saved_path_viz).stem
        if dx in pnt_fn_per_dx:
            save_dir = os.path.join(args.save_base_dir, 'pnt_on_cxr', dx)
            os.makedirs(save_dir, exist_ok=True)

            viz_df = pd.read_csv(saved_path_viz)
            # only keep rows marked as valid visualization to avoid eval on empty fields
            if 'has_valid_viz' in viz_df.columns and not args.ignore_has_valid_viz:
                try:
                    before_cnt = len(viz_df)
                    viz_df = viz_df[viz_df['has_valid_viz'] == True]
                    after_cnt = len(viz_df)
                    print(f"  Info: filtered viz rows by has_valid_viz: {before_cnt} -> {after_cnt}")
                except Exception:
                    # sometimes CSV stores boolean as string 'True'/'False'
                    before_cnt = len(viz_df)
                    viz_df = viz_df[viz_df['has_valid_viz'].astype(str).str.lower() == 'true']
                    after_cnt = len(viz_df)
                    print(f"  Info: filtered viz rows by has_valid_viz (string): {before_cnt} -> {after_cnt}")
            elif 'has_valid_viz' in viz_df.columns and args.ignore_has_valid_viz:
                print(f"  Info: 'has_valid_viz' column present but ignored due to --ignore_has_valid_viz; processing all {len(viz_df)} rows")
            dicom_list = viz_df['image_file'].tolist()
            if args.test_one_per_dx:
                if len(dicom_list) > 0:
                    dicom_list = dicom_list[:1]
                    print(f"  ⚙️ Test mode: processing 1 image for {dx} ({dicom_list[0]})")
            
            # 統計已完成的輸出檔案數量：在診斷資料夾內遞迴搜尋所有影像檔（包含子資料夾）
            completed_count = 0
            existing_basenames = set()
            if os.path.exists(save_dir):
                try:
                    # glob for image files recursively to include subfolders like mask_n_line
                    pattern = os.path.join(save_dir, '**', '*')
                    all_files = glob(pattern, recursive=True)
                except Exception:
                    all_files = []

                for fpath in all_files:
                    try:
                        low = str(fpath).lower()
                        if low.endswith('.png') or low.endswith('.jpg') or low.endswith('.jpeg'):
                            existing_basenames.add(os.path.splitext(os.path.basename(str(fpath)))[0])
                    except Exception:
                        continue

                # 正規化 dicom 列表（確保為檔名，無副檔名）
                normalized_dicoms = [os.path.splitext(os.path.basename(str(d)))[0] for d in dicom_list]
                for d in normalized_dicoms:
                    if d in existing_basenames:
                        completed_count += 1
            
            completion_rate = (completed_count / len(dicom_list) * 100) if len(dicom_list) > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"診斷任務: {dx}")
            print(f"  總影像數: {len(dicom_list)}")
            print(f"  已完成: {completed_count} ({completion_rate:.1f}%)")
            
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
                viz_df_data = viz_df.to_dict('list')  # 轉為字典格式傳遞
                task_args = [(dicom, dx, save_dir, viz_df_data, args_dict) for dicom in dicom_list]
                
                print(f">>> 啟動 Pool，準備處理 {len(task_args)} 個影像...")
                try:
                    with Pool(processes=args.num_workers) as pool:
                        results = list(tqdm(
                            pool.imap(process_single_pnt_image, task_args),
                            total=len(task_args),
                            desc=f'{dx}'
                        ))
                    
                    processed_count = sum(1 for r in results if r['status'] == 'success')
                    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
                    error_count = sum(1 for r in results if r['status'] == 'error')
                    
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
                # 單進程處理
                processed_count = 0
                skipped_count = 0
                error_count = 0
                
                for dicom in tqdm(dicom_list, total=len(dicom_list), desc=f'{dx}'):
                    if check_if_processed(dicom, save_dir):
                        skipped_count += 1
                        continue
                    
                    try:
                        target_df = viz_df[viz_df['image_file'] == dicom]
                        pnt_fn_per_dx[dx](args, dicom, target_df, save_dir)
                        processed_count += 1
                    except Exception as e:
                        print(f"\nError processing {dicom}: {e}")
                        error_count += 1
                        continue
            
            print(f"✅ {dx} 完成 - 新處理: {processed_count}, 跳過: {skipped_count}, 錯誤: {error_count}")
