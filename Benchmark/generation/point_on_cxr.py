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
import matplotlib.pyplot as plt
from adjustText import adjust_text
from detectron2.utils.visualizer import ColorMode, Visualizer

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

    args = parser.parse_args()
    return args

mask_color = [np.array([0.5, 0., 0.], dtype=np.float32)]

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

    coord_mask = eval(target_df['coord_mask'].values[0])
    xmin_heart, ymin_heart, xmax_heart, ymax_heart = coord_mask
    ymean_heart = (ymin_heart + ymax_heart) // 2

    xmin_lung = target_df['xmin_lung'].values[0]
    xmax_lung = target_df['xmax_lung'].values[0]

    fname_rlung = os.path.join(args.cxas_base_dir, dicom, f"right lung.png")
    fname_llung = os.path.join(args.cxas_base_dir, dicom, f"left lung.png")

    mask_rlung = cv2.imread(fname_rlung, 0)
    mask_llung = cv2.imread(fname_llung, 0)

    mask_rlung_refined = select_max_area_mask(mask_rlung)[ymin_heart:]
    mask_llung_refined = select_max_area_mask(mask_llung)[ymin_heart:]
    lungs = (mask_rlung_refined | mask_llung_refined)

    y_xmin_indices = lungs[:, xmin_lung].nonzero()[0].tolist()
    y_xmax_indices = lungs[:, xmax_lung].nonzero()[0].tolist()

    y_indices = set(y_xmax_indices + y_xmin_indices)
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
    refined_tgpnt = eval(target_df['refined_tgpnt'].values[0])
    pnt_rl, pnt_c, pnt_ll = refined_tgpnt

    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 10
    fontsize = 25
    plt.imshow(cxr)

    bbox_props = dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="square,pad=0")

    texts = []
    plt.plot([pnt_rl[0]], [pnt_rl[-1]], color='red', marker='o', markersize=markersize)
    texts.append(plt.text(pnt_rl[0] - 20, pnt_rl[-1] + 15, f'A: {pnt_rl}',
                          fontsize=fontsize, color='red', bbox=bbox_props,
                          verticalalignment='top', horizontalalignment='right'))

    plt.plot([pnt_c[0]], [pnt_c[-1]], color='green', marker='o', markersize=markersize)
    texts.append(plt.text(pnt_c[0], pnt_c[-1] - 25, f'B: {pnt_c}',
                          fontsize=fontsize, color='green', bbox=bbox_props,
                          verticalalignment='bottom', horizontalalignment='center'))

    plt.plot([pnt_ll[0]], [pnt_ll[-1]], color='blue', marker='o', markersize=markersize)
    texts.append(plt.text(pnt_ll[0] + 20, pnt_ll[-1] + 15, f'C: {pnt_ll}',
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
        pnt_r = eval(target_df[f'pnt_r_{i + 1}'].values[0])
        pnt_l = eval(target_df[f'pnt_{i + 1}'].values[0])

        pnt_width = (pnt_r[0] - pnt_l[0])
        pnt_width_lst.append(pnt_width)

    pnt_indices = np.array(pnt_width_lst).argmax()
    tg_pnt_r = eval(target_df[f'pnt_r_{pnt_indices + 1}'].values[0])
    tg_pnt_l = eval(target_df[f'pnt_{pnt_indices + 1}'].values[0])

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
    plt.plot([tg_pnt_r[0], tg_pnt_r[0]], [tg_pnt_r[-1] - 50, tg_pnt_r[-1] + 50], color="red", linewidth=markersize)
    texts.append(plt.text(tg_pnt_r[0] + 20, tg_pnt_r[-1] + ((img_height - tg_pnt_r[-1]) * 0.05), f'{tg_pnt_r[0]}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='left'))

    plt.plot([tg_pnt_l[0], tg_pnt_l[0]], [tg_pnt_l[-1] - 50, tg_pnt_l[-1] + 50], color="red", linewidth=markersize)
    texts.append(plt.text(tg_pnt_l[0] - 20, tg_pnt_l[-1] + ((img_height - tg_pnt_l[-1]) * 0.05), f'{tg_pnt_l[0]}',
             fontsize=fontsize, color='red', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='right'))

    plt.plot([tg_pnt_l[0], tg_pnt_r[0]], [tg_pnt_l[-1], tg_pnt_l[-1]], color="red", linewidth=markersize)
    texts.append(plt.text(tg_pnt_r[0] + 10, tg_pnt_l[-1] + ((img_height - tg_pnt_l[-1]) * 0.15), 'Descending Aorta Width',
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
            pnt_r = eval(target_df[column].values[0])
            plt.plot([pnt_r[0]], [pnt_r[-1]], color="red", marker='o', markersize=markersize)
            plt.text(pnt_r[0] + 10, pnt_r[-1] - 5, f'{pnt_r}',
                     fontsize=fontsize, color='red', bbox=bbox_props,
                     verticalalignment='top', horizontalalignment='left')

    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_aortic_knob_enlarged(args, dicom, target_df, save_dir):
    y_width = target_df['trachea_y_width'].values[0]
    xmin_width = target_df['trachea_xmin_width'].values[0]
    xmax_width = target_df['trachea_xmax_width'].values[0]

    aortic_arch_start = round(target_df['xmax_trachea_mean'].values[0])

    xmax_aortic_arch = target_df['xmax_aortic_arch'].values[0]
    xmax_desc_sub = target_df['xmax_desc_sub'].values[0]
    aortic_arch_end = xmax_aortic_arch if (xmax_aortic_arch > xmax_desc_sub) else xmax_desc_sub

    ymin_aortic_arch = target_df['ymin_aortic_arch'].values[0]
    ymax_aortic_arch = target_df['ymax_aortic_arch'].values[0]
    y_aortic_arch = (ymin_aortic_arch + ymax_aortic_arch) // 2

    cxr = return_cxr(args, dicom)

    img_height, img_width = cxr.size
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 10
    fontsize = 25

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
    texts.append(plt.text(xmax_width + 10, y_width - (y_width * 0.2), 'Trachea Width',
             fontsize=fontsize, color='blue', bbox=bbox_props,
             verticalalignment='top', horizontalalignment='center'))


    # ================
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

    fname_ascending_aorta = os.path.join(args.cxas_base_dir, dicom, f"ascending aorta.png")
    mask_ascending_aorta = cv2.imread(fname_ascending_aorta, 0)
    mask_refined = select_max_area_mask(mask_ascending_aorta)

    pnt_heart = eval(target_df['pnt_heart'].values[0])
    pnt_trachea = eval(target_df['pnt_trachea'].values[0])

    cxr = return_cxr(args, dicom)
    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    markersize = 1.5
    fontsize = 25

    plt.imshow(cxr)
    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    plt.imshow(vs.overlay_instances(masks=[mask_refined], assigned_colors=mask_color).get_image())
    plt.plot([pnt_heart[0], pnt_trachea[0]], [pnt_heart[-1], pnt_trachea[-1]], color="blue", linewidth=markersize)

    plt.axis('off')
    save_dir_all = f'{save_dir}/mask_n_line'
    os.makedirs(save_dir_all, exist_ok=True)
    plt.savefig(f'{save_dir_all}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_inclusion(args, dicom, target_df, save_dir):
    apex_right = eval(target_df['apex_right'].values[0])
    apex_left = eval(target_df['apex_left'].values[0])
    side_right = eval(target_df['side_right'].values[0])
    side_left = eval(target_df['side_left'].values[0])
    cpa_right = eval(target_df['cpa_right'].values[0])
    cpa_left = eval(target_df['cpa_left'].values[0])

    pnt_lst = [apex_right, apex_left, side_right, side_left, cpa_right, cpa_left]

    cxr = return_cxr(args, dicom)

    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    plt.imshow(cxr)
    markersize = 10
    color_lst = ['red', 'red', 'green', 'green', 'blue', 'blue']
    for idx, pnt_xy in enumerate(pnt_lst):
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
    fname_rib = os.path.join(args.cxas_base_dir, dicom, f'{rib_posterior[(label - 1)]} right.png')

    mask_rib = cv2.imread(fname_rib, 0)
    mask_rib = (mask_rib / 255).astype(int)
    mask_rib_refined, _ = select_max_width_mask(mask_rib)

    cxr = return_cxr(args, dicom)

    h, w = cxr.size
    dpi = 100
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    x_lung_mid_combined = target_df['x_lung_mid_combined'].values[0]
    lung_y_lst_x_lung_mid = eval(target_df['lung_y_lst_x_lung_mid'].values[0])
    plt.plot([x_lung_mid_combined, x_lung_mid_combined],
             [lung_y_lst_x_lung_mid[0], lung_y_lst_x_lung_mid[-1]], color="red", linewidth=10)

    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    plt.imshow(vs.overlay_instances(masks=[mask_rib_refined], assigned_colors=mask_color).get_image())
    plt.axis('off')
    plt.savefig(f'{save_dir}/{dicom}.png', dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()

def pnt_mw(args, dicom, target_df, save_dir):
    y_medi = target_df['y_medi'].values[0]
    xmin_medi = target_df['xmin_medi'].values[0]
    xmax_medi = target_df['xmax_medi'].values[0]

    xmin_rlung = target_df['xmin_rlung'].values[0]
    xmax_llung = target_df['xmax_llung'].values[0]

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

    mask_r_scapular = cv2.imread(fname_r_scapular, 0) // 255  # .sum() 0 ~ 1 scale
    mask_l_scapular = cv2.imread(fname_l_scapular, 0) // 255

    mask_r_lung = cv2.imread(fname_r_lung, 0) // 255
    mask_l_lung = cv2.imread(fname_l_lung, 0) // 255

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

    right_end_pnt = eval(target_df['right_end_pnt'].values[0])
    left_end_pnt = eval(target_df['left_end_pnt'].values[0])

    midline_pnts = eval(target_df['target_coords'].values[0])
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

    y_pos_carina_start = mask_carina.sum(axis=-1).nonzero()[0][0]
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


if __name__ == "__main__":
    args = config()
    
    # Load metadata based on dataset type
    if args.dataset_name == 'mimic-cxr-jpg':
        args.mimic_meta = pd.read_csv(args.mimic_meta_file)
    else:
        args.mimic_meta = None  # Not needed for NIH dataset
    
    args.saved_dir_viz = os.path.join(args.saved_base_dir, f"{args.dataset_name}_viz")
    saved_path_viz_list = glob(os.path.join(args.saved_dir_viz, '*.csv'))
    
    for saved_path_viz in saved_path_viz_list:
        dx = Path(saved_path_viz).stem
        if dx in pnt_fn_per_dx:
            save_dir = os.path.join(args.save_base_dir, 'pnt_on_cxr', dx)
            os.makedirs(save_dir, exist_ok=True)

            viz_df = pd.read_csv(saved_path_viz)
            dicom_list = viz_df['image_file'].tolist()
            for dicom in tqdm(dicom_list, total=len(dicom_list)):
                target_df = viz_df[viz_df['image_file'] == dicom]
                pnt_fn_per_dx[dx](args, dicom, target_df, save_dir)
