import os
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode, Visualizer


def select_max_area_mask(mask):
    label_im, nb_labels = ndimage.label(mask)
    max_area = 0
    max_mask = mask
    for i in range(nb_labels):
        mask_compare = np.full(np.shape(label_im), i + 1)
        separate_mask = np.equal(label_im, mask_compare).astype(int)
        mask_region = separate_mask.sum()
        if mask_region > max_area:
            max_area = mask_region
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


def visualize_mask(args, mask_lst, save_path, dicom):
    mask_colors = [np.array([0.5, 0., 0.], dtype=np.float32)]

    os.makedirs(save_path, exist_ok=True)
    sid = args.metadata[args.metadata['dicom_id'] == dicom]['study_id'].values[0]
    pid = args.metadata[args.metadata['dicom_id'] == dicom]['subject_id'].values[0]

    cxr_path = f'{args.mimic_cxr_base_dir}/p{str(pid)[:2]}/p{pid}/s{sid}/{dicom}.jpg'
    cxr = Image.open(cxr_path).convert('RGB')

    vs = Visualizer(img_rgb=cxr, instance_mode=ColorMode.SEGMENTATION)
    plt.imshow(vs.overlay_instances(masks=mask_lst, assigned_colors=mask_colors * len(mask_lst)).get_image())
    plt.axis('off')
    plt.savefig(f'{save_path}/{dicom}.png', bbox_inches='tight')
    plt.close()

def visualize_midline(args, pnt_start, pnt_end, save_path, dicom):
    os.makedirs(save_path, exist_ok=True)

    sid = args.metadata[args.metadata['dicom_id'] == dicom]['study_id'].values[0]
    pid = args.metadata[args.metadata['dicom_id'] == dicom]['subject_id'].values[0]

    cxr_path = f'{args.mimic_cxr_base_dir}/p{str(pid)[:2]}/p{pid}/s{sid}/{dicom}.jpg'
    cxr = Image.open(cxr_path).convert('RGB')

    plt.imshow(cxr)
    plt.plot([pnt_start[0], pnt_end[0]], [pnt_start[-1], pnt_end[-1]], color="red", linewidth=1.5)
    plt.axis('off')
    plt.savefig(f'{save_path}/{dicom}.png', bbox_inches='tight')
    plt.close()