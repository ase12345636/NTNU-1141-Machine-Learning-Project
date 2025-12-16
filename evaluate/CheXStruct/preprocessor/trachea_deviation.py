import os
import cv2
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from collections import Counter

from utils.constants import TARGET_MASK_LIST, MD_RANGES
from utils.utils import select_max_area_mask, select_max_width_mask


def line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)  # Slope
    b = y1 - m * x1  # Intercept
    return m, b


def residue(coords):
    coords = np.array(coords)
    p1 = coords[0]
    p_last = coords[-1]
    residues = []

    if p1[0] == p_last[0]:  # Vertical line case
        for p in coords[1:-1]:  # Exclude the first and last points
            residue = abs(p[0] - p1[0])
            residues.append(residue)
    else:  # Non-vertical line
        m, b = line_equation(p1, p_last)
        for p in coords[1:-1]:  # Exclude the first and last points
            x, y = p
            residue = abs(m * x - y + b) / np.sqrt(m ** 2 + 1)
            residues.append(residue)

    return max(residues)

def x_std(coords):
    x_coords = coords[:, 0]
    return x_coords.std()

def least_squares_fit_vertical(points):
    y = points[:, 1]
    x = points[:, 0]

    y_mean = np.mean(y)
    x_mean = np.mean(x)

    m = np.sum((y - y_mean) * (x - x_mean)) / np.sum((y - y_mean) ** 2)
    b = x_mean - m * y_mean

    return m, b

def find_x_vertical(y, m, b):
    x = m * y + b
    return x


def is_ascending(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

def extract_trachea_deviation(meta_trachea, meta_vertebrae):
    meta_trachea_deviation = dict()

    md_range_midline = {'threshold_vertebrae_residue': 20, 'threshold_vertebrae_std': 30}

    if ('qc_valid_trachea' in meta_trachea):
        if meta_trachea['qc_valid_trachea']:
            xy_coords = meta_vertebrae['xy_coords']
            y_coords = xy_coords[:, -1]

            y_min = meta_trachea['trachea_y_min']
            y_max = meta_trachea['trachea_y_max']
            mask_refined = meta_trachea['trachea_mask_refined']
            mask_btw_trachea = (y_coords >= y_min) & (y_coords <= y_max)
            xy_coords_btw_trachea = xy_coords[mask_btw_trachea]

            idx_btw_trachea = np.where(mask_btw_trachea == 1)[0]
            if len(idx_btw_trachea):
                if mask_btw_trachea.sum() == (idx_btw_trachea[-1] - idx_btw_trachea[0]) + 1:  # check whether float('inf') is located btw the TracheaDeviation
                    if is_ascending(xy_coords_btw_trachea[:, -1]) & (len(xy_coords_btw_trachea[:, -1]) == len(set(xy_coords_btw_trachea[:, -1]))):
                        if len(xy_coords_btw_trachea) > 2:  # For quality control, we need more than 3 coords
                            midline_residue = residue(xy_coords_btw_trachea)
                            midline_x_std = x_std(xy_coords_btw_trachea)

                            qc_valid_midline = (midline_residue < md_range_midline['threshold_vertebrae_residue']) \
                                               & (midline_x_std < md_range_midline['threshold_vertebrae_std'])
                            if qc_valid_midline:  #
                                label_lst, ratio_lst = [], []
                                if len(set(xy_coords_btw_trachea[:, 0])) != 1:
                                    m, b = least_squares_fit_vertical(xy_coords_btw_trachea)
                                    x_min = find_x_vertical(y_min, m, b)
                                    x_max = find_x_vertical(y_max, m, b)
                                else:
                                    midline_x = xy_coords_btw_trachea[0][0]
                                    x_min, x_max = midline_x, midline_x
                                for row in range(y_min, y_max + 1):
                                    if len(set(xy_coords_btw_trachea[:, 0])) != 1:
                                        midline_x = find_x_vertical(row, m, b)
                                    x_coords = np.where(mask_refined[row] == mask_refined.max())[0]
                                    if len(x_coords) > 0:
                                        rlmost_x = x_coords[0]  # right lung side
                                        llmost_x = x_coords[-1]  # left lung side

                                        rl_distance = midline_x - rlmost_x
                                        ll_distance = llmost_x - midline_x

                                        if rlmost_x > midline_x:  # midline_x < rlmost_x < llmost_x
                                            blank_distance = rlmost_x - midline_x
                                            width = llmost_x - midline_x
                                            ratio = blank_distance / width
                                            label = 'left'
                                        elif llmost_x < midline_x:  # rlmost_x < llmost_x < midline_x
                                            blank_distance = llmost_x - midline_x
                                            width = rlmost_x - midline_x
                                            ratio = blank_distance / width
                                            label = 'right'
                                        else:  # rlmost_x <= midline_x <= llmost_x
                                            ratio = (rl_distance / ll_distance) if ll_distance >= rl_distance \
                                                else (ll_distance / rl_distance)
                                            label = 'flat'

                                        label_lst.append(label)
                                        ratio_lst.append(ratio)

                                if len(label_lst):
                                    label_lst = np.array(label_lst)
                                    ratio_lst = np.array(ratio_lst)

                                    target_points_y = np.linspace(y_min, y_max, 9, dtype=int)

                                    label_lst_tgt_pnt, ratio_lst_tgt_pnt, target_points = [], [], []
                                    for idx, y in enumerate(range(y_min, y_max + 1)):
                                        if y in target_points_y:
                                            label_lst_tgt_pnt.append(label_lst[idx])
                                            ratio_lst_tgt_pnt.append(ratio_lst[idx])

                                            if x_min == x_max:
                                                target_x = x_max
                                            else:
                                                m = (x_max - x_min) / (y_max - y_min)
                                                b = x_min - (m * y_min)
                                                target_x = find_x_vertical(y, m, b)
                                            target_points.append([target_x, y])

                                    label_lst_tgt_pnt = np.array(label_lst_tgt_pnt)
                                    ratio_lst_tgt_pnt = np.array(ratio_lst_tgt_pnt)

                                    mask_label_shift = (label_lst_tgt_pnt != 'flat')
                                    mask_ratio_flat = (ratio_lst_tgt_pnt <= 0.0)

                                    mask_flat = (mask_label_shift & mask_ratio_flat)
                                    label_lst_tgt_pnt[mask_flat] = 'flat'

                                    label_counter = Counter(label_lst_tgt_pnt)
                                    max_value = max(label_counter.values())

                                    label = sorted([key for key, value in label_counter.items() if value == max_value])

                                    def first_satisfied_label(label_lst):
                                        count = Counter()
                                        threshold = len(label_lst) // len(set(label_lst))

                                        first_satisfied = []
                                        for label in label_lst:
                                            count[label] += 1
                                            if count[label] == threshold:
                                                first_satisfied.append(label)

                                        final_label = "&".join(first_satisfied)
                                        return final_label

                                    if len(label) == 1:
                                        final_label = label[0]
                                    else:
                                        label_lst_tgt_pnt = list(filter(lambda x: x != 'flat', label_lst_tgt_pnt))
                                        final_label = first_satisfied_label(label_lst_tgt_pnt)

                                    meta_trachea_deviation['direction'] = final_label
                                    meta_trachea_deviation['direction_per_pnt'] = label_lst_tgt_pnt
                                    for idx, pnt in enumerate(target_points):
                                        meta_trachea_deviation[f'point_{idx + 1}'] = pnt

                                    if final_label in ['flat']:
                                        meta_trachea_deviation['label'] = 0
                                    else:
                                        meta_trachea_deviation['label'] = 1

                                    meta_trachea_deviation['viz_midline_x_min'] = x_min
                                    meta_trachea_deviation['viz_midline_x_max'] = x_max
                                    meta_trachea_deviation['viz_y_min'] = y_min
                                    meta_trachea_deviation['viz_y_max'] = y_max
    return meta_trachea_deviation












