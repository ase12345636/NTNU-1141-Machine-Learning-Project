import os
import cv2
import numpy as np
from scipy import ndimage
from collections import defaultdict

from utils.constants import TARGET_MASK_LIST, MD_RANGES
from utils.utils import select_max_area_mask


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

def extract_lung_mid_xpos(lung_mask):
    if lung_mask.sum() == 0:
        lung_mid_xpos = (0, 0)
    else:
        row_indices = lung_mask.sum(axis=-1).nonzero()[0]
        middle_area_indices = np.array_split(row_indices, 3)[1]
        middle_area = lung_mask[middle_area_indices]
        width_idx_per_line = [np.where(row == 1)[0] for row in middle_area]
        lung_mid_xpos_lst = [(line[0], line[-1]) for line in width_idx_per_line if line[0] != line[-1]]

        lung_mid_xpos = lung_mid_xpos_lst[int(len(lung_mid_xpos_lst) * (1 / 3))]
    return lung_mid_xpos


def extract_inspiration(image_data):
    meta_inspiration = defaultdict()

    margin = 1 / 2.5
    rib_posterior = ["posterior 1st rib", "posterior 2nd rib", "posterior 3rd rib",
                     "posterior 4th rib", "posterior 5th rib", "posterior 6th rib",
                     "posterior 7th rib", "posterior 8th rib", "posterior 9th rib",
                     "posterior 10th rib", "posterior 11th rib"]

    cxas_files = image_data['cxas']
    chexmask_files = image_data['chexmask']

    mask_dp = cv2.imread(cxas_files["right hemidiaphragm"], 0)
    mask_lung = cv2.imread(cxas_files["right lung"], 0)

    if (mask_dp.sum() != 0) and (mask_lung.sum() != 0):
        mask_lung_max_area = select_max_area_mask(mask_lung)
        img_height, img_width = mask_dp.shape

        fname_lung_chexmask = chexmask_files['RL_mask']
        if os.path.isfile(fname_lung_chexmask):
            mask_lung_chexmask = cv2.imread(fname_lung_chexmask, 0)
            mask_dp = np.logical_and(mask_dp, np.logical_not(mask_lung_chexmask))
            mask_lung_combined = select_max_area_mask(np.logical_or(mask_lung, mask_lung_chexmask))
        else:
            mask_lung_combined = select_max_area_mask(mask_lung)

        total_rib_masks = defaultdict(list)
        for cxas_mask_name in rib_posterior:
            mask_rib = cv2.imread(cxas_files[f'{cxas_mask_name} right'], 0)
            mask_rib = (mask_rib / 255).astype(int)
            total_rib_masks['cxas'].append(mask_rib)

            lung_mid_xpos_max = extract_lung_mid_xpos(mask_lung_max_area)
            lung_mid_xpos_combined = extract_lung_mid_xpos(mask_lung_combined)
            x_lung_mid_max = int(lung_mid_xpos_max[0] + (lung_mid_xpos_max[-1] - lung_mid_xpos_max[0]) * margin)
            x_lung_mid_combined = int(lung_mid_xpos_combined[0] + (lung_mid_xpos_combined[-1] - lung_mid_xpos_combined[0]) * margin)

        lung_y_lst_x_lung_mid = mask_lung_combined[:, x_lung_mid_combined].nonzero()[0].tolist()
        dp_y_lst_x_lung_mid = mask_dp[:, x_lung_mid_combined].nonzero()[0].tolist()

        if len(lung_y_lst_x_lung_mid) != 0 and len(dp_y_lst_x_lung_mid) != 0:
            indices_lung = list(range(lung_y_lst_x_lung_mid[0], lung_y_lst_x_lung_mid[-1] + 1))
            indices_dp = list(range(dp_y_lst_x_lung_mid[0], dp_y_lst_x_lung_mid[-1] + 1))

            intersect_lung_dp = len(set(indices_lung).intersection(indices_dp))
            iou_lung_dp = (intersect_lung_dp / len(set(indices_lung)))
        else:
            iou_lung_dp = -1

        pairs = [(i, i + 1) for i in range(len(rib_posterior) - 1)]
        iou_lst, inter_ratio_small_lst, inter_ratio_large_lst = [], [], []
        for idx, pair in enumerate(pairs):
            first_rib = total_rib_masks['cxas'][pair[0]]
            second_rib = total_rib_masks['cxas'][pair[-1]]
            first_rib_refined, _ = select_max_width_mask(first_rib)
            second_rib_refined, _ = select_max_width_mask(second_rib)

            # ======================
            # IoU (1st_rib, 2nd_rib)
            # ======================
            intersection = np.logical_and(first_rib_refined, second_rib_refined)
            union = np.logical_or(first_rib_refined, second_rib_refined)
            iou = np.sum(intersection) / np.sum(union)
            iou_lst.append(iou)

            # ======================
            # inter ratio
            # ======================
            inter_ratio_first = np.sum(intersection) / np.sum(first_rib_refined)
            inter_ratio_second = np.sum(intersection) / np.sum(second_rib_refined)
            if np.sum(first_rib_refined) > np.sum(second_rib_refined):
                inter_ratio_small_lst.append(inter_ratio_second)
                inter_ratio_large_lst.append(inter_ratio_first)
            else:
                inter_ratio_small_lst.append(inter_ratio_first)
                inter_ratio_large_lst.append(inter_ratio_second)

        exist_lst, intersect_lst = [], []
        rib_xpos_lst, rib_ypos_lst, rib_y_lst_x_lung_mid = [], [], []
        for idx, rib_mask in enumerate(total_rib_masks['cxas']):
            rib_mask_refined, rib_xpos = select_max_width_mask(rib_mask)

            # ===============
            # Mask Existence
            # ===============
            mask_existence = (rib_mask_refined.sum() != 0)
            exist_lst.append(mask_existence)

            # ============
            # Rib x, y indices
            # =============
            rib_xpos_lst.append(rib_xpos)

            if rib_xpos == (0, 0):
                rib_ypos = (0, 0)
            else:
                rib_y0 = rib_mask_refined[:, rib_xpos[0]].nonzero()[0][-1]
                rib_y1 = rib_mask_refined[:, rib_xpos[-1]].nonzero()[0][0]
                rib_ypos = (rib_y0, rib_y1)

            rib_ypos_lst.append(rib_ypos)

            # ============
            # Rib y indices
            # =============
            rib_y_lst_x_lung_mid.append(rib_mask_refined[:, x_lung_mid_combined].nonzero()[0].tolist())

            # =======================================
            # Intersection (Rib, Diaphragm)
            # =======================================
            intersect_region = np.logical_and(mask_dp, rib_mask_refined)
            intersect_ratio = intersect_region.sum() / rib_mask_refined.sum()
            intersect_lst.append(intersect_ratio)

        # Quality Control

        # Standard Line
        standard_line_diff = abs(x_lung_mid_max - x_lung_mid_combined) / img_width
        qc_valid_line_diff = (standard_line_diff < 0.15)
        qc_valid_line_exist = (x_lung_mid_combined != 0)
        qc_valid_line = (qc_valid_line_diff & qc_valid_line_exist)

        # Diaphragm
        qc_dp_invalid = (iou_lung_dp == -1)
        qc_dp_blank = (iou_lung_dp == 0.0)
        qc_dp_over_lung = (iou_lung_dp >= 0.7)

        qc_valid_dp = ~(qc_dp_invalid | qc_dp_blank | qc_dp_over_lung)

        if qc_valid_dp & qc_valid_line:
            # Ribs
            threshold_iou_pair = [0.5, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
            threshold_y_height = list({'12': 0.25, '23': 0.3, '34': 0.25,
                                       '45': 0.25, '56': 0.25, '67': 0.25,
                                       '78': 0.25, '89': 0.25, '910': 0.25,
                                       '1011': 0.25}.values())
            threshold_x_pos = list({'12': 0.0, '23': 0.0, '34': 0.05,
                                    '45': 0.05, '56': 0.05, '67': 0.05,
                                    '78': 0.05, '89': 0.05, '910': 0.05,
                                    '1011': 0.05}.values())
            # diff: 0 ~ 0.1
            threshold_large_overlap_01 = list({'12': 0.6, '23': 0.8, '34': 0.4,
                                               '45': 0.4, '56': 0.4, '67': 0.4,
                                               '78': 0.4, '89': 0.4, '910': 0.4,
                                               '1011': 0.4}.values())
            # diff: 0.1 ~ 0.2
            threshold_large_overlap_02 = list({'12': 1.0, '23': 0.8, '34': 0.8,
                                               '45': 0.6, '56': 0.6, '67': 0.6,
                                               '78': 0.6, '89': 0.6, '910': 0.6,
                                               '1011': 0.6}.values())

            threshold_ratio_rib_dp = 0.1

            y_rib_thick_lst, target_idx_lst = [], []
            standard_line = x_lung_mid_combined
            rib_xpos_lst = np.array(rib_xpos_lst)
            rib_ypos_lst = np.array(rib_ypos_lst)
            dp_y_lst = np.array(dp_y_lst_x_lung_mid)

            for y1 in rib_ypos_lst[:, 1]:
                y1_first_valid_rib = 0
                if y1 != 0:
                    y1_first_valid_rib = y1
                    break

            dp_y_lst = dp_y_lst[dp_y_lst > y1_first_valid_rib]
            for idx in range(rib_ypos_lst.shape[0]):
                rib_y_lst = np.array(rib_y_lst_x_lung_mid[idx])
                flag_rib_dp = set(rib_y_lst).intersection(dp_y_lst)

                if len(rib_y_lst):
                    y_rib_thick = (rib_y_lst[0], rib_y_lst[-1])
                    y_rib_thick_lst.append(y_rib_thick)
                    target_idx_lst.append(idx)

                if flag_rib_dp:
                    flag_intersection_rib = (rib_y_lst.min() < dp_y_lst.min())
                    if flag_intersection_rib:
                        target_idx = (idx + 1)
                        expected_rib = (idx + 1)

                    else:
                        target_idx = (idx + 1)
                        expected_rib = f'post'
                    break
                else:
                    target_idx = -1

            if target_idx == -1:
                if len(y_rib_thick_lst) == 0:
                    target_idx = -1
                else:
                    if len(dp_y_lst) == 0:
                        target_idx = -1
                    else:
                        y_height = (dp_y_lst[0] - y_rib_thick_lst[0][0])
                        if y_height <= 0:
                            target_idx = -1
                        else:
                            for idx, y_rib_thick in zip(target_idx_lst[::-1], y_rib_thick_lst[::-1]):
                                if (dp_y_lst[0] - y_rib_thick[-1]) > 0:
                                    y_last_height = (dp_y_lst[0] - y_rib_thick[-1])
                                    break
                            ratio_rib_dp = (y_last_height / y_height)
                            target_idx = (idx + 1) if ratio_rib_dp <= threshold_ratio_rib_dp else -1
                            expected_rib = (idx + 1)

            if target_idx != -1:
                if expected_rib == 'post':
                    assert (target_idx_lst[-1] + 1) == target_idx

                    if len(target_idx_lst) > 1 and (target_idx - (target_idx_lst[-2] + 1)) == 1:
                        if dp_y_lst[0] <= y_rib_thick_lst[-2][-1]:
                            expected_rib = (target_idx_lst[-2] + 1)
                        else:
                            expected_rib = (target_idx_lst[-2] + 1)
                    else:
                        if len(target_idx_lst) > 1:
                            target = list(range(target_idx_lst[-2] + 1, target_idx_lst[-1]))[::-1]
                            expected_rib = None
                            for idx in target:
                                y1 = rib_ypos_lst[:, 1][idx]
                                x1 = rib_xpos_lst[:, 1][idx]

                                y0 = rib_ypos_lst[:, 0][idx]
                                if y1 < dp_y_lst[0] and x1 < standard_line:
                                    expected_rib = (idx + 1)
                                    break
                                elif y1 < dp_y_lst[0] and y0 < dp_y_lst[0]:
                                    expected_rib = (idx + 1)
                                    break
                            if expected_rib == None:
                                expected_rib = (target_idx_lst[-2] + 1)
                        else:
                            target = list(range(target_idx_lst[0]))[::-1]
                            for idx in target:
                                y1 = rib_ypos_lst[:, 1][idx]
                                x1 = rib_xpos_lst[:, 1][idx]

                                y0 = rib_ypos_lst[:, 0][idx]
                                if y1 < dp_y_lst[0] and x1 < standard_line:
                                    expected_rib = (idx + 1)
                                    break
                                elif y1 < dp_y_lst[0] and y0 < dp_y_lst[0]:
                                    expected_rib = (idx + 1)
                                    break

            if target_idx != -1:
                exist_lst_target = np.array(exist_lst)[:target_idx]
                if False not in exist_lst_target:
                    lung_mid_xpos_combined = np.array(lung_mid_xpos_combined)
                    iou_lst = np.array(iou_lst)
                    inter_ratio_small_lst = np.array(inter_ratio_small_lst)

                    rib_x0_lst = rib_xpos_lst[:, 0][:target_idx]
                    rib_x1_lst = rib_xpos_lst[:, 1][:target_idx]
                    rib_y0_lst = rib_ypos_lst[:, 0][:target_idx]
                    rib_y1_lst = rib_ypos_lst[:, 1][:target_idx]

                    lung_valid_lst = []
                    for x0, x1, y0 in zip(rib_x0_lst, rib_x1_lst, rib_y0_lst):
                        flag_lung_valid = (lung_mid_xpos_combined[1] > x0) & (lung_mid_xpos_combined[0] < x1)
                        lung_valid_lst.append(flag_lung_valid)

                    if (sum(lung_valid_lst) / len(lung_valid_lst)) >= 0.7:
                        target_ribs = rib_y1_lst

                        y_height = abs(target_ribs[0] - target_ribs[-1]) + 1
                        y_normalized = target_ribs / y_height
                        y1_diff = np.diff(y_normalized)

                        mask_y_height_min = []
                        mask_y_height = []
                        mask_iou = []
                        for idx, diff in enumerate(y1_diff):
                            inter_small = inter_ratio_small_lst[idx]
                            if (int(diff * 100) / 100) < 0.1:
                                if (int(inter_small * 100) / 100) > threshold_large_overlap_01[idx]:
                                    mask_y_height_min.append(True)

                            elif (int(diff * 100) / 100) > 0.1 and (int(diff * 100) / 100) <= 0.2:
                                if (int(inter_small * 100) / 100) > threshold_large_overlap_02[idx]:
                                    mask_y_height_min.append(True)

                            mask_y_height.append((int(diff * 100) / 100) <= threshold_y_height[idx])
                            mask_iou.append((int(iou_lst[idx] * 100) / 100) <= threshold_iou_pair[idx])

                        mask_x_pos = []
                        for i, x0_prev in enumerate(rib_x0_lst):
                            if i < (target_idx - 1):
                                x0_post = rib_x0_lst[i + 1]

                                x1_prev, x1_post = rib_x1_lst[i], rib_x1_lst[i + 1]

                                y0_prev, y0_post = (rib_y0_lst[i] / y_height), (rib_y0_lst[i + 1] / y_height)
                                y1_prev, y1_post = (rib_y1_lst[i] / y_height), (rib_y1_lst[i + 1] / y_height)

                                if x1_prev > x1_post:
                                    if (int(abs(y0_prev - y1_post) * 100) / 100) <= threshold_x_pos[i]:
                                        mask_poor = (abs(x1_post - x1_prev) >= abs(x1_post - x0_prev)) \
                                                    | (abs(x0_prev - x0_post) >= abs(x0_prev - x1_post))
                                    else:
                                        mask_poor = False
                                elif x1_post > x1_prev:
                                    if (int(abs(y0_post - y1_prev) * 100) / 100) <= threshold_x_pos[i]:
                                        mask_poor = (abs(x1_prev - x1_post) >= abs(x1_prev - x0_post)) \
                                                    | (abs(x0_post - x0_prev) >= abs(x0_post - x1_prev))
                                    else:
                                        mask_poor = False
                                elif x1_post == x1_prev:
                                    if (int(abs(y1_prev - y1_post) * 100) / 100) <= threshold_x_pos[i]:
                                        mask_poor = (abs(x0_prev - x0_post) >= abs(x0_prev - x1_post)) \
                                                    | (abs(x0_post - x0_prev) >= abs(x0_post - x1_prev))

                                    else:
                                        mask_poor = False
                                mask_x_pos.append(mask_poor)

                        if (False not in mask_y_height) and (True not in mask_y_height_min) and (False not in mask_iou) and (True not in mask_x_pos):
                            meta_inspiration['mid-clavicular_line(x)'] = standard_line
                            meta_inspiration['rib_position'] = expected_rib
                            if expected_rib >= MD_RANGES['Inspiration']:
                                meta_inspiration['label'] = 0
                            else:
                                meta_inspiration['label'] = 1

                            meta_inspiration['viz_label'] = expected_rib
                            meta_inspiration['viz_x_lung_mid_combined'] = x_lung_mid_combined
                            meta_inspiration['viz_lung_y_lst_x_lung_mid'] = lung_y_lst_x_lung_mid

    return meta_inspiration
