import cv2
import numpy as np
from collections import defaultdict

from utils.constants import TARGET_MASK_LIST, MD_RANGES, ROUND_VALUE
from utils.utils import select_max_height_mask


def compute_slope(x1, y1, x2, y2):
    if x2 == x1:
        m = 0
    else:
        m = (y2 - y1) / (x2 - x1)
    return m

def refine_mask(target_mask, filter_mask):
    overlap = (target_mask & filter_mask)
    overlap = select_max_height_mask(overlap)
    if overlap.sum():
        ymax = overlap.sum(axis=-1).nonzero()[0][-1]

        mask_ = np.zeros_like(target_mask)
        mask_[:ymax, ] = 1

        refined_mask = select_max_height_mask(target_mask & mask_)
    else:
        refined_mask = np.zeros_like(target_mask)
    return refined_mask


def split_mask_y(mask, ymin_start, ymin_end, num_parts=3):
    h = ymin_end - ymin_start  # Total height to split
    split_height = h // num_parts  # Height of each part
    remainder = h % num_parts  # Handle rounding issues

    parts = []
    start = ymin_start

    for i in range(num_parts):
        # Adjust height for remainder spread
        extra = 1 if i < remainder else 0
        end = start + split_height + extra

        # Copy mask and zero out everything outside the segment
        part = np.zeros_like(mask)
        part[start:end, :] = mask[start:end, :]

        parts.append(part)
        start = end  # Update start for next part

    return parts


def manual_first_derivative(f, h=1):
    """Compute first derivative using finite differences."""
    df = np.zeros_like(f)

    # Forward difference for first point
    df[:, 0] = (f[:, 1] - f[:, 0]) / h

    # Central difference for middle points
    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * h)

    # Backward difference for last point
    df[:, -1] = (f[:, -1] - f[:, -2]) / h

    return df


def manual_second_derivative(f):
    """Compute second derivative using finite differences."""
    ddf = np.zeros_like(f)

    # Forward difference for first point
    ddf[:, 0] = f[:, 2] - 2 * f[:, 1] + f[:, 0]

    # Central difference for middle points
    ddf[:, 1:-1] = f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]

    # Backward difference for last point
    ddf[:, -1] = f[:, -1] - 2 * f[:, -2] + f[:, -3]

    return ddf


def calculate_curvature_manual(points):
    """Calculate curvature manually using finite difference methods."""

    x = points[:, :, 0]  # Extract x-coordinates (N, 6)
    y = points[:, :, 1]  # Extract y-coordinates (N, 6)

    dx = manual_first_derivative(x)
    dy = manual_first_derivative(y)
    ddx = manual_first_derivative(dx)
    ddy = manual_first_derivative(dy)

    # Compute curvature: κ = |x''y' - y''x'| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(ddx * dy - ddy * dx)
    denominator = (dx ** 2 + dy ** 2) ** (3 / 2)

    curvature = numerator / np.where(denominator == 0, np.nan, denominator)

    # Remove NaN values (e.g., division by zero)
    curvature = np.nan_to_num(curvature, nan=0.0)

    return curvature


def extract_desc_aorta(image_data, meta_trachea):
    meta_tortuous = defaultdict()
    meta_enlarged = defaultdict()

    num_subparts = 5
    cxas_files = image_data['cxas']

    mask_descending_aorta = cv2.imread(cxas_files["descending aorta"], 0)
    mask_heart = cv2.imread(cxas_files["heart"], 0)
    if (mask_descending_aorta.sum() != 0) and (mask_heart.sum() != 0):
        img_height, img_width = mask_descending_aorta.shape
        mask_descending_aorta_refined = refine_mask(mask_descending_aorta, mask_heart)
        if mask_descending_aorta_refined.sum() != 0:
            height_idx = mask_descending_aorta_refined.sum(axis=-1).nonzero()[0]
            ymin, ymax = height_idx[0], height_idx[-1]
            height = (ymax - ymin)

            # trimm
            ymin_start = (ymin + int(height * (5 / 100)))
            ymin_end = (ymax - int(height * (5 / 100)))

            height_refined = (ymin_end - ymin_start)
            mask_descending_aorta_refined[:ymin_start] = 0
            mask_descending_aorta_refined[ymin_end:] = 0

            height_ratio = height_refined / img_height

            splited_masks = split_mask_y(mask_descending_aorta_refined, ymin_start, ymin_end, num_subparts)
            mask_sum = [mask.sum() for mask in splited_masks]
            if 0 not in mask_sum:
                def find_top_left_position(mask):
                    y_indices, x_indices = mask.nonzero()
                    pos = (x_indices[0], y_indices[0])
                    return pos

                def find_bottom_left_position(mask):
                    y_indices, x_indices = mask.nonzero()
                    bottom_idx = y_indices.argmax()
                    pos = (x_indices[bottom_idx], y_indices[bottom_idx])
                    return pos

                target_pnts_left = []
                target_pnts_right = []
                for mask in splited_masks:
                    pnt_left = find_top_left_position(mask)
                    pnt_right = (mask[pnt_left[-1]].nonzero()[0][-1], pnt_left[-1])
                    target_pnts_left.append(pnt_left)
                    target_pnts_right.append(pnt_right)

                pnt_left = find_bottom_left_position(splited_masks[-1])
                pnt_right = (splited_masks[-1][pnt_left[-1]].nonzero()[0][-1], pnt_left[-1])
                target_pnts_left.append(pnt_left)
                target_pnts_right.append(pnt_right)

                # tortuous
                target_pnts_right_ = np.array(target_pnts_right)[None, :, :]
                curvatures = calculate_curvature_manual(target_pnts_right_)
                curvature_mean = curvatures.mean()

                qc_valid_height = (height_ratio >= 0.2)

                width_lst = (np.array(target_pnts_right)[:, 0] - np.array(target_pnts_left)[:, 0]) / img_width
                width_std = width_lst.std()
                qc_valid_width_std = (width_std <= 0.01)

                # enlargement
                max_width_idx = width_lst.argmax()
                pnt_right = target_pnts_right[max_width_idx]
                pnt_left = target_pnts_left[max_width_idx]
                desc_aorta_width = (pnt_right[0] - pnt_left[0])

                qc_valid_desc_aorta = qc_valid_height & qc_valid_width_std
                if qc_valid_desc_aorta:
                    for idx, pnt in enumerate(target_pnts_right):
                        meta_tortuous[f'point_{idx + 1}'] = pnt
                        meta_tortuous[f'viz_pnt_r_{idx + 1}'] = pnt
                    meta_tortuous['curvature'] = curvature_mean

                    if (round(curvature_mean, ROUND_VALUE['Descending_Aorta_Tortuous']) >= MD_RANGES['Descending_Aorta_Tortuous']):
                        meta_tortuous['label'] = 1
                    else:
                        meta_tortuous['label'] = 0
                    meta_tortuous['viz_ymin_start'] = ymin_start
                    meta_tortuous['viz_ymin_end'] = ymin_end


                    if ('qc_valid_trachea' in meta_trachea):
                        if meta_trachea['qc_valid_trachea']:
                            trachea_width = meta_trachea['trachea_width']
                            meta_enlarged['trachea_width'] = trachea_width
                            meta_enlarged['trachea_point_right'] = meta_trachea['trachea_point_right']
                            meta_enlarged['trachea_point_left'] = meta_trachea['trachea_point_left']

                            meta_enlarged['desc_aorta_width'] = desc_aorta_width
                            meta_enlarged['desc_aorta_point_right'] = pnt_right
                            meta_enlarged['desc_aorta_point_left'] = pnt_left

                            ratio_enlarged = (desc_aorta_width / trachea_width)
                            meta_enlarged['ratio'] = ratio_enlarged

                            if (round(ratio_enlarged, ROUND_VALUE['Descending_Aorta_Enlargement']) >= MD_RANGES['Descending_Aorta_Enlargement']):
                                meta_enlarged['label'] = 1
                            else:
                                meta_enlarged['label'] = 0

                            meta_enlarged['viz_ymin_start'] = ymin_start
                            meta_enlarged['viz_ymin_end'] = ymin_end

                            meta_enlarged['viz_trachea_y_width'] = meta_trachea['trachea_point_right'][-1]
                            meta_enlarged['viz_trachea_xmin_width'] = meta_trachea['trachea_point_left'][0]
                            meta_enlarged['viz_trachea_xmax_width'] = meta_trachea['trachea_point_right'][0]

                            for idx, (pnt_r, pnt_l) in enumerate(zip(target_pnts_right, target_pnts_left)):
                                meta_enlarged[f'viz_pnt_r_{idx + 1}'] = pnt_r
                                meta_enlarged[f'viz_pnt_{idx + 1}'] = pnt_l

    return meta_tortuous, meta_enlarged











