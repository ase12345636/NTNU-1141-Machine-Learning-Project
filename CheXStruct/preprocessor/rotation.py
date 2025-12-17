import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import distance
from collections import defaultdict
from skimage.morphology import skeletonize

from utils.constants import TARGET_MASK_LIST, MD_RANGES, ROUND_VALUE
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


def is_ascending(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

def find_target_midlines(right_end_pnt, left_end_pnt, midline_xy_coords):
    clavicle_y_min = right_end_pnt[-1] if right_end_pnt[-1] <= left_end_pnt[-1] else left_end_pnt[-1]
    clavicle_y_max = right_end_pnt[-1] if right_end_pnt[-1] > left_end_pnt[-1] else left_end_pnt[-1]

    midline_ys = midline_xy_coords[:, -1]

    diff_y_min = abs(midline_ys - clavicle_y_min)
    diff_y_max = abs(midline_ys - clavicle_y_max)

    idx_closest_ymin = diff_y_min.argmin()
    idx_closest_ymax = diff_y_max.argmin()

    if idx_closest_ymin == idx_closest_ymax:
        if idx_closest_ymin == 0:
            target_indices = [idx_closest_ymin, idx_closest_ymin + 1, idx_closest_ymin + 2]
        elif idx_closest_ymin == len(midline_ys) - 1:
            target_indices = [idx_closest_ymin - 2, idx_closest_ymin - 1, idx_closest_ymin]
        else:
            target_indices = [idx_closest_ymin - 1, idx_closest_ymin, idx_closest_ymin + 1]
    else:
        if abs(idx_closest_ymin - idx_closest_ymax) == 1:
            if idx_closest_ymin == 0:
                target_indices = [idx_closest_ymin, idx_closest_ymin + 1, idx_closest_ymin + 2]
            elif idx_closest_ymax == len(midline_ys) - 1:
                target_indices = [idx_closest_ymax - 2, idx_closest_ymax - 1, idx_closest_ymax]
            else:
                target_indices = [idx_closest_ymin - 1, idx_closest_ymin, idx_closest_ymax,
                                  idx_closest_ymax + 1]
        else:
            target_indices = range(idx_closest_ymin, idx_closest_ymax + 1)

    target_coords = midline_xy_coords[target_indices]

    if (float('inf') in target_coords):
        return target_indices, -1
    elif not is_ascending(midline_xy_coords[:, -1]):
        return target_indices, -1
    else:
        return target_indices, target_coords.tolist()

def least_squares_fit_vertical(points):
    points = np.array(points)
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

def line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Slope
    b = y1 - m * x1  # Intercept
    return m, b


def residue(coords):
    residues = []
    if coords[0][0] == coords[-1][0]:
        for p in coords[1:-1]:
            residue = abs(coords[0][0] - p[0])
            residues.append(residue)
    else:
        m, b = line_equation(coords[0], coords[-1])
        for p in coords[1:-1]:
            x, y = p
            residue = abs(m * x - y + b) / np.sqrt(m ** 2 + 1)
            residues.append(residue)
    residue_max = np.array(residues).max()
    return residue_max

def x_std(coords):
    coords = np.array(coords)
    x_coords = coords[:, 0]
    return x_coords.std()


def extract_meta_clavicle(image_data):
    meta_clavicle = dict()
    for side in ['right', 'left']:
        cxas_files = image_data['cxas']
        mask_lung = cv2.imread(cxas_files[f"{side} lung"], 0)
        mask_clavicle = cv2.imread(cxas_files[f"clavicle {side}"], 0)

        if mask_clavicle.sum() != 0 and mask_lung.sum() != 0:
            mask_lung_refined = select_max_area_mask(mask_lung)
            longest_mask, _ = select_max_width_mask(mask_clavicle)
            ys, xs = longest_mask.nonzero()
            idx_xmin, idx_xmax = xs.argmin(), xs.argmax()

            img_height, img_width = longest_mask.shape
            meta_clavicle['image_height'] = img_height
            meta_clavicle['image_width'] = img_width

            if side == 'right':
                x1, y1 = xs[idx_xmin], ys[idx_xmin]
                x2, y2 = xs[idx_xmax], ys[idx_xmax]

            elif side == 'left':
                x2, y2 = xs[idx_xmin], ys[idx_xmin]
                x1, y1 = xs[idx_xmax], ys[idx_xmax]

            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                initial_intercept = y1 - slope * x1
            else:
                slope = None  # Vertical line
                initial_intercept = x1

            if slope is not None:
                height, width = longest_mask.shape
                max_intercept = height - 1  # Limit to the height of the mask
                valid_intercept = []  # Store valid intercept

                # Function to check for intersections
                def has_intersection(mask, slope, intercept, height):
                    for x in range(width):
                        y = int(slope * x + intercept)
                        if 0 <= y < height and mask[y, x] > 0:
                            return True
                    return False

                # Start from initial intercept and move upwards
                intercept = initial_intercept
                while intercept <= max_intercept and has_intersection(longest_mask, slope, intercept, height):
                    valid_intercept.append(intercept)
                    intercept += 1

                # Start from initial intercept and move downwards
                intercept = initial_intercept - 1
                while intercept >= 0 and has_intersection(longest_mask, slope, intercept, height):
                    valid_intercept.append(intercept)
                    intercept -= 1

                if valid_intercept:
                    intercept_max = max(valid_intercept)
                    intercept_min = min(valid_intercept)

                    filled_mask = np.zeros_like(mask_clavicle, dtype=np.uint8)
                    for x in range(width):
                        y_min = int(slope * x + intercept_min)
                        y_max = int(slope * x + intercept_max)

                        if 0 <= y_min < height and 0 <= y_max < height:
                            filled_mask[min(y_min, y_max):max(y_min, y_max) + 1, x] = 1

                    intersection_mask = np.bitwise_and(mask_clavicle, filled_mask)

                    y_indices, x_indices = intersection_mask.nonzero()
                    if len(x_indices) != 0:
                        if side == 'right':
                            idx = x_indices.argmax()
                        elif side == 'left':
                            idx = x_indices.argmin()
                        end_pnt = (x_indices[idx], y_indices[idx])

                        y_maxwidth = mask_lung_refined.sum(axis=-1).argmax()
                        xs_lung_max_width = mask_lung_refined[y_maxwidth].nonzero()[0]
                        if len(xs_lung_max_width) != 0:
                            if side == 'right':
                                lung_inner_pnt_max = (xs_lung_max_width[-1], y_maxwidth)
                                lung_outter_pnt_max = (xs_lung_max_width[0], y_maxwidth)
                                quality_x_pos_max = (end_pnt[0] - lung_outter_pnt_max[0]) / (
                                        lung_inner_pnt_max[0] - lung_outter_pnt_max[0])
                            elif side == 'left':
                                lung_inner_pnt_max = (xs_lung_max_width[0], y_maxwidth)
                                lung_outter_pnt_max = (xs_lung_max_width[-1], y_maxwidth)
                                quality_x_pos_max = (lung_outter_pnt_max[0] - end_pnt[0]) / (
                                        lung_outter_pnt_max[0] - lung_inner_pnt_max[0])
                        else:
                            quality_x_pos_max = -1

                        meta_clavicle[f'{side}_quality_x_pos_max'] = quality_x_pos_max
                        meta_clavicle[f'{side}_end_pnt_xy'] = end_pnt
                        meta_clavicle[f'{side}_slope'] = slope
                        meta_clavicle[f'{side}_intercept_min'] = intercept_min
                        meta_clavicle[f'{side}_intercept_max'] = intercept_max
                else:
                    print(image_data['cxr'])
    return meta_clavicle

def extract_meta_midline(image_data):
    meta_midline = dict()

    total_rib_masks = defaultdict(list)
    vertebrae_lst = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']

    cxas_files = image_data['cxas']
    for idx, vertebrae in enumerate(vertebrae_lst):
        meta_midline[vertebrae] = -1
        cxas_mask = cv2.imread(cxas_files[f"vertebrae {vertebrae}"], 0)

        if cxas_mask.sum() != 0:
            cxas_mask = (cxas_mask / 255).astype(int)
            cxas_mask = select_max_area_mask(cxas_mask)
            total_rib_masks['cxas'].append(cxas_mask)

            # Step 1: Skeletonize the binary mask
            skeleton = skeletonize(cxas_mask)

            # Step 2: Get the coordinates of skeleton pixels
            skeleton_coords = np.column_stack(np.where(skeleton > 0))

            # Step 3: Calculate the approximate centroid
            approx_centroid = skeleton_coords.mean(axis=0)

            # Step 4: Find the skeleton pixel closest to the approximate centroid
            closest_pixel_index = distance.cdist([approx_centroid], skeleton_coords).argmin()
            centroid_on_skeleton = skeleton_coords[closest_pixel_index]

            # Step 5: Find convex and concave points
            ret, thresh = cv2.threshold(cxas_mask.astype(np.uint8), 0.5, 1, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                # define left and right wing
                convex_points = set()
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])

                    convex_points.update([start])
                    convex_points.update([end])

                distances = distance.cdist([centroid_on_skeleton], list(convex_points)).flatten()
                longest_index = distances.argmax()
                longest_convexpoint = list(convex_points)[longest_index]

                x_diffs = [abs(point[0] - longest_convexpoint[0]) for point in convex_points]
                largest_x_diff_convexpoint = list(convex_points)[np.array(x_diffs).argmax()]

                # # # left wing ------- right wing
                if largest_x_diff_convexpoint[0] < longest_convexpoint[0]:
                    left_wing = largest_x_diff_convexpoint
                    right_wing = longest_convexpoint
                else:
                    left_wing = longest_convexpoint
                    right_wing = largest_x_diff_convexpoint

                # define candidate points that exist btw left and right wing
                candidate_points = []
                pnts_per_line = {}
                far_points = []
                b4_left_wing = 0
                af_right_wing = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    line = abs(start[0] - end[0])
                    far_points.append(far)

                    # counter-clockwise
                    if b4_left_wing == 0:
                        candidate_points.append(far)
                        pnts_per_line[line] = [start, end, far]
                    if (start == left_wing) or (end == left_wing):
                        b4_left_wing += 1
                        if start == left_wing:
                            if far in candidate_points:
                                candidate_points.remove(far)
                            if line in pnts_per_line:
                                del pnts_per_line[line]

                    if (start == right_wing) or (end == right_wing):
                        af_right_wing += 1
                    if af_right_wing != 0:
                        candidate_points.append(far)
                        pnts_per_line[line] = [start, end, far]
                        if end == right_wing:
                            if far in candidate_points:
                                candidate_points.remove(far)
                            if line in pnts_per_line:
                                del pnts_per_line[line]

                if len(pnts_per_line) != 0:
                    line_rl = abs(left_wing[0] - right_wing[0])
                    line_max = max(pnts_per_line.keys())
                    ratio = round((line_max / line_rl), 2)
                    if ratio >= 0.45 or ratio <= 0.55:  # concave
                        start, end, far = pnts_per_line[line_max]
                        target_pnt = far
                        meta_midline[vertebrae] = target_pnt

                    else:  # convex
                        if len(pnts_per_line) != 1:
                            sorted_keys = sorted(pnts_per_line.keys(), reverse=True)
                            first_max_key = sorted_keys[0]  # 1st key
                            second_max_key = sorted_keys[1]  # 2nd key

                            start_1st, end_1st, _ = pnts_per_line[first_max_key]
                            start_2nd, end_2nd, _ = pnts_per_line[second_max_key]

                            dist_s1_e2 = abs(start_1st[0] - end_2nd[0])
                            dist_s2_e1 = abs(start_2nd[0] - end_1st[0])

                            if dist_s1_e2 > dist_s2_e1:
                                target_pnt = [(start_2nd[0] + end_1st[0]) // 2, (start_2nd[1] + end_1st[1]) // 2]
                            else:
                                target_pnt = [(start_1st[0] + end_2nd[0]) // 2, (start_1st[1] + end_2nd[1]) // 2]
                            meta_midline[vertebrae] = target_pnt

    # vertebrae
    xy_coords = []
    for vertebrae, pnt in meta_midline.items():
        if pnt == -1:
            xy_coords.append([float('inf'), float('inf')])
        else:
            xy_coords.append(pnt)
    xy_coords = np.array(xy_coords)
    meta_midline['xy_coords'] = xy_coords

    return meta_midline

def extract_rotation(image_data):
    meta_rotation = dict()

    # clavicle, midline
    meta_clavicle = extract_meta_clavicle(image_data)
    meta_midline = extract_meta_midline(image_data)

    tg_keys_clavicle = ['image_height', 'image_width',
                        'right_quality_x_pos_max', 'left_quality_x_pos_max',
                        'right_end_pnt_xy', 'left_end_pnt_xy']
    if all(k in meta_clavicle for k in tg_keys_clavicle):
        image_height = meta_clavicle['image_height']
        image_width = meta_clavicle['image_width']

        clavicle_right_x_pos_max = meta_clavicle['right_quality_x_pos_max']
        clavicle_left_x_pos_max = meta_clavicle['left_quality_x_pos_max']

        clavicle_y_right = meta_clavicle['right_end_pnt_xy'][-1]
        clavicle_y_left = meta_clavicle['left_end_pnt_xy'][-1]
        clavicle_y_diff = abs(clavicle_y_right - clavicle_y_left) / image_height

        clavicle_y_mean = (clavicle_y_right + clavicle_y_left) / 2
        clavicle_y_pos = (clavicle_y_mean / image_height)

        clavicle_x_right = meta_clavicle['right_end_pnt_xy'][0]
        clavicle_x_left = meta_clavicle['left_end_pnt_xy'][0]
        clavicle_x_diff = (clavicle_x_left - clavicle_x_right) / image_width

        md_range_clavicle = {'x_position': [0.8, 2.0], 'x_diff': [-0.1, 0.3],
                             'y_position': [0.05, 0.5], 'y_diff': [0, 0.05]}

        qc_valid_clavicle = (clavicle_right_x_pos_max >= md_range_clavicle['x_position'][0]) \
                            & (clavicle_right_x_pos_max < md_range_clavicle['x_position'][-1]) \
                            & (clavicle_left_x_pos_max >= md_range_clavicle['x_position'][0]) \
                            & (clavicle_left_x_pos_max < md_range_clavicle['x_position'][-1]) \
                            & (clavicle_x_diff >= md_range_clavicle['x_diff'][0]) \
                            & (clavicle_x_diff < md_range_clavicle['x_diff'][-1]) \
                            & (clavicle_y_diff >= md_range_clavicle['y_diff'][0]) \
                            & (clavicle_y_diff < md_range_clavicle['y_diff'][-1]) \
                            & (clavicle_y_pos >= md_range_clavicle['y_position'][0]) \
                            & (clavicle_y_pos < md_range_clavicle['y_position'][-1])

        if qc_valid_clavicle:
            # clavicle
            right_end_pnt = meta_clavicle['right_end_pnt_xy']
            left_end_pnt = meta_clavicle['left_end_pnt_xy']

            # clavicle2midline
            xy_coords = meta_midline['xy_coords']
            target_indices, target_coords = find_target_midlines(right_end_pnt, left_end_pnt, xy_coords)

            if target_coords != -1:
                m, b = least_squares_fit_vertical(target_coords)
                right_midline_x = find_x_vertical(right_end_pnt[-1], m, b)
                left_midline_x = find_x_vertical(left_end_pnt[-1], m, b)

                right_midline_xy = [right_midline_x, right_end_pnt[-1]]
                left_midline_xy = [left_midline_x, left_end_pnt[-1]]

                distance_right_clavicle = right_midline_x - right_end_pnt[0]
                distance_left_clavicle = left_end_pnt[0] - left_midline_x

                ratio = (distance_right_clavicle / distance_left_clavicle) \
                    if distance_left_clavicle >= distance_right_clavicle \
                    else (distance_left_clavicle / distance_right_clavicle)

                # qc - midline
                md_range_midline = {
                    'threshold_vertebrae_residue': 20,
                    'threshold_vertebrae_std': 30
                }

                midline_residue = residue(target_coords)
                midline_x_std = x_std(target_coords)

                qc_valid_midline = (midline_residue < md_range_midline['threshold_vertebrae_residue']) \
                                   & (midline_x_std < md_range_midline['threshold_vertebrae_std'])

                qc_valid_dist_both_pos = (distance_right_clavicle >= 0) & (distance_left_clavicle >= 0)

                qc_valid = qc_valid_clavicle & qc_valid_midline & qc_valid_dist_both_pos

                if qc_valid:
                    meta_rotation['distance_right'] = distance_right_clavicle
                    meta_rotation['distance_left'] = distance_left_clavicle

                    meta_rotation['medial_end_right_clavicle'] = right_end_pnt
                    meta_rotation['medial_end_left_clavicle'] = left_end_pnt

                    meta_rotation['midline_points'] = target_coords

                    meta_rotation['midline_point_right_clavicle'] = right_midline_xy
                    meta_rotation['midline_point_left_clavicle'] = left_midline_xy

                    meta_rotation['ratio'] = ratio
                    if round(ratio, ROUND_VALUE['Rotation']) >= MD_RANGES['Rotation']:
                        meta_rotation['label'] = 0
                    else:
                        meta_rotation['label'] = 1


                    for side in ['right', 'left']:
                        meta_rotation[f'viz_{side}_slope'] = meta_clavicle[f'{side}_slope']
                        meta_rotation[f'viz_{side}_intercept_min'] = meta_clavicle[f'{side}_intercept_min']
                        meta_rotation[f'viz_{side}_intercept_max'] = meta_clavicle[f'{side}_intercept_max']
                        meta_rotation[f'viz_{side}_end_pnt'] = meta_clavicle[f'{side}_end_pnt_xy']

                    meta_rotation['viz_target_coords'] = target_coords
                    meta_rotation['viz_m'] = m
                    meta_rotation['viz_b'] = b

    return meta_rotation, meta_midline
