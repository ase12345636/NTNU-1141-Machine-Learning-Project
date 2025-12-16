import cv2
import numpy as np
from scipy import ndimage

from utils.constants import TARGET_MASK_LIST, MD_RANGES, ROUND_VALUE


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


def find_filtered_zero_groups(row):
    zero_indices = [i for i, val in enumerate(row) if val == 0]
    groups = []
    temp_group = []

    for i in zero_indices:
        if not temp_group or i == temp_group[-1] + 1:
            temp_group.append(i)
        else:
            groups.append(temp_group)
            temp_group = [i]

    if temp_group:
        groups.append(temp_group)

    filtered_groups = [g for g in groups if 0 not in g and (len(row) - 1) not in g]

    return filtered_groups


def euclidean_distance(row1, row2):
    return np.linalg.norm(row1 - row2)

def is_ascending(lst):
    return lst == sorted(lst)

def calculate_angle(A, B, C):
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)

    dot_product = np.dot(BA, BC)

    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)

    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def extract_contour_height(approx_pnts):
    approx_pnts = np.array(approx_pnts)
    y_pnts = approx_pnts[:, -1]
    y_max, y_min = y_pnts.max(), y_pnts.min()

    height = (y_max - y_min)
    return height


def extract_contour_width(approx_pnts):
    approx_pnts = np.array(approx_pnts)
    x_pnts = approx_pnts[:, 0]
    x_max, x_min = x_pnts.max(), x_pnts.min()

    width = (x_max - x_min)
    return width


def cal_angle_std(angles):
    angles = np.array(angles)
    target_angles = angles[np.where(angles != -1)[0]]
    return target_angles.std()

def extract_carina(image_data):
    meta_carina = dict()

    cxas_files = image_data['cxas']

    mask = cv2.imread(cxas_files["tracheal bifurcation"], 0)
    mask = select_max_area_mask(mask)
    if mask.sum() != 0:
        _, binary_mask = cv2.threshold(mask, (mask.max() / 2), mask.max(), cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, closed=False)
        approx = cv2.approxPolyDP(largest_contour, epsilon, closed=False)
        approx_points = np.array(approx).reshape(-1, 2)
        top_y_pnt = approx_points[np.argmin(approx_points[:, -1])]
        x_value = top_y_pnt[0]
        left_rows = approx_points[approx_points[:, 0] <= x_value]
        right_rows = approx_points[approx_points[:, 0] > x_value]

        if len(left_rows) and len(right_rows):
            left_distances = [euclidean_distance(top_y_pnt, row) for row in left_rows]
            right_distances = [euclidean_distance(top_y_pnt, row) for row in right_rows]

            max_left_row = left_rows[np.argmax(left_distances)]
            max_right_row = right_rows[np.argmax(right_distances)]

            max_left_index = np.where((approx_points == max_left_row).all(axis=1))[0][0]
            max_right_index = np.where((approx_points == max_right_row).all(axis=1))[0][0]

            rows_between = approx_points[min(max_left_index, max_right_index) + 1: max(max_left_index, max_right_index)]

            if len(rows_between):
                center_point = rows_between[np.argmin(rows_between[:, -1])].tolist()
                x_center, y_center = center_point
                y_bottom = approx_points[:, -1].max()

                angle_lst = []
                margin_y_lst = []
                target_points_lst = []

                margins_y = [int((y_bottom - y_center) * (val / 100)) for val in [10, 20, 30]]
                for margin_y in margins_y:
                    target_row = mask[(y_center + margin_y)]
                    idx_last_row_lst = find_filtered_zero_groups(target_row)

                    if len(idx_last_row_lst) == 1:
                        idx_last_row = idx_last_row_lst[-1]
                        if (idx_last_row[0] != 0) and (idx_last_row[-1] != len(target_row)):
                            pnt_rl = [idx_last_row[0], y_center + margin_y]
                            pnt_ll = [idx_last_row[-1], y_center + margin_y]

                            target_point = [pnt_rl, center_point, pnt_ll]

                            if is_ascending(np.array(target_point)[:, 0].tolist()):
                                angle = calculate_angle(pnt_rl, center_point, pnt_ll)
                                target_point = [pnt_rl, center_point, pnt_ll]

                                angle_lst.append(angle)
                                margin_y_lst.append(margin_y)
                                target_points_lst.append(target_point)

                if len(angle_lst) == len(margins_y):
                    pnt_c = np.array(target_points_lst)[:, 1][0].tolist()
                    pnt_rl = np.array(target_points_lst)[:, 0].mean(axis=0).round().astype(int).tolist()
                    pnt_ll = np.array(target_points_lst)[:, -1].mean(axis=0).round().astype(int).tolist()

                    refined_tgpnt = [pnt_rl, pnt_c, pnt_ll]
                    refined_angle = calculate_angle(*refined_tgpnt)

                    img_height, img_width = mask.shape

                    x_top, y_top = top_y_pnt

                    contour_height = extract_contour_height(approx_points)
                    contour_width = extract_contour_width(approx_points)
                    contour_xmin = np.array(approx_points)[:, 0].min()

                    position_center_y = (y_center - y_top) / contour_height
                    position_center_x = (x_center - contour_xmin) / contour_width

                    contour_height = contour_height / img_height
                    contour_width = contour_width / img_width

                    x_diff_top_center = abs(x_center - x_top) / img_width
                    y_diff_top_center = abs(y_center - y_top) / img_height

                    angle_std = cal_angle_std(angle_lst) if -1 not in angle_lst else -1


                    qc_valid_x_diff = (x_diff_top_center <= 0.03)
                    qc_valid_y_diff = (y_diff_top_center > 0)

                    qc_valid_contour_width = (contour_width >= 0.175)
                    qc_valid_contour_height = (contour_height >= 0.075)

                    qc_valid_position_center_x = (position_center_x >= 0.25) & (position_center_x < 0.6)
                    qc_valid_position_center_y = (position_center_y >= 0.1) & (position_center_y < 0.65)

                    qc_valid_angle_std = (angle_std >= 0) & (angle_std <= 20)

                    qc_valid = qc_valid_x_diff & qc_valid_y_diff \
                               & qc_valid_contour_width & qc_valid_contour_height \
                               & qc_valid_position_center_x & qc_valid_position_center_y \
                               & qc_valid_angle_std

                    if qc_valid:
                        for idx, pnt in enumerate(refined_tgpnt):
                            meta_carina[f'point_{idx+1}'] = pnt
                        meta_carina['angle'] = refined_angle


                        if (round(refined_angle, ROUND_VALUE['Carina_Angle']) >= MD_RANGES['Carina_Angle'][0]) \
                                & (round(refined_angle, ROUND_VALUE['Carina_Angle']) <= MD_RANGES['Carina_Angle'][-1]):
                            meta_carina['label'] = 0
                        else:
                            meta_carina['label'] = 1

                        meta_carina['viz_refined_tgpnt'] = refined_tgpnt

    return meta_carina











