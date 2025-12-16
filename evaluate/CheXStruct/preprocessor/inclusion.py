import cv2
import numpy as np
from scipy.ndimage import label

from utils.constants import TARGET_MASK_LIST, MD_RANGES, md_range_per_part


def select_max_area_mask(mask):
    labeled, num_components = label(mask)

    component_sizes = [np.sum(labeled == i) for i in range(1, num_components + 1)]

    if len(component_sizes) == 0:
        return mask

    largest_component = np.argmax(component_sizes) + 1

    mask = (labeled == largest_component)

    return mask

def extract_apex(lung, cxr):
    # refine mask
    mask_apex = np.logical_and(lung, cxr)
    mask_apex = select_max_area_mask(mask_apex)

    values = None
    if int(np.sum(mask_apex)) != 0:
        # extract point
        nonzero_indices = np.nonzero(mask_apex)
        min_y = np.min(nonzero_indices[0])

        pixel_line = mask_apex[min_y, :]
        nonzero_indices = np.nonzero(pixel_line)
        mid_x = nonzero_indices[0][len(nonzero_indices[0]) // 2]
        apex = [int(mid_x), int(min_y)]

        # calculate distance and ratio
        line_b4_apex = np.prod((cxr[:apex[1], (apex[0] - 2): (apex[0] + 3)]), axis=1)
        non_black_indices_b4_apex = np.where(line_b4_apex != 0)[0]
        if len(non_black_indices_b4_apex):
            first_non_black_index = non_black_indices_b4_apex[0]
        else:
            first_non_black_index = apex[1]

        line_af_apex = np.prod((cxr[apex[1]:, (apex[0] - 2): (apex[0] + 3)]), axis=1)
        non_black_indices_af_apex = np.where(line_af_apex != 0)[0]
        if len(non_black_indices_af_apex):
            last_non_black_index = non_black_indices_af_apex[-1] + apex[1]
        else:
            last_non_black_index = apex[1]

        total_distance = last_non_black_index - first_non_black_index
        if total_distance != 0:
            apex_distance = apex[1] - first_non_black_index
            ratio = apex_distance / total_distance

            values = {
                'point': apex,
                'ratio': ratio,
                'distance': apex_distance,
                'total_distance': total_distance
            }
    return values

def extract_side(lung, dp, cxr, direction):
    # refine mask
    diff = lung & ~dp
    diff = np.logical_and(diff, cxr)
    mask_side = select_max_area_mask(diff)

    values = None
    if int(np.sum(mask_side)) != 0:
        # extract point
        nonzero_indices = np.nonzero(mask_side)

        if direction == 'right':
            x = np.min(nonzero_indices[1])
        elif direction == 'left':
            x = np.max(nonzero_indices[1])

        pixel_line = mask_side[:, x]
        nonzero_indices = np.nonzero(pixel_line)
        mid_y = nonzero_indices[0][len(nonzero_indices[0]) // 2]  #
        side = [int(x), int(mid_y)]

        # calculate distsance, ratio
        line_b4_side = np.prod((cxr[(side[1] - 2): (side[1] + 3), :side[0]]), axis=0)
        non_black_indices_b4_side = np.where(line_b4_side != 0)[0]
        if len(non_black_indices_b4_side):
            first_non_black_index = non_black_indices_b4_side[0]
        else:
            first_non_black_index = side[0]

        line_af_side = np.prod((cxr[(side[1] - 2): (side[1] + 3), side[0]:]), axis=0)
        non_black_indices_af_side = np.where(line_af_side != 0)[0]
        if len(non_black_indices_af_side):
            last_non_black_index = non_black_indices_af_side[-1] + side[0]
        else:
            last_non_black_index = side[0]

        total_distance = last_non_black_index - first_non_black_index

        if total_distance != 0:
            if direction == 'right':
                side_distance = side[0] - first_non_black_index
            elif direction == 'left':
                side_distance = last_non_black_index - side[0]

            ratio = side_distance / total_distance

            values = {
                'refined_mask': mask_side,
                'point': side,
                'ratio': ratio,
                'distance': side_distance,
                'total_distance': total_distance
            }
    return values

def extract_bottom(lung, dp, cxr, direction):
    # refine mask
    if int(np.sum(dp)) == 0:
        mask_bottom = dp
    else:
        nonzero_indices = np.nonzero(dp)

        max_x = np.max(nonzero_indices[1])
        min_x = np.min(nonzero_indices[1])

        lung[:, max_x + 1:] = 0
        lung[:, :min_x] = 0

        diff = lung & ~dp
        diff = np.logical_and(diff, cxr)

        mask_bottom = select_max_area_mask(diff)

    values = None
    if int(np.sum(mask_bottom)) != 0:
        # extract point
        nonzero_indices = np.nonzero(mask_bottom)

        if direction == 'right':
            idp = int((2 * np.max(nonzero_indices[1]) + 1 * np.min(nonzero_indices[1])) / 3)
            mask_bottom[:, :idp] = 0
        elif direction == 'left':
            idp = int((1 * np.max(nonzero_indices[1]) + 2 * np.min(nonzero_indices[1])) / 3)
            mask_bottom[:, idp + 1:] = 0

        nonzero_indices = np.nonzero(mask_bottom)
        max_y = np.max(nonzero_indices[0])
        pixel_line = mask_bottom[max_y, :]
        nonzero_indices = np.nonzero(pixel_line)

        if direction == 'right':
            max_x = np.max(nonzero_indices[0])
            bottom = [int(max_x), int(max_y)]
        elif direction == 'left':
            min_x = np.min(nonzero_indices[0])
            bottom = [int(min_x), int(max_y)]

        # calculate distsance, ratio
        line_b4_bottom = np.prod((cxr[:bottom[1], (bottom[0] - 2): (bottom[0] + 3)]), axis=1)
        non_black_indices_b4_bottom = np.where(line_b4_bottom != 0)[0]
        if len(non_black_indices_b4_bottom):
            first_non_black_index = non_black_indices_b4_bottom[0]
        else:
            first_non_black_index = bottom[1]

        line_af_bottom = np.prod((cxr[bottom[1]:, (bottom[0] - 2): (bottom[0] + 3)]), axis=1)
        non_black_indices_af_bottom = np.where(line_af_bottom != 0)[0]
        if len(non_black_indices_af_bottom):
            last_non_black_index = non_black_indices_af_bottom[-1] + bottom[1]
        else:
            last_non_black_index = bottom[1]

        total_distance = last_non_black_index - first_non_black_index
        if total_distance != 0:
            bottom_distance = last_non_black_index - bottom[1]
            ratio = bottom_distance / total_distance

            values = {
                'point': bottom,
                'ratio': ratio,
                'distance': bottom_distance,
                'total_distance': total_distance,
                'nonblack_idx': (first_non_black_index, last_non_black_index),
            }

    return values


def compute_cpa(side, bottom, cpa_range):
    nonblack_idx = bottom['nonblack_idx']
    ratio = bottom['ratio']
    total_distance = bottom['total_distance']
    x = side['point'][0]
    y = nonblack_idx[-1] - ((ratio - cpa_range) * total_distance)
    return [x, y]

def return_label_per_position(position, values_r, values_l):
    ratio_r, ratio_l = values_r['ratio'], values_l['ratio']

    md_range_inclusion = MD_RANGES['Inclusion'][position]['inclusion']
    md_range_exclusion = MD_RANGES['Inclusion'][position]['exclusion']

    mask_inclusion_r = (ratio_r >= md_range_inclusion[0]) & (ratio_r < md_range_inclusion[-1])
    mask_inclusion_l = (ratio_l >= md_range_inclusion[0]) & (ratio_l < md_range_inclusion[-1])

    mask_exclusion_r = (ratio_r >= md_range_exclusion[0]) & (ratio_r < md_range_exclusion[-1])
    mask_exclusion_l = (ratio_l >= md_range_exclusion[0]) & (ratio_l < md_range_exclusion[-1])

    if mask_inclusion_r & mask_inclusion_l:
        label = [1, 1]
    elif mask_exclusion_r & mask_exclusion_l:
        label = [0, 0]
    elif mask_inclusion_r & mask_exclusion_l:
        label = [1, 0]
    elif mask_exclusion_r & mask_inclusion_l:
        label = [0, 1]
    else:
        label = None
    return label

def quality_y_diff_position(position, values_r, values_l, cxr):
    img_height, img_width = cxr.shape

    # y diff
    point_r, point_l = values_r['point'], values_l['point']
    y_diff = np.abs(point_r[-1] - point_l[-1]) / img_height

    # y position
    ratio_r, ratio_l = values_r['ratio'], values_l['ratio']
    y_position = (ratio_r + ratio_l) / 2


    qc_y_diff = (y_diff >= md_range_per_part[position]['y_diff'][0]) \
                & (y_diff < md_range_per_part[position]['y_diff'][-1])

    qc_y_pos = (y_position >= md_range_per_part[position]['y_position'][0]) \
               & (y_position < md_range_per_part[position]['y_position'][-1])

    qc_valid = qc_y_diff & qc_y_pos

    return qc_valid

def extract_rb_bottom_point(lung_mask, bottom_point):
    x, y = bottom_point
    blue_point = bottom_point

    labeled_mask, num_features = label(lung_mask)
    target_label = labeled_mask[y, x]

    new_lung_mask = (labeled_mask == target_label)

    lung_mask_line = new_lung_mask[:, x]

    indices = np.where(lung_mask_line)[0]

    max_y = int(indices[-1])

    red_point = [x, max_y]

    return blue_point, red_point

def quality_dp_bottom_point_diff(lung_r, lung_l, values_r, values_l):
    img_height, img_width = lung_r.shape
    point_r, point_l = values_r['point'], values_l['point']

    blue_pnt_r, red_pnt_r = extract_rb_bottom_point(lung_r, point_r)
    blue_pnt_l, red_pnt_l = extract_rb_bottom_point(lung_l, point_l)

    right_dp_rb_diff = (red_pnt_r[1] - blue_pnt_r[1]) / img_height

    left_dp_rb_diff = (red_pnt_l[1] - blue_pnt_l[1]) / img_height

    qc_right_dp_rb_diff = (right_dp_rb_diff >= md_range_per_part['bottom']['dp_rb_diff'][0]) \
                          & (right_dp_rb_diff < md_range_per_part['bottom']['dp_rb_diff'][-1])

    qc_left_dp_rb_diff = (left_dp_rb_diff >= md_range_per_part['bottom']['dp_rb_diff'][0]) \
                         & (left_dp_rb_diff < md_range_per_part['bottom']['dp_rb_diff'][-1])

    qc_dp_rb_diff = (qc_right_dp_rb_diff & qc_left_dp_rb_diff)

    return qc_dp_rb_diff

def quality_side_width_dff(values_r, values_l):
    point_r, point_l = values_r['point'], values_l['point']
    lung_r_refined, lung_l_refined = values_r['refined_mask'], values_l['refined_mask']

    rl_max_x = np.max(np.nonzero(lung_r_refined)[1])
    ll_min_x = np.min(np.nonzero(lung_l_refined)[1])

    right_line_start = point_r
    right_line_end = [rl_max_x, point_r[-1]]

    left_line_start = point_l
    left_line_end = [ll_min_x, point_l[-1]]

    width_lung_right = (right_line_end[0] - right_line_start[0])
    width_lung_left = (left_line_start[0] - left_line_end[0])

    width_ratio_r_l = (width_lung_right / width_lung_left)
    mask_width_ratio_r_l = (width_lung_left > width_lung_right)

    width_ratio_l_r = (width_lung_left / width_lung_right)
    mask_width_ratio_l_r = (width_lung_right >= width_lung_left)

    width_ratio = (mask_width_ratio_r_l * width_ratio_r_l) + (mask_width_ratio_l_r * width_ratio_l_r)

    label = return_label_per_position('SIDE', values_r, values_l)
    if label == [1, 1]:
        qc_valid_side = (width_ratio >= md_range_per_part['side']['both_inclusion'])
    elif label == [0, 0]:
        qc_valid_side = (width_ratio >= md_range_per_part['side']['both_exclusion'])
    elif label == [1, 0]:
        qc_valid_side = (width_ratio_r_l >= md_range_per_part['side']['r_in_l_ex'])
    elif label == [0, 1]:
        qc_valid_side = (width_ratio_l_r >= md_range_per_part['side']['r_ex_l_in'])
    else:
        qc_valid_side = None

    values_r['right_lung_inner_point'] = right_line_end
    values_l['left_lung_inner_point'] = left_line_end

    return qc_valid_side, label


def extract_inclusion(image_data):
    meta_inclusion = dict()

    cxas_files = image_data['cxas']

    cxr = cv2.imread(image_data['cxr'], cv2.IMREAD_GRAYSCALE)
    lung_r = cv2.imread(cxas_files['right lung'], cv2.IMREAD_GRAYSCALE)
    lung_l = cv2.imread(cxas_files['left lung'], cv2.IMREAD_GRAYSCALE)
    dp_r = cv2.imread(cxas_files['right hemidiaphragm'], cv2.IMREAD_GRAYSCALE)
    dp_l = cv2.imread(cxas_files['left hemidiaphragm'], cv2.IMREAD_GRAYSCALE)

    # meta
    apex_r = extract_apex(lung_r, cxr)
    apex_l = extract_apex(lung_l, cxr)

    side_r = extract_side(lung_r, dp_r, cxr, 'right')
    side_l = extract_side(lung_l, dp_l, cxr, 'left')

    bottom_r = extract_bottom(lung_r, dp_r, cxr, 'right')
    bottom_l = extract_bottom(lung_l, dp_l, cxr, 'left')

    if all(x is not None for x in [apex_r, apex_l, side_r, side_l, bottom_r, bottom_l]):
        # quality control
        qc_valid_apex = quality_y_diff_position('apex', apex_r, apex_l, cxr)
        label_apex = return_label_per_position('APEX', apex_r, apex_l)

        qc_valid_bottom_y = quality_y_diff_position('bottom', bottom_r, bottom_l, cxr)
        qc_valid_bottom_dp = quality_dp_bottom_point_diff(lung_r, lung_l, bottom_r, bottom_l)
        qc_valid_bottom = (qc_valid_bottom_y & qc_valid_bottom_dp)
        label_bottom = return_label_per_position('BOTTOM', bottom_r, bottom_l)

        qc_valid_side, label_side = quality_side_width_dff(side_r, side_l)

        if qc_valid_side:
            # for cardiomegaly, mw
            meta_inclusion['label_side_right_lung'], meta_inclusion['label_side_left_lung'] = label_side
            meta_inclusion['right_lung_inner_point'] = side_r['right_lung_inner_point']
            meta_inclusion['left_lung_inner_point'] = side_l['left_lung_inner_point']

            if qc_valid_apex & qc_valid_bottom:
                if all(x is not None for x in [label_apex, label_side, label_bottom]):
                    meta_inclusion['label_apex_right_lung'], meta_inclusion['label_apex_left_lung'] = label_apex
                    meta_inclusion['label_bottom_right_lung'], meta_inclusion['label_bottom_left_lung'] = label_bottom

                    for info in ['point', 'ratio']:
                        meta_inclusion[f'{info}_apex_right_lung'] = apex_r[info]
                        meta_inclusion[f'{info}_apex_left_lung'] = apex_l[info]

                        meta_inclusion[f'{info}_side_right_lung'] = side_r[info]
                        meta_inclusion[f'{info}_side_left_lung'] = side_l[info]

                        meta_inclusion[f'{info}_bottom_right_lung'] = bottom_r[info]
                        meta_inclusion[f'{info}_bottom_left_lung'] = bottom_l[info]

                    if all(x == [1, 1] for x in [label_apex, label_side, label_bottom]):
                        meta_inclusion['label'] = 1
                    else:
                        meta_inclusion['label'] = 0

                    # approximate cpa point
                    cpa_range = (MD_RANGES['Inclusion']['BOTTOM']['inclusion'][0] - MD_RANGES['Inclusion']['BOTTOM']['exclusion'][-1])

                    cpa_right = compute_cpa(side_r, bottom_r, cpa_range)
                    cpa_left = compute_cpa(side_l, bottom_l, cpa_range)


                    meta_inclusion['point_cpa_right_lung'] = cpa_right
                    meta_inclusion['point_cpa_left_lung'] = cpa_left

                    meta_inclusion['viz_apex_right'] = apex_r['point']
                    meta_inclusion['viz_apex_left'] = apex_l['point']
                    meta_inclusion['viz_side_right'] = side_r['point']
                    meta_inclusion['viz_side_left'] = side_l['point']
                    meta_inclusion['viz_cpa_right'] = cpa_right
                    meta_inclusion['viz_cpa_left'] = cpa_left


    return meta_inclusion











