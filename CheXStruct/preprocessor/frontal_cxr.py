import cv2
import numpy as np

from utils.constants import TARGET_MASK_LIST, MD_RANGES
from utils.utils import select_max_area_mask

def extract_mask_num(image_data):
    mask_num = 0
    for target_mask in TARGET_MASK_LIST:
        mask = cv2.imread(image_data['cxas'][target_mask])
        if mask.sum() != 0:
            mask_num += 1

    label = 0 if mask_num < MD_RANGES['Frontal_CXR']['mask_num'] else 1

    meta_mask_num = {'mask_count': mask_num,
                     'label': label}
    return meta_mask_num

def extract_abdomial(image_data):
    meta_abdomial = dict()
    mask_heart = cv2.imread(image_data['cxas']['heart'])
    if mask_heart.sum() != 0:
        cxr = cv2.imread(image_data['cxr'], cv2.IMREAD_GRAYSCALE)

        height, width = cxr.shape

        cxr_mid_value = cxr[:, width // 2]
        cxr_mid_value_prev = cxr[:, width // 2 - 1]
        cxr_mid_value_post = cxr[:, width // 2 + 1]

        # black edge
        cxr_mid_value_merge = cxr_mid_value * cxr_mid_value_prev * cxr_mid_value_post
        if 0 in cxr_mid_value_merge:
            try:
                start = cxr_mid_value_merge.nonzero()[0][0]
                end = cxr_mid_value_merge.nonzero()[0][-1]
            except IndexError:
                start = 0
                end = 0
        else:
            start = 0
            end = len(cxr_mid_value_merge)

        mask_heart_max_area = select_max_area_mask(mask_heart)
        y_indices_heart = mask_heart_max_area.sum(axis=-1).nonzero()[0]
        ymin_heart, ymax_heart = y_indices_heart[0], y_indices_heart[-1]

        distance_upper_part = abs(start - ymax_heart)
        distance_lower_part = abs(end - ymax_heart)

        ratio = distance_lower_part / distance_upper_part

        meta_abdomial['ratio'] = ratio
        meta_abdomial['distance_upper'] = distance_upper_part
        meta_abdomial['distance_lower'] = distance_lower_part
        meta_abdomial['y_start'] = start
        meta_abdomial['y_heart_max'] = ymax_heart
        meta_abdomial['y_end'] = end

        label = 0 if ratio >= MD_RANGES['Frontal_CXR']['abdomial_ratio'] else 1
        meta_abdomial['label'] = label
    else:
        meta_abdomial['label'] = -1
    return meta_abdomial

def extract_window(image_data):
    cxr = cv2.imread(image_data['cxr'], cv2.IMREAD_GRAYSCALE)

    sobel_x = cv2.Sobel(cxr, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(cxr, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_mean = np.mean(gradient_magnitude)

    meta_window = {'gradient_mean': gradient_mean}

    label = 0 if gradient_mean >= MD_RANGES['Frontal_CXR']['window'] else 1
    meta_window['label'] = label

    return meta_window

