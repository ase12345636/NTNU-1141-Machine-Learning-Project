import cv2
import numpy as np

from utils.utils import select_max_area_mask, select_max_width_mask

def calculate_median_width(mask):
    img_height, img_width = mask.shape
    row_width_lst, row_width_idx_lst = [], []
    for idx, row in enumerate(mask):
        if row.sum() != 0:
            x_idx_nonzero_per_row = row.nonzero()[0]
            row_width = (x_idx_nonzero_per_row[-1] - x_idx_nonzero_per_row[0])
            row_width_lst.append(row_width)
            row_width_idx_lst.append(idx)

    median_value = np.sort(row_width_lst)[len(row_width_lst) // 2]
    median_index = np.where(row_width_lst == median_value)[0][0]

    y_width = row_width_idx_lst[median_index]
    width_idx = mask[y_width].nonzero()[0]
    xmin_width, xmax_width = width_idx[0], width_idx[-1]

    width_median = median_value / img_width
    width_std = np.array(row_width_lst).std()

    return width_median, width_std, y_width, xmin_width, xmax_width


def extract_trachea(image_data):
    meta_trachea = dict()

    cxas_files = image_data['cxas']

    mask_trachea = cv2.imread(cxas_files["trachea"], 0)
    mask_carina = cv2.imread(cxas_files["tracheal bifurcation"], 0)

    if mask_trachea.sum() != 0 and mask_carina.sum() != 0:
        img_height, img_width = mask_trachea.shape

        y_pos_carina_start = mask_carina.sum(axis=-1).nonzero()[0][0]
        mask_trachea_refined = mask_trachea.copy()
        mask_trachea_refined[y_pos_carina_start:] = 0

        mask_trachea_refined = select_max_area_mask(mask_trachea_refined)
        trachea_height_idx = mask_trachea_refined.sum(axis=-1).nonzero()[0]

        if len(trachea_height_idx):
            # trachea - height
            y_min, y_max = trachea_height_idx[0], trachea_height_idx[-1]
            y_subpart = int((y_max - y_min) * (1 / 4))
            mask_trachea_refined[:(y_min + y_subpart)] = 0

            y_min = y_min + y_subpart
            trachea_height = (y_max - y_min) / img_height

            # trachea - width, std
            trachea_width_median, trachea_width_std, y_width, xmin_width, xmax_width = calculate_median_width(mask_trachea_refined)

            qc_valid_trachea_height = (trachea_height >= 0.1)
            qc_valid_trachea_width_std = (trachea_width_std <= 20)
            qc_valid_trachea_width_median = (trachea_width_median >= 0.03) & (trachea_width_median <= 0.08)

            qc_valid_trachea = qc_valid_trachea_height & qc_valid_trachea_width_std & qc_valid_trachea_width_median
            meta_trachea['qc_valid_trachea'] = qc_valid_trachea
            if qc_valid_trachea:
                meta_trachea['trachea_width'] = (xmax_width - xmin_width)
                meta_trachea['trachea_point_right'] = (xmax_width, y_width)
                meta_trachea['trachea_point_left'] = (xmin_width, y_width)


                meta_trachea['trachea_y_min'] = y_min
                meta_trachea['trachea_y_max'] = y_max
                meta_trachea['trachea_mask_refined'] = mask_trachea_refined

    return meta_trachea











