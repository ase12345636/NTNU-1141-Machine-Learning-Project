import cv2
import numpy as np

from utils.constants import TARGET_MASK_LIST, MD_RANGES
from utils.utils import select_max_area_mask, select_max_width_mask


def check_intersection(mask, point1, point2):
    (x1, y1), (x2, y2) = point1, point2

    if x1 == x2:
        mask_points = np.argwhere(mask > 0)
        return np.any(mask_points[:, 1] == x1)

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    mask_points = np.argwhere(mask > 0)

    for y, x in mask_points:
        if (y < (m * x + b)):
            return True


def split_mask_by_line(mask, point1, point2):
    (x1, y1), (x2, y2) = point1, point2

    mask_top = np.zeros_like(mask, dtype=np.uint8)
    mask_bottom = np.zeros_like(mask, dtype=np.uint8)

    if x1 == x2:
        mask_top[:, :x1] = mask[:, :x1]
        mask_bottom[:, x1:] = mask[:, x1:]
        return 0, mask_top, mask_bottom

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    mask_points = np.argwhere(mask > 0)

    for y, x in mask_points:
        if y > m * x + b:
            mask_bottom[y, x] = mask[y, x]
        else:
            mask_top[y, x] = mask[y, x]

    return m, mask_top, mask_bottom

def extract_enlarged_asc_aorta(image_data, meta_cardiomegaly, meta_trachea):
    meta_enlarged_asc_aorta = dict()

    if ('qc_valid_trachea' in meta_trachea) & ('qc_valid_heart' in meta_cardiomegaly):
        if meta_trachea['qc_valid_trachea'] & meta_cardiomegaly['qc_valid_heart']:
            cxas_files = image_data['cxas']

            mask_trachea = cv2.imread(cxas_files["trachea"], 0)
            mask_carina = cv2.imread(cxas_files["tracheal bifurcation"], 0)
            mask_heart = cv2.imread(cxas_files["heart"], 0)
            mask_ascending_aorta = cv2.imread(cxas_files["ascending aorta"], 0)

            if mask_ascending_aorta.sum() != 0 and mask_heart.sum() != 0 and mask_trachea.sum() != 0 and mask_carina.sum() != 0:
                y_pos_carina_start = mask_carina.sum(axis=-1).nonzero()[0][0]
                mask_trachea_refined = mask_trachea.copy()
                mask_trachea_refined[y_pos_carina_start:] = 0

                img_height, img_width = mask_ascending_aorta.shape
                if mask_trachea_refined.sum() != 0:
                    mask_heart_refined, _ = select_max_width_mask(mask_heart)
                    mask_ascending_aorta_refined = select_max_area_mask(mask_ascending_aorta)

                    # pnt for trachea - bottom
                    mask_trachea_refined = select_max_area_mask(mask_trachea_refined)
                    y_trachea = mask_trachea_refined.sum(axis=-1).nonzero()[0][-1]
                    x_trachea = mask_trachea_refined[y_trachea].nonzero()[0][0]
                    pnt_trachea = [x_trachea, y_trachea]

                    # pnt for heart - right lung side
                    y_indices, x_indices = mask_heart_refined.nonzero()
                    x_min_idx = x_indices.argmin()
                    pnt_heart = [x_indices[x_min_idx], y_indices[x_min_idx]]

                    # pnt for asc aorta - top
                    ymin_asc = mask_ascending_aorta_refined.sum(axis=-1).nonzero()[0][0]
                    ymax_asc = mask_ascending_aorta_refined.sum(axis=-1).nonzero()[0][-1]

                    y_diff_trachea_asc = (ymin_asc - y_trachea) / img_height

                    height_asc = (ymax_asc - ymin_asc) / img_height
                    if check_intersection(mask_ascending_aorta_refined, pnt_heart, pnt_trachea):
                        m, mask_split_top, mask_split_bottom = split_mask_by_line(mask_ascending_aorta_refined,
                                                                                  pnt_heart, pnt_trachea)
                        if m <= 0:
                            ratio = mask_split_top.sum() / mask_ascending_aorta_refined.sum()
                        else:
                            ratio = mask_split_bottom.sum() / mask_ascending_aorta_refined.sum()
                    else:
                        ratio = 0.0

                    qc_valid_asc_aorta = (height_asc >= 0.15) \
                                         & (y_diff_trachea_asc <= 0.075) & (y_diff_trachea_asc >= -0.025)
                    qc_valid_line = ((pnt_trachea[0] - pnt_heart[0]) >= 0)

                    if qc_valid_asc_aorta & qc_valid_line:
                        meta_enlarged_asc_aorta['heart_point'] = pnt_heart
                        meta_enlarged_asc_aorta['trachea_point'] = pnt_trachea
                        meta_enlarged_asc_aorta['ratio'] = ratio
                        if (ratio >= MD_RANGES['Ascending_Aorta_Enlargemnet'][0]) \
                                & (ratio < MD_RANGES['Ascending_Aorta_Enlargemnet'][-1]):
                            meta_enlarged_asc_aorta['label'] = 1
                        elif ratio == 0.0:
                            meta_enlarged_asc_aorta['label'] = 0

                        meta_enlarged_asc_aorta['viz_pnt_heart'] = pnt_heart
                        meta_enlarged_asc_aorta['viz_pnt_trachea'] = pnt_trachea

    return meta_enlarged_asc_aorta












