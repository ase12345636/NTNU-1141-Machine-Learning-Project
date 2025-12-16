import cv2
import numpy as np

from utils.constants import TARGET_MASK_LIST, MD_RANGES, ROUND_VALUE
from utils.utils import select_max_area_mask, select_max_width_mask


def calculate_width_n_height(scapular_mask, lung_mask):
    x_indices = scapular_mask.sum(axis=0).nonzero()[0]
    xmin_scapular = x_indices[0]
    xmax_scapular = x_indices[-1]
    scapular_width = abs(xmax_scapular - xmin_scapular)

    y_indices = scapular_mask.sum(axis=-1).nonzero()[0]
    ymin_scapular = y_indices[0]
    ymax_scapular = y_indices[-1]
    scapular_height = abs(ymax_scapular - ymin_scapular)

    lung_mask_target = lung_mask[ymin_scapular: ymax_scapular + 1]
    lung_area = lung_mask_target.sum()
    if lung_area != 0:
        lung_x_indices = lung_mask_target.sum(axis=0).nonzero()[0]
        xmin_lung = lung_x_indices[0]
        xmax_lung = lung_x_indices[-1]
        lung_width = abs(xmax_lung - xmin_lung)

        lung_y_indices = lung_mask_target.sum(axis=-1).nonzero()[0]
        ymin_lung = lung_y_indices[0]
        ymax_lung = lung_y_indices[-1]
        lung_height = abs(ymax_lung - ymin_lung)

        overlap_region = np.logical_and(lung_mask, scapular_mask)

        values = {'scapular_height': scapular_height,
                'scapular_width': scapular_width,
                'lung_height': lung_height,
                'lung_width': lung_width,
                'overlap_region': overlap_region.sum(),
               'scapular_region': scapular_mask.sum(),
               'lung_region': lung_mask.sum()}
    else:
        values = None
    return values

def extract_projection(image_data):
    meta_projection = dict()

    cxas_files = image_data['cxas']


    mask_r_scapular = cv2.imread(cxas_files["scapula right"], 0) // 255
    mask_l_scapular = cv2.imread(cxas_files["scapula left"], 0) // 255
    mask_r_lung = cv2.imread(cxas_files["right lung"], 0) // 255
    mask_l_lung = cv2.imread(cxas_files["left lung"], 0) // 255

    if mask_r_scapular.sum() != 0 and mask_l_scapular.sum() != 0 \
            and mask_r_lung.sum() != 0 and mask_l_lung.sum() != 0:
        img_height, img_width = mask_r_scapular.shape

        mask_r_scapular_refined = select_max_area_mask(mask_r_scapular)
        mask_l_scapular_refined = select_max_area_mask(mask_l_scapular)

        values_r = calculate_width_n_height(mask_r_scapular_refined, mask_r_lung)
        values_l = calculate_width_n_height(mask_l_scapular_refined, mask_l_lung)

        if isinstance(values_r, dict) and isinstance(values_l, dict):
            # Quality Control
            md_range_per_mask = {'scapular': {'width': [0.2, 0.5], 'height': [0.3, 0.5], 'limit': 0.45},
                                 'lung': {'width': 0.2, 'height': 0.2}}

            # Scapular
            height_r_scapular = (values_r['scapular_height'] / img_height)
            height_l_scapular = (values_l['scapular_height'] / img_height)

            qc_height_r_scapular = (height_r_scapular >= md_range_per_mask['scapular']['height'][0]) \
                                     & (height_r_scapular < md_range_per_mask['scapular']['height'][-1])

            qc_height_l_scapular = (height_l_scapular >= md_range_per_mask['scapular']['height'][0]) \
                                     & (height_l_scapular < md_range_per_mask['scapular']['height'][-1])

            width_r_scapular = (values_r['scapular_width'] / img_width)
            width_l_scapular = (values_l['scapular_width'] / img_width)

            qc_width_r_scapular = (width_r_scapular >= md_range_per_mask['scapular']['width'][0]) \
                                    & (width_r_scapular < md_range_per_mask['scapular']['width'][-1])

            qc_width_l_scapular = (width_l_scapular >= md_range_per_mask['scapular']['width'][0]) \
                                    & (width_l_scapular < md_range_per_mask['scapular']['width'][-1])

            qc_r_scapular_over = (width_r_scapular >= md_range_per_mask['scapular']['limit']) \
                                   & (height_r_scapular >= md_range_per_mask['scapular']['limit'])

            qc_l_scapular_over = (width_l_scapular >= md_range_per_mask['scapular']['limit']) \
                                   & (height_l_scapular >= md_range_per_mask['scapular']['limit'])


            qc_valid_r_scapular = qc_height_r_scapular & qc_width_r_scapular
            qc_valid_l_scapular = qc_height_l_scapular & qc_width_l_scapular


            # Lung
            height_r_lung = (values_r['lung_height'] / img_height)
            height_l_lung = (values_l['lung_height'] / img_height)

            qc_height_r_lung = (height_r_lung >= md_range_per_mask['lung']['height'])
            qc_height_l_lung = (height_l_lung >= md_range_per_mask['lung']['height'])

            width_r_lung = (values_r['lung_width'] / img_width)
            width_l_lung = (values_l['lung_width'] / img_width)

            qc_width_r_lung = (width_r_lung >= md_range_per_mask['lung']['width'])
            qc_width_l_lung = (width_l_lung >= md_range_per_mask['lung']['width'])

            qc_valid_r_lung = qc_height_r_lung & qc_width_r_lung
            qc_valid_l_lung = qc_height_l_lung & qc_width_l_lung

            qc_valid_right = qc_valid_r_lung & qc_valid_r_scapular & (~qc_r_scapular_over)
            qc_valid_left = qc_valid_l_lung & qc_valid_l_scapular & (~qc_l_scapular_over)

            qc_valid_both = qc_valid_right & qc_valid_left

            if qc_valid_both:
                overlap_ratio_right = (values_r['overlap_region'] / values_r['scapular_region']).round(ROUND_VALUE['Projection'])
                overlap_ratio_left = (values_l['overlap_region'] / values_l['scapular_region']).round(ROUND_VALUE['Projection'])

                overlap_large_right = (overlap_ratio_right >= MD_RANGES['Projection'])
                overlap_large_left = (overlap_ratio_left >= MD_RANGES['Projection'])

                overlap_large = (overlap_large_right & overlap_large_left)
                overlap_small = (~overlap_large_right & ~overlap_large_left)

                if (overlap_large | overlap_small):
                    meta_projection['scapular_region_right'] = values_r['scapular_region']
                    meta_projection['overlap_region_right'] = values_r['overlap_region']
                    meta_projection['ratio_right'] = overlap_ratio_right

                    meta_projection['scapular_region_left'] = values_l['scapular_region']
                    meta_projection['overlap_region_left'] = values_l['overlap_region']
                    meta_projection['ratio_left'] = overlap_ratio_left


                    if overlap_large:
                        meta_projection['label'] = 1 # 'AP'
                    elif overlap_small:
                        meta_projection['label'] = 0  # 'PA'

                    meta_projection['viz_overlap_region_right'] = values_r['overlap_region']
                    meta_projection['viz_scapular_region_right'] = values_r['scapular_region']

                    meta_projection['viz_overlap_region_left'] = values_l['overlap_region']
                    meta_projection['viz_scapular_region_left'] = values_l['scapular_region']


    return meta_projection











