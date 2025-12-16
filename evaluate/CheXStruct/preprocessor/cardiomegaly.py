import os
import cv2
import json
import numpy as np

from utils.constants import TARGET_MASK_LIST, MD_RANGES, ROUND_VALUE
from utils.utils import select_max_area_mask, select_max_width_mask

def diff_center(mask_heart, bbox_coord, mask_coord):
    img_height, img_width = mask_heart.shape
    center_cxas = [(mask_coord[0] + mask_coord[2]) // 2, (mask_coord[1] + mask_coord[3]) // 2]
    center_CI = [(bbox_coord[0] + bbox_coord[2]) // 2, (bbox_coord[1] + bbox_coord[3]) // 2]

    normalized_center_distance = np.linalg.norm([(center_cxas[0] - center_CI[0]) / img_width,
                                                 (center_cxas[-1] - center_CI[-1]) / img_height])

    return center_cxas, center_CI, normalized_center_distance


def iou_bbox_mask(coord_bbox, coord_mask):
    x1_1, y1_1, x2_1, y2_1 = coord_bbox
    x1_2, y1_2, x2_2, y2_2 = coord_mask

    # Intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Intersection dimensions
    intersection_width = max(0, xi2 - xi1)
    intersection_height = max(0, yi2 - yi1)
    intersection_area = intersection_width * intersection_height

    # Areas of the boxes
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Union area
    union_area = area_box1 + area_box2 - intersection_area

    # IoU calculation
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou


def extract_cardiomegaly(image_data, meta_inclusion):
    meta_cardiomegaly = dict()

    if ('label_side_right_lung' in meta_inclusion) & ('label_side_left_lung' in meta_inclusion):
        if meta_inclusion['label_side_right_lung'] & meta_inclusion['label_side_left_lung']:
            bbox_file = image_data['bbox']
            cxas_files = image_data['cxas']
            if os.path.isfile(bbox_file):
                with open(bbox_file, 'r') as f:
                    bboxes = json.load(f)

                coord_bbox = []
                for obj in bboxes['objects']:
                    object_name = obj["bbox_name"]
                    if object_name == 'cardiac silhouette':
                        x1 = obj["original_x1"]
                        y1 = obj["original_y1"]
                        x2 = obj["original_x2"]
                        y2 = obj["original_y2"]
                        coord_bbox = [x1, y1, x2, y2]

                cxas_heart = cv2.imread(cxas_files['heart'], 0)
                cxas_rlung = cv2.imread(cxas_files['right lung'], 0)
                cxas_llung = cv2.imread(cxas_files['left lung'], 0)

                if cxas_heart.sum() != 0 and cxas_rlung.sum() != 0 and cxas_llung.sum() != 0 and len(coord_bbox) == 4:
                    # heart
                    cxas_heart_refined, coord_mask = select_max_width_mask(cxas_heart)
                    xmin_heart, ymin_heart, xmax_heart, ymax_heart = coord_mask
                    heart_width = xmax_heart - xmin_heart

                    # lungs
                    cxas_rlung_refined = select_max_area_mask(cxas_rlung)[coord_mask[1]:]
                    cxas_llung_refined = select_max_area_mask(cxas_llung)[coord_mask[1]:]
                    if cxas_rlung_refined.sum() != 0 and cxas_llung_refined.sum() != 0:
                        lungs = (cxas_rlung_refined | cxas_llung_refined)
                        x_indices = lungs.sum(axis=0).nonzero()[0]
                        xmin_lung, xmax_lung = x_indices[0], x_indices[-1]
                        lung_width = (xmax_lung - xmin_lung)

                        ctr = heart_width / lung_width

                        # quality control
                        center_cxas, center_bbox, normalized_center_distance = diff_center(cxas_heart, coord_bbox, coord_mask)
                        iou = iou_bbox_mask(coord_bbox, coord_mask)

                        qc_valid_heart = ((iou >= 0.5) & (iou < 0.6) & (normalized_center_distance <= 0.2)) \
                                   or ((iou >= 0.6) & (normalized_center_distance <= 0.15))


                        rlung_end = meta_inclusion['right_lung_inner_point']
                        llung_end = meta_inclusion['left_lung_inner_point']

                        width_rl = rlung_end[0] - xmin_lung
                        width_ll = xmax_lung - llung_end[0]

                        mask_r_l = (width_ll > width_rl)
                        lung_width_ratio_r_l = (width_rl / width_ll)

                        mask_l_r = (width_rl > width_ll)
                        lung_width_ratio_l_r = (width_ll / width_rl)

                        lung_width_ratio = (mask_r_l * lung_width_ratio_r_l) + (mask_l_r * lung_width_ratio_l_r)
                        qc_valid_lung = (lung_width_ratio >= 0.7)

                        meta_cardiomegaly['qc_valid_heart'] = qc_valid_heart
                        if qc_valid_heart & qc_valid_lung:
                            meta_cardiomegaly['heart_xmin'] = xmin_heart
                            meta_cardiomegaly['heart_xmax'] = xmax_heart
                            meta_cardiomegaly['heart_width'] = heart_width

                            meta_cardiomegaly['lung_xmin'] = xmin_lung
                            meta_cardiomegaly['lung_xmax'] = xmax_lung
                            meta_cardiomegaly['lung_width'] = lung_width

                            meta_cardiomegaly['ctr'] = ctr
                            if image_data['viewposition'] == 'PA':
                                if round(ctr, ROUND_VALUE['Cardiomegaly']) >= MD_RANGES['Cardiomegaly']['PA']:
                                    meta_cardiomegaly['label'] = 1
                                else:
                                    meta_cardiomegaly['label'] = 0

                            elif image_data['viewposition'] == 'AP':
                                if round(ctr, ROUND_VALUE['Cardiomegaly']) >= MD_RANGES['Cardiomegaly']['AP']:
                                    meta_cardiomegaly['label'] = 1
                                else:
                                    meta_cardiomegaly['label'] = 0

                            else:
                                meta_cardiomegaly['label'] = 'N/A'

                            meta_cardiomegaly['viz_coord_mask'] = coord_mask
                            meta_cardiomegaly['viz_xmin_lung'] = xmin_lung
                            meta_cardiomegaly['viz_xmax_lung'] = xmax_lung

    return meta_cardiomegaly











