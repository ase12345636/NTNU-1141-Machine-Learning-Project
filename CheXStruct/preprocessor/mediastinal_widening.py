import os
import cv2
import json
import numpy as np
from PIL import Image

from utils.constants import TARGET_MASK_LIST, MD_RANGES, ROUND_VALUE
from utils.utils import select_max_area_mask, select_max_width_mask


def extract_mw(image_data, meta_inclusion):
    meta_mw = dict()

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
                    if object_name == 'upper mediastinum':
                        x1 = obj["original_x1"]
                        y1 = obj["original_y1"]
                        x2 = obj["original_x2"]
                        y2 = obj["original_y2"]
                        coord_bbox = [x1, y1, x2, y2]
                cxas_medi = cv2.imread(cxas_files['upper mediastinum'], 0)
                cxas_rlung = cv2.imread(cxas_files['right lung'], 0)
                cxas_llung = cv2.imread(cxas_files['left lung'], 0)

                if cxas_medi.sum() != 0 and cxas_rlung.sum() != 0 and cxas_llung.sum() != 0 and len(coord_bbox) == 4:
                    # mediastinum
                    cxas_medi_refined, coord_mask = select_max_width_mask(cxas_medi)
                    cxas_rlung_refined = select_max_area_mask(cxas_rlung)
                    cxas_llung_refined = select_max_area_mask(cxas_llung)

                    row_lst = []
                    for row in cxas_medi_refined:
                        if row.sum() != 0:
                            row_width = abs(row.nonzero()[0][0] - row.nonzero()[0][-1])
                        else:
                            row_width = 0
                        row_lst.append(row_width)

                    y_medi = np.array(row_lst).argmax()

                    if cxas_rlung_refined[y_medi].sum() != 0 and cxas_llung_refined[y_medi].sum() != 0:
                        xmin_medi = cxas_medi_refined[y_medi].nonzero()[0][0]
                        xmax_medi = cxas_medi_refined[y_medi].nonzero()[0][-1]

                        xmin_rlung = cxas_rlung_refined[y_medi].nonzero()[0][0]
                        xmax_rlung = cxas_rlung_refined[y_medi].nonzero()[0][-1]

                        xmin_llung = cxas_llung_refined[y_medi].nonzero()[0][0]
                        xmax_llung = cxas_llung_refined[y_medi].nonzero()[0][-1]

                        medi_width = xmax_medi - xmin_medi
                        lung_width = xmax_llung - xmin_rlung

                        mcr = medi_width / lung_width


                        # Quality Control
                        # Mediastinum - Width
                        medi_width_bbox = coord_bbox[2] - coord_bbox[0]
                        medi_width_ratio = medi_width / medi_width_bbox
                        qc_valid_medi_width = (medi_width_ratio >= 0.5) \
                                              & (medi_width_ratio < 1.0)

                        # Mediastinum - X diff
                        xmid_medi = (xmax_medi + xmin_medi) / 2
                        xmid_medi_bbox = (coord_bbox[0] + coord_bbox[2]) / 2
                        medi_xdiff = abs(xmid_medi_bbox - xmid_medi) / lung_width
                        qc_valid_medi_xdiff = (medi_xdiff >= 0.0) & (medi_xdiff < 0.1)

                        # Mediastinum - Y Pos
                        cxr = np.array(Image.open(image_data['cxr']))
                        medi_mid = ((xmin_medi + xmax_medi) // 2, y_medi)
                        line_b4_medi = np.prod((cxr[:medi_mid[-1], (medi_mid[0] - 2): (medi_mid[0] + 3)]), axis=1)

                        non_black_indices_b4_medi = np.where(line_b4_medi != 0)[0]
                        if len(non_black_indices_b4_medi):
                            first_non_black_index = non_black_indices_b4_medi[0]
                        else:
                            first_non_black_index = medi_mid[0]

                        line_af_medi = np.prod((cxr[medi_mid[-1]:, (medi_mid[0] - 2): (medi_mid[0] + 3)]), axis=1)
                        non_black_indices_af_medi = np.where(line_af_medi != 0)[0]
                        if len(non_black_indices_af_medi):
                            last_non_black_index = non_black_indices_af_medi[-1] + medi_mid[-1]
                        else:
                            last_non_black_index = medi_mid[-1]

                        medi_distance = y_medi - first_non_black_index
                        total_distance = last_non_black_index - first_non_black_index
                        ratio_ypos = (medi_distance / total_distance)
                        qc_valid_medi_ypos = (ratio_ypos >= 0.0) & (ratio_ypos < 0.5)

                        qc_valid_medi = qc_valid_medi_width & qc_valid_medi_xdiff & qc_valid_medi_ypos

                        # Lung
                        rlung_end = meta_inclusion['right_lung_inner_point']
                        llung_end = meta_inclusion['left_lung_inner_point']

                        width_rl = rlung_end[0] - xmin_rlung
                        width_ll = xmax_llung - llung_end[0]

                        mask_r_l = (width_ll > width_rl)
                        lung_width_ratio_r_l = (width_rl / width_ll)

                        mask_l_r = (width_rl > width_ll)
                        lung_width_ratio_l_r = (width_ll / width_rl)

                        lung_width_ratio = (mask_r_l * lung_width_ratio_r_l) + (mask_l_r * lung_width_ratio_l_r)
                        qc_valid_lung = (lung_width_ratio >= 0.7)

                        if qc_valid_medi & qc_valid_lung:
                            meta_mw['mediastinum_xmin'] = xmin_medi
                            meta_mw['mediastinum_xmax'] = xmax_medi
                            meta_mw['mediastinum_width'] = medi_width

                            meta_mw['lung_xmin'] = xmin_rlung
                            meta_mw['lung_xmax'] = xmax_llung
                            meta_mw['lung_width'] = lung_width

                            meta_mw['mcr'] = mcr

                            if image_data['viewposition'] == 'PA':
                                if round(mcr, ROUND_VALUE['Cardiomegaly']) >= MD_RANGES['Mediastinal_Widening']['PA']:
                                    meta_mw['label'] = 1
                                else:
                                    meta_mw['label'] = 0


                            elif image_data['viewposition'] == 'AP':
                                if round(mcr, ROUND_VALUE['Cardiomegaly']) >= MD_RANGES['Mediastinal_Widening']['AP']:
                                    meta_mw['label'] = 1
                                else:
                                    meta_mw['label'] = 0

                            else:
                                meta_mw['label'] = 'N/A'

                            meta_mw['viz_y_medi'] = y_medi
                            meta_mw['viz_xmin_rlung'] = xmin_rlung
                            meta_mw['viz_xmax_llung'] = xmax_llung

                            meta_mw['viz_xmin_medi'] = xmin_medi
                            meta_mw['viz_xmax_medi'] = xmax_medi
    return meta_mw











