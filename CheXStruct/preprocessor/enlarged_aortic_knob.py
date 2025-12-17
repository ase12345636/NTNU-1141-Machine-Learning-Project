import cv2
import numpy as np
from scipy import ndimage

from utils.constants import TARGET_MASK_LIST, MD_RANGES, ROUND_VALUE
from utils.utils import select_max_area_mask, select_max_width_mask

def select_target_axis_mask(mask, target_area):
    mask_width_idx = target_area.sum(axis=0).nonzero()[0]
    xmin, xmax = mask_width_idx[0], mask_width_idx[-1]
    label_im, nb_labels = ndimage.label(mask)
    target_mask = np.zeros_like(mask)
    for i in range(nb_labels):
        mask_compare = np.full(np.shape(label_im), i + 1)
        separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
        intersection_area = separate_mask[:, xmin:(xmax + 1)].sum()
        if intersection_area:
            target_mask = (target_mask | separate_mask)
    return target_mask

def select_target_area_mask(mask, target_area):
    label_im, nb_labels = ndimage.label(mask)
    target_mask = np.zeros_like(mask)
    for i in range(nb_labels):
        mask_compare = np.full(np.shape(label_im), i + 1)
        separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
        intersection_area = (separate_mask & target_area).sum()
        if intersection_area:
            target_mask = (target_mask | separate_mask)
    return target_mask

def extract_enlarged_aortic_knob(image_data, meta_trachea):
    meta_enlarged_aortic_knob = dict()

    if ('qc_valid_trachea' in meta_trachea):
        if meta_trachea['qc_valid_trachea']:
            subpart = 30
            cxas_files = image_data['cxas']

            mask_trachea = cv2.imread(cxas_files["trachea"], 0)
            mask_carina = cv2.imread(cxas_files["tracheal bifurcation"], 0)
            mask_descending_aorta = cv2.imread(cxas_files["descending aorta"], 0)
            mask_aortic_arch = cv2.imread(cxas_files["aortic arch"], 0)

            if mask_trachea.sum() != 0 and mask_carina.sum() != 0 \
                    and mask_descending_aorta.sum() != 0 and mask_aortic_arch.sum() != 0:
                img_height, img_width = mask_trachea.shape

                y_pos_carina_start = mask_carina.sum(axis=-1).nonzero()[0][0]
                mask_trachea_refined = mask_trachea.copy()
                mask_trachea_refined[y_pos_carina_start:] = 0

                mask_trachea_refined = select_max_area_mask(mask_trachea_refined)
                trachea_height_idx = mask_trachea_refined.sum(axis=-1).nonzero()[0]

                if len(trachea_height_idx):
                    y_min, y_max = trachea_height_idx[0], trachea_height_idx[-1]
                    y_subpart = int((y_max - y_min) * (1 / 4))
                    mask_trachea_refined[:(y_min + y_subpart)] = 0

                    # descending aorta mask refinement
                    mask_descending_aorta_refined = mask_descending_aorta.copy()
                    mask_descending_aorta_refined[:y_max] = 0
                    if mask_descending_aorta_refined.sum():
                        desc_aorta_height_idx = mask_descending_aorta_refined.sum(axis=-1).nonzero()[0]
                        ymin_desc, ymax_desc = desc_aorta_height_idx[0], desc_aorta_height_idx[-1]
                        ysub_desc = int((ymax_desc - ymin_desc) * (subpart / 100))
                        mask_descending_aorta_refined[(ymin_desc + ysub_desc):] = 0

                        if mask_descending_aorta_refined.sum():
                            mask_descending_aorta_refined, _ = select_max_width_mask(mask_descending_aorta_refined)

                            # Calculate width of descending aorta
                            xmin_desc_sub = mask_descending_aorta_refined.sum(axis=0).nonzero()[0][0]
                            y_desc_sub = mask_descending_aorta_refined[:, xmin_desc_sub].nonzero()[0][0]
                            xmax_desc_sub = mask_descending_aorta_refined[y_desc_sub].nonzero()[0][-1]

                            desc_aorta_sub_height_idx = mask_descending_aorta_refined.sum(axis=-1).nonzero()[0]
                            ymin_desc_sub, ymax_desc_sub = desc_aorta_sub_height_idx[0], desc_aorta_sub_height_idx[-1]
                            desc_aorta_sub_height = (ymax_desc_sub - ymin_desc_sub) / img_height

                            # Aortic knob
                            mask_aortic_arch[ymax_desc_sub:] = 0
                            mask_aortic_arch_refined = select_target_axis_mask(mask_aortic_arch, mask_descending_aorta_refined)
                            aortic_arch_width_idx = mask_aortic_arch_refined.sum(axis=0).nonzero()[0]
                            aortic_arch_height_idx = mask_aortic_arch_refined.sum(axis=-1).nonzero()[0]

                            if len(aortic_arch_width_idx) and len(aortic_arch_height_idx):
                                xmin_aortic_arch, xmax_aortic_arch = aortic_arch_width_idx[0], aortic_arch_width_idx[-1]

                                # return the original mask region of the refined mask
                                mask_desc_aorta_target = select_target_area_mask(mask_descending_aorta, mask_descending_aorta_refined)
                                mask_desc_aorta_target[: y_max] = 0
                                blank_y_btw_trachea = (ymin_desc_sub - y_max) / img_height

                                ymin_aortic_arch, ymax_aortic_arch = aortic_arch_height_idx[0], aortic_arch_height_idx[-1]
                                mask_trachea_within_aortic_arch = mask_trachea_refined[ymin_aortic_arch: (ymax_aortic_arch + 1)]

                                if mask_trachea_within_aortic_arch.sum():
                                    rightmost_xs = []
                                    for row in mask_trachea_within_aortic_arch:
                                        ones_indices = np.where(row == 1)[0]  # Find indices of 1s
                                        if ones_indices.size > 0:
                                            rightmost_xs.append(ones_indices[-1])  # Last index where 1 appears
                                    rightmost_xs = np.array(rightmost_xs)
                                    xmax_trachea_mean = rightmost_xs.mean()

                                    aortic_knob_width_refined = (xmax_aortic_arch - xmax_trachea_mean) \
                                        if (xmax_aortic_arch > xmax_desc_sub) else (xmax_desc_sub - xmax_trachea_mean)

                                    trachea_width = meta_trachea['trachea_width']
                                    ratio_enlarged_median_refined = (aortic_knob_width_refined / trachea_width)


                                    qc_valid_desc_aorta_height = (desc_aorta_sub_height >= 0.07)
                                    x_diff_aortic_knob = (xmax_aortic_arch - xmax_desc_sub) / img_width
                                    qc_valid_aortic_knob_x_diff = x_diff_aortic_knob >= -0.02
                                    qc_valid_blank_y_btw_trachea = blank_y_btw_trachea <= 0.09
                                    qc_valid_positive_aortic_knob_width = (aortic_knob_width_refined / img_width) > 0

                                    qc_valid = qc_valid_desc_aorta_height & qc_valid_aortic_knob_x_diff \
                                               & qc_valid_blank_y_btw_trachea & qc_valid_positive_aortic_knob_width

                                    if qc_valid:
                                        meta_enlarged_aortic_knob['aortic_knob_width'] = aortic_knob_width_refined
                                        meta_enlarged_aortic_knob['aortic_knob_xmin'] = xmax_trachea_mean
                                        meta_enlarged_aortic_knob['aortic_knob_xmax'] = xmax_aortic_arch if (xmax_aortic_arch > xmax_desc_sub) else xmax_desc_sub

                                        meta_enlarged_aortic_knob['trachea_width'] = trachea_width
                                        meta_enlarged_aortic_knob['trachea_point_right'] = meta_trachea['trachea_point_right']
                                        meta_enlarged_aortic_knob['trachea_point_left'] = meta_trachea['trachea_point_left']

                                        meta_enlarged_aortic_knob['ratio'] = ratio_enlarged_median_refined

                                        if round(ratio_enlarged_median_refined, ROUND_VALUE['Aortic_Knob_Enlargement']) >= MD_RANGES['Aortic_Knob_Enlargement']:
                                            meta_enlarged_aortic_knob['label'] = 1
                                        else:
                                            meta_enlarged_aortic_knob['label'] = 0

                                        meta_enlarged_aortic_knob['viz_y_max'] = y_max
                                        meta_enlarged_aortic_knob['viz_ymin_desc'] = ymin_desc
                                        meta_enlarged_aortic_knob['viz_ysub_desc'] = ysub_desc
                                        meta_enlarged_aortic_knob['viz_ymax_desc_sub'] = ymax_desc_sub
                                        meta_enlarged_aortic_knob['viz_xmax_trachea_mean'] = xmax_trachea_mean

                                        meta_enlarged_aortic_knob['viz_trachea_y_width'] = meta_trachea['trachea_point_right'][-1]
                                        meta_enlarged_aortic_knob['viz_trachea_xmin_width'] = meta_trachea['trachea_point_left'][0]
                                        meta_enlarged_aortic_knob['viz_trachea_xmax_width'] = meta_trachea['trachea_point_right'][0]

                                        meta_enlarged_aortic_knob['viz_xmax_aortic_arch'] = xmax_aortic_arch
                                        meta_enlarged_aortic_knob['viz_xmax_desc_sub'] = xmax_desc_sub
                                        meta_enlarged_aortic_knob['viz_ymin_aortic_arch'] = ymin_aortic_arch
                                        meta_enlarged_aortic_knob['viz_ymax_aortic_arch'] = ymax_aortic_arch

    return meta_enlarged_aortic_knob












