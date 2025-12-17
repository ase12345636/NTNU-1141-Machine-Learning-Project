import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict

# Global variable to store dataset configuration
_dataset_config = None

class DatasetConfig:
    """Helper class to handle dataset-specific operations"""
    def __init__(self, args, mimic_meta=None):
        self.args = args
        self.mimic_meta = mimic_meta
        self.dataset_name = args.dataset_name
        self.chexstruct_base_dir = args.chexstruct_base_dir
        
    def get_view_position(self, dicom, dx):
        """Get view position for a given image"""
        if self.dataset_name == 'nih-cxr14':
            df = pd.read_csv(os.path.join(self.chexstruct_base_dir, f'{dx}.csv'))
            if 'viewposition' in df.columns:
                return df[df['image_file'] == dicom]['viewposition'].values[0]
            return 'PA'  # Default
        else:
            return self.mimic_meta[self.mimic_meta['dicom_id'] == dicom]['ViewPosition'].values[0]
    
    def get_cxr_path(self, dicom):
        """Get image path for a given DICOM ID"""
        if self.dataset_name == 'nih-cxr14':
            return f'{dicom}.png'
        else:
            sid = self.mimic_meta[self.mimic_meta['dicom_id'] == dicom]['study_id'].values[0]
            pid = self.mimic_meta[self.mimic_meta['dicom_id'] == dicom]['subject_id'].values[0]
            return f'p{str(pid)[:2]}/p{pid}/s{sid}/{dicom}.jpg'

def set_dataset_config(config):
    """Set global dataset configuration"""
    global _dataset_config
    _dataset_config = config

def get_dataset_config():
    """Get global dataset configuration"""
    return _dataset_config

def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inference_path', default='path1', type=str, choices=['path1', 'path2', 're-path1'])
    parser.add_argument('--dataset_name', default='mimic-cxr-jpg', type=str, choices=['mimic-cxr-jpg', 'nih-cxr14'])

    parser.add_argument('--save_base_dir', default='path/to/save/output', type=str)

    parser.add_argument('--chexstruct_base_dir', default='path/to/chexstruct_data', type=str)
    parser.add_argument('--cxreasonbench_base_dir', default=f'path/to/cxreasonbench_metadata', type=str)

    # MIMIC-CXR arguments
    parser.add_argument('--mimic_cxr_base', default='', type=str)
    parser.add_argument('--mimic_meta_path', default='', type=str)

    # NIH CXR-14 arguments
    parser.add_argument('--nih_image_base_dir', default='/mnt/d/CXReasonBench/dataset', type=str, help='Path to NIH images folders')

    parser.add_argument('--num_options', default=5, type=int)

    args = parser.parse_args()
    return args

threshold_per_dx = {
    'inclusion': '', 'trachea_deviation': '',
    'cardiomegaly': {'PA': 0.5, 'AP': 0.55}, 'mediastinal_widening': {'PA': 0.25, 'AP': 0.33},
    'carina_angle': [40, 80],
    'inspiration': 9, 'rotation': 0.4, 'projection': 0.3,
    'ascending_aorta_enlargement': 0.1, 'aortic_knob_enlargement': 2.5,
    'descending_aorta_enlargement': 2.5, 'descending_aorta_tortuous': 0.0009,
}
round_per_dx = {
    'rotation': 2, 'projection': 2,
    'cardiomegaly': 2, 'mediastinal_widening': 2, 'carina_angle': 0,
    'aortic_knob_enlargement': 2, 'descending_aorta_enlargement': 2, 'descending_aorta_tortuous': 4,
}
map_tolerance_per_dx = {
            'cardiomegaly_PA': {'tolerance': 0.01, 'margin': 0.01, 'round': 2,
                                'min_limit': 0.3, 'max_limit': 0.95},

            'cardiomegaly_AP': {'tolerance': 0.01, 'margin': 0.01, 'round': 2,
                                'min_limit': 0.3, 'max_limit': 0.95},

            'mediastinal_widening_PA': {'tolerance': 0.015, 'margin': 0.001, 'round': 3,
                                        'min_limit': 0.1, 'max_limit': 0.95},

            'mediastinal_widening_AP': {'tolerance': 0.015, 'margin': 0.001, 'round': 3,
                                        'min_limit': 0.1, 'max_limit': 0.95},

            'projection': {'tolerance': 0.1, 'margin': 0.01, 'round': 2,
                           'min_limit': 0, 'max_limit': 1.0},
            'rotation': {'tolerance': 0.1, 'margin': 0.01, 'round': 2,
                         'min_limit': 0.0, 'max_limit': 1.0},

            'carina_angle': {'tolerance': 10, 'margin': 1, 'round': 0,
                             'min_limit': 0, 'max_limit': 180},

            'aortic_knob_enlargement': {'tolerance': 0.1, 'margin': 0.01, 'round': 2,
                                     'min_limit': 0.0, 'max_limit': 10},

            'descending_aorta_enlargement': {'tolerance': 0.1, 'margin': 0.01, 'round': 2,
                                          'min_limit': 0.0, 'max_limit': 10},

            'descending_aorta_tortuous': {'tolerance': 0.00015, 'margin': 0.00001, 'round': 5,
                                          'min_limit': 0.0, 'max_limit': 1.0},
        }


initial_question_per_dx = {
    'inclusion': "Is the entire thoracic cage - including the lung apices, "
                 "inner margins of the lateral ribs, and costophrenic angles (CPAs) - "
                 "fully visible in this chest X-ray without being cropped? "
                 "Options: (a) Yes, (b) No, (c) I don't know",

    'inspiration': "Assess the level of inspiration in this chest X-ray. Was it taken with good or poor inspiration? "
                   "Options: (a) Good, (b) Poor, (c) I don't know",

    'rotation': "Was the patient rotated during the chest X-ray? "
                "Options: (a) Yes, (b) No, (c) I don't know",

    'projection': "Identify the view of this chest X-ray. "
                  "Options: (a) PA (Posteroanterior), (b) AP (Anteroposterior), (c) I don't know ",

    'cardiomegaly': "Does this patient have cardiomegaly? "
                    "Options: (a) Yes, (b) No, (c) I don't know",

    'mediastinal_widening': "Does this patient have mediastinal widening? "
                            "Options: (a) Yes, (b) No, (c) I don't know",

    'carina_angle': "Does this chest X-ray show a normal carina angle? "
                    "Options: (a) Yes, (b) No, (c) I don't know",

    'trachea_deviation': "Is the trachea deviated in this chest X-ray? "
                         "Options: (a) Yes (b) No (c) I don't know ",

    'aortic_knob_enlargement': "Does the aortic knob appear enlarged in this chest X-ray? "
                            "Options: (a) Yes, (b) No, (c) I don't know",
    'ascending_aorta_enlargement': "Does the ascending aorta appear enlarged in this chest X-ray? "
                                "Options: (a) Yes, (b) No, (c) I don't know",
    'descending_aorta_enlargement': "Does the descending aorta appear enlarged in this chest X-ray? "
                                 "Options: (a) Yes, (b) No, (c) I don't know",
    'descending_aorta_tortuous': "Is the descending aorta tortuous in this chest X-ray? "
                                 "Options: (a) Yes, (b) No, (c) I don't know",
}
criteria_per_dx = {
    'inclusion': "By checking if the chest X-ray includes the lung apices, "
                 "inner margins of the lateral ribs, and costophrenic angles.",

    'inspiration': "By counting the number of right posterior ribs or anterior rib "
                   "visible above the right hemidiaphragm.",

    'rotation': "By checking if the spinous processes are equidistant "
                "from the medial ends of the clavicles.",

    'projection': "By checking if the scapulae were laterally retracted "
                  "or if they overlapped the lung fields.",

    'cardiomegaly': "By calculating the cardiothoracic ratio, "
                    "which is the ratio of the maximal horizontal cardiac diameter "
                    "to the maximal horizontal thoracic diameter.",

    'mediastinal_widening': "By evaluating the width of the mediastinum.",

    'carina_angle': "By evaluating the angle of the carina.",

    'trachea_deviation': "By checking if the trachea is displaced to one side from the midline.",

    'aortic_knob_enlargement': "By checking for any prominent bulge or abnormal widening of the aortic knob.",

    'ascending_aorta_enlargement': "By checking for any bulge or abnormal widening of the ascending aorta.",

    'descending_aorta_enlargement': "By checking for any abnormal dilation or widening of the descending aorta.",

    'descending_aorta_tortuous': "By evaluating the shape of the descending aorta "
                                 "and checking for any signs of tortuosity, such as irregular bends or twists.",
}
refined_criteria_per_dx = {
    'inspiration': "The criterion has been refined to reduce ambiguity and ensure consistency "
                   "in measurements between individuals, "
                   "as the previous criterion lacked a clear reference point, which could result in "
                   "variations in interpretation. Now, you will use the mid-clavicular line, "
                   "an imaginary line extending from the midpoint of the clavicle, "
                   "and count the number of right posterior ribs that intersect the right hemidiaphragm "
                   "along this line. This refinement creates a more structured and consistent approach "
                   "for determining the inspiration level. "
                   "Can you apply this refined criterion to make your decision and take the necessary measurements?",

    'projection': "The previous method for determining chest X-ray projection (PA or AP) relied on "
                  "visually assessing scapular retraction or overlap with the lung fields. "
                  "While this approach reflects the correct fundamental concept — "
                  "that scapular positioning is indicative of projection — "
                  "it lacked clear thresholds, leading to variability in interpretation among individuals. "
                  "To reduce this ambiguity and enhance measurement consistency between evaluators, "
                  "the criterion has been refined. The refined method maintains the original diagnostic logic "
                  "but introduces a more structured process: measuring the ratio of the overlapping area "
                  "between the scapula and the lung to the total scapular area for "
                  "both the right and left scapula. This update ensures a more objective and "
                  "standardized application of the same underlying principle. "
                  "Can you apply this refined criterion to determine the overlap ratio for each side?",


    'mediastinal_widening': "The original criterion for assessing mediastinal widening involves measuring "
                            "the mediastinal width in centimeters, typically using metadata or physical markers. "
                            "However, in this evaluation setting where only images are provided without accompanying metadata, "
                            "absolute measurements are not feasible. To maintain the fundamental diagnostic approach "
                            "while adapting to this constraint, the criterion has been refined to use a ratio-based measurement. "
                            "Specifically, evaluators are now instructed to measure both the mediastinal width and "
                            "the thoracic width at the same level, then calculate the ratio by dividing the mediastinal width "
                            "by the thoracic width. This updated approach enables consistent and objective assessments "
                            "even in the absence of physical measurement units. "
                            "Can you apply this refined method to assess and take the necessary measurements?",

    'aortic_knob_enlargement': "The original criterion for assessing an enlarged aortic knob involved visually inspecting "
                            "for any prominent bulge or abnormal widening of the aortic knob. "
                            "However, this approach can be subjective and may vary among evaluators. "
                            "To enhance consistency and objectivity in assessment, the criterion has been refined. "
                            "The refined criterion involves measuring both the maximum width of the aortic knob and "
                            "the median width of the trachea. Then, calculate the ratio by dividing the aortic knob width "
                            "by the trachea width. This updated approach provides a more standardized and "
                            "reliable method for identifying an enlarged aortic knob. "
                            "Can you apply this refined criterion to assess and take the necessary measurements?",

    'ascending_aorta_enlargement': "The original criterion for assessing an enlarged ascending aorta involved visually inspecting "
                                "for any bulge or abnormal widening of the ascending aorta. "
                                "However, this approach can be subjective and may vary among evaluators. "
                                "To enhance consistency and objectivity in assessment, the criterion has been refined. "
                                "The refined criterion involves drawing an imaginary straight line connecting "
                                "the inner boundary of the right lung and the right heart side. "
                                "Then, determine whether the ascending aorta extends beyond this line. "
                                "This approach provides a more reproducible and objective way to assess aortic enlargement. "
                                "Can you apply this refined criterion to assess and take the necessary measurements?",

    'descending_aorta_enlargement': "The original criterion for assessing an enlarged descending aorta involved visually inspecting "
                                "for any bulge or abnormal widening of the descending aorta. "
                                "However, this approach can be subjective and may vary among evaluators. "
                                "To enhance consistency and objectivity in assessment, the criterion has been refined. "
                                "The refined criterion involves measuring both the maximum width of the descending aorta and "
                                 "the median width of the trachea. Then, calculate the ratio by dividing the descending aorta width "
                                 "by the trachea width. This updated approach provides a more standardized and "
                                 "reliable method for identifying an enlarged descending aorta. "
                                "Can you apply this refined criterion to assess and take the necessary measurements?",

    'descending_aorta_tortuous': "The original criterion for assessing descending aorta tortuosity involved visually inspecting "
                             "the shape of the descending aorta and checking for any signs of tortuosity. "
                             "However, this approach can be subjective and may vary among evaluators. "
                             "To enhance consistency and objectivity in assessment, the criterion has been refined "
                             "to include a more quantitative approach. "
                             "This refined method involves focusing on the thoracic portion of the descending aorta, "
                             "particularly the region at the upper part of the heart. "
                             "The region is divided into five equal sections, and six coordinates are determined "
                             "at the top-left lung side of each division. Curvature is then calculated at each of "
                             "these six coordinates using finite difference methods: "
                             "forward and backward differences for the first and last points, "
                             "and central differences for the middle points to ensure greater accuracy. "
                             "The average curvature is computed across all six points to quantify the tortuosity of the descending aorta. "
                             "This refined approach minimizes evaluator variability and provides a more objective and "
                             "reproducible assessment of aortic tortuosity. "
                             "Can you apply this refined criterion to assess and take the necessary measurements?",
}
measurement_q_per_dx = {
    # ========================
    #      RECOGNITION
    # ========================
    'inclusion': {
        'Explanation': "Determine whether the chest X-ray includes the lung apices, inner margins of the lateral ribs, and costophrenic angles "
                       "based on the image that was provided earlier and "
                       "select the appropriate label. Choose only one.",
        'Simple': "Use the criterion you selected to assess which parts of the thoracic cage are "
                  "included or excluded, and select the correct option. Choose only one."},

    'inspiration': {
        'Explanation': "Using the previously provided chest X-ray, draw an imaginary mid-clavicular line extending vertically from the midpoint of the clavicle. "
                       "Count the number of right posterior ribs that intersect the right hemidiaphragm along this line, and select the correct number. Choose only one.",
        'Simple': "Use the refined criterion outlined earlier to count the rib number and "
                  "select the correct option. Choose only one."
    },

    'trachea_deviation': {
        'Explanation': "In the previously provided chest X-ray, using the spinous processes as a reference to "
                       "draw an imaginary vertical line down the center of the vertebral bodies. "
                       "Assess whether the trachea is aligned with this vertical line "
                       "and select the appropriate label. Choose only one.",
        'Simple': "Use the criterion you selected to assess the deviation and select the correct option. Choose only one."
    },

    # recognition - our own metric
    'ascending_aorta_enlargement': {
        'Explanation': "Using the previously provided chest X-ray, draw an imaginary straight line "
                       "connecting the inner boundary of the right lung and the right heart side. "
                       "Determine whether the ascending aorta extends beyond this line "
                       "and select the appropriate label. Choose only one.",
        'Simple': "Use the refined criterion outlined earlier to assess "
                  "whether the ascending aorta extends beyond the line. "
                  "Select the appropriate label based on your measurement. Choose only one."
    },

    # ========================
    #      MEASUREMENT
    # ========================
    # measurement - prevalent
    'rotation': {
        'Explanation': "In the previously provided chest X-ray, use the spinous processes as a reference "
                       "to draw an imaginary vertical line down the center of the vertebral bodies. "
                       "Measure the distance from the medial end of each clavicle to the nearest point on this vertical line. "
                       "Compute the ratio of the shorter distance to the longer distance, round it to two decimal places. "
                       "Determine which range the measured value falls into and select the correct option. Choose only one.",
        'Simple': "Measure the horizontal distance (x-coordinate difference) from the medial end of each clavicle "
                  "to the nearest point on the vertical line through the spinous processes. "
                  "Then, compute the ratio of the shorter distance to the longer distance. "
                  "Round the result to two decimal places, and determine which range the measured value falls into "
                  "and select the correct option. Choose only one."
    },

    'cardiomegaly': {
        'Explanation': 'Measure the cardiothoracic ratio based on the image provided earlier. '
                       'Round the ratio to two decimal places, then determine which range it falls into '
                       'and select the correct option. Choose only one.',
        'Simple': "Use the criterion you selected to make the decision for the first question to measure the ratio. "
                  "Round the ratio to two decimal places, then determine which range it falls into "
                  "and select the correct option. Choose only one."
    },

    'mediastinal_widening': {
        'Explanation': "Measure the ratio by dividing the width of the mediastinum "
                       "by the width of the lungs at the same level. "
                       "Round the ratio to two decimal places, then determine which range it falls into "
                       "and select the correct option. Choose only one.",
        'Simple': "Use the refined criterion outlined earlier to measure the ratio."
                  "Round the ratio to two decimal places, then determine which range it falls into "
                  "and select the correct option. Choose only one.",
    },

    'carina_angle': {
        'Explanation': "Measure the angle of the carina based on the image that was provided earlier. "
                       "Round the angle to the nearest whole number, then determine which range it falls into "
                       "and select the correct option. Choose only one.",
        'Simple': "Then, use the criterion you selected to make the decision for the first question to measure the angle. "
                  "Round the value to the nearest whole number, then determine which range it falls into "
                  "and select the correct option. Choose only one."
    },

    'projection': {
        'Explanation': "Measure the ratio of the overlapping area between the scapula and the lung to the total scapular area "
                       "for both the right and left scapula based on the image that was provided earlier. "
                       "Round the ratio to two decimal places, then determine which range the value falls into "
                       "for both the right and left sides, and select the correct option, choosing one option for each.",
        'Simple': "Use the refined criterion outlined earlier to measure the ratio."
                  "Round the ratio to two decimal places, then determine which range it falls into "
                  "for both the right and left sides, and select the correct option, choosing one option for each.",

    },

    # measurement - our own metric
    'aortic_knob_enlargement': {
        'Explanation': "Measure the width of the aortic knob and the width of the trachea in the provided chest X-ray. "
                       "Calculate the ratio of the aortic knob width to the trachea width. "
                       "Round the ratio to two decimal places, then determine which range it falls into "
                       "and select the correct option. Choose only one.",
        'Simple': "Use the refined criterion outlined earlier to measure the ratio."
                  "Round the ratio to two decimal places, then determine which range it falls into "
                  "and select the correct option. Choose only one.",
    },
    'descending_aorta_enlargement': {
        'Explanation': "Measure the width of the descending aorta and the width of the trachea in the provided chest X-ray."
                       "Calculate the ratio of the descending aorta width to the trachea width. "
                       "Round the ratio to two decimal places, then determine which range it falls into "
                       "and select the correct option. Choose only one.",

        'Simple': "Use the refined criterion outlined earlier to measure the ratio."
                  "Round the ratio to two decimal places, then determine which range it falls into "
                  "and select the correct option. Choose only one.",
    },
    'descending_aorta_tortuous': {
        'Explanation': "Using only the thoracic aorta of the descending aorta, specifically the region at the upper part of the heart, "
                       "divide this region into five equal sections. For each section, calculate the coordinates on the top-left lung side of the division, "
                       "resulting in a total of 6 coordinates. Then, calculate the curvature at each of these coordinates "
                       "and compute the average curvature value. "
                       "Round the value to four decimal places, then determine which range it falls into "
                       "and select the correct option. Choose only one.",

        'Simple': "Use the refined criterion outlined earlier to measure the ratio."
                  "Round the ratio to four decimal places, then determine which range it falls into "
                  "and select the correct option. Choose only one.",

    }
}
prefix_final_q = "Based on the measurement results from the previous question"
final_question_per_dx = {
    'inclusion': f"{prefix_final_q}, is the entire thoracic cage - including the lung apices, "
                 "inner margins of the lateral ribs, and costophrenic angles (CPAs) - "
                 "fully visible in this chest X-ray without being cropped? "
                 "Options: (a) Yes, (b) No",

    'inspiration': f"{prefix_final_q}, what is the inspiration level of the chest X-ray? "
                   f"For a chest X-ray to have a good inspiration level, at least 9 right posterior ribs "
                   f"should be visible above the right hemidiaphragm in the mid-clavicular line. "
                   f"Options: (a) Good, (b) Poor",

    'trachea_deviation': f"{prefix_final_q}, is the trachea deviated in this chest X-ray? "
                         "Options: "
                         "(a) Yes "
                         "(b) No",

    'ascending_aorta_enlargement': f"{prefix_final_q}, does the ascending aorta appear enlarged in the chest X-ray? "
                                f"If the ascending aorta extends beyond the line connecting the inner boundary "
                                f"of the right lung and the right heart side, it indicates that the ascending aorta is enlarged. "
                                f"Options: (a) Yes, (b) No",

    # ============# ============# ============# ============# ============
    #                       MEASUREMENT
    # ============# ============# ============# ============# ============

    'rotation': f"{prefix_final_q}, was the patient rotated during the chest X-ray? "
                f"If the ratio is greater than 0.4, the patient was not rotated. "
                f"Options: (a) Yes, (b) No "
                "Also, report the calculated ratio using the following format: Value: [Value]. "
                "Do not include any explanations.",

    'projection': f"{prefix_final_q}, identify the view of the chest X-ray. "
                  f"If the ratio of both sides is less than 0.3, "
                  f"regard the scapulae as laterally retracted from the lung fields. "
                  f"Options: (a) PA (Posteroanterior), (b) AP (Anteroposterior)"
                  " Also, report the calculated ratio using the following format: "
                  "Value: Right - [Value], Left - [Value]. "
                  "Do not include any explanations.",

    'cardiomegaly': f"{prefix_final_q}, does this patient have cardiomegaly? "
                    f"For the AP view, the CTR of 0.55 or higher is considered indicative of cardiomegaly, "
                    f"while the PA view follows the standardized criteria."
                    f"Options: (a) Yes, (b) No "
                    "Also, report the calculated ratio using the following format: Value: [Value]. "
                    "Do not include any explanations.",

    'mediastinal_widening': f"{prefix_final_q}, does this patient have mediastinal widening? "
                            f"For the AP view, the ratio of 0.33 or higher indicates mediastinal widening, "
                            f"while for the PA view, the ratio of 0.28 or higher indicates mediastinal widening."
                            f"Options: (a) Yes, (b) No "
                            "Also, report the calculated ratio using the following format: Value: [Value]. Do not include any explanations.",

    'carina_angle': f"{prefix_final_q}, does this chest X-ray show a normal carina angle? "
                    f"The normal carina angle is between 40-80 degrees. "
                    f"Options: (a) Yes, (b) No "
                    "Also, report the calculated angle using the following format: Value: [Value]. "
                    "Do not include any explanations.",

    'aortic_knob_enlargement': f"{prefix_final_q}, does the aortic knob appear enlarged in the chest X-ray? "
                            f"If the ratio is 2.5 or higher, consider it enlarged. "
                            f"Options: (a) Yes, (b) No "
                            f"Also, report the calculated value using the following format: Value: [Value]. "
                            "Do not include any explanations.",

    'descending_aorta_enlargement': f"{prefix_final_q}, does the descending aorta appear "
                                 f"enlarged in the chest X-ray? "
                                 f"If the ratio is 2.5 or higher, consider it enlarged. "
                                 f"Options: (a) Yes, (b) No "
                                 f"Also, report the calculated value using the following format: Value: [Value]. "
                                 f"Do not include any explanations.",

    'descending_aorta_tortuous': f"{prefix_final_q}, is the descending aorta tortuous in the chest X-ray? "
                                 f"If the average curvature value is 0.0009 or higher, consider it tortuous. "
                                 f"Options: (a) Yes, (b) No "
                                 "Also, report the calculated value using the following format: Value: [Value]. "
                                 "Do not include any explanations.",
}

def mk_answer_q_dx(target_dx, dicom, label_measured):
    if target_dx in ['inclusion']:
        GT = '(a) Yes'
        for label in label_measured:
            if '_ex' in label:
                GT = '(b) No'

    elif target_dx in ['projection']:
        threshold = threshold_per_dx[target_dx]

        label_measured = np.array(label_measured).round(round_per_dx[target_dx])
        tolerance = map_tolerance_per_dx[target_dx]['tolerance']

        mask_large = (label_measured > threshold + tolerance)
        mask_small = (label_measured < threshold - tolerance)

        if mask_large.sum() == len(label_measured):
            GT = '(b) AP (Anteroposterior)'
        elif mask_small.sum() == len(label_measured):
            GT = '(a) PA (Posteroanterior)'
        else:
            GT = '(a) PA (Posteroanterior), (b) AP (Anteroposterior)'

    elif target_dx in ['inspiration']:
        if int(label_measured) >= threshold_per_dx[target_dx]:
            GT = '(a) Good'
        elif int(label_measured) < threshold_per_dx[target_dx]:
            GT = '(b) Poor'

    elif target_dx in ['ascending_aorta_enlargement']:
        threshold = threshold_per_dx[target_dx]
        if label_measured >= threshold:
            GT = '(a) Yes'
        else:
            GT = '(b) No'

    elif target_dx in ['trachea_deviation']:
        if label_measured in ['flat']:
            GT = '(b) No'
        else:
            GT = '(a) Yes'

    elif target_dx in ['rotation']:
        threshold = threshold_per_dx[target_dx]
        label_measured = round(label_measured, round_per_dx[target_dx])
        tolerance = map_tolerance_per_dx[target_dx]['tolerance']

        if label_measured > (threshold + tolerance):
            GT = '(b) No'
        elif label_measured < (threshold - tolerance):
            GT = '(a) Yes'
        else:
            GT = '(a) Yes, (b) No'


    elif target_dx in ['cardiomegaly', 'mediastinal_widening']:
        label_measured = round(label_measured, round_per_dx[target_dx])

        # Get view position using dataset config
        config = get_dataset_config()
        if config:
            view = config.get_view_position(dicom, target_dx)
        else:
            # Fallback to global mimic_meta for backward compatibility
            view = mimic_meta[mimic_meta['dicom_id'] == dicom]['ViewPosition'].values[0]
        
        threshold_per_view = threshold_per_dx[target_dx][view]

        tolerance = map_tolerance_per_dx[f'{target_dx}_{view}']['tolerance']

        if label_measured > (threshold_per_view + tolerance):
            GT = '(a) Yes'
        elif label_measured < (threshold_per_view - tolerance):
            GT = '(b) No'
        else:
            GT = '(a) Yes, (b) No'

    elif target_dx in ['carina_angle']:
        threshold = threshold_per_dx[target_dx]
        label_measured = round(label_measured, round_per_dx[target_dx])
        tolerance = map_tolerance_per_dx[target_dx]['tolerance']
        if (label_measured >= threshold[0]) and (label_measured <= threshold[-1]):
            GT = '(a) Yes'
        elif (label_measured < threshold[0] - tolerance) or (label_measured > threshold[-1] + tolerance):
            GT = '(b) No'
        else:
            GT = '(a) Yes, (b) No'

    elif target_dx in ['aortic_knob_enlargement', 'descending_aorta_enlargement',
                       'descending_aorta_tortuous']:
        threshold = threshold_per_dx[target_dx]
        label_measured = round(label_measured, round_per_dx[target_dx])
        tolerance = map_tolerance_per_dx[target_dx]['tolerance']

        if label_measured > (threshold + tolerance):
            GT = '(a) Yes'
        elif label_measured < (threshold - tolerance):
            GT = '(b) No'
        else:
            GT = '(a) Yes, (b) No'
    return GT

def get_option(number, distractor):
    def get_ordinal(n):
        suffix = 'th'
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            if n % 10 == 1:
                suffix = 'st'
            elif n % 10 == 2:
                suffix = 'nd'
            elif n % 10 == 3:
                suffix = 'rd'
        return str(n) + suffix

    result = []

    for i in range(1, number):
        option = get_ordinal(i)
        result.append(f"({chr(97 + i - 1)}) {option} image")
    result.append(f"({chr(97 + (number-1))}) {distractor}")
    return ', '.join(result)

def mk_options_q_bodypart(target_dx, dicom, segmask_base_dir, pitfall, num_options):
    target_dx_bodypart_path_lst = glob(f'{segmask_base_dir}/{target_dx}/*/{dicom}.png')
    target_dx_bodypart = [bodypart.split('/')[-2] for bodypart in target_dx_bodypart_path_lst]

    available_bodypart_path = []
    available_bodypart_lst = []
    exist_bodypart_path = glob(f'{segmask_base_dir}/*/*/{dicom}.png')
    for path in exist_bodypart_path:
        dx = path.split('/')[-3]
        bodypart = path.split('/')[-2]
        if dx != target_dx:
            if target_dx in ['cardiomegaly', 'trachea_deviation', 'carina_angle', 'aortic_knob_enlargement',
                             'ascending_aorta_enlargement', 'descending_aorta_enlargement', 'descending_aorta_tortuous']:
                if (bodypart not in target_dx_bodypart) and (bodypart not in available_bodypart_lst) and (bodypart not in ['mediastinum']):
                    available_bodypart_path.append(path)
                    available_bodypart_lst.append(bodypart)
            else:
                if (bodypart not in target_dx_bodypart) and (bodypart not in available_bodypart_lst):
                    available_bodypart_path.append(path)
                    available_bodypart_lst.append(bodypart)

    if pitfall in ['basic']:
        random_bodypart_path = random.sample(available_bodypart_path, (num_options - len(target_dx_bodypart) - 1))
        options = get_option(num_options, 'None of the above')
        bodypart_path_lst = target_dx_bodypart_path_lst + random_bodypart_path
        random.shuffle(bodypart_path_lst)
        answer_lst = []
        for target_dx_bodypart_path in target_dx_bodypart_path_lst:
            answer_idx = bodypart_path_lst.index(target_dx_bodypart_path)
            answer = options.split(', ')[answer_idx]
            answer_lst.append(answer)

        answer = ', '.join(sorted(answer_lst))

        bodypart_path_lst = [path.replace(segmask_base_dir, '') for path in bodypart_path_lst]
        return [options], [bodypart_path_lst], [answer]

    elif pitfall in ['two-round_none_included']:
        options_1st = get_option(num_options, 'Need new options')
        bodypart_path_lst_1st = random.sample(available_bodypart_path, (num_options - 1))
        random.shuffle(bodypart_path_lst_1st)

        options_list = options_1st.split(', ')
        for option in options_list:
            option = option + '.'
            if 'Need new options' in option:
                answer_1st = option

        available_bodypart_path_2nd = set(available_bodypart_path).difference(bodypart_path_lst_1st)
        options_2nd = get_option(num_options, 'None of the above')
        bodypart_path_lst_2nd = random.sample(available_bodypart_path_2nd,
                                              (num_options - len(target_dx_bodypart) - 1))
        bodypart_path_lst_2nd += target_dx_bodypart_path_lst
        random.shuffle(bodypart_path_lst_2nd)

        answer_lst = []
        for target_dx_bodypart_path in target_dx_bodypart_path_lst:
            answer_idx = bodypart_path_lst_2nd.index(target_dx_bodypart_path)
            answer = options_2nd.split(', ')[answer_idx]
            answer_lst.append(answer)

        answer_2nd = ', '.join(sorted(answer_lst))

        bodypart_path_lst_1st = [path.replace(segmask_base_dir, '') for path in bodypart_path_lst_1st]
        bodypart_path_lst_2nd = [path.replace(segmask_base_dir, '') for path in bodypart_path_lst_2nd]

        return [options_1st, options_2nd], [bodypart_path_lst_1st, bodypart_path_lst_2nd], [answer_1st, answer_2nd]

    elif pitfall in ['two-round_partial_inclusion']:
        idx_partial = len(target_dx_bodypart_path_lst) // 2

        options_1st = get_option(num_options, 'Need new options')
        bodypart_path_lst_1st = random.sample(available_bodypart_path, (num_options - len(target_dx_bodypart_path_lst[:idx_partial]) - 1))
        bodypart_path_lst_1st += target_dx_bodypart_path_lst[:idx_partial]
        random.shuffle(bodypart_path_lst_1st)

        answer_lst = []
        for target_dx_bodypart_path in target_dx_bodypart_path_lst[:idx_partial]:
            answer_idx = bodypart_path_lst_1st.index(target_dx_bodypart_path)
            answer = options_1st.split(', ')[answer_idx]
            answer_lst.append(answer)

        options_list = options_1st.split(', ')
        for option in options_list:
            option = option + '.'
            if 'Need new options' in option:
                answer_lst.append(option)

        answer_1st = ', '.join(sorted(answer_lst))

        available_bodypart_path_2nd = set(available_bodypart_path).difference(bodypart_path_lst_1st)

        options_2nd = get_option(num_options, 'None of the above')

        bodypart_path_lst_2nd = random.sample(available_bodypart_path_2nd,
                                              (num_options - len(target_dx_bodypart_path_lst[idx_partial:]) - 1))
        bodypart_path_lst_2nd += target_dx_bodypart_path_lst[idx_partial:]
        random.shuffle(bodypart_path_lst_2nd)

        answer_lst = []
        for target_dx_bodypart_path in target_dx_bodypart_path_lst[idx_partial:]:
            answer_idx = bodypart_path_lst_2nd.index(target_dx_bodypart_path)
            answer = options_2nd.split(', ')[answer_idx]
            answer_lst.append(answer)
        answer_2nd = ', '.join(sorted(answer_lst))

        bodypart_path_lst_1st = [path.replace(segmask_base_dir, '') for path in bodypart_path_lst_1st]
        bodypart_path_lst_2nd = [path.replace(segmask_base_dir, '') for path in bodypart_path_lst_2nd]

        return [options_1st, options_2nd], [bodypart_path_lst_1st, bodypart_path_lst_2nd], [answer_1st, answer_2nd]

def mk_options_q_criteria(target_dx, criteria_selection_list, num_options, pitfall):
    def mk_options(criteria_lst, target_criteria):
        options = ', '.join([f"({chr(97 + i)}) {criteria}" for i, criteria in enumerate(criteria_lst)])
        options_list = options.split('., ')

        for option in options_list:
            option = option + '.'
            if target_criteria in option:
                answer = option

        return options, answer

    if pitfall in ['two-round']:
        available_dx = list(set(criteria_selection_list.keys()) - set([target_dx]))
        random_dx_1st = random.sample(available_dx, num_options - 1)
        random_criteria_1st = [criteria_selection_list[key] for key in random_dx_1st]
        criteria_lst_1st = random_criteria_1st + ['Need new options']

        options_1st, answer_1st = mk_options(criteria_lst_1st, 'Need new options')


        available_dx_2nd = list(set(criteria_selection_list.keys()) - set([target_dx] + random_dx_1st))
        random_dx_2nd = random.sample(available_dx_2nd, num_options - 2)
        random_criteria_2nd = [criteria_selection_list[key] for key in random_dx_2nd]
        target_criteria = criteria_selection_list[target_dx]

        criteria_lst_2nd = [target_criteria] + random_criteria_2nd
        random.shuffle(criteria_lst_2nd)
        criteria_lst_2nd.append('None of the above')

        options_2nd, answer_2nd = mk_options(criteria_lst_2nd, target_criteria)
        options = [options_1st, options_2nd]
        answers = [answer_1st, answer_2nd]

        return options, answers

    elif pitfall in ['basic']:
        available_dx = list(set(criteria_selection_list.keys()) - set([target_dx]))
        random_dx = random.sample(available_dx, (num_options-2))

        random_criteria = [criteria_selection_list[key] for key in random_dx]
        target_criteria = criteria_selection_list[target_dx]

        criteria_lst = [target_criteria] + random_criteria
        random.shuffle(criteria_lst)
        criteria_lst.append('None of the above')

        options = ', '.join([f"({chr(97+i)}) {criteria}" for i, criteria in enumerate(criteria_lst)])
        options_list = options.split('., ')

        for option in options_list:
            option = option + '.'
            if target_criteria in option:
                answer = option
        return [options], [answer]

def mk_options_q_measurement(target_dx, dicom, label_measured, num_options):
    if target_dx in ['inclusion']:
        map_label_measured2answer = {
            'apex_both_in': 'Right apex: Included, Left apex: Included',
            'apex_both_ex': 'Right apex: Excluded, Left apex: Excluded',
            'apex_r_in_l_ex': 'Right apex: Included, Left apex: Excluded',
            'apex_r_ex_l_in': 'Right apex: Excluded, Left apex: Included',

            'side_both_in': 'Right rib edge: Included, Left rib edge: Included',
            'side_both_ex': 'Right rib edge: Excluded, Left rib edge: Excluded',
            'side_r_in_l_ex': 'Right rib edge: Included, Left rib edge: Excluded',
            'side_r_ex_l_in': 'Right rib edge: Excluded, Left rib edge: Included',

            'bottom_both_in': 'Right costophrenic angle: Included, '
                              'Left costophrenic angle: Included',
            'bottom_both_ex': 'Right costophrenic angle: Excluded, '
                              'Left costophrenic angle: Excluded',
        }

        def generate_options(input_list, map_label_measured2answer, num_samples):
            # Desired prefixes
            prefixes = ['apex_', 'side_', 'bottom_']

            # Create a list of possible values for each prefix
            possible_values = defaultdict(list)
            for label_measured in map_label_measured2answer.keys():
                position = label_measured.split('_')[0]
                possible_values[f'{position}_'].append(label_measured)


            # To store the unique lists
            unique_lists = set()

            # Keep generating until we have the required number of unique lists
            while len(unique_lists) < num_samples:
                # Randomly select one value from each prefix group
                new_list = [random.choice(possible_values[prefix]) for prefix in prefixes]

                # Ensure the generated list is not the same as the input list
                if new_list != input_list:
                    unique_lists.add(tuple(new_list))  # Store as a tuple since lists are not hashable

            # Convert the sets back to lists
            return [list(unique_list) for unique_list in unique_lists]

        def sort_based_on_prefix(original_list, desired_order):
            # Extract the prefix from each string and sort accordingly
            return sorted(original_list, key=lambda x: desired_order.index(x.split('_')[0]))

        sorted_labels = sort_based_on_prefix(label_measured, ['apex', 'side', 'bottom'])

        option_lst = [sorted_labels] + generate_options(sorted_labels, map_label_measured2answer, (num_options - 1))
        random.shuffle(option_lst)

        options = ''
        for i, option in enumerate(option_lst):
            option_str = ', '.join([map_label_measured2answer[label] for label in option])
            options += f"({chr(97 + i)}) {option_str}, "
            if option == sorted_labels:
                answer = f"({chr(97 + i)}) {option_str}"

        return options, answer

    elif target_dx in ['inspiration']:
        def get_adjacent_samples(candidates, target, sample_size):
            target_index = candidates.index(target)

            # Randomly determine where in the final list the target range should appear
            available_positions = list(range(sample_size))
            random.shuffle(available_positions)
            target_position = available_positions[0]  # Choose a random position

            # Compute the starting index based on the random target_position
            start_idx = target_index - target_position
            end_idx = start_idx + sample_size

            # Adjust start/end indices to keep within bounds
            if start_idx < 0:
                start_idx = 0
                end_idx = min(sample_size, len(candidates))
            elif end_idx > len(candidates):
                end_idx = len(candidates)
                start_idx = max(0, end_idx - sample_size)

            return candidates[start_idx:end_idx]

        candidates = [5, 6, 7, 8, 9, 10, 11]
        option_lst = get_adjacent_samples(candidates, label_measured, num_options)

        options = ''
        for i, option in enumerate(option_lst):
            options += f"({chr(97 + i)}) {option}, "
            if option == label_measured:
                answer = f"({chr(97 + i)}) {option}"

        return options, answer

    elif target_dx in ['trachea_deviation']:
        map_measured2label = {
            'flat': 'Not deviated',
            'right': 'Deviated to the right',
            'left': 'Deviated to the left',
            'right&left': 'Deviated to the right and then left',
            'left&right': 'Deviated to the left and then right',
            }

        option_lst = list(map_measured2label.values())
        random.shuffle(option_lst)

        options = ''
        for i, option in enumerate(option_lst):
            options += f"({chr(97+i)}) {option}, "
            if option == map_measured2label[label_measured]:
                answer = f"({chr(97+i)}) {option}"

        return options, answer

    elif target_dx in ['ascending_aorta_enlargement']:
        threshold = threshold_per_dx[target_dx]  # .round(round_per_dx[target_dx])

        if label_measured >= threshold:
            answer = '(a) Extends beyond the line'
        else:
            answer = '(b) Does not extend beyond the line'

        options = '(a) Extends beyond the line (b) Does not extend beyond the line'

        return options, answer

    else:
        def generate_multiple_choices(label_measured, margin, tolerance, num_choices,
                                      min_limit, max_limit, round_):
            answer_range = [round(label_measured - tolerance, round_),
                            round(label_measured + tolerance, round_)]

            if answer_range[0] <= min_limit:
                answer_range[0] = min_limit

                choices = [answer_range]
                for _ in range(num_choices):
                    cur_max = choices[-1][-1]
                    option = [round((cur_max + margin), round_),
                              round((cur_max + margin) + (tolerance * 2), round_)]
                    if option[-1] >= max_limit:
                        option[-1] = max_limit
                        choices.append(option)
                        break
                    else:
                        choices.append(option)

            elif answer_range[-1] >= max_limit:
                answer_range[-1] = max_limit

                choices = [answer_range]
                for _ in range(num_choices):
                    cur_min = choices[-1][0]
                    option = [round((cur_min - margin) - (tolerance * 2), round_),
                              round(cur_min - margin, round_)]
                    if option[0] <= min_limit:
                        option[0] = min_limit
                        choices.append(option)
                        break
                    else:
                        choices.append(option)

            else:
                choices_lower = [answer_range]
                for _ in range(num_choices):
                    cur_min = choices_lower[-1][0]
                    option = [round((cur_min - margin) - (tolerance * 2), round_),
                              round(cur_min - margin, round_)]
                    if option[0] <= min_limit:
                        option[0] = min_limit
                        choices_lower.append(option)
                        break
                    else:
                        choices_lower.append(option)

                choices_upper = [answer_range]
                for _ in range(num_choices):
                    cur_max = choices_upper[-1][-1]
                    option = [round((cur_max + margin), round_),
                              round((cur_max + margin) + (tolerance * 2), round_)]
                    if option[-1] >= max_limit:
                        option[-1] = max_limit
                        choices_upper.append(option)
                        break
                    else:
                        choices_upper.append(option)

                choices = choices_upper + choices_lower

            answer_ = f'[{answer_range[0]} - {answer_range[-1]}]'
            choices_ = []
            for choice in choices:
                choice_ = f"[{choice[0]} - {choice[-1]}]"
                if choice_ not in choices:
                    choices_.append(choice_)

            choices_ = set(choices_)
            choices_ = sorted(choices_, key=lambda x: eval(x.split(' - ')[0][1:].strip()))

            if len(choices_) <= num_choices:
                return choices_, answer_

            else:
                target_index = choices_.index(answer_)

                # Randomly determine where in the final list the target range should appear
                available_positions = list(range(num_choices))
                random.shuffle(available_positions)
                target_position = available_positions[0]  # Choose a random position

                # Compute the starting index based on the random target_position
                start_idx = target_index - target_position
                end_idx = start_idx + num_choices

                # Adjust start/end indices to keep within bounds
                if start_idx < 0:
                    start_idx = 0
                    end_idx = min(num_choices, len(choices_))
                elif end_idx > len(choices_):
                    end_idx = len(choices_)
                    start_idx = max(0, end_idx - num_choices)

                return choices_[start_idx:end_idx], choices_[target_index]

        view = mimic_meta[mimic_meta['dicom_id'] == dicom]['ViewPosition'].values[0]
        key_dx = f"{target_dx}_{view}" if target_dx in ['cardiomegaly', 'mediastinal_widening'] else target_dx
        tolerance = map_tolerance_per_dx[key_dx]['tolerance']
        min_limit = map_tolerance_per_dx[key_dx]['min_limit']
        max_limit = map_tolerance_per_dx[key_dx]['max_limit']
        margin = map_tolerance_per_dx[key_dx]['margin']
        round_ = map_tolerance_per_dx[key_dx]['round']

        if target_dx in ['projection']:
            label_measured = np.array(label_measured).round(round_per_dx[target_dx])
            option_lst_r, answer_range_r = generate_multiple_choices(label_measured[0], margin, tolerance, num_options, min_limit, max_limit, round_)
            option_lst_l, answer_range_l = generate_multiple_choices(label_measured[-1], margin, tolerance, num_options, min_limit, max_limit, round_)

            options = ''
            for i, option_r in enumerate(option_lst_r):
                options += f"({chr(97 + i)}) Right: {option_r}, "
                if option_r == answer_range_r:
                    answer = f"({chr(97 + i)}) Right: {option_r}, "

            for j, option_l in enumerate(option_lst_l):
                options += f"({chr(97 + i + j + 1)}) Left: {option_l}, "
                if option_l == answer_range_l:
                    answer += f"({chr(97 + i + j + 1)}) Left: {option_l}"

        else:
            label_measured = round(label_measured, round_per_dx[target_dx])
            option_lst, answer_range = generate_multiple_choices(label_measured, margin, tolerance, num_options, min_limit, max_limit, round_)
            options = ''
            for i, option in enumerate(option_lst):
                options += f"({chr(97 + i)}) {option}, "
                if option == answer_range:
                    answer = f"({chr(97 + i)}) {option}"

        return options, answer

def return_cxr_path(mimic_meta, dicom, dataset_name='mimic-cxr-jpg'):
    """Get CXR path - kept for backward compatibility, use DatasetConfig instead"""
    config = get_dataset_config()
    if config:
        return config.get_cxr_path(dicom)
    elif dataset_name == 'nih-cxr14':
        return f'{dicom}.png'
    else:
        sid = mimic_meta[mimic_meta['dicom_id'] == dicom]['study_id'].values[0]
        pid = mimic_meta[mimic_meta['dicom_id'] == dicom]['subject_id'].values[0]
        return f'p{str(pid)[:2]}/p{pid}/s{sid}/{dicom}.jpg'

def qa_init_question(stage, dicom, target_dx, measured_value, mimic_meta, save_dir_gold_qa):
    cxr_path_lst = [return_cxr_path(mimic_meta, dicom)]

    guidance_init_q = "Please base your decision on the most established and clearly defined diagnostic criterion " \
                      "used in standard radiologic references. Avoid relying on indirect factors, " \
                      "which, while potentially relevant, are not the direct and primary criteria. " \
                      "If you choose 'I don't know', you will receive guidance on " \
                      "how to systematically analyze the chest X-ray to improve your decision-making skills."

    # Get view position using dataset config
    config = get_dataset_config()
    if config:
        view = config.get_view_position(dicom, target_dx)
    else:
        view = mimic_meta[mimic_meta['dicom_id'] == dicom]['ViewPosition'].values[0]
    if target_dx in ['cardiomegaly', 'mediastinal_widening']:
        if view in ['AP']:
            view_ = 'AP (Anteroposterior)'
        else:
            view_ = 'PA (Posteroanterior)'

        question_init_dx = f"{guidance_init_q} {initial_question_per_dx[target_dx]} The chest X-ray was taken in the {view_} view."
    else:
        question_init_dx = f"{guidance_init_q} {initial_question_per_dx[target_dx]}"

    answer_dx = mk_answer_q_dx(target_dx, dicom, measured_value)

    qa_dict = {'question': question_init_dx,
               'answer': answer_dx,
               'img_path': cxr_path_lst}
    # # 'measured_value': measured_value,
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'inference_type': 'reasoning',
    #                'stage': 'init',

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

    return answer_dx

def qa_criteria(stage, dicom, target_dx, num_options, pitfall, save_dir_gold_qa):
    # ====================================================================
    #                       Question - Criteria
    # ====================================================================
    options_criteria, answers_criteria = mk_options_q_criteria(target_dx, criteria_per_dx, num_options, pitfall)
    questions_criteria = []
    for idx, options_criteria_per_round in enumerate(options_criteria):
        if idx != len(options_criteria) - 1:
            question_criteria = "What criterion was used to make the decision for the first question? " \
                                f"Options: {options_criteria_per_round}. " \
                                f"If none of the options reflect a correct and " \
                                f"direct criterion, select 'Need new options.'." \
                                f"Do not choose an option just because it appears similar or somewhat related. " \
                                f"If you select 'Need new options.', additional options will be provided."
        else:
            question_criteria = f"What criterion was used to make the decision for the first question? " \
                                f"Options: {options_criteria_per_round}. " \
                                f"If none of the options reflect a correct and " \
                                f"direct criterion, select 'None of the above', and explain the criterion you applied."

        questions_criteria.append(question_criteria)

    qa_dict = {
               'question': questions_criteria,

               'answer': answers_criteria}
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'inference_type': 'reasoning',
    #                'stage': 'criteria',
    # 'option': options_criteria,

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/{pitfall}"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

def qa_custom_criteria(stage, dicom, target_dx, save_dir_gold_qa):
    answer_refined_criteria = "(a) Yes"
    question_refined_criteria = f"{refined_criteria_per_dx[target_dx]} " \
                                f"Options: (a) Yes (b) No. " \
                                f"If you choose '(b) No', you will receive guidance on " \
                                f"how to systematically analyze the chest X-ray to improve your decision-making skills. " \
                                f"Do not include any explanations."

    qa_dict = {
               'question': question_refined_criteria,
               'answer': answer_refined_criteria}

    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'inference_type': 'reasoning',
    #                'stage': 'custom_criteria',

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

def qa_bodypart(stage, inference_type, dicom, target_dx, num_options, pitfall, segmask_base_dir, save_dir_gold_qa):
    options_bodypart_lst, img_path_lst_bodypart_lst, answer_bodypart_lst = \
        mk_options_q_bodypart(target_dx, dicom, segmask_base_dir, pitfall=pitfall, num_options=num_options)

    dx_multi_bodyparts = ['inspiration', 'rotation', 'projection', 'cardiomegaly', 'trachea_deviation',
                         'mediastinal_widening', 'aortic_knob_enlargement', 'ascending_aorta_enlargement', 'descending_aorta_enlargement',]

    if target_dx in dx_multi_bodyparts:
        if target_dx in refined_criteria_per_dx.keys():
            question_bodypart = "In the following images, either a segmentation mask of a specific body part or a reference line is shown. " \
                                "Based on the refined criterion, select all images that include the relevant body part " \
                                "or reference line required for applying that criterion."

        else:
            question_bodypart = "In the following images, either a segmentation mask of a specific body part or a reference line is shown. " \
                                "Based on the selected criterion, select all images that include the relevant body part " \
                                "or reference line required for applying that criterion."

    else:  # inclusion, carina-angle, desc aorta tortuous
        if target_dx in refined_criteria_per_dx.keys():
            question_bodypart = "In the following images, either a segmentation mask of a specific body part or a reference line is shown. " \
                                "Based on the refined criterion, select the image that include the relevant body part " \
                                "or reference line required for applying that criterion."
        else:
            question_bodypart = "In the following images, either a segmentation mask of a specific body part or a reference line is shown. " \
                                "Based on the selected criterion, select the image that include the relevant body part " \
                                "or reference line required for applying that criterion."

    questions_bodypart = []
    if len(options_bodypart_lst) == 1:
        question_bodypart += "If you choose 'None of the above', please explain which body parts you used. "
        question_bodypart += f"Options: {options_bodypart_lst[0]} "
        questions_bodypart.append(question_bodypart)

    else:
        for idx, (options_bodypart, img_path_lst_bodypart, answer_bodypart) in enumerate(zip(options_bodypart_lst, img_path_lst_bodypart_lst, answer_bodypart_lst)):
            if idx == 0:
                if target_dx in dx_multi_bodyparts:
                    question_bodypart += "If only some of the required body parts or reference lines are present, " \
                                         "select the corresponding image(s) and choose 'Need new options' to request the missing ones. " \
                                         "If none of the relevant body parts or reference lines are present, select only 'Need new options'. "
                else:
                    question_bodypart += "If none of the relevant body parts or reference lines are present, " \
                                         "select only 'Need new options'. "
            else:
                if target_dx in refined_criteria_per_dx.keys():
                    question_bodypart = "Additional options are now provided. Select any additional image(s) " \
                                        "that includes the relevant body part or reference line required " \
                                        "for applying the refined criterion. If you choose 'None of the above', " \
                                        "please explain which body parts you used. "
                else:
                    question_bodypart = "Additional options are now provided. Select any additional image(s) " \
                                        "that includes the relevant body part or reference line required " \
                                        "for applying the selected criterion. If you choose 'None of the above', " \
                                        "please explain which body parts you used. "
            question_bodypart += f"Options: {options_bodypart}"
            questions_bodypart.append(question_bodypart)

    qa_dict = {
               'question': questions_bodypart,

               'answer': answer_bodypart_lst,
               'img_path': img_path_lst_bodypart_lst}
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'inference_type': inference_type,
    #                'stage': 'bodypart',
    # 'option': options_bodypart_lst,

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/{pitfall}"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

def qa_measurement(stage, inference_type, dicom, target_dx, measured_value, num_options, save_dir_gold_qa):
    option, answer_measurement = mk_options_q_measurement(target_dx, dicom, measured_value, num_options=num_options)

    question_measurement = measurement_q_per_dx[target_dx]['Simple'] + " " + option

    qa_dict = {
               'question': question_measurement,
               'option': option,
               'answer': answer_measurement}
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'measured_value': measured_value,
    #                'inference_type': inference_type,
    #                'stage': 'measurement',

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

def qa_final(stage, dicom, target_dx, measured_value, answer_init, save_dir_gold_qa):
    question_final_dx = final_question_per_dx[target_dx]

    qa_dict = {
               'question': question_final_dx,
               'answer': answer_init
               }
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'measured_value': measured_value,
    #                'inference_type': 'reasoning',
    #                'stage': 'final',

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)


map_bodypart_fname2real = {
    'lung_both': 'Lungs',
    'clavicle_both': 'Clavicles',
    'scapular_both': 'Scapulae',
    'diaphragm_right': 'Right hemidiaphragm',
    'right_posterior_rib': 'Right posterior rib',
    'heart': 'Heart',
    'thoracic_width_heart': 'the maximal horizontal thoracic diameter',
    'thoracic_width_mw': 'The thoracic width at the same level with the mediastinum',
    'mediastinum': 'Mediastinum',
    'carina': 'Carina',
    'trachea': 'Trachea',

    'aortic_knob': 'Aortic knob',
    'ascending_aorta': 'Ascending aorta',
    'descending_aorta': 'Descending aorta',

    'borderline': 'the straight line connecting the inner boundary of the right lung and the right heart border',
    'midline': 'the straight line representing the spinous processes',
    'midclavicularline': 'Mid-Clavicular line'
}

measurement_instruction_per_dx = {

    # the inner margins of the lateral ribs
    'inclusion': "To check whether the entire thoracic cage is shown in the chest X-ray, "
                 "look to see if three important parts are visible: "
                 "the tops of the lungs (lung apices), the inner edges of the side ribs, "
                 "and the costophrenic angles (CPAs), which are the corners at the bottom of the lungs. "
                 "In the image, a colored point has been placed at each of these areas: red for the lung apices, "
                 "green for the inner rib edges, and blue for the CPAs. "
                 "Keep in mind: the point doesn't always mark the exact part of the body. "
                 "If the part is visible in the image, the point shows its actual location. "
                 "If it's missing from the image (for example, if the top of the lung is cut off), "
                 "the point just shows the general level where it should have appeared. "
                 "If a part is excluded, that means the body part isn't visible at the point's location "
                 "- it was cut off in the X-ray. "
                 "Examine each point and decide whether the corresponding body part is visible in the image. "
                 "Then, select the appropriate option based on your assessment.",


    'inspiration': 'To assess the level of inspiration, draw an imaginary vertical line '
                   'from the midpoint of the clavicle (mid-clavicular line). '
                   'Then, count how many right posterior ribs intersect the right hemidiaphragm along this line. '
                   'In the provided image, the mid-clavicular line and the right posterior rib '
                   'intersecting the right hemidiaphragm along the line are marked. '
                   'Look at the image and count which rib is intersecting. '
                   'Based on your assessment, select the appropriate option.',

    'rotation': "To assess the rotation of the patient during an X-ray, "
                "check whether the spinous processes are equidistant from the medial ends of the clavicles. "
                "In the provided image, the points corresponding to the medial ends of the clavicles "
                "and their coordinate values are marked. Additionally, a straight line "
                "representing the spinous processes is given, with its slope and intercept provided. "
                "This line is defined in a way that for any given y-coordinate, "
                "the corresponding x-coordinate can be determined using the slope and intercept. "
                "To measure the rotation: "
                "1.	For each medial clavicle point, use its y-coordinate to determine the corresponding x-coordinate on the spinous process line. "
                "2.	Compute the difference between this x-coordinate and the original x-coordinate of the medial clavicle point to obtain the distance. "
                "3.	Compare the two distances and determine the ratio of the shorter distance to the longer distance. "
                "4.	Round the result to two decimal places, check which range it falls into, and select the correct option.",

    'projection': "To assess the projection, check whether the scapulae are laterally retracted "
                  "or overlapping with the lung fields. "
                  "In the provided image, segmentation masks for the left and right scapulae are drawn. "
                  "The overlapping regions between the scapulae and the lung fields are highlighted in purple, "
                  "while the remaining scapular regions are marked in red. "
                  "Each mask also displays numerical values indicating the overlapping area and the total scapular area. "
                  "If there is no overlapping region, no purple markings are displayed."
                  "To determine whether the scapulae are retracted or overlapping, "
                  "calculate the ratio of the overlapping area between the scapula and the lung "
                  "to the total scapular area for both the right and left scapula. "
                  "Round the ratio to two decimal places, then determine which range it falls into "
                  "for both the right and left sides, and select the correct option, "
                  "choosing one option for each.",

    'cardiomegaly': "To assess cardiomegaly, calculate the cardiothoracic ratio, "
                    "which is the ratio of the maximal horizontal cardiac diameter "
                    "to the maximal horizontal thoracic diameter. "
                    "In the provided image, the x-coordinates for measuring cardiac width and thoracic width "
                    "are marked, along with lines representing both measurements. "
                    "The coordinates and lines associated with the heart are highlighted in red, "
                    "while those related to the lungs are highlighted in blue. "
                    "Round the ratio to two decimal places, then determine which range it falls into "
                    "and select the correct option. Choose only one.",

    'mediastinal_widening': "To assess mediastinal widening, measure the mediastinal width and the thoracic width "
                            "at the same level, then calculate their ratio by dividing the mediastinal width by the thoracic width. "
                            "In the provided image, the x-coordinates for measuring mediastinal width and thoracic width are marked, "
                            "along with lines representing both measurements. "
                            "The coordinates and lines associated with the mediastinum are highlighted in red, "
                            "while those related to the lungs are highlighted in blue. "
                            "Round the ratio to two decimal places, then determine which range it falls into "
                            "and select the correct option. Choose only one.",

    'carina_angle': "To assess whether the carina angle is normal, measure the angle between the left and right main bronchi. "
                    "In the provided image, the central point at the carina is marked as B, "
                    "the right main bronchus as A, and the left main bronchus as C, "
                    "with their respective coordinate values also indicated. "
                    "Use these points to determine the angle formed between the two bronchi. "
                    "Round the value to the nearest whole number, then determine which range it falls into "
                    "and select the correct option. Choose only one.",

    'trachea_deviation': "To assess tracheal deviation, use the spinous processes as a reference "
                         "to draw an imaginary straight line down the center of the vertebral bodies "
                         "and evaluate whether the trachea aligns with this line. "
                         "In the provided image, the trachea segmentation mask and nine points along the line are marked. "
                         "For each point, determine whether the trachea is on, deviated toward the left (left lung side), "
                         "or deviated toward the right (right lung side) of the point. "
                         "The final label is determined by majority vote. If multiple labels share the highest count, "
                         "assign the label based on the order in which the majority count was first reached. "
                         "Based on your assessment, select the appropriate option.",

    'aortic_knob_enlargement': "To assess aortic knob enlargement, measure the maximum width of the aortic knob and the median width of the trachea, "
                            "then calculate their ratio by dividing the aortic knob width by the trachea width. "
                            "In the provided image, the x-coordinates for measuring the trachea and aortic knob widths are marked, "
                            "along with lines representing both measurements. "
                            "The coordinates and lines associated with the aortic knob are highlighted in red, "
                            "while those related to the trachea are highlighted in blue. "
                            "Round the ratio to two decimal places, then determine the corresponding range "
                            "and select the appropriate option. Choose only one.",

    'ascending_aorta_enlargement': "To assess ascending aorta enlargement, determine whether the ascending aorta extends "
                                "beyond an imaginary straight line connecting the inner boundary of the right lung "
                                "and the right heart border. In the provided image, the ascending aorta segmentation mask "
                                "and this reference line are marked. Select the appropriate label "
                                "based on your measurement. Choose only one.",

    'descending_aorta_enlargement': "To assess descending aorta enlargement, measure the maximum width of the descending aorta "
                                 "and the median width of the trachea, then calculate their ratio by dividing "
                                 "the descending aorta width by the trachea width. "
                                 "In the provided image, the x-coordinates for measuring the trachea and descending aorta widths "
                                 "are marked, along with lines representing both measurements. "
                                 "The coordinates and lines associated with the descending aorta are highlighted in red, "
                                 "while those related to the trachea are highlighted in blue. "
                                 "Round the ratio to two decimal places, then determine the corresponding range "
                                 "and select the appropriate option. Choose only one.",

    'descending_aorta_tortuous': "To assess descending aorta tortuosity, focus on the thoracic portion of the descending aorta, specifically the region at the upper part of the heart. Divide this region into five equal sections and determine the coordinates at the top-left lung side of each division, resulting in six total coordinates. "
                                 "Calculate the curvature at each of these six coordinates using finite difference methods: "
                                 "- The first and last points use forward and backward differences, respectively. "
                                 "- The middle points use central differences for higher accuracy. "
                                 "Compute the average curvature across all six points to quantify tortuosity. "
                                 "In the provided images, these six coordinates are marked. "
                                 "Round the ratio to four decimal places, then determine which range it falls into "
                                 "and select the correct option. Choose only one.",
}

def mk_qa_guidance_bodypart(target_dx, dicom, segmask_base_dir, num_options):
    target_dx_bodypart_path_lst = glob(f'{segmask_base_dir}/{target_dx}/*/{dicom}.png')
    target_dx_bodypart = [bodypart.split('/')[-2] for bodypart in target_dx_bodypart_path_lst]

    available_bodypart_path = []
    available_bodypart_lst = []
    exist_bodypart_path = glob(f'{segmask_base_dir}/*/*/{dicom}.png')
    for path in exist_bodypart_path:
        dx = path.split('/')[-3]
        bodypart = path.split('/')[-2]
        if dx != target_dx:
            if (bodypart not in target_dx_bodypart) and (bodypart not in available_bodypart_lst):
                available_bodypart_path.append(path)
                available_bodypart_lst.append(bodypart)

    random_bodypart_path = random.sample(available_bodypart_path, (num_options - len(target_dx_bodypart) - 1))
    options = get_option(num_options, 'None of the above')
    bodypart_path_lst = target_dx_bodypart_path_lst + random_bodypart_path
    random.shuffle(bodypart_path_lst)
    question_lst, answer_lst = [], []
    for idx, bodypart_path in enumerate(bodypart_path_lst):
        if bodypart_path in target_dx_bodypart_path_lst:
            bodypart_name = bodypart_path.split('/')[-2]
            bodypart_placeholder = map_bodypart_fname2real[bodypart_name]

            if len(question_lst) == 0:
                question = 'Among the following images, each image either contains a segmentation mask ' \
                           'highlighting a specific body part or a reference line necessary for decision. ' \
                           f'Which image represents {bodypart_placeholder.lower()}?  Options: {options}.'
            else:
                question = f"Continuing from the previous question, " \
                           f"which image corresponds to {bodypart_placeholder.lower()}?  Options: {options}."

            question_lst.append(question)

            answer = options.split(', ')[idx]
            answer_lst.append(answer)

    bodypart_path_lst = [path.replace(segmask_base_dir, '') for path in bodypart_path_lst]
    return question_lst, answer_lst, options, bodypart_path_lst

def qa_bodypart_guidance(stage, dicom, target_dx, segmask_base_dir, num_options, save_dir_gold_qa):
    q_lst_bodypart, a_lst_bodypart, options_bodypart, img_lst_bodypart = mk_qa_guidance_bodypart(target_dx, dicom,
                                                                                                 segmask_base_dir, num_options)

    qa_dict = {
               'question': q_lst_bodypart,

               'answer': a_lst_bodypart,
               'img_path': img_lst_bodypart}
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'inference_type': 'guidance',
    #                'stage': 'bodypart',
    # 'option': options_bodypart,

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

def qa_measurement_guidance(stage, dicom, target_dx, pnt_on_cxr_base_dir, measured_value, num_options, save_dir_gold_qa):
    options_measurement, answer_measurement = mk_options_q_measurement(target_dx, dicom, measured_value, num_options)

    question_measurement = measurement_instruction_per_dx[target_dx] + " " + options_measurement
    img_path_lst = [f"/{target_dx}/{dicom}.png"]

    qa_dict = {
               'question': question_measurement,

               'answer': answer_measurement,
               'img_path': img_path_lst}
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'inference_type': 'guidance',
    #                'stage': 'measurement',
    #                'measured_value': measured_value,
    # 'option': options_measurement,

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

prefix_final_q = "Based on the measurement results from the previous question"
final_question_per_dx_guidance = {
    'inclusion': f"{prefix_final_q}, is the entire thoracic cage - including the lung apices, "
                 "inner margins of the lateral ribs, and costophrenic angles (CPAs) - "
                 "fully visible in this chest X-ray without being cropped? "
                 "Options: (a) Yes, (b) No",

    'inspiration': f"{prefix_final_q}, what is the inspiration level of the chest X-ray? "
                   f"For a chest X-ray to have a good inspiration level, at least 9 right posterior ribs "
                   f"should be visible above the right hemidiaphragm in the mid-clavicular line. "
                   f"Options: (a) Good, (b) Poor",

    'trachea_deviation': f"{prefix_final_q}, is the trachea deviated in this chest X-ray? "
                         "Options: "
                         "(a) Yes "
                         "(b) No "
                         "(c) I don't know ",

    'ascending_aorta_enlargement': f"{prefix_final_q}, does the ascending aorta appear enlarged in the chest X-ray? "
                                f"If the ascending aorta extends beyond the line connecting the inner boundary "
                                f"of the right lung and the right heart side, it indicates that the ascending aorta is enlarged. "
                                f"Options: (a) Yes, (b) No",

    # ============# ============# ============# ============# ============
    #                       MEASUREMENT
    # ============# ============# ============# ============# ============

    'rotation': f"{prefix_final_q}, was the patient rotated during the chest X-ray? "
                f"If the ratio is greater than 0.4, the patient was not rotated. "
                f"Options: (a) Yes, (b) No "
                "Also, report the calculated ratio using the following format: Value: [Value]. "
                "Do not include any explanations.",

    'projection': f"{prefix_final_q}, identify the view of the chest X-ray. "
                  f"If the ratio of both sides is less than 0.3, "
                  f"regard the scapulae as laterally retracted from the lung fields. "
                  f"Options: (a) PA (Posteroanterior), (b) AP (Anteroposterior)"
                  " Also, report the calculated ratio using the following format: "
                  "Value: Right - [Value], Left - [Value]. "
                  "Do not include any explanations.",

    'cardiomegaly': f"{prefix_final_q}, does this patient have cardiomegaly? "
                    f"A cardiothoracic ratio (CTR) of 0.55 or higher is considered indicative of "
                    f"cardiomegaly on AP (anteroposterior) view chest radiographs, "
                    f"whereas a threshold of 0.50 is used for PA (posteroanterior) views. "
                    f"Options: (a) Yes, (b) No "
                    "Also, report the calculated ratio using the following format: Value: [Value]. "
                    "Do not include any explanations.",

    'mediastinal_widening': f"{prefix_final_q}, does this patient have mediastinal widening? "
                            f"For the AP view, the ratio of 0.33 or higher indicates mediastinal widening, "
                            f"while for the PA view, the ratio of 0.28 or higher indicates mediastinal widening."
                            f"Options: (a) Yes, (b) No "
                            "Also, report the calculated ratio using the following format: Value: [Value]. Do not include any explanations.",

    'carina_angle': f"{prefix_final_q}, does this chest X-ray show a normal carina angle? "
                    f"The normal carina angle is between 40-80 degrees. "
                    f"Options: (a) Yes, (b) No "
                    "Also, report the calculated angle using the following format: Value: [Value]. "
                    "Do not include any explanations.",

    'aortic_knob_enlargement': f"{prefix_final_q}, does the aortic knob appear enlarged in the chest X-ray? "
                            f"If the ratio is 2.5 or higher, consider it enlarged. "
                            f"Options: (a) Yes, (b) No "
                            f"Also, report the calculated value using the following format: Value: [Value]. "
                            "Do not include any explanations.",

    'descending_aorta_enlargement': f"{prefix_final_q}, does the descending aorta appear "
                                 f"enlarged in the chest X-ray? "
                                 f"If the ratio is 2.5 or higher, consider it enlarged. "
                                 f"Options: (a) Yes, (b) No "
                                 f"Also, report the calculated value using the following format: Value: [Value]. "
                                 f"Do not include any explanations.",

    'descending_aorta_tortuous': f"{prefix_final_q}, is the descending aorta tortuous in the chest X-ray? "
                                 f"If the average curvature value is 0.0009 or higher, consider it tortuous. "
                                 f"Options: (a) Yes, (b) No "
                                 "Also, report the calculated value using the following format: Value: [Value]. "
                                 "Do not include any explanations.",
}

def qa_final_guidance(stage, dicom, target_dx, measured_value, save_dir_gold_qa):
    # Get view position using dataset config
    config = get_dataset_config()
    if config:
        view = config.get_view_position(dicom, target_dx)
    else:
        view = mimic_meta[mimic_meta['dicom_id'] == dicom]['ViewPosition'].values[0]
    
    question_final_dx = final_question_per_dx_guidance[target_dx]
    if target_dx in ['cardiomegaly', 'mediastinal_widening']:
        if view in ['AP']:
            view_ = 'AP (Anteroposterior)'
        else:
            view_ = 'PA (Posteroanterior)'
        question_final_dx = f"The chest X-ray was taken in the {view_} view. " \
                            f"{final_question_per_dx[target_dx]}."

        if target_dx in ['cardiomegaly']:
            final_question_w_threshold = f"{prefix_final_q}, does this patient have cardiomegaly? "
            f"For the AP view, the CTR of 0.55 or higher is considered indicative of cardiomegaly, "
            f"while the PA view follows the standardized criteria (0.5)."
            f"Options: (a) Yes, (b) No "
            "Also, report the calculated ratio using the following format: Value: [Value]. "
            "Do not include any explanations."
            question_final_dx = f"The chest X-ray was taken in the {view_} view. " \
                                f"{final_question_w_threshold}."

    answer_dx = mk_answer_q_dx(target_dx, dicom, measured_value)

    qa_dict = {
               'question': question_final_dx,
               'answer': answer_dx
               }
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'measured_value': measured_value,
    #                'inference_type': 'guidance',
    #                'stage': 'final',
    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

def qa_init_after_guidance(stage, dicom, target_dx, measured_value, save_dir_gold_qa):

    cxr_path_lst = [return_cxr_path(mimic_meta, dicom)]

    guidance_init_q = "With the understanding you've gained from the previous steps, " \
                      "answer the following question accordingly."



    view = mimic_meta[mimic_meta['dicom_id'] == dicom]['ViewPosition'].values[0]
    if target_dx in ['cardiomegaly', 'mediastinal_widening']:
        if view in ['AP']:
            view_ = 'AP (Anteroposterior)'
        else:
            view_ = 'PA (Posteroanterior)'

        question_init_dx = f"{guidance_init_q} {initial_question_per_dx[target_dx]} The chest X-ray was taken in the {view_} view."
    else:
        question_init_dx = f"{guidance_init_q} {initial_question_per_dx[target_dx]}"


    answer_dx = mk_answer_q_dx(target_dx, dicom, measured_value)

    qa_dict = {
               'question': question_init_dx,
               'answer': answer_dx,
               'img_path': cxr_path_lst}
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'measured_value': measured_value,
    #                'inference_type': 'review',
    #                'stage': 'init',

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/basic"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

    return answer_dx

criteria_per_dx_after_guidance = {
    'inclusion': "By checking if the chest X-ray includes the lung apices, "
                 "inner margins of the lateral ribs, and costophrenic angles.",

    'rotation': "By checking if the spinous processes are equidistant "
                "from the medial ends of the clavicles.",

    'cardiomegaly': "By calculating the cardiothoracic ratio, "
                    "which is the ratio of the maximal horizontal cardiac diameter "
                    "to the maximal horizontal thoracic diameter.",

    'carina_angle': "By evaluating the angle of the carina.",

    'trachea_deviation': "By checking if the trachea is displaced to one side from the midline.",

    'inspiration': "By counting the number of right posterior ribs that "
                   "intersect the right hemidiaphragm along the midclavicular line.",

    'projection': "By checking if the scapulae were laterally retracted "
                  "or if they overlapped the lung fields. "
                  "Specifically, measure the ratio of the overlapping area "
                  "between the scapula and the lung to the total scapular area "
                  "for both the right and left scapula.",

    'mediastinal_widening': "By calculating the ratio of the mediastinal width to the thoracic width at the same level.",


    'aortic_knob_enlargement': "By calculating the ratio of the maximum width of the aortic knob to the median width of the trachea.",

    'ascending_aorta_enlargement': "By checking whether the ascending aorta extends "
                                "beyond an imaginary straight line connecting "
                                "the inner boundary of the right lung and the right heart border.",

    'descending_aorta_enlargement': "By calculating the ratio of the maximum width of the descending aorta to the median width of the trachea.",

    'descending_aorta_tortuous': "By calculating the average curvature across the extracted points "
                                 "along the contour of the descending aorta.",
}

def qa_criteria_after_guidance(stage, dicom, target_dx, num_options, pitfall, save_dir_gold_qa):
    # ====================================================================
    #                       Question - Criteria
    # ====================================================================
    options_criteria, answers_criteria = mk_options_q_criteria(target_dx, criteria_per_dx_after_guidance,
                                                               num_options, pitfall)
    questions_criteria = []
    for idx, options_criteria_per_round in enumerate(options_criteria):
        if idx != len(options_criteria) - 1:
            question_criteria = "What criterion was used to make the decision for the first question? " \
                                f"Options: {options_criteria_per_round}. " \
                                f"If none of the options reflect a correct and direct diagnostic criterion, " \
                                f"select 'Need new options', and additional options will be provided." \
                                f"Do not choose an option just because it appears similar or somewhat related."

        else:
            question_criteria = f"What criterion was used to make the decision for the first question? " \
                                f"Options: {options_criteria_per_round}. " \
                                f"If none of the options reflect a correct and direct diagnostic criterion, " \
                                f"select 'None of the above', and explain the criterion you applied."
        questions_criteria.append(question_criteria)

    qa_dict = {
               'question': questions_criteria,

               'answer': answers_criteria}
    # 'dicom': dicom,
    #                'dx': target_dx,
    #                'inference_type': 'review',
    #                'stage': 'criteria',
    # 'option': options_criteria,

    save_dir_gold_qa = f"{save_dir_gold_qa}/{stage}/{pitfall}"
    os.makedirs(save_dir_gold_qa, exist_ok=True)
    with open(f"{save_dir_gold_qa}/{dicom}.json", "w") as file:
        json.dump(qa_dict, file, indent=4)

def return_mc_format_list(args, dx):
    bodypart_num_per_dx = glob(f'{args.segmask_base_dir}/{dx}/*')

    if args.inference_path in ['path1']:
        mc_format_lst_criteria = ['basic'] * 50 + ['two-round'] * 50
        if len(bodypart_num_per_dx) == 1:
            mc_format_lst_bodypart = ['basic'] * 50 + ['two-round_none_included'] * 50
        else:
            mc_format_lst_bodypart = ['basic'] * 34 + ['two-round_none_included'] * 33 + ['two-round_partial_inclusion'] * 33

    elif args.inference_path in ['re-path1']:
        mc_format_lst_criteria = ['basic', 'two-round'] * 50
        if len(bodypart_num_per_dx) == 1:
            mc_format_lst_bodypart = ['basic', 'two-round_none_included', 'two-round_none_included', 'basic'] * 25
        else:
            mc_format_lst_bodypart = ['two-round_none_included', 'two-round_partial_inclusion',
                                      'two-round_none_included',
                                      'basic', 'two-round_partial_inclusion', 'two-round_partial_inclusion',
                                      'basic', 'basic', 'two-round_none_included'] * 11 + ['basic']
    
    elif args.inference_path in ['path2']:
        # Path 2 doesn't use segmentation masks, so bodypart questions aren't relevant
        mc_format_lst_criteria = ['basic'] * 100
        mc_format_lst_bodypart = []  # Empty list since path2 doesn't generate bodypart questions

    return mc_format_lst_criteria, mc_format_lst_bodypart

def return_measured_value(args, dicom, dx):
    dx_by_colname_measurement = {
        'inspiration': 'rib_position', 'trachea_deviation': 'direction',
        'cardiomegaly': 'ctr', 'mediastinal_widening': 'mcr', 'carina_angle': 'angle', 'rotation': 'ratio',
        'aortic_knob_enlargement': 'ratio', 'ascending_aorta_enlargement': 'ratio',
        'descending_aorta_enlargement': 'ratio', 'descending_aorta_tortuous': 'curvature',
    }

    df_chexstruct = pd.read_csv(os.path.join(args.chexstruct_base_dir, f'{dx}.csv'))
    if dx in ['inclusion']:
        def get_lung_label(df, dicom, region):

            right_col = f'label_{region}_right_lung'
            left_col = f'label_{region}_left_lung'

            right = df[df['image_file'] == dicom][right_col].values[0]
            left = df[df['image_file'] == dicom][left_col].values[0]

            if right == 1 and left == 1:
                return f'{region}_both_in'
            elif right == 1 and left == 0:
                return f'{region}_r_in_l_ex'
            elif right == 0 and left == 1:
                return f'{region}_r_ex_l_in'
            else:
                return f'{region}_both_ex'

        label_side = get_lung_label(df_chexstruct, dicom, 'side')
        label_apex = get_lung_label(df_chexstruct, dicom, 'apex')
        label_bottom = get_lung_label(df_chexstruct, dicom, 'bottom')

        measured_value = [label_side, label_apex, label_bottom]

    elif dx in ['projection']:
        measured_value_right = df_chexstruct[df_chexstruct['image_file'] == dicom]['ratio_right'].values[0]
        measured_value_left = df_chexstruct[df_chexstruct['image_file'] == dicom]['ratio_left'].values[0]
        measured_value = [measured_value_right, measured_value_left]
    else:
        measured_value = df_chexstruct[df_chexstruct['image_file'] == dicom][dx_by_colname_measurement[dx]].values[0]
    return measured_value



if __name__ == '__main__':
    args = config()

    args.segmask_base_dir = os.path.join(args.cxreasonbench_base_dir, 'segmask_bodypart')
    args.pnt_base_dir = os.path.join(args.cxreasonbench_base_dir, 'pnt_on_cxr')
    args.dx_by_dicoms_file = os.path.join(args.cxreasonbench_base_dir, 'dx_by_dicoms.json')

    # Load metadata based on dataset type
    if args.dataset_name == 'mimic-cxr-jpg':
        mimic_meta = pd.read_csv(args.mimic_meta_path)
    else:
        mimic_meta = None  # Not needed for NIH dataset

    # Initialize dataset configuration
    dataset_config = DatasetConfig(args, mimic_meta)
    set_dataset_config(dataset_config)

    with open(args.dx_by_dicoms_file, "r") as file:
        dx_by_dicoms = json.load(file)

    for dx, dicom_lst in dx_by_dicoms.items():
        save_dir_qa = os.path.join(args.save_base_dir, 'qa', dx, args.inference_path)
        mc_format_lst_criteria, mc_format_lst_bodypart = return_mc_format_list(args, dx)
        for idx, dicom in tqdm(enumerate(dicom_lst), total=len(dicom_lst), desc=f'{args.inference_path}-{dx}'):
            measured_value = return_measured_value(args, dicom, dx)
            if args.inference_path in ['path1']:
                random.seed(42)

                answer_init = qa_init_question('init', dicom, dx, measured_value, mimic_meta, save_dir_qa)
                qa_criteria('stage1', dicom, dx, args.num_options, mc_format_lst_criteria[idx], save_dir_qa)
                if dx in refined_criteria_per_dx.keys():
                    qa_custom_criteria('stage1.5', dicom, dx, save_dir_qa)
                qa_bodypart('stage2', args.inference_path, dicom, dx, args.num_options, mc_format_lst_bodypart[idx], args.segmask_base_dir, save_dir_qa)
                qa_measurement('stage3', args.inference_path, dicom, dx, measured_value, args.num_options, save_dir_qa)
                qa_final('stage4', dicom, dx, measured_value, answer_init, save_dir_qa)

            elif args.inference_path in ['path2']:
                random.seed(33)
                qa_bodypart_guidance('stage1', dicom, dx, args.segmask_base_dir, args.num_options, save_dir_qa)
                qa_measurement_guidance('stage2', dicom, dx,  args.pnt_base_dir, measured_value, args.num_options, save_dir_qa)
                qa_final_guidance('stage3', dicom, dx, measured_value, save_dir_qa)

            elif args.inference_path in ['re-path1']:
                random.seed(24)
                answer_init = qa_init_after_guidance('init', dicom, dx, measured_value, save_dir_qa)
                qa_criteria_after_guidance('stage1', dicom, dx, args.num_options, mc_format_lst_criteria[idx], save_dir_qa)
                qa_bodypart('stage2', args.inference_path, dicom, dx, args.num_options, mc_format_lst_bodypart[idx], args.segmask_base_dir, save_dir_qa)
                qa_measurement('stage3', args.inference_path, dicom, dx, measured_value, args.num_options, save_dir_qa)
                qa_final('stage4', dicom, dx, measured_value, answer_init, save_dir_qa)





