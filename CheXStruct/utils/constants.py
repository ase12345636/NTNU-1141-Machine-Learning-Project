thoracic_spine = ["thoracic spine",
                          "vertebrae T1",
                          "vertebrae T2",
                          "vertebrae T3",
                          "vertebrae T4",
                          "vertebrae T5",
                          "vertebrae T6",
                          "vertebrae T7",
                          "vertebrae T8",
                          "vertebrae T9",
                          "vertebrae T10",
                          "vertebrae T11",
                          "vertebrae T12",
                          ]

clavicle_set = ['clavicles', "clavicle left", "clavicle right"]

scapula_set = [
    "scapulas",
    "scapula left",
    "scapula right",
]

humerus = [
    "humerus",
    "humerus left",
    "humerus right",
]

rib = [
    "posterior 12th rib right",
    "posterior 12th rib left",
    "anterior 11th rib right",
    "posterior 11th rib right",
    "anterior 11th rib left",
    "posterior 11th rib left",
    "anterior 10th rib right",
    "posterior 10th rib right",
    "anterior 10th rib left",
    "posterior 10th rib left",
    "anterior 9th rib right",
    "posterior 9th rib right",
    "anterior 9th rib left",
    "posterior 9th rib left",
    "anterior 8th rib right",
    "posterior 8th rib right",
    "anterior 8th rib left",
    "posterior 8th rib left",
    "anterior 7th rib right",
    "posterior 7th rib right",
    "anterior 7th rib left",
    "posterior 7th rib left",
    "anterior 6th rib right",
    "posterior 6th rib right",
    "anterior 6th rib left",
    "posterior 6th rib left",
    "anterior 5th rib right",
    "posterior 5th rib right",
    "anterior 5th rib left",
    "posterior 5th rib left",
    "anterior 4th rib right",
    "posterior 4th rib right",
    "anterior 4th rib left",
    "posterior 4th rib left",
    "anterior 3rd rib right",
    "posterior 3rd rib right",
    "anterior 3rd rib left",
    "posterior 3rd rib left",
    "anterior 2nd rib right",
    "posterior 2nd rib right",
    "anterior 2nd rib left",
    "posterior 2nd rib left",
    "anterior 1st rib right",
    "posterior 1st rib right",
    "anterior 1st rib left",
    "posterior 1st rib left",
]

diaphragm = ["diaphragm", "left hemidiaphragm", "right hemidiaphragm", ]

mediastinum = [
    "cardiomediastinum",
    "upper mediastinum",
    "lower mediastinum",
    # "anterior mediastinum",
    # "middle mediastinum",
    # "posterior mediastinum",
]

heart = [
    "heart",
    "heart atrium left",
    "heart atrium right",
    # "heart myocardium",
    "heart ventricle left",
    "heart ventricle right",
]

trachea = [
    "trachea",
    "tracheal bifurcation",
]

zones = [
    "right upper zone lung",
    "right mid zone lung",
    "right lung base",
    "right apical zone lung",
    "left upper zone lung",
    "left mid zone lung",
    "left lung base",
    "left apical zone lung",
]

lung_halves = [
    "right lung",
    "left lung",
]

vessels = [
    # "heart",
    "ascending aorta",
    "descending aorta",
    "aortic arch",

    "pulmonary artery",
    # "inferior vena cava",
]

lobes = [
    "lung lower lobe left",
    "lung upper lobe left",
    "lung lower lobe right",
    "lung middle lobe right",
    "lung upper lobe right",
]

TARGET_MASK_LIST = thoracic_spine + clavicle_set + scapula_set + rib + diaphragm + mediastinum + heart + trachea + zones + vessels + lobes + lung_halves

MD_RANGES = {
    'Frontal_CXR': {'mask_num': 70, 'abdomial_ratio': 1.0, 'window': 450},
    'Inclusion': {'APEX': {'exclusion': [0.0, 0.01], 'inclusion': [0.01, 1.0]},
                  'SIDE': {'exclusion': [0.0, 0.005], 'inclusion': [0.005, 1.0]},
                  'BOTTOM': {'exclusion': [0.0, 0.03], 'mixed': [0.03, 0.09], 'inclusion': [0.09, 1.0]}},
    'Inspiration': 9,
    'Rotation': 0.4,
    'Projection': 0.3,
    'Cardiomegaly': {'PA': 0.5, 'AP': 0.55},
    'Mediastinal_Widening': {'PA': 0.25, 'AP': 0.33},
    'Trachea_Deviation': {},
    'Carina_Angle': [40, 80],
    'Aortic_Knob_Enlargement': 2.5,
    'Ascending_Aorta_Enlargemnet': [0.1, 0.3],
    'Descending_Aorta_Enlargement': 2.5,
    'Descending_Aorta_Tortuous': 0.0009,
}

md_range_per_part = {'apex': {'y_diff': [0.0, 0.05], 'y_position': [0.0, 0.6]},
                     'bottom': {'y_diff': [0.0, 0.05], 'y_position': [0.0, 0.4], 'dp_rb_diff': [0.0, 0.25]},
                     'side': {'both_inclusion': 0.7, 'both_exclusion': 0.6, 'r_in_l_ex': 0.7, 'r_ex_l_in': 0.7}}

ROUND_VALUE = {
    'Rotation': 2,
    'Projection': 2,

    'Cardiomegaly': 2,
    'Mediastinal_Widening': 2,
    'Carina_Angle': 0,

    'Aortic_Knob_Enlargement': 2,
    'Descending_Aorta_Enlargement': 2,
    'Descending_Aorta_Tortuous': 4,
}