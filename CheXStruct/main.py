import os
import csv
import parmap
import argparse
import pandas as pd
from glob import glob

from preprocessor.frontal_cxr import extract_mask_num, extract_abdomial, extract_window
from preprocessor.inclusion import extract_inclusion
from preprocessor.cardiomegaly import extract_cardiomegaly
from preprocessor.mediastinal_widening import extract_mw

from preprocessor.carina_angle import extract_carina

from preprocessor.trachea import extract_trachea
from preprocessor.descending_aorta_tortuous_enlarged import extract_desc_aorta
from preprocessor.enlarged_aortic_knob import extract_enlarged_aortic_knob
from preprocessor.enlarged_ascending_aorta import extract_enlarged_asc_aorta
from preprocessor.trachea_deviation import extract_trachea_deviation

from preprocessor.inspiration import extract_inspiration
from preprocessor.projection import extract_projection
from preprocessor.rotation import extract_rotation

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='mimic-cxr-jpg')

    parser.add_argument('--pm_processes', default=8, type=int)

    parser.add_argument('--save_base_dir', default='path/to/save/output', type=str)

    parser.add_argument('--mimic_cxr_base_dir', default="<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files", type=str)
    parser.add_argument('--mimic_meta_file', default='<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv', type=str)


    parser.add_argument('--cxas_base_dir', type=str, default='path/to/cxas_segmentation_folders')
    parser.add_argument('--chexmask_base_dir', type=str, default='path/to/chexmask_segmentation_folders')
    parser.add_argument('--chest_imagenome_base_dir', default='<path_to_physionet_download_dir>/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_graph', type=str)

    args = parser.parse_args()
    return args


def write_csv_row(save_dir, info, header, row):
    file_path = os.path.join(save_dir, f"{info}.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def save_metadata(args, target_data, viewposition, meta_gathered):
    for info, meta_data in meta_gathered.items():
        if 'label' in meta_data:
            view = viewposition if viewposition in ['AP', 'PA'] else 'N/A'
            base_info = [target_data, view]

            # ----------- meta data -----------
            meta_data_ = {k: str(v) for k, v in meta_data.items() if not k.startswith("viz_")}
            header_ = ["image_file", "viewposition"] + list(meta_data_.keys())
            row_ = base_info + list(meta_data_.values())
            write_csv_row(args.save_dir, info, header_, row_)

            # ----------- visualization data -----------
            meta_data_viz = {k.replace("viz_", ""): str(v) for k, v in meta_data.items() if k.startswith("viz_")}
            header_viz = ["image_file", "viewposition"] + list(meta_data_viz.keys())
            row_viz = base_info + list(meta_data_viz.values())
            write_csv_row(args.save_dir_viz, info, header_viz, row_viz)

def log_saved_data(args, img_path):
    log_file_path = args.log_file_path
    header_log = ['img_path']
    fname = os.path.basename(img_path)
    dicom = os.path.splitext(fname)[0]
    row_log = [dicom]

    if not os.path.isfile(log_file_path):
        with open(log_file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(header_log)

    with open(log_file_path, 'a', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(row_log)

def extract_meta(target_data, args):
    # build image data
    subject_id = str(args.metadata[args.metadata['dicom_id'] == target_data]['subject_id'].values[0])
    study_id = str(args.metadata[args.metadata['dicom_id'] == target_data]['study_id'].values[0])
    img_path = f'{args.mimic_cxr_base_dir}/p{subject_id[:2]}/p{subject_id}/s{study_id}/{target_data}.jpg'

    viewposition = args.metadata[args.metadata['dicom_id'] == target_data]['ViewPosition'].values[0]

    cxas_files = glob(os.path.join(args.cxas_base_dir, target_data, '*.png'))
    chexmask_files = glob(os.path.join(args.chexmask_base_dir, 'RL', f'{target_data}.png'))
    ci_file = os.path.join(args.chest_imagenome_base_dir, f'{target_data}_SceneGraph.json')

    image_data = {'dicom': target_data, 'cxr': img_path, 'viewposition': viewposition,
                  'cxas': {os.path.splitext(os.path.basename(file))[0]: file for file in cxas_files},
                  'chexmask': {'RL_mask': file for file in chexmask_files}, 'bbox': ci_file}

    if not len(chexmask_files):
        image_data['chexmask']['RL_mask'] = ''

    meta_mask_num = extract_mask_num(image_data)
    meta_abdomial = extract_abdomial(image_data)
    meta_window = extract_window(image_data)

    meta_all_base = {'mask_number': meta_mask_num, 'abdomial_xray': meta_abdomial, 'window': meta_window}
    save_metadata(args, target_data, viewposition, meta_all_base)

    label_mask_num = meta_mask_num['label']
    label_abdomial = meta_abdomial['label']
    label_window = meta_window['label']
    if label_mask_num & label_abdomial & label_window:
        meta_carina_angle = extract_carina(image_data)
        meta_inspiration = extract_inspiration(image_data)
        meta_projection = extract_projection(image_data)

        meta_inclusion = extract_inclusion(image_data)

        meta_rotation, meta_midline = extract_rotation(image_data)
        meta_trachea = extract_trachea(image_data)

        meta_trachea_deviation = extract_trachea_deviation(meta_trachea, meta_midline)
        meta_desc_tortuous, meta_desc_enlarged = extract_desc_aorta(image_data, meta_trachea)
        meta_knob_enlarged = extract_enlarged_aortic_knob(image_data, meta_trachea)

        meta_mw = extract_mw(image_data, meta_inclusion)
        meta_cardiomegaly = extract_cardiomegaly(image_data, meta_inclusion)

        meta_asc_enlarged = extract_enlarged_asc_aorta(image_data, meta_cardiomegaly, meta_trachea)

        meta_inclusion.pop('right_lung_inner_point', None)
        meta_inclusion.pop('left_lung_inner_point', None)
        meta_cardiomegaly.pop('qc_valid_heart', None)

        meta_all_dx = {
            'inspiration': meta_inspiration,
            'rotation': meta_rotation,
            'projection': meta_projection,
            'trachea_deviation': meta_trachea_deviation,
            'carina_angle': meta_carina_angle,
            'descending_aorta_enlargement': meta_desc_enlarged,
            'descending_aorta_tortuous': meta_desc_tortuous,

            'inclusion': meta_inclusion,
            'cardiomegaly': meta_cardiomegaly,
            'mediastinal_widening': meta_mw,
            'ascending_aorta_enlargement': meta_asc_enlarged,
            'aortic_knob_enlargement': meta_knob_enlarged,
        }

        save_metadata(args, target_data, viewposition, meta_all_dx)
        log_saved_data(args, img_path)

if __name__ == "__main__":
    args = config()

    args.save_dir = os.path.join(args.save_base_dir, args.dataset_name)
    args.save_dir_viz = os.path.join(args.save_base_dir, f"{args.dataset_name}_viz")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir_viz, exist_ok=True)

    args.log_file_path = os.path.join(args.save_dir, 'log_saved_file.csv')

    args.metadata = pd.read_csv(args.mimic_meta_file)
    img_paths = args.metadata[args.metadata['ViewPosition'].isin(['PA', 'AP'])]['dicom_id']

    if os.path.isfile(args.log_file_path):
        saved_img_paths = pd.read_csv(args.log_file_path, on_bad_lines='skip')['img_path']
    else:
        saved_img_paths = []
    remained_img_paths = list(set(img_paths).difference(saved_img_paths))

    print('=*' * 20)
    print('Dataset:', args.dataset_name)
    print('Saved Dir:', args.save_dir)
    print('# Total:', len(img_paths))
    print('# Saved:', len(saved_img_paths))
    print('# Remained:', len(remained_img_paths))
    print('=*' * 20)

    parmap.map(extract_meta, remained_img_paths, args, pm_pbar=True, pm_processes=args.pm_processes)




