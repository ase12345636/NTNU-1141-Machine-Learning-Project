"""
Generate dx_by_dicoms.json for NIH CXR-14 dataset
This script reads all CSV files in the chexstruct directory and creates a JSON file
mapping diagnostic tasks to their corresponding DICOM IDs.
"""
import os
import json
import argparse
import pandas as pd
from glob import glob


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chexstruct_base_dir', default='/mnt/d/CXReasonBench/nih_cxr14', type=str,
                        help='Directory containing CSV files for each diagnostic task')
    parser.add_argument('--output_file', default='/mnt/d/CXReasonBench/output_nih/dx_by_dicoms.json', type=str,
                        help='Output JSON file path')
    args = parser.parse_args()
    return args


def main():
    args = config()
    
    # Diagnostic tasks that are used in CXReasonBench
    diagnostic_tasks = [
        'inclusion',
        'inspiration',
        'rotation',
        'projection',
        'cardiomegaly',
        'mediastinal_widening',
        'carina_angle',
        'trachea_deviation',
        'aortic_knob_enlargement',
        'ascending_aorta_enlargement',
        'descending_aorta_enlargement',
        'descending_aorta_tortuous',
    ]
    
    dx_by_dicoms = {}
    
    for dx in diagnostic_tasks:
        csv_path = os.path.join(args.chexstruct_base_dir, f'{dx}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Get unique image files (DICOM IDs)
            if 'image_file' in df.columns:
                dicom_list = df['image_file'].unique().tolist()
                dx_by_dicoms[dx] = dicom_list
                print(f"{dx}: {len(dicom_list)} images")
            else:
                print(f"Warning: {csv_path} does not have 'image_file' column")
        else:
            print(f"Warning: {csv_path} not found, skipping {dx}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(dx_by_dicoms, f, indent=2)
    
    print(f"\nGenerated dx_by_dicoms.json with {len(dx_by_dicoms)} diagnostic tasks")
    print(f"Saved to: {args.output_file}")
    
    # Print summary
    total_images = sum(len(dicoms) for dicoms in dx_by_dicoms.values())
    print(f"Total image-task pairs: {total_images}")


if __name__ == '__main__':
    main()
