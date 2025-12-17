"""
Create visualization CSV files for NIH dataset
These CSVs list which images need segmentation mask visualization for each diagnostic task
"""
import os
import json
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx_by_dicoms_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--nih_csv_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load dx_by_dicoms.json
    with open(args.dx_by_dicoms_file, 'r') as f:
        dx_by_dicoms = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # For each diagnostic task, create a CSV with image_file column
    for dx, dicom_list in dx_by_dicoms.items():
        print(f"Processing {dx}: {len(dicom_list)} images")
        
        # Create DataFrame with image_file column (DICOM IDs without extension)
        viz_df = pd.DataFrame({
            'image_file': dicom_list
        })
        
        # Save to CSV
        output_file = os.path.join(args.output_dir, f'{dx}.csv')
        viz_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
    
    print(f"\nCreated {len(dx_by_dicoms)} visualization CSV files in {args.output_dir}")

if __name__ == "__main__":
    main()
