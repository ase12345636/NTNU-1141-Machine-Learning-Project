"""
Create separate dx_by_dicoms files for different inference paths
- dx_by_dicoms.json: All images (for Path 1 - reasoning)
- dx_by_dicoms_path2.json: Only images with valid viz data (for Path 2 - guidance)
"""
import os
import json
import argparse
import pandas as pd
from glob import glob


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_dir', type=str, required=True,
                        help='Directory containing viz CSV files')
    parser.add_argument('--dx_by_dicoms_file', type=str, required=True,
                        help='Path to original dx_by_dicoms.json')
    parser.add_argument('--output_path2_file', type=str, required=True,
                        help='Path to save dx_by_dicoms_path2.json')
    args = parser.parse_args()
    return args


def main():
    args = config()
    
    # Read original dx_by_dicoms.json
    with open(args.dx_by_dicoms_file, 'r') as f:
        dx_by_dicoms_all = json.load(f)
    
    dx_by_dicoms_path2 = {}
    
    print("Creating dx_by_dicoms_path2.json (only valid viz data)...")
    print("=" * 70)
    
    total_all = 0
    total_path2 = 0
    
    for dx in dx_by_dicoms_all.keys():
        viz_csv = os.path.join(args.viz_dir, f'{dx}.csv')
        
        if os.path.exists(viz_csv):
            df = pd.read_csv(viz_csv)
            
            # Filter to only images with valid viz data
            if 'has_valid_viz' in df.columns:
                df_valid = df[df['has_valid_viz'] == True]
                dicom_list_path2 = df_valid['image_file'].unique().tolist()
            else:
                # If no has_valid_viz column, assume all are valid
                dicom_list_path2 = df['image_file'].unique().tolist()
            
            dx_by_dicoms_path2[dx] = dicom_list_path2
            
            count_all = len(dx_by_dicoms_all[dx])
            count_path2 = len(dicom_list_path2)
            excluded = count_all - count_path2
            
            total_all += count_all
            total_path2 += count_path2
            
            status = "✓" if excluded == 0 else "⚠"
            print(f"{status} {dx:30s}: {count_path2:5d}/{count_all:5d} images", end="")
            if excluded > 0:
                print(f" ({excluded} excluded, {(excluded/count_all)*100:.1f}%)")
            else:
                print()
        else:
            print(f"✗ {dx:30s}: viz CSV not found, excluding from Path 2")
            dx_by_dicoms_path2[dx] = []
    
    print("=" * 70)
    
    # Save Path 2 version
    with open(args.output_path2_file, 'w') as f:
        json.dump(dx_by_dicoms_path2, f, indent=2)
    
    print(f"\n📁 Files created:")
    print(f"   {args.dx_by_dicoms_file}")
    print(f"     → For Path 1 (reasoning): {total_all} total images")
    print(f"   {args.output_path2_file}")
    print(f"     → For Path 2 (guidance):  {total_path2} images with valid viz")
    
    excluded_total = total_all - total_path2
    if excluded_total > 0:
        print(f"\n⚠️  {excluded_total} images ({(excluded_total/total_all)*100:.1f}%) excluded from Path 2")
        print(f"   These images still available for Path 1")
    else:
        print(f"\n✓ All images have valid viz data!")


if __name__ == '__main__':
    main()
