"""
Create visualization CSV files for NIH dataset by computing required viz fields from CXAS masks
This replaces the original viz CSV generation to work with NIH dataset structure
"""
import os
import json
import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
import argparse
from tqdm import tqdm


def compute_aortic_knob_viz(dicom, cxas_dir, source_row):
    """Compute viz fields for aortic knob enlargement from CXAS masks"""
    try:
        # Load CXAS masks
        fname_descending_aorta = os.path.join(cxas_dir, dicom, "descending aorta.png")
        fname_aortic_arch = os.path.join(cxas_dir, dicom, "aortic arch.png")
        fname_trachea = os.path.join(cxas_dir, dicom, "trachea.png")
        
        if not os.path.exists(fname_descending_aorta) or not os.path.exists(fname_aortic_arch):
            return None
            
        mask_descending_aorta = cv2.imread(fname_descending_aorta, 0)
        mask_aortic_arch = cv2.imread(fname_aortic_arch, 0)
        
        # Calculate y_max (top of descending aorta)
        desc_height_idx = mask_descending_aorta.sum(axis=-1).nonzero()[0]
        if len(desc_height_idx) == 0:
            return None
        ymin_desc = desc_height_idx[0]
        ymax_desc = desc_height_idx[-1]
        y_subpart_desc = int((ymax_desc - ymin_desc) * (1 / 3))
        ymin_desc_sub = ymin_desc + y_subpart_desc
        ysub_desc = y_subpart_desc
        ymax_desc_sub = ymax_desc
        
        # Get aortic arch height
        aortic_arch_height_idx = mask_aortic_arch.sum(axis=-1).nonzero()[0]
        if len(aortic_arch_height_idx) == 0:
            return None
        ymin_aortic_arch = aortic_arch_height_idx[0]
        ymax_aortic_arch = aortic_arch_height_idx[-1]
        y_max = ymin_aortic_arch
        
        # Calculate xmax_trachea_mean
        if os.path.exists(fname_trachea):
            mask_trachea = cv2.imread(fname_trachea, 0)
            mask_trachea_within_aortic_arch = mask_trachea[ymin_aortic_arch: (ymax_aortic_arch + 1)]
            
            if mask_trachea_within_aortic_arch.sum() > 0:
                rightmost_xs = []
                for row in mask_trachea_within_aortic_arch:
                    ones_indices = np.where(row > 0)[0]
                    if ones_indices.size > 0:
                        rightmost_xs.append(ones_indices[-1])
                if rightmost_xs:
                    xmax_trachea_mean = int(np.array(rightmost_xs).mean())
                else:
                    xmax_trachea_mean = 0
            else:
                xmax_trachea_mean = 0
        else:
            xmax_trachea_mean = 0
        
        return {
            'y_max': y_max,
            'ymin_desc': ymin_desc,
            'ysub_desc': ysub_desc,
            'ymax_desc_sub': ymax_desc_sub,
            'xmax_trachea_mean': xmax_trachea_mean
        }
    except Exception as e:
        print(f"    Error computing viz for {dicom}: {e}")
        return None


def compute_cardiomegaly_viz(dicom, cxas_dir, source_row):
    """Compute viz fields for cardiomegaly from CXAS masks"""
    try:
        fname_heart = os.path.join(cxas_dir, dicom, "heart.png")
        fname_lung = os.path.join(cxas_dir, dicom, "right lung.png")
        
        if not os.path.exists(fname_heart) or not os.path.exists(fname_lung):
            return None
            
        mask_heart = cv2.imread(fname_heart, 0)
        mask_lung = cv2.imread(fname_lung, 0)
        
        # Get heart bounding box
        heart_idx = np.where(mask_heart > 0)
        if len(heart_idx[0]) == 0:
            return None
        ymin_heart = heart_idx[0].min()
        ymax_heart = heart_idx[0].max()
        xmin_heart = heart_idx[1].min()
        xmax_heart = heart_idx[1].max()
        
        # Get lung x bounds
        lung_idx = np.where(mask_lung > 0)
        if len(lung_idx[0]) == 0:
            return None
        xmin_lung = lung_idx[1].min()
        xmax_lung = lung_idx[1].max()
        
        coord_mask = f"({xmin_heart}, {ymin_heart}, {xmax_heart}, {ymax_heart})"
        
        return {
            'coord_mask': coord_mask,
            'xmin_lung': xmin_lung,
            'xmax_lung': xmax_lung
        }
    except Exception as e:
        print(f"    Error computing viz for {dicom}: {e}")
        return None


def compute_mediastinal_widening_viz(dicom, cxas_dir, source_row):
    """Compute viz fields for mediastinal widening"""
    try:
        fname_mediastinum = os.path.join(cxas_dir, dicom, "cardiomediastinum.png")
        fname_rlung = os.path.join(cxas_dir, dicom, "right lung.png")
        fname_llung = os.path.join(cxas_dir, dicom, "left lung.png")
        
        if not os.path.exists(fname_mediastinum):
            return None
            
        mask_medi = cv2.imread(fname_mediastinum, 0)
        if mask_medi is None:
            return None
        
        # Get mediastinum top y
        medi_height_idx = mask_medi.sum(axis=-1).nonzero()[0]
        if len(medi_height_idx) == 0:
            return None
        y_medi = int(medi_height_idx[0])
        
        # Get mediastinum x bounds at y_medi
        medi_row = mask_medi[y_medi, :]
        medi_x_idx = np.where(medi_row > 0)[0]
        if len(medi_x_idx) == 0:
            return None
        xmin_medi = int(medi_x_idx.min())
        xmax_medi = int(medi_x_idx.max())
        
        # Get lung x bounds
        xmin_rlung = None
        xmax_llung = None
        if os.path.exists(fname_rlung):
            mask_rlung = cv2.imread(fname_rlung, 0)
            if mask_rlung is not None:
                rlung_idx = np.where(mask_rlung > 0)
                if len(rlung_idx[1]) > 0:
                    xmin_rlung = int(rlung_idx[1].min())
        
        if os.path.exists(fname_llung):
            mask_llung = cv2.imread(fname_llung, 0)
            if mask_llung is not None:
                llung_idx = np.where(mask_llung > 0)
                if len(llung_idx[1]) > 0:
                    xmax_llung = int(llung_idx[1].max())
        
        # Return None if any required field is missing
        if xmin_rlung is None or xmax_llung is None:
            return None
        
        return {
            'y_medi': y_medi,
            'xmin_medi': xmin_medi,
            'xmax_medi': xmax_medi,
            'xmin_rlung': xmin_rlung,
            'xmax_llung': xmax_llung
        }
    except Exception as e:
        print(f"    Error computing viz for {dicom}: {e}")
        return None


def compute_inspiration_viz(dicom, cxas_dir, source_row):
    """Compute viz fields for inspiration"""
    try:
        fname_lung = os.path.join(cxas_dir, dicom, "right lung.png")
        
        if not os.path.exists(fname_lung):
            return None
            
        mask_lung = cv2.imread(fname_lung, 0)
        
        # Get x_lung_mid_combined from source if available
        x_lung_mid_combined = source_row.get('mid-clavicular_line(x)', None)
        if x_lung_mid_combined is None:
            # Calculate from mask center
            lung_idx = np.where(mask_lung > 0)
            if len(lung_idx[1]) > 0:
                x_lung_mid_combined = int(np.median(lung_idx[1]))
            else:
                return None
        
        # Get lung y values at midline
        lung_y_lst = []
        for y in range(mask_lung.shape[0]):
            if mask_lung[y, int(x_lung_mid_combined)] > 0:  # Check for any non-zero value
                lung_y_lst.append(y)
        
        # Also need label from source
        label = source_row.get('label', None)
        
        return {
            'x_lung_mid_combined': x_lung_mid_combined,
            'lung_y_lst_x_lung_mid': str(lung_y_lst),
            'label': label
        }
    except Exception as e:
        print(f"    Error computing viz for {dicom}: {e}")
        return None


def compute_rotation_viz(dicom, cxas_dir, source_row):
    """Compute viz fields for rotation"""
    try:
        # Get midline points from source
        midline_points = source_row.get('midline_points', None)
        if midline_points is None or midline_points == '[]':
            return None
        
        # Parse midline_points
        points = eval(midline_points) if isinstance(midline_points, str) else midline_points
        if not points or len(points) == 0:
            return None
        
        # Calculate slope (m) and intercept (b) from midline points
        # Points are in format [(x1, y1), (x2, y2), ...]
        points = np.array(points)
        xs = points[:, 0]
        ys = points[:, 1]
        
        # Linear regression: y = mx + b
        # But in CXR, we use vertical line so x = my + b
        if len(ys) > 1:
            m, b = np.polyfit(ys, xs, 1)
        else:
            m = 0
            b = xs[0] if len(xs) > 0 else 0
        
        return {
            'right_end_pnt': source_row.get('medial_end_right_clavicle', None),
            'left_end_pnt': source_row.get('medial_end_left_clavicle', None),
            'target_coords': str(points.tolist()),
            'm': m,
            'b': b
        }
    except Exception as e:
        print(f"    Error computing viz for {dicom}: {e}")
        return None


def compute_inclusion_viz(dicom, cxas_dir, source_row):
    """Extract viz fields for inclusion (copy all point coordinates and label)"""
    return {
        'apex_right': source_row.get('point_apex_right_lung', None),
        'apex_left': source_row.get('point_apex_left_lung', None),
        'side_right': source_row.get('point_side_right_lung', None),
        'side_left': source_row.get('point_side_left_lung', None),
        'cpa_right': source_row.get('point_bottom_right_lung', None),
        'cpa_left': source_row.get('point_bottom_left_lung', None),
        'label': source_row.get('label', None)
    }


def compute_projection_viz(dicom, cxas_dir, source_row):
    """Extract viz fields for projection (copy overlap and scapular region data)"""
    return {
        'overlap_region_right': source_row.get('overlap_region_right', None),
        'overlap_region_left': source_row.get('overlap_region_left', None),
        'scapular_region_right': source_row.get('scapular_region_right', None),
        'scapular_region_left': source_row.get('scapular_region_left', None)
    }


def compute_carina_angle_viz(dicom, cxas_dir, source_row):
    """Extract viz fields for carina angle (copy point coordinates and angle)"""
    return {
        'point_1': source_row.get('point_1', None),
        'point_2': source_row.get('point_2', None),
        'point_3': source_row.get('point_3', None),
        'angle': source_row.get('angle', None)
    }


def compute_trachea_deviation_viz(dicom, cxas_dir, source_row):
    """Extract viz fields for trachea deviation
    
    Computes midline_x_min, midline_x_max, y_min, y_max from point_1 to point_9
    which are needed by point_on_cxr.py for visualization.
    """
    try:
        # These fields exist in source CSV
        result = {
            'direction_per_pnt': source_row.get('direction_per_pnt', None)
        }
        
        # Copy point_1 to point_9 as-is from source
        for i in range(1, 10):
            result[f'point_{i}'] = source_row.get(f'point_{i}', None)
        
        # Compute midline bounds from point_1 and point_9 for visualization
        # point_on_cxr.py needs midline_x_min, midline_x_max, y_min, y_max
        if pd.notna(source_row.get('point_1')) and pd.notna(source_row.get('point_9')):
            import ast
            point_1 = ast.literal_eval(source_row['point_1'])
            point_9 = ast.literal_eval(source_row['point_9'])
            
            # midline_x_min is x coordinate at the top (point_1[0])
            # midline_x_max is x coordinate at the bottom (point_9[0])
            result['midline_x_min'] = point_1[0]
            result['midline_x_max'] = point_9[0]
            
            # y_min and y_max are y coordinates of the endpoints
            result['y_min'] = point_1[1]
            result['y_max'] = point_9[1]
        
        return result
        
    except Exception as e:
        print(f"    Error computing trachea_deviation viz for {dicom}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_ascending_aorta_viz(dicom, cxas_dir, source_row):
    """Extract viz fields for ascending aorta enlargement (copy point coordinates)
    Note: Renaming fields to match point_on_cxr.py expectations
    """
    return {
        'pnt_heart': source_row.get('heart_point', None),
        'pnt_trachea': source_row.get('trachea_point', None)
    }


def compute_descending_aorta_viz(dicom, cxas_dir, source_row):
    """Compute viz fields for descending aorta (enlargement and tortuous)
    
    For descending_aorta_enlargement:
    - Renames trachea_point_right/left to trachea_y_width, trachea_xmin_width, trachea_xmax_width
    - Renames desc_aorta_point_right/left to pnt_r_1, pnt_1
    
    For descending_aorta_tortuous:
    - Renames point_N to pnt_r_N (since they're on the right side)
    """
    try:
        fname_descending_aorta = os.path.join(cxas_dir, dicom, "descending aorta.png")
        
        if not os.path.exists(fname_descending_aorta):
            return None
        
        mask_descending_aorta = cv2.imread(fname_descending_aorta, 0)
        
        # Get ymin and ymax of descending aorta
        desc_height_idx = mask_descending_aorta.sum(axis=-1).nonzero()[0]
        if len(desc_height_idx) == 0:
            return None
        
        ymin_desc = desc_height_idx[0]
        ymax_desc = desc_height_idx[-1]
        
        # Calculate ymin_start and ymin_end for segmentation
        y_subpart = int((ymax_desc - ymin_desc) * (1 / 3))
        ymin_start = ymin_desc + y_subpart
        ymin_end = ymax_desc
        
        result = {
            'ymin_start': ymin_start,
            'ymin_end': ymin_end
        }
        
        # For descending_aorta_enlargement: rename trachea and desc_aorta points
        if 'trachea_point_right' in source_row and pd.notna(source_row['trachea_point_right']):
            # Parse trachea points: "(x, y)" -> extract x and y
            import ast
            trachea_right = ast.literal_eval(source_row['trachea_point_right'])
            trachea_left = ast.literal_eval(source_row['trachea_point_left'])
            
            result['trachea_y_width'] = trachea_right[1]  # y coordinate
            result['trachea_xmin_width'] = trachea_left[0]  # left x
            result['trachea_xmax_width'] = trachea_right[0]  # right x
            
        if 'desc_aorta_point_right' in source_row and pd.notna(source_row['desc_aorta_point_right']):
            # Rename desc_aorta points to pnt_r_1 and pnt_1
            result['pnt_r_1'] = source_row['desc_aorta_point_right']
            result['pnt_1'] = source_row['desc_aorta_point_left']
        
        # For descending_aorta_tortuous: rename point_N to pnt_r_N
        for i in range(1, 10):
            point_col = f'point_{i}'
            if point_col in source_row and pd.notna(source_row[point_col]):
                result[f'pnt_r_{i}'] = source_row[point_col]
        
        return result
        
    except Exception as e:
        print(f"    Error computing viz for {dicom}: {e}")
        import traceback
        traceback.print_exc()
        return None


# Mapping of diagnostic tasks to computation functions
VIZ_COMPUTERS = {
    'aortic_knob_enlargement': compute_aortic_knob_viz,
    'cardiomegaly': compute_cardiomegaly_viz,
    'mediastinal_widening': compute_mediastinal_widening_viz,
    'inspiration': compute_inspiration_viz,
    'rotation': compute_rotation_viz,
    'projection': compute_projection_viz,
    'carina_angle': compute_carina_angle_viz,
    'trachea_deviation': compute_trachea_deviation_viz,
    'ascending_aorta_enlargement': compute_ascending_aorta_viz,
    'inclusion': compute_inclusion_viz,
    'descending_aorta_enlargement': compute_descending_aorta_viz,
    'descending_aorta_tortuous': compute_descending_aorta_viz,
}


def process_task_with_viz_computation(dx, source_df, cxas_dir):
    """Process a diagnostic task by computing viz fields from CXAS masks
    
    Strategy: Keep ALL images, but mark which ones have valid viz data
    - For Path 1 (reasoning): Use all images (reads from original CSV)
    - For Path 2 (guidance): Filter to only valid viz data
    
    Returns DataFrame with all rows, adding 'has_valid_viz' column
    """
    viz_df = source_df[['image_file', 'viewposition']].copy()
    
    if dx not in VIZ_COMPUTERS:
        viz_df['has_valid_viz'] = True
        return viz_df
    
    print(f"  Computing viz fields from CXAS masks...")
    compute_func = VIZ_COMPUTERS[dx]
    
    # Process all rows, tracking success/failure
    viz_data = []
    valid_count = 0
    failed_count = 0
    
    for idx, row in tqdm(source_df.iterrows(), total=len(source_df), desc=f"  {dx}"):
        dicom = row['image_file']
        viz_fields = compute_func(dicom, cxas_dir, row)
        
        if viz_fields and all(v is not None for v in viz_fields.values()):
            # Successfully computed with no NaN values
            viz_fields['has_valid_viz'] = True
            valid_count += 1
        else:
            # Computation failed - keep row but mark as invalid
            viz_fields = {'has_valid_viz': False}
            failed_count += 1
        
        viz_data.append(viz_fields)
    
    # Report statistics
    failure_rate = (failed_count / len(source_df)) * 100
    print(f"  Results: {valid_count} valid, {failed_count} invalid ({failure_rate:.1f}% failure rate)")
    
    if failure_rate > 10:
        print(f"  ⚠️  WARNING: High failure rate! Consider checking CXAS data quality")
    
    # Merge computed viz fields
    viz_fields_df = pd.DataFrame(viz_data)
    viz_df = pd.concat([viz_df, viz_fields_df], axis=1)
    
    return viz_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx_by_dicoms_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--nih_csv_dir', type=str, required=True)
    parser.add_argument('--cxas_dir', type=str, default='d:/CXReasonBench/CXAS/cxas',
                       help='Directory containing CXAS segmentation masks')
    args = parser.parse_args()
    
    # Load dx_by_dicoms.json
    with open(args.dx_by_dicoms_file, 'r') as f:
        dx_by_dicoms = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # For each diagnostic task, create a CSV with visualization fields
    for dx, dicom_list in dx_by_dicoms.items():
        print(f"\nProcessing {dx}: {len(dicom_list)} images")
        
        # Load source CSV
        source_csv = os.path.join(args.nih_csv_dir, f'{dx}.csv')
        if not os.path.exists(source_csv):
            print(f"  Warning: Source CSV not found: {source_csv}")
            # Create minimal viz CSV
            viz_df = pd.DataFrame({'image_file': dicom_list, 'viewposition': 'PA'})
        else:
            source_df = pd.read_csv(source_csv)
            
            # Filter to only requested dicoms
            source_df = source_df[source_df['image_file'].isin(dicom_list)]
            print(f"  Found {len(source_df)} images in source CSV")
            
            # Process with viz computation
            viz_df = process_task_with_viz_computation(dx, source_df, args.cxas_dir)
        
        # Save to CSV
        output_file = os.path.join(args.output_dir, f'{dx}.csv')
        viz_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file} ({len(viz_df)} rows, {len(viz_df.columns)} columns)")
    
    print(f"\n✅ Created {len(dx_by_dicoms)} visualization CSV files in {args.output_dir}")


if __name__ == "__main__":
    main()
