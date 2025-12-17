"""
Convert Hugging Face CheXStruct CSV format to bodypart_segmask.py expected format
Creates _viz directory with properly formatted CSV files
"""
import os
import pandas as pd
import argparse
from pathlib import Path

def convert_cardiomegaly(df):
    """Convert cardiomegaly CSV to viz format"""
    viz_df = pd.DataFrame()
    viz_df['image_file'] = df['image_file']
    
    # Copy all available columns
    for col in df.columns:
        if col != 'image_file':
            viz_df[col] = df[col]
    
    # Rename columns if needed
    if 'lung_xmin' in df.columns:
        viz_df['xmin_lung'] = df['lung_xmin']
        viz_df['xmax_lung'] = df['lung_xmax']
    
    return viz_df

def convert_inclusion(df):
    """Convert inclusion CSV to viz format"""
    viz_df = pd.DataFrame()
    viz_df['image_file'] = df['image_file']
    
    # Copy all relevant columns
    for col in df.columns:
        if col != 'image_file':
            viz_df[col] = df[col]
    
    return viz_df

def convert_generic(df, task_name):
    """Generic converter for tasks that don't need special handling"""
    viz_df = pd.DataFrame()
    viz_df['image_file'] = df['image_file']
    
    # Copy all columns
    for col in df.columns:
        if col != 'image_file':
            viz_df[col] = df[col]
    
    return viz_df

def main():
    parser = argparse.ArgumentParser(description='Convert Hugging Face CSV to viz format')
    parser.add_argument('--hf_csv_dir', type=str, required=True,
                        help='Directory containing Hugging Face NIH CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for viz CSV files')
    parser.add_argument('--add_cxas_coords', action='store_true',
                        help='Extract y coordinates from CXAS masks (slower)')
    parser.add_argument('--cxas_base_dir', type=str, default=None,
                        help='Path to CXAS segmentation results (required if --add_cxas_coords)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Task-specific converters
    converters = {
        'cardiomegaly': convert_cardiomegaly,
        'inclusion': convert_inclusion,
    }
    
    # List of diagnostic tasks
    tasks = [
        'inclusion', 'inspiration', 'rotation', 'projection',
        'cardiomegaly', 'mediastinal_widening', 'carina_angle', 'trachea_deviation',
        'aortic_knob_enlargement', 'ascending_aorta_enlargement',
        'descending_aorta_enlargement', 'descending_aorta_tortuous'
    ]
    
    print(f"Converting Hugging Face CSV files from {args.hf_csv_dir} to viz format...")
    print(f"Output directory: {args.output_dir}")
    print()
    
    for task in tasks:
        input_file = os.path.join(args.hf_csv_dir, f'{task}.csv')
        
        if not os.path.exists(input_file):
            print(f"⚠️  Skipping {task}: file not found")
            continue
        
        # Read input CSV
        df = pd.read_csv(input_file)
        print(f"Processing {task}: {len(df)} rows")
        
        # Convert using appropriate converter
        if task in converters:
            viz_df = converters[task](df)
        else:
            viz_df = convert_generic(df, task)
        
        # Add CXAS coordinates if requested
        if args.add_cxas_coords and args.cxas_base_dir:
            viz_df = add_cxas_coordinates(viz_df, task, args.cxas_base_dir)
        
        # Save output
        output_file = os.path.join(args.output_dir, f'{task}.csv')
        viz_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
    
    print(f"\n✓ Conversion complete! Created {len(tasks)} viz CSV files in {args.output_dir}")
    print(f"\nNote: Some tasks may need additional coordinate extraction from CXAS masks.")
    print(f"      Run with --add_cxas_coords to extract missing coordinates (this is slower).")

def add_cxas_coordinates(df, task, cxas_base_dir):
    """Extract missing coordinates from CXAS segmentation masks"""
    import cv2
    import numpy as np
    from tqdm import tqdm
    from scipy import ndimage
    
    print(f"    Extracting coordinates from CXAS masks...")
    
    def get_mask_bounds(mask):
        """Get bounding box and other metrics from mask"""
        if mask is None or mask.sum() == 0:
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return {'ymin': int(ymin), 'ymax': int(ymax), 'xmin': int(xmin), 'xmax': int(xmax)}
    
    def select_max_area_mask(mask):
        """Select the largest connected component"""
        label_im, nb_labels = ndimage.label(mask)
        max_area = 0
        max_mask = mask
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(int)
            if separate_mask.sum() > max_area:
                max_area = separate_mask.sum()
                max_mask = separate_mask
        return max_mask
    
    def select_max_width_mask(mask):
        """Select the widest connected component"""
        label_im, nb_labels = ndimage.label(mask)
        max_mask = mask
        max_width = 0
        coordinates = (0, 0, 0, 0)
        for i in range(nb_labels):
            mask_compare = np.full(np.shape(label_im), i + 1)
            separate_mask = np.equal(label_im, mask_compare).astype(np.uint8)
            x_indices = separate_mask.sum(axis=0).nonzero()[0]
            y_indices = separate_mask.sum(axis=1).nonzero()[0]
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x1, x2 = x_indices[0], x_indices[-1]
            y1, y2 = y_indices[0], y_indices[-1]
            width = abs(x_indices[0] - x_indices[-1]) + 1
            if width > max_width:
                max_width = width
                max_mask = separate_mask
                coordinates = (x1, y1, x2, y2)
        return max_mask, coordinates
    
    if task == 'cardiomegaly':
        # Extract heart bounding box y coordinates
        coords_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            heart_mask_path = os.path.join(cxas_base_dir, dicom, 'heart.png')
            
            if os.path.exists(heart_mask_path):
                mask = cv2.imread(heart_mask_path, 0)
                bounds = get_mask_bounds(mask)
                if bounds:
                    coords_list.append(bounds)
                else:
                    coords_list.append({'ymin': 0, 'ymax': 0, 'xmin': 0, 'xmax': 0})
            else:
                coords_list.append({'ymin': 0, 'ymax': 0, 'xmin': 0, 'xmax': 0})
        
        df['heart_ymin'] = [c['ymin'] for c in coords_list]
        df['heart_ymax'] = [c['ymax'] for c in coords_list]
        
        # Create coord_mask column as string
        df['coord_mask'] = df.apply(
            lambda row: f"[{int(row['heart_xmin'])}, {int(row['heart_ymin'])}, "
                       f"{int(row['heart_xmax'])}, {int(row['heart_ymax'])}]",
            axis=1
        )
    
    elif task == 'aortic_knob_enlargement':
        # Extract descending aorta and aortic arch coordinates
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            desc_aorta_path = os.path.join(cxas_base_dir, dicom, 'descending aorta.png')
            aortic_arch_path = os.path.join(cxas_base_dir, dicom, 'aortic arch.png')
            trachea_path = os.path.join(cxas_base_dir, dicom, 'trachea.png')
            
            coords = {}
            
            # Descending aorta
            if os.path.exists(desc_aorta_path):
                mask = cv2.imread(desc_aorta_path, 0)
                bounds = get_mask_bounds(mask)
                if bounds:
                    coords['ymin_desc'] = bounds['ymin']
                    coords['y_max'] = bounds['ymax']
                    coords['ysub_desc'] = bounds['ymax'] - bounds['ymin']
                else:
                    coords['ymin_desc'] = 0
                    coords['y_max'] = 0
                    coords['ysub_desc'] = 0
            else:
                coords['ymin_desc'] = 0
                coords['y_max'] = 0
                coords['ysub_desc'] = 0
            
            # Aortic arch
            if os.path.exists(aortic_arch_path):
                mask = cv2.imread(aortic_arch_path, 0)
                bounds = get_mask_bounds(mask)
                if bounds:
                    coords['ymax_desc_sub'] = bounds['ymax']
                else:
                    coords['ymax_desc_sub'] = 0
            else:
                coords['ymax_desc_sub'] = 0
            
            # Trachea
            if os.path.exists(trachea_path):
                mask = cv2.imread(trachea_path, 0)
                bounds = get_mask_bounds(mask)
                if bounds:
                    coords['xmax_trachea_mean'] = bounds['xmax']
                else:
                    coords['xmax_trachea_mean'] = 0
            else:
                coords['xmax_trachea_mean'] = 0
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task == 'ascending_aorta_enlargement':
        # Extract pnt_heart and pnt_trachea coordinates
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            heart_path = os.path.join(cxas_base_dir, dicom, 'heart.png')
            trachea_path = os.path.join(cxas_base_dir, dicom, 'trachea.png')
            
            coords = {}
            
            # Heart top center point
            if os.path.exists(heart_path):
                mask = cv2.imread(heart_path, 0)
                bounds = get_mask_bounds(mask)
                if bounds:
                    x_center = (bounds['xmin'] + bounds['xmax']) // 2
                    coords['pnt_heart'] = str([x_center, bounds['ymin']])
                else:
                    coords['pnt_heart'] = str([0, 0])
            else:
                coords['pnt_heart'] = str([0, 0])
            
            # Trachea bottom center point
            if os.path.exists(trachea_path):
                mask = cv2.imread(trachea_path, 0)
                bounds = get_mask_bounds(mask)
                if bounds:
                    x_center = (bounds['xmin'] + bounds['xmax']) // 2
                    coords['pnt_trachea'] = str([x_center, bounds['ymax']])
                else:
                    coords['pnt_trachea'] = str([0, 0])
            else:
                coords['pnt_trachea'] = str([0, 0])
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task in ['descending_aorta_enlargement', 'descending_aorta_tortuous']:
        # Extract ymin_start and ymin_end from descending aorta
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            desc_aorta_path = os.path.join(cxas_base_dir, dicom, 'descending aorta.png')
            heart_path = os.path.join(cxas_base_dir, dicom, 'heart.png')
            
            coords = {}
            
            if os.path.exists(desc_aorta_path):
                mask_desc = cv2.imread(desc_aorta_path, 0)
                # Apply heart mask refinement similar to bodypart_segmask.py
                if os.path.exists(heart_path):
                    mask_heart = cv2.imread(heart_path, 0)
                    # Refine mask - keep parts overlapping with heart
                    mask_refined = select_max_area_mask(mask_desc & mask_heart)
                    if mask_refined.sum() > 0:
                        y_indices = mask_refined.sum(axis=-1).nonzero()[0]
                        if len(y_indices) > 0:
                            ymin = y_indices[0]
                            ymax = y_indices[-1]
                            height = ymax - ymin
                            coords['ymin_start'] = int(ymin)
                            coords['ymin_end'] = int(ymin + height * 0.8)  # Use 80% of height
                        else:
                            coords['ymin_start'] = 0
                            coords['ymin_end'] = 0
                    else:
                        coords['ymin_start'] = 0
                        coords['ymin_end'] = 0
                else:
                    coords['ymin_start'] = 0
                    coords['ymin_end'] = 0
            else:
                coords['ymin_start'] = 0
                coords['ymin_end'] = 0
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task == 'inclusion':
        # coord_mask (lung bounds) and xmin_lung, xmax_lung
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            rlung_path = os.path.join(cxas_base_dir, dicom, 'right lung.png')
            llung_path = os.path.join(cxas_base_dir, dicom, 'left lung.png')
            
            coords = {}
            
            if os.path.exists(rlung_path) and os.path.exists(llung_path):
                mask_rlung = cv2.imread(rlung_path, 0)
                mask_llung = cv2.imread(llung_path, 0)
                mask_rlung = select_max_area_mask(mask_rlung)
                mask_llung = select_max_area_mask(mask_llung)
                mask_lungs = mask_rlung | mask_llung
                
                bounds = get_mask_bounds(mask_lungs)
                if bounds:
                    coords['coord_mask'] = str([bounds['xmin'], bounds['ymin'], bounds['xmax'], bounds['ymax']])
                    coords['xmin_lung'] = bounds['xmin']
                    coords['xmax_lung'] = bounds['xmax']
                else:
                    coords['coord_mask'] = str([0, 0, 0, 0])
                    coords['xmin_lung'] = 0
                    coords['xmax_lung'] = 0
            else:
                coords['coord_mask'] = str([0, 0, 0, 0])
                coords['xmin_lung'] = 0
                coords['xmax_lung'] = 0
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task == 'inspiration':
        # x_lung_mid_combined, lung_y_lst_x_lung_mid
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            rlung_path = os.path.join(cxas_base_dir, dicom, 'right lung.png')
            llung_path = os.path.join(cxas_base_dir, dicom, 'left lung.png')
            
            coords = {}
            
            if os.path.exists(rlung_path) and os.path.exists(llung_path):
                mask_rlung = cv2.imread(rlung_path, 0)
                mask_llung = cv2.imread(llung_path, 0)
                mask_rlung = select_max_area_mask(mask_rlung)
                mask_llung = select_max_area_mask(mask_llung)
                
                # Find medial edges
                rlung_x = mask_rlung.sum(axis=0).nonzero()[0]
                llung_x = mask_llung.sum(axis=0).nonzero()[0]
                
                if len(rlung_x) > 0 and len(llung_x) > 0:
                    xmax_rlung = rlung_x[-1]
                    xmin_llung = llung_x[0]
                    x_lung_mid = (xmax_rlung + xmin_llung) // 2
                    
                    # Get y coordinates at midline
                    mask_lungs = mask_rlung | mask_llung
                    y_lst = mask_lungs[:, x_lung_mid].nonzero()[0]
                    
                    if len(y_lst) > 0:
                        coords['x_lung_mid_combined'] = int(x_lung_mid)
                        coords['lung_y_lst_x_lung_mid'] = str(y_lst.tolist())
                    else:
                        coords['x_lung_mid_combined'] = 0
                        coords['lung_y_lst_x_lung_mid'] = str([0])
                else:
                    coords['x_lung_mid_combined'] = 0
                    coords['lung_y_lst_x_lung_mid'] = str([0])
            else:
                coords['x_lung_mid_combined'] = 0
                coords['lung_y_lst_x_lung_mid'] = str([0])
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task == 'mediastinal_widening':
        # y_medi, xmin_rlung, xmax_llung
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            rlung_path = os.path.join(cxas_base_dir, dicom, 'right lung.png')
            llung_path = os.path.join(cxas_base_dir, dicom, 'left lung.png')
            
            coords = {}
            
            if os.path.exists(rlung_path) and os.path.exists(llung_path):
                mask_rlung = cv2.imread(rlung_path, 0)
                mask_llung = cv2.imread(llung_path, 0)
                
                # Get medial edges (closest to midline)
                rlung_x = mask_rlung.sum(axis=0).nonzero()[0]
                llung_x = mask_llung.sum(axis=0).nonzero()[0]
                
                if len(rlung_x) > 0 and len(llung_x) > 0:
                    xmin_rlung = rlung_x[0]
                    xmax_llung = llung_x[-1]
                    
                    # Find vertical center
                    mask_lungs = mask_rlung | mask_llung
                    y_indices = mask_lungs.sum(axis=-1).nonzero()[0]
                    if len(y_indices) > 0:
                        y_medi = int((y_indices[0] + y_indices[-1]) // 2)
                    else:
                        y_medi = 0
                    
                    coords['y_medi'] = y_medi
                    coords['xmin_rlung'] = int(xmin_rlung)
                    coords['xmax_llung'] = int(xmax_llung)
                else:
                    coords['y_medi'] = 0
                    coords['xmin_rlung'] = 0
                    coords['xmax_llung'] = 0
            else:
                coords['y_medi'] = 0
                coords['xmin_rlung'] = 0
                coords['xmax_llung'] = 0
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task == 'rotation':
        # Calculate slope and intercept for clavicle lines
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            rclavicle_path = os.path.join(cxas_base_dir, dicom, 'clavicle right.png')
            lclavicle_path = os.path.join(cxas_base_dir, dicom, 'clavicle left.png')
            spine_path = os.path.join(cxas_base_dir, dicom, 'spine.png')
            
            coords = {}
            
            # Process each clavicle
            for side, clavicle_path in [('right', rclavicle_path), ('left', lclavicle_path)]:
                if os.path.exists(clavicle_path):
                    mask = cv2.imread(clavicle_path, 0)
                    mask_refined, _ = select_max_width_mask(mask)
                    
                    # Get upper and lower bounds
                    y_indices = mask_refined.sum(axis=-1).nonzero()[0]
                    if len(y_indices) >= 2:
                        ymin = y_indices[0]
                        ymax = y_indices[-1]
                        
                        # Get x coordinates at these y positions
                        x_at_ymin = mask_refined[ymin, :].nonzero()[0]
                        x_at_ymax = mask_refined[ymax, :].nonzero()[0]
                        
                        if len(x_at_ymin) > 0 and len(x_at_ymax) > 0:
                            x1 = int(np.mean(x_at_ymin))
                            x2 = int(np.mean(x_at_ymax))
                            
                            # Calculate slope
                            if x2 != x1:
                                slope = (ymax - ymin) / (x2 - x1)
                            else:
                                slope = 0
                            
                            intercept_min = ymin - slope * x1
                            intercept_max = ymax - slope * x2
                            
                            coords[f'{side}_slope'] = float(slope)
                            coords[f'{side}_intercept_min'] = float(intercept_min)
                            coords[f'{side}_intercept_max'] = float(intercept_max)
                        else:
                            coords[f'{side}_slope'] = 0
                            coords[f'{side}_intercept_min'] = 0
                            coords[f'{side}_intercept_max'] = 0
                    else:
                        coords[f'{side}_slope'] = 0
                        coords[f'{side}_intercept_min'] = 0
                        coords[f'{side}_intercept_max'] = 0
                else:
                    coords[f'{side}_slope'] = 0
                    coords[f'{side}_intercept_min'] = 0
                    coords[f'{side}_intercept_max'] = 0
            
            # Get spine midline coordinates
            if os.path.exists(spine_path):
                mask_spine = cv2.imread(spine_path, 0)
                if mask_spine.sum() > 0:
                    y_indices = mask_spine.sum(axis=-1).nonzero()[0]
                    if len(y_indices) > 0:
                        target_coords = []
                        for y in y_indices:
                            x_vals = mask_spine[y, :].nonzero()[0]
                            if len(x_vals) > 0:
                                target_coords.append([int(np.mean(x_vals)), int(y)])
                        coords['target_coords'] = str(target_coords) if target_coords else str([[0, 0]])
                        
                        # Calculate line fit
                        if len(target_coords) > 1:
                            points = np.array(target_coords)
                            # Fit line: y = mx + b
                            A = np.vstack([points[:, 0], np.ones(len(points))]).T
                            m, c = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
                            coords['m'] = float(m)
                            coords['b'] = float(c)
                        else:
                            coords['m'] = 0
                            coords['b'] = 0
                    else:
                        coords['target_coords'] = str([[0, 0]])
                        coords['m'] = 0
                        coords['b'] = 0
                else:
                    coords['target_coords'] = str([[0, 0]])
                    coords['m'] = 0
                    coords['b'] = 0
            else:
                coords['target_coords'] = str([[0, 0]])
                coords['m'] = 0
                coords['b'] = 0
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task == 'trachea_deviation':
        # midline_x_min, midline_x_max, y_min, y_max
        result_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"    {task}"):
            dicom = row['image_file']
            trachea_path = os.path.join(cxas_base_dir, dicom, 'trachea.png')
            carina_path = os.path.join(cxas_base_dir, dicom, 'tracheal bifurcation.png')
            
            coords = {}
            
            if os.path.exists(trachea_path) and os.path.exists(carina_path):
                mask_trachea = cv2.imread(trachea_path, 0)
                mask_carina = cv2.imread(carina_path, 0)
                
                # Cut trachea at carina
                carina_y = mask_carina.sum(axis=-1).nonzero()[0]
                if len(carina_y) > 0:
                    y_pos_carina = carina_y[0]
                    mask_refined = mask_trachea.copy()
                    mask_refined[y_pos_carina:] = 0
                else:
                    mask_refined = mask_trachea
                
                mask_refined = select_max_area_mask(mask_refined)
                
                # Get upper 3/4 region
                y_indices = mask_refined.sum(axis=-1).nonzero()[0]
                if len(y_indices) > 0:
                    y_min = y_indices[0]
                    y_max = y_indices[-1]
                    y_subpart = int((y_max - y_min) * 0.25)
                    y_min_adjusted = y_min + y_subpart
                    
                    # Get x coordinates at top and bottom
                    x_top = mask_refined[y_min_adjusted, :].nonzero()[0]
                    x_bottom = mask_refined[y_max, :].nonzero()[0]
                    
                    if len(x_top) > 0 and len(x_bottom) > 0:
                        coords['midline_x_min'] = int(np.mean(x_top))
                        coords['midline_x_max'] = int(np.mean(x_bottom))
                        coords['y_min'] = int(y_min_adjusted)
                        coords['y_max'] = int(y_max)
                    else:
                        coords['midline_x_min'] = 0
                        coords['midline_x_max'] = 0
                        coords['y_min'] = 0
                        coords['y_max'] = 0
                else:
                    coords['midline_x_min'] = 0
                    coords['midline_x_max'] = 0
                    coords['y_min'] = 0
                    coords['y_max'] = 0
            else:
                coords['midline_x_min'] = 0
                coords['midline_x_max'] = 0
                coords['y_min'] = 0
                coords['y_max'] = 0
            
            result_rows.append(coords)
        
        for key in result_rows[0].keys():
            df[key] = [r[key] for r in result_rows]
    
    elif task == 'carina_angle':
        # No additional coords needed - uses CXAS directly
        print(f"    Task '{task}' uses CXAS masks directly, no coordinate extraction needed")
    
    elif task == 'projection':
        # No additional coords needed - just label
        print(f"    Task '{task}' uses label only, no coordinate extraction needed")
    
    return df

if __name__ == '__main__':
    main()
