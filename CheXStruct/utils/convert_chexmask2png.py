import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_base_dir', default='path/to/save/results', type=str)
    parser.add_argument('--chexmask_meta_file', type=str, default='<path_to_physionet_download_dir>/physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/OriginalResolution/MIMIC-CXR-JPG.csv')

    args = parser.parse_args()
    return args

def get_mask_from_RLE(rle, height, width):
    runs = np.array([int(x) for x in rle.split()])
    starts = runs[::2]
    lengths = runs[1::2]

    mask = np.zeros((height * width), dtype=np.uint8)

    for start, length in zip(starts, lengths):
        start -= 1
        end = start + length
        mask[start:end] = 1

    mask = mask.reshape((height, width))

    return mask

def convert(args):
    print('Loading chexmask df ...')
    start = time.time()
    chunk = pd.read_csv(args.chexmask_meta_file, chunksize=50000)
    chexmask_df = pd.concat(chunk)
    print('chexmask df loaded:', time.time() - start)

    target_dicoms = chexmask_df['dicom_id']
    for dicom in tqdm(target_dicoms, total=len(target_dicoms)):
        target_df = chexmask_df[chexmask_df['dicom_id'] == dicom]
        height, width = target_df["Height"].values[0], target_df["Width"].values[0]

        RLMask_RLE = target_df[f"Right Lung"].values[0]
        if type(RLMask_RLE) != float:
            RLMask = get_mask_from_RLE(RLMask_RLE, height, width)
            save_dir = os.path.join(args.save_base_dir, 'RL')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(f'{save_dir}/{dicom}.png'), RLMask)

        LLMask_RLE = target_df[f"Left Lung"].values[0]
        if type(LLMask_RLE) != float:
            LLMask = get_mask_from_RLE(LLMask_RLE, height, width)
            save_dir = os.path.join(args.save_base_dir, 'LL')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(f'{save_dir}/{dicom}.png'), LLMask)

        Heart_RLE = target_df[f"Heart"].values[0]
        if type(Heart_RLE) != float:
            Heart = get_mask_from_RLE(Heart_RLE, height, width)
            save_dir = os.path.join(args.save_base_dir, 'Heart')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(f'{save_dir}/{dicom}.png'), Heart)

if __name__ == "__main__":
    args = config()
    convert(args)
