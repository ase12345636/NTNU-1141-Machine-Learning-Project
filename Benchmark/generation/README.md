# CXReasonBench Generation


This README provides instructions and scripts for generating the CXReasonBench benchmark dataset.
However, the dataset is already available on PhysioNet, so if you only want to use it, you can skip running these scripts.

For instructions on downloading CXReasonBench and evaluating models, see [/CXReasonBench/Benchmark/Evaluation/README.md](https://github.com/ttumyche/CXReasonBench/blob/main/Benchmark/Evaluation/README.md).


# Environment Setup

Follow the steps below to set up the required environment for running the scripts.

```bash
# Create a conda environment
conda create -n cxrb_gen python=3.9

# Activate the environment
conda activate cxrb_gen

# Install required packages
pip install -r requirements.txt
```
**Note**: For anatomical and landmark visualizations (needed when running `bodypart_segmask.py` and `point_on_cxr.py`), you also need detectron2. Follow the official installation guide: 👉 https://detectron2.readthedocs.io/en/latest/tutorials/install.html 

# Usage Options

There are two main ways to generate the CXReasonBench benchmark dataset.

## Option 1. Using precomputed results from PhysioNet

### (1) **Download the precomputed outputs** from PhysioNet.
   
The files are located at: `<path_to_physionet_download_dir>/physionet.org/files/chexstruct-cxreasonbench/<version>/`

This precomputed download contains:
- CXReasonBench folder (`CXReasonBench/`) with:
    - segmask_bodypart.zip — Contains chest X-ray images overlaid with anatomical segmentation masks.
    - pnt_on_cxr.zip — Contains chest X-ray images with overlaid anatomical landmarks
    - dx_by_dicoms.json — This file contains a dictionary listing DICOM IDs for each diagnostic task used in CXReasonBench. 

- CheXStruct dataset (`CheXStruct/diagnostic_tasks/`): Containing structured outputs extracted by the CheXStruct pipeline.

Unzip the .zip files in the CXReasonBench/ folder. All contents, including CheXStruct dataset, will be used by `generate_benchmark.py`.

   
### (2) **Run `generate_benchmark.py`** using the downloaded files.

Below are the key configuration options used to run the script:

| Argument | Description | Example |
|----------|-------------|---------|
| `--inference_path` | Choice of path for benchmark generation: 'path1', 'path2', 're-path1' | `path1` |
| `--save_base_dir` | Directory to save generated benchmark | `cxreasonbench` |
| `--chexstruct_base_dir` | Directory containing the downloaded CheXStruct dataset | `<path_to_physionet_download_dir>/physionet.org/files/chexstruct-cxreasonbench/<version>/CheXStruct/diagnostic_tasks` |
| `--cxreasonbench_base_dir` | Directory containing the downloaded CXReasonBench metadata (pnt_on_cxr, segmask_bodypart, dx_by_dicoms.json) | `<path_to_physionet_download_dir>/physionet.org/files/chexstruct-cxreasonbench/<version>/CXReasonBench` |
| `--mimic_cxr_base` | Path to the MIMIC-CXR-JPG dataset | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files` |
| `--mimic_meta_file` | Path to the MIMIC-CXR-JPG meta file | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv` |

- **Note on `--inference_path` choices:**
    - `path1`: Path1. Direct Reasoning Process Evaluation  
    - `path2`: Path2. Guided Reasoning and Re-evaluation (guidance part)  
    - `re-path1`: Path2. Guided Reasoning and Re-evaluation (re-evaluation part)

Once all arguments are properly configured, simply run: `python generate_benchmark.py`

---

## Option 2. Generate benchmark from scratch

### (1) Run the CheXStruct Pipeline

Refer to `/CXReasonBench/CheXStruct/README.md` to run the pipeline (`main.py)`.
The `--save_base_dir` used in this step will be the same as the `--saved_base_dir` for the following scripts.

### (2) Run the `bodypart_segmask.py` script

This script highlights anatomical regions (e.g., heart, trachea) and reference lines (e.g., spinous process line) on chest X-rays.
Used in Stage 2 of Path 1, the re-evaluation of Path 1, and Stage 1 of Path 2.

Below are the key configuration options used to run the script

| Argument | Description | Example |
|----------|-------------|---------|
| `--save_base_dir` | Directory to save results. | `result` |
| `--saved_base_dir` | Path to the base directory specified by `CXReasonBench/CheXStruct/main.py --save_base_dir`  | `chexstruct_output` |
| `--dataset_name` | Dataset Name (default) | `mimic-cxr-jpg` |
| `--mimic_cxr_base_dir` | Path to the MIMIC-CXR-JPG dataset | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files` |
| `--mimic_meta_file` | Path to the MIMIC-CXR-JPG meta file | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv` |
| `--cxas_base_dir` | Path to the anatomical segmentation results | `<path_to_cxas_output_dir>` |
| `--chexmask_base_dir` | Path to the converted CheXmask dataset| `<path_to_chexmask_output_dir>` |

Once all arguments are properly configured, simply run: `python bodypart_segmask.py`


### (3) Run the `point_on_cxr.py` script

This script generates detailed visual annotations used in Path 2, Stage 2: Guided Measurement or Recognition, such as labeled landmarks and coordinates on chest X-rays.


Below are the key configuration options used to run the script:

| Argument | Description | Example |
|----------|-------------|---------|
| `--save_base_dir` | Directory to save results. | `result` |
| `--saved_base_dir` | Path to the base directory specified by `CXReasonBench/CheXStruct/main.py --save_base_dir`  | `chexstruct_output` |
| `--dataset_name` | Dataset Name (default) | `mimic-cxr-jpg` |
| `--mimic_cxr_base_dir` | Path to the MIMIC-CXR-JPG dataset | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files` |
| `--mimic_meta_file` | Path to the MIMIC-CXR-JPG meta file | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv` |
| `--cxas_base_dir` | Path to the anatomical segmentation results | `<path_to_cxas_output_dir>` |

Once all arguments are properly configured, simply run: `python point_on_cxr.py`

### (4) Run the `generate_benchmark.py` script

Uses the outputs from the previous steps to generate the CXReasonBench benchmark dataset.

Required arguments can be found in Option 1 - (2).

**Note on dx_by_dicoms.json:**
This file contains a dictionary listing the DICOM IDs for each diagnostic task used in CXReasonBench.
If you skip Option 1 and want to run the pipeline from scratch (Option 2), you can either use the precomputed `dx_by_dicoms.json` or create your own dictionary with DICOM IDs of your choice for generating a custom benchmark.