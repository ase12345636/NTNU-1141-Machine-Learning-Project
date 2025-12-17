# CheXStruct


The dataset extracted using the CheXStruct pipeline is publicly available at: 👉 https://physionet.org/content/chexstruct-cxreasonbench 

This README provides instructions for reproducing the dataset or running the CheXStruct pipeline.

For implementation details and the full design of each pipeline step, refer to Appendix A: Details of CheXStruct in the [paper](https://arxiv.org/pdf/2505.18087).

## Environment Setup


### Conda Environment

Follow the steps below to set up the required environment for running the chexstruct pipeline:

```bash
# Create a conda environment
conda create -n chexstruct python=3.9

# Activate the environment
conda activate chexstruct

# Install required packages
pip install -r requirements.txt
```


## Prerequisites

Before running the CheXStruct pipeline, the required datasets should be downloaded and prepared in advance.


### 1. Get PhysioNet Access

To download the datasets, you first need credentialed access from PhysioNet.
Follow the instructions here to complete the credentialing process:
👉 https://physionet.org/settings/credentialing/

Once your account is approved, log in to PhysioNet and download the necessary datasets listed below.

### 2. Download MIMIC-CXR-JPG

Download the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) dataset from PhysioNet.
This serves as the base dataset containing chest X-ray images and corresponding metadata.

### 3. Generate Anatomical Segmentation Masks

Use the official implementation from the [Chest X-ray Anatomy Segmentation (CXAS)](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation) repository to generate anatomical segmentation masks for MIMIC-CXR-JPG.

For efficiency, anatomical segmentation masks are generated first using the CXAS model before running the CheXStruct pipeline.

Each mask represents key anatomical regions such as the heart, lungs, diaphragm, and trachea.
Masks should be saved in the following structure:

```
<path_to_cxas_output_dir>/
 ├── <dicom_id>/
 │    ├── heart.png
 │    ├── right lung.png
 │    ├── diaphragm.png
 │    └── ...
```

### 4. Download and Convert CheXmask Database

Download the [CheXmask Database](https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/OriginalResolution/#files-panel) from PhysioNet.
The dataset provides segmentation masks in Run-Length Encoding (RLE) format, so for convenience when running the CheXStruct pipeline, you need to convert them into .png images.

Run the following script to perform the conversion:
```
python utils/convert_chexmask2png.py \
    --save_base_dir <path_to_chexmask_output_dir> \
    --chexmask_meta_file <path_to_physionet_download_dir>/physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/OriginalResolution/MIMIC-CXR-JPG.csv
```

This dataset is used in the CheXStruct pipeline to extract inspiration level information.

### 5. Download Chest ImaGenome Dataset

Download the [Chest ImaGenome Dataset](https://physionet.org/content/chest-imagenome/1.0.0/) from PhysioNet.

This dataset is used in the CheXStruct pipeline to extract information related to cardiomegaly and mediastinal widening.


## Run the CheXStruct Pipeline

### Configurations
`main.py` uses pre-defined arguments for the scripts. 

Below are the key configuration options:

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset_name` | Dataset Name (default) | `mimic-cxr-jpg` |
| `--save_base_dir` | Base directory for the extraction results; folder where outputs will be saved | `chexstruct_output` |
| `--mimic_cxr_base` | Path to the MIMIC-CXR-JPG dataset | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/files` |
| `--mimic_meta_file` | Path to the MIMIC-CXR-JPG meta file | `<path_to_physionet_download_dir>/physionet.org/files/mimic-cxr-jpg/<version>/mimic-cxr-2.0.0-metadata.csv` |
| `--cxas_base_dir` | Path to the anatomical segmentation results (from Prerequisite 2) | `<path_to_cxas_output_dir>` |
| `--chexmask_base_dir` | Path to the converted CheXmask dataset (from Prerequisite 3)| `<path_to_chexmask_output_dir>` |
| `--chest_imagenome_base_dir` | Path to the Chest ImaGenome dataset | `<path_to_physionet_download_dir>/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_graph` |

---
### Run the Pipeline
Once all paths are properly configured, execute the full pipeline using:

```
# Activate the environment
conda activate chexstruct

# Navigate to the CheXStruct folder
cd CXReasonBench/CheXStruct

# Run the script
python main.py
```

---
### Output Structure
After running `main.py`, all outputs will be stored under the directory specified by `--save_base_dir`:
```
<save_base_dir>/
 ├── <dataset_name>/
 │    ├── cardiomegaly.csv
 │    ├── inclusion.csv
 │    ├── trachea_deviation.csv
 │    └── ...
 └── <dataset_name>_viz/
      ├── cardiomegaly.csv
      ├── inclusion.csv
      ├── trachea_deviation.csv
      └── ...
```

- The `*_viz` folders (e.g., `mimic-cxr-jpg_viz/`) contain visualization metadata required when running the scripts located in `CXReasonBench/Benchmark/generation/` — namely, `bodypart_segmask.py` and `point_on_cxr.py`.
These files are used internally for anatomical highlighting and detailed landmark visualization within the CXReasonBench generation process.
