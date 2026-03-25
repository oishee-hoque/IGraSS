# [IGraSS: Iterative Graph-constrained Semantic Segmentation](https://www.ijcai.org/proceedings/2025/1076) (📍 IJCAI 2025 )

This repository contains an implementation of an iterative segmentation pipeline for extracting irrigation canal networks from satellite imagery.

At a high level, the pipeline repeats three stages:
1. **Generate patch datasets** from large `.npy` imagery and mask arrays.
2. **Train a segmentation model** (ResUNet, DeepLabV3+, or ResNet50-UNet).
3. **Refine ground-truth masks** using graph reachability constraints, then feed refined labels into the next iteration.

The orchestration entry point is `Framework/run_framework.py`.  

## Repository structure

```text
Framework/
├── run_framework.py        # End-to-end iterative pipeline CLI
├── generate_data.py        # Patch extraction and dataset generation
├── resunet_train.py        # Model training routine
├── run_single_process.py   # Prediction + graph-based refinement logic
├── ResUNet.py              # ResUNet model definition
├── deeplabModelV3.py       # DeepLabV3+ model definition
├── network_utils.py        # Graph / connectivity utility functions
├── path_utils.py           # Data handling helpers (patching, dilation, loaders)
├── utils.py                # Dataset + preprocessing utilities for training
└── refined_metrcis.py      # Custom segmentation metrics
```

## How the pipeline works

`run_framework.py` loops for `--iterations` and executes behavior based on `--process_type`:

- `d`: generate training patches only.
- `t`: train model only.
- `f`: full pipeline (generate patches, train, refine GT).

In full mode (`f`):
- A patch dataset is generated via `generate_dataset_set(...)`.
- A model is trained via `run_resunet(...)`.
- The generated predictions are used by `gen_refine_gt(...)` (from `run_single_process.py`) to create a refined ground truth.
- The next iteration uses that refined GT and previous model weights as initialization.

## Data expectations

The current code expects **NumPy arrays** on disk and relies on filename conventions.

### Image files
- `train_img_2020.npy`
- `train_img_2021.npy`
- `train_img_2022.npy`
- `train_img_2023.npy`

### Mask files
By default the code reads masks like:
- `train_mask_2020.npy` … `train_mask_2023.npy`

The refinement and generation modules currently include several hard-coded absolute paths under `/scratch/...` used in the original research environment. You should update those for your machine (see **Porting notes**).

## Installation

> Recommended: Python 3.10+ in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python patchify numba networkx pillow
pip install tensorflow keras
pip install segmentation-models
```

Depending on your TensorFlow/Keras version, you may need version pinning for compatibility.

## Running

### 1) Full iterative framework

```bash
python Framework/run_framework.py \
  --process_type f \
  --iterations 5 \
  --model_type resunet \
  --output_path /path/to/output/ \
  --from_scratch \
  --dilation \
  --k 4 \
  --R 150 \
  --th 0.5 \
  --r_th 0.1 \
  --epochs 10
```

### 2) Generate patches only

```bash
python Framework/run_framework.py \
  --process_type d \
  --output_path /path/to/output/
```

### 3) Train only

```bash
python Framework/run_framework.py \
  --process_type t \
  --model_type resunet \
  --data_dir /path/to/data/root \
  --img_folder_name <generated_image_folder> \
  --mask_folder_name <generated_mask_folder> \
  --output_path /path/to/output/ \
  --epochs 10 \
  --from_scratch
```

### CLI help

```bash
python Framework/run_framework.py -h
```

## Key arguments

### General
- `--process_type`: `d`, `t`, or `f`.
- `--iterations`: number of refinement cycles.
- `--output_path`: root folder for generated patches, checkpoints, logs, and refinement artifacts.

### Dataset generation
- `--image_path`: folder with yearly image `.npy` arrays.
- `--gt_path`: optional path to a GT `.npy`; if omitted, defaults are used.
- `--patch_size`: patch size (default `512`).
- `--dilation`: enable GT dilation pre-processing.
- `--k`: dilation kernel size.

### Training
- `--model_type`: `resunet`, `deeplabv3+`, or `resnet`.
- `--data_dir`: data root used by dataset loaders.
- `--batch_size`, `--learning_rate`, `--epochs`, `--optimizer`.
- `--from_scratch`: train new model; if omitted, `--pretrained_weights` is required.

### Refinement
- `--R`: radius for source-terminal pairing.
- `--th`: prediction threshold.
- `--r_th`: source-terminal connectivity threshold.

## Outputs

Under `--output_path`, the pipeline writes:
- Generated patch folders (`*_images_512`, `*_masks_512`).
- Model checkpoints in `models/`.
- Training CSV logs in `logs/`.
- Intermediate `.npy` graph/refinement artifacts.
- Iteration summary CSV files.

## Porting notes (important)

The codebase was developed in a fixed filesystem layout and still contains hard-coded paths (for example under `/scratch/gza5dr/...`). To run this project elsewhere, you will likely need to:

1. Replace hard-coded data paths in:
   - `Framework/generate_data.py`
   - `Framework/resunet_train.py`
   - `Framework/run_single_process.py`
   - `Framework/configure_p.py`
2. Verify filename conventions for image/mask arrays.
3. Ensure generated test/train split folders match what `DatasetHandler` expects.

## Reproducibility and caveats

- This repository appears research-oriented and may require environment-specific adaptation before first successful run.
- Some defaults in CLI arguments reference machine-specific paths and should be overridden.
- There is no pinned `requirements.txt` yet; dependency resolution may vary by platform.

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ijcai2025p1076,
  title     = {IGraSS: Learning to Identify Infrastructure Networks from Satellite Imagery by Iterative Graph-constrained Semantic Segmentation},
  author    = {Hoque, Oishee Bintey and Adiga, Abhijin and Adiga, Aniruddha and Chaudhary, Siddharth and Marathe, Madhav V. and Ravi, S.S. and Rajagopalan, Kirti and Wilson, Amanda and Swarup, Samarth},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)},
  pages     = {9683--9691},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/1076},
  url       = {https://doi.org/10.24963/ijcai.2025/1076}
}
```


