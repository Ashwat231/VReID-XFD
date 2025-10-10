# VSLA-CLIP - Baseline

This repository contains code for training and evaluation on the AG-VPReID dataset.

## Environment Setup

```bash
conda create -n vslaclip_new python=3.8
conda activate vslaclip_new
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs timm scikit-image tqdm ftfy regex
```

## Training

To train the model:

```bash
CUDA_VISIBLE_DEVICES=0 python train_reidadapter.py --config_file configs/adapter/vit_adapter.yml
```

## Evaluation

The repository supports three cross-view matching scenarios:

**Evaluate all cases at once:**
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_all_cases.py \
    --config_file configs/adapter/vit_adapter.yml \
    --model_path logs/ViT-B-16_5.pth
```

## Submission File

After completing evaluation, the `evaluation_rankings.csv` file can be found in:

```
logs/evaluation_rankings.csv
```

This file contains all three cases combined in the required order with tracklet rankings for submission.

## Hardware Requirements

This code has been tested on NVIDIA A100 GPUs.

## Acknowledgment

This baseline is based on the work of [VSLA-CLIP](https://github.com/FHR-L/VSLA-CLIP). We appreciate the authors for their excellent contribution.
