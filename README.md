# HSI-based CNN-MoE Framework for Face Anti-Spoofing and Identification

This repository implements the **CNN-MoE FAS** framework, a deep learning approach for Hyperspectral Imaging (HSI)-based Face Anti-Spoofing (FAS) and Identity Recognition.

As described in the associated research, HSI data provides useful material and spectral clues for distinguishing real skin from fake mediums (e.g., masks, paper). However, high dimensionality and redundancy in HSI data increase computational costs. This project solves this by combining a VGG-style CNN backbone with a **Sparsely-Gated Mixture-of-Experts (MoE)** module.

## Key Features

* **Multi-Task Learning:** Simultaneously performs **Face Anti-Spoofing (Live/Spoof)** and **Face Identification**.


* **Sparsely-Gated MoE:** Uses a router to activate only the Top-k experts per sample, reducing computation while increasing model expressiveness.


* **ArcFace Loss:** Utilizes ArcFace loss for robust identity verification alongside Cross-Entropy loss for spoof detection.


* **Spatial-Spectral Stacking:** Preprocesses HSI images by cropping them into 36 patches and stacking them as channels to capture material properties.



## Project Structure

```
.
├── cnn.py  # Baseline VGG-style CNN backbone [cite: 21]
├── main_cnn.py         # Training script for the baseline CNN
├── main_moe.py         # Training script for the proposed CNN-MoE model
├── model.py            # MoE Backbone, Router, ClassificationHead, and ArcFaceHead [cite: 30, 36]
├── preprocessing.py    # HSI cropping (6x6 grid) and dataset construction [cite: 20]
└── results/            # Saved model checkpoints and logs

```

## Methodology

### 1. Input & Preprocessing

The model does not use the raw HSI image directly. Instead, it employs a patch-based approach:

* The raw HSI image is cropped into **36 non-overlapping patches** (6x6 grid).


* These patches are stacked along the channel dimension, resulting in a tensor input of shape `(Batch, 36, H, W)`.


* This is handled in `preprocessing.py` via `hsi_crop` and `hsi_preprocessing`.

### 2. Architecture

* **Backbone:** A shared VGG-style CNN extracts a feature vector (size ).


* **Router (MoE):** A lightweight gating network calculates routing probabilities and selects the **Top-k** (default ) experts.


* **Heads:**
  * **Classification Head:** Detects 4 classes: Live (Real), Paper, iPhone, iPad.

  * **ArcFace Head:** Learns identity embeddings for subject verification.





## Dataset

The project utilizes a custom HSI dataset constructed for this research:

* **Subjects:** 54 individuals.
* **Images:** 4,374 images total.
* **Conditions:** Various lighting (Fluorescent, LED) and accessories (Mask, Hat, Sunglasses).


* **Attacks:** Paper, iPhone, and iPad spoofing attacks.



## Usage

### Prerequisites

* Python  3.10.14
* PyTorch 2.9.1
* OpenCV  4.12.0.88
* NumPy   2.2.6

### Training

To reproduce the results, use the provided `main` scripts. The default seed is set to `43` as per the paper.

**1. Train Proposed MoE Model**

```bash
python main_moe.py \
    --seed 43 \
    --num_experts 4 \
    --batch 64 \
    --cls_weight 0.8 \
    --arc_weight 0.8

```

**2. Train Baseline CNN Model**

```bash
python main_cnn.py \
    --seed 43 \
    --batch 64 \
    --cls_weight 0.8 \
    --arc_weight 0.8

```

### Arguments

* `--seed`: Random seed (default: 43).


* `--num_experts`: Number of experts in the MoE layer (default: 4).


* `--cls_weight`: Weight for classification loss ().


* `--arc_weight`: Weight for identity ArcFace loss ().



## Experimental Results

The proposed CNN-MoE framework significantly outperforms the baseline CNN, particularly in reducing the Attack Presentation Classification Error Rate (APCER).

| Model | APCER | NPCER | ACER | Cls Accuracy | ID Accuracy |
| --- | --- | --- | --- | --- | --- |
| **Baseline CNN** | 97.53% | 0.00% | 48.77% | 82.79% | 73.75% |
| **Ours (CNN-MoE)** | **12.96%** | **0.00%** | **6.48%** | **93.68%** | **75.93%** |

Data sourced from Table 1 of the associated paper.

## Acknowledgments

This research was supported by the Ministry of Science and ICT (MSIT), Korea, under the ITRC support program (IITP-2026-RS-2021-1211835) and the National Research Foundation of Korea (NRF) (RS-2026-22932973).
