# CNN vs. Mixture of Experts for Face Anti-Spoofing

This project implements and compares a standard Convolutional Neural Network (CNN) with a Mixture of Experts (MoE) model for a Face Anti-Spoofing (FAS) task. The models are trained to distinguish between live subjects and various spoofing attacks (e.g., photos on iPhones, iPads, or paper). Additionally, the models learn to identify the subject's ID using ArcFace loss.

## Project Structure

```
.
├── cnn.py              # Baseline CNN model architecture
├── main_cnn.py         # Main script to train and evaluate the CNN model
├── main_moe.py         # Main script to train and evaluate the MoE model
├── model.py            # MoE model architecture, ClassificationHead, and ArcFaceHead
├── preprocessing.py    # Data loading and preprocessing scripts
├── README.md           # This file
├── fas_project/        # Project data
└── results/            # Directory to save model results
    ├── cnn_result
    └── moe_result
```

## Key Files

*   `preprocessing.py`: This script handles loading and preprocessing the Hyperspectral Imaging (HSI) data from the `./fas_project/dataset/hsi_raw` directory. It creates `Dataset` objects for training, validation, and testing.
*   `cnn.py`: Defines the architecture of the baseline CNN model.
*   `model.py`: Contains the implementation of the Mixture of Experts (MoE) model, as well as the `ClassificationHead` and `ArcFaceHead` used by both the CNN and MoE models.
*   `main_cnn.py`: The main script for training and evaluating the baseline CNN model. It uses a combination of standard cross-entropy loss for classification and ArcFace loss for subject identification.
*   `main_moe.py`: The main script for training and evaluating the MoE model. It follows a similar training and evaluation procedure as `main_cnn.py`.

## Usage

To train and evaluate the models, run the respective main scripts:

### CNN Model

```bash
python main_cnn.py --seed 42 --batch 64 --cls_weight 0.8 --arc_weight 0.8
```

### MoE Model

```bash
python main_moe.py --seed 42 --num_experts 4 --batch 64 --cls_weight 0.8 --arc_weight 0.8
```

### Arguments

*   `--seed`: Random seed for reproducibility.
*   `--batch`: Batch size for training and evaluation.
*   `--cls_weight`: Weight for the classification loss.
*   `--arc_weight`: Weight for the ArcFace loss.
*   `--num_experts` (for MoE model only): The number of experts in the Mixture of Experts model.

## Evaluation

The models are evaluated using the following metrics:

*   **APCER** (Attack Presentation Classification Error Rate)
*   **NPCER** (Normal Presentation Classification Error Rate)
*   **ACER** (Average Classification Error Rate)

The scripts will print these metrics to the console after training and evaluation are complete.