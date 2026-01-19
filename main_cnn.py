# Import necessary libraries
import torch
import argparse
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from preprocessing import construction
from cnn import Backbone
from model import ClassificationHead, ArcFaceHead
from sklearn.metrics import confusion_matrix

def seed_worker(worker_id):
    """
    Sets the random seed for a DataLoader worker.

    This function is used to ensure that the data loading process is reproducible
    across different runs.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Main execution block
if __name__ == "__main__":
    # --- Argument Parsing ---
    # Sets up the command-line arguments that can be used to configure the script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 43)
    parser.add_argument("--batch", type = int, default = 64)
    parser.add_argument("--cls_weight", type = float, default = 0.8)
    parser.add_argument("--arc_weight", type = float, default = 0.8)
    args = parser.parse_args()

    # --- Reproducibility ---
    # Sets the random seed for all relevant libraries to ensure that the results
    # can be reproduced.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Device Configuration ---
    # Selects the appropriate device (GPU, MPS, or CPU) for training and evaluation.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Creates a torch.Generator for reproducible data loading and initializes the datasets
    # and DataLoaders for training, validation, and testing.
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset, val_dataset, test_dataset = construction(seed = args.seed)
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = 64,
                                               shuffle = True,
                                               num_workers = 4,
                                               worker_init_fn = seed_worker,
                                               generator = g)
    
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                             batch_size = args.batch,
                                             shuffle = False,
                                             num_workers = 4,
                                             worker_init_fn = seed_worker,
                                             generator = g)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = args.batch,
                                              shuffle = False,
                                              num_workers = 4,
                                              worker_init_fn = seed_worker,
                                              generator = g)
    
    # --- Model Initialization ---
    # Initializes the backbone network, classification head, and ArcFace head, and moves
    # them to the selected device.
    model = Backbone(input_channels = 36, embed_dim = 128).to(device)
    classification = ClassificationHead(input_dim = 128, num_classes = 4).to(device)
    archead = ArcFaceHead(in_features = 128, out_features = 54, s = 30.0, m = 0.50).to(device)

    # --- Loss and Optimizer ---
    # Defines the loss functions and the optimizer. A class weight is used for the
    # classification loss to handle class imbalance.
    class_weight = torch.tensor([1.0, 13.0, 13.0, 13.0]).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(weight = class_weight).to(device)
    criterion_arc = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': classification.parameters()},
        {'params': archead.parameters()}
    ], lr = 0.001, weight_decay = 1e-4)

    # --- Training Configuration ---
    # Sets the weights for the different loss components and the number of training epochs.
    lambda_cls = args.cls_weight
    lambda_arc = args.arc_weight
    epochs = 20

    # --- Training Loop ---
    # The main loop for training the model. It iterates over the specified number of epochs.
    for ep in range(epochs):
        model.train()
        classification.train()
        archead.train()

        running_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{ep+1}/{epochs}] Train")
        for images, labels, ids in pbar:
            images, labels, ids = images.to(device), labels.to(device), ids.to(device)

            # Forward pass: Get features and outputs from the model.
            features = model(images)
            out_cls = classification(features)
            out_arc = archead(features, ids)

            # Calculate loss: Combine the classification and ArcFace losses.
            loss_cls = criterion_cls(out_cls, labels)
            loss_arc = criterion_arc(out_arc, ids)
            loss = (lambda_cls * loss_cls) + (lambda_arc * loss_arc)

            # Backward pass and optimization: Update the model's weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = running_loss / train_steps

        # --- Validation Loop ---
        # Evaluates the model on the validation set after each epoch.
        model.eval()
        classification.eval()
        archead.eval()
        
        val_loss = 0.0
        val_correct_cls = 0
        val_correct_ids = 0
        val_total_cls = 0
        val_total_ids = 0
        val_steps = 0

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels, ids in tqdm(val_loader, desc=f"Epoch [{ep+1}/{epochs}] Val"):
                images, labels, ids = images.to(device), labels.to(device), ids.to(device)

                # Forward pass
                features = model(images)
                norm_features = F.normalize(features)
                out_cls = classification(norm_features)
                out_arc = archead(norm_features, ids)

                # Calculate loss
                loss_cls = criterion_cls(out_cls, labels)
                loss_arc = criterion_arc(out_arc, ids)
                loss = (lambda_cls * loss_cls) + (lambda_arc * loss_arc)
                
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate validation metrics
                _, predicted_cls = torch.max(out_cls.data, 1)
                _, predicted_arc = torch.max(out_arc.data, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted_cls.cpu().numpy())

                val_total_cls += labels.size(0)
                val_total_ids += ids.size(0)
                val_correct_cls += (predicted_cls == labels).sum().item()
                val_correct_ids += (predicted_arc == ids).sum().item()

        # Calculate and print validation results, including APCER, NPCER, and ACER.
        cm = confusion_matrix(all_labels, all_preds, labels = [0, 1, 2, 3])

        TP = cm[0,0]
        FN = np.sum(cm[0, 1:])
        FP = np.sum(cm[1:, 0])
        TN = np.sum(cm[1:, 1:])

        NPCER = FN / (TP + FN) if (TP + FN) > 0 else 0
        apcer = FP / (FP + TN) if (FP + TN) > 0 else 0
        ACER = (apcer + NPCER) / 2

        avg_val_loss = val_loss / val_steps
        val_acc_cls = 100 * val_correct_cls / val_total_cls
        val_acc_ids = 100 * val_correct_ids / val_total_ids
        
        print(f"\nEpoch [{ep+1}/{epochs}] Results:")
        print("-" * 50)
        print(f" APCER:      {apcer*100:.2f}%")
        print(f" NPCER:      {NPCER*100:.2f}%")
        print(f" ACER:       {ACER*100:.2f}%")
        print(f" Train Loss: {avg_train_loss:.4f}")
        print(f" Val Loss:  {avg_val_loss:.4f} | Val Acc (Cls): {val_acc_cls:.2f}% | Val Acc (Arc): {val_acc_ids:.2f}%")
        print(f"FP: {FP}, FN: {FN}, TP: {TP}, TN: {TN}")
        print("-" * 50)

    # --- Testing Loop ---
    # After training is complete, this loop evaluates the final model on the test set.
    model.eval()
    classification.eval()
    archead.eval()
    with torch.no_grad():
        correct_cls, total_cls = 0, 0
        correct_ids, total_ids = 0, 0

        all_labels = []
        all_preds = []

        for images, labels, ids in tqdm(test_loader, desc = "Testing model"):
            images, labels, ids = images.to(device), labels.to(device), ids.to(device)

            # Forward pass
            features = model(images)
            norm_features = F.normalize(features)
            out_cls = classification(norm_features)
            out_arc = archead(norm_features, ids)

            # Get predictions
            _, predicted_cls = torch.max(out_cls.data, 1)
            _, predicted_arc = torch.max(out_arc.data, 1)
            
            # Calculate test metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_cls.cpu().numpy())
            
            total_cls += labels.size(0)
            total_ids += ids.size(0)
            correct_cls += (predicted_cls == labels).sum().item()
            correct_ids += (predicted_arc == ids).sum().item()

        # Calculate and print test results, including APCER, NPCER, and ACER.
        cm = confusion_matrix(all_labels, all_preds, labels = [0, 1, 2, 3])

        TP = cm[0,0]
        FN = np.sum(cm[0, 1:])
        FP = np.sum(cm[1:, 0])
        TN = np.sum(cm[1:, 1:])

        NPCER = FN / (TP + FN) if (TP + FN) > 0 else 0
        apcer = FP / (FP + TN) if (FP + TN) > 0 else 0
        ACER = (apcer + NPCER) / 2

        print("\nTest Results:")
        print("-" * 50)
        print(f" APCER: {apcer*100:.2f}%")
        print(f" NPCER: {NPCER*100:.2f}%")
        print(f" ACER: {ACER*100:.2f}%")
        print(f" Accuracy (classification): {100 * correct_cls / total_cls:.2f}%")
        print(f" Accuracy (arcface): {100 * correct_ids / total_ids:.2f}%")
        print(f"FP: {FP}, FN: {FN}, TP: {TP}, TN: {TN}")
        print("-" * 50)