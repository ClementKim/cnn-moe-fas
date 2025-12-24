import torch
import argparse
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from preprocessing import construction
from model import Backbone, ClassificationHead, ArcFaceHead

if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--num_experts", type = int, default = 4)
    parser.add_argument("--batch", type = int, default = 64)
    parser.add_argument("--cls_weight", type = float, default = 0.5)
    parser.add_argument("--arc_weight", type = float, default = 0.7)
    args = parser.parse_args()

    # for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    ## Testing model
    train_dataset, val_dataset, test_dataset = construction(seed = args.seed)
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = 64,
                                               shuffle = True,
                                               num_workers = 4)
    
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                             batch_size = args.batch,
                                             shuffle = False,
                                             num_workers = 4)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = args.batch,
                                              shuffle = False,
                                              num_workers = 4)
    
    model = Backbone(input_channels = 36, embed_dim = 128, num_experts = args.num_experts).to(device)
    classification = ClassificationHead(input_dim = 128, num_classes = 4).to(device)
    archead = ArcFaceHead(in_features = 128, out_features = 54, s = 30.0, m = 0.50).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': classification.parameters()},
        {'params': archead.parameters()}
    ], lr = 0.001, weight_decay = 1e-4)

    # lambda_cls = 0.5
    # lambda_arc = 0.7
    lambda_cls = args.cls_weight
    lambda_arc = args.arc_weight
    epochs = 10

    for ep in range(epochs):
        model.train()
        classification.train()
        archead.train()

        running_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{ep+1}/{epochs}] Train")
        for images, labels, ids in pbar:
            images = images.to(device)
            labels = labels.to(device)
            ids = ids.to(device)

            features = model(images)
            out_cls = classification(features)
            out_arc = archead(features, ids)

            loss_cls = criterion(out_cls, labels)
            loss_arc = criterion(out_arc, ids)

            loss = (lambda_cls * loss_cls) + (lambda_arc * loss_arc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = running_loss / train_steps

        model.eval()
        classification.eval()
        archead.eval()
        
        val_loss = 0.0
        val_correct_cls = 0
        val_total = 0
        val_steps = 0
        
        with torch.no_grad():
            for images, labels, ids in tqdm(val_loader, desc=f"Epoch [{ep+1}/{epochs}] Val"):
                images = images.to(device)
                labels = labels.to(device)
                ids = ids.to(device)

                features = model(images)
                norm_features = F.normalize(features)

                out_cls = classification(norm_features)
                out_arc = archead(norm_features, ids)

                loss_cls = criterion(out_cls, labels)
                loss_arc = criterion(out_arc, ids)
                loss = (lambda_cls * loss_cls) + (lambda_arc * loss_arc)
                
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate validation accuracy for classification
                _, predicted = torch.max(out_cls.data, 1)
                val_total += labels.size(0)
                val_correct_cls += (predicted == labels).sum().item()

        avg_val_loss = val_loss / val_steps
        val_acc = 100 * val_correct_cls / val_total
        
        print(f"Epoch [{ep+1}/{epochs}] Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc (Cls): {val_acc:.2f}%")
        print("-" * 50)


    model.eval()
    classification.eval()
    archead.eval()
    with torch.no_grad():
        correct_label = 0
        correct_ids = 0
        total_label = 0
        total_ids = 0
        for images, labels, ids in tqdm(test_loader, desc = "Testing model"):
            images = images.to(device)
            labels = labels.to(device)
            ids = ids.to(device)

            features = model(images)
            norm_features = F.normalize(features)
            out_cls = classification(norm_features)
            out_arc = archead(norm_features, ids)

            _, predicted_cls = torch.max(out_cls.data, 1)
            _, predicted_arc = torch.max(out_arc.data, 1)
            
            total_label += labels.size(0)
            correct_label += (predicted_cls == labels).sum().item()
            
            total_ids += ids.size(0)
            correct_ids += (predicted_arc == ids).sum().item()

        print(f"Test Accuracy (classification): {100 * correct_label / total_label:.2f}%")
        print(f"Test Accuracy (arcface): {100 * correct_ids / total_ids:.2f}%")