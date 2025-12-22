import torch
import argparse
import random
import numpy as np
import torchvision

from preprocessing import custum_dataset
from model import Backbone, ClassificationHead, IdentificationHead, VerificationHead

if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--num_experts", type = int, default = 4)
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
    train_dataset, val_dataset, test_dataset = custum_dataset(opt = "other", seed = args.seed)
    # train_dataset, val_dataset, test_dataset = custum_dataset(opt = "classification", seed = args.seed)

    exit(1)
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = 64,
                                               shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = 64,
                                              shuffle = False)
    
    model = Backbone(num_classes = 2, num_experts = args.num_experts).to(device)
    identification = IdentificationHead(input_size = 32, num_classes = 2).to(device)
    verification = VerificationHead(input_size = 32, num_classes = 2).to(device)
    classification = ClassificationHead(input_size = 32, num_classes = 2).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")


    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")
