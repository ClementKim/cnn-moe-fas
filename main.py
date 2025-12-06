import torch
import argparse
import random
import numpy as np
import torchvision

from model import Model

if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 42)
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
    train_dataset = torchvision.datasets.MNIST(root = "./data", 
                                               train = True, 
                                               transform = torchvision.transforms.ToTensor(), 
                                               download = True)
    
    test_dataset = torchvision.datasets.MNIST(root = "./data", 
                                              train = False, 
                                              transform = torchvision.transforms.ToTensor(), 
                                              download = True)
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = 64,
                                               shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = 64,
                                              shuffle = False)
    
    model = Model(num_classes = 10, num_experts = 10).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    oprimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            oprimizer.zero_grad()
            loss.backward()
            oprimizer.step()

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
