import os
import random
import torch
import torch.utils.data as data

from random import shuffle

class ClassificationDataset(data.Dataset):
    def __init__(self, x):
        super(ClassificationDataset,self).__init__()

        self.x = x
        self.y = ["live" if "live" in item else "spoof" for item in self.x]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
class IdDataset(data.Dataset):
    def __init__(self, x):
        super(IdDataset,self).__init__()

        self.x = x
        self.y = [item.split("/")[-3] for item in self.x]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
def custum_dataset(opt, seed):
    root = "./data/CelebA_Spoof/Data"
    live = []
    spoof = []

    random.seed(seed)
    
    for opt in ["train", "test"]:
        for num in os.listdir(os.path.join(root, opt)):
            for img in os.listdir(os.path.join(root, opt, num, "live")):
                if img.endswith(".jpg") or img.endswith(".png"):
                    live.append(os.path.join(root, opt, num, "live", img))

            for img in os.listdir(os.path.join(root, opt, num, "spoof")):
                if img.endswith(".jpg") or img.endswith(".png"):
                    spoof.append(os.path.join(root, opt, num, "spoof", img))

    shuffle(live)
    shuffle(spoof)
    
    train = []
    val = []
    test = []

    train.extend(live[:int(0.6*len(live))])
    train.extend(spoof[:int(0.6*len(spoof))])

    val.extend(live[int(0.6*len(live)):int(0.8*len(live))])
    val.extend(spoof[int(0.6*len(spoof)):int(0.8*len(spoof))])

    test.extend(live[int(0.8*len(live)):])
    test.extend(spoof[int(0.8*len(spoof)):])

    shuffle(train)
    shuffle(val)
    shuffle(test)

    if opt == "classification":
        train_dataset = ClassificationDataset(train)
        val_dataset = ClassificationDataset(val)
        test_dataset = ClassificationDataset(test)

    else:
        train_dataset = IdDataset(train)
        val_dataset = IdDataset(val)
        test_dataset = IdDataset(test)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    custum_dataset(seed=42)