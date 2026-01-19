# Import necessary libraries
import os
import cv2
import pickle
import torch
import random
import numpy as np
import PIL.Image as Image
import torch.utils.data as data

from random import shuffle
from itertools import product
from torchvision import transforms

def hsi_crop(img):
    """
    Crops a large Hyperspectral Imaging (HSI) image into a grid of smaller images.

    This function takes a large HSI image and divides it into a 6x6 grid of
    smaller, overlapping images.

    Args:
        img (str): The file path to the input HSI image.

    Returns:
        list: A list of the cropped image sections as NumPy arrays.
    """
    # Read the image in grayscale.
    img_raw = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = img_raw.shape

    # Define the center and step size for cropping.
    cy, cx = int(h / 2), int(w / 2)
    sz_step = 300

    # Define the starting point (top-left corner) of the cropping grid.
    lt = (cx - 3 * sz_step, cy - 3 * sz_step)

    # Generate the anchor points for the cropping grid.
    anch_x = [lt[0] + sz_step * i for i in range(6)]
    anch_y = [lt[1] + sz_step * i for i in range(6)]

    anchors = [(x, y) for (y, x) in product(anch_y, anch_x)]

    # Crop the image at each anchor point.
    imgs = []
    for idx, (x, y) in enumerate(anchors):
        crop_img = img_raw[y : y + sz_step, x : x + sz_step]
        imgs.append(crop_img)

    return imgs

def hsi_preprocessing(image_path):
    """
    Preprocesses an HSI image by cropping it and converting it to a tensor.

    Args:
        image_path (str): The file path to the input HSI image.

    Returns:
        torch.Tensor: A tensor representing the preprocessed image.
    """
    # Crop the image and stack the cropped sections to form a data cube.
    imgs = hsi_crop(image_path)
    cube = np.stack(imgs, axis=0)

    # Convert the cube to a PyTorch tensor and normalize its values to the [0, 1] range.
    tensor = torch.from_numpy(cube).float()
    tensor /= 255.0

    return tensor

class CustumDataset(data.Dataset):
    """
    A custom dataset class for loading and preprocessing HSI images.
    """
    def __init__(self, x, transform = None):
        """
        Initializes the dataset.

        Args:
            x (list): A list of file paths to the HSI images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(CustumDataset,self).__init__()

        self.x = x
        # Assign labels based on keywords in the image file names.
        self.label = [3 if "iphone" in item else 2 if "ipad" in item else 1 if "paper" in item else 0 for item in self.x]
        # Extract subject IDs from the file paths.
        self.id = [int(item.split("/")[-3][-3:]) - 1 for item in self.x]
        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image tensor, its label, and the subject ID.
        """
        img_path = self.x[index]
        label = self.label[index]
        id = self.id[index]

        # Preprocess the image to get the HSI tensor.
        img_tensor = hsi_preprocessing(img_path)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label, id
    
    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.x)
    
def train_val_test(seed):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        seed (int): A random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training, validation, and test sets.
    """
    # Set the root directory for the dataset.
    root = "./fas_project/dataset/hsi_raw"
    random.seed(seed)
    np.random.seed(seed)

    # Initialize lists for the different data splits.
    train = []
    val = []
    test = []

    # Define the ranges for different types of images.
    live = [i for i in range(1, 67)]
    iphone = [i for i in range(67, 72)]
    ipad = [i for i in range(72, 77)]
    paper = [i for i in range(77, 82)]
    
    # Iterate over each subject in the dataset.
    for subject_num in sorted(os.listdir(root)):
        all = []

        # Define the directories for live and fake images for the current subject.
        target_live_dir = os.path.join(root, subject_num, "real")
        target_fake_dir = os.path.join(root, subject_num, "fake")

        # Gather all image paths for the current subject.
        all.extend([f"{os.path.join(target_live_dir, str(i).zfill(4) + '_hsi.jpg')}" for i in live])
        all.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_iphone.jpg')}" for i in iphone])
        all.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_ipad.jpg')}" for i in ipad])
        all.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_paper.jpg')}" for i in paper])

        shuffle(all)

        # Split the data into training, validation, and test sets.
        train.extend(all[:int(len(all)*0.6)])
        val.extend(all[int(len(all)*0.6):int(len(all)*0.8)])
        test.extend(all[int(len(all)*0.8):])

    # Shuffle the datasets to ensure randomness.
    shuffle(train)
    shuffle(val)
    shuffle(test)

    # Save the data splits to pickle files for later use.
    with open(f"dataset/train_{seed}.pkl", "wb") as f:
        pickle.dump(train, f)

    with open(f"dataset/val_{seed}.pkl", "wb") as f:
        pickle.dump(val, f)

    with open(f"dataset/test_{seed}.pkl", "wb") as f:
        pickle.dump(test, f)

    return train, val, test
    
def construction(seed):
    """
    Constructs the training, validation, and test datasets.

    This function first tries to load pre-split datasets from pickle files.
    If they don't exist, it calls `train_val_test` to create them.

    Args:
        seed (int): A random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        # Try to load the pre-split datasets.
        with open(f"dataset/train_{seed}.pkl", "rb") as f:
            train = pickle.load(f)

        with open(f"dataset/val_{seed}.pkl", "rb") as f:
            val = pickle.load(f)

        with open(f"dataset/test_{seed}.pkl", "rb") as f:
            test = pickle.load(f)

    except:
        # If the pickle files don't exist, create the splits.
        train, val, test = train_val_test(seed)
    
    # Return the custom datasets.
    return CustumDataset(train, transform = None), CustumDataset(val, transform = None), CustumDataset(test, transform = None)

if __name__ == "__main__":
    # This block is for testing the dataset and DataLoader.
    train, val, test = construction(seed = 42)
    print(len(train), len(val), len(test))
    
    train_loader = torch.utils.data.DataLoader(dataset = train,
                                               batch_size = 64,
                                               shuffle = True)
    
    # Print the shape of a single batch to verify the data loading process.
    for img, label, id in train_loader:
        print(img.shape)
        print(label.shape)
        print(id.shape)
        break
