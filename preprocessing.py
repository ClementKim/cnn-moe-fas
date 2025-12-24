import os
import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
import torch.utils.data as data

from random import shuffle
from itertools import product

def hsi_crop(img):
    img_raw = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = img_raw.shape

    cy, cx = int(h / 2), int(w / 2)
    sz_step = 300

    lt = (cx - 3 * sz_step, cy - 3 * sz_step)

    anch_x = [lt[0] + sz_step * i for i in range(6)]
    anch_y = [lt[1] + sz_step * i for i in range(6)]

    anchors = [(x, y) for (y, x) in product(anch_y, anch_x)]

    imgs = []
    for idx, (x, y) in enumerate(anchors):
        crop_img = img_raw[y : y + sz_step, x : x + sz_step]

        imgs.append(crop_img)

    return imgs

def hsi_preprocessing(image_path):
    imgs = hsi_crop(image_path)
    cube = np.stack(imgs, axis=0)

    tensor = torch.from_numpy(cube).float()
    tensor /= 255.0

    return tensor

class CustumDataset(data.Dataset):
    def __init__(self, x):
        super(CustumDataset,self).__init__()

        self.x = x
        self.label = [3 if "iphone" in item else 2 if "ipad" in item else 1 if "paper" in item else 0 for item in self.x]
        self.id = [int(item.split("/")[-3][-3:]) - 1 for item in self.x]

    def __getitem__(self, index):
        img_path = self.x[index]
        label = self.label[index]
        id = self.id[index]

        img_tensor = hsi_preprocessing(img_path)

        return img_tensor, label, id
    
    def __len__(self):
        return len(self.x)
    
def construction(seed):
    root = "./fas_project/dataset/hsi_raw"
    random.seed(seed)

    train = []
    val = []
    test = []

    for subject_num in os.listdir(root):
        live = [i for i in range(1, 67)]
        shuffle(live)

        iphone = [i for i in range(67, 72)]
        shuffle(iphone)

        ipad = [i for i in range(72, 77)]
        shuffle(ipad)

        paper = [i for i in range(77, 82)]
        shuffle(paper)

        target_live_dir = os.path.join(root, subject_num, "real")
        target_fake_dir = os.path.join(root, subject_num, "fake")

        train.extend([f"{os.path.join(target_live_dir, str(i).zfill(4) + '_hsi.jpg')}" for i in live[:int(len(live)*0.6)]])
        train.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_iphone.jpg')}" for i in iphone[:int(len(iphone)*0.6)]])
        train.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_ipad.jpg')}" for i in ipad[:int(len(ipad)*0.6)]])
        train.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_paper.jpg')}" for i in paper[:int(len(paper)*0.6)]])

        val.extend([f"{os.path.join(target_live_dir, str(i).zfill(4) + '_hsi.jpg')}" for i in live[int(len(live)*0.6):int(len(live)*0.8)]])
        val.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_iphone.jpg')}" for i in iphone[int(len(iphone)*0.6):int(len(iphone)*0.8)]])
        val.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_ipad.jpg')}" for i in ipad[int(len(ipad)*0.6):int(len(ipad)*0.8)]])
        val.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_paper.jpg')}" for i in paper[int(len(paper)*0.6):int(len(paper)*0.8)]])

        test.extend([f"{os.path.join(target_live_dir, str(i).zfill(4) + '_hsi.jpg')}" for i in live[int(len(live)*0.8):]])
        test.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_iphone.jpg')}" for i in iphone[int(len(iphone)*0.8):]])
        test.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_ipad.jpg')}" for i in ipad[int(len(ipad)*0.8):]])
        test.extend([f"{os.path.join(target_fake_dir, str(i).zfill(4) + '_hsi_paper.jpg')}" for i in paper[int(len(paper)*0.8):]])

    shuffle(train)
    shuffle(val)
    shuffle(test)

    return CustumDataset(train), CustumDataset(val), CustumDataset(test)

if __name__ == "__main__":
    construction(seed=42)