import os
import glob
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt

class Dataset(data.Dataset):
    def __init__(self, imageDir, imageList, transform = None): #imageDir can be either train, test or validation"""
        self.imageDir = imageDir
        self.imageList = imageList #with or without label?
        self.transform = transform
        # self.imagePaths = glob.glob(os.path.join(self.imageList))

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, item):
        image_path = os.path.join(self.imageDir, self.imageList[item].split()[0])
        Im = Image.open(image_path).convert("RGB")
        # print(Im)
        # Im = np.array(Im)
        # Im = Im/Im.max() #converts image to 0~1
        # print(np.min(Im))
        # Im = ToTensor()(Im)
        X = self.imageList[item].split()[1]
        X = float(X)
        # print("X: ", X, type(X))
        X = torch.tensor(X)
        # X = ToTensor()(np.array(X))
        Y = self.imageList[item].split()[2]
        Y = float(Y)
        Y = torch.tensor(Y)

        if self.transform is not None:
            Im = self.transform(Im)
        # print("Min: ", torch.min(Im))
        return Im, X, Y





