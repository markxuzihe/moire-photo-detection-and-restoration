import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import torchvision.transforms as transforms
import cv2
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset

def default_loader(path1,path2,size):
    labels = []
    images = []
    for i in range(1,size+1):
        images.append(Image.open(path1+str(i)+".jpg"))
        labels.append(0)
        images.append(Image.open(path2+str(i)+".jpg"))
        labels.append(1)
    return labels,images

class MyDataset(Dataset):

    def __init__(self, path1, path2, train=True, size=400):
        super(MyDataset, self).__init__()
        if train:

            self.labels, self.images = default_loader(path1,path2,size)
        else:
            self.labels, self.images = default_loader(path1,path2,size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = np.asarray(image)
        image = np.transpose(image,(2,0,1))
        image = torch.from_numpy(image).div(255.0)
        label = self.labels[index]
        label = int(label)
        return image, label