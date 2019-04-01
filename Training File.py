import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np


TRAIN_PATH = './Dataset/train.txt'
TEST_PATH = './Dataset/test.txt'



class MyDataset(Dataset):
    def __init__(self,txt_path,image = True):
        self.image = image
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_lists = [i.split(" ")[0] for i in lines]


    def __getitem__(self, index):
        if self.image == False:
            img = cv.imread(self.img_lists[index], 0)
            img = img[np.newaxis, :, :]  # add dimention
        else:
            img = cv.imread(self.img_lists[index])
            img = img[np.newaxis, :, :, :]  # add dimention

    def __len__(self):
        return len(self.img_lists)

if __name__ == '__main__':
    pass