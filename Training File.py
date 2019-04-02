import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np


TRAIN_PATH = './Dataset/train.txt'
TEST_PATH = './Dataset/test.txt'
BATCH_SIZE = 10
EPOCH = 100





class MyDataset(Dataset):
    def __init__(self,txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_lists = [i.split(" ")[0] for i in lines]
            # print(self.img_lists)
            self.label_lists = [i.split(" ")[1][:-1] for i in lines]


    def __getitem__(self, index):
        img = cv.imread(self.img_lists[index])
        img = cv.resize(img,(100,100),interpolation=cv.INTER_CUBIC)
        img = img.reshape([3,100,100])
        # img = img[np.newaxis, :, :, :]  # add dimention

        label = cv.imread(self.label_lists[index],0)
        label = cv.resize(label,(100,100),interpolation=cv.INTER_CUBIC)
        label = label.flatten()
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        return img,label


    def __len__(self):
        return len(self.img_lists)




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 10,kernel_size = 5,stride = 1,padding = 2)
        self.mp = nn.MaxPool2d(kernel_size = 4)
        self.fc = nn.Linear(25 * 25 * 10,10000)

    def forward(self, x):
        out = F.relu(self.mp(self.conv1(x)))
        out = out.view(BATCH_SIZE,-1)
        # print(out.shape)
        out = self.fc(out)
        return F.log_softmax(out)




class Main():
    def train(self,epoch):
        train_dataset = MyDataset(TRAIN_PATH)
        train_loader = DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = True)

        for step,(data,target) in enumerate(train_loader):
            data = data.type('torch.FloatTensor').cuda()
            target = target.cuda()
            self.optimizer.zero_grad()
            output = self.net(data)
            print(output.shape)
            print(target.shape)
            loss = nn.MSELoss(output, target)
            loss.backward()
            self.optimizer.step()
            print(epoch)


    def main(self):
        self.net = Net().cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(),lr = 0.001,momentum = 0.5)
        for epoch in range(EPOCH):
            self.train(epoch)

if __name__ == '__main__':
    t = Main()
    t.main()