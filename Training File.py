import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np


TRAIN_PATH = './Dataset/train.txt'
# TEST_PATH = './Dataset/test.txt'
BATCH_SIZE = 2
EPOCH = 10
EDGE_THRESHOLD = 128.0




class MyDataset(Dataset):
    def __init__(self,txt_path):
        with open(txt_path, 'r') as f:
            self.img_lists = []
            self.label_lists = []
            while True:
                line = f.readline()
                if not line:
                    break

                self.img_lists.append(line.split(" ")[0])
                self.label_lists.append((line.split(" ")[1][:-1]))
            # lines = f.readlines()
            # self.img_lists = [i.split(" ")[0] for i in lines]
            # # print(self.img_lists)
            # self.label_lists = [i.split(" ")[1][:-1] for i in lines]
        print(self.label_lists)

    def __getitem__(self, index):
        img = cv.imread(self.img_lists[index])
        img = cv.resize(img,(224,224),interpolation=cv.INTER_CUBIC)
        img = img.reshape([3,224,224])
        # img = img[np.newaxis, :, :, :]  # add dimention

        label = cv.imread(self.label_lists[index],0)
        # print(index,label)
        label = cv.resize(label,(224,224),interpolation=cv.INTER_CUBIC)
        label = torch.from_numpy(label)
        label = torch.unsqueeze(label,0)
        label = label.type(torch.FloatTensor)
        return img,label


    def __len__(self):
        print(len(self.img_lists))
        return len(self.img_lists)




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 3,stride = 1,padding = 1)
        self.conv1_2 = nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,stride = 1,padding = 1)
        self.conv1 = nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 1,stride = 1)
        self.conv2_1 = nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 3,stride = 1,padding = 1)
        self.conv2_2 = nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size = 3,stride = 1,padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 128,out_channels = 1,kernel_size = 1,stride = 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3_1 = nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 3,stride = 1,padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3,stride = 1,padding = 1)
        self.conv3_3 = nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3,stride = 1,padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 256,out_channels = 1,kernel_size = 1,stride = 1)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv4_1 = nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 3,stride = 1,padding = 1)
        self.conv4_2 = nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1)
        self.conv4_3 = nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 512,out_channels = 1,kernel_size = 1,stride = 1)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv5_1 = nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1)
        self.conv5_2 = nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1)
        self.conv5_3 = nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 512,out_channels = 1,kernel_size = 1,stride = 1)
        self.s = nn.Conv2d(in_channels = 5,out_channels = 1,kernel_size = 1,stride = 1)
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.mp = nn.MaxPool2d(kernel_size = 2,stride = 2)





    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        s1 = F.relu(self.conv1(x))
        x = self.mp(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        s2 = self.up2(F.relu(self.conv2(x)))
        x = self.mp(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        s3 = self.up3(F.relu(self.conv3(x)))
        x = self.mp(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        s4 = self.up4(F.relu(self.conv4(x)))
        x = self.mp(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        s5 = self.up5(F.relu(self.conv5(x)))
        s = self.s(torch.cat([s1,s2,s3,s4,s5],1))
        # print(s.shape)
        return s




class Main():
    def train(self,epoch):
        train_dataset = MyDataset(TRAIN_PATH)
        train_loader = DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = True)

        for step,(data,target) in enumerate(train_loader):
            data = data.type('torch.FloatTensor').cuda()
            target = target.cuda()
            print(target.shape)
            self.optimizer.zero_grad()
            output = self.net(data)
            batch_loss = sum([self.weighted_cross_entropy_loss(output, target) for preds in output])
            eqv_iter_loss = batch_loss /10

            # Generate the gradient and accumulate (using equivalent average loss).
            eqv_iter_loss.backward(torch.ones_like(eqv_iter_loss))
            # batch_loss.backward()
            # print(output.shape)
            # print(target.shape)
            # loss_fn = nn.MSELoss(reduce=True, size_average=True)
            # loss = loss_fn(target,output)
            # loss.backward()
            # print(output)
            self.optimizer.step()
            print(epoch,batch_loss)


    def weighted_cross_entropy_loss(self,preds, edges):
        mask = (edges > EDGE_THRESHOLD).float()
        b,c,h,w = mask.shape
        num_pos = torch.sum(mask,dim = [1,2,3]).float()
        # print(num_pos)
        # print(num_pos.shape)
        num_neg = c * h * w - num_pos
        # print(num_neg)
        weight = torch.zeros_like(mask)
        # weight.cuda()
        # print(weight)
        w_neg = num_neg / (num_pos + num_neg)
        w_pos = num_pos / (num_pos + num_neg)
        for i in range(b):
            for j in range(c):
                for m in range(h):
                    for n in range(w):
                        # print(weight[i][j][m][n])
                        if edges[i][j][m][n] > EDGE_THRESHOLD:

                            weight[i][j][m][n] = w_neg[i]

                        else:
                            weight[i][j][m][n] = w_pos[i]
        # weight[edges > EDGE_THRESHOLD] = num_neg / (num_pos + num_neg)
        # weight[edges <= EDGE_THRESHOLD] = num_pos / (num_pos + num_neg)

        losses = F.binary_cross_entropy_with_logits(preds.float(),edges.float(),weight = weight,reduction = 'none')
        loss = losses / b
        return loss


    def main(self):
        self.net = Net().cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(),lr = 0.000001,momentum = 0.5)
        for epoch in range(EPOCH):
            self.train(epoch)

if __name__ == '__main__':
    t = Main()
    t.main()