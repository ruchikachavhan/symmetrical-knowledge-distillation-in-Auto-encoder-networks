import torch
import torch.nn as nn
import numpy as np
import os
import math
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import random
from torch.utils.data.sampler import SubsetRandomSampler
import time
from torch.optim.lr_scheduler import StepLR
import imageio

use_cuda = 1
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {}

#Helper function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.eval()
    return model

def KL(Q, P):
    F.kl_div(Q.log(), P, None, None, 'sum')

#--------------------------------------------------------------
#---------------------Loading Data-----------------------------
#--------------------------------------------------------------
def train_val_data(batch_size, shuffle, num_workers, valid_size):
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose([transforms.ToTensor(), normalize,])
    dataset = torchvision.datasets.CIFAR10(root = './cifardata', download = 1, transform = transform) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.shuffle(indices)
    print("Length of dataset", num_train)
    valid_idx, train_idx = indices[split:], indices[:split]
    print("training", len(train_idx))
    print("validation", len(valid_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
                                dataset, batch_size=batch_size, sampler=train_sampler,
                                num_workers=num_workers,
                                )

    valid_loader = torch.utils.data.DataLoader(
                                dataset, batch_size=batch_size, sampler=valid_sampler,
                                num_workers=num_workers,
                                )
    return train_loader, valid_loader


#------------------------------------------
#-----------Defining ResNet Block----------
#------------------------------------------
class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()
    conv_1 = [  nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.BatchNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.BatchNorm2d(in_features)  ]
    self.conv_1 = nn.Sequential(*conv_1)
  def forward(self, x):
    return x + self.conv_1(x)

class Network(nn.Module):
    def __init__(self, ch, in_features):
        super().__init__()
        self.in_features = in_features
        self.ch = ch
        self.conv1 = nn.Conv2d(ch, in_features, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_features, in_features*2 , 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(infeatures*2, infeatures*4, 3)
        self.conv4 = nn.Conv2d(infeatures*4, infeatures*8, 3)
        self.norm1 = nn.BatchNorm2d(in_features)
        self.norm2 = nn.BatchNorm2d(in_features*2)
        self.norm3 = nn.BatchNorm2d(in_features*4)
        self.norm4 = nn.BatchNorm2d(in_features*8)
        self.resnet1 = ResidualBlock(in_features)
        self.resnet2 = ResidualBlock(in_features*2)
        self.resnet3 = ResidualBlock(in_features*4)
        self.resnet4 = ResidualBlock(in_features*8)
        bottle1 =   [nn.Conv2d(in_features, in_features*2 , 4, stride=1, padding=0), 
                    nn.ReLU(inplace=True), 
                    nn.BatchNorm2d(in_features*2),
                    nn.Conv2d(in_features*2, in_features*4 , 4, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_features*4),
                    nn.Conv2d(in_features*4, in_features*4 , 4, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_features*4),
                    nn.Conv2d(in_features*4, in_features*8 , 4, stride=1, padding=0), 
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_features*8)]
        self.bottle1 = nn.Sequential(*bottle1)
        bottle2 = [nn.Conv2d(in_features*2, in_features*4 , 3, stride=1, padding=0), 
                    nn.ReLU(inplace=True), 
                    nn.BatchNorm2d(in_features*4),
                    nn.Conv2d(in_features*4, in_features*8 , 3, stride=1, padding=0), 
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_features*8)]
        self.bottle2 = nn.Sequential(*bottle2)
        bottle3 = [ nn.Conv2d(in_features*4, in_features*8 , 3, stride=1, padding=0), 
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_features*8)]
        self.bottle3 = nn.Sequential(*bottle3)
        self.fc_teach = nn.Linear(512*4*4, 10)
        self.fc1 = nn.Linear(512*4*4, 10)
        self.fc2 = nn.Linear(512*4*4, 10)
        self.fc3 = nn.Linear(512*4*4, 10)
    def forward(self, images):
        out = self.norm1(F.relu(self.conv1(images)))
        res1 = self.resnet1.forward(out)
        
        out = self.norm2(F.relu(self.conv2(res1)))
        res2 = self.resnet2.forward(out)
        
        out = self.norm3(F.relu(self.conv3(res2)))
        res3 = self.resnet3.forward(out)
        
        out = self.norm4(F.relu(self.conv4(res3)))
        res4 = self.resnet4.forward(out)
        
        bottle1 = self.bottle1(res1)
        bottle2 = self.bottle2(res2)
        bottle3 = self.bottle3(res3)

        bottle1 , bottle2, bottle3 = bottle1.view(bottle1.shape[0], -1), bottle2.view(bottle2.shape[0], -1), bottle3.view(bottle3.shape[0], -1)
        res4 = res4.view(res4.shape[0], -1)

        fc_teach = self.fc_teach(res4)
        fc1 = self.fc1(bottle1)
        fc2 = self.fc2(bottle2)
        fc3 = self.fc3(bottle3)

        return bottle1, bottle2, bottle3, res4, fc_teach, fc1, fc2, fc3
    def calculate_loss(self, images, label):
        bottle1, bottle2, bottle3, res4, fc_teach, fc1, fc2, fc3 = self.forward(images)
        criterion = torch.nn.CrossEntropyLoss()
        L2FeatLoss = F.mse_loss(bottle1, res4) + F.mse_loss(bottle2, res4) + F.mse_loss(bottle3, res4)
        KL_loss = F.kl_div(F.softmax(fc1), F.softmax(fc_teach)) + F.kl_div(F.softmax(fc2), F.softmax(fc_teach)) + F.kl_div(F.softmax(fc3), F.softmax(fc_teach))
        CELoss =  criterion(fc1, label) + criterion(fc2, label) + criterion(fc3, label)
        teachCE =criterion(fc_teach, label)
        alpha = 0.2
        gamma = 0.5
        selfStudentLoss = (1-alpha)*CELoss + alpha*KL_loss + gamma*L2FeatLoss
        return teachCE + selfStudentLoss
        
#---------------------------------------------------------------------
#------------Initializing hyperparameters and network-----------------
#---------------------------------------------------------------------
lr = 0.0007
batch_size = 100
shuffle = True 
num_workers = 2
valid_size = 0.99
num_channels = 3 
infeatures = 64
train_loader, valid_loader = train_val_data(batch_size, shuffle, num_workers, valid_size)
net = Network(num_channels, infeatures).to(device)
optim = torch.optim.Adamax(list(net.parameters()), lr)
#---------------------------------
#---------Validation set---------
#---------------------------------
def validate(valid_loader):
    accuracy = 0.0
    print("VALIDATING")
    index = 0
    for iter, (image, label) in  enumerate(valid_loader):
        # if(iter<100):
        with torch.no_grad():
            print("Image number", iter+1)
            image = Variable(image).to(device)
            label = Variable(label).to(device)
            # print("Image number", iter+1)
            _, _, _, _, _, _, _, out = net.forward(image)
            prob = F.softmax(out, dim = 1)
            _, max = prob.max(1)
            acc = torch.sum(max == label)
            accuracy += float(acc)
            index+=1
            print("accuracy on validation set", acc)
    print("AVERAGE ACCURACY", float(accuracy/index))


#-------------------------------------
#------------Training-----------------
#-------------------------------------
for epoch in range(0, 50):
    print("-------------------------EPOCH NUMBER-----------------------------", epoch+1)
    for iter, (image, label) in  enumerate(train_loader):
        print("------------------------------This is image number---------------------------", iter+1)
        image = Variable(image).to(device)
        label = Variable(label).to(device)
        net.zero_grad()
        Loss = net.calculate_loss(image, label)
        print("Loss", Loss.item())
        Loss.backward()
        optim.step()
    checkpoint_en = {'model': Network(num_channels, infeatures) ,'state_dict': net.state_dict(), 'optimizer' : optim.state_dict()}
    torch.save(checkpoint_en, 'self_distillation.pth')
    validate(valid_loader)
