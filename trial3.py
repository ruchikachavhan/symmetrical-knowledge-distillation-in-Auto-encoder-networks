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
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
args = parser.parse_args()

use_cuda = 1
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {}
import torchvision.utils as vutils



loss_file = open("with_distillation1_2.txt", "w")

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size


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
    imageSize = (64, 64)
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), normalize,])
    
    train_dataX = []
    train_dataY = []
    val_dataX = []
    val_dataY = []

    data = []


    X_dir = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/X/"
    Y_dir = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/Y/"
    X_dir_val = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/X_val/"
    Y_dir_val = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/Y_val/"
    
    index = 0

    for image in sorted(os.listdir(X_dir)):
        print("Reading training data X", index+1)
        index+=1
        img = Image.open(X_dir+ image)
        img = transform(img)
        train_dataX.append(img)

    index = 0
    for image in sorted(os.listdir(Y_dir)):
        print("Reading training data Y", index+1)
        index+=1
        img = Image.open(Y_dir+image)
        img = transform(img)
        train_dataY.append(img)

    index = 0
    for image in sorted(os.listdir(X_dir_val)):
        print("Reading val data X", index+1)
        index+=1
        img = Image.open(X_dir_val+image)
        img = transform(img)
        val_dataX.append(img)

    index = 0
    for image in sorted(os.listdir(Y_dir_val)):
        print("Reading val data Y", index+1)
        index+=1
        img = Image.open(Y_dir_val+image)
        img = transform(img)
        val_dataY.append(img)

    print("Training data X", len(train_dataX))
    print("Training data Y", len(train_dataY))
    print("Validation data X", len(val_dataX))
    print("Validation data Y", len(val_dataY))


    train_loaderX = torch.utils.data.DataLoader(train_dataX, batch_size=batch_size,shuffle = shuffle, num_workers=num_workers)
    train_loaderY = torch.utils.data.DataLoader(train_dataY, batch_size=batch_size,shuffle = shuffle, num_workers=num_workers)
    valid_loaderX = torch.utils.data.DataLoader(val_dataX, batch_size=1, shuffle = shuffle, num_workers=num_workers)
    valid_loaderY = torch.utils.data.DataLoader(val_dataY, batch_size=1, shuffle = shuffle, num_workers=num_workers)
    return train_loaderX, train_loaderY, valid_loaderX, valid_loaderY
        


#---------------------------------------------------------------------
#-------------------------Defining Network----------------------------
#---------------------------------------------------------------------
class encoder_decoder(nn.Module):
    def __init__(self):
        super(encoder_decoder, self).__init__()
        self.enc_conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)  # b, 16, 10, 10
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(2, stride = 1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # b, 8, 3, 3
        self.enc_conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.enc_conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=0)  # b, 8, 3, 3

        self.dec_conv1 = nn.ConvTranspose2d(512, 512, 4, 1, 0, bias = False)
        self.norm1 = nn.BatchNorm2d(512)
        self.dec_conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
        self.norm2 = nn.BatchNorm2d(256)
        self.dec_conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
        self.norm3 = nn.BatchNorm2d(128)
        self.dec_conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False)
        self.norm4 = nn.BatchNorm2d(64)
        self.dec_conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False)
        self.tanh = nn.Tanh()

        self.conv_external = nn.Conv2d(64, 64, 2, stride = 1, padding = 0)

    def forward(self, image_x):
        enc_out1 = self.maxpool(self.relu(self.enc_conv1(image_x)))
        enc_out2 = self.maxpool(self.relu(self.enc_conv2(enc_out1)))
        enc_out3 = self.maxpool(self.relu(self.enc_conv3(enc_out2)))
        enc_out4 = self.maxpool(self.relu(self.enc_conv4(enc_out3)))
        enc_out5 = self.enc_conv5(enc_out4)

        dec_out1 = self.norm1(self.relu(self.dec_conv1(enc_out5)))
        dec_out2 = self.norm2(self.relu(self.dec_conv2(dec_out1)))
        dec_out3 = self.norm3(self.relu(self.dec_conv3(dec_out2)))
        dec_out4 = self.norm4(self.relu(self.dec_conv4(dec_out3)))
        dec_out5 = self.tanh(self.dec_conv5(dec_out4))

        return enc_out1, enc_out2, enc_out3, enc_out4, enc_out5, dec_out1, dec_out2, dec_out3, dec_out4, dec_out5

#---------------------------------------------------------------------
#------------Initializing hyperparameters and network-----------------
#---------------------------------------------------------------------
lr = 0.0001
batch_size = 100
shuffle = False 
num_workers = 2
valid_size = 0.99
num_channels = 3 
infeatures = 64
net1 = encoder_decoder().to(device)
net2 = encoder_decoder().to(device)
optim1 = torch.optim.RMSprop(list(net1.parameters()), lr)
optim2 = torch.optim.RMSprop(list(net2.parameters()), lr)


def validate(epoch):
    avg_loss = 0.0
    print("-------------------------VALIDATION GOING ON------------------------------")
    for (iter, image_x), (iter1, image_y) in  zip(enumerate(valid_loaderX), enumerate(valid_loaderY)):
        image_x = Variable(image_x).to(device)
        image_y = Variable(image_y).to(device)

        _, _, _, _, _, _, _, _, _, out1 = net1.forward(image_x)
        _, _, _, _, _, _, _, _, _, out2 = net1.forward(image_x)

        avg_loss += EPE(out1, image_y).detach()  

        to_save = torch.cat([out1, image_x, image_y], dim = 3)
        vutils.save_image(to_save, "results_val_model1_2/"+str(epoch)+"_"+str(iter)+".png" , normalize = True)
        to_save = torch.cat([out2, image_x, image_y], dim = 3)
        vutils.save_image(to_save, "results_val_model2_2/"+str(epoch)+"_"+str(iter)+".png" , normalize = True)
    print("Average loss for validation set", avg_loss.item()/500.00)


alpha = 0.2
maxpool = nn.MaxPool2d(2, stride=1)

if(args.mode == "train"):
    train_loaderX, train_loaderY, valid_loaderX, valid_loaderY = train_val_data(batch_size, shuffle, num_workers, valid_size)
    for epoch in range(0, 500):
        print("-------------------------EPOCH NUMBER-----------------------------", epoch+1)
        for (iter, image_x), (iter1, image_y) in  zip(enumerate(train_loaderX), enumerate(train_loaderY)):
            print("------------------------------This is image number---------------------------", iter+1)
            image_x = Variable(image_x).to(device)
            image_y = Variable(image_y).to(device)
            net1.zero_grad()
            net2.zero_grad()
            enc_out1x, enc_out2x, enc_out3x, enc_out4x, enc_out5x, dec_out1x, dec_out2x, dec_out3x, dec_out4x, dec_out5x = net1.forward(image_x)
            enc_out1y, enc_out2y, enc_out3y, enc_out4y, enc_out5y, dec_out1y, dec_out2y, dec_out3y, dec_out4y, dec_out5y = net2.forward(image_y)

            loss1 = EPE(enc_out1x, net2.conv_external(dec_out4y))
            # loss2 = EPE(enc_out2x, maxpool(dec_out3y))
            # loss3 = EPE(enc_out3x, maxpool(dec_out2y))
            # loss4 = EPE(enc_out4x, maxpool(dec_out1y))
            loss5 = EPE(dec_out5x, image_y)
            loss6 = EPE(dec_out5y, image_x)

            Loss = (1-alpha)*(loss5+loss6) + alpha*(loss1)
            print("Loss", Loss.item())
            loss_file.write(str(Loss.item()))
            loss_file.write("\n")
            Loss.backward()
            optim1.step()
            optim2.step()

            to_save1 = torch.cat([dec_out5x, image_x, image_y], dim = 3)
            vutils.save_image(to_save1, "results_train_model1_2/"+str(epoch)+"_"+str(iter)+".png" , normalize = True)
            to_save2 = torch.cat([dec_out5y, image_x, image_y], dim = 3)
            vutils.save_image(to_save2, "results_train_model2_2/"+str(epoch)+"_"+str(iter)+".png" , normalize = True)

        checkpoint_en = {'model': encoder_decoder() ,'state_dict': net1.state_dict(), 'optimizer' : optim1.state_dict()}
        torch.save(checkpoint_en, 'model1_2.pth')
        checkpoint_en = {'model': encoder_decoder() ,'state_dict': net2.state_dict(), 'optimizer' : optim2.state_dict()}
        torch.save(checkpoint_en, 'model2_2.pth')
        validate(epoch)

elif(args.mode == "test"):

    net1 = load_checkpoint('model1_2.pth').to(device)
    net2 = load_checkpoint('model2_2.pth').to(device)

    X_test = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/test_X/"
    Y_test = "/home/ruchika/GNR638Project/cityscapes-image-pairs/cityscapes_data/test_Y/"

    save_dir = "/home/ruchika/GNR638Project/RESULTS_ON_DEMO/outputs_AE3/"
    imageSize = (64, 64)
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), normalize,])

    test_X = []
    test_Y = []

    for image in os.listdir(X_test):
        img = Image.open(X_test+ image)
        img = transform(img)
        test_X.append(img)

    for image in os.listdir(Y_test):
        img = Image.open(Y_test+ image)
        img = transform(img)
        test_Y.append(img)

    test_loaderX = torch.utils.data.DataLoader(test_X, batch_size=batch_size,shuffle = shuffle, num_workers=num_workers)
    test_loaderY = torch.utils.data.DataLoader(test_Y, batch_size=batch_size,shuffle = shuffle, num_workers=num_workers)

    for (iter, image_x), (iter1, image_y) in  zip(enumerate(test_loaderX), enumerate(test_loaderY)):
        print("Producing output in directory", save_dir)
        image_x = Variable(image_x).to(device)
        image_y = Variable(image_y).to(device)
        net1.zero_grad()
        net2.zero_grad()
        enc_out1x, enc_out2x, enc_out3x, enc_out4x, enc_out5x, dec_out1x, dec_out2x, dec_out3x, dec_out4x, dec_out5x = net1.forward(image_x)
        enc_out1y, enc_out2y, enc_out3y, enc_out4y, enc_out5y, dec_out1y, dec_out2y, dec_out3y, dec_out4y, dec_out5y = net2.forward(image_y)

        to_save = torch.cat([dec_out5x, image_x, image_y], dim = 3)
        vutils.save_image(to_save, save_dir+"1"+str(iter)+".png" , normalize = True)
        to_save = torch.cat([dec_out5y, image_x, image_y], dim = 3)
        vutils.save_image(to_save, save_dir+"2"+str(iter)+".png" , normalize = True)
