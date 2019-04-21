#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:41:15 2019

@author: zhouminghao
"""

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator_ori as dg
from data_generator_ori import DenoisingDataset
import matplotlib.pyplot as plt


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--sigma', default=50, type=float, help='noise level')
parser.add_argument('--lam', default=0.1, type=float, help='regularization weight')
parser.add_argument('--epoch', default=10, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma
lam = args.lam

save_dir = os.path.join('models', 'cat_gauss50_tao1_decay0.92_eta0.1_lam0.1_855_epo10')

if not os.path.exists('models/'):
    os.mkdir('models/')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)


class CNNBlock0(nn.Module): #momentum
    def __init__(self, image_channels, n_channels, level):
        super(CNNBlock0, self).__init__()
        layers2=[]
        layers2.append(nn.Conv2d(in_channels=image_channels,
                                 out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers2.append(nn.ReLU(inplace=True))
        for _ in range(level-2):
            layers2.append(nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers2.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.9))
            layers2.append(nn.ReLU(inplace=True))
#        layers2.append(nn.Conv2d(in_channels=n_channels,
#                                 out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers2)
        
    def forward(self, x):
        out = self.cnnblock(x)
        return out

class CNNBlock1(nn.Module): #momentum
    def __init__(self, image_channels, n_channels):
        super(CNNBlock1, self).__init__()
        layers2=[]
        layers2.append(nn.Conv2d(in_channels=n_channels,
                                 out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers2)
        
    def forward(self, x, x0):
        out = self.cnnblock(x)
        return out+x0

class CNNBlock2(nn.Module): #momentum
    def __init__(self, image_channels, n_channels, level):
        super(CNNBlock2, self).__init__()
        layers2=[]
        layers2.append(nn.Conv2d(in_channels=image_channels+n_channels,
                                 out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers2.append(nn.ReLU(inplace=True))
        for _ in range(level-2):
            layers2.append(nn.Conv2d(in_channels=n_channels,
                                     out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers2.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.9))
            layers2.append(nn.ReLU(inplace=True))
#        layers2.append(nn.Conv2d(in_channels=n_channels,
#                                 out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers2)
        
    def forward(self, x,x0):
        x1=torch.cat((x0,x),1)
        out = self.cnnblock(x1)
        return out


class PADMM(nn.Module):
    def __init__(self,tao=1,eta=0.1,decay=0.92, level=8, subnet=5, n_channels=64, image_channels=1):
        super(PADMM, self).__init__()
        self.tao=tao
        self.level = level
        self.eta=eta
        self.decay = decay
        self.proxNet0 = CNNBlock0(image_channels,n_channels,subnet)
        self.proxNet1 = CNNBlock1(image_channels,n_channels)
        self.proxNet2 = CNNBlock2(image_channels,n_channels,subnet)
        #self.proxNet_final = CNNBlock1(image_channels,n_channels,final_subnet)
        self._initialize_weights()

    def updateX(self,V,Y,tao):
        #return (V-tao+torch.sqrt_((V-tao)**2+4*tao*Y)).div_(2)
        return (Y+tao*V).div_(1+tao)

    def forward(self, Y):
        X = Y
        X_p= Y
        listV = []
        tao = self.tao
        temp = self.proxNet0(X)
        for i in range(self.level-1):
            V = self.proxNet1(temp,X)
            listV.append(V)
            V=V+self.eta*(X-X_p)
            X_p=X
            X = self.updateX(V,Y,tao)
            tao *=self.decay
            temp = self.proxNet2(X,temp)
        V = self.proxNet1(temp,X)
        return listV, V
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               #init.kaiming_normal_(m.weight,nonlinearity='relu')
                init.orthogonal_(m.weight)
               #print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1/100)
                init.constant_(m.bias, 0)

class Loss_func(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(Loss_func, self).__init__(size_average, reduce, reduction)

    def forward(self, listV, V, X):
        loss1 = torch.nn.functional.mse_loss(V, X, size_average=None, reduce=None, reduction='sum').div_(2)
        level = len(listV)
        loss2 = 0
        for i in range(level):
            loss2 += lam*torch.nn.functional.mse_loss(listV[i], X, size_average=None, reduce=None, reduction='sum').div_(2)
        print('loss:{0}__loss1:{1}__loss2:{2}'.format((loss1+loss2)/batch_size,loss1/batch_size,loss2/batch_size))
        return loss1+loss2


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def imshow(X):
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imshow(X.squeeze(),cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = PADMM()
    #model2 = DnCNN(depth=3)
    
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = Loss_func()
    if cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[6,9], gamma=0.2)  # learning rates
    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
                optimizer.zero_grad()
                if cuda:
                    batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
                else:
                    batch_x = batch_yx[1]
                    batch_y = batch_yx[0]
                loss = criterion(model(batch_y)[0],model(batch_y)[1], batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if n_count % 10 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
                    show1 = batch_yx[1].cpu().detach().numpy()[0][0]
                    show2 = batch_yx[0].cpu().detach().numpy()[0][0]
                    show3 = model(batch_y)[1].cpu().detach().numpy()[0][0]
                    toshow = np.hstack((show1,show2,show3))
                    imshow(toshow)
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))






