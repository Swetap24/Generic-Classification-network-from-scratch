#!/usr/bin/env python
# coding: utf-8

# 
# IMPORTING PACKAGES
# This is the work performed by me tohelp people perform some basic DL applications on their own
# Sweta Priyadarshi
# 
# 
# ---
# 
# 

# In[ ]:


import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms 
import torchvision 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets, transforms, models
cuda = torch.cuda.is_available()
cuda


# 
# DATA LOADER
# ---
# 
# 
# 
# ---
# 
# 

# In[ ]:


def dataload_torchvision(folder_path='_path', batch_size=1, workers=1, shuffle=True):  
  data_set= torchvision.datasets.ImageFolder(root=folder_path, transform=torchvision.transforms.ToTensor())
  data_loader= DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
  return data_set, data_loader


# In[ ]:


train_set,train_loader = dataload_torchvision(folder_path= 'give your train data path', batch_size=64, workers=4, shuffle=True)
train_label= train_set.classes
print((train_label))


# In[ ]:


val_set,val_loader = dataload_torchvision(folder_path= 'give your validation data path', batch_size=64, workers=1, shuffle=True)
val_label= val_set.classes
print((val_label))


# In[ ]:


test_set,test_loader = dataload_torchvision(folder_path= 'give your test data path', batch_size=4, workers=1, shuffle=False)


# 
# RESNET BLOCKS
# ---
# 
# 
# 
# ---
# 
# 

# In[ ]:


class baseBlock(torch.nn.Module):
    expansion = 1
    def __init__(self,input_planes,planes,stride=1,dim_change=None):
        super(baseBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(input_planes,planes,stride=stride,kernel_size=3,padding=1)
        self.bn1   = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes,planes,stride=1,kernel_size=3,padding=1)
        self.bn2   = torch.nn.BatchNorm2d(planes)
        self.dim_change = dim_change
    def forward(self,x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        if self.dim_change is not None:
            res = self.dim_change(res)
        output = output + res
        output = F.relu(output)
        return output

class bottleNeck(torch.nn.Module):
    expansion = 4
    def __init__(self,input_planes,planes,stride=1,dim_change=None):
        super(bottleNeck,self).__init__()
        self.conv1 = torch.nn.Conv2d(input_planes,planes,kernel_size=1,stride=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes,planes*self.expansion,kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(planes*self.expansion)
        self.dim_change = dim_change
    
    def forward(self,x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output = output + res
        output = F.relu(output)
        return output

class Resnet(torch.nn.Module):
    def __init__(self,block,num_layers,classes=2):
        super(Resnet,self).__init__()
        self.input_planes = 64
        self.conv1 = torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._layer(block,64,num_layers[0],stride=1)
        self.layer2 = self._layer(block,256,num_layers[1],stride=2)
        self.layer3 = self._layer(block,512,num_layers[2],stride=2)
        self.layer4 = self._layer(block,512,num_layers[3],stride=2)
        self.averagePool = torch.nn.AvgPool2d(kernel_size=4,stride=1)
        self.fc = torch.nn.Linear(512*block.expansion,classes)
        self.sigmoid = nn.Sigmoid()
    
    def _layer(self,block,planes,num_layers,stride=1):
        dim_change = None
        if stride!=1 or planes!= self.input_planes*block.expansion:
            dim_change = torch.nn.Sequential(torch.nn.Conv2d(self.input_planes,planes*block.expansion,kernel_size=1,stride=stride),
                                             torch.nn.BatchNorm2d(planes*block.expansion))
        netLayers =[]
        netLayers.append(block(self.input_planes,planes,stride=stride,dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1,num_layers):
            netLayers.append(block(self.input_planes,planes))
        return torch.nn.Sequential(*netLayers)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,4)        
        x = x.view(x.size(0),-1)
        x = self.sigmoid(x)
        return x


# 
# Training
# ---
# 
# 
# 
# ---
# 
# 

# In[ ]:


def train():  
    device = torch.device( "cuda" if cuda else "cpu")
    print(device)
    model =  Resnet(bottleNeck,[3,4,6,3],2)
#     model.load_state_dict(torch.load("model_path"))
    model.to(device)
    costFunc = torch.nn.CrossEntropyLoss()
    optimizer =  torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(10):
        loss = 0
        iter1 = 0
        total_train = 0
        pred_train = 0
        for i,batch in enumerate(train_loader):
            data,output = batch            
            data,output = data.to(device),output.to(device)
            prediction = model(data)
            loss = costFunc(prediction,output)
            loss = loss.item()
            _,prediction_new = torch.max(prediction.data,1)
            total_train += output.size(0)
            pred_train += (prediction_new.cpu()==output.cpu()).sum().item()                    
            if i%1000 == 0:
              torch.save(model.state_dict(), "give your path /resnet_iter_{}_ep_{}.pth".format(i, epoch))
              print('[%d  %d] loss: %.4f'% (epoch+1,i+1,loss/100))
              loss = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            iter1= iter1+1


# 
# Test_Evaluation
# ---
# 
# 
# 
# ---
# 
# 
# 

# In[ ]:


test = test_dataset
testset = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False)
data_set, data_loader = dataload_torchvision( batch_size=1, workers=4, shuffle=False)
for i, [data, label] in enumerate(data_loader):
  nlabel = label.cpu().numpy().tolist()[0]


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  Resnet(bottleNeck,[3,4,6,3],2300)
model.load_state_dict(torch.load("saved model path"))
model.to(device)
model.eval()
test_loss = []
accuracy = 0
total = 0
arr=[]
actual_arr = []
for i, batch in enumerate(testset):
  data, output, file_id = batch
  file_id = file_id[0][len("test folder path"):-4]
  data_numpy = data.to(device)
  outputs = model(data_numpy)
  _, pred_labels = torch.max(outputs, 1)
  arr.append(pred_labels)
  pred_labels = pred_labels.view(-1).cpu().numpy().tolist()[0]
  actual_label = data_set.classes[pred_labels]
  actual_arr.append(actual_label)s


# In[ ]:


full_arr = []
for each in arr:
  full_arr.append(each.cpu().numpy().tolist()[0])


# In[ ]:


import pandas as pd
df = pd.DataFrame(np.vstack([range(0, max_dataset), actual_arr]).T, columns=["id","label"])
df.to_csv('filename.csv', index=False)

