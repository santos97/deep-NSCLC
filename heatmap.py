#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models
from torch import nn
import os
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import torchvision
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 32 #need to change this
num_workers = 6


# In[2]:


import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(128, 256, 3,stride=1, padding=1)
        self.bn4=nn.BatchNorm2d(num_features=256)
        # pool
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected
        self.fc1 = nn.Linear(16*16*256, 1024)
        self.fbn1=nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fbn2=nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(512,3)
        
        # drop-out
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x1 = self.pool(x)
        middle =x1
        # flatten
        x=x1
        x = x.view(-1, 16*16*256)
    
        x = F.relu(self.fbn1(self.fc1(x)))
        
        x = self.dropout(x)
        x = F.relu(self.fbn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, middle

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model = Net()


# In[23]:


from collections import OrderedDict
new_checkpoint = OrderedDict()
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
# move tensors to GPU if CUDA is available
#if use_cuda:
    #model_scratch = nn.DataParallel(model_scratch)
model.to(device)

modelCheckpoint = torch.load("../models/model_final_new_custom.pt")
for key in modelCheckpoint:
    print(key)
model.load_state_dict(modelCheckpoint)
#model = model.features
model.eval()
weights = list(model.parameters())[3]


# In[118]:


from PIL import Image
import matplotlib.pyplot as plt
standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
transform_seq = transforms.Compose([transforms.Resize(size=(256,256)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     standard_normalization])

pathImageFile = "/data/santosh_proj/datasets/valid/LUSC/559.jpeg"
imageData = Image.open(pathImageFile).convert('RGB')
imageData = transform_seq(imageData)
imageData = imageData.unsqueeze_(0)

input = torch.autograd.Variable(imageData)

output,middle = model(input.cuda())
output2=middle

output = torch.nn.functional.softmax(output)
print(output)
max_value, max_index = torch.max(output,1)
print(max_value,"pos=",max_index)


# In[51]:


()
print(middle.shape)
img = middle.squeeze_(0)

#img = Image.fromarray(img, 'RGB')
#img.show()
from torchvision import transforms
im = transforms.ToPILImage()(middle).convert("RGB")


# In[119]:


heatmap = None
import cv2
for i in range (0, len(weights)):
        
    map = output2[0,i,:,:]
    #print(map.shape)
    if i == 0: heatmap = weights[i] * map
    else: heatmap += weights[i] * map
        
        #---- Blend original and heatmap 
npHeatmap = heatmap.cpu().data.numpy()

imgOriginal = cv2.imread(pathImageFile, 1)
imgOriginal = cv2.resize(imgOriginal, (256, 256))
        
cam = npHeatmap / np.max(npHeatmap)
cam = cv2.resize(cam, (256, 256))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
img = heatmap * 0.5 + imgOriginal
            
cv2.imwrite("../models/heatmap2.jpeg", img)
cv2.imwrite("../models/original2.jpeg", imgOriginal)
class_names=["LUAD","LUSC","Normal"]
img = Image.open("../models/heatmap2.jpeg").convert('RGB')
fig, ax = plt.subplots(1, 2,figsize=(12, 12))
ax[0].imshow(imgOriginal)
ax[1].imshow(img)
print(max_value[0])
fig.text(0.02, 0.8, "Prediction probabilities: " + str('{:.4f}'.format(max_value[0])  ),fontsize=25)
fig.text(0.02, 0.85, "Predicted Class: " +  ' (' + class_names[max_index] + ')', fontsize=25)
#if label is not None:
#fig.text(0.02, 0.84, "Ground Truth Class: " + str(label) + ' (' + class_names[label] + ')', fontsize=10)
fig.suptitle("Heatmap for the input slide sample ",fontsize=25)# + img_filename, fontsize=13)
plt.savefig("../models/lusc_comp.jpeg")


# In[22]:


print(weights)


# In[9]:


for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor])


# In[ ]:





# In[ ]:





# In[ ]:




