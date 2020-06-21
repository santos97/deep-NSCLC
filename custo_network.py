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

data_dir = 'datasets'
train_dir = os.path.join(data_dir, 'train/')
valid_dir = os.path.join(data_dir, 'vvalid/')
#test_dir = os.path.join(data_dir, 'test/')
class_names = ["LUAD","LUSC","Normal"]


# In[10]:


standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
data_transforms = {'train': transforms.Compose([transforms.Resize(size=(256,256)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   'val': transforms.Compose([transforms.Resize(size=(256,256)),
                                     transforms.ToTensor(),
                                     standard_normalization]),
                   #'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                    # transforms.ToTensor(), 
                                    # standard_normalization])
                  }
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
#test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_data,
                                           #batch_size=batch_size, 
                                           #num_workers=num_workers,
                                           #shuffle=False)
loaders_scratch = {
    'train': train_loader,
    'valid': valid_loader,
    #'test': test_loader
}


# In[3]:



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes = 3


# In[4]:


import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(128, 256, 3,stride=1, padding=1)
        self.bn4=nn.BatchNorm2d(num_features=256)
        # pool
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16*16*256, 1024)
        self.fbn1=nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fbn2=nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(512,3)
        
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        # flatten
        x = x.view(-1, 16*16*256)
    
        x = F.relu(self.fbn1(self.fc1(x)))
        
        x = self.dropout(x)
        x = F.relu(self.fbn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model_scratch = Net()
print(model_scratch)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
# move tensors to GPU if CUDA is available
if use_cuda:
    #model_scratch = nn.DataParallel(model_scratch)
    model_scratch.to(device)


# In[5]:


import torch.optim as optim

criterion= nn.CrossEntropyLoss()

# only train the classifier! -> model_transfer.classifier.parameters()
optimizer = optim.Adam(model_scratch.parameters(), lr=0.001)


# In[6]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[7]:


training_losses =[]
validation_losses=[]
accuracy=[]
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

    valid_loss_min = np.Inf
    
    print(f"Batch Size: {loaders['train'].batch_size}\n")
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        epoch_acc=0
        correct = 0
        epoch_acc_t=0
        correct_t = 0
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            _, preds_t= torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            correct_t += torch.sum(preds_t == target.data)
        epoch_acc_t = 100. * correct_t / len(loaders['train'].dataset)
            #if (batch_idx + 1) % 5 == 0:
                #print(f'Epoch:{epoch}/{n_epochs} \tBatch:{batch_idx + 1}')
                #print(f'Train Loss: {train_loss}\n')

        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
                _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            correct += torch.sum(preds == target.data)
        epoch_acc = 100. * correct / len(loaders['valid'].dataset)
        # print training/validation statistics 
        print('Epoch: {} \tTrain Loss: {:.6f} \t train acc  {:.6f} \t Val Loss: {:.6f} \t Val Acc: {:.6f}'.format(
            epoch, 
            train_loss,
            epoch_acc_t,
            valid_loss,
            epoch_acc
            ))
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        accuracy.append(epoch_acc)
        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model



model_transfer = train(40, loaders_scratch, model_scratch, optimizer,
                       criterion, use_cuda, 'models/model_final_new_custom.pt')

# load the model that got the best validation accuracy
#model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# In[8]:


import matplotlib.pyplot as plt

loss_train = [0.152519,0.135324,0.126982,0.116104,0.110863,0.098519,0.095795,0.085791,0.084670,0.080025]
loss_val = [8.200484,8.290196, 6.233508,7.335477,13.756619,10.444750,14.094317,15.913617,11.181411,21.083136]
epochs=[1,2,3,4,5,6,7,8,9,10]
#accurac_y=accuracy 
#epochs = np.array(range(1,10))
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
#plt.plot(epochs, accurac_y, 'r', label='Accuracy')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[6]:


model =  Net()
model.load_state_dict(torch.load('models/model_final_new_custom.pt'))
print("weights loaded")


# In[7]:


for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# In[ ]:




