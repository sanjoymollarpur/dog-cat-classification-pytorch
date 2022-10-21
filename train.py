import os
import cv2
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet50
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split

from PIL import Image

import matplotlib.pyplot as plt
# %matplotlib inline

# DIR_TRAIN = "dogcat/training_set/"
# DIR_TEST = "dogcat/test_set/"
# imgs = os.listdir(DIR_TRAIN) 
# test_imgs = os.listdir(DIR_TEST)

# print(imgs[:5])
# print(test_imgs[:5])
# print(DIR_TRAIN)


lr = 0.001 # learning_rate
batch_size = 16 # we will use mini-batch method
epochs = 10 # How much to train a model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1234)
if device =='cuda':
    torch.cuda.manual_seed_all(1234)

train_dir = "dogcat/training_set/cats"
train_dir1 = "dogcat/training_set/dogs"
test_dir = "dogcat/test_set/cats"
test_dir1 = "dogcat/test_set/dogs"

import glob




train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
print(len(train_list))
train_list+=glob.glob(os.path.join(train_dir1,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
print("test", len(test_list))
test_list += glob.glob(os.path.join(test_dir1, '*.jpg'))
print(len(train_list))
print("test", len(test_list))

print(train_list[0])

from PIL import Image
random_idx = np.random.randint(1,8000,size=10)

fig = plt.figure()
i=1
# for idx in random_idx:
#     ax = fig.add_subplot(2,5,i)
#     img = Image.open(train_list[idx])
#     plt.imshow(img)
#     i+=1

# plt.axis('off')
# plt.show()

print(train_list[0].split('/')[-1].split('.')[0])

# print(int(test_list[0].split('/')[-1].split('.')[0]))


from sklearn.model_selection import train_test_split
train_list, val_list = train_test_split(train_list, test_size=0.2)

print(len(train_list), len(val_list))



train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


test_transforms = transforms.Compose([   
    transforms.Resize((224, 224)),
     transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])



class dataset(torch.utils.data.Dataset):
    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform
        
        
    #dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    #load an one of images
    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label=1
        elif label == 'cat':
            label=0
            
        return img_transformed,label
        

train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=test_transforms)



train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))

print(train_data[0][0].shape)



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr







class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out



model = Cnn().to(device)
model.train()

optimizer = optim.Adam(params = model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 4
train_acc=[]
test_acc=[]
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    torch.cuda.empty_cache() 
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        
        output = model(data)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
        
        train_acc.append(epoch_accuracy)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
    
    
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            
            val_output = model(data)
            val_loss = criterion(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)
            
        if epoch_val_accuracy>=0.5:
            save_checkpoint(model, optimizer, filename=f"weight/wt_lr_epoch-{epochs}.pth.tar")
        
        test_acc.append(epoch_val_accuracy)

        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))


