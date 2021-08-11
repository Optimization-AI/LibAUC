"""
Author: Zhuoning Yuan
Contact: yzhuoning@gmail.com
"""

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import ResNet20
from libauc.datasets import CIFAR10
from libauc.datasets import imbalance_generator 

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
       self.images = images.astype(np.uint8)
       self.targets = targets
       self.mode = mode
       self.transform_train = transforms.Compose([                                                
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size)),
                              ])
       self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                              ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target

# paramaters
SEED = 123
BATCH_SIZE = 128
imratio = 0.1
lr = 0.1
gamma = 500
weight_decay = 1e-4
margin = 1.0


# dataloader 
(train_data, train_label), (test_data, test_label) = CIFAR10()
(train_images, train_labels) = imbalance_generator(train_data, train_label, imratio=imratio, shuffle=True, random_seed=SEED)
(test_images, test_labels) = imbalance_generator(test_data, test_label, is_balanced=True,  random_seed=SEED)

trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader( ImageDataset(test_images, test_labels, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=1,  pin_memory=True)

# model 
model = ResNet20(pretrained=False, num_classes=1)
model = model.cuda()

# loss & optimizer
Loss = AUCMLoss(imratio=imratio)
optimizer = PESG(model, 
                 a=Loss.a, 
                 b=Loss.b, 
                 alpha=Loss.alpha, 
                 imratio=imratio, 
                 lr=lr, 
                 gamma=gamma, 
                 margin=margin, 
                 weight_decay=weight_decay)

def reset_model():
    # loss & optimizer
    Loss = AUCMLoss(imratio=imratio)
    optimizer = PESG(model, 
                    a=Loss.a, 
                    b=Loss.b, 
                    alpha=Loss.alpha, 
                    imratio=imratio, 
                    lr=lr, 
                    gamma=gamma, 
                    margin=margin, 
                    weight_decay=weight_decay)
    return Loss, optimizer

total_epochs = 10
Loss, optimizer = reset_model()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader)*total_epochs)

model.train()    
for epoch in range(total_epochs):
     for i, (data, targets) in enumerate(trainloader):
         data, targets  = data.cuda(), targets.cuda()
         y_pred = model(data)
         loss = Loss(y_pred, targets)
         optimizer.zero_grad()
         loss.backward(retain_graph=True)
         optimizer.step()
         scheduler.step()
     print("epoch: {}, loss: {:4f}, lr:{:4f}".format(epoch, loss.item(), optimizer.lr))          

total_epochs = 20
Loss, optimizer = reset_model()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       patience=3,  
                                                       verbose=True, 
                                                       factor=0.5, 
                                                       threshold=0.001,
                                                       min_lr=0.00001)

model.train()    
for epoch in range(total_epochs):
     for i, (data, targets) in enumerate(trainloader):
         data, targets  = data.cuda(), targets.cuda()
         y_pred = model(data)
         loss = Loss(y_pred, targets)
         optimizer.zero_grad()
         loss.backward(retain_graph=True)
         optimizer.step()
     scheduler.step(loss)
     print("epoch: {}, loss: {:4f}, lr:{:4f}".format(epoch, loss.item(), optimizer.lr))          

total_epochs = 20
Loss, optimizer = reset_model()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)

# reset model
model.train()    
for epoch in range(total_epochs):
     for i, (data, targets) in enumerate(trainloader):
         data, targets  = data.cuda(), targets.cuda()
         y_pred = model(data)
         loss = Loss(y_pred, targets)
         optimizer.zero_grad()
         loss.backward(retain_graph=True)
         optimizer.step()
     scheduler.step()
     print("epoch: {}, loss: {:4f}, lr:{:4f}".format(epoch, loss.item(), optimizer.lr))          
