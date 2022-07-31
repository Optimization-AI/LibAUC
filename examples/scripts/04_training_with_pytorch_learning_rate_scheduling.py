# -*- coding: utf-8 -*-
"""04_Training_with_Pytorch_Learning_Rate_Scheduling.ipynb

**Author**: Zhuoning Yuan
**Introduction**

In this tutorial, you will learn how to quickly train models using LibAUC with [Pytorch Learning Rate Scheduler](https:/https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook/). After completion of this tutorial, you should be able to use LibAUC to train your own models on your own datasets.

**Useful Resources**:
* Website: https://libauc.org
* Github: https://github.com/Optimization-AI/LibAUC

**Reference**:  

If you find this tutorial helpful in your work,  please acknowledge our library and cite the following paper:

@inproceedings{yuan2021large,
  title={Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification},
  author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3040--3049},
  year={2021}
  }
"""

"""# **Importing AUC Training Pipeline**"""

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import resnet20 as ResNet20
from libauc.datasets import CIFAR10
from libauc.utils import ImbalancedDataGenerator
from libauc.metrics import auc_roc_score

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

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
epoch_decay = 2e-3 # 1/gamma
weight_decay = 1e-4
margin = 1.0


# dataloader 
(train_data, train_label) = CIFAR10(root='./data', train=True) 
(test_data, test_label) = CIFAR10(root='./data', train=False) 

generator = ImbalancedDataGenerator(verbose=True, random_seed=0)
(train_images, train_labels) = generator.transform(train_data, train_label, imratio=imratio)
(test_images, test_labels) = generator.transform(test_data, test_label, imratio=0.5)

trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader(ImageDataset(test_images, test_labels, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=1,  pin_memory=True)

# model 
model = ResNet20(pretrained=False, num_classes=1)
model = model.cuda()

# loss & optimizer
loss_fn = AUCMLoss()
optimizer = PESG(model, 
                 loss_fn=loss_fn,
                 lr=lr, 
                 margin=margin,
                 epoch_decay=epoch_decay, 
                 weight_decay=weight_decay)

"""# **Pytorch Learning Rate Scheduling**
We will cover three scheduling functions in this section: 
*   CosineAnnealingLR
*   ReduceLROnPlateau
*   MultiStepLR

For more details, please refer to orginal PyTorch [doc](https://pytorch.org/docs/stable/optim.html).

"""

def reset_model():
    # loss & optimizer
    loss_fn = AUCMLoss()
    optimizer = PESG(model, 
                    loss_fn=loss_fn,
                    lr=lr, 
                    epoch_decay=epoch_decay, 
                    margin=margin, 
                    weight_decay=weight_decay)
    return loss_fn, optimizer

"""### CosineAnnealingLR"""

total_epochs = 10
loss_fn, optimizer = reset_model()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader)*total_epochs)

model.train()    
for epoch in range(total_epochs):
     for i, (data, targets) in enumerate(trainloader):
         data, targets  = data.cuda(), targets.cuda()
         y_pred = model(data)
         y_pred = torch.sigmoid(y_pred)
         loss = loss_fn(y_pred, targets)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         scheduler.step()
     print("epoch: {}, loss: {:4f}, lr:{:4f}".format(epoch, loss.item(), optimizer.lr))

"""### ReduceLROnPlateau"""

total_epochs = 20
loss_fn, optimizer = reset_model()
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
         y_pred = torch.sigmoid(y_pred)
         loss = loss_fn(y_pred, targets)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
     scheduler.step(loss)
     print("epoch: {}, loss: {:4f}, lr:{:4f}".format(epoch, loss.item(), optimizer.lr))

"""### MultiStepLR"""

total_epochs = 20
loss_fn, optimizer = reset_model()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)

# reset model
model.train()    
for epoch in range(total_epochs):
     for i, (data, targets) in enumerate(trainloader):
         data, targets  = data.cuda(), targets.cuda()
         y_pred = model(data)
         y_pred = torch.sigmoid(y_pred)
         loss = loss_fn(y_pred, targets)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
     scheduler.step()
     print("epoch: {}, loss: {:4f}, lr:{:4f}".format(epoch, loss.item(), optimizer.lr))