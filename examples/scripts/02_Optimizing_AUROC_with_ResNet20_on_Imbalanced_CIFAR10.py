"""
Author: Zhuoning Yuan
Contact: yzhuoning@gmail.com
"""

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import ResNet20
from libauc.datasets import CIFAR10
from libauc.datasets import ImbalanceGenerator 

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
(train_images, train_labels) = ImbalanceGenerator(train_data, train_label, imratio=imratio, shuffle=True, random_seed=SEED)
(test_images, test_labels) = ImbalanceGenerator(test_data, test_label, is_balanced=True,  random_seed=SEED)

trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader( ImageDataset(test_images, test_labels, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=1,  pin_memory=True)

# You need to include sigmoid activation in the last layer for any customized models!
model = ResNet20(pretrained=False, num_classes=1)
model = model.cuda()

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

print ('Start Training')
print ('-'*30)
for epoch in range(100):
    
     if epoch == 50 or epoch==75:
         # decrease learning rate by 10x & update regularizer
         optimizer.update_regularizer(decay_factor=10)
   
     train_pred = []
     train_true = []
     model.train()    
     for data, targets in trainloader:
         data, targets  = data.cuda(), targets.cuda()
         y_pred = model(data)
         loss = Loss(y_pred, targets)
         optimizer.zero_grad()
         loss.backward(retain_graph=True)
         optimizer.step()
        
         train_pred.append(y_pred.cpu().detach().numpy())
         train_true.append(targets.cpu().detach().numpy())

     train_true = np.concatenate(train_true)
     train_pred = np.concatenate(train_pred)
     train_auc = roc_auc_score(train_true, train_pred) 

     model.eval()
     test_pred = []
     test_true = [] 
     for j, data in enumerate(testloader):
         test_data, test_targets = data
         test_data = test_data.cuda()
         y_pred = model(test_data)
         test_pred.append(y_pred.cpu().detach().numpy())
         test_true.append(test_targets.numpy())
     test_true = np.concatenate(test_true)
     test_pred = np.concatenate(test_pred)
     val_auc =  roc_auc_score(test_true, test_pred) 
     model.train()
   
     # print results
     print("epoch: {}, train_loss: {:4f}, train_auc:{:4f}, test_auc:{:4f}, lr:{:4f}".format(epoch, loss.item(), train_auc, val_auc, optimizer.lr ))          
