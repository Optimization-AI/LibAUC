"""Optimizing AUROC loss on imbalanced dataset**

 Author: Zhuoning Yuan

If you find this tutorial helpful in your work,  please acknowledge our library and cite the following paper:

@inproceedings{yuan2021large,
  title={Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification},
  author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3040--3049},
  year={2021}
}

@misc{libauc2022,
      title={LibAUC: A Deep Learning Library for X-Risk Optimization.},
      author={Zhuoning Yuan, Zi-Hao Qiu, Gang Li, Dixian Zhu, Zhishuai Guo, Quanqi Hu, Bokun Wang, Qi Qi, Yongjian Zhong, Tianbao Yang},
      year={2022}
    }
"""

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import resnet20 as ResNet20
from libauc.datasets import CIFAR10
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler

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
imratio = 0.1 # for demo 
lr = 0.1
gamma = 500
weight_decay = 1e-4
margin = 1.0


# dataloader 
train_data, train_targets = CIFAR10(root='./data', train=True)
test_data, test_targets  = CIFAR10(root='./data', train=False)

generator = ImbalancedDataGenerator(verbose=True, random_seed=0)
(train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
(test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5) 

trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader( ImageDataset(test_images, test_labels, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=1,  pin_memory=True)


model = ResNet20(pretrained=False, last_activation=None, num_classes=1)
model = model.cuda()

Loss = AUCMLoss()
optimizer = PESG(model, 
                 a=Loss.a, 
                 b=Loss.b, 
                 alpha=Loss.alpha, 
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
         y_pred = torch.sigmoid(y_pred)
         loss = Loss(y_pred, targets)
         optimizer.zero_grad()
         loss.backward()
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