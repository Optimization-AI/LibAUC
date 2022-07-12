# -*- coding: utf-8 -*-
"""11_Optimizing_pAUC_Loss_with_SOPAs_on_Imbalanced_data.ipynb

Author: Gang Li  
Edited by: Zhuoning Yuan

Introduction

In this tutorial, we'll show how to use **pAUC_DRO** loss to train a Resnet18 model to maximize the `partial Area Under the Curve (pAUC)` on a binary image classification task with CIFAR-10 dataset. After completion of this tutorial, you should be able to use LibAUC to train your own models on your own datasets.

References:

If you find this tutorial helpful in your work,  please acknowledge our library and cite the following papers:
@article{zhu2022auc,
  title={When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee},
  author={Zhu, Dixian and Li, Gang and Wang, Bokun and Wu, Xiaodong and Yang, Tianbao},
  journal={arXiv preprint arXiv:2203.00176},
  year={2022}
}

"""

from libauc.losses.auc import pAUC_DRO_Loss
from libauc.optimizers import SOPAs
from libauc.models import resnet18
from libauc.datasets import CIFAR10
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler # data resampling (for binary class)
from libauc.metrics import pauc_roc_score

import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
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
       
       # for loss function
       self.pos_indices = np.flatnonzero(targets==1)
       self.pos_index_map = {}
       for i, idx in enumerate(self.pos_indices):
           self.pos_index_map[idx] = i

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            idx = self.pos_index_map[idx] if idx in self.pos_indices else -1
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target, idx


# paramaters
SEED = 123
batch_size = 64
total_epochs = 60
weight_decay = 5e-4 # regularization weight decay
lr = 1e-3  # learning rate
eta = 1e1 # learning rate for control negative samples weights
decay_epochs = [20, 40]
decay_factor = 10

gamma = 0.1 
margin = 1.0
Lambda = 1.0

sampling_rate = 0.5 
num_pos = round(sampling_rate*batch_size) 
num_neg = batch_size - num_pos


train_data, train_targets = CIFAR10(root='./data', train=True)
test_data, test_targets  = CIFAR10(root='./data', train=False)

imratio = 0.2
generator = ImbalancedDataGenerator(shuffle=True, verbose=True, random_seed=0)
(train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
(test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5) 

trainDataset = ImageDataset(train_images, train_labels)
testDataset = ImageDataset(test_images, test_labels, mode='test')

sampler = DualSampler(trainDataset, batch_size, sampling_rate=sampling_rate)
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size, sampler=sampler, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=1)

"""## **Model and Loss setup**"""

seed = 123
set_all_seeds(seed)
model = resnet18(pretrained=False, num_classes=1, last_activation=None) 
model = model.cuda()

loss_fn = pAUC_DRO_Loss(pos_len=sampler.pos_len, margin=margin, gamma=gamma, Lambda=Lambda)
optimizer = SOPAs(model.parameters(), loss_fn=loss_fn, mode='adam', lr=lr, weight_decay=weight_decay)


print ('Start Training')
print ('-'*30)
test_best = 0
train_list, test_list = [], []
for epoch in range(total_epochs):
    
    if epoch in decay_epochs:
         # decrease learning rate by 10x 
        optimizer.update_lr(decay_factor=10)
            
    train_pred, train_true = [], []
    model.train() 
    for idx, (data, targets, index) in enumerate(trainloader):
        data, targets  = data.cuda(), targets.cuda()
        y_pred = model(data)
        y_prob = torch.sigmoid(y_pred)
        loss = loss_fn(y_prob, targets, index_p=index) # postive index is selected inside loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_pred.append(y_prob.cpu().detach().numpy())
        train_true.append(targets.cpu().detach().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_pauc = pauc_roc_score(train_true, train_pred, max_fpr=0.3)
    train_list.append(train_pauc)
    
   # evaluation
    model.eval()
    test_pred, test_true = [], [] 
    for j, data in enumerate(testloader):
        test_data, test_targets, index = data
        test_data = test_data.cuda()
        y_pred = model(test_data)
        y_prob = torch.sigmoid(y_pred)
        test_pred.append(y_prob.cpu().detach().numpy())
        test_true.append(test_targets.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_pauc = pauc_roc_score(test_true, test_pred, max_fpr=0.3)
    test_list.append(val_pauc)
    
    if test_best < val_pauc:
       test_best = val_pauc
    
    model.train()
    print("epoch: %s, lr: %.4f, train_pauc: %.4f, test_pauc: %.4f, test_best: %.4f"%(epoch, optimizer.lr, train_pauc, val_pauc, test_best))

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
###
aucm_train= [0.6069506135036445, 0.6978036932301377, 0.7463894673857081, 0.7776219948089698, 0.8027497552574762, 0.8194925184801071, 0.8326414836870717, 0.8418876286652094, 0.8545082542202714, 0.8613377917975251, 0.8663205447982314, 0.8720521974101583, 0.8797435655295311, 0.8809396206077954, 0.8836973394615497, 0.8862993963546479, 0.8907247376100926, 0.8913716767613735, 0.8904567210873551, 0.8936225492461372, 0.9540770048876036, 0.9723488133661535, 0.9781509710156533, 0.9817765145216534, 0.9844834466810455, 0.986610225327087, 0.9881808086925543, 0.9899628359094543, 0.9907810165593278, 0.9914713525781993, 0.9920222707564783, 0.9929317113720121, 0.9935210368721419, 0.993572794854696, 0.9938371343578507, 0.9935073845281966, 0.9947178127253539, 0.9941674834729468, 0.9945044055783974, 0.9948909958585506, 0.9955329256867305, 0.9961777539970935, 0.9958743720990391, 0.9961726341719099, 0.995941568038736, 0.9963039071928088, 0.9965874872432375, 0.9965767861211055, 0.9964081514670973, 0.9963576105770835, 0.9963734406282638, 0.9967885112505668, 0.9964143992297314, 0.9963215825031975, 0.9964172999093988, 0.9968314808643665, 0.9968811317731188, 0.9967179493923207, 0.9970772553897975, 0.9968263538188764]
aucm_test= [0.6624721568627451, 0.6585558039215686, 0.7224398823529412, 0.7461185882352941, 0.7209969411764705, 0.7715108235294117, 0.739145568627451, 0.7252269803921568, 0.7362687450980392, 0.7746165098039216, 0.7529456862745099, 0.8016385098039216, 0.8120752549019608, 0.7091570588235294, 0.7759422745098039, 0.7294065490196078, 0.7001150588235294, 0.761414, 0.7787230980392157, 0.8003133333333334, 0.862998980392157, 0.862715294117647, 0.8590309411764705, 0.8663745098039217, 0.8619192156862745, 0.8615073333333334, 0.8627211764705882, 0.8655584705882353, 0.8625684705882353, 0.8610763921568627, 0.8653654509803922, 0.8593325490196078, 0.8611757254901962, 0.8573677647058824, 0.8564478039215686, 0.8580612549019608, 0.8614994901960784, 0.861043725490196, 0.8583090980392156, 0.853949137254902, 0.8608719215686275, 0.862686431372549, 0.8628177254901961, 0.86389, 0.8628747058823528, 0.8629372549019607, 0.8625298431372549, 0.8619865490196079, 0.8633648235294118, 0.8624133725490196, 0.8632398823529412, 0.863091568627451, 0.8631138823529412, 0.862576, 0.8623480392156863, 0.8626336078431373, 0.8629759607843137, 0.8622864705882353, 0.8625338431372549, 0.8614151372549019]
plt.plot(train_list, label='KLDRO_pAUC Training', linewidth=3)
plt.plot(aucm_train, label='AUCM Training', linewidth=3)
plt.plot(test_list, marker='_' , linestyle='dashed', label='KLDRO_pAUC Test', linewidth=3)
plt.plot(aucm_test, marker='_' , linestyle='dashed', label='AUCM Test', linewidth=3)
plt.title('pAUC Performance(FPR≤0.3)',fontsize=20)
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('pAUC',fontsize=20)
plt.legend(fontsize=15)
plt.show()