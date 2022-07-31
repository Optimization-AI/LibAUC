# -*- coding: utf-8 -*-
"""11_Optimizing_pAUC_Loss_on_Imbalanced_data_wrapper.ipynb

Author: Zhuoning Yuan

Introduction:

In this tutorial, you will learn how to quickly train a ResNet18 model by optimizing **Partial AUC (pAUC)** score using our novel optimization methods on a binary image classification task on Cifar10. 
This is a **wrapper** of original implementations of partial AUC losses. For orginal tutorials, please refer to  
- SOPA: https://github.com/Optimization-AI/LibAUC/blob/main/examples/11_Optimizing_pAUC_Loss_with_SOPA_on_Imbalanced_data.ipynb 
- SOPA-s: https://github.com/Optimization-AI/LibAUC/blob/main/examples/11_Optimizing_pAUC_Loss_with_SOPAs_on_Imbalanced_data.ipynb
- SOTA-s: https://github.com/Optimization-AI/LibAUC/blob/main/examples/11_Optimizing_pAUC_Loss_with_SOTAs_on_Imbalanced_data.ipynb 
After completion of this tutorial, you should be able to use LibAUC to train your own models on your own datasets.

References:

If you find this tutorial helpful in your work,  please acknowledge our library and cite the following papers:
@article{zhu2022auc,
  title={When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee},
  author={Zhu, Dixian and Li, Gang and Wang, Bokun and Wu, Xiaodong and Yang, Tianbao},
  journal={arXiv preprint arXiv:2203.00176},
  year={2022}
}

"""

from libauc.models import resnet18
from libauc.datasets import CIFAR10
from libauc.losses.auc import pAUCLoss  # default: SOPA
from libauc.optimizers import SOPA
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler  # data resampling (for binary class)
from libauc.metrics import pauc_roc_score

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import torch 

import warnings
warnings.filterwarnings("ignore")


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
        return image, target, int(idx)



# general params
lr = 1e-3
weight_decay = 5e-4
total_epoch = 60
decay_epochs = [20, 40]
batch_size = 64

# By default, we use one-way partial AUC loss (SOPA)
alpha = 0.  # a: min_tpr=0. This is fixed (for reference only)
beta = 0.1  # b: max_fpr=0.1

# By default, pAUCLoss calls SOPA in the backend
margin = 1.0
eta = 1e1 # learning rate for control negative samples weights

# sampling parameters
sampling_rate = 0.5

train_data, train_targets = CIFAR10(root='./data', train=True)
test_data, test_targets  = CIFAR10(root='./data', train=False)

imratio = 0.2
generator = ImbalancedDataGenerator(shuffle=True, verbose=True, random_seed=0)
(train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
(test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5) 

trainDataset = ImageDataset(train_images, train_labels)
testDataset = ImageDataset(test_images, test_labels, mode='test')

"""Now, we can import data and load the dataset!"""

trainSet = ImageDataset(train_images, train_labels)
testSet = ImageDataset(test_images, test_labels, mode='test')

sampler = DualSampler(trainSet, batch_size, sampling_rate=sampling_rate)
trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,  sampler=sampler,  shuffle=False,  num_workers=1)
testloader = torch.utils.data.DataLoader(testSet , batch_size=batch_size, shuffle=False, num_workers=1)


seed = 123
set_all_seeds(seed)
model = resnet18(pretrained=False, num_classes=1, last_activation=None) 
model = model.cuda()

loss_fn = pAUCLoss(pos_len=sampler.pos_len, backend='SOPA', beta=beta, margin=margin)
optimizer = SOPA(model.parameters(), loss_fn=loss_fn.loss_fn, mode='adam', lr=lr, eta=eta, weight_decay=weight_decay)


print ('Start Training')
print ('-'*30)
test_best = 0
train_list, test_list = [], []
for epoch in range(total_epoch):
    
    if epoch in decay_epochs:
        optimizer.update_lr(decay_factor=10, coef_decay_factor=10)
            
    train_pred, train_true = [], []
    model.train() 
    for idx, (data, targets, index) in enumerate(trainloader):
        data, targets  = data.cuda(), targets.cuda()
        y_pred = model(data)
        y_prob = torch.sigmoid(y_pred)
        loss = loss_fn(y_prob, targets, index_p=index) # Notes: make index>0 for positive samples, and index<0 for negative samples
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
plt.rcParams["figure.figsize"] = (9,5)
x=np.arange(60)
aucm_tr_pAUC = [0.6429585593723751, 0.705619972966429, 0.69894917006115, 0.7947875283260033, 0.7677915794870532, 0.7694341408326258, 0.8345354696048994, 0.8353778185042913, 0.8261352422411384, 0.850143596501959, 0.8749833709434203, 0.8778469862659533, 0.8833579163047021, 0.8755243973017512, 0.874157483715067, 0.8366073362501383, 0.8920074624226216, 0.8941630657880417, 0.8697558150194012, 0.8712549503124775, 0.9686330493047073, 0.9769933546518772, 0.9815388471447516, 0.9843683141845079, 0.9862415423489106, 0.9870076275453813, 0.9891961884708532, 0.9900528747015318, 0.9912304734205963, 0.9912435867529628, 0.9929736720264557, 0.9928194070381502, 0.9935700668345326, 0.9933461158425558, 0.9939052576360476, 0.9946626137958484, 0.9945831000132086, 0.9943817922164405, 0.9950893414460829, 0.9952917756106698, 0.9958267789776483, 0.9955192943758522, 0.9957648139930699, 0.9957864767962223, 0.9961289076816299, 0.99591124055382, 0.9961174449741217, 0.9958902131376424, 0.996271318496243, 0.9960302217876651, 0.9959968608322265, 0.995976135413218, 0.9958279797460171, 0.9962699055136474, 0.9961840783577622, 0.996047774666769, 0.9964594341651255, 0.9964745506616648, 0.9960480697575578, 0.996416429077646]
aucm_te_pAUC = [0.6279169411764706, 0.696358431372549, 0.673276862745098, 0.7493461960784313, 0.7261282745098039, 0.7153201960784313, 0.7762015294117646, 0.781920274509804, 0.778698862745098, 0.773890431372549, 0.8124459607843137, 0.802709294117647, 0.8055946274509804, 0.8052427843137255, 0.8037347843137256, 0.789566862745098, 0.8226527450980392, 0.819271725490196, 0.8068901176470589, 0.8064509019607844, 0.8594197647058823, 0.8614054901960784, 0.8630056862745098, 0.8605498823529412, 0.8648399607843138, 0.8607361176470588, 0.8658290196078431, 0.8603041176470588, 0.8643641960784314, 0.8594430588235293, 0.8661986274509804, 0.8604558823529411, 0.8617808235294118, 0.8613043529411765, 0.8562662352941177, 0.859865137254902, 0.8595644705882353, 0.865197294117647, 0.858666862745098, 0.8554585490196078, 0.8611540392156862, 0.8627088627450981, 0.8615218823529411, 0.8601713725490197, 0.8619247450980392, 0.8619268627450981, 0.8633670196078431, 0.8617367058823528, 0.8618482352941177, 0.859918, 0.8597246666666667, 0.860948705882353, 0.8607091372549018, 0.8608580784313726, 0.8610979215686274, 0.8626889803921569, 0.8603958039215687, 0.8605810980392157, 0.8616598039215686, 0.8612713725490196]
plt.figure()
plt.plot(x, train_list, LineStyle='--', label='SOPA train', linewidth=3)
plt.plot(x, test_list, label='SOPA test', linewidth=3)
plt.plot(x, aucm_tr_pAUC, LineStyle='--', label='PESG train', linewidth=3)
plt.plot(x, aucm_te_pAUC, label='PESG test', linewidth=3)
plt.title('CIFAR-10 (20% imbalanced)',fontsize=30)
plt.legend(fontsize=15)
plt.ylabel('OPAUC(0.3)',fontsize=25)
plt.xlabel('epochs',fontsize=25)
