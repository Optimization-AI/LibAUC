# -*- coding: utf-8 -*-
"""11_Optimizing_pAUC_Loss_with_SOTAs_on_Imbalanced_data.ipynb

Author: Dixian Zhu  
Edited by: Zhuoning Yuan

Introduction:

In this tutorial, we will learn how to quickly train a ResNet18 model by optimizing **two way partial AUC** score using our novel **`SOTA-s`** method on a binary image classification task on Cifar10. After completion of this tutorial, you should be able to use LibAUC to train your own models on your own datasets.

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
from libauc.losses.auc import tpAUC_KL_Loss
from libauc.optimizers import SOTAs
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler # data resampling (for binary class)
from libauc.metrics import pauc_roc_score

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

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
        return image, target, idx


# paramaters
SEED = 123
batch_size = 64
total_epochs = 60
weight_decay = 1e-4
lr = 1e-3 
decay_epochs = [20, 40]
decay_factor = 10

gamma0 = 0.5 # learning rate for control negative samples weights
gamma1 = 0.5 # learning rate for control positive samples weights

tau = 1.0 # KL-DRO regularization for outer positive samples part  # 
Lambda = 0.5 # KL-DRO regularization for inner negative samples part # 

sampling_rate = 0.5 
num_pos = round(sampling_rate*batch_size) 
num_neg = batch_size - num_pos


train_data, train_targets = CIFAR10(root='./data', train=True)
test_data, test_targets  = CIFAR10(root='./data', train=False)

imratio = 0.2
g = ImbalancedDataGenerator(shuffle=True, verbose=True, random_seed=0)
(train_images, train_labels) = g.transform(train_data, train_targets, imratio=imratio)
(test_images, test_labels) = g.transform(test_data, test_targets, imratio=0.5) 

traindSet = ImageDataset(train_images, train_labels)
testSet = ImageDataset(test_images, test_labels, mode = 'test')
sampler = DualSampler(dataset=traindSet, batch_size=batch_size, shuffle=True, sampling_rate=sampling_rate)
trainloader =  torch.utils.data.DataLoader(traindSet, sampler=sampler, batch_size=batch_size, shuffle=False, num_workers=1)   
testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)

"""## **Model and Loss setup**"""

set_all_seeds(SEED)
model = resnet18(pretrained=False, num_classes=1, last_activation=None)
model = model.cuda() 

loss_fn = tpAUC_KL_Loss(pos_len=sampler.pos_len, Lambda=Lambda, tau=tau)
optimizer = SOTAs(model, loss_fn=loss_fn, mode='adam', lr=lr, gammas=(gamma0, gamma1), weight_decay=weight_decay)

"""## **Training**"""

print ('Start Training')
print ('-'*30)

tr_tpAUC=[]
te_tpAUC=[]

for epoch in range(total_epochs):
    if epoch in decay_epochs:
        optimizer.update_lr(decay_factor=decay_factor, coef_decay_factor=decay_factor)
 
    train_loss = 0
    model.train()
    for idx, data in enumerate(trainloader):
        train_data, train_labels, index = data
        train_data, train_labels = train_data.cuda(), train_labels.cuda()
        y_pred = model(train_data)
        y_prob = torch.sigmoid(y_pred)
        loss = loss_fn(y_prob, train_labels, index[:num_pos])
        train_loss = train_loss  + loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/(idx+1)
   
    # evaluation
    model.eval()
    with torch.no_grad():    
        train_pred = []
        train_true = []
        for jdx, data in enumerate(trainloader):
            train_data, train_labels,_ = data
            train_data = train_data.cuda()
            y_pred = model(train_data)    
            y_prob = torch.sigmoid(y_pred)
            train_pred.append(y_prob.cpu().detach().numpy())
            train_true.append(train_labels.numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        single_train_auc =  pauc_roc_score(train_true, train_pred, max_fpr = 0.3, min_tpr=0.7) 

        test_pred = []
        test_true = [] 
        for jdx, data in enumerate(testloader):
            test_data, test_labels, _ = data
            test_data = test_data.cuda()
            y_pred = model(test_data)    
            y_prob = torch.sigmoid(y_pred)
            test_pred.append(y_prob.cpu().detach().numpy())
            test_true.append(test_labels.numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        single_test_auc =  pauc_roc_score(test_true, test_pred, max_fpr = 0.3, min_tpr=0.7) 
        print('Epoch=%s, Loss=%0.4f, Train_tpAUC(0.3,0.7)=%.4f, Test_tpAUC(0.3,0.7)=%.4f, lr=%.4f'%(epoch, train_loss, single_train_auc, single_test_auc, optimizer.lr))
        
        tr_tpAUC.append(single_train_auc)
        te_tpAUC.append(single_test_auc)



import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,5)
x=np.arange(60)
aucm_tr_tpAUC = [0.0, 0.0036274632780175535, 0.005047971892709222, 0.16224877002854712, 0.08772204107278256, 0.09231572626711304, 0.2732654643710093, 0.26647582593555136, 0.24948327111467297, 0.3138958243403896, 0.41470878408333095, 0.43854233806173326, 0.44279415206018125, 0.41232459713094294, 0.42158369465786594, 0.28455608967939594, 0.47755676075725645, 0.4896781046120168, 0.3863189337375359, 0.41758365936793596, 0.842293465507402, 0.8849664436167597, 0.908744785930645, 0.925100663457797, 0.9326945885667315, 0.9383691079068582, 0.9493701298918441, 0.953469756067609, 0.959401363614275, 0.9611639035102166, 0.9687268876750703, 0.96784177567828, 0.9702642497143897, 0.9702752956047458, 0.9733664551144465, 0.9760006331554156, 0.9760579703978677, 0.9755799377033129, 0.9770459817114208, 0.9787550086264667, 0.9813317160205113, 0.9802740408912104, 0.9814502948096553, 0.9811205278462805, 0.9821393644867049, 0.9823378881296415, 0.9827693626631546, 0.9815768245053843, 0.9835002857701681, 0.9825182242961066, 0.9819347131316462, 0.9819089304938505, 0.9819178330190087, 0.9830069649230193, 0.982775401438961, 0.9835906895346361, 0.9843369096878757, 0.9844427527966737, 0.9829346774859711, 0.9838700118199921]
aucm_te_tpAUC = [0.0, 0.000301777777777778, 0.0, 0.050526666666666664, 0.022042666666666665, 0.006600444444444444, 0.09903022222222221, 0.10774866666666666, 0.1029888888888889, 0.08559822222222221, 0.17839955555555553, 0.16736266666666666, 0.1591968888888889, 0.14970355555555553, 0.16401288888888887, 0.10874133333333333, 0.2060488888888889, 0.20340622222222224, 0.15917866666666666, 0.15321466666666667, 0.3240817777777778, 0.3328191111111111, 0.33990511111111105, 0.33390533333333333, 0.3470675555555556, 0.3300217777777778, 0.35229422222222223, 0.32868844444444445, 0.34444044444444444, 0.33212177777777785, 0.3557311111111111, 0.33471155555555554, 0.332584, 0.3388742222222223, 0.31508044444444444, 0.3285328888888889, 0.32913555555555557, 0.35162800000000005, 0.32234244444444443, 0.31436844444444445, 0.331578888888889, 0.3387742222222222, 0.3338368888888889, 0.3277377777777778, 0.335048, 0.3345755555555555, 0.34122044444444444, 0.3344044444444444, 0.33397066666666664, 0.32639111111111113, 0.3255946666666667, 0.3287395555555556, 0.3287168888888889, 0.33072799999999997, 0.3306648888888889, 0.33826266666666666, 0.3271982222222222, 0.3275493333333333, 0.33227422222222225, 0.33158488888888893]
plt.figure()
plt.plot(x, tr_tpAUC, LineStyle='--', label='SOTA-s train', linewidth=3)
plt.plot(x, te_tpAUC, label='SOTA-s test', linewidth=3)
plt.plot(x, aucm_tr_tpAUC, LineStyle='--', label='PESG train', linewidth=3)
plt.plot(x, aucm_te_tpAUC, label='PESG test', linewidth=3)
plt.title('CIFAR-10 (20% imbalanced)',fontsize=25)
plt.legend(fontsize=15)
plt.ylabel('TPAUC(0.7,0.3)',fontsize=25)
plt.xlabel('epochs',fontsize=25)