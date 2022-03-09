"""
Author: Zhuoning Yuan
Contact: yzhuoning@gmail.com
"""

# **Citation**
"""
If you use this work,  please acknowledge our library and cite the following paper:
@inproceedings{
    yuan2022compositional,
    title={Compositional Training for End-to-End Deep {AUC} Maximization},
    author={Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=gPvB4pdu_Z}
}
@inproceedings{yuan2021robust,
	title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
	author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	year={2021}
	}
"""

"""# **Importing LibAUC**"""

from libauc.losses import CompositionalLoss
from libauc.optimizers import PDSCA
from libauc.models import ResNet20
from libauc.datasets import CIFAR10, CIFAR100, CAT_VS_DOG, STL10 
from libauc.datasets import ImbalanceGenerator

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

"""# **Reproducibility**"""

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""# **Image Dataset**"""

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

"""# **Paramaters**"""

# all paramaters
total_epochs = 200 
SEED = 123
dataset = 'C2' # choose dataset to use
imratio = 0.1
BATCH_SIZE = 128

# tunable paramaters
margin = 1.0
lr = 0.1  
#lr0 = 0.1 # refers to line 5 in algorithm 1. By default, lr0=lr unless you specify the value and pass it to optimizer
gamma = 500 
weight_decay = 1e-4
beta1 = 0.9   # try different values: e.g., [0.999, 0.99, 0.9]
beta2 = 0.999 # try different values: e.g., [0.999, 0.99, 0.9]

"""# **Loading datasets**"""

if dataset == 'C10':
    IMG_SIZE = 32
    (train_data, train_label), (test_data, test_label) = CIFAR10()
elif dataset == 'C100':
    IMG_SIZE = 32
    (train_data, train_label), (test_data, test_label) = CIFAR100()
elif dataset == 'STL10':
    BATCH_SIZE = 32
    IMG_SIZE = 96
    (train_data, train_label), (test_data, test_label) = STL10()
elif dataset == 'C2':
    IMG_SIZE = 50
    (train_data, train_label), (test_data, test_label) = CAT_VS_DOG()

(train_images, train_labels) = ImbalanceGenerator(train_data, train_label, imratio=imratio, shuffle=True, random_seed=0) # fixed seed
(test_images, test_labels) = ImbalanceGenerator(test_data, test_label, is_balanced=True,  random_seed=0)

trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels, image_size=IMG_SIZE, crop_size=IMG_SIZE-2), batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)
testloader = torch.utils.data.DataLoader(ImageDataset(test_images, test_labels, image_size=IMG_SIZE, crop_size=IMG_SIZE-2, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=8,  pin_memory=False)

"""# **Training**"""

set_all_seeds(123)
model = ResNet20(pretrained=False, last_activation=None, activations='relu', num_classes=1)
model = model.cuda()
    
# Compositional Training
Loss = CompositionalLoss(imratio=imratio)  
optimizer = PDSCA(model, 
                  a=Loss.a, 
                  b=Loss.b, 
                  alpha=Loss.alpha, 
                  lr=lr,
                  beta1=beta1,
                  beta2=beta2, 
                  gamma=gamma, 
                  margin=margin, 
                  weight_decay=weight_decay)

test_auc_max = 0
print ('-'*30)
for epoch in range(total_epochs):
    if epoch == int(0.5*total_epochs) or epoch==int(0.75*total_epochs):
      optimizer.update_regularizer(decay_factor=10)

    train_pred = []
    train_true = []
    for idx, (data, targets) in enumerate(trainloader):
        model.train()  
        data, targets  = data.cuda(), targets.cuda()
        y_pred = model(data)
        loss = Loss(y_pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_pred.append(y_pred.cpu().detach().numpy())
        train_true.append(targets.cpu().detach().numpy())
    
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_auc = roc_auc_score(train_true, train_pred) 
    
    # evaluations
    model.eval()
    test_pred = []
    test_true = [] 
    for j, data in enumerate(testloader):
        test_data, test_targets = data
        test_data = test_data.cuda()
        outputs = model(test_data)
        y_pred = torch.sigmoid(outputs)
        test_pred.append(y_pred.cpu().detach().numpy())
        test_true.append(test_targets.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_auc =  roc_auc_score(test_true, test_pred) 
    model.train()

    if test_auc_max<val_auc:
       test_auc_max = val_auc
      
    # print results
    print("epoch: {}, train_auc:{:4f}, test_auc:{:4f}, test_auc_max:{:4f}".format(epoch, train_auc, val_auc, test_auc_max, optimizer.lr ))