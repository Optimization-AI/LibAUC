"""09_Optimizing_CompositionalAUC_Loss_with_ResNet20_on_CIFAR10.ipynb

**Author**: Zhuoning Yuan

**Introduction**
In this tutorial, we will learn how to quickly train a ResNet20 model by optimizing AUC score using our novel compositional training framework [[Ref]](https://openreview.net/forum?id=gPvB4pdu_Z) on an binary image classification task on Cifar10. After completion of this tutorial, you should be able to use LibAUC to train your own models on your own datasets.  

**Useful Resources**
* Website: https://libauc.org
* Github: https://github.com/Optimization-AI/LibAUC


**References**
If you find this tutorial helpful in your work,  please acknowledge our library and cite the following papers:

@inproceedings{yuan2022compositional,
    title={Compositional Training for End-to-End Deep AUC Maximization},
    author={Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=gPvB4pdu_Z}
}

"""

"""# **Importing LibAUC**
Import required packages to use
"""

from libauc.losses import CompositionalAUCLoss
from libauc.optimizers import PDSCA
from libauc.models import resnet20 as ResNet20
from libauc.datasets import CIFAR10, CIFAR100, STL10, CAT_VS_DOG
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler
from libauc.metrics import auc_roc_score

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

"""# **Reproducibility**
The following function `set_all_seeds` limits the number of sources of randomness behaviors, such as model intialization, data shuffling, etcs. However, completely reproducible results are not guaranteed across PyTorch releases [[Ref]](https://pytorch.org/docs/stable/notes/randomness.html#:~:text=Completely%20reproducible%20results%20are%20not,even%20when%20using%20identical%20seeds.).
"""

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""# **Image Dataset**
Now that we defined the data input pipeline such as data augmentations. In this tutorials, we use `RandomCrop`, `RandomHorizontalFlip` as stated in the original paper. 
"""

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


# HyperParameters
SEED = 123
dataset = 'C10'
imratio = 0.1
BATCH_SIZE = 128
total_epochs = 100 
decay_epochs=[int(total_epochs*0.5), int(total_epochs*0.75)]

margin = 1.0
lr = 0.1  
#lr0 = 0.1 # (default: lr0=lr unless you specify the value and pass it to optimizer)
epoch_decay = 2e-3 
weight_decay = 1e-4
beta0 = 0.9   # e.g., [0.999, 0.99, 0.9]
beta1 = 0.9   # e.g., [0.999, 0.99, 0.9] 

sampling_rate = 0.2


"""# **Loading datasets**
In this step, we will use the [CIFAR10](http://yann.lecun.com/exdb/mnist/) as benchmark dataset. Before importing data to `dataloader`, we construct imbalanced version for CIFAR10 by `ImbalanceGenerator`. Specifically, it first randomly splits the training data by class ID (e.g., 10 classes) into two even portions as the positive and negative classes, and then it randomly removes some samples from the positive class to make
it imbalanced. We keep the testing set untouched. We refer `imratio` to the ratio of number of positive examples to number of all examples.
"""
if dataset == 'C10':
    IMG_SIZE = 32
    train_data, train_targets = CIFAR10(root='./data', train=True)
    test_data, test_targets  = CIFAR10(root='./data', train=False)
elif dataset == 'C100':
    IMG_SIZE = 32
    train_data, train_targets = CIFAR100(root='./data', train=True)
    test_data, test_targets  = CIFAR100(root='./data', train=False)
elif dataset == 'STL10':
    BATCH_SIZE = 32
    IMG_SIZE = 96
    train_data, train_targets = STL10(root='./data/', split='train')
    test_data, test_targets = STL10(root='./data/', split='test')
elif dataset == 'C2':
    IMG_SIZE = 50
    train_data, train_targets  = CAT_VS_DOG('./data/', train=True)
    test_data, test_targets = CAT_VS_DOG('./data/', train=False)

(train_images, train_labels) = ImbalancedDataGenerator(verbose=True, random_seed=0).transform(train_data, train_targets, imratio=imratio)
(test_images, test_labels) = ImbalancedDataGenerator(verbose=True, random_seed=0).transform(test_data, test_targets, imratio=0.5) 

trainSet = ImageDataset(train_images, train_labels, image_size=IMG_SIZE, crop_size=IMG_SIZE-2)
trainSet_eval = ImageDataset(train_images, train_labels, image_size=IMG_SIZE, crop_size=IMG_SIZE-2, mode='test')
testSet = ImageDataset(test_images, test_labels, image_size=IMG_SIZE, crop_size=IMG_SIZE-2, mode='test')

# parameters for sampler
sampler = DualSampler(trainSet, batch_size=BATCH_SIZE, sampling_rate=sampling_rate)
trainloader = torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=2)
trainloader_eval = torch.utils.data.DataLoader(trainSet_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


"""# **Model, Loss & Optimizer**
Before training, we need to define **model**, **loss function**, **optimizer**. 
"""
set_all_seeds(123)
model = ResNet20(pretrained=False, last_activation=None, activations='relu', num_classes=1)
model = model.cuda()
    
# Compositional Training
loss_fn = CompositionalAUCLoss()  
optimizer = PDSCA(model, 
                  loss_fn=loss_fn,
                  lr=lr,
                  beta1=beta0,
                  beta2=beta1, 
                  margin=margin, 
                  epoch_decay=epoch_decay, 
                  weight_decay=weight_decay)

"""# **Training**
Now it's time for training
"""
print ('Start Training')
print ('-'*30)

train_log = []
test_log = []
for epoch in range(total_epochs):
     if epoch in decay_epochs:
         optimizer.update_regularizer(decay_factor=10, decay_factor0=10) # decrease learning rate by 10x & update regularizer
   
     train_loss = []
     model.train()    
     for data, targets in trainloader:
         data, targets  = data.cuda(), targets.cuda()
         y_pred = model(data)
         loss = loss_fn(y_pred, targets)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         train_loss.append(loss.item())
    
     # evaluation on train & test sets
     model.eval()
     train_pred_list = []
     train_true_list = []
     for train_data, train_targets in trainloader_eval:
         train_data = train_data.cuda()
         train_pred = model(train_data)
         train_pred_list.append(train_pred.cpu().detach().numpy())
         train_true_list.append(train_targets.numpy())
     train_true = np.concatenate(train_true_list)
     train_pred = np.concatenate(train_pred_list)
     train_auc = auc_roc_score(train_true, train_pred)
     train_loss = np.mean(train_loss)
  
     test_pred_list = []
     test_true_list = [] 
     for test_data, test_targets in testloader:
         test_data = test_data.cuda()
         test_pred = model(test_data)
         test_pred_list.append(test_pred.cpu().detach().numpy())
         test_true_list.append(test_targets.numpy())
     test_true = np.concatenate(test_true_list)
     test_pred = np.concatenate(test_pred_list)
     val_auc =  auc_roc_score(test_true, test_pred) 
     model.train()
 
     # print results
     print("epoch: %s, train_loss: %.4f, train_auc: %.4f, test_auc: %.4f, lr: %.4f"%(epoch, train_loss, train_auc, val_auc, optimizer.lr ))    
     train_log.append(train_auc) 
     test_log.append(val_auc)


"""# **Visualization**
Now, let's see the change of AUC scores on training and testing set. 
"""
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,5)
x=np.arange(len(train_log))
plt.figure()
plt.plot(x, train_log, LineStyle='-', label='Train Set', linewidth=3)
plt.plot(x, test_log,  LineStyle='-', label='Test Set', linewidth=3)
plt.title('CompositionalAUCLoss (10% CIFAR10)',fontsize=25)
plt.legend(fontsize=15)
plt.ylabel('AUROC', fontsize=25)
plt.xlabel('Epoch', fontsize=25)