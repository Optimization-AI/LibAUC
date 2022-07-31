"""03_Optimizing_AUPRC_Loss_on_Imbalanced_dataset.ipynb
# **Optimizing AUPRC Loss on imbalanced dataset**

**Author**: Gang Li  
**Edited by**: Zhuoning Yuan

In this tutorial, you will learn how to quickly train a Resnet18 model by optimizing **AUPRC** loss with **SOAP** optimizer [[ref]](https://arxiv.org/abs/2104.08736) on a binary image classification task with CIFAR-10 dataset. After completion of this tutorial, you should be able to use LibAUC to train your own models on your own datasets.

**Useful Resources**:
* Website: https://libauc.org
* Github: https://github.com/Optimization-AI/LibAUC

**Reference**:  
If you find this tutorial helpful,  please acknowledge our library and cite the following paper:

@article{qi2021stochastic,
  title={Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence},
  author={Qi, Qi and Luo, Youzhi and Xu, Zhao and Ji, Shuiwang and Yang, Tianbao},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
"""

!pip install libauc==1.2.0

"""# **Importing LibAUC**

Import required packages to use
"""

from libauc.losses import APLoss
from libauc.optimizers import SOAP
from libauc.models import resnet18 as ResNet18
from libauc.datasets import CIFAR10
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler
from libauc.metrics import auc_prc_score

import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image

"""# **Configurations**
**Reproducibility**
The following function `set_all_seeds` limits the number of sources of randomness behaviors, such as model intialization, data shuffling, etcs. However, completely reproducible results are not guaranteed across PyTorch releases [[Ref]](https://pytorch.org/docs/stable/notes/randomness.html#:~:text=Completely%20reproducible%20results%20are%20not,even%20when%20using%20identical%20seeds.).
"""

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""# **Loading datasets**
In this step, , we will use the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) as benchmark dataset. Before importing data to `dataloader`, we construct imbalanced version for CIFAR10 by `ImbalanceDataGenerator`. Specifically, it first randomly splits the training data by class ID (e.g., 10 classes) into two even portions as the positive and negative classes, and then it randomly removes some samples from the positive class to make
it imbalanced. We keep the testing set untouched. We refer `imratio` to the ratio of number of positive examples to number of all examples.
"""

train_data, train_targets = CIFAR10(root='./data', train=True)
test_data, test_targets  = CIFAR10(root='./data', train=False)

imratio = 0.02
generator = ImbalancedDataGenerator(verbose=True, random_seed=2022)
(train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
(test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5)

"""Now that we defined the data input pipeline such as data augmentations. In this tutorials, we use `RandomCrop`, `RandomHorizontalFlip`."""

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
        return idx, image, target

"""We define `dataset`, `DualSampler` and `dataloader` here. By default, we use `batch_size` 64 and we oversample the minority class with `pos:neg=1:1` by setting `sampling_rate=0.5`."""

batch_size = 64
sampling_rate = 0.5

trainSet = ImageDataset(train_images, train_labels)
trainSet_eval = ImageDataset(train_images, train_labels,mode='test')
testSet = ImageDataset(test_images, test_labels, mode='test')

sampler = DualSampler(trainSet, batch_size, sampling_rate=sampling_rate)
trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, sampler=sampler, num_workers=2)
trainloader_eval = torch.utils.data.DataLoader(trainSet_eval, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=2)


# Parameters

lr = 1e-3
margin = 0.6
gamma = 0.1
weight_decay = 0
total_epoch = 60
decay_epoch = [30]
SEED = 2022

"""# **Model and Loss Setup**
"""

set_all_seeds(SEED)
model = ResNet18(pretrained=False, last_activation=None) 
model = model.cuda()

Loss = APLoss(pos_len=sampler.pos_len, margin=margin, gamma=gamma)
optimizer = SOAP(model.parameters(), lr=lr, mode='adam', weight_decay=weight_decay)

"""# **Training**"""
print ('Start Training')
print ('-'*30)
test_best = 0
train_list, test_list = [], []
for epoch in range(total_epoch):
    if epoch in decay_epoch:
        optimizer.update_lr(decay_factor=10)
    model.train() 
    for idx, (index, data, targets) in enumerate(trainloader):
        data, targets  = data.cuda(), targets.cuda() 
        y_pred = model(data)
        y_prob = torch.sigmoid(y_pred)
        loss = Loss(y_prob, targets, index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # evaluation
    model.eval()
    train_pred, train_true = [], []
    for i, data in enumerate(trainloader_eval):
        _, train_data, train_targets = data
        train_data = train_data.cuda()
        y_pred = model(train_data)
        y_prob = torch.sigmoid(y_pred)
        train_pred.append(y_prob.cpu().detach().numpy())
        train_true.append(train_targets.cpu().detach().numpy())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_ap = auc_prc_score(train_true, train_pred)
    train_list.append(train_ap)
    
    test_pred, test_true = [], [] 
    for j, data in enumerate(testloader):
        _, test_data, test_targets = data
        test_data = test_data.cuda()
        y_pred = model(test_data)
        y_prob = torch.sigmoid(y_pred)
        test_pred.append(y_prob.cpu().detach().numpy())
        test_true.append(test_targets.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    val_ap = auc_prc_score(test_true, test_pred)
    test_list.append(val_ap)
    if test_best < val_ap:
       test_best = val_ap
    
    model.train()
    print("epoch: %s, train_ap: %.4f, test_ap: %.4f, lr: %.4f, test_best: %.4f"%(epoch, train_ap, val_ap, optimizer.lr, test_best))

