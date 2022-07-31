"""05_Optimizing_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb

**Author**: Zhuoning Yuan

**Introduction**
In this tutorial, you will learn how to quickly train a DenseNet121 model by optimizing **AUROC** using our novel **`AUCMLoss`** and **`PESG`** optimizer on Chest X-Ray dataset, e.g., [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). After completion of this tutorial, you should be able to use LibAUC to train your own models on your own datasets.

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

"""# **Downloading CheXpert**
*   To request dataset access, you need to apply from CheXpert website: https://stanfordmlgroup.github.io/competitions/chexpert/
*   In this tutorial, we use the smaller version of dataset with lower image resolution, i.e., *CheXpert-v1.0-small.zip*

"""

"""
# **Importing LibAUC**"""

from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert

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

"""# **Pretraining**
* Multi-label classification (5 tasks)   
* Adam + CrossEntropy Loss 
* This step is optional



"""

# dataloader
root = './CheXpert/CheXpert-v1.0-small/'
# Index: -1 denotes multi-label mode including 5 diseases
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)

# paramaters
SEED = 123
BATCH_SIZE = 32
lr = 1e-4
weight_decay = 1e-5

# model
set_all_seeds(SEED)
model = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=5)
model = model.cuda()

# define loss & optimizer
CELoss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training
best_val_auc = 0 
for epoch in range(1):
    for idx, data in enumerate(trainloader):
      train_data, train_labels = data
      train_data, train_labels  = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      loss = CELoss(y_pred, train_labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        
      # validation  
      if idx % 400 == 0:
         model.eval()
         with torch.no_grad():    
              test_pred = []
              test_true = [] 
              for jdx, data in enumerate(testloader):
                  test_data, test_labels = data
                  test_data = test_data.cuda()
                  y_pred = model(test_data)
                  test_pred.append(y_pred.cpu().detach().numpy())
                  test_true.append(test_labels.numpy())
            
              test_true = np.concatenate(test_true)
              test_pred = np.concatenate(test_pred)
              val_auc_mean =  roc_auc_score(test_true, test_pred) 
              model.train()

              if best_val_auc < val_auc_mean:
                 best_val_auc = val_auc_mean
                 torch.save(model.state_dict(), 'ce_pretrained_model.pth')

              print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc ))

"""# **Optimizing AUCM Loss**
*   Binary Classification
*   PESG + AUCM Loss
Note: you can also try other losses in this task, e.g., [CompositionalAUCLoss](https://github.com/Optimization-AI/LibAUC/blob/main/examples/09_Optimizing_CompositionalAUC_Loss_with_ResNet20_on_CIFAR10.ipynb).
"""

# parameters
class_id = 1 # 0:Cardiomegaly, 1:Edema, 2:Consolidation, 3:Atelectasis, 4:Pleural Effusion 
root = './CheXpert/CheXpert-v1.0-small/'

# You can set use_upsampling=True and pass the class name by upsampling_cols=['Cardiomegaly'] to do upsampling. This may improve the performance
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=True, use_frontal=True, image_size=224, mode='train', class_index=class_id)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=class_id)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)

# paramaters
SEED = 123
BATCH_SIZE = 32
imratio = traindSet.imratio
lr = 0.05 # using smaller learning rate is better
epoch_decay = 2e-3
weight_decay = 1e-5
margin = 1.0

# model
set_all_seeds(SEED)
model = DenseNet121(pretrained=False, last_activation=None, activations='relu', num_classes=1)
model = model.cuda()


# load pretrained model
if True:
  PATH = 'ce_pretrained_model.pth' 
  state_dict = torch.load(PATH)
  state_dict.pop('classifier.weight', None)
  state_dict.pop('classifier.bias', None) 
  model.load_state_dict(state_dict, strict=False)


# define loss & optimizer
loss_fn = AUCMLoss()
optimizer = PESG(model, 
                 loss_fn=loss_fn, 
                 lr=lr, 
                 margin=margin, 
                 epoch_decay=epoch_decay, 
                 weight_decay=weight_decay)

best_val_auc = 0
for epoch in range(2):
  if epoch > 0:
     optimizer.update_regularizer(decay_factor=10)
  for idx, data in enumerate(trainloader):
      train_data, train_labels = data
      train_data, train_labels = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      y_pred = torch.sigmoid(y_pred)
      loss = loss_fn(y_pred, train_labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # validation
      if idx % 400 == 0:
        model.eval()
        with torch.no_grad():    
              test_pred = []
              test_true = [] 
              for jdx, data in enumerate(testloader):
                  test_data, test_label = data
                  test_data = test_data.cuda()
                  y_pred = model(test_data)
                  test_pred.append(y_pred.cpu().detach().numpy())
                  test_true.append(test_label.numpy())
              
              test_true = np.concatenate(test_true)
              test_pred = np.concatenate(test_pred)
              val_auc =  roc_auc_score(test_true, test_pred) 
              model.train()

              if best_val_auc < val_auc:
                 best_val_auc = val_auc
              
        print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, lr=%.4f'%(epoch, idx, val_auc,  optimizer.lr))

print ('Best Val_AUC is %.4f'%best_val_auc)