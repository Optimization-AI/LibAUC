"""
Authors: Zhuoning Yuan
Contact: yzhuoning@gmail.com
"""

from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import DenseNet121, DenseNet169
from libauc.datasets import CheXpert

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""# **Multi-Label Training**
* Optimizing Multi-Label AUC (5 tasks)   
"""

# dataloader
root = './CheXpert/CheXpert-v1.0-small/'
# Index: -1 denotes multi-label mode including 5 diseases
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='train', class_index=-1, verbose=False)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=224, mode='valid', class_index=-1, verbose=False)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, shuffle=False)

# paramaters
SEED = 123
BATCH_SIZE = 32
 
lr = 0.1 
gamma = 500
imratio = traindSet.imratio_list 
weight_decay = 1e-5
margin = 1.0

# model
set_all_seeds(SEED)
model = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=5)
model = model.cuda()

# define loss & optimizer
Loss = AUCM_MultiLabel(imratio=imratio, num_classes=5)
optimizer = PESG(model, 
                 a=Loss.a, 
                 b=Loss.b, 
                 alpha=Loss.alpha, 
                 lr=lr, 
                 gamma=gamma, 
                 margin=margin, 
                 weight_decay=weight_decay, device='cuda')


# training
best_val_auc = 0 
for epoch in range(2):
    if epoch > 0:
        optimizer.update_regularizer(decay_factor=10)       
    for idx, data in enumerate(trainloader):
      train_data, train_labels = data
      train_data, train_labels  = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      y_pred = torch.sigmoid(y_pred)
      loss = Loss(y_pred, train_labels)
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
                  y_pred = torch.sigmoid(y_pred)
                  test_pred.append(y_pred.cpu().detach().numpy())
                  test_true.append(test_labels.numpy())
            
              test_true = np.concatenate(test_true)
              test_pred = np.concatenate(test_pred)
              val_auc_mean =  roc_auc_score(test_true, test_pred) 
              model.train()

              if best_val_auc < val_auc_mean:
                 best_val_auc = val_auc_mean
                 torch.save(model.state_dict(), 'aucm_multi_label_pretrained_model.pth')

              print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc))