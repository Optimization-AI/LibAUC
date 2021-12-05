"""
Author: Zhuoning Yuan
Contact: yzhuoning@gmail.com
"""

"""
# **Importing LibAUC**"""

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import DenseNet121, DenseNet169
from libauc.datasets import Melanoma
from libauc.utils import auroc

import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

"""# **Reproducibility**"""

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""# **Data Augmentation**"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensor

def augmentations(image_size=256, is_test=True):
    # https://www.kaggle.com/vishnus/a-simple-pytorch-starter-code-single-fold-93
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    train_tfms = A.Compose([
        A.Cutout(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.5), 
        ToTensor(normalize=imagenet_stats)
        ])
    
    test_tfms = A.Compose([ToTensor(normalize=imagenet_stats)])
    if is_test:
        return test_tfms
    else:
        return train_tfms

"""# **Optimizing AUCM Loss**
* Installation of `albumentations` is required!
"""

# dataset
trainSet = Melanoma(root='./melanoma/', is_test=False, test_size=0.2, transforms=augmentations)
testSet = Melanoma(root='./melanoma/', is_test=True, test_size=0.2, transforms=augmentations)

# paramaters
SEED = 123
BATCH_SIZE = 64
lr = 0.1 
gamma = 500
imratio = trainSet.imratio
weight_decay = 1e-5
margin = 1.0

# model
set_all_seeds(SEED)
model = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=1)
model = model.cuda()

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

# load your own pretrained model here
#  PATH = 'ce_pretrained_model.pth' 
#  state_dict = torch.load(PATH)
#  state_dict.pop('classifier.weight', None)
#  state_dict.pop('classifier.bias', None) 
#  model.load_state_dict(state_dict, strict=False)

# define loss & optimizer
Loss = AUCMLoss(imratio=imratio)
optimizer = PESG(model, 
                 a=Loss.a, 
                 b=Loss.b, 
                 alpha=Loss.alpha, 
                 lr=lr, 
                 gamma=gamma, 
                 margin=margin, 
                 weight_decay=weight_decay)

total_epochs = 16
best_val_auc = 0
for epoch in range(total_epochs):

  # reset stages 
  if epoch== int(total_epochs*0.5) or epoch== int(total_epochs*0.75):
     optimizer.update_regularizer(decay_factor=10) 

  # training 
  for idx, data in enumerate(trainloader):
      train_data, train_labels = data
      train_data, train_labels = train_data.cuda(), train_labels.cuda()
      y_pred = model(train_data)
      y_pred = torch.sigmoid(y_pred)
      loss = Loss(y_pred, train_labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # validation
  model.eval()
  with torch.no_grad():    
       test_pred = []
       test_true = [] 
       for jdx, data in enumerate(testloader):
           test_data, test_label = data
           test_data = test_data.cuda()
           y_pred = model(test_data)
           y_pred = torch.sigmoid(y_pred)
           test_pred.append(y_pred.cpu().detach().numpy())
           test_true.append(test_label.numpy())
              
       test_true = np.concatenate(test_true)
       test_pred = np.concatenate(test_pred)
       val_auc =  auroc(test_true, test_pred) 
       model.train()

       if best_val_auc < val_auc:
          best_val_auc = val_auc
              
       print ('Epoch=%s, Loss=%.4f, Val_AUC=%.4f, lr=%.4f'%(epoch, loss, val_auc, optimizer.lr))

print ('Best Val_AUC is %.4f'%best_val_auc)