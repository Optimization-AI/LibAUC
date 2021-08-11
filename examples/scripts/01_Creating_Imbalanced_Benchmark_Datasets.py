"""
Author: Zhuoning Yuan
Contact: yzhuoning@gmail.com
"""

from libauc.datasets import CIFAR10
(train_data, train_label), (test_data, test_label) = CIFAR10()

from libauc.datasets import CIFAR100
(train_data, train_label), (test_data, test_label) = CIFAR100()

from libauc.datasets import CAT_VS_DOG
(train_data, train_label), (test_data, test_label) = CAT_VS_DOG()

from libauc.datasets import STL10
(train_data, train_label), (test_data, test_label) = STL10()

from libauc.datasets import imbalance_generator 

SEED = 123
imratio = 0.1 # postive_samples/(total_samples)

from libauc.datasets import CIFAR10
(train_data, train_label), (test_data, test_label) = CIFAR10()
(train_images, train_labels) = imbalance_generator(train_data, train_label, imratio=imratio, shuffle=True, random_seed=SEED)
(test_images, test_labels) = imbalance_generator(test_data, test_label, is_balanced=True, random_seed=SEED)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

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
  

trainloader = DataLoader(ImageDataset(train_images, train_labels, mode='train'), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
testloader = DataLoader(ImageDataset(test_images, test_labels, mode='test'), batch_size=128, shuffle=False, num_workers=2,  pin_memory=True)
