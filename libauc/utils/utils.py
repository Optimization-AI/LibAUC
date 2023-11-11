import torch 
import numpy as np
import datetime
import os
import sys
import time
import random
import shutil
import numpy as np
from collections import Counter
from tqdm import tqdm, trange

def set_all_seeds(SEED):
    # for reproducibility 
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_tensor_shape(tensor, shape):
    # check tensor shape 
    if not torch.is_tensor(tensor):
        raise ValueError('Input is not a valid torch tensor!')
    if not isinstance(shape, (tuple, list, int)):
        raise ValueError("Shape must be a tuple, an integer or a list!")
    if isinstance(shape, int):
        shape = torch.Size([shape])
    tensor_shape = tensor.shape
    if len(tensor_shape) != len(shape):
        tensor = tensor.reshape(shape)
    return tensor

def check_array_type(array):
    # convert to array type 
    if not isinstance(array, (np.ndarray, np.generic)):
        array = np.array(array)
    return array

def check_array_shape(array, shape):
    # check array shape
    array = check_array_type(array)  
    if array.size == 0:
        raise ValueError("Array is empty.")
    if array.shape != shape and len(array.shape) != 1:
        try:
            array = array.reshape(shape)
        except ValueError as e:
            raise ValueError(f"Could not reshape array of shape {array.shape} to {shape}.") from e
    return array

def check_class_labels(labels):
    # check if labels are valid
    labels = check_array_type(labels)  
    unique_values = np.unique(labels)
    num_classes = len(unique_values)
    if not np.all(unique_values == np.arange(num_classes)):
        raise ValueError("Labels should be integer values starting from 0.")

def select_mean(array, threshold=0):
    # select elements for average based on threshold
    array = check_array_type(array)  
    select_array = array[array >= threshold] 
    if len(select_array) != 0: 
        return np.mean(select_array)
    else:
        return None 

def check_imbalance_ratio(labels):
    # check data imbalance ratio for the labels
    labels = check_array_type(labels)
    check_class_labels(labels)

    # Flatten the labels array if it's 2D (n, 1)
    if len(labels.shape) > 1 and labels.shape[1] == 1:
        labels = labels.flatten()

    num_samples = len(labels)
    class_counts = Counter(labels)

    for class_label, count in class_counts.items():
        class_ratio = count / num_samples
        print (f'#SAMPLES: {num_samples}, CLASS {class_label:.1f} COUNT: {count}, CLASS RATIO: {class_ratio:.4f}')

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

class ImbalancedDataGenerator(object):
    def __init__(self, imratio=None, shuffle=True, random_seed=0, verbose=False):
        self.imratio = imratio
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.verbose = verbose

    @staticmethod
    def _get_split_index(num_classes):
        split_index = num_classes // 2 - 1
        if split_index < 0:
            raise NotImplementedError
        return split_index

    @staticmethod
    def _get_class_num(targets):
        return np.unique(targets).size

    def transform(self, data, targets, imratio=None):
        data = check_array_type(data)
        targets = check_array_type(targets)
        targets = np.maximum(targets, 0)
        if imratio is not None:
            self.imratio = imratio
        if self.imratio is None:
            raise ValueError("imratio is None.")
        assert self.imratio > 0 and self.imratio <= 0.5, 'imratio needs to be in (0, 0.5)!'
       
        if self.shuffle:
            np.random.seed(self.random_seed)
            idx = np.random.permutation(len(targets))
            data, targets = data[idx], targets[idx]

        num_classes = self._get_class_num(targets)
        split_index = self._get_split_index(num_classes)
        targets = np.where(targets <= split_index, 0, 1)

        if self.imratio < 0.5:
            neg_ids = np.where(targets == 0)[0]
            pos_ids = np.where(targets == 1)[0]
            pos_ids = pos_ids[:int((self.imratio / (1 - self.imratio)) * len(neg_ids))]
            idx = np.concatenate([neg_ids, pos_ids])
            data, targets = data[idx], targets[idx]
            targets = targets.reshape(-1, 1).astype(np.float32)

        if self.shuffle:
            np.random.seed(self.random_seed)
            idx = np.random.permutation(len(targets))
            data, targets = data[idx], targets[idx]
            
        if self.verbose:
            check_imbalance_ratio(targets)

        return data, targets
