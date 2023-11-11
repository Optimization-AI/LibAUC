import numpy as np
import random
import torch
import torchvision
from torch.utils.data.sampler import Sampler

__all__ = [
        'ControlledDataSampler', 
        'DualSampler',
        'TriSampler']

class ControlledDataSampler(Sampler):
    r""" Base class for Controlled Data Sampler."""
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 labels=None, 
                 shuffle=True, 
                 num_pos=None, 
                 num_sampled_tasks=None, 
                 sampling_rate=0.5,
                 random_seed=2023): 

        assert batch_size is not None, 'batch_size can not be None!'
        assert (num_pos is None) or (sampling_rate is None), 'only one of {pos_num} and {sampling_rate} is needed!'
        
        if sampling_rate:
           assert sampling_rate>0.0 and sampling_rate<1.0, 'sampling rate is not a valid number!'
        if labels is None:
           labels = self._get_labels(dataset)
        self.labels = self._check_labels(labels) # return: (N, ) or (N, T)
        
        self.random_seed = random_seed
        self.shuffle = shuffle
       
        self.num_samples = int(len(labels))
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
       
        np.random.seed(self.random_seed)
        
        total_tasks = 0
        if num_sampled_tasks is None:
           total_tasks = self._get_num_tasks(self.labels) 
        self.total_tasks = total_tasks
        self.num_sampled_tasks = num_sampled_tasks
        self.pos_indices, self.neg_indices = self._get_sample_class_indices(self.labels) # task_id: 0, 1, 2, 3, ...
        self.class_counts = self._get_sample_class_counts(self.labels)                   # pos_len & neg_len 
 
        if self.sampling_rate:
           self.num_pos = int(self.sampling_rate*batch_size) 
           if self.num_pos == 0:
              self.num_pos = 1
           self.num_neg = batch_size - self.num_pos
        elif num_pos:
            self.num_pos = num_pos
            self.num_neg = batch_size - num_pos
        else:
            NotImplementedError

        self.num_batches = len(labels)//batch_size 
        self.sampled = []
        
    def _check_array(self, data, squeeze=True):
        if not isinstance(data, (np.ndarray, np.generic)):
           data = np.array(data)
        if squeeze:
           data = np.squeeze(data)
        return data

    def _get_labels(self, dataset):
        r"""Extract labels from given any dataset object."""
        if isinstance(dataset, torch.utils.data.Dataset):
           return np.array(dataset.targets)       
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
           return np.array(dataset.targets)    
        else:
           raise NotImplementedError # TODO: support more Dataset types
    
    def _check_labels(self, labels): 
        r"""Validate labels on three cases: nan, negative, one-hot."""
        if np.isnan(labels).sum()>0:
           raise ValueError('labels contain NaN value!') 
        labels = self._check_array(labels, squeeze=True)
        if (labels<0).sum() > 0 :
           raise ValueError('labels contain negative value!') 
        if len(labels.shape) == 1:
           num_classes = np.unique(labels).size
           assert num_classes > 1, 'labels must have >= 2 classes!'
           if num_classes > 2: # format multi-class to multi-label
              num_samples = len(labels)
              new_labels = np.eye(num_classes)[labels]  
              return new_labels
        return labels

    def _get_num_tasks(self, labels):
        r"""Compute number of unique labels for binary and multi-label datasets."""
        if len(labels.shape) == 1:
            return len(np.unique(labels)) 
        else: 
            return labels.shape[-1] 
            
    def _get_unique_labels(self, labels):
        r"""Extract unique labels for binary and multi-label (task) datasets."""
        unique_labels = np.unique(labels) if len(labels.shape)==1 else np.arange(labels.squeeze().shape[-1])
        assert len(unique_labels) > 1, 'labels must have >=2 classes!'
        return unique_labels

    def _get_sample_class_counts(self, labels):
       r"""Compute number of postives and negatives per label (task). """
       num_sampled_task = self._get_num_tasks(labels)
       dict = {}
       if num_sampled_task == 2: 
           task_id = 0   # binary data, i.e. num_sampled_task == 1
           dict[task_id] = (np.count_nonzero(labels == 1), np.count_nonzero(labels == 0) )
       else:
           task_ids = np.arange(num_sampled_task)              
           for task_id in task_ids:
               dict[task_id] = (np.count_nonzero(labels[:, task_id] > 0), np.count_nonzero(labels[:, task_id] == 0) )
       return dict

    def _get_sample_class_indices(self, labels, num_sampled_task=None):
        r"""Extract sample indices for postives and negatives per label (task)."""
        if not num_sampled_task:
           num_sampled_task = self._get_num_tasks(labels)
        num_sampled_task = num_sampled_task - 1 if num_sampled_task == 2 else num_sampled_task    
        pos_indices, neg_indices = {}, {}
        for task_id in range(num_sampled_task):
             label_t = labels[:, task_id] if num_sampled_task > 2 else labels
             pos_idx = np.flatnonzero(label_t>0)
             neg_idx = np.flatnonzero(label_t==0)
             if self.shuffle:
                np.random.shuffle(pos_idx), np.random.shuffle(neg_idx)
             pos_indices[task_id] = pos_idx
             neg_indices[task_id] = neg_idx
        return pos_indices, neg_indices
    
    def __iter__(self):
        r"""Naive implementation for Controlled Data Sampler."""
        pos_id = 0
        neg_id = 0
        if self.shuffle:
           np.random.shuffle(self.pos_pool)
           np.random.shuffle(self.neg_pool)
        for i in range(self.num_batches):
            for j in range(self.num_pos):
                self.sampled.append(self.pos_indices[pos_id % self.pos_len])
                pos_id += 1
            for j in range(self.num_neg):
                self.sampled.append(self.neg_indices[neg_id % self.neg_len])
                neg_id += 1    
        return iter(self.sampled)

    def __len__ (self):
        return len(self.sampled)
    
    
class DualSampler(ControlledDataSampler):
    r"""
        Dual Sampler aims to customize the number of positives and negatives in mini-batch data for binary classification tasks. 
        For more details, please refer to LibAUC paper[1]_.

        Args:
            dataset (torch.utils.data.Dataset): pytorch dataset object for training or evaluation.
            batch_size (int): number of samples per mini-batch.
            sampling_rate (float): the ratio of number of positive samples to total number of samples per task in a mini-batch (default: ``0.5``).
            num_pos (int, optional): number of positive samples in a batch (default: ``None``).
            labels (list or array, optional): A list or array of labels for the dataset (default: ``None``).
            shuffle (bool): Whether to shuffle the data before sampling mini-batch data (default: ``True``).
            num_sampled_tasks (int): number of sampled tasks from original dataset. If None is given, then all labels (tasks) are used for training (default: ``None``).
            random_seed (int): random seed for reproducibility (default: ``2023``).

        Example:
            >>> sampler = libauc.sampler.DualSampler(trainSet, batch_size=32, sampling_rate=0.5)
            >>> trainloader = torch.utils.data.DataLoader(trainSet, batch_size=32, sampler=sampler, shuffle=False)
            >>> data, targets, index = next(iter(trainloader))


        .. note::

            Practical Tips: 

            - In `DualSampler`, ``num_pos`` is equivalent to ``int(sampling_rate * batch_size)``. You can choose to use ``num_pos`` if you want to define the exact number of positive samples per mini-batch. Otherwise, ``sampling_rate`` will be the required parameter by default.
            - For ``sampling_rate``, we recommended to set a value slightly higher than the proportion of positive samples in your training dataset. For instance, if the ratio of positive sample in your dataset is 0.01, you might consider setting ``sampling_rate`` to 0.05, 0.1, or 0.2.

        Reference:
            .. [1] Zhuoning Yuan, Dixian Zhu, Zi-Hao Qiu, Gang Li, Xuanhui Wang, Tianbao Yang.
               "LibAUC: A Deep Learning Library for X-Risk Optimization."
               29th SIGKDD Conference on Knowledge Discovery and Data Mining.
               https://arxiv.org/abs/2306.03065

    """
    def __init__(self, 
                  dataset, 
                  batch_size, 
                  labels=None, 
                  shuffle=True, 
                  num_pos=None,  
                  num_sampled_tasks=None, 
                  sampling_rate=0.5,
                  random_seed=2023):
        super().__init__(dataset, batch_size, labels, shuffle, num_pos, num_sampled_tasks, sampling_rate, random_seed)
        
        assert self.total_tasks > 1, 'Labels are not binary, e.g., [0, 1]!'
        self.pos_len = self.class_counts[0][0]
        self.neg_len = self.class_counts[0][1]
        self.pos_indices, self.neg_indices = self.pos_indices[0], self.neg_indices[0]
        
        np.random.seed(self.random_seed)
        if shuffle:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

        self.num_batches = max(self.pos_len//self.num_pos, self.neg_len//self.num_neg)
        self.pos_ptr, self.neg_ptr = 0, 0
        self.sampled = np.zeros(self.num_batches*self.batch_size, dtype=np.int64)
        
    def __iter__(self):
        self.sampled = np.zeros(self.num_batches*self.batch_size, dtype=np.int64)
        for i in range(self.num_batches):
            start_index = i*self.batch_size
            if self.pos_ptr+self.num_pos > self.pos_len:
                # TODO: edge case - dataset has very limited positive samples e.g., < half of batch size
                temp = self.pos_indices[self.pos_ptr:]
                np.random.shuffle(self.pos_indices)
                self.pos_ptr = (self.pos_ptr+self.num_pos)%self.pos_len
                self.sampled[start_index:start_index+self.num_pos] = np.concatenate((temp, self.pos_indices[:self.pos_ptr]))
            else:
                self.sampled[start_index:start_index+self.num_pos]= self.pos_indices[self.pos_ptr:self.pos_ptr+self.num_pos]
                self.pos_ptr += self.num_pos
            start_index += self.num_pos
            if self.neg_ptr+self.num_neg > self.neg_len:
                temp = self.neg_indices[self.neg_ptr:]
                np.random.shuffle(self.neg_indices)
                self.neg_ptr = (self.neg_ptr+self.num_neg)%self.neg_len
                self.sampled[start_index:start_index+self.num_neg] = np.concatenate((temp, self.neg_indices[:self.neg_ptr]))
            else:
                self.sampled[start_index:start_index+self.num_neg] = self.neg_indices[self.neg_ptr:self.neg_ptr+self.num_neg]
                self.neg_ptr += self.num_neg    

        return iter(self.sampled)

    def __len__ (self):
        return len(self.sampled)
    
    
class TriSampler(ControlledDataSampler):
    r"""
        TriSampler aims to customize the number of positives and negatives in mini-batch data for multi-label classification or ranking tasks. For more details, 
        please refer to LibAUC paper[1]_.

        Args:
            dataset (torch.utils.data.Dataset): pytorch dataset object for training or evaluation.
            batch_size_per_task (int): number of samples per mini-batch for each task.
            num_sampled_tasks (int): number of sampled tasks from original dataset. If None is given, then all labels (tasks) are used for training (default: ``None``).
            sampling_rate (float): the ratio of number of positive samples to total number of samples per task in a mini-batch (default: ``0.5``).
            num_pos (int, optional): number of positive samples in a batch (default: ``None``).
            mode (str, optional): sampling mode for classification or ranking tasks (default: ``'classification'``).
            labels (list or array, optional): A list or array of labels for the dataset (default: ``None``).
            shuffle (bool): Whether to shuffle the data before sampling mini-batch data (default: ``True``).
            random_seed (int): random seed for reproducibility (default: ``2023``).

        Example:
            >>> sampler = libauc.sampler.TriSampler(trainSet, batch_size_per_task=32, num_sampled_tasks=10, sampling_rate=0.5)
            >>> trainloader = torch.utils.data.DataLoader(trainSet, batch_size=320, sampler=sampler, shuffle=False)
            >>> data, targets, index = next(iter(trainloader))
            >>> data_id, task_id = index

        .. note::
          `TriSampler` will return an index tuple of ``(sample_id, task_id)`` and it requires a slight change in your dataloader for the training. See the example below:

            .. code-block:: python

                class SampleDataset(torch.utils.data.Dataset):
                    def __init__(self, inputs, targets):
                        self.inputs = inputs
                        self.targets = targets

                    def __len__(self):
                        return len(self.inputs)

                    def __getitem__(self, index):
                        index, task_id = index
                        data = self.inputs[index]
                        target = self.targets[index]
                        return data, target, (index, task_id)

        .. note::
            
            Practical Tips: 

            - In `classification` mode, ``batch_size_per_task * num_sampled_tasks`` is the total ``batch_size``. If ``num_sampled_tasks`` is not specified, all labels will be used. 
            - In `ranking` mode, ``batch_size_per_task`` is the number of queries, ``num_pos`` is the number of positive items per user, and ``num_sampled_tasks`` is the number of users sampled from the dataset for mini-batch. For example, ``batch_size_per_task=310``, ``num_pos=10``, ``num_sampled_tasks=256`` implies that we sample 256 users per mini-batch data where each user has 10 positive items and 300 negative items. 
    """
    def __init__(self, 
                  dataset, 
                  batch_size_per_task,  
                  num_sampled_tasks=None,  
                  sampling_rate=0.5,
                  mode='classification',   
                  labels=None, 
                  shuffle=True, 
                  num_pos=None,                 
                  random_seed=2023):         
        super().__init__(dataset, batch_size_per_task, labels, shuffle, num_pos, None, sampling_rate, random_seed)

        self.mode = mode
        assert self.mode in ['classification', 'ranking'], 'TriSampler mode should be classification or ranking'

        assert self.total_tasks >=3, "TriSampler requires number of tasks >= 3 for given dataset!"
        self.batch_size_per_task = batch_size_per_task

        self.num_sampled_tasks = num_sampled_tasks if num_sampled_tasks != None else self.total_tasks # if num_sampled_tasks is not specified, it uses all tasks by default. 
        self.batch_size = self.batch_size_per_task*self.num_sampled_tasks
        
        if self.mode == 'classification':
            self.num_batches = self.labels.shape[0]//(self.batch_size_per_task*self.num_sampled_tasks)
        else:
            self.num_batches = self.labels.shape[1]// self.num_sampled_tasks

        self.num_pos = int(self.batch_size_per_task*self.sampling_rate) if not num_pos else num_pos  
        if self.num_pos < 1:
           print('batch_size_per_task x sampling_rate < 1 !')
           self.num_pos = 1
        self.num_neg = self.batch_size_per_task - self.num_pos 
        self.pos_len = [self.class_counts[task_id][0] for task_id in range(self.total_tasks)]
        self.neg_len = [self.class_counts[task_id][1] for task_id in range(self.total_tasks)]
        self.tasks_ids = np.arange(self.total_tasks)
            
        np.random.seed(self.random_seed)
        if shuffle:
            np.random.shuffle(self.tasks_ids)
            for task_id in range(len(self.pos_indices)):
                np.random.shuffle(self.pos_indices[task_id])
                np.random.shuffle(self.neg_indices[task_id])

        self.pos_ptr, self.neg_ptr, self.task_ptr = np.zeros(self.total_tasks, dtype=np.int32), np.zeros(self.total_tasks, dtype=np.int32), 0

        if self.mode == 'classification':
            self.sampled = np.zeros(self.num_batches*self.batch_size, dtype=np.int64)
            self.sampled_tasks = np.zeros(self.num_batches*self.batch_size, dtype=np.int32)
        else:
            self.sampled = np.zeros((self.num_batches*self.num_sampled_tasks, self.num_pos+self.num_neg), dtype=np.int32)
            self.sampled_tasks = np.zeros(self.num_batches*self.num_sampled_tasks, dtype=np.int32)
          
    def __iter__(self):
        sid = 0 
        for batch_id in range(self.num_batches):
            start_index = batch_id*self.batch_size 
            if self.num_sampled_tasks < self.total_tasks:
                task_ids = []
                if self.task_ptr + self.num_sampled_tasks >= self.total_tasks:
                    temp = self.tasks_ids[self.task_ptr:]
                    self.task_ptr = (self.task_ptr + self.num_sampled_tasks) % len(self.tasks_ids)
                    np.random.shuffle(self.tasks_ids)  
                    task_ids = np.concatenate((temp, self.tasks_ids[:self.task_ptr]))    
                else:
                    task_ids = self.tasks_ids[self.task_ptr:self.task_ptr+self.num_sampled_tasks]
                    self.task_ptr += self.num_sampled_tasks                    
            else:
                self.num_sampled_tasks = self.total_tasks
                task_ids = self.tasks_ids
                np.random.shuffle(self.tasks_ids)

            for idx, task_id in enumerate(task_ids):
                if self.pos_ptr[task_id]+self.num_pos >= self.pos_len[task_id]:
                    temp = self.pos_indices[task_id][self.pos_ptr[task_id]:]
                    np.random.shuffle(self.pos_indices[task_id])
                    self.pos_ptr[task_id] = (self.pos_ptr[task_id]+self.num_pos)%self.pos_len[task_id]
                    pos_list = np.concatenate((temp, self.pos_indices[task_id][:self.pos_ptr[task_id]]))
                else:
                    pos_list = self.pos_indices[task_id][self.pos_ptr[task_id]:self.pos_ptr[task_id]+self.num_pos]
                    self.pos_ptr[task_id] += self.num_pos

                if self.mode == 'classification':
                    self.sampled[start_index:start_index+self.num_pos] = pos_list
                    self.sampled_tasks[start_index:start_index+self.num_pos] = task_id
                    start_index += self.num_pos
                else:
                    self.sampled[sid, :self.num_pos] = pos_list
                
                if self.neg_ptr[task_id]+self.num_neg >= self.neg_len[task_id]:
                    temp = self.neg_indices[task_id][self.neg_ptr[task_id]:]
                    np.random.shuffle(self.neg_indices[task_id])
                    self.neg_ptr[task_id] = (self.neg_ptr[task_id]+self.num_neg)%self.neg_len[task_id]
                    neg_list = np.concatenate((temp, self.neg_indices[task_id][:self.neg_ptr[task_id]]))
                else:
                    neg_list = self.neg_indices[task_id][self.neg_ptr[task_id]:self.neg_ptr[task_id]+self.num_neg]
                    self.neg_ptr[task_id] += self.num_neg

                if self.mode == 'classification':
                    self.sampled[start_index:start_index+self.num_neg] = neg_list
                    self.sampled_tasks[start_index:start_index+self.num_neg] = task_id 
                    start_index += self.num_neg
                else:
                    self.sampled[sid, self.num_pos:] = neg_list
                
                if self.mode == 'ranking':
                    self.sampled_tasks[sid] = task_id
                    sid += 1
                
        return iter(zip(self.sampled, self.sampled_tasks)) # potential issue: task_id can be zero!

    def __len__ (self):
        return len(self.sampled)
    



    
