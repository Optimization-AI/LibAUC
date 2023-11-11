from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CustomDataset(Dataset):
    r"""
        Custom Dataset Template for loading numpy array-like data & targets into the PyTorch dataloader.

        Args:
            data (numpy.ndarray): numpy array-like data 
            targets (numpy.ndarray): numpy array-like targets 
            transform (callable, optional): optional transform to be applied on the training/testing data (default: ``None``)
            return_index (bool, optional): returns a tuple containing data, target, and index if return_index is set to True. Otherwise, it returns a tuple containing data and target only (default: ``False``)
    """
    def __init__(self, data, targets, transform=None, return_index=False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.return_index = return_index
        assert len(data) == len(targets), 'The length of data and targets must match!'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try: 
            sample_id, task_id = index
        except:
            sample_id, task_id = index, None
        data = self.data[sample_id]
        target = self.targets[sample_id]
        if self.transform:
            data = self.transform(data)
        if self.return_index:
            if task_id != None:
                index = (sample_id, task_id)
            return data, target, index
        return data, target
