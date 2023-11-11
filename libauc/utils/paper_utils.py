import torch 
import numpy as np
import datetime
import os
import sys
import time
import shutil
import math
import numpy as np
import logging
from typing import Dict, Any
from collections import Counter
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
from ..metrics import ndcg_at_k, map_at_k

_logger = logging.getLogger(__name__)


'''
Helper functions for MIDAM
'''
def collate_fn(list_items):
    r"""
        The basic collate function takes a list of (x, y, index) and collate them separately.

        Args:
            list_items (list, required): list of tuples (x, y, index)

        Example:
            >>> traindSet = TabularDataset(data, label)
            >>> trainloader =  torch.utils.data.DataLoader(dataset=traindSet, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    """
    x = []
    y = []
    index = []
    for x_, y_, index_ in list_items:
        x.append(x_)
        y.append(y_)
        index.append(index_)
    return x, y, index


def MIL_sampling(bag_X, model, instance_batch_size=4, mode='mean', tau=0.1, device=None):
    r"""
    The multiple instance sampling for the stochastic pooling operations. It uniformly randomly samples instances from each bag and take different pooling calculations for different pooling methods.

    Args:
        bag_X (array-like, required): data features for all instances from a bag with shape [number_of_instance, ...].
        model (pytorch model, required): model that generates predictions (or more generally related outputs) from instance-level.
        instance_batch_size (int, required): the maximal instance batch size for each bag, default: 4.
        mode (str, required): the stochastic pooling mode for MIL, default: mean.
        tau (float, optional): the temperature parameter for stochastic softmax (smoothed-max) pooling, default: 0.1.
        device (torch.device, optional): device for running the code. default: none (use GPU if available)

    Example:
        >>> model = FFNN_stoc_MIL(num_classes=1, dims=DIMS)
        >>> train_data_bags, train_labels, index = data
        >>> for i in range(len(train_data_bags)):
        >>>   y_pred[i] = MIL_sampling(bag_X=train_data_bags[i], model=model, instance_batch_size=instance_batch_size, mode='att')

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning 2023.
           https://arxiv.org/abs/2305.08040
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type(bag_X) == list:
      X = torch.from_numpy(np.concatenate(bag_X, axis=0)).to(device) 
    else: # it is a tensor
      if isinstance(bag_X, np.ndarray): # check if it is still numpy array
        bag_X = torch.from_numpy(bag_X)
      X = bag_X.to(device) 
    bag_size = X.shape[0]
    weights = torch.ones(bag_size)
    sample_size = min(bag_size, instance_batch_size)
    ids = torch.multinomial(weights, sample_size, replacement=False) # uniformly randomly sample instances from each bag
    X = X[ids,...]
    if mode=='mean':
      y_pred_bag = model(X.float())
      y_pred = torch.mean(y_pred_bag.view([1,-1]), dim=1, keepdim=True)
      return y_pred
    elif mode=='max':
      y_pred_bag = model(X.float())
      y_pred = torch.max(y_pred_bag.view([1,-1]), dim=1, keepdim=True).values
      return y_pred
    elif mode=='exp':
      y_pred_bag = torch.exp(model(X.float())/tau)
      y_pred = torch.mean(y_pred_bag.view([1,-1]), dim=1, keepdim=True)
      return y_pred
    elif mode=='att':
      y_pred_bag, weights_bag = model(X.float())
      sn_bag = y_pred_bag * weights_bag
      sn = torch.mean(sn_bag.view([1,-1]), dim=1, keepdim=True)
      sd = torch.mean(weights_bag.view([1,-1]), dim=1, keepdim=True)
      return sn, sd


def MIL_aggregation(bag_X, model, mode='mean', tau=0.1, device=None):
    r"""
    The bag-level prediction aggregated from all the instances from the input bag. Notice that MIL_aggregation is not recommended for back-propagation, which may exceede GPU memory limits.

    Args:
        bag_X (array-like, required): data features for all instances from a bag with shape [number_of_instance, ...].
        model (pytorch model, required): model that generates predictions (or more generally related outputs) from instance-level.
        mode (str, required): the stochastic pooling mode for MIL, default: mean.
        tau (float, optional): the temperature parameter for stochastic softmax (smoothed-max) pooling, default: 0.1.
        device (torch.device, optional): device for running the code. default: none (use GPU if available)

    Example:
        >>> model = FFNN_stoc_MIL(num_classes=1, dims=DIMS)
        >>> train_data_bags, train_labels, index = data
        >>> for i in range(len(train_data_bags)):
        >>>   y_pred[i] = MIL_aggregation(bag_X=train_data_bags[i], model=model, mode='att')

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning 2023.
           https://arxiv.org/abs/2305.08040
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type(bag_X) == list:
      X = torch.from_numpy(np.concatenate(bag_X, axis=0)).to(device) 
    else: # it is a tensor
      if isinstance(bag_X, np.ndarray):
        bag_X = torch.from_numpy(bag_X)
      X = bag_X.to(device) 
    y_pred_bag = model(X.float())
    if mode=='max':
      y_pred = torch.max(y_pred_bag.view([1,-1]), dim=1, keepdim=True).values 
    elif mode=='mean':
      y_pred = torch.mean(y_pred_bag.view([1,-1]), dim=1, keepdim=True)
    elif mode=='softmax':
      y_pred = tau*torch.log(torch.mean(torch.exp(y_pred_bag.view([1,-1])/tau), dim=1, keepdim=True))
    elif mode=='att':
      w_pred_bag = y_pred_bag[1] # don't switch order of these two lines
      y_pred_bag = y_pred_bag[0]
      y_pred = torch.sum(y_pred_bag.view([1,-1]) * torch.nn.functional.normalize(w_pred_bag.view([1,-1]),p=1.0,dim=-1), dim=1, keepdim=True)
    return y_pred


def MIL_evaluate_auc(dataloader, model, mode='max', tau=0.1):
    r"""
    The high-level wrapper for AUC evaluation under Multiple Instance Learning setting.

    Args:
        dataloader (torch.utils.data.dataloader, required): dataloader for loading data.
        model (pytorch model, required): model that generates predictions (or more generally related outputs) from instance-level.
        mode (str, required): the stochastic pooling mode for MIL, default: mean.
        tau (float, optional): the temperature parameter for stochastic softmax (smoothed-max) pooling, default: 0.1.

    Example:
        >>> traindSet = TabularDataset(data, label)
        >>> trainloader =  torch.utils.data.DataLoader(dataset=traindSet, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        >>> model = FFNN_stoc_MIL(num_classes=1, dims=DIMS)
        >>> tr_auc = evaluate_auc(trainloader, model, mode='att') 

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
           https://arxiv.org/abs/2305.08040
    """
    test_pred = []
    test_true = []
    for jdx, data in enumerate(dataloader):
      test_data_bags, test_labels, ids = data
      y_pred = []
      for i in range(len(ids)):
        tmp_pred = MIL_aggregation(test_data_bags[i],model,mode=mode,tau=tau)
        y_pred.append(tmp_pred)
      y_pred = torch.cat(y_pred, dim=0)
      test_pred.append(y_pred.cpu().detach().numpy())
      test_true.append(test_labels)
    test_true = np.concatenate(test_true, axis=0)
    test_pred = np.concatenate(test_pred, axis=0)
    single_te_auc =  roc_auc_score(test_true, test_pred) 
    return single_te_auc


'''
Helper functions for NDCG
'''

def batch_to_gpu(batch, device='cuda'):
    for c in batch:
        if type(batch[c]) is torch.Tensor:
            batch[c] = batch[c].to(device)
    return batch

def adjust_lr(learning_rate, lr_schedule, optimizer, epoch):
    lr = learning_rate
    for milestone in eval(lr_schedule):
        lr *= 0.25 if epoch >= milestone else 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate_method(predictions, ratings, topk, metrics):
    """
    :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
    :param ratings: (# of users, # of pos items)
    :param topk: top-K value list
    :param metrics: metric string list
    :return: a result dict, the keys are metric@topk
    """
    evaluations = dict()
    for k in topk:
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'NDCG':
                evaluations[key] = ndcg_at_k(ratings, predictions, k)
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    return evaluations


def evaluate(model, data_set, topks, metrics):
    """
    The returned prediction is a 2D-array, each row corresponds to all the candidates,
    and the ground-truth item poses the first.
    Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
             predictions like: [[1,3,4], [2,5,6]]
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = list()
    ratings = list()
    for idx in trange(0, len(data_set), EVAL_BATCH_SIZE):
        batch = data_set.get_batch(idx, EVAL_BATCH_SIZE)
        prediction = model(batch_to_gpu(batch, DEVICE))['prediction']
        predictions.extend(prediction.cpu().data.numpy())
        ratings.extend(batch['rating'].cpu().data.numpy())

    predictions = np.array(predictions)                                 # [# of users, # of items]
    ratings = np.array(ratings)[:, :NUM_POS]                            # [# of users, # of pos items]

    return evaluate_method(predictions, ratings, topks, metrics)

def format_metric(result_dict):
    assert type(result_dict) == dict
    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys()])
    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = '{}@{}'.format(metric, topk)
            m = result_dict[name]
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('{}:{:<.4f}'.format(name, m))
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


'''
Helper functions for iSogCLR
'''
class Scheduler:
    """ 
    Parameter Scheduler Base Class.
    
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
        
        - At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
        - At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.

    Reference:
        - https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
        - https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs


class CosineLRScheduler(Scheduler):
    """
        Cosine decay with restarts. This is described in the paper https://arxiv.org/abs/1608.03983. Inspiration from
        https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=True,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))


