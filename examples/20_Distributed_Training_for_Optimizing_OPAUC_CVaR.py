from libauc.losses import pAUC_CVaR_Loss
from libauc.optimizers import SOPA
from libauc.models import resnet18 as ResNet18
from libauc.datasets import CIFAR10
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DistributedDualSampler
from libauc.metrics import auc_roc_score

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os



local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group("nccl", device_id=local_rank)
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)


def set_all_seeds(SEED):
   # REPRODUCIBILITY
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.cuda.manual_seed(SEED)
   torch.cuda.manual_seed_all(SEED)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
set_all_seeds(2026)

class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
        self.images = images.astype(np.uint8)
        self.targets = targets
        self.mode = mode
        self.transform_train = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size), antialias=True),
                              ])
        self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size), antialias=True),
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
        return image, target, idx
    
def distributed_gather_tensor(local_tensor, device):
    """
    Gathers a 1-D float tensor from all ranks onto rank 0.
    Handles uneven shard sizes.
    Returns concatenated tensor on rank 0 (CPU), None on other ranks.
    """
    assert local_tensor.dim() == 1, "Expected 1-D tensor"
    local_tensor = local_tensor.to(device).contiguous()

    # share sizes and pad to max size
    local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device)
                 for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(s.item() for s in all_sizes)
    pad = max_size - local_tensor.shape[0]
    if pad > 0:
        local_tensor = torch.cat([local_tensor,
                                   torch.zeros(pad, dtype=local_tensor.dtype, device=device)])

    # gather
    gathered = [torch.zeros(max_size, dtype=local_tensor.dtype, device=device)
                for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    # trim and concat on rank 0 only
    if local_rank == 0:
        return torch.cat([g[:all_sizes[i].item()]
                          for i, g in enumerate(gathered)]).cpu()
    return None


def evaluate(model, loader, device):
    model.eval()
    pred_list, true_list = [], []

    with torch.no_grad():
        for data, targets, _ in loader:
            data = data.to(device)
            pred = model(data)
            pred_list.append(pred.detach())        # stay on GPU
            true_list.append(targets.to(device))   # move labels to GPU too

    local_preds  = torch.cat(pred_list).ravel()
    local_labels = torch.cat(true_list).ravel().float()

    dist.barrier()
    all_preds  = distributed_gather_tensor(local_preds,  device)
    all_labels = distributed_gather_tensor(local_labels, device)

    if local_rank == 0:
        return auc_roc_score(all_labels.numpy(), all_preds.numpy(), max_fpr=0.3)
    return None

# Hyper-Parameters
lr = 1e-3
eta = 0.1 # learning rate for control negative samples weights
beta = 0.3 # upper bound for FPR
weight_decay = 2e-4
total_epoch = 60
decay_epoch = [30, 45]
batch_size_per_gpu = 64
sampling_rate = 0.5

imratio = 0.2  ##we set the imratio as 0.2 here.


train_data, train_targets = CIFAR10(root='./data', train=True).as_array()
test_data, test_targets  = CIFAR10(root='./data', train=False).as_array()

generator = ImbalancedDataGenerator(verbose=True, random_seed=2026)
(train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
(test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5)
    

trainSet = ImageDataset(train_images, train_labels)
trainSet_eval = ImageDataset(train_images, train_labels,mode='test')
testSet = ImageDataset(test_images, test_labels, mode='test')


# dataloaders
train_sampler  = DistributedDualSampler(trainSet, batch_size_per_gpu, sampling_rate=sampling_rate)
trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=2)

train_eval_sampler = DistributedSampler(trainSet_eval, shuffle=False, drop_last=False)
trainloader_eval   = torch.utils.data.DataLoader(trainSet_eval, batch_size=batch_size_per_gpu, sampler=train_eval_sampler, num_workers=2)
test_sampler = DistributedSampler(testSet, shuffle=False, drop_last=False)
testloader   = torch.utils.data.DataLoader(testSet, batch_size=batch_size_per_gpu,sampler=test_sampler, num_workers=2)



# You can include sigmoid/l2 activations on model's outputs before computing loss
model = ResNet18(pretrained=False, last_activation=None, num_classes=1)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])



loss_fn = pAUC_CVaR_Loss(pos_len=train_sampler.pos_len, data_len=len(trainSet), beta=beta, eta=eta, device=device)
optimizer = SOPA(model.parameters(), loss_fn=loss_fn, mode='adam', lr=lr, weight_decay=weight_decay, device=device)

print ('Start Training')
print ('-'*30)

train_log = []
test_log = []
best_test = 0
for epoch in range(total_epoch):
   if hasattr(train_sampler, 'set_epoch'):
      train_sampler.set_epoch(epoch)
   if epoch in decay_epoch:
      optimizer.update_lr(decay_factor=10)

   train_loss = []
   model.train()
   for idx, (data, targets, index) in enumerate(trainloader):
      # data, targets  = data.cuda(), targets.cuda()
      data, targets, index  = data.to(device), targets.to(device), index.to(device)
      y_pred = model(data)
      y_pred = torch.sigmoid(y_pred)
      loss = loss_fn(y_pred, targets, index)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())


   # evaluation on train & test sets
   train_pauc = evaluate(model, trainloader_eval, device)
   test_pauc  = evaluate(model, testloader,       device)

   if local_rank == 0:
      train_loss_mean = np.mean(train_loss)
      if best_test < test_pauc:
            best_test = test_pauc
      print("epoch: %s, train_loss: %.4f, train_pauc: %.4f, test_pauc: %.4f, "
            "best_test_pauc: %.4f, lr: %.4f"
            % (epoch, train_loss_mean, train_pauc, test_pauc, best_test, optimizer.lr))
      train_log.append(train_pauc)
      test_log.append(test_pauc)

if local_rank == 0:
   print('train_log', train_log)
   print('test_log',  test_log)

