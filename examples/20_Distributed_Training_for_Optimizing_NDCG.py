import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader

from libauc.datasets import MoiveLens
from libauc.sampler import DistributedTriSampler  
from libauc.losses import NDCGLoss  
from libauc.optimizers import SONG
from libauc.models import NeuMF
from libauc.utils.utils import get_time
from libauc.utils.paper_utils import batch_to_gpu, adjust_lr, format_metric 
from libauc.metrics.metrics_k import ndcg_at_k


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
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
seed=2026
set_all_seeds(seed)


DATA_PATH = 'ml-20m'                 # path for the dataset file
BATCH_SIZE_PER_GPU = 256                     # training batch size
EVAL_BATCH_SIZE = 2048               # evaluation batch size
EPOCH = 120                          # total training epochs
NUM_WORKERS = 4                     # number of workers in the dataloader
LR_SCHEDULE = '[80]'                 # the lr will multiple 0.25 at 80 epochs
TOPKS = eval('[5,10,20,50]')         # k values for model evaluation (seperated by comma)
METRICS = eval('["NDCG"]')           # the list of evaluation metrics (seperated by comma)
MAIN_METRIC = "NDCG@5"               # main metric when evaluation


trainSet = MoiveLens(root=DATA_PATH, phase='train', random_seed=seed)
valSet = MoiveLens(root=DATA_PATH, phase='dev', random_seed=seed)
testSet = MoiveLens(root=DATA_PATH, phase='test', random_seed=seed)

# training function
def train(model, train_set, train_sampler, eval_set, optimizer, device):
    main_metric_results, dev_results = list(), list()
    try:
        for epoch in range(EPOCH):
            if hasattr(train_sampler, 'set_epoch'):
                    train_sampler.set_epoch(epoch)

            time_s = time.time()
            adjust_lr(LR, LR_SCHEDULE, optimizer, epoch + 1)
            model.train()
            loss_lst = list()

            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, sampler=train_sampler,
                                      num_workers=NUM_WORKERS, collate_fn=train_set.collate_batch, pin_memory=True)

            for batch in tqdm(train_loader, leave=False, desc='Epoch {:<3}'.format(epoch + 1), ncols=100, mininterval=1):
                batch = batch_to_gpu(batch, device)
                optimizer.zero_grad()
                out_dict = model(batch)
                loss = criterion(out_dict['prediction'], batch)
                loss.backward()
                optimizer.step()
                loss_lst.append(loss.detach().cpu().data.numpy())

            loss = np.mean(loss_lst).item()

            training_time = time.time() - time_s

            # Record dev results
            dev_result = evaluate(model, eval_set, TOPKS[:1], METRICS, device)
            dev_results.append(dev_result)
            main_metric_results.append(dev_result[MAIN_METRIC])
            logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    dev=({})'.format(
                epoch + 1, loss, training_time, format_metric(dev_result))

            # Save model and early stop
            if max(main_metric_results) == main_metric_results[-1]:
                model.module.save_model(os.path.join(RES_PATH, 'pretrained_model.pkl'))
                logging_str += ' *'
            #logging.info(logging_str)
            print(logging_str)

    except KeyboardInterrupt:
        #logging.info("Early stop manually")
        print ("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            #logging.info(os.linesep + '-' * 45 + ' END: ' + get_time() + ' ' + '-' * 45)
            print(os.linesep + '-' * 45 + ' END: ' + get_time() + ' ' + '-' * 45)
            exit(1)
    print("main_metric_results:", main_metric_results)
    print("dev_results:", dev_results)

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


def evaluate(model, data_set, topks, metrics, device):
    """
    The returned prediction is a 2D-array, each row corresponds to all the candidates,
    and the ground-truth item poses the first.
    Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
             predictions like: [[1,3,4], [2,5,6]]
    """

    model.eval()
    predictions = list()
    ratings = list()
    for idx in trange(0, len(data_set), EVAL_BATCH_SIZE):
        batch = data_set.get_batch(idx, EVAL_BATCH_SIZE)
        prediction = model(batch_to_gpu(batch, device))['prediction']
        predictions.extend(prediction.cpu().data.numpy())
        ratings.extend(batch['rating'].cpu().data.numpy())

    predictions = np.array(predictions)                                 # [# of users, # of items]
    ratings = np.array(ratings)[:, :NUM_POS]                            # [# of users, # of pos items]

    return evaluate_method(predictions, ratings, topks, metrics)


LOSS = 'SONG'
LR = 0.001                # learning rate of model parameters, \eta in the paper
NUM_POS = 10              # number of positive items sampled per user
NUM_NEG = 300             # number of negative items sampled per user
L2 = 1e-7                 # weight_decay
OPTIMIZER_STYLE = 'adam'  # 'sgd' or 'adam'

# GAMMA0 is the moving average factor in our algo, you can tune BETA0 in (0.0, 1.0) for better performance
GAMMA0 = 0.1
TOPK = 300
TOPK_V = 'theo' # 'prac' or 'theo'

n_users = 138493
n_items = 26744
num_relevant_pairs = trainSet.get_num_televant_pairs()

# save the model and log file
RES_PATH = 'k_song'
os.makedirs(RES_PATH, exist_ok=True)



labels = trainSet.targets.toarray().T
train_sampler = DistributedTriSampler(dataset=None, labels=labels, batch_size_per_task=(NUM_POS+NUM_NEG),
                           num_sampled_tasks_per_gpu=BATCH_SIZE_PER_GPU, num_pos=NUM_POS, mode='ranking', sampling_rate=None)


model = NeuMF(n_users, n_items)
model.apply(model.init_weights)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])


SONG_GAMMA0 = 0.1
criterion = NDCGLoss(num_relevant_pairs, n_users, n_items, NUM_POS, gamma0=SONG_GAMMA0, topk=TOPK, topk_version=TOPK_V, device=device)
optimizer = SONG(params=model.parameters(), lr=LR, weight_decay=L2, mode=OPTIMIZER_STYLE,  device=device)

print ('Start Training')
print ('-'*30)

train(model, trainSet, train_sampler, valSet, optimizer, device)

result_dict = evaluate(model, testSet, TOPKS, METRICS, device)
print("test results:" + format_metric(result_dict))