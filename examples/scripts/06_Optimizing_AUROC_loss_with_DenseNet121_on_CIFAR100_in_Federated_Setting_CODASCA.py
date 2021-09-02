"""
Authors: Zhuoning Yuan, Zhishuai Guo
Contact: yzhuoning@gmail.com
Reference:
    Zhuoning Yuan*, Zhishuai Guo*, Yi Xu, Yiming Ying, Tianbao Yang (equal contribution).
    Federated Deep AUC Maximization for Hetergeneous Data with a Constant Communication Complexity. 
    ICML 2021: 12219-12229 
    
How to run the code:  
    
    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr='YOUR IP' --master_port=8888 \
            main_codasca_cifar.py --T0=4000 --imratio=0.1 --gamma=500 --lr=0.1 --I=8 --local_batchsize=32 --total_iter=20000
    
"""
import torch
import torch.distributed as dist
import numpy as np
import copy 
import os,re,time, random
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from libauc.models import DenseNet121

physical_devices = tf.config.list_physical_devices('GPU')
AUTO = tf.data.experimental.AUTOTUNE

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--local_batchsize', type=int, default=32)
parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument('--model_name', type=str, default='densenet121')
parser.add_argument('--pretrained', type=bool, default=False)  # single or mixed
parser.add_argument('--ft', type=bool, default=True)  # single or mixed
parser.add_argument('--imratio', type=float, default=0.1)

parser.add_argument('--T0', type=int, default=4000)
parser.add_argument('--lr', type=float, default=0.1) 
parser.add_argument('--weight_decay', type=float, default=1e-5) 
parser.add_argument('--gamma', type=float, default=500)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--I', type=int, default=1)

parser.add_argument('--total_iter', type=int, default=20000)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--master_addr', type=str)

para = parser.parse_args()


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
class CODASCA:
    def __init__(self, imratio = 0.1, margin = 1.0, model=None, **kwargs):
        self.p = imratio
        self.margin = margin
        self.model = model
        self.model_ref = {}

        # PESG
        for name, var in self.model.state_dict().items(): 
            self.model_ref[name] = torch.empty(var.shape).normal_(mean=0, std=0.01).cuda()
        self.model_acc = copy.deepcopy(model.state_dict()) 
        self.a = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.b = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        self.alpha = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=True).cuda()
        
        self.a_ref = torch.empty(self.a.shape).normal_(mean=0,std=0.01).cuda() 
        self.b_ref = torch.empty(self.b.shape).normal_(mean=0,std=0.01).cuda() 
        self.a_acc = self.a.clone().detach().requires_grad_(False)
        self.b_acc = self.b.clone().detach().requires_grad_(False) 
        
        # CODASCA
        self.model_c_x = {}
        for name, var in self.model.state_dict().items(): 
                self.model_c_x[name] = torch.zeros(var.shape, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.a_c_x = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.b_c_x = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.alpha_c_y = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
            
        ## prev
        self.model_prev = copy.deepcopy(self.model.state_dict()) 
        self.a_prev = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.b_prev = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.alpha_prev = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        
        ## SCAFFOLD
        self.model_grad_acc = {}
        for name, var in self.model.state_dict().items(): 
            self.model_grad_acc[name] =  torch.zeros(var.shape, requires_grad=False).cuda()
        self.a_grad_acc = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.b_grad_acc = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.alpha_grad_acc = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        
        # others
        self.T = 0
        self.T_grad =  0

    def AUCMLoss(self, y_pred, y_true):
        '''
        AUC Margin Loss
        Reference:
            Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao,
            Proceedings of the IEEE/CVF International Conference on Computer Vision 2021.
        '''
        auc_loss = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                    self.p*torch.mean((y_pred - self.b)**2*(-1==y_true).float())   + \
                    2*self.alpha*(self.p*(1-self.p) + \
                    torch.mean((self.p*y_pred*(-1==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                    self.p*(1-self.p)*self.alpha**2
        return auc_loss

    
    def PESG(self, model_c_x=None, a_c_x=None, b_c_x=None, alpha_c_y=None, lr=0.1, gamma=500, clip_value=1.0, weight_decay=1e-4):
        # Primal
        for name, param in self.model.named_parameters(): 
            param.data = param.data - lr*( torch.clamp(param.grad.data , -clip_value, clip_value) - self.model_c_x[name].data +  model_c_x[name].data + 1/gamma*(param.data - self.model_ref[name].data)) - lr*weight_decay*param.data
            self.model_acc[name].data = self.model_acc[name].data + param.data
            self.model_grad_acc[name].data = self.model_grad_acc[name].data + param.grad.data
      
        self.a.data = self.a.data - lr*(self.a.grad.data - self.a_c_x + a_c_x + 1/gamma*(self.a.data - self.a_ref.data))- lr*weight_decay*self.a.data 
        self.b.data = self.b.data - lr*(self.b.grad.data - self.b_c_x + b_c_x  + 1/gamma*(self.b.data - self.b_ref.data))- lr*weight_decay*self.b.data 
        
        # dual
        self.alpha.data = self.alpha.data + lr*(2*(self.margin + self.b.data - self.a.data)-2*self.alpha.data - self.alpha_c_y + alpha_c_y) 
        self.alpha.data  = torch.clamp(self.alpha.data,  0, 999)
        
        self.a_acc.data = self.a_acc.data + self.a.data
        self.b_acc.data = self.b_acc.data + self.b.data
        
        self.a_grad_acc.data = self.a_grad_acc.data + self.a.grad.data
        self.b_grad_acc.data = self.b_grad_acc.data + self.b.grad.data
        self.alpha_grad_acc.data = self.alpha_grad_acc.data + 2*(self.margin + self.b.data - self.a.data)-2*self.alpha.data 
        
        self.T = self.T + 1
        self.T_grad = self.T_grad + 1
        
    def update_SCAFFOLD(self, I, lr, model_c_x=None, a_c_x=None, b_c_x=None, alpha_c_y=None):
        for name, param in self.model.named_parameters():
            self.model_c_x[name].data =  self.model_grad_acc[name].data/I
        self.a_c_x.data = self.a_grad_acc.data/I
        self.b_c_x.data = self.b_grad_acc.data/I
        self.alpha_c_y.data = self.alpha_grad_acc.data/I
        self.T_grad  = 0
                    
        # update model prev
        self.model_prev = copy.deepcopy(self.model.state_dict()) 
        self.a_prev = self.a.clone().detach().requires_grad_(False)
        self.b_prev = self.b.clone().detach().requires_grad_(False)
        self.alpha_prev = self.alpha.clone().detach().requires_grad_(False)
        
        # reset grad acc
        for name, var in self.model.state_dict().items(): 
            self.model_grad_acc[name] =  torch.zeros(var.shape, requires_grad=False).cuda()
        self.a_grad_acc = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.b_grad_acc = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        self.alpha_grad_acc = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
        
        # update global c_x;c_y 
        ## tmp vars for recieving vars from all nodes
        model_c_x_tmp = copy.deepcopy(self.model_c_x)
        a_c_x_tmp = self.a_c_x.clone().detach().requires_grad_(False)
        b_c_x_tmp = self.b_c_x.clone().detach().requires_grad_(False)
        alpha_c_y_tmp = self.alpha_c_y.clone().detach().requires_grad_(False)

        ## reduce all vars 
        size = float(dist.get_world_size())
        for name, var in self.model.state_dict().items(): 
            dist.all_reduce(self.model_c_x[name].data, op=dist.ReduceOp.SUM)
            model_c_x[name].data = self.model_c_x[name].data/size
                            
        dist.all_reduce(self.a_c_x.data, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.b_c_x.data, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.alpha_c_y.data, op=dist.ReduceOp.SUM)
                            
        a_c_x.data = self.a_c_x.data/size
        b_c_x.data = self.b_c_x.data/size
        alpha_c_y.data = self.alpha_c_y.data/size
    
        ## assign back to global vars
        for name, var in self.model.state_dict().items(): 
            self.model_c_x[name].data = model_c_x_tmp[name]
        self.a_c_x.data = a_c_x_tmp
        self.b_c_x.data =  b_c_x_tmp
        self.alpha_c_y.data =  alpha_c_y_tmp 
                  
        # average over all clients    
        momentum_beta = 1
        for name, param in self.model.named_parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= size
            param.data = momentum_beta*param.data + (1-momentum_beta)*self.model_prev[name]
        
        dist.all_reduce(self.a.data, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.b.data, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.alpha.data, op=dist.ReduceOp.SUM)
                        
        self.a.data /= float(size)
        self.b.data /= float(size)
        self.alpha.data /= float(size)
        
        self.a.data = momentum_beta*self.a.data + (1-momentum_beta)*self.a_prev.data
        self.b.data = momentum_beta*self.b.data + (1-momentum_beta)*self.b_prev.data
        self.alpha.data = momentum_beta*self.alpha.data + (1-momentum_beta)*self.alpha_prev.data
      
    def zero_grad(self):
        self.model.zero_grad()
        self.a.grad = None
        self.b.grad = None
        self.alpha.grad =None

    def update_regularizer(self):
        print ('Update regularizer!', self.T)
        for name, param in self.model.named_parameters():
            self.model_ref[name].data = self.model_acc[name].data/self.T
        self.a_ref.data = self.a_acc.data/self.T
        self.b_ref.data = self.b_acc.data/self.T

        # reset
        self.a_acc = self.a.clone().detach().requires_grad_(False)
        self.b_acc = self.b.clone().detach().requires_grad_(False)
        self.model_acc = copy.deepcopy(self.model.state_dict())  
        self.T = 0

def partition (list_in, n, seed=123):
    random.seed(seed)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)] 
  
def generate_imbalance_dataset(raw_data, raw_labels, imratio=0.5, size=1, rank=0, is_balanced=False, shuffle=True, seed=123):
    data = raw_data.copy()
    labels = raw_labels.copy()
    if labels.max() < 2:  #C2
        split_index = 0 
    if labels.max() == 99: #C100
        split_index = 49 
    if labels.max() == 9:  #C10;
        split_index = 4
        
    if shuffle:
        ids = list(range(data.shape[0]))
        np.random.seed(seed)
        np.random.shuffle(ids)
        data = data[ids]
        labels = labels[ids]

    if is_balanced == False and size > 1:

        neg_indices = list(range(50))
        pos_indices = list(range(50, 100))
        pos_indices_k = partition(pos_indices, size, seed)
        neg_indices_k = partition(neg_indices, size, seed)

        select_idx_list= []
        
        # delete sampels from each class based on pos ratio
        for pos_class_id, neg_class_id in zip(range(50, 100), range(50)):
            num_neg = np.where(labels==neg_class_id)[0].shape[0]
            assert num_neg == 500, 'error!'
            keep_num_pos = int((imratio/(1-imratio))*num_neg )
            idx_pos_tmp = np.where(labels==pos_class_id)[0][:keep_num_pos] 
            idx_neg_tmp = np.where(labels==neg_class_id)[0] 
            select_idx_list.extend(idx_neg_tmp.tolist() + idx_pos_tmp.tolist())
            
        data = data[select_idx_list] 
        labels = labels[select_idx_list]    
        
        # select data group by rank 
        select_data = data[np.isin(labels, pos_indices_k[rank] + neg_indices_k[rank]).squeeze()].copy()
        select_label = labels[np.isin(labels, pos_indices_k[rank] + neg_indices_k[rank])].copy()
             
        select_label[select_label<=split_index] = -1 # [0, ....]
        select_label[select_label>=split_index+1] = 1 # [0, ....]

        data = select_data.copy()
        labels = select_label.copy()
    else:
        #labels = labels.reshape((-1, 1))
        labels[labels<=split_index] = -1 # [0, ....]
        labels[labels>=split_index+1] = 1 # [0, ....]
      
    pos_count = np.count_nonzero(labels == 1)
    neg_count = np.count_nonzero(labels == -1)
    print ('Rank:%d/%d, Pos:Neg: [%d : %d], Pos Ratio: %.4f'%(rank, size, pos_count,neg_count, pos_count/ (pos_count + neg_count) ) )
    return data, labels.reshape(-1, 1)

def prepare_image(img, augment=True, dim=256,):    
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_crop(img, [dim-2, dim-2, 3])    
        img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [dim, dim])                  
    img = tf.reshape(img, [dim,dim, 3])
    return img

def get_dataset(dataset, augment = False, shuffle = False, repeat = False, labeled=True, return_image_names=True, batch_size=16, dim=32):
    ds = dataset
    ds = ds.cache()
    if repeat:
        ds = ds.repeat()
    if shuffle: 
        ds = ds.shuffle(40000)
        opt = tf.data.Options()
        opt.experimental_deterministic = True
        ds = ds.with_options(opt)
    ds = ds.map(lambda img, label: (prepare_image(img, augment=augment, dim=dim), label), num_parallel_calls=AUTO) 
    ds = ds.batch(batch_size) 
    # for pytorch 
    ds = ds.map(lambda x, y: (tf.transpose(x, (0, 3, 1, 2)), y), num_parallel_calls=AUTO) 
    ds = ds.prefetch(AUTO)
    return ds


def train(rank, size, group):
    torch.cuda.set_device(para.local_rank)
    
    # Load datasets
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar100.load_data()
    (train_data, train_label) = (train_data.astype(float), train_label.astype(np.int32))
    (test_data, test_label) = (test_data.astype(float), test_label.astype(np.int32)) 
    
    (train_images, train_labels) = generate_imbalance_dataset(train_data, train_label, imratio=para.imratio, seed=para.random_seed, rank=rank, size=size)
    (test_images, test_labels) = generate_imbalance_dataset(test_data, test_label, imratio=para.imratio, seed=para.random_seed, rank=rank, is_balanced=True)
 
    # assign data by rank id
    tf.config.experimental.set_visible_devices(physical_devices[para.local_rank], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[para.local_rank], True)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    trainloader = get_dataset(train_dataset, augment=True, shuffle=True, repeat=True, dim=para.image_size, batch_size = para.local_batchsize) 
    
    if rank == 0 and para.local_rank ==0:
        test_dataset = tf.data.Dataset.from_tensor_slices( (test_images, test_labels))
        testloader = get_dataset(test_dataset, augment=False, shuffle=False, repeat=False, dim=para.image_size, batch_size = para.local_batchsize*8) 
        trainloader_eval = get_dataset(train_dataset, augment=False, shuffle=False, repeat=False, dim=para.image_size, batch_size = para.local_batchsize) 

    # model & optimizer
    set_all_seeds(para.random_seed)
    model = DenseNet121(pretrained=True, last_activation='sigmoid', num_classes=1)
    model = model.cuda() 
    optimizer = CODASCA(imratio=para.imratio, margin=para.margin, model=model)   

    # global vars for reducing client shift
    model_c_x = {}
    for name, var in model.state_dict().items(): 
        model_c_x[name] = torch.zeros(var.shape, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
    a_c_x = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
    b_c_x = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
    alpha_c_y = torch.zeros(1, dtype=torch.float32, device="cuda", requires_grad=False).cuda()
            
    if rank == 0:
        init_weights = [w.data.cpu().clone() for w in list(model.parameters())]
        print ('Init weights:', init_weights[0].numpy().sum())
    
    start_time = time.time()
    total_iter = 0
    best_val_auc = 0
    
    for epoch in range(200):
        
        for i, data in enumerate(trainloader):
            model.train()
            
            if i == para.total_iter:
                os.system('pkill python')
                break
                        
            # decay lr & update regularizer
            if i % para.T0 == 0 and i > 0:
               para.lr = para.lr/3
               optimizer.update_regularizer()
              
            # load datasets
            train_data, train_labels = data
            train_data, train_labels = train_data.numpy(), train_labels.numpy()
            train_data, train_labels = torch.from_numpy(train_data), torch.from_numpy(train_labels)
            train_data, train_labels  = train_data.cuda(), train_labels.cuda()

            # forward + backward + optimization
            optimizer.zero_grad()
            pred_prob = model(train_data)
            loss = optimizer.AUCMLoss(pred_prob, train_labels.view(-1, 1))
            loss.backward()
            optimizer.PESG(model_c_x=model_c_x, a_c_x=a_c_x, b_c_x=b_c_x, alpha_c_y=alpha_c_y, lr=para.lr, gamma=para.gamma, clip_value=1.0, weight_decay=para.weight_decay)
    
            # communicatios over all machines
            if 0 == i %(para.I):
                if size > 1:  
                    with torch.no_grad():
                        optimizer.update_SCAFFOLD(I=para.I, lr=para.lr, model_c_x=model_c_x, a_c_x=a_c_x, b_c_x=b_c_x, alpha_c_y=alpha_c_y) 

            # evaluation
            if i % 100 == 0 and rank == 0:
                model.eval()
                with torch.no_grad():
                    
                    train_pred = []
                    train_true = [] 
                    for j, data in enumerate(trainloader_eval):
                        train_data, train_label = data
                        train_data, train_label = train_data.numpy(), train_label.numpy()
                        train_data, train_label = torch.from_numpy(train_data), torch.from_numpy(train_label)
                        train_data = train_data.cuda()
                        y_pred = model(train_data)
                        train_pred.append(y_pred.cpu().detach().numpy())
                        train_true.append(train_label.numpy())
                    train_true = np.concatenate(train_true)
                    train_pred = np.concatenate(train_pred)
                    train_auc =  roc_auc_score(train_true, train_pred)  
                    
                    test_pred = []
                    test_true = [] 
                    for j, data in enumerate(testloader):
                        test_data, test_label = data
                        test_data, test_label = test_data.numpy(), test_label.numpy()
                        test_data, test_label = torch.from_numpy(test_data), torch.from_numpy(test_label)
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
                
                line_log = ("iter: {}, train_loss: {:4f}, train_auc:{:4f}, test_auc:{:4f}, best_test_auc:{:4f},  lr:{:4f}, time:{:4f}".format(total_iter, loss.item(), train_auc, val_auc, best_val_auc, para.lr, time.time()-start_time ))          
                print (line_log)
                start_time = time.time()
                
            total_iter += 1


if __name__ == "__main__":
    dist.init_process_group('nccl')
    size = dist.get_world_size()
    group = dist.new_group(range(size))
    rank = dist.get_rank()
    print ('Current Rank: %s, Number of nodes: %s '%(str(rank), str(size)))
    train(rank, size, group)
