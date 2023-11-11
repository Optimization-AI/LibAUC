import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import dok_matrix
from .surrogate import get_surrogate_loss


class ListwiseCELoss(torch.nn.Module):
    r"""Stochastic Optimization of Listwise CE loss. The objective function is defined as

        .. math::

            F(\mathbf{w})=\frac{1}{N}\sum_{q=1}^{N} \frac{1}{N_q}\sum_{\mathbf{x}_i^q \in S_q^+} - y_i^q \ln \left(\frac{\exp(h_q(\mathbf{x}_i^q;\mathbf{w}))}{\sum_{\mathbf{x}_j^q \in S_q} \exp(h_q(\mathbf{x}_j^q;\mathbf{w})) }\right)

        where :math:`h_q(\mathbf{x}_i^q;\mathbf{w})` is the predicted score of :math:`\mathbf{x}_i^q` with respect to :math:`q`, :math:`y_i^q` is the relvance score of :math:`x_i^q` with respect to :math:`q`, :math:`N` is the number of total queries, :math:`N_q` is the total number of items to be ranked for query q,
        :math:`S_q` denotes the set of items to be ranked by query :math:`q`, and :math:`S_q^+` denotes the set of relevant items for query :math:`q`.

        Args: 
            N (int): number of all relevant pairs
            num_pos (int): number of positive items sampled for each user
            gamma (float): the factor for moving average, i.e., \gamma in our paper [1]_.
            eps (float, optional): a small value to avoid divide-zero error (default: ``1e-10``)

        Example:
            >>> loss_fn = libauc.losses.ListwiseCELoss(N=1000, num_pos=10, gamma=0.1)      # assume we have 1000 relevant query-item pairs
            >>> predictions = torch.randn((32, 10+20), requires_grad=True)                   # we sample 32 queries/users, and 10 positive items and 20 negative items for each query/user
            >>> batch = {'user_item_id': torch.randint(low=0, high=1000-1, size=(32,10+20))} # ids for all sampled query-item pairs in the batch
            >>> loss = loss_fn(predictions, batch)
            >>> loss.backward()

        Reference:
            .. [1] Qiu, Zi-Hao, Hu, Quanqi, Zhong, Yongjian, Zhang, Lijun, and Yang, Tianbao.
                   "Large-scale Stochastic Optimization of NDCG Surrogates for Deep Learning with Provable Convergence."
                   Proceedings of the 39th International Conference on Machine Learning. 2022.
                   https://arxiv.org/abs/2202.12183
    """
    def __init__(self,
                  N, 
                  num_pos, 
                  gamma, 
                  eps=1e-10,
                  device=None):
        super(ListwiseCELoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.num_pos = num_pos
        self.gamma = gamma
        self.eps = eps
        self.u = torch.zeros(N).to(self.device)

    def forward(self, predictions, batch):
        """
        Args:
            predictions:  predicted socres from the model, shape: [batch_size, num_pos + num_neg]
            batch:        a dict that contains two keys: user_id and item_id        
        """
        batch_size = predictions.size(0)
        neg_pred = torch.repeat_interleave(predictions[:, self.num_pos:], self.num_pos, dim=0)                   # [batch_size * num_pos, num_neg]
        pos_pred = torch.cat(torch.chunk(predictions[:, :self.num_pos], batch_size, dim=0), dim=1).permute(1,0)  # [batch_size * num_pos, 1]

        margin = neg_pred - pos_pred
        exp_margin = torch.exp(margin - torch.max(margin)).detach_()
    
        user_item_ids = batch['user_item_id'][:, :self.num_pos].reshape(-1)

        self.u[user_item_ids] = (1-self.gamma) * self.u[user_item_ids] + self.gamma * torch.mean(exp_margin, dim=1)

        exp_margin_softmax = exp_margin / (self.u[user_item_ids][:, None] + self.eps)

        loss = torch.sum(margin * exp_margin_softmax)
        loss /= batch_size

        return loss


class NDCGLoss(torch.nn.Module):
    r"""Stochastic Optimization of NDCG (SONG) and top-K NDCG (K-SONG). The objective function of K-SONG is a bilevel optimization problem as presented below:

        .. math::
            & \min \frac{1}{|S|} \sum_{(q,\mathbf{x}_i^q)\in S} \psi(h_q(\mathbf{x}_i^q;\mathbf{w})-\hat{\lambda}_q(\mathbf{w})) f_{q,i}(g(\mathbf{w};\mathbf{x}_i^q,S_q))

            & s.t. \hat{\lambda}_q(\mathbf{w})=\arg\min_{\lambda} \frac{K+\epsilon}{N_q}\lambda + \frac{\tau_2}{2}\lambda^2 + \frac{1}{N_q} \sum_{\mathbf{x}_i^q \in S_q} \tau_1 \ln(1+\exp((h_q(\mathbf{x}_i^q;\mathbf{w})-\lambda)/\tau_1)) ,

            &  \forall q\in\mathbf{Q}

        where :math:`\psi(\cdot)` is a smooth Lipschtiz continuous function to approximate :math:`\mathbb{I}(\cdot\ge 0)`, e.g., sigmoid function, :math:`f_{q,i}(g)` denotes :math:`\frac{1}{Z_q^K}\frac{1-2^{y_i^q}}{\log_2(N_q g+1)}`. The objective formulation for SONG is a special case of
        that for K-SONG, where the :math:`\psi(\cdot)` function is a constant. 
            
        Args:
            N (int): number of all relevant pairs
            num_user (int): number of users in the dataset
            num_item (int): number of items in the dataset
            num_pos (int): number of positive items sampled for each user
            gamma0 (float): the moving average factor of u_{q,i}, i.e., \beta_0 in our paper, in range (0.0, 1.0)
                this hyper-parameter can be tuned for better performance (default: ``0.9``)
            gamma1 (float, optional): the moving average factor of s_{q} and v_{q} (default: ``0.9``)
            eta0 (float, optional): step size of \lambda (default: ``0.01``)
            margin (float, optional): margin for squared hinge loss (default: ``1.0``)
            topk (int, optional): NDCG@k optimization is activated if topk > 0; topk=-1 represents SONG (default: ``1e-10``)
            topk_version (string, optional): 'theo' or 'prac'  (default: ``theo``)
            tau_1 (float, optional): \tau_1 in Eq. (6), \tau_1 << 1 (default: ``0.01``)
            tau_2 (float, optional): \tau_2 in Eq. (6), \tau_2 << 1 (default: ``0.0001``) 
            sigmoid_alpha (float, optional): a hyperparameter for sigmoid function, psi(x) = sigmoid(x * sigmoid_alpha) (default: ``1.0``)

        Example:
            >>> loss_fn = libauc.losses.NDCGLoss(N=1000, num_user=100, num_item=5000, num_pos=10, gamma0=0.1, topk=-1)  # SONG (with topk = -1)/K-SONG (with topk = 100)
            >>> predictions = torch.randn((32, 10+20), requires_grad=True)              # we sample 32 queries/users, and 10 positive items and 20 negative items for each query/user
            >>> batch = {
                    'rating': torch.randint(low=0, high=5, size=(32,10+20)),            # ratings (e.g., in the range of [0,1,2,3,4]) for each sampled query-item pair
                    'user_id': torch.randint(low=0, high=100-1, size=32),               # id for each sampled query
                    'num_pos_items': torch.randint(low=0, high=1000, size=32),          # number of all relevant items for each sampled query
                    'ideal_dcg': torch.rand(32),                                        # ideal DCG precomputed for each sampled query (in the range of (0.0, 1.0))
                    'user_item_id': torch.randint(low=0, high=1000-1, size=(32,10+20))} # ids for all sampled query-item pairs in the batch
                }  
            >>> loss = loss_fn(predictions, batch)
            >>> loss.backward()

        Reference:
            .. [1] Qiu, Zi-Hao, Hu, Quanqi, Zhong, Yongjian, Zhang, Lijun, and Yang, Tianbao.
                   "Large-scale Stochastic Optimization of NDCG Surrogates for Deep Learning with Provable Convergence."
                   Proceedings of the 39th International Conference on Machine Learning. 2022.
                   https://arxiv.org/abs/2202.12183
    """
    def __init__(self, 
                  N, 
                  num_user, 
                  num_item, 
                  num_pos,
                  gamma0=0.9, 
                  gamma1=0.9, 
                  eta0=0.01,
                  margin=1.0, 
                  topk=-1, 
                  topk_version='theo', 
                  tau_1=0.01, 
                  tau_2=0.0001,
                  sigmoid_alpha=2.0, 
                  surrogate_loss='squared_hinge',
                  device=None):
        super(NDCGLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.num_pos = num_pos
        self.margin = margin
        self.gamma0 = gamma0
        self.topk = topk                              
        self.lambda_q = torch.zeros(num_user+1).to(self.device)   # learnable thresholds for all querys (users)
        self.gamma1 = gamma1                        
        self.tau_1 = tau_1                            
        self.tau_2 = tau_2                       
        self.eta0 = eta0                  
        self.num_item = num_item
        self.topk_version = topk_version
        self.s_q = torch.zeros(num_user+1).to(self.device)        # moving average estimator for \nabla_{\lambda}^2 L_q
        self.sigmoid_alpha = sigmoid_alpha
        self.u = torch.zeros(N).to(self.device) 
        self.surrogate_loss = get_surrogate_loss(surrogate_loss)
    
    def forward(self, predictions, batch):
        
        device = predictions.device
        ratings = batch['rating'][:, :self.num_pos]                                                                    # [batch_size, num_pos]
        batch_size = ratings.size()[0]
        predictions_expand = torch.repeat_interleave(predictions, self.num_pos, dim=0)                                 # [batch_size*num_pos, num_pos+num_neg]
        predictions_pos = torch.cat(torch.chunk(predictions[:, :self.num_pos], batch_size, dim=0), dim=1).permute(1,0) # [batch_suze*num_pos, 1]

        num_pos_items = batch['num_pos_items'].float()  # [batch_size], the number of positive items for each user
        ideal_dcg = batch['ideal_dcg'].float()          # [batch_size], the ideal dcg for each user
        
        g = torch.mean(self.surrogate_loss(self.margin, predictions_pos-predictions_expand), dim=-1)   # [batch_size*num_pos]
        g = g.reshape(batch_size, self.num_pos)                                                        # [batch_size, num_pos], line 5 in Algo 2.

        G = (2.0 ** ratings - 1).float()

        user_ids = batch['user_id']
        user_item_ids = batch['user_item_id'][:, :self.num_pos].reshape(-1)

        self.u[user_item_ids] = (1-self.gamma0) * self.u[user_item_ids] + self.gamma0 * g.clone().detach_().reshape(-1)
        g_u = self.u[user_item_ids].reshape(batch_size, self.num_pos)

        nabla_f_g = (G * self.num_item) / ((torch.log2(1 + self.num_item*g_u))**2 * (1 + self.num_item*g_u) * np.log(2)) # \nabla f(g)

        if self.topk > 0:
            user_ids = user_ids.long()
            pos_preds_lambda_diffs = predictions[:, :self.num_pos].clone().detach_() - self.lambda_q[user_ids][:, None].to(device)
            preds_lambda_diffs = predictions.clone().detach_() - self.lambda_q[user_ids][:, None].to(device)

            # the gradient of lambda
            grad_lambda_q = self.topk/self.num_item + self.tau_2*self.lambda_q[user_ids] - torch.mean(torch.sigmoid(preds_lambda_diffs.to(device) / self.tau_1), dim=-1)
            self.lambda_q[user_ids] = self.lambda_q[user_ids] - self.eta0 * grad_lambda_q

            if self.topk_version == 'prac':
                nabla_f_g *= torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)

            elif self.topk_version == 'theo':
                nabla_f_g *= torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)
                d_psi = torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha) * (1 - torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha))
                f_g_u = -G / torch.log2(1 + self.num_item*g_u)
         
                # part 2 of eqn. (5)
                temp_term = torch.sigmoid(preds_lambda_diffs / self.tau_1) * (1 - torch.sigmoid(preds_lambda_diffs / self.tau_1)) / self.tau_1
                L_lambda_hessian = self.tau_2 + torch.mean(temp_term, dim=1)                                     # \nabla_{\lambda}^2 L_q in Eq. (5) in the paper
                self.s_q[user_ids] = self.gamma1 * L_lambda_hessian.to(device) + (1-self.gamma1) * self.s_q[user_ids] # line 10 in Algorithm 2 in the paper
                hessian_term = torch.mean(temp_term * predictions, dim=1) / self.s_q[user_ids].to(device)        # \nabla_{\lambda,w}^2 L_q * s_q in Eq. (5) in the paper
                
                # based on eqn. (5)
                loss = (num_pos_items * torch.mean(nabla_f_g * g + d_psi * f_g_u * (predictions[:, :self.num_pos] - hessian_term[:, None]), dim=-1) / ideal_dcg).mean()
                return loss

        loss = (num_pos_items * torch.mean(nabla_f_g * g, dim=-1) / ideal_dcg).mean()
        return loss
